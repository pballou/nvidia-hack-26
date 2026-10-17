"""The full loop end to end: real headless Chromium on the fake sandbox, the DOM oracle for
grounding and summaries, and a scripted FakeLightning. No GPU and no network.

Each run starts from /api/reset with a fresh sid and is verified from /api/state with the
real sandbox.checks.evaluate (tests may read state; the agent never does).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from agent.config import AgentConfig
from agent.core import run_task
from agent.testing.fake_lightning import FakeLightning, Reply, click, done, type_text
from agent.testing.fake_sandbox import TASKS_BY_ID, evaluate

OVERLAY_COLOR = (255, 0, 160)  # agent/trace.py draw_overlay default

SCRIPTS = {
    "fake_high_contrast": [click("the High contrast toggle"), done("Turned on high contrast.")],
    "fake_gear_high_contrast": [click("the gear icon in the top-right toolbar"), click("the High contrast toggle"),
                                done("Opened settings and turned on high contrast.")],
    "fake_display_name": [type_text("Alex Kim", "the Display name field"), click("the Save button"),
                          done("Changed the display name to Alex Kim.")],
}


async def _run(task_id: str, script: list, base_url: str, sid: str, tmp_path: Path):
    task = TASKS_BY_ID[task_id]
    async with httpx.AsyncClient(base_url=base_url, timeout=10) as http:
        (await http.post("/api/reset", json={"sid": sid, "task_id": task_id})).raise_for_status()
        fake = FakeLightning(script)
        received = []

        async def on_event(ev):
            received.append(json.loads(ev.to_json()))  # JSON round trip: tuples become lists, as in the file

        cfg = AgentConfig().with_options(runs_dir=str(tmp_path), use_oracle_grounder=True)
        result = await run_task(task["request"], start_url=f"{base_url}{task['start_path']}?sid={sid}", cfg=cfg,
                                on_event=on_event, planner_client=fake.client)
        state = (await http.get("/api/state", params={"sid": sid})).json()
    trace = Path(result.trace_dir)
    lines = [json.loads(ln) for ln in (trace / "events.jsonl").read_text().splitlines()]
    return task, result, state, trace, lines, received, fake


@pytest.mark.asyncio
@pytest.mark.parametrize("task_id", list(SCRIPTS))
async def test_full_loop_solves_task(task_id, fake_sandbox_url, sid, browser_pool, tmp_path):
    script = SCRIPTS[task_id]
    task, result, state, trace, lines, received, fake = await _run(task_id, script, fake_sandbox_url, sid, tmp_path)

    passed, explanation = evaluate(state, task["check"])
    assert passed, explanation
    assert result.success_claimed is True and result.steps == len(script)
    assert fake.exhausted and len(fake.requests) == len(script)

    kinds = [ln["kind"] for ln in lines]
    assert {"start", "observation", "plan", "tool_call", "grounding", "action", "done"} <= set(kinds)
    assert kinds[0] == "start" and kinds[-1] == "done" and "error" not in kinds
    done_ev = lines[-1]["payload"]
    assert done_ev["reason"] == "done" and done_ev["stats"]["n_invalid_calls"] == 0
    assert done_ev["stats"]["n_failed_actions"] == 0 and done_ev["stats"]["n_no_change"] == 0
    for ln in lines:
        if ln["kind"] in ("observation", "plan", "grounding"):
            assert ln.get("timings_ms"), ln["kind"]
        if ln["kind"] == "action" and ln["payload"]["name"] != "done":
            assert {"act", "settle"} <= set(ln["timings_ms"])
            assert ln["payload"]["ok"] and ln["payload"]["screen_changed"] is True
    assert "oracle" in {ln["payload"]["model"] for ln in lines if ln["kind"] == "grounding"}

    assert received == lines  # the callback saw exactly the trace, in order
    assert (trace / "step_00.png").exists() and (trace / "step_01.png").exists()
    overlay = trace / "step_01_overlay.png"
    assert overlay.exists()
    g = next(ln for ln in lines if ln["kind"] == "grounding")
    assert g["overlay_path"] == str(overlay)
    with Image.open(overlay) as im:
        assert im.size == (1280, 800)
        assert im.convert("RGB").getpixel((round(g["payload"]["x"]), round(g["payload"]["y"]))) == OVERLAY_COLOR


@pytest.mark.asyncio
async def test_wrong_argument_name_is_not_executed_on_a_real_page(fake_sandbox_url, sid, browser_pool, tmp_path):
    script = [Reply([("click", {"target": "the High contrast toggle"})]), click("the High contrast toggle"),
              done("Turned on high contrast.")]
    task, result, state, trace, lines, _, fake = await _run("fake_high_contrast", script, fake_sandbox_url, sid,
                                                            tmp_path)
    assert evaluate(state, task["check"])[0]  # toggled exactly once: the invalid call clicked nothing
    assert [ln["kind"] for ln in lines if ln["step"] == 1] == ["plan", "tool_call"]
    assert not (trace / "step_01_overlay.png").exists() and (trace / "step_02_overlay.png").exists()
    stats = lines[-1]["payload"]["stats"]
    assert (stats["n_tool_calls"], stats["n_invalid_calls"], stats["n_clicks"]) == (3, 1, 1)
    assert fake.requests[1]["chat_template_kwargs"] == {"enable_thinking": True}  # thinking=retry after a bad call
