"""Live smoke test: the configured Lightning (configs/agent.yaml; local :8000 by default) plus the
DOM oracle on the fake sandbox. Opt-in, because it needs a running Lightning:

  AGENT_LIVE_TESTS=1 .venv-gym/bin/python -m pytest agent/tests/test_live_lightning.py -q -s
  AGENT_LIVE_TESTS=1 AGENT_LIGHTNING=hosted ...   # build.nvidia.com (25-80 s per call)

Loop plumbing only: the oracle is not the grounder, so nothing measured here is reported.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from agent.config import load_config
from agent.core import run_task
from agent.testing.fake_sandbox import TASKS_BY_ID, evaluate

pytestmark = pytest.mark.skipif(os.environ.get("AGENT_LIVE_TESTS") != "1",
                                reason="live Lightning test; set AGENT_LIVE_TESTS=1 to run")

# The acceptance bar: >= 3 easy tasks (a toggle, a menu then a toggle, a text field).
TASK_IDS = ["fake_high_contrast", "fake_gear_high_contrast", "fake_display_name"]


@pytest.mark.asyncio
@pytest.mark.parametrize("task_id", TASK_IDS)
async def test_live_lightning_solves_an_easy_task(task_id, fake_sandbox_url, sid, browser_pool, tmp_path):
    cfg = load_config(overrides={"runs_dir": str(tmp_path), "use_oracle_grounder": True, "max_steps": 8})
    try:
        httpx.get(cfg.lightning.base_url.rstrip("/") + "/models", timeout=5,
                  headers={"Authorization": f"Bearer {cfg.lightning.api_key}"}).raise_for_status()
    except Exception as e:
        pytest.fail(f"Lightning at {cfg.lightning.base_url} is not reachable: {type(e).__name__}: {e}")

    task = TASKS_BY_ID[task_id]
    async with httpx.AsyncClient(base_url=fake_sandbox_url, timeout=10) as http:
        (await http.post("/api/reset", json={"sid": sid, "task_id": task_id})).raise_for_status()
        result = await run_task(task["request"], start_url=f"{fake_sandbox_url}{task['start_path']}?sid={sid}",
                                cfg=cfg, pool=browser_pool)
        state = (await http.get("/api/state", params={"sid": sid})).json()

    lines = [json.loads(ln) for ln in (Path(result.trace_dir) / "events.jsonl").read_text().splitlines()]
    stats = lines[-1]["payload"]["stats"]
    root = lines[0]["payload"]["config"]["lightning"].get("root")
    print(f"\n{task_id}: {cfg.lightning.model} @ {cfg.lightning.base_url} (serving {root}): steps={result.steps} "
          f"claimed={result.success_claimed} tool_calls={stats['n_tool_calls']} invalid={stats['n_invalid_calls']} "
          f"trace={result.trace_dir}")
    passed, explanation = evaluate(state, task["check"])
    assert passed, f"{explanation}; agent said: {result.summary}"
    assert lines[-1]["payload"]["reason"] == "done"
