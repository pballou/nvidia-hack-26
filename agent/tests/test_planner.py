"""The interactive loop (agent/planner.py + agent/core.py) against a scripted Lightning.

No network and no browser: FakeLightning is a real AsyncOpenAI client over an in-process
transport; the page is test_session's in-memory FakeBrowser.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agent import planner as planner_mod
from agent import prompts
from agent.config import AgentConfig
from agent.core import NO_TOOL_CALL_CORRECTION, TRUNCATED_CORRECTION, run_task
from agent.observation import NO_CHANGE_NOTE
from agent.planner import INITIAL_SCREEN_OMITTED, NOT_EXECUTED, LightningPlanner, PlanResult, ToolCall, \
    parse_content_tool_calls
from agent.session import validate_args
from agent.testing.fake_lightning import FakeLightning, Reply, click, done, say, type_text
from agent.tests.test_session import FakeGrounder, FakePool, FakeSummarizer, ShotFailsAfterClick

START_URL = "http://sandbox.test/settings?sid=t"
SECRET = "nvapi-test-secret-do-not-print"


@pytest.fixture(autouse=True)
def _no_retry_waits(monkeypatch):
    monkeypatch.setattr(planner_mod, "RATE_LIMIT_WAIT_S", 0.0)
    monkeypatch.setattr(planner_mod, "TRANSIENT_RETRY_S", 0.0)


async def _run(tmp_path: Path, script, *, on_event=None, client=None, grounder=None, **cfg_changes):
    fake = None if client is not None else FakeLightning(script)
    pool = FakePool()
    grounder = grounder or FakeGrounder()
    events = []

    async def collect(ev):
        events.append(ev)
        if on_event is not None:
            await on_event(ev)

    cfg = AgentConfig().with_options(runs_dir=str(tmp_path), **cfg_changes)
    result = await run_task("Open the gear menu.", start_url=START_URL, cfg=cfg, on_event=collect,
                            planner_client=client or fake.client, grounder=grounder,
                            summarizer=FakeSummarizer(), pool=pool)
    return SimpleNamespace(result=result, events=events, fake=fake, browser=pool.browser, cfg=cfg,
                           trace=Path(result.trace_dir), grounder=grounder, pool=pool)


def _lines(trace: Path) -> list[dict]:
    return [json.loads(line) for line in (trace / "events.jsonl").read_text().splitlines()]


def _of(events, kind: str) -> list:
    return [e for e in events if e.kind == kind]


def _done(events) -> dict:
    (ev,) = _of(events, "done")
    return ev.payload


def _thinking(fake: FakeLightning) -> list[bool]:
    return [r["chat_template_kwargs"]["enable_thinking"] for r in fake.requests]


def _tool_results(messages: list[dict]) -> list[dict]:
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


def assert_pairing(messages: list[dict]) -> None:
    """system, user, then records: assistant+tool_calls followed by one tool message per id, or assistant+user."""
    assert [m["role"] for m in messages[:2]] == ["system", "user"]
    i = 2
    while i < len(messages):
        m = messages[i]
        assert m["role"] == "assistant", (i, m)
        ids = [tc["id"] for tc in m.get("tool_calls") or []]
        for tc in m.get("tool_calls") or []:
            json.loads(tc["function"]["arguments"])  # history never carries broken JSON
        if ids:
            answers = messages[i + 1:i + 1 + len(ids)]
            assert [a["role"] for a in answers] == ["tool"] * len(ids)
            assert [a["tool_call_id"] for a in answers] == ids
            i += 1 + len(ids)
        else:
            assert messages[i + 1]["role"] == "user"
            i += 2


# --- happy path + request shape ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_writes_trace(tmp_path):
    r = await _run(tmp_path, [click("the gear icon in the top-right toolbar"), done("Opened the gear menu.")])
    assert r.result.success_claimed is True and r.result.steps == 2
    assert r.result.summary == "Opened the gear menu."
    assert r.trace.parent == tmp_path and r.trace.name == r.result.run_id
    for name in ("events.jsonl", "step_00.png", "step_01.png", "step_01_overlay.png"):
        assert (r.trace / name).exists(), name
    kinds = [e.kind for e in r.events]
    assert kinds == ["start", "observation", "plan", "tool_call", "grounding", "action", "observation",
                     "plan", "tool_call", "action", "done"]
    assert [ln["kind"] for ln in _lines(r.trace)] == kinds  # file == stream
    assert all(e.run_id == r.result.run_id for e in r.events)

    (g,) = _of(r.events, "grounding")
    assert (g.payload["x"], g.payload["y"]) == (1225.0, 30.0) and g.payload["model"] == "base"
    assert g.overlay_path.endswith("step_01_overlay.png") and "ground" in g.timings_ms
    action = _of(r.events, "action")[0].payload
    assert action["ok"] and action["screen_changed"] is True and action["click"] == {"x": 1225, "y": 30}
    assert _of(r.events, "tool_call")[0].payload["description"] == "Clicking: the gear icon in the top-right toolbar"
    assert _of(r.events, "tool_call")[1].payload["description"] == "Finishing: Opened the gear menu."
    assert all("plan" in e.timings_ms for e in _of(r.events, "plan"))
    assert _of(r.events, "observation")[1].screenshot_path.endswith("step_01.png")

    d = _done(r.events)
    assert d["reason"] == "done" and d["success_claimed"] is True and d["steps"] == 2
    s = d["stats"]
    assert (s["n_actions"], s["n_clicks"], s["n_failed_actions"], s["n_invalid_calls"]) == (2, 1, 0, 0)
    assert len(s["latency_ms"]["plan"]) == 2 and len(s["latency_ms"]["ground"]) == 1
    assert {"summary", "act", "settle", "screenshot"} <= set(s["latency_ms"])
    assert r.browser.closed
    assert r.browser.log[0] == ("goto", START_URL)
    assert r.pool.kwargs == {"navigation_timeout_s": r.cfg.timeouts.navigation_s}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,first", [("retry", False), ("off", False), ("on", True)])
async def test_request_kwargs(tmp_path, mode, first):
    r = await _run(tmp_path, [done("Nothing to do.")], thinking=mode)
    (body,) = r.fake.requests
    assert body["model"] == r.cfg.lightning.model
    assert body["tools"] == prompts.chat_tools()
    assert body["tool_choice"] == "required" and body["parallel_tool_calls"] is False
    s = r.cfg.sampling
    assert body["temperature"] == 1.0 and body["top_p"] == 0.95
    assert body["max_tokens"] == (s.thinking_max_tokens if first else s.max_tokens)  # reasoning counts against it
    assert body["chat_template_kwargs"] == {"enable_thinking": first}
    assert "speculative" not in json.dumps(body)
    msgs = body["messages"]
    assert msgs[0] == {"role": "system", "content": prompts.SYSTEM_PROMPT}
    assert msgs[1]["role"] == "user" and msgs[1]["content"].startswith("User request: Open the gear menu.")
    assert "Current screen:\nStep 0 | URL: /settings" in msgs[1]["content"]
    assert len(msgs) == 2


class _StampedGrounder(FakeGrounder):
    async def locate(self, img, target):
        loc = await super().locate(img, target)
        loc.prompt_sha, loc.usage = "abc1234567", {"prompt_tokens": 900, "completion_tokens": 12}
        return loc


@pytest.mark.asyncio
async def test_events_record_models_and_prompt_ids(tmp_path):
    """Additive fields for the console Logs page: which model and prompt produced each event."""
    r = await _run(tmp_path, [click("the gear icon"), done("Opened it.")], grounder=_StampedGrounder())
    for p in (e.payload for e in _of(r.events, "plan")):
        assert p["model"] == r.cfg.lightning.model and p["prompt_version"] == prompts.PROMPT_VERSION
        assert p["prompt_sha"] == prompts.PROMPT_SHA and len(p["prompt_sha"]) == 10
    (g,) = _of(r.events, "grounding")
    assert g.payload["prompt_sha"] == "abc1234567" and g.payload["usage"]["completion_tokens"] == 12
    obs = [e.payload for e in _of(r.events, "observation")]
    assert obs[0]["summary"] == {"model": "base"} and obs[1]["summary"] == {"model": "base"}  # two fresh calls


@pytest.mark.asyncio
async def test_reused_description_is_marked(tmp_path):
    r = await _run(tmp_path, [click("the gear icon"), done("Could not.", success=False)],
                   grounder=FakeGrounder(fail=True))
    obs = [e.payload for e in _of(r.events, "observation")]
    assert "reused" not in obs[0]["summary"]
    assert obs[1]["summary"] == {"model": "base", "reused": True}  # failed grounding: no new summary call
    (g,) = _of(r.events, "grounding")
    assert "prompt_sha" not in g.payload and "usage" not in g.payload  # empty optional fields are left out


@pytest.mark.asyncio
async def test_tool_choice_required_falls_back_to_auto(tmp_path):
    rejected = Reply(status=400, content="tool_choice 'required' is not supported by this server")
    r = await _run(tmp_path, [rejected, click("the gear icon"), done("Opened it.")])
    first, retried, second = r.fake.requests
    assert first["tool_choice"] == "required"
    assert retried["tool_choice"] == "auto" and second["tool_choice"] == "auto"  # for the rest of the run
    assert r.result.success_claimed is True


@pytest.mark.asyncio
async def test_tool_choice_auto_by_config(tmp_path):
    r = await _run(tmp_path, [done("Nothing to do.")], tool_choice="auto")
    assert r.fake.requests[0]["tool_choice"] == "auto"


# --- recovery ----------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_malformed_json_arguments(tmp_path):
    r = await _run(tmp_path, [Reply([("click", '{"target_description": "the gear')]),
                              click("the gear icon"), done("Opened it.")])
    assert r.result.success_claimed and len(r.browser.clicks) == 1
    msgs = r.fake.requests[1]["messages"]
    assert_pairing(msgs)
    (bad,) = _tool_results(msgs)
    assert bad["ok"] is False and bad["error"].startswith("invalid JSON arguments")
    assert "Step 0 |" in bad["observation"]  # the last observation, so the model keeps its bearings
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == "{}"
    tc = _of(r.events, "tool_call")[0].payload
    assert tc["valid"] is False and "invalid JSON" in tc["error"]
    assert "Expected click(target_description: string)" in bad["error"]
    s = _done(r.events)["stats"]
    # invalid calls execute nothing, so they are not failed actions
    assert (s["n_invalid_calls"], s["n_failed_actions"], s["n_recoveries"]) == (1, 0, 1)
    assert _thinking(r.fake) == [False, True, False]


@pytest.mark.asyncio
async def test_unknown_tool(tmp_path):
    r = await _run(tmp_path, [Reply([("open_settings", {"page": "accessibility"})]), done("Gave up.", success=False)])
    assert r.result.success_claimed is False and r.result.summary == "Gave up."
    assert r.browser.clicks == []
    (res,) = _tool_results(r.fake.requests[1]["messages"])
    assert res["ok"] is False and "unknown tool 'open_settings'" in res["error"]
    tc = _of(r.events, "tool_call")[0].payload
    assert tc["valid"] is False and tc["description"] == "Unknown action: open_settings"
    assert "click(target_description: string)" in res["error"]  # lists the real tools
    s = _done(r.events)["stats"]
    assert (s["n_invalid_calls"], s["n_failed_actions"]) == (1, 0)


@pytest.mark.asyncio
async def test_bad_arguments(tmp_path):
    r = await _run(tmp_path, [Reply([("click", {"target": "the gear icon"})]), click("the gear icon"), done("ok")])
    (res,) = _tool_results(r.fake.requests[1]["messages"])
    assert res["ok"] is False and "target_description" in res["error"]
    assert len(r.browser.clicks) == 1 and r.result.success_claimed


@pytest.mark.asyncio
async def test_reply_without_tool_call(tmp_path):
    r = await _run(tmp_path, [say("I will click the gear icon now."), click("the gear icon"), done("Opened it.")])
    assert r.result.success_claimed and r.result.steps == 3
    msgs = r.fake.requests[1]["messages"]
    assert msgs[-2] == {"role": "assistant", "content": "I will click the gear icon now."}
    assert msgs[-1] == {"role": "user", "content": NO_TOOL_CALL_CORRECTION}
    assert_pairing(r.fake.requests[2]["messages"])
    (err,) = _of(r.events, "error")
    assert err.payload["recoverable"] is True and err.payload["where"] == "planner"
    s = _done(r.events)["stats"]
    assert s["n_no_tool_call"] == 1 and s["n_recoveries"] == 1


@pytest.mark.asyncio
async def test_tool_call_written_in_content_is_recovered(tmp_path):
    xml = ("<tool_call>\n<function=click>\n<parameter=target_description>\nthe gear icon\n</parameter>\n"
           "</function>\n</tool_call>")
    r = await _run(tmp_path, [say(xml), done("Opened it.")])
    assert len(r.browser.clicks) == 1 and r.result.success_claimed
    tc = _of(r.events, "tool_call")[0].payload
    assert tc["from_content"] is True and tc["arguments"] == {"target_description": "the gear icon"}
    assistant = r.fake.requests[1]["messages"][2]
    assert assistant["content"] == ""
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"target_description": "the gear icon"}
    assert_pairing(r.fake.requests[1]["messages"])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,expected", [("retry", [False, True, False, False]), ("off", [False] * 4),
                                           ("on", [True] * 4)])
async def test_no_change_click_enables_thinking_for_next_call_only(tmp_path, mode, expected):
    r = await _run(tmp_path, [click("nowhere in particular"), click("the gear icon"), click("the bell icon"),
                              done("Opened it.")], thinking=mode)
    assert _thinking(r.fake) == expected
    assert [e.payload["thinking"] for e in _of(r.events, "plan")] == expected
    first = _tool_results(r.fake.requests[1]["messages"])[0]
    assert first["ok"] is True and NO_CHANGE_NOTE in first["observation"]
    s = _done(r.events)["stats"]
    assert s["n_no_change"] == 1 and s["n_recoveries"] == 1 and s["n_thinking_calls"] == sum(expected)


@pytest.mark.asyncio
async def test_truncated_thinking_reply_turns_thinking_off(tmp_path):
    cut = Reply(reasoning="Let me think about which element " * 50, finish_reason="length")
    r = await _run(tmp_path, [click("nowhere in particular"), cut, cut, click("the gear icon"), done("Opened it.")])
    # no change -> one thinking call; it is cut off -> the next call doesn't think; cut off again
    # (not thinking) -> one more thinking call
    assert _thinking(r.fake) == [False, True, False, True, False]
    assert [b["max_tokens"] for b in r.fake.requests] == [1024, 4096, 1024, 4096, 1024]
    assert r.fake.requests[2]["messages"][-1] == {"role": "user", "content": TRUNCATED_CORRECTION}
    errors = _of(r.events, "error")
    assert len(errors) == 2 and "cut off at the token limit" in errors[0].payload["message"]
    assert errors[0].payload["finish_reason"] == "length"
    s = _done(r.events)["stats"]
    assert (s["n_truncated"], s["n_no_tool_call"], s["n_thinking_calls"]) == (2, 2, 2)
    assert r.result.success_claimed


@pytest.mark.asyncio
async def test_uncapturable_screen_still_streams_an_observation(tmp_path):
    fake = FakeLightning([click("the gear icon"), done("Opened it.")])
    pool = FakePool(ShotFailsAfterClick(fails=2))
    cfg = AgentConfig().with_options(runs_dir=str(tmp_path))
    events = []

    async def collect(ev):
        events.append(ev)

    await run_task("Open the gear menu.", start_url=START_URL, cfg=cfg, on_event=collect, planner_client=fake.client,
                   grounder=FakeGrounder(), summarizer=FakeSummarizer(), pool=pool)
    obs = [e for e in _of(events, "observation") if e.step == 1]
    assert len(obs) == 1 and obs[0].screenshot_path is None
    assert "screen unavailable" in obs[0].payload["text"] and "could not be captured" in obs[0].payload["text"]
    s = _done(events)["stats"]
    assert (s["n_failed_actions"], s["n_no_change"], s["n_clicks"]) == (0, 0, 1)


@pytest.mark.asyncio
async def test_retry_never_thinks_twice_in_a_row_without_a_tool_call(tmp_path):
    r = await _run(tmp_path, [say("hmm")] * 4, max_steps=4)
    assert _thinking(r.fake) == [False, True, False, True]


@pytest.mark.asyncio
async def test_invalid_enum_enables_thinking(tmp_path):
    r = await _run(tmp_path, [Reply([("scroll", {"direction": "left", "amount": "page"})]),
                              type_text("Alex Kim", "the name field"), done("Done.")])
    assert _thinking(r.fake) == [False, True, False]
    assert [e[0] for e in r.browser.log[1:]] == ["click", "select_all", "type"]


@pytest.mark.asyncio
async def test_failed_action_enables_thinking(tmp_path):
    r = await _run(tmp_path, [click("the gear icon"), done("Gave up.", success=False)],
                   grounder=FakeGrounder(fail=True))
    assert _thinking(r.fake) == [False, True]
    assert r.browser.clicks == []
    action = _of(r.events, "action")[0].payload
    assert action["ok"] is False and "could not locate" in action["error"]
    (g,) = _of(r.events, "grounding")
    assert g.payload["error"] == "no point in reply" and g.overlay_path.endswith("step_01_overlay.png")
    s = _done(r.events)["stats"]
    assert (s["n_failed_actions"], s["n_invalid_calls"], s["n_actions"]) == (1, 0, 2)


@pytest.mark.asyncio
async def test_max_steps_is_a_hard_stop(tmp_path):
    r = await _run(tmp_path, [click("nowhere")] * 5, max_steps=3)
    assert r.result.steps == 3 and r.result.success_claimed is False
    assert len(r.fake.requests) == 3 and len(r.browser.clicks) == 3
    d = _done(r.events)
    assert d["reason"] == "max_steps" and "3-step limit" in d["summary"]
    assert r.result.summary == d["summary"] and r.browser.closed


@pytest.mark.asyncio
async def test_max_steps_counts_replies_without_tool_calls(tmp_path):
    r = await _run(tmp_path, [say("hmm")] * 5, max_steps=2)
    assert r.result.steps == 2 and len(r.fake.requests) == 2 and _done(r.events)["reason"] == "max_steps"


@pytest.mark.asyncio
async def test_two_tool_calls_only_first_runs(tmp_path):
    r = await _run(tmp_path, [Reply([click("the gear icon"), click("the bell icon")]), done("Opened it.")])
    assert r.browser.clicks == [("click", 1225.0, 30.0)]
    msgs = r.fake.requests[1]["messages"]
    assert_pairing(msgs)
    assert len(msgs[2]["tool_calls"]) == 2
    first, second = _tool_results(msgs)
    assert first["ok"] is True and second == {"ok": False, "error": NOT_EXECUTED}
    assert len(_of(r.events, "tool_call")) == 2  # one per step: the gear click, then done


# --- schema mismatch (vllm-metal doesn't enforce the tool schema) -------------------------------

def _kinds_at(events, step: int) -> list[str]:
    return [e.kind for e in events if e.step == step]


@pytest.mark.asyncio
async def test_wrong_argument_name_is_corrected_not_executed(tmp_path):
    r = await _run(tmp_path, [Reply([("click", {"target": "the gear icon"})]), click("the gear icon"),
                              done("Opened it.")])
    assert r.result.success_claimed and r.result.steps == 3
    assert r.grounder.calls == ["the gear icon"] and len(r.browser.clicks) == 1  # only the second call ran
    msgs = r.fake.requests[1]["messages"]
    assert_pairing(msgs)
    (res,) = _tool_results(msgs)
    assert res["ok"] is False and set(res) == {"ok", "error", "observation"}
    assert "unknown argument(s) target" in res["error"]
    assert "missing required argument(s) target_description" in res["error"]
    assert "Expected click(target_description: string)" in res["error"]
    assert res["observation"] == _of(r.events, "observation")[0].payload["text"]  # the unchanged last screen
    # the model's own (wrong) key goes back into history: never renamed silently
    assert json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"]) == {"target": "the gear icon"}

    assert _kinds_at(r.events, 1) == ["plan", "tool_call"]  # nothing executed: no grounding/action/observation
    tc = _of(r.events, "tool_call")[0].payload
    assert tc["valid"] is False and tc["error"] == res["error"] and tc["arguments"] == {"target": "the gear icon"}
    assert tc["description"] == "Invalid click call (not executed)"
    assert _kinds_at(r.events, 2) == ["plan", "tool_call", "grounding", "action", "observation"]
    obs2 = _of(r.events, "observation")[1]
    assert obs2.step == 2 and obs2.payload["observation"]["step"] == 2  # numbering follows Lightning calls
    assert obs2.screenshot_path.endswith("step_02.png")

    s = _done(r.events)["stats"]
    assert (s["n_tool_calls"], s["n_invalid_calls"], s["n_actions"], s["n_failed_actions"]) == (3, 1, 2, 0)
    assert s["n_recoveries"] == 1
    assert _thinking(r.fake) == [False, True, False]


@pytest.mark.asyncio
async def test_done_without_success_is_corrected(tmp_path):
    r = await _run(tmp_path, [Reply([("done", {"summary": "Turned it on."})]), done("Turned it on.")])
    assert r.result.success_claimed is True and r.result.steps == 2
    (res,) = _tool_results(r.fake.requests[1]["messages"])
    assert res["ok"] is False and "missing required argument(s) success" in res["error"]
    assert "Expected done(summary: string, success: boolean)" in res["error"]
    d = _done(r.events)
    assert d["reason"] == "done" and d["stats"]["n_invalid_calls"] == 1
    assert [e.payload["valid"] for e in _of(r.events, "tool_call")] == [False, True]


@pytest.mark.asyncio
async def test_invalid_call_kinds_are_all_counted(tmp_path):
    bad = [Reply([("click", {"target": "the gear icon"})]),                      # wrong key
           Reply([("click", {"target_description": "the gear", "x": 10})]),     # extra key
           Reply([("scroll", {"direction": "sideways", "amount": "page"})]),    # bad enum
           Reply([("done", {"summary": "ok", "success": "maybe"})]),            # bad type
           Reply([("open_settings", {})]),                                       # unknown tool
           Reply([("type_text", '{"text": "Alex')])]                            # bad JSON
    r = await _run(tmp_path, bad + [done("Gave up.", success=False)])
    assert r.browser.log == [("goto", START_URL)] and r.grounder.calls == []  # nothing ran
    s = _done(r.events)["stats"]
    assert (s["n_tool_calls"], s["n_invalid_calls"], s["n_actions"], s["n_failed_actions"]) == (7, 6, 1, 0)
    assert [e.payload["valid"] for e in _of(r.events, "tool_call")] == [False] * 6 + [True]
    assert [e.step for e in _of(r.events, "action")] == [7]  # only done executed
    assert _thinking(r.fake) == [False] + [True] * 6
    last = r.fake.requests[-1]["messages"]
    assert_pairing(last)
    errors = [res["error"] for res in _tool_results(last)]
    assert "Expected click(target_description: string)" in errors[0] and "unknown argument(s) x" in errors[1]
    assert 'scroll(direction: "up"|"down", amount: "small"|"page")' in errors[2]
    assert "done.success must be true or false" in errors[3] and "unknown tool 'open_settings'" in errors[4]
    assert errors[5].startswith("invalid JSON arguments") and "type_text(text: string" in errors[5]


@pytest.mark.asyncio
async def test_invalid_calls_use_up_steps(tmp_path):
    r = await _run(tmp_path, [Reply([("click", {"target": "the gear icon"})])] * 5, max_steps=3)
    d = _done(r.events)
    assert d["reason"] == "max_steps" and r.result.steps == 3 and len(r.fake.requests) == 3
    assert d["stats"]["n_invalid_calls"] == 3 and r.browser.clicks == []


@pytest.mark.asyncio
async def test_collapsed_invalid_call_shows_argument_names(tmp_path):
    r = await _run(tmp_path, [Reply([("click", {"target": "the gear icon"})])] * 7 + [done("x", success=False)])
    msgs = r.fake.requests[7]["messages"]
    assert_pairing(msgs)
    assert 'Earlier steps:\n1. click(target="the gear icon") -> invalid call, not executed' in msgs[1]["content"]


# --- context budget ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_budget(tmp_path):
    targets = ["the gear icon", "the bell icon"] * 5
    r = await _run(tmp_path, [click(t) for t in targets[:9]] + [done("Clicked around.")])
    assert len(r.fake.requests) == 10 and r.result.success_claimed
    for body in r.fake.requests:
        assert_pairing(body["messages"])

    seventh = r.fake.requests[6]["messages"]  # 6 records: all verbatim
    assert len(seventh) == 2 + 6 * 2 and "Earlier steps" not in seventh[1]["content"]
    assert "Current screen:\nStep 0 |" in seventh[1]["content"]

    last = r.fake.requests[9]["messages"]  # 9 records: 3 collapsed + 6 verbatim
    assert len(last) == 2 + 6 * 2
    first_user = last[1]["content"]
    assert first_user.startswith("User request: Open the gear menu.")
    assert INITIAL_SCREEN_OMITTED in first_user and "Step 0 |" not in first_user
    earlier = first_user.split("Earlier steps:\n", 1)[1].splitlines()
    assert earlier == ['1. click("the gear icon") -> ok, screen changed',
                       '2. click("the bell icon") -> ok, screen changed',
                       '3. click("the gear icon") -> ok, screen changed']
    verbatim_ids = [m["tool_call_id"] for m in last if m["role"] == "tool"]
    assert len(verbatim_ids) == 6
    assert "Step 9 |" in json.loads(last[-1]["content"])["observation"]  # the latest observation is kept


def _plan(calls: list[ToolCall], content: str = "") -> PlanResult:
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments}}
                             for c in calls]
    return PlanResult(calls, content, None, False, 1.0, assistant_message=msg)


def test_collapsed_history_keeps_latest_screen_when_only_corrections_remain():
    planner = LightningPlanner(AgentConfig(), client=SimpleNamespace())
    planner.start("Do it.", "Step 0 | initial")
    call = ToolCall("c1", "click", {"target_description": "x"}, '{"target_description": "x"}')
    planner.record_step(_plan([call]), [("c1", {"ok": True, "observation": "Step 1 | after click"})], "click(x) -> ok")
    for _ in range(6):
        planner.record_correction(_plan([], "thinking out loud"), NO_TOOL_CALL_CORRECTION)
    msgs = planner.messages()
    assert_pairing(msgs)
    assert "Current screen:\nStep 1 | after click" in msgs[1]["content"]
    assert "Earlier steps:\n1. click(x) -> ok" in msgs[1]["content"]
    assert len(msgs) == 2 + 6 * 2


def test_record_step_answers_every_tool_call_id():
    planner = LightningPlanner(AgentConfig(), client=SimpleNamespace())
    planner.start("Do it.", "Step 0")
    calls = [ToolCall("a", "click", {"target_description": "x"}, "{}"), ToolCall("b", "done", {}, "{}")]
    planner.record_step(_plan(calls), [("a", {"ok": True, "observation": "o"}), ("zzz", {"ok": True})], "click")
    msgs = planner.messages()
    assert_pairing(msgs)
    assert [json.loads(m["content"]).get("error") for m in msgs if m["role"] == "tool"] == [None, NOT_EXECUTED]


# --- content parsing ---------------------------------------------------------------------------

@pytest.mark.parametrize("content,name,args,rest", [
    ("<tool_call><function=click><parameter=target_description>the gear icon</parameter></function></tool_call>",
     "click", {"target_description": "the gear icon"}, ""),
    ("Let me finish.\n<function=done>\n<parameter=summary>\nAll set.\n</parameter>\n<parameter=success>\ntrue\n"
     "</parameter>\n</function>", "done", {"summary": "All set.", "success": "true"}, "Let me finish."),
    ('{"name": "click", "arguments": {"target_description": "the Save button"}}',
     "click", {"target_description": "the Save button"}, ""),
    ('<tool_call>\n{"name": "press_key", "arguments": "{\\"key\\": \\"Enter\\"}"}\n</tool_call>',
     "press_key", {"key": "Enter"}, ""),
    ('Sure: {"type": "function", "function": {"name": "scroll", "parameters": {"direction": "down", "amount": "page"}}}',
     "scroll", {"direction": "down", "amount": "page"}, "Sure:"),
])
def test_parse_content_tool_calls(content, name, args, rest):
    calls, remaining = parse_content_tool_calls(content)
    assert len(calls) == 1
    c = calls[0]
    assert (c.name, c.arguments, c.from_content, c.parse_error) == (name, args, True, None)
    assert remaining == rest
    assert validate_args(c.name, c.arguments)[1] is None  # "true" strings are coerced by validate_args


@pytest.mark.parametrize("content", ["I will click the gear icon.", "", "The value is {not json}.",
                                     '{"summary": "no name key"}'])
def test_parse_content_without_calls(content):
    calls, remaining = parse_content_tool_calls(content)
    assert calls == [] and remaining == content


# --- endpoint behavior -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parallel_tool_calls_rejected_is_dropped(tmp_path):
    replies = iter([click("the gear icon"), done("Opened it.")])

    def server(body: dict) -> Reply:
        if "parallel_tool_calls" in body:
            return Reply(status=400, content="Unsupported parameter: parallel_tool_calls")
        return Reply([next(replies)])

    fake = FakeLightning(server)
    r = await _run(tmp_path, None, client=fake.client)
    assert r.result.success_claimed and r.result.steps == 2
    assert ["parallel_tool_calls" in b for b in fake.requests] == [True, False, False]
    assert _of(r.events, "error") == []


@pytest.mark.asyncio
async def test_two_consecutive_api_errors_stop_the_run(tmp_path):
    # each step retries a 5xx once, so two failed steps are four failed requests
    r = await _run(tmp_path, [Reply(status=500, content="boom"), Reply(status=503, content="down")] * 2
                   + [done("never")])
    assert r.result.success_claimed is False and r.result.steps == 2 and len(r.fake.requests) == 4
    errors = _of(r.events, "error")
    assert [e.payload["recoverable"] for e in errors] == [True, False]
    assert "InternalServerError" in errors[0].payload["message"]
    d = _done(r.events)
    assert d["reason"] == "error" and d["stats"]["n_api_errors"] == 2
    assert r.result.summary.startswith("Stopped after an error") and r.browser.closed


@pytest.mark.asyncio
async def test_single_api_error_is_recoverable(tmp_path):
    r = await _run(tmp_path, [Reply(status=500, content="boom")] * 2 + [click("the gear icon"), done("Opened it.")])
    assert r.result.success_claimed and r.result.steps == 3
    assert _done(r.events)["stats"]["n_api_errors"] == 1


@pytest.mark.asyncio
async def test_transient_error_is_retried_once_within_the_step(tmp_path):
    r = await _run(tmp_path, [Reply(status=502, content="bad gateway"), click("the gear icon"), done("Opened it.")])
    assert r.result.success_claimed and r.result.steps == 2 and len(r.fake.requests) == 3
    assert _done(r.events)["stats"]["n_api_errors"] == 0 and _of(r.events, "error") == []


@pytest.mark.asyncio
async def test_rate_limit_waits_without_costing_a_step(tmp_path):
    r = await _run(tmp_path, [Reply(status=429, content="slow down")] * 3 + [click("the gear icon"), done("Opened it.")],
                   max_steps=2)
    assert r.result.success_claimed and r.result.steps == 2 and len(r.fake.requests) == 5
    d = _done(r.events)
    assert d["stats"]["n_api_errors"] == 0 and _of(r.events, "error") == []
    assert "wait_ms" in _of(r.events, "plan")[0].payload


@pytest.mark.asyncio
async def test_rate_limit_honors_retry_after(tmp_path, monkeypatch):
    monkeypatch.setattr(planner_mod, "RATE_LIMIT_WAIT_S", 1000.0)  # would hang if Retry-After were ignored
    r = await asyncio.wait_for(_run(tmp_path, [Reply(status=429, headers={"retry-after": "0"}),
                                               done("Nothing to do.")]), 10)
    assert r.result.success_claimed and len(r.fake.requests) == 2


@pytest.mark.asyncio
async def test_persistent_rate_limit_becomes_an_api_error(tmp_path):
    r = await _run(tmp_path, [Reply(status=429, content="quota")] * 20, max_steps=3)
    n = planner_mod.RATE_LIMIT_MAX_WAITS + 1
    assert len(r.fake.requests) == 2 * n and r.result.steps == 2
    d = _done(r.events)
    assert d["reason"] == "error" and d["stats"]["n_api_errors"] == 2 and "RateLimitError" in r.result.summary


@pytest.mark.asyncio
async def test_timeout_is_not_retried(tmp_path):
    def server(body: dict) -> Reply:
        raise httpx.ReadTimeout("slow")

    fake = FakeLightning(server)
    r = await _run(tmp_path, None, client=fake.client, max_steps=3)
    assert len(fake.requests) == 2 and r.result.steps == 2  # one request per step, then fatal
    assert "APITimeoutError" in _of(r.events, "error")[0].payload["message"]


def test_owned_client_does_not_retry_on_its_own():
    planner = LightningPlanner(AgentConfig().with_options(lightning="local"))
    assert planner.client.max_retries == 0 and planner.client.timeout == 180


@pytest.mark.asyncio
async def test_start_event_records_the_served_checkpoint(tmp_path):
    r = await _run(tmp_path, [done("Nothing to do.")], lightning={"model": "fake-lightning"})
    cfg = _of(r.events, "start")[0].payload["config"]
    assert cfg["lightning"] == {"base_url": r.cfg.lightning.base_url, "model": "fake-lightning",
                                "root": r.fake.root}
    assert cfg["prompt_version"] == prompts.PROMPT_VERSION
    assert cfg["sampling"]["thinking_max_tokens"] == r.cfg.sampling.thinking_max_tokens
    assert cfg["history_full_steps"] == r.cfg.history_full_steps and "lightning_s" in cfg["timeouts"]

    other = await _run(tmp_path, [done("Nothing to do.")])  # the configured model isn't served there
    lightning = _of(other.events, "start")[0].payload["config"]["lightning"]
    assert lightning["root"] is None and "not in the server's /models list" in lightning["server_error"]


@pytest.mark.asyncio
async def test_reasoning_is_previewed_but_not_sent_back(tmp_path):
    long = "The gear icon usually opens settings. " * 40
    r = await _run(tmp_path, [Reply([click("the gear icon")], reasoning=long), done("Opened it.")], thinking="on")
    plan = _of(r.events, "plan")[0].payload
    assert plan["thinking"] is True and plan["reasoning"].startswith("The gear icon")
    assert len(plan["reasoning"]) <= 603 and plan["reasoning"].endswith("...")
    assistant = r.fake.requests[1]["messages"][2]
    assert set(assistant) == {"role", "content", "tool_calls"}
    assert "gear icon usually" not in json.dumps(r.fake.requests[1]["messages"])


@pytest.mark.asyncio
async def test_on_event_errors_do_not_break_the_run(tmp_path):
    async def broken(ev):
        raise RuntimeError("console went away")

    r = await _run(tmp_path, [click("the gear icon"), done("Opened it.")], on_event=broken)
    assert r.result.success_claimed and _lines(r.trace)[-1]["kind"] == "done"


class _HangingClient:
    """AsyncOpenAI-shaped client whose call never returns (to test cancellation)."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=self)
        self.started = asyncio.Event()

    async def create(self, **kwargs):
        self.started.set()
        await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_cancel_emits_stopped_and_closes(tmp_path):
    client, pool = _HangingClient(), FakePool()
    cfg = AgentConfig().with_options(runs_dir=str(tmp_path))
    task = asyncio.create_task(run_task("Wait.", start_url=START_URL, cfg=cfg, planner_client=client,
                                        grounder=FakeGrounder(), summarizer=FakeSummarizer(), pool=pool,
                                        run_id="cancel-me"))
    await asyncio.wait_for(client.started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    last = _lines(tmp_path / "cancel-me")[-1]
    assert last["kind"] == "done" and last["payload"]["reason"] == "stopped"
    assert last["payload"]["success_claimed"] is False and pool.browser.closed


@pytest.mark.asyncio
async def test_second_cancel_during_cleanup_still_finishes_it(tmp_path):
    class SlowCloseBrowser(type(FakePool().browser)):
        async def close(self):
            closing.set()
            await asyncio.sleep(0.3)
            self.closed = True

    closing = asyncio.Event()
    client, pool = _HangingClient(), FakePool(SlowCloseBrowser())
    cfg = AgentConfig().with_options(runs_dir=str(tmp_path))
    task = asyncio.create_task(run_task("Wait.", start_url=START_URL, cfg=cfg, planner_client=client,
                                        grounder=FakeGrounder(), summarizer=FakeSummarizer(), pool=pool,
                                        run_id="cancel-twice"))
    await asyncio.wait_for(client.started.wait(), 5)
    task.cancel()
    await asyncio.wait_for(closing.wait(), 5)
    task.cancel()  # Stop clicked again while the browser is closing
    with pytest.raises(asyncio.CancelledError):
        await task
    lines = _lines(tmp_path / "cancel-twice")
    assert lines[-1]["kind"] == "done" and lines[-1]["payload"]["reason"] == "stopped"
    assert pool.browser.closed


@pytest.mark.asyncio
async def test_setup_failure_is_reported_not_raised(tmp_path):
    class BrokenPool:
        async def new_session(self, viewport, **kwargs):
            raise RuntimeError("chromium missing")

    cfg = AgentConfig().with_options(runs_dir=str(tmp_path))
    result = await run_task("x", start_url=START_URL, cfg=cfg, planner_client=FakeLightning([]).client,
                            grounder=FakeGrounder(), summarizer=FakeSummarizer(), pool=BrokenPool())
    assert result.success_claimed is False and "chromium missing" in result.summary
    kinds = [ln["kind"] for ln in _lines(Path(result.trace_dir))]
    assert kinds == ["start", "error", "done"]


@pytest.mark.asyncio
async def test_api_key_never_reaches_events(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_TEST_LIGHTNING_KEY", SECRET)
    cfg_key = {"lightning": {"api_key_env": "AGENT_TEST_LIGHTNING_KEY"}}
    r = await _run(tmp_path, [click("the gear icon"), done("Opened it.")], **cfg_key)
    assert SECRET not in (r.trace / "events.jsonl").read_text()
    planner = LightningPlanner(r.cfg)  # default client: reads the key
    try:
        assert SECRET not in planner._describe(RuntimeError(f"401: bad key {SECRET}"))
    finally:
        await planner.aclose()
