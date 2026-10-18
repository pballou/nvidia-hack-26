"""agent/clarify.py: the pre-flight "understand the request" call, its run_task wiring, and its replay.

No network: FakeLightning answers. The replay tests feed the Lightning replies recorded by
scripts/clarify_eval.py --record back through clarify(), so the decision logic reproduces each
live pass: fixtures/clarify_cases.json (the acceptance set: the 12 task phrasings, spoken
variants, the spec's ask cases), clarify_heldout.json and clarify_heldout2.json (blind sets).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from agent import clarify as clarify_mod
from agent.clarify import (CLARIFY_PROMPT_SHA, CLARIFY_PROMPT_VERSION, CLARIFY_SAMPLING, DEFAULT_QUESTION,
                           MAX_CLARIFICATIONS, Clarification, clarify, follow_up_request)
from agent.config import AgentConfig, load_config
from agent.planner import LightningPlanner
from agent.testing.fake_lightning import FakeLightning, Reply, done
from agent.tests.test_planner import START_URL, _lines
from agent.tests.test_session import FakeGrounder, FakePool, FakeSummarizer

FIXTURES = Path(__file__).parent / "fixtures"
CASES = FIXTURES / "clarify_cases.json"
RECORDED = [CASES, FIXTURES / "clarify_heldout.json", FIXTURES / "clarify_heldout2.json"]


def proceed(text: str) -> tuple[str, dict]:
    return ("proceed", {"normalized_request": text})


def ask(question: str) -> tuple[str, dict]:
    return ("ask_user", {"question": question})


async def _clarify(script, request: str = "Pay $50 toward my bill.") -> tuple[Clarification, FakeLightning]:
    fake = FakeLightning(script)
    planner = LightningPlanner(AgentConfig(), client=fake.client)
    return await clarify(planner, request), fake


# --- the call ----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_is_one_short_deterministic_tool_call_without_thinking():
    c, fake = await _clarify([proceed("Pay $50 toward my bill.")])
    body = fake.last_request
    assert len(fake.requests) == 1
    assert [t["function"]["name"] for t in body["tools"]] == ["proceed", "ask_user"]
    assert body["tool_choice"] == "required" and body["parallel_tool_calls"] is False
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert (body["temperature"], body["top_p"], body["max_tokens"]) == (0.0, 1.0, CLARIFY_SAMPLING["max_tokens"])
    assert body["messages"][0] == {"role": "system", "content": clarify_mod.SYSTEM_PROMPT}
    assert body["messages"][1] == {"role": "user", "content": "Request: Pay $50 toward my bill."}
    assert c.action == "proceed" and not c.asks and c.normalized_request == "Pay $50 toward my bill."
    assert c.latency_ms is not None and c.usage["completion_tokens"] > 0 and c.error is None


@pytest.mark.asyncio
async def test_ask_user_returns_the_question():
    c, _ = await _clarify([ask("How much would you like to pay?")], "Pay something toward my bill")
    assert c.asks and c.question == "How much would you like to pay?"
    assert c.tool_call == {"name": "ask_user", "arguments": {"question": "How much would you like to pay?"}}


@pytest.mark.asyncio
@pytest.mark.parametrize("reply, error", [
    (Reply(status=500, content="boom"), "InternalServerError"),
    (Reply(content="Sure, I can do that."), "no tool call"),
    (Reply([("plan_steps", {"x": "y"})]), "unknown tool 'plan_steps'"),
])
async def test_failures_proceed(reply, error):
    c, _ = await _clarify([reply])
    assert c.action == "proceed" and error in c.error and c.question is None


@pytest.mark.asyncio
async def test_malformed_proceed_still_proceeds():
    c, _ = await _clarify([Reply([("proceed", '{"normalized_request": ')])])
    assert c.action == "proceed" and c.normalized_request is None


@pytest.mark.asyncio
async def test_ask_user_with_a_wrong_argument_name_or_no_question():
    c, _ = await _clarify([Reply([("ask_user", {"q": "  Which   setting? "})])])
    assert c.asks and c.question == "Which setting?"
    c, _ = await _clarify([Reply([("ask_user", {})])])
    assert c.asks and c.question == DEFAULT_QUESTION
    c, _ = await _clarify([ask("x" * 500)])
    assert len(c.question) == clarify_mod.MAX_QUESTION_CHARS


@pytest.mark.asyncio
async def test_call_written_as_text_is_recovered():
    c, _ = await _clarify([Reply(content='{"name": "ask_user", "arguments": {"question": "Turn off what?"}}')],
                          "turn it off")
    assert c.asks and c.question == "Turn off what?"


@pytest.mark.asyncio
async def test_tool_choice_rejected_falls_back_to_auto_for_the_planner_too():
    fake = FakeLightning([Reply(status=400, content="tool_choice 'required' is not supported"), proceed("x")])
    planner = LightningPlanner(AgentConfig(), client=fake.client)
    c = await clarify(planner, "Turn on high contrast.")
    assert c.action == "proceed" and c.error is None
    assert [r["tool_choice"] for r in fake.requests] == ["required", "auto"]
    assert planner.required_rejected and planner.tool_choice == "auto"


@pytest.mark.asyncio
async def test_skips_without_a_call():
    c, fake = await _clarify([], "   ")
    assert c.asks and c.skipped == "empty request" and fake.requests == []
    clarified = "Pay part of it"
    for answer in ["some", "fifty"][:MAX_CLARIFICATIONS]:
        clarified = follow_up_request(clarified, answer)
    c, fake = await _clarify([], clarified)
    assert c.action == "proceed" and c.skipped and fake.requests == []
    c, fake = await _clarify([proceed("Pay $50")], follow_up_request("Pay part of it", "fifty dollars"))
    assert len(fake.requests) == 1  # one answer: still checked


def test_follow_up_request():
    first = follow_up_request("Pay something toward my bill", "  fifty\ndollars ")
    assert first == "Pay something toward my bill (clarified: fifty dollars)"
    assert follow_up_request(first, "yes") == first + " (clarified: yes)"
    assert follow_up_request("Turn it off", "  ") == "Turn it off"


def test_event_payload_drops_empty_fields():
    c = Clarification("ask", question="Turn off what?", latency_ms=812.34, usage={"completion_tokens": 9})
    p = c.event_payload("turn it off", "lightning")
    assert p == {"action": "ask", "request": "turn it off", "model": "lightning", "prompt_version": CLARIFY_PROMPT_VERSION,
                 "prompt_sha": CLARIFY_PROMPT_SHA, "question": "Turn off what?", "usage": {"completion_tokens": 9},
                 "latency_ms": 812.3}


# --- run_task wiring ---------------------------------------------------------------------------


def _script(clarify_reply, *planner_replies):
    """Callable FakeLightning script: the clarify call by its tools, then the planner's replies in order."""
    queue = list(planner_replies)

    def reply(body):
        if body["tools"][0]["function"]["name"] == "proceed":
            return clarify_reply
        return queue.pop(0) if queue else Reply([done("script exhausted", success=False)])
    return reply


async def _run(tmp_path: Path, script, request: str = "Pay something toward my bill", **kwargs):
    from agent.core import run_task

    fake = FakeLightning(script)
    pool = FakePool()
    events = []

    async def collect(ev):
        events.append(ev)

    cfg = AgentConfig().with_options(runs_dir=str(tmp_path), **kwargs.pop("cfg", {}))
    result = await run_task(request, start_url=START_URL, cfg=cfg, on_event=collect, planner_client=fake.client,
                            grounder=FakeGrounder(), summarizer=FakeSummarizer(), pool=pool, **kwargs)
    return result, events, fake, pool


@pytest.mark.asyncio
async def test_ask_ends_the_run_before_the_browser_opens(tmp_path):
    result, events, fake, pool = await _run(tmp_path, _script(Reply([ask("How much would you like to pay?")])),
                                            cfg={"clarify": True})
    assert [e.kind for e in events] == ["start", "clarify", "done"]
    assert pool.viewports == [] and len(fake.requests) == 1   # no browser session, no planner call
    cl, dn = events[1].payload, events[2].payload
    assert cl["action"] == "ask" and cl["question"] == "How much would you like to pay?"
    assert cl["request"] == "Pay something toward my bill" and cl["prompt_sha"] == CLARIFY_PROMPT_SHA
    assert events[1].timings_ms["clarify"] >= 0
    assert dn["reason"] == "needs_input" and dn["question"] == dn["summary"] == "How much would you like to pay?"
    assert dn["request"] == "Pay something toward my bill" and dn["steps"] == 0 and dn["success_claimed"] is False
    assert dn["stats"]["latency_ms"]["clarify"]
    assert (result.reason, result.question, result.steps) == ("needs_input", "How much would you like to pay?", 0)
    assert [e["kind"] for e in _lines(Path(result.trace_dir))] == ["start", "clarify", "done"]


@pytest.mark.asyncio
async def test_proceed_runs_the_loop_on_the_users_own_words(tmp_path):
    script = _script(Reply([proceed("Pay $50 toward my bill.")]), Reply([done("Paid $50.")]))
    result, events, fake, pool = await _run(tmp_path, script, request="pay fifty dollars towards my bill",
                                            cfg={"clarify": True})
    kinds = [e.kind for e in events]
    assert kinds[:3] == ["start", "clarify", "observation"] and kinds[-1] == "done"
    assert events[1].payload["normalized_request"] == "Pay $50 toward my bill."
    assert [r["tools"][0]["function"]["name"] for r in fake.requests] == ["proceed", "click"]
    assert "User request: pay fifty dollars towards my bill" in fake.requests[1]["messages"][1]["content"]
    assert result.reason == "done" and result.success_claimed and result.question is None


@pytest.mark.asyncio
async def test_a_failed_clarify_call_proceeds(tmp_path):
    script = _script(Reply(status=500, content="down"), Reply([done("Done.")]))
    result, events, _, _ = await _run(tmp_path, script, cfg={"clarify": True})
    assert events[1].kind == "clarify" and events[1].payload["action"] == "proceed" and events[1].payload["error"]
    assert result.reason == "done"


@pytest.mark.asyncio
async def test_clarify_off_makes_no_call_and_records_the_parent(tmp_path):
    result, events, fake, _ = await _run(tmp_path, _script(Reply([ask("never asked")]), Reply([done("Done.")])),
                                         parent_run_id="20261017-230000-abc123")
    assert "clarify" not in [e.kind for e in events]
    assert [r["tools"][0]["function"]["name"] for r in fake.requests] == ["click"]
    assert events[0].payload["parent_run_id"] == "20261017-230000-abc123"
    assert events[0].payload["config"]["clarify"] is False and result.reason == "done"


# --- where it is on and off --------------------------------------------------------------------


def test_flag_defaults_and_values(monkeypatch):
    monkeypatch.delenv("AGENT_CLARIFY", raising=False)
    monkeypatch.delenv("AGENT_CONFIG", raising=False)
    assert AgentConfig().clarify is False
    assert load_config().clarify is False                     # configs/agent.yaml: off (held-out false asks)
    assert load_config(env={"AGENT_CLARIFY": "on"}).clarify is True
    assert AgentConfig().with_options(clarify="on").clarify is True
    with pytest.raises(ValueError, match="clarify"):
        AgentConfig(clarify="maybe")


def test_cli_flag():
    from agent.__main__ import config_from_args, main  # noqa: F401
    import argparse

    args = argparse.Namespace(config=None, grounder=None, summary=None, thinking=None, lightning=None, max_steps=None,
                              clarify="off", headed=False, use_oracle_grounder=False)
    assert config_from_args(args).clarify is False


# --- replay of the recorded live pass ----------------------------------------------------------


def _cases(path: Path = CASES) -> dict:
    return json.loads(path.read_text())


@pytest.mark.parametrize("path", RECORDED, ids=lambda p: p.stem)
def test_recorded_pass_is_for_the_current_prompt(path):
    data = _cases(path)
    assert data["prompt_sha"] == CLARIFY_PROMPT_SHA, "clarify prompt changed: rerun scripts/clarify_eval.py --record"
    assert all(c.get("reply") for c in data["cases"]), "cases without a recorded reply: rerun --record"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", RECORDED, ids=lambda p: p.stem)
async def test_replay_reproduces_the_live_decisions(path):
    data = _cases(path)
    counts: Counter = Counter()
    for case in data["cases"]:
        r = case["reply"]
        reply = Reply([(r["name"], r["arguments"])]) if r.get("name") else Reply(content=r.get("content", ""))
        c, _ = await _clarify([reply], case["text"])
        assert c.action == case["live_action"], case["text"]
        counts[(case["expect"], c.action)] += 1
    assert counts == Counter({tuple(k.split("->")): v for k, v in data["confusion"].items()})


def test_live_pass_meets_the_acceptance_bar():
    """0 false asks on the 12 task phrasings and the spoken variants; the spec's ask cases ask."""
    cases = _cases()["cases"]
    false_asks = [c["text"] for c in cases if c["expect"] == "proceed" and c["live_action"] == "ask"]
    assert false_asks == []
    task_ids = {c["task_id"] for c in cases if c["source"] == "task"}
    assert len(task_ids) == 12
    must_ask = [c for c in cases if c["source"] == "spec"]
    assert {c["text"] for c in must_ask} >= {"Allors towards my bill.", "Pay something toward my bill", "turn it off"}
    assert [c["text"] for c in must_ask if c["live_action"] != "ask"] == []
