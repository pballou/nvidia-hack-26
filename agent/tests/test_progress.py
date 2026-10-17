"""agent/progress.py (the "stuck" stop) as a unit, on real traces, and inside run_task."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.progress import CLICK_RADIUS_PX, STUCK_WARNING, ProgressTracker
from agent.testing.fake_lightning import click, done
from agent.tests.test_planner import _of, _run

FIXTURES = Path(__file__).parent / "fixtures" / "progress_traces.json"


def feed(tr: ProgressTracker, name: str, target: str = "x", *, ok: bool = True, changed: bool | None = False,
         xy: tuple[float, float] | None = None) -> str | None:
    return tr.record(name, {"target_description": target}, ok=ok, screen_changed=changed, click_xy=xy)


def test_four_in_a_row_warns_then_three_more_stop():
    tr = ProgressTracker()
    verdicts = [feed(tr, "click", f"target {i}", xy=(100.0 * i, 50.0)) for i in range(7)]
    assert verdicts == [None, None, None, "warn", None, None, "stuck"]
    assert tr.reason == "4 actions in a row changed nothing" and tr.n_warnings == 1
    assert "target 3" in tr.warning() or "4 actions" in tr.warning()
    s = tr.summary()
    assert s.startswith("I'm stuck: 4 actions in a row changed nothing") and "What would you like me to do next?" in s
    assert 'click "target 6"' in s and 'click "target 1"' not in s  # the last 5 distinct attempts


def test_same_target_three_times_warns_early():
    tr = ProgressTracker()
    assert [feed(tr, "click", "the L option", xy=(1100.0 + 40 * i, 440.0)) for i in range(3)] == [None, None, "warn"]
    assert tr.reason == 'click "the L option" made no progress 3 times'


def test_nearby_clicks_with_different_wording_count_as_the_same_action():
    tr = ProgressTracker()
    r = CLICK_RADIUS_PX
    assert feed(tr, "click", "the L option", xy=(1139.0, 442.0)) is None
    assert feed(tr, "click", "the L button at the right", xy=(1139.0 + r * 0.6, 442.0)) is None
    assert feed(tr, "click", "Large text", xy=(1139.0, 442.0 + r * 0.6)) == "warn"
    far = ProgressTracker()
    assert [feed(far, "click", f"t{i}", xy=(1139.0 + 3 * r * i, 442.0)) for i in range(3)] == [None, None, None]


def test_a_screen_change_clears_the_stretch_and_the_warning():
    tr = ProgressTracker()
    for i in range(4):
        feed(tr, "click", f"a{i}", xy=(10.0 * i * 5, 0.0))
    assert tr.warned_at == 4
    assert feed(tr, "click", "the right one", changed=True) is None
    assert (tr.streak, tr.warned_at, tr.reason, tr.no_progress) == (0, None, None, [])
    assert [feed(tr, "click", f"b{i}", xy=(500.0 + 50 * i, 0.0)) for i in range(3)] == [None, None, None]


def test_failed_actions_count_and_unknown_change_is_neutral():
    tr = ProgressTracker()
    assert feed(tr, "click", "a", ok=False, changed=None) is None      # grounding failed: no progress
    assert feed(tr, "click", "b", ok=True, changed=None) is None       # couldn't capture: ignored
    assert tr.streak == 1
    assert tr.record("done", {"summary": "s", "success": True}, ok=True, screen_changed=None) is None
    for t in ("c", "d"):
        feed(tr, "click", t, xy=(float(ord(t)) * 30, 0.0))
    assert feed(tr, "scroll", "x") == "warn"  # 4th no-progress action
    other = ProgressTracker()
    for _ in range(2):
        other.record("scroll", {"direction": "down", "amount": "page"}, ok=True, screen_changed=False)
    assert other.record("scroll", {"direction": "down", "amount": "page"}, ok=True, screen_changed=False) == "warn"


@pytest.mark.parametrize("trace", json.loads(FIXTURES.read_text()), ids=lambda t: t["run_id"])
def test_real_traces(trace):
    """Real runs (Sat 2026-10-17): stuck loops stop; runs that recovered after 5-6 no-change clicks don't."""
    tr = ProgressTracker()
    verdicts = []
    for a in trace["actions"]:
        c = a["click"] or {}
        verdicts.append(tr.record(a["name"], a["arguments"], ok=a["ok"], screen_changed=a["screen_changed"],
                                  click_xy=(c["x"], c["y"]) if "x" in c else None))
    assert ("stuck" in verdicts) == trace["expect_stuck"]
    if trace["verified_success"]:
        assert "stuck" not in verdicts  # never cut off a run that succeeded


@pytest.mark.asyncio
async def test_run_task_warns_then_stops_as_stuck(tmp_path):
    r = await _run(tmp_path, [click("nowhere in particular")] * 8 + [done("Never reached.")])
    d = _of(r.events, "done")[-1].payload
    assert d["reason"] == "stuck" and d["success_claimed"] is False and d["steps"] == 6
    assert d["summary"].startswith("I'm stuck:") and "What would you like me to do next?" in d["summary"]
    assert d["stats"]["n_stuck_warnings"] == 1 and d["stats"]["n_stuck_stops"] == 1
    stuck = [e.payload["stuck"]["verdict"] for e in _of(r.events, "action") if "stuck" in e.payload]
    assert stuck == ["warn", "stuck"]
    # the warning reached Lightning as part of the 3rd tool result
    tool_msgs = [m for m in r.fake.requests[3]["messages"] if m["role"] == "tool"]
    assert STUCK_WARNING.split(":")[0] in json.loads(tool_msgs[-1]["content"])["warning"]
    assert r.result.success_claimed is False and r.result.steps == 6


@pytest.mark.asyncio
async def test_progress_between_misses_keeps_the_run_going(tmp_path):
    script = [click("nowhere in particular"), click("nowhere in particular"), click("the gear icon"),
              click("nowhere in particular"), click("nowhere in particular"), done("Opened settings.")]
    r = await _run(tmp_path, script)
    d = _of(r.events, "done")[-1].payload
    assert d["reason"] == "done" and d["stats"]["n_stuck_warnings"] == 0 and d["stats"]["n_stuck_stops"] == 0
