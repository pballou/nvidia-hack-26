"""scripts/agent_metrics.py on small synthetic traces and rollouts (no browser, no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "agent_metrics.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("agent_metrics", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up in sys.modules
    spec.loader.exec_module(mod)
    return mod


am = _load_script()

T0 = 1792263000.0
SCREEN = "Screen:\nPAGE: Settings | headings: Settings\nELEMENTS:\n- toggle: High contrast @ top"
CHANGED_OBS = f"Step 2 | URL: /settings | Title: Settings\nLast action: clicked\nThe screen changed.\n{SCREEN}"
NO_CHANGE_OBS = f"Step 3 | URL: /billing | Title: Billing\nLast action: clicked\n{am.NO_CHANGE_NOTE}\n{SCREEN}"
BAD_ARGS_ERROR = ("click got unknown argument(s) target and missing required argument(s) target_description. "
                  "Expected click(target_description: string). Call click again with exactly these argument names.")


# ---------------------------------------------------------------- interactive fixtures

def ev(run_id: str, step: int, kind: str, payload: dict, timings: dict | None = None, ts: float | None = None) -> dict:
    d = {"run_id": run_id, "step": step, "kind": kind, "ts": ts if ts is not None else T0 + step, "payload": payload}
    if timings:
        d["timings_ms"] = timings
    return d


def start(run_id: str, request: str, thinking: str, oracle: bool = False, ts: float = T0) -> dict:
    config = {"thinking": thinking, "grounder_model": "base", "summary_model": "base", "max_steps": 12,
              "lightning": {"base_url": "http://127.0.0.1:8000/v1", "model": "lightning"},
              "use_oracle_grounder": oracle}
    return ev(run_id, 0, "start", {"request": request, "start_url": "http://x/?sid=1", "config": config}, ts=ts)


def plan(run_id: str, step: int, name: str | None, args: dict | None = None, ms: float = 800, thinking: bool = False):
    calls = [{"name": name, "arguments": args or {}}] if name else []
    return ev(run_id, step, "plan", {"content": "", "reasoning": None, "thinking": thinking, "tool_calls": calls},
              {"plan": ms})


def tool_call(run_id: str, step: int, name: str, args: dict, valid: bool = True) -> dict:
    p = {"name": name, "arguments": args, "valid": valid, "description": name}
    if not valid:
        p["error"] = BAD_ARGS_ERROR
    return ev(run_id, step, "tool_call", p)


def action(run_id: str, step: int, name: str, ok: bool = True, changed: bool | None = True, done: bool = False):
    return ev(run_id, step, "action", {"name": name, "arguments": {}, "ok": ok, "screen_changed": changed,
                                       "done": done}, None if done else {"act": 10, "settle": 350})


def done_ev(run_id: str, steps: int, claimed: bool, reason: str, stats: dict, latency: dict) -> dict:
    full = {k: 0 for k in ("n_tool_calls", "n_invalid_calls", "n_actions", "n_clicks", "n_failed_actions",
                           "n_no_tool_call", "n_no_change", "n_recoveries", "n_thinking_calls", "n_api_errors")}
    full.update(stats, latency_ms=latency)
    return ev(run_id, steps, "done", {"success_claimed": claimed, "summary": "s", "steps": steps, "reason": reason,
                                      "stats": full}, ts=T0 + 30)


def run_a() -> list[dict]:
    """thinking=retry; invalid call, corrected, click changes the screen (a recovery), done(success)."""
    r = "20261017-150000-aaaaaa"
    return [
        start(r, "Make the text bigger, I can't read this.", "retry"),
        ev(r, 0, "observation", {"observation": {}, "text": ""}, {"screenshot": 20, "summary": 400, "settle": 300}),
        plan(r, 1, "click", {"target": "the L button"}, ms=800),
        tool_call(r, 1, "click", {"target": "the L button"}, valid=False),
        plan(r, 2, "click", {"target_description": "the L button"}, ms=1200, thinking=True),
        tool_call(r, 2, "click", {"target_description": "the L button"}),
        ev(r, 2, "grounding", {"target": "the L button", "x": 1, "y": 2, "model": "base"}, {"ground": 150}),
        action(r, 2, "click"),
        ev(r, 2, "observation", {"observation": {}, "text": ""}, {"screenshot": 25, "summary": 380}),
        plan(r, 3, "done", {"summary": "ok", "success": True}, ms=700),
        tool_call(r, 3, "done", {"summary": "ok", "success": True}),
        action(r, 3, "done", changed=None, done=True),
        done_ev(r, 3, True, "done",
                {"n_tool_calls": 3, "n_invalid_calls": 1, "n_actions": 2, "n_clicks": 1, "n_recoveries": 1,
                 "n_thinking_calls": 1},
                {"screenshot": [20, 25], "summary": [400, 380], "settle": [300, 350], "plan": [800, 1200, 700],
                 "ground": [150], "act": [10]}),
    ]


def run_b() -> list[dict]:
    """thinking=retry; a failed click, then a click that changes nothing; max_steps."""
    r = "20261017-150100-bbbbbb"
    return [
        start(r, "Download my September statement.", "retry"),
        plan(r, 1, "click", {"target_description": "x"}, ms=900),
        tool_call(r, 1, "click", {"target_description": "x"}),
        action(r, 1, "click", ok=False, changed=None),
        plan(r, 2, "click", {"target_description": "y"}, ms=1000, thinking=True),
        tool_call(r, 2, "click", {"target_description": "y"}),
        action(r, 2, "click", changed=False),
        done_ev(r, 2, False, "max_steps",
                {"n_tool_calls": 2, "n_actions": 2, "n_clicks": 2, "n_failed_actions": 1, "n_no_change": 1,
                 "n_thinking_calls": 1},
                {"plan": [900, 1000], "act": [10], "settle": [350]}),
    ]


def run_simple(run_id: str, request: str, thinking: str, *, oracle: bool = False, ts: float = T0) -> list[dict]:
    """click (screen changes) then done(success=true)."""
    return [
        start(run_id, request, thinking, oracle=oracle, ts=ts),
        plan(run_id, 1, "click", {"target_description": "x"}, ms=500),
        tool_call(run_id, 1, "click", {"target_description": "x"}),
        ev(run_id, 1, "grounding", {"target": "x", "x": 1, "y": 2, "model": "oracle" if oracle else "base"},
           {"ground": 100}),
        action(run_id, 1, "click"),
        plan(run_id, 2, "done", {"summary": "ok", "success": True}, ms=400),
        tool_call(run_id, 2, "done", {"summary": "ok", "success": True}),
        action(run_id, 2, "done", changed=None, done=True),
        done_ev(run_id, 2, True, "done", {"n_tool_calls": 2, "n_actions": 2, "n_clicks": 1},
                {"plan": [500, 400], "ground": [100], "act": [10], "settle": [350]}),
    ]


def write_run(runs: Path, events: list[dict], *extra_lines: str) -> None:
    d = runs / events[0]["run_id"]
    d.mkdir(parents=True)
    lines = [json.dumps(e) for e in events] + list(extra_lines)
    (d / "events.jsonl").write_text("\n".join(lines) + "\n")


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    write_run(runs, run_a())
    write_run(runs, run_b())
    # thinking=off: incomplete run (crashed mid-step; partial last line)
    c = "20261017-150200-cccccc"
    write_run(runs, [start(c, "Mark all my messages as read.", "off"), plan(c, 1, "click", {"target_description": "x"}),
                     tool_call(c, 1, "click", {"target_description": "x"}), action(c, 1, "click")],
              '{"run_id": "20261017-150200-cccccc", "step": 2, "ki')
    write_run(runs, run_simple("20261017-150300-dddddd", "I keep forgetting to pay. Turn on autopay.", "off"))
    write_run(runs, run_simple("20261017-150400-ffffff", "Download my September statement.", "off"))
    # no summary row: task id and tags come from the tasks file
    write_run(runs, run_simple("20261017-150500-gggggg", "Make the text bigger, I can't read this.", "off"))
    # test run (oracle grounder): excluded by default
    write_run(runs, run_simple("20261017-150600-eeeeee", "Turn on high contrast.", "off", oracle=True))
    summary = {"created": "20261017-150000", "fake_sandbox": False, "tasks": [
        {"task_id": "text-size-large", "tags": ["easy", "settings"], "verified_success": True,
         "success_claimed": True, "run_id": "20261017-150000-aaaaaa", "explanation": "ok"},
        {"task_id": "download-september-statement", "tags": ["medium", "billing"], "verified_success": False,
         "success_claimed": False, "run_id": "20261017-150100-bbbbbb", "explanation": "billing.downloaded != ..."},
        {"task_id": "turn-on-autopay", "tags": ["easy", "billing"], "verified_success": True,
         "success_claimed": True, "run_id": "20261017-150300-dddddd", "explanation": "ok"},
        {"task_id": "download-september-statement", "tags": ["medium", "billing"], "verified_success": False,
         "success_claimed": True, "run_id": "20261017-150400-ffffff",
         "explanation": "harness error: ConnectError: sandbox down"},
        {"task_id": "x", "tags": [], "verified_success": False, "run_id": None, "explanation": "harness error: ..."},
    ]}
    (runs / "summary_20261017-150000.json").write_text(json.dumps(summary))
    (runs / "summary_broken.json").write_text("{not json")
    return runs


@pytest.fixture
def tasks_file(tmp_path: Path) -> Path:
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps([{"id": "text-size-large", "request": "Make the text bigger, I can't read this.",
                              "tags": ["easy", "settings"]}]))
    return p


def run_main(*args: str) -> int:
    return am.main([str(a) for a in args])


def test_interactive_metrics(runs_dir: Path, tasks_file: Path, tmp_path: Path, capsys):
    out_json, out_md = tmp_path / "out" / "m.json", tmp_path / "out" / "m.md"
    assert run_main("--runs", runs_dir, "--tasks", tasks_file, "--json", out_json, "--markdown", out_md) == 0
    report = json.loads(out_json.read_text())
    assert list(report["groups"]) == ["interactive thinking=off", "interactive thinking=retry"]
    assert report["skipped"]["test_runs"] == 1 and report["skipped"]["corrupt_lines"] == 1
    assert any("summary_broken.json" in u for u in report["skipped"]["unreadable"])

    retry = report["groups"]["interactive thinking=retry"]
    assert (retry["n_scored"], retry["n_verified"], retry["n_success"]) == (2, 2, 1)
    assert retry["success_rate"] == 0.5 and retry["claimed_rate"] == 0.5
    assert retry["claim_agreement"] == 1.0 and retry["false_claims"] == 0
    assert (retry["tool_calls"], retry["invalid_calls"], retry["validity_rate"]) == (5, 1, 0.8)
    assert (retry["failed_actions"], retry["no_change"], retry["problems"], retry["recovered"]) == (1, 1, 3, 1)
    assert (retry["runs_with_problem"], retry["runs_succeeded_after_problem"]) == (2, 1)
    assert retry["steps"] == {"n": 2, "mean": 2.5, "median": 2.5}
    assert (retry["thinking_calls"], retry["lightning_calls"]) == (2, 5)
    assert retry["latency_ms"]["plan"] == {"n": 5, "p50": 900.0, "p90": 1120.0}
    assert retry["latency_ms"]["summary"]["n"] == 2 and retry["wall_s"] == {"n": 2, "p50": 30.0, "p90": 30.0}
    assert retry["by_tag"]["easy"]["n_success"] == 1 and retry["by_tag"]["medium"]["success_rate"] == 0.0
    assert retry["end_reasons"] == {"done": 1, "max_steps": 1}
    assert retry["lightning_models"] == ["lightning"]

    off = report["groups"]["interactive thinking=off"]
    assert off["n_runs"] == 4 and off["excluded"] == {"incomplete": 1}
    # D verified; F had a harness error (claimed, not verified); G has no summary row
    assert (off["n_scored"], off["n_verified"], off["n_success"], off["n_claimed"]) == (3, 1, 1, 3)
    assert off["by_tag"]["settings"] == {"n": 1, "n_verified": 0, "n_success": 0, "success_rate": None,
                                         "n_claimed": 1, "steps_mean": 2.0}
    runs = {r["run_id"]: r for r in report["runs"]}
    assert runs["20261017-150500-gggggg"]["task_id"] == "text-size-large"
    assert runs["20261017-150400-ffffff"]["verified"] is None
    c = runs["20261017-150200-cccccc"]
    assert c["excluded"] == "incomplete" and c["n_tool_calls"] == 1 and c["steps"] == 1

    md = out_md.read_text()
    assert md == capsys.readouterr().out
    assert md.startswith("| Group | Runs | Verified success |")
    assert "| interactive thinking=retry | 2 | 1/2 (50%) | 1/2 (50%) | 2/2 (100%) | 2.5 (2.5) | 4/5 (80%) " \
           "| 3 (1 / 1 / 1) | 1 | 1/2 | 2/5 |" in md
    assert "| interactive thinking=off | 3 (+1 incomplete) |" in md
    assert "| interactive thinking=retry | 900 / 1120 |" in md
    assert "| easy | 1/1 | 1/1 |" in md and "Skipped: 1 test runs" in md


def test_event_recompute_matches_done_stats():
    events = run_a()
    stats, latency = am._stats_from_events(events[:-1])
    done = events[-1]["payload"]["stats"]
    for key in ("n_tool_calls", "n_invalid_calls", "n_failed_actions", "n_no_change", "n_no_tool_call",
                "n_recoveries", "n_thinking_calls"):
        assert stats[key] == done[key], key
    assert {k: sorted(v) for k, v in latency.items()} == {k: sorted(v) for k, v in done["latency_ms"].items()}
    assert am._stats_from_events(run_b()[:-1])[0]["n_failed_actions"] == 1


def test_lightning_outage_run_is_excluded_not_failed():
    """A run that ended because Lightning was unreachable (done reason 'error') is infra, like a Gym mask."""
    r = "20261017-150200-cccccc"
    events = [start(r, "Turn on autopay.", "retry"),
              ev(r, 1, "error", {"message": "APIConnectionError: Connection error.", "where": "planner",
                                 "recoverable": False}),
              done_ev(r, 1, False, "error", {"n_api_errors": 2}, {})]
    ep = am.interactive_episode(r, events, None, {}, ["thinking"])
    assert ep.excluded == "infra_error" and ep.end_reason == "error"
    assert am.interactive_episode(r, run_b(), None, {}, ["thinking"]).excluded is None  # max_steps still counts


def test_test_runs_group_by_and_summary_selection(runs_dir: Path, tasks_file: Path, tmp_path: Path, capsys):
    out = tmp_path / "m.json"
    run_main("--runs", runs_dir, "--tasks", tasks_file, "--include-test-runs", "--json", out,
             "--group-by", "thinking,use_oracle_grounder")
    groups = json.loads(out.read_text())["groups"]
    assert groups["interactive thinking=off, use_oracle_grounder=True"]["n_runs"] == 1
    assert groups["interactive thinking=off, use_oracle_grounder=False"]["n_runs"] == 4

    # explicit summary: only that batch's runs (the row without a run_id is skipped)
    run_main("--summary", runs_dir / "summary_20261017-150000.json", "--runs", runs_dir, "--json", out)
    report = json.loads(out.read_text())
    assert {r["run_id"] for r in report["runs"]} == {"20261017-150000-aaaaaa", "20261017-150100-bbbbbb",
                                                     "20261017-150300-dddddd", "20261017-150400-ffffff"}
    capsys.readouterr()


def test_since_uses_start_time(tmp_path: Path, capsys):
    runs = tmp_path / "runs"
    early, late = am.parse_since("20261017-1400"), am.parse_since("20261017-1600")
    write_run(runs, run_simple("20261017-140000-aaaaaa", "a", "off", ts=early))
    write_run(runs, run_simple("20261017-160000-bbbbbb", "b", "off", ts=late))
    out = tmp_path / "m.json"
    run_main("--runs", runs, "--since", "2026-10-17T15:00", "--json", out)
    report = json.loads(out.read_text())
    assert [r["run_id"] for r in report["runs"]] == ["20261017-160000-bbbbbb"]
    assert report["skipped"]["before_since"] == 1
    capsys.readouterr()


# ---------------------------------------------------------------- Gym rollouts

def fc(call_id: str, name: str, args) -> dict:
    return {"type": "function_call", "call_id": call_id, "name": name,
            "arguments": args if isinstance(args, str) else json.dumps(args), "status": "completed"}


def fco(call_id: str, payload) -> dict:
    return {"type": "function_call_output", "call_id": call_id,
            "output": payload if isinstance(payload, str) else json.dumps(payload)}


def msg(text: str = "Done.") -> dict:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def rollout(task_id: str, tags: list[str], output: list[dict], reward: float, claimed: bool, **extra) -> dict:
    row = {"task_id": task_id, "tags": tags, "reward": reward, "claimed_success": claimed, "mask_sample": False,
           "failure_reason": None, "response": {"model": "lightning", "output": output}, "_ng_task_index": 0,
           "_ng_rollout_index": 0}
    row.update(extra)
    return row


DONE_OBS = {"ok": True, "observation": "Step 3 | URL: /settings | Title: Settings\nLast action: done recorded"}


def gym_rows() -> list[dict]:
    success = rollout("text-size-large", ["easy", "settings"], [
        fc("a1", "click", {"target": "the L button"}),
        fco("a1", {"ok": False, "error": BAD_ARGS_ERROR, "observation": "..."}),
        fc("a2", "click", {"target_description": "the L button"}),
        fco("a2", {"ok": True, "observation": CHANGED_OBS, "click": {"x": 1, "y": 2}}),
        fc("a3", "done", {"summary": "Made the text larger.", "success": True}), fco("a3", DONE_OBS), msg(),
    ], 1.0, True, steps=2, n_invalid_calls=1)
    masked = rollout("turn-on-autopay", ["easy", "billing"], [
        fc("b1", "click", {"target_description": "x"}),
        fco("b1", {"ok": False, "error": "environment failed to start: ConnectError", "observation": ""}), msg(),
    ], 0.0, False, mask_sample=True, failure_reason="sandbox unreachable", failure_kind="provider_unavailable",
        _ng_task_index=3)
    failed = rollout("download-september-statement", ["easy", "billing", "icon_only"], [
        fc("c1", "click", {"target_description": "the download icon"}),
        fco("c1", {"ok": False, "error": "could not locate 'the download icon': no point", "observation": "..."}),
        fc("c2", "click", {"target_description": "the September row"}),
        fco("c2", {"ok": True, "observation": NO_CHANGE_OBS}),
        fc("c3", "click", '{"target_description": "the'),
        fco("c3", '{"error": "Invalid tool call arguments: JSONDecodeError(...)"}'),
        fc("c4", "press_key", {"key": "Enter"}),
        fco("c4", {"ok": True, "observation": CHANGED_OBS}),
    ], 0.0, False, _ng_task_index=1)
    false_claim = rollout("high-contrast", ["medium", "settings"], [
        fc("d1", "navigate", {"url": "/settings"}),
        fco("d1", {"ok": False, "error": "unknown tool 'navigate'; available: click, type_text, press_key, "
                                         "scroll, done", "observation": "..."}),
        fc("d2", "done", {"summary": "Turned it on.", "success": "true"}), fco("d2", DONE_OBS),
        fc("d3", "click", {"target_description": "x"}),
        fco("d3", {"ok": False, "error": am.ALREADY_DONE, "observation": "..."}), msg(),
    ], 0.0, True, _ng_task_index=2)
    return [success, masked, failed, false_claim]


def write_jsonl(path: Path, rows: list, *extra_lines: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([json.dumps(r) for r in rows] + list(extra_lines)) + "\n")


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    res = tmp_path / "results"
    write_jsonl(res / "agent_base.jsonl", gym_rows(), '{"reward": 1.0, "respo')
    write_jsonl(res / "agent_base_failures.jsonl", [{"_ng_failure_class": "agent_run_error", "task_id": "x",
                                                     "_ng_failure_message": "verify returned 500"}])
    write_jsonl(res / "agent_base_materialized_inputs.jsonl", [{"task_id": "should-be-ignored"}])
    write_jsonl(res / "agent_grounder.jsonl", gym_rows()[:1])
    return res


def test_gym_rollout_metrics(results_dir: Path, tmp_path: Path, capsys):
    out = tmp_path / "m.json"
    files = sorted(results_dir.glob("*.jsonl"))  # like results/*.jsonl: sidecars and inputs included
    assert run_main("--no-runs", "--rollouts", *files, "--label", "base", "--label", "grounder", "--json", out) == 0
    report = json.loads(out.read_text())
    assert list(report["groups"]) == ["base", "grounder"]
    assert [s["dropped"] for s in report["sources"]["rollouts"]] == [1, 0]
    assert report["skipped"]["corrupt_lines"] == 1

    g = report["groups"]["base"]
    assert (g["n_runs"], g["n_scored"], g["excluded"]) == (5, 3, {"dropped": 1, "masked": 1})
    assert g["failure_kinds"] == {"provider_unavailable": 1, "agent_run_error": 1}
    assert (g["n_verified"], g["n_success"], g["n_claimed"]) == (3, 1, 2)
    assert (g["claim_agreement"], g["false_claims"], g["missed_claims"]) == (round(2 / 3, 4), 1, 0)
    assert (g["tool_calls"], g["invalid_calls"], g["validity_rate"]) == (10, 3, 0.7)
    assert (g["failed_actions"], g["no_change"], g["recovered"], g["after_done_calls"]) == (1, 1, 2, 1)
    assert (g["runs_with_problem"], g["runs_succeeded_after_problem"]) == (3, 1)
    assert g["steps"] == {"n": 3, "mean": 3.33, "median": 3}
    assert g["end_reasons"] == {"done": 2, "max_steps": 1}
    assert g["by_tag"]["easy"]["n_verified"] == 2 and g["by_tag"]["easy"]["n_success"] == 1
    assert g["latency_ms"] == {} and g["lightning_models"] == ["lightning"]
    assert report["groups"]["grounder"]["success_rate"] == 1.0

    runs = {r["run_id"]: r for r in report["runs"]}
    assert len(runs) == len(report["runs"]) == 6
    assert set(runs) == {"agent_base#0.0", "agent_base#3.0", "agent_base#1.0", "agent_base#2.0",
                         "agent_base#failed.0", "agent_grounder#0.0"}
    first = runs["agent_base#0.0"]
    assert (first["n_lightning_calls"], first["steps"], first["n_recoveries"]) == (4, 3, 1)
    assert runs["agent_base#2.0"]["claimed"] is True and runs["agent_base#2.0"]["n_after_done"] == 1
    assert runs["agent_base#1.0"]["end_reason"] == "max_steps"
    md = capsys.readouterr().out
    assert "| base | 3 (+1 dropped, 1 masked) | 1/3 (33%) | 2/3 (67%) | 2/3 (67%) | 3.3 (3) | 7/10 (70%) " \
           "| 5 (3 / 1 / 1) | 2 | 1/3 | 0/12 |" in md  # 4 model calls in each scored rollout
    assert "Latency per stage" not in md  # Gym rows carry no timings


def test_gym_default_label_and_label_errors(results_dir: Path, tmp_path: Path, capsys):
    out = tmp_path / "m.json"
    run_main("--no-runs", "--rollouts", results_dir / "agent_base.jsonl", "--json", out)
    report = json.loads(out.read_text())
    assert list(report["groups"]) == ["gym agent_base"]
    assert report["sources"]["rollouts"][0]["failures"].endswith("agent_base_failures.jsonl")
    with pytest.raises(SystemExit):
        run_main("--no-runs", "--rollouts", results_dir / "agent_base.jsonl", results_dir / "agent_grounder.jsonl",
                 "--label", "a", "--label", "b", "--label", "c")
    capsys.readouterr()


def test_gym_rows_without_output_use_verify_counters():
    row = {"task_id": "t", "reward": 1.0, "claimed_success": True, "done_called": True, "steps": 4,
           "n_invalid_calls": 1, "n_failed_actions": 2, "response": {"output": []}}
    ep = am.gym_episode(row, "x", 0)
    assert (ep.steps, ep.n_tool_calls, ep.n_invalid_calls, ep.n_failed_actions) == (4, 4, 1, 2)
    assert ep.verified is True and ep.claimed is True and ep.end_reason == "done"
    assert am.gym_episode({"task_id": "t"}, "x", 1).excluded == "no_reward"


@pytest.mark.parametrize("name,args,output,expected", [
    ("click", {"target_description": "x"}, json.dumps({"ok": True, "observation": CHANGED_OBS}), am.CHANGED),
    ("click", {"target_description": "x"}, json.dumps({"ok": True, "observation": NO_CHANGE_OBS}), am.NO_CHANGE),
    ("done", {"summary": "s", "success": True}, json.dumps(DONE_OBS), am.OK),
    ("click", {"target": "x"}, json.dumps({"ok": False, "error": BAD_ARGS_ERROR}), am.INVALID),
    ("click", "{not json", '{"error": "Invalid tool call arguments: JSONDecodeError()"}', am.INVALID),
    ("scroll", {"direction": "sideways", "amount": "page"}, json.dumps({"ok": False, "error": "x"}), am.INVALID),
    ("click", {"target_description": "x"}, json.dumps({"detail": "Not Found"}), am.INVALID),
    ("click", {"target_description": "x"}, json.dumps({"ok": False, "error": "click failed: TimeoutError"}),
     am.FAILED),
    ("click", {"target_description": "x"}, '"RuntimeError(\'boom\')"', am.FAILED),
    ("click", {"target_description": "x"}, "Internal Server Error", am.FAILED),
    ("click", {"target_description": "x"}, json.dumps({"ok": False, "error": am.ALREADY_DONE}), am.AFTER_DONE),
    ("click", {"target_description": "x"}, None, am.NOT_EXECUTED),
    ("click", {"target_description": "x"}, [{"type": "input_text", "text": json.dumps({"ok": True,
                                                                                         "observation": ""})}], am.OK),
])
def test_classify_gym_call(name, args, output, expected):
    arguments = args if isinstance(args, str) else json.dumps(args)
    assert am.classify_gym_call(name, arguments, output)[0] == expected


def test_percentile_and_since_parsing():
    assert am.percentile([], 0.5) is None
    assert am.percentile([4, 1, 3, 2], 0.5) == 2.5
    assert am.percentile([1, 2, 3, 4], 0.9) == pytest.approx(3.7)
    assert am.percentile([7], 0.9) == 7
    assert am.parse_since("20261017-1800") == am.parse_since("2026-10-17T18:00")
    with pytest.raises(am.argparse.ArgumentTypeError):
        am.parse_since("yesterday")


def test_empty_inputs_do_not_crash(tmp_path: Path, capsys):
    assert run_main("--runs", tmp_path / "missing", "--json", tmp_path / "m.json") == 0
    report = json.loads((tmp_path / "m.json").read_text())
    assert report["groups"] == {} and report["runs"] == []
    assert "no runs found" in capsys.readouterr().err
