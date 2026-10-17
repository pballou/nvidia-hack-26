"""Agent metrics for the Lightning bounty: interactive runs (runs/) and Gym rollouts (results/).

Sources
  interactive  runs/<run_id>/events.jsonl (agent/trace.py), joined by run_id with
               runs/summary_*.json (scripts/run_agent_tasks.py) for verified success. Runs
               without a summary row have a claimed result only (verified = n/a); their task
               id and tags come from matching the request against sandbox/tasks.json.
  gym          results/*.jsonl from `gym eval run`, one web_agent VerifyResponse per line
               (reward, mask_sample, failure_reason, task_id, tags, steps, claimed_success, ...,
               and response.output). `<stem>_failures.jsonl` sidecars count as dropped
               rollouts; `*_materialized_inputs.jsonl` files are skipped.

Metrics, per group (interactive runs by start-event config.thinking, or --group-by; Gym
files by --label, default the file stem)
  success    verified success rate (sandbox check / Gym reward). Claimed = done(success=true);
             "claim = verified" is their agreement on runs that have both.
  steps      Lightning calls per run (interactive: done.steps; Gym: model turns that called a
             tool, from response.output).
  validity   1 - invalid / tool calls. Invalid = malformed JSON, unknown tool, wrong, missing
             or extra argument names, bad enum or type (agent/session.py validate_args). Gym
             calls are re-validated from response.output, so calls simple_agent never sent to
             the server (bad JSON) count too.
  problems   corrections (invalid calls + replies without a tool call) + failed actions
             (ok=false) + actions after which the screen did not change. "Recovered" = problems
             later followed by an ok, screen-changing action (agent/core.py n_recoveries);
             "succeeded after a problem" = verified successes among runs with >= 1 problem.
  latency    p50/p90 per stage (plan, summary, ground, act, settle, screenshot) from the
             events' timings_ms; wall = start -> done per run, in seconds.
  coverage   masked (Gym mask_sample) and dropped (sidecar) rollouts, and interactive runs
             that are incomplete (no done event) or were stopped by the user, are excluded from
             every quality metric and counted separately.

Test runs (oracle grounder or fake sandbox) are excluded unless --include-test-runs. Report
only runs made with the official BF16 Lightning: select them with --since or --summary.

Examples
  python scripts/agent_metrics.py
  python scripts/agent_metrics.py --summary runs/summary_20261017-190000.json --markdown /tmp/lightning.md
  python scripts/agent_metrics.py --no-runs --rollouts results/agent_base.jsonl results/agent_grounder.jsonl
  python scripts/agent_metrics.py --since 20261017-1800 --rollouts results/agent_base.jsonl \\
      --label "gym base, thinking off" --json /tmp/lightning.json
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Run as a file (python scripts/agent_metrics.py), so put the repo root on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.observation import NO_CHANGE_NOTE  # noqa: E402
from agent.session import ALREADY_DONE, validate_args  # noqa: E402

STAGES = ("plan", "summary", "ground", "act", "settle", "screenshot")
THINKING_ORDER = {"off": 0, "retry": 1, "on": 2}
NO_TOOL_CALL_PREFIX = "reply had no tool call"   # agent/core.py error event message
SCREEN_CHANGED_LINE = "The screen changed."        # agent/observation.py Observation.render
TAG_ORDER = {"easy": 0, "medium": 1, "hard": 2}

# Gym tool-call classes (one per function_call in response.output)
INVALID, FAILED, AFTER_DONE, NO_CHANGE, CHANGED, OK, NOT_EXECUTED = (
    "invalid", "failed", "after_done", "no_change", "changed", "ok", "not_executed")


@dataclass
class Episode:
    """One interactive run or one Gym rollout, normalized."""
    source: str                        # interactive | gym
    group: str
    run_id: str
    task_id: str | None = None
    tags: list[str] = field(default_factory=list)
    verified: bool | None = None       # None: no verifier result (no summary row, harness error, masked)
    claimed: bool | None = None
    excluded: str | None = None        # masked | dropped | incomplete | stopped | no_reward -> not scored
    end_reason: str | None = None      # done | max_steps | error | no_tool_call | ...
    steps: int | None = None
    n_lightning_calls: int = 0
    n_tool_calls: int = 0
    n_invalid_calls: int = 0
    n_no_tool_call: int = 0
    n_failed_actions: int = 0
    n_no_change: int = 0
    n_recoveries: int = 0
    n_after_done: int = 0
    n_thinking_calls: int = 0
    latency_ms: dict[str, list[float]] = field(default_factory=dict)
    wall_s: float | None = None
    lightning: str | None = None
    failure_kind: str | None = None
    failure_reason: str | None = None

    @property
    def scored(self) -> bool:
        return self.excluded is None

    @property
    def n_problems(self) -> int:
        return self.n_invalid_calls + self.n_no_tool_call + self.n_failed_actions + self.n_no_change


@dataclass
class Skipped:
    corrupt_lines: int = 0
    unreadable: list[str] = field(default_factory=list)
    test_runs: int = 0
    before_since: int = 0


# ---------------------------------------------------------------- small helpers

def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def _int(v: Any) -> int:
    n = _num(v)
    return int(n) if n is not None else 0


def _latency(d: Any) -> dict[str, list[float]]:
    """{stage: [ms, ...]} with non-numbers dropped."""
    if not isinstance(d, dict):
        return {}
    out = {str(k): [x for x in (_num(v) for v in vs) if x is not None] for k, vs in d.items() if isinstance(vs, list)}
    return {k: v for k, v in out.items() if v}


def read_jsonl(path: Path, skipped: Skipped) -> list[dict]:
    """JSON objects from a .jsonl file; corrupt or partial lines are counted and skipped."""
    out: list[dict] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    skipped.corrupt_lines += 1
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
                else:
                    skipped.corrupt_lines += 1
    except OSError as e:
        skipped.unreadable.append(f"{path}: {type(e).__name__}")
    return out


def percentile(xs: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in [0, 1]); None for no data."""
    s = sorted(xs)
    if not s:
        return None
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _rate(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def _dig(d: dict, dotted: str) -> Any:
    for key in dotted.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def parse_since(value: str) -> float:
    """Local time 'YYYYMMDD-HHMM[SS]' / 'YYYY-MM-DDTHH:MM[:SS]' / 'YYYY-MM-DD', or epoch seconds."""
    for fmt in ("%Y%m%d-%H%M%S", "%Y%m%d-%H%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M",
                "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            continue
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad --since {value!r}; use e.g. 20261017-1800") from None


def _run_started(run_id: str, events: list[dict]) -> float | None:
    for ev in events:
        if ev.get("kind") == "start" and _num(ev.get("ts")) is not None:
            return float(ev["ts"])
    try:
        return datetime.strptime(run_id[:15], "%Y%m%d-%H%M%S").timestamp()
    except ValueError:
        return None


# ---------------------------------------------------------------- interactive runs

def load_tasks(path: Path | None) -> dict[str, dict]:
    """request text -> task (sandbox/tasks.json schema)."""
    if path is None or not path.is_file():
        return {}
    try:
        tasks = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {t["request"].strip(): t for t in tasks if isinstance(t, dict) and isinstance(t.get("request"), str)}


def load_summaries(paths: list[Path], skipped: Skipped) -> dict[str, dict]:
    """run_id -> summary row from scripts/run_agent_tasks.py, with the batch's fake_sandbox flag."""
    rows: dict[str, dict] = {}
    for p in paths:
        try:
            data = json.loads(p.read_text())
            tasks = data.get("tasks") or []
        except (OSError, ValueError, AttributeError) as e:
            skipped.unreadable.append(f"{p}: {type(e).__name__}")
            continue
        for row in tasks:
            if isinstance(row, dict) and row.get("run_id"):
                rows[str(row["run_id"])] = dict(row, _fake_sandbox=bool(data.get("fake_sandbox")))
    return rows


def _stats_from_events(events: list[dict]) -> tuple[Counter, dict[str, list[float]]]:
    """Recompute the done-event stats (agent/core.py) for a run that has no done event."""
    s: Counter = Counter()
    lat: dict[str, list[float]] = {}
    pending = False
    for ev in events:
        kind, p = ev.get("kind"), ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        timings = ev.get("timings_ms") if isinstance(ev.get("timings_ms"), dict) else {}
        for stage, ms in timings.items():
            if _num(ms) is not None:
                lat.setdefault(stage, []).append(float(ms))
        if kind == "plan":
            s["n_lightning_calls"] += 1
            s["n_thinking_calls"] += bool(p.get("thinking"))
        elif kind == "tool_call":
            s["n_tool_calls"] += 1
            if p.get("valid") is False:
                s["n_invalid_calls"] += 1
                pending = True
        elif kind == "error" and str(p.get("message", "")).startswith(NO_TOOL_CALL_PREFIX):
            s["n_no_tool_call"] += 1
            pending = True
        elif kind == "action" and not p.get("done"):
            if not p.get("ok"):
                s["n_failed_actions"] += 1
                pending = True
            elif p.get("screen_changed") is False:
                s["n_no_change"] += 1
                pending = True
            elif p.get("screen_changed") and pending:
                s["n_recoveries"] += 1
                pending = False
    return s, lat


def interactive_episode(run_id: str, events: list[dict], summary: dict | None, tasks_by_request: dict[str, dict],
                        group_by: list[str]) -> Episode:
    start = next((e for e in events if e.get("kind") == "start"), None)
    done = next((e for e in reversed(events) if e.get("kind") == "done"), None)
    sp = start.get("payload") if start and isinstance(start.get("payload"), dict) else {}
    config = sp.get("config") if isinstance(sp.get("config"), dict) else {}
    group = "interactive " + ", ".join(f"{k}={_dig(config, k) if _dig(config, k) is not None else '?'}"
                                       for k in group_by)
    ep = Episode(source="interactive", group=group, run_id=run_id, lightning=_dig(config, "lightning.model"))

    if summary:
        ep.task_id, ep.tags = summary.get("task_id"), list(summary.get("tags") or [])
        if not str(summary.get("explanation", "")).startswith("harness error"):  # harness error: not verified
            ep.verified = bool(summary.get("verified_success"))
    elif isinstance(sp.get("request"), str):
        task = tasks_by_request.get(sp["request"].strip())
        if task:
            ep.task_id, ep.tags = task.get("id"), list(task.get("tags") or [])

    dp = done.get("payload") if done and isinstance(done.get("payload"), dict) else None
    if dp is not None:
        stats = dp.get("stats") if isinstance(dp.get("stats"), dict) else {}
        ep.end_reason = str(dp.get("reason") or "done")
        ep.claimed = bool(dp.get("success_claimed"))
        ep.steps = _int(dp.get("steps"))
        ep.n_lightning_calls = ep.steps
        for key in ("n_tool_calls", "n_invalid_calls", "n_no_tool_call", "n_failed_actions", "n_no_change",
                    "n_recoveries", "n_thinking_calls"):
            setattr(ep, key, _int(stats.get(key)))
        ep.latency_ms = _latency(stats.get("latency_ms"))
        if not ep.latency_ms:
            ep.latency_ms = _stats_from_events(events)[1]
        if start and _num(start.get("ts")) is not None and _num(done.get("ts")) is not None:
            ep.wall_s = round(float(done["ts"]) - float(start["ts"]), 3)
        if ep.end_reason == "stopped":
            ep.excluded = "stopped"
    else:
        stats, ep.latency_ms = _stats_from_events(events)
        for key, value in stats.items():
            setattr(ep, key, value)
        ep.steps = ep.n_lightning_calls
        ep.end_reason, ep.excluded = "incomplete", "incomplete"
    return ep


def _is_test_run(events: list[dict], summary: dict | None) -> bool:
    for ev in events:
        if ev.get("kind") == "start" and _dig(ev, "payload.config.use_oracle_grounder"):
            return True
        if ev.get("kind") == "grounding" and _dig(ev, "payload.model") == "oracle":
            return True
    return bool(summary and summary.get("_fake_sandbox"))


def collect_interactive(runs_dir: Path, summary_files: list[Path] | None, tasks_by_request: dict[str, dict],
                        group_by: list[str], since: float | None, include_test_runs: bool,
                        skipped: Skipped) -> tuple[list[Episode], list[str]]:
    """Episodes for runs/<run_id>/events.jsonl. With explicit summary files, only their runs."""
    explicit = summary_files is not None
    summary_paths = summary_files if explicit else sorted(runs_dir.glob("summary_*.json"))
    summaries = load_summaries(summary_paths, skipped)
    if explicit:
        dirs = []
        for rid, row in summaries.items():
            d = Path(row["trace_dir"]) if row.get("trace_dir") else runs_dir / rid
            dirs.append(d if (d / "events.jsonl").is_file() else runs_dir / rid)
    elif (runs_dir / "events.jsonl").is_file():   # --runs pointed at one run directory
        dirs = [runs_dir]
    else:
        dirs = sorted(d for d in runs_dir.iterdir() if (d / "events.jsonl").is_file()) if runs_dir.is_dir() else []

    episodes = []
    for d in dirs:
        if not (d / "events.jsonl").is_file():
            skipped.unreadable.append(f"{d / 'events.jsonl'}: missing")
            continue
        events = read_jsonl(d / "events.jsonl", skipped)
        run_id = next((str(e["run_id"]) for e in events if e.get("run_id")), d.name)
        summary = summaries.get(run_id)
        if not include_test_runs and _is_test_run(events, summary):
            skipped.test_runs += 1
            continue
        if since is not None:
            started = _run_started(run_id, events)
            if started is None or started < since:
                skipped.before_since += 1
                continue
        episodes.append(interactive_episode(run_id, events, summary, tasks_by_request, group_by))
    return episodes, [str(p) for p in summary_paths]


# ---------------------------------------------------------------- Gym rollouts

def _is_validation_error(err: str) -> bool:
    """agent/session.py validate_args / invalid_json_error, the server's unknown-tool route, simple_agent's
    bad-JSON output."""
    return (err.startswith(("unknown tool", "invalid JSON arguments", "Invalid tool call arguments"))
            or "again with exactly these argument names" in err)


def _output_text(output: Any) -> str | None:
    if output is None:
        return None
    if isinstance(output, list):  # input_text parts
        return "".join(str(p.get("text", "")) for p in output if isinstance(p, dict))
    return str(output)


def classify_gym_call(name: Any, arguments: Any, output: Any) -> tuple[str, dict | None]:
    """(class, clean args) for one function_call and its function_call_output text (None if never executed)."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except ValueError:
        return INVALID, None
    clean, err = validate_args(str(name), args)
    if err:
        return INVALID, None
    text = _output_text(output)
    if text is None:
        return NOT_EXECUTED, clean
    try:
        payload = json.loads(text)
    except ValueError:
        return FAILED, clean  # not our payload: e.g. a server error body
    if not isinstance(payload, dict):
        return FAILED, clean
    if "ok" not in payload:
        if "detail" in payload:  # FastAPI 404/422
            return INVALID, clean
        return (INVALID if _is_validation_error(str(payload.get("error") or "")) else FAILED), clean
    if payload["ok"] is not True:
        err = str(payload.get("error") or "")
        if _is_validation_error(err):
            return INVALID, clean
        return (AFTER_DONE if err.startswith(ALREADY_DONE) else FAILED), clean
    obs = str(payload.get("observation") or "")
    if NO_CHANGE_NOTE in obs:
        return NO_CHANGE, clean
    if SCREEN_CHANGED_LINE in obs.splitlines():
        return CHANGED, clean
    return OK, clean


def _model_turns(output: list) -> list[list[dict]]:
    """Model items grouped per model call (function_call_output items separate the calls)."""
    turns: list[list[dict]] = []
    cur: list[dict] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call_output":
            if cur:
                turns.append(cur)
                cur = []
        else:
            cur.append(item)
    if cur:
        turns.append(cur)
    return turns


def gym_episode(row: dict, label: str, index: int, file_stem: str | None = None) -> Episode:
    """One rollout line; run_id = <file stem>#<_ng_task_index>.<_ng_rollout_index> (line index if absent)."""
    rid = f"{file_stem or label}#{row.get('_ng_task_index', index)}.{row.get('_ng_rollout_index', 0)}"
    response = row.get("response") if isinstance(row.get("response"), dict) else {}
    output = response.get("output") if isinstance(response.get("output"), list) else []
    tags = row.get("tags") if isinstance(row.get("tags"), list) else []
    ep = Episode(source="gym", group=label, run_id=rid, task_id=str(row["task_id"]) if row.get("task_id") else None,
                 tags=[str(t) for t in tags if isinstance(t, (str, int))],
                 lightning=response.get("model"), failure_kind=row.get("failure_kind"),
                 failure_reason=row.get("failure_reason"), latency_ms=_latency(row.get("latency_ms")))

    claimed_from_output = None
    if output:
        outputs = {o.get("call_id"): o.get("output") for o in output
                   if isinstance(o, dict) and o.get("type") == "function_call_output"}
        pending = False
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            ep.n_tool_calls += 1
            kind, clean = classify_gym_call(item.get("name"), item.get("arguments"), outputs.get(item.get("call_id")))
            if kind == INVALID:
                ep.n_invalid_calls += 1
                pending = True
            elif kind == FAILED:
                ep.n_failed_actions += 1
                pending = True
            elif kind == NO_CHANGE:
                ep.n_no_change += 1
                pending = True
            elif kind == CHANGED and pending:
                ep.n_recoveries += 1
                pending = False
            elif kind == AFTER_DONE:
                ep.n_after_done += 1
            if (item.get("name") == "done" and clean is not None and kind not in (AFTER_DONE, NOT_EXECUTED)
                    and claimed_from_output is None):
                claimed_from_output = bool(clean.get("success"))
        turns = _model_turns(output)
        ep.n_lightning_calls = len(turns)
        ep.n_thinking_calls = sum(any(i.get("type") == "reasoning" for i in t) for t in turns)
        ep.steps = sum(any(i.get("type") == "function_call" for i in t) for t in turns)
    else:  # no output: fall back to the VerifyResponse counters
        ep.steps = _int(row.get("steps"))
        ep.n_tool_calls = ep.steps
        ep.n_invalid_calls = _int(row.get("n_invalid_calls"))
        ep.n_failed_actions = _int(row.get("n_failed_actions"))

    done_called = bool(row.get("done_called")) or claimed_from_output is not None
    if isinstance(row.get("claimed_success"), bool):
        ep.claimed = row["claimed_success"]
    else:
        ep.claimed = bool(claimed_from_output)
    if done_called:
        ep.end_reason = "done"
    elif output and isinstance(output[-1], dict) and output[-1].get("type") == "function_call_output":
        ep.end_reason = "max_steps"
    elif response.get("incomplete_details"):
        ep.end_reason = "incomplete_response"
    else:
        ep.end_reason = "no_tool_call"

    reward = _num(row.get("reward"))
    if row.get("mask_sample"):
        ep.excluded = "masked"
    elif reward is None:
        ep.excluded = "no_reward"
    else:
        ep.verified = reward > 0.5
    return ep


def _failure_episode(row: dict, label: str, index: int, file_stem: str) -> Episode:
    """A `<stem>_failures.jsonl` row: the agent /run failed, so there is no rollout to score."""
    return Episode(source="gym", group=label, run_id=f"{file_stem}#failed.{index}",
                   task_id=str(row["task_id"]) if row.get("task_id") else None, excluded="dropped",
                   end_reason="run_failed", failure_kind=row.get("_ng_failure_class"),
                   failure_reason=str(row.get("_ng_failure_message") or row.get("error") or "")[:300] or None)


def rollout_files(paths: list[Path]) -> list[tuple[Path, Path | None]]:
    """(main jsonl, failures sidecar or None), in the given order; materialized inputs skipped."""
    mains: list[Path] = []
    sidecars: dict[Path, Path] = {}
    for p in paths:
        if p.stem.endswith("_materialized_inputs"):
            continue
        if p.stem.endswith("_failures"):
            sidecars[p.with_name(p.stem[: -len("_failures")] + p.suffix)] = p
        elif p not in mains:
            mains.append(p)
    out = []
    for m in mains:
        side = sidecars.pop(m, None) or m.with_name(m.stem + "_failures" + m.suffix)
        out.append((m, side if side.is_file() else None))
    out.extend((m, s) for m, s in sidecars.items() if m not in mains)  # sidecar given without its main file
    return out


def collect_gym(paths: list[Path], labels: list[str], skipped: Skipped) -> tuple[list[Episode], list[dict]]:
    files = rollout_files(paths)
    if len(labels) not in (0, 1, len(files)):
        raise SystemExit(f"--label: give one label for all rollout files or one per file "
                         f"({len(files)} main files: {', '.join(str(m) for m, _ in files)})")
    episodes, sources = [], []
    for i, (main, side) in enumerate(files):
        label = labels[i] if len(labels) == len(files) and len(labels) > 1 else (
            labels[0] if labels else "gym " + main.stem)
        rows = read_jsonl(main, skipped) if main.is_file() else []
        if not main.is_file() and side is None:
            skipped.unreadable.append(f"{main}: missing")
        eps = [gym_episode(r, label, j, main.stem) for j, r in enumerate(rows)]
        dropped = ([_failure_episode(r, label, j, main.stem) for j, r in enumerate(read_jsonl(side, skipped))]
                   if side else [])
        episodes += eps + dropped
        sources.append({"path": str(main), "failures": str(side) if side else None, "label": label,
                        "rows": len(rows), "dropped": len(dropped)})
    return episodes, sources


# ---------------------------------------------------------------- aggregation

def _dist(xs: list[float]) -> dict:
    return {"n": len(xs), "p50": _round(percentile(xs, 0.5)), "p90": _round(percentile(xs, 0.9))}


def _round(x: float | None, nd: int = 1) -> float | None:
    return round(x, nd) if x is not None else None


def summarize(eps: list[Episode]) -> dict:
    scored = [e for e in eps if e.scored]
    verified = [e for e in scored if e.verified is not None]
    claimed = [e for e in scored if e.claimed is not None]
    both = [e for e in verified if e.claimed is not None]
    steps = [e.steps for e in scored if e.steps is not None]
    problem_runs = [e for e in scored if e.n_problems]
    problem_verified = [e for e in problem_runs if e.verified is not None]
    n_calls = sum(e.n_tool_calls for e in scored)
    n_invalid = sum(e.n_invalid_calls for e in scored)
    latency: dict[str, list[float]] = {}
    for e in scored:
        for stage, xs in e.latency_ms.items():
            latency.setdefault(stage, []).extend(xs)
    stages = [s for s in STAGES if s in latency] + sorted(s for s in latency if s not in STAGES)
    tasks: dict[str, list[Episode]] = {}
    for e in verified:
        if e.task_id:
            tasks.setdefault(e.task_id, []).append(e)
    tags = sorted({t for e in scored for t in e.tags}, key=lambda t: (TAG_ORDER.get(t, 9), t))
    return {
        "n_runs": len(eps),
        "n_scored": len(scored),
        "excluded": dict(Counter(e.excluded for e in eps if e.excluded)),
        "end_reasons": dict(Counter(e.end_reason for e in scored if e.end_reason)),
        "failure_kinds": dict(Counter(e.failure_kind for e in eps if e.failure_kind)),
        "failure_reasons": dict(Counter(e.failure_reason for e in eps if e.failure_reason).most_common(10)),
        "n_verified": len(verified),
        "n_success": sum(bool(e.verified) for e in verified),
        "success_rate": _rate(sum(bool(e.verified) for e in verified), len(verified)),
        "n_claim_known": len(claimed),
        "n_claimed": sum(bool(e.claimed) for e in claimed),
        "claimed_rate": _rate(sum(bool(e.claimed) for e in claimed), len(claimed)),
        "n_claim_and_verified": len(both),
        "n_claim_agrees": sum(e.claimed == e.verified for e in both),
        "claim_agreement": _rate(sum(e.claimed == e.verified for e in both), len(both)),
        "false_claims": sum(bool(e.claimed) and not e.verified for e in both),
        "missed_claims": sum(not e.claimed and bool(e.verified) for e in both),
        "steps": {"n": len(steps), "mean": _round(statistics.fmean(steps), 2) if steps else None,
                  "median": statistics.median(steps) if steps else None},
        "tool_calls": n_calls,
        "invalid_calls": n_invalid,
        "validity_rate": _rate(n_calls - n_invalid, n_calls),
        "no_tool_call": sum(e.n_no_tool_call for e in scored),
        "failed_actions": sum(e.n_failed_actions for e in scored),
        "no_change": sum(e.n_no_change for e in scored),
        "problems": sum(e.n_problems for e in scored),
        "recovered": sum(e.n_recoveries for e in scored),
        "after_done_calls": sum(e.n_after_done for e in scored),
        "runs_with_problem": len(problem_runs),
        "runs_with_problem_verified": len(problem_verified),
        "runs_succeeded_after_problem": sum(bool(e.verified) for e in problem_verified),
        "lightning_calls": sum(e.n_lightning_calls for e in scored),
        "thinking_calls": sum(e.n_thinking_calls for e in scored),
        "latency_ms": {s: _dist(latency[s]) for s in stages},
        "wall_s": _dist([e.wall_s for e in scored if e.wall_s is not None]),
        "tasks": {"n": len(tasks), "solved_any": sum(any(e.verified for e in v) for v in tasks.values())},
        "lightning_models": sorted({e.lightning for e in eps if e.lightning}),
        "by_tag": {t: _tag_summary([e for e in scored if t in e.tags]) for t in tags},
    }


def _tag_summary(eps: list[Episode]) -> dict:
    verified = [e for e in eps if e.verified is not None]
    steps = [e.steps for e in eps if e.steps is not None]
    n_ok = sum(bool(e.verified) for e in verified)
    return {"n": len(eps), "n_verified": len(verified), "n_success": n_ok, "success_rate": _rate(n_ok, len(verified)),
            "n_claimed": sum(bool(e.claimed) for e in eps),
            "steps_mean": _round(statistics.fmean(steps), 2) if steps else None}


def _group_order(name: str) -> tuple:
    if name.startswith("interactive "):
        first = name[len("interactive "):].split(", ")[0].partition("=")[2]
        return (0, THINKING_ORDER.get(first, 9), name)
    return (1,)


def build_report(episodes: list[Episode], *, command: str, sources: dict, skipped: Skipped) -> dict:
    groups: dict[str, list[Episode]] = {}
    for e in episodes:
        groups.setdefault(e.group, []).append(e)
    names = sorted((n for n in groups if n.startswith("interactive ")), key=_group_order)
    names += [n for n in groups if not n.startswith("interactive ")]  # Gym groups in input order
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": command,
        "sources": sources,
        "skipped": asdict(skipped),
        "groups": {n: summarize(groups[n]) for n in names},
        "runs": [asdict(e) for e in episodes],
    }


# ---------------------------------------------------------------- markdown

def _frac(k: int, n: int) -> str:
    return f"{k}/{n} ({k / n:.0%})" if n else "n/a"


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _runs_cell(g: dict) -> str:
    extra = ", ".join(f"{v} {k}" for k, v in sorted(g["excluded"].items()))
    return f"{g['n_scored']}" + (f" (+{extra})" if extra else "")


def _ms(d: dict | None, scale: float = 1.0, nd: int = 0) -> str:
    if not d or d.get("p50") is None:
        return "-"
    fmt = f"{{:.{nd}f}}"
    return f"{fmt.format(d['p50'] / scale)} / {fmt.format(d['p90'] / scale)}"


def render_markdown(report: dict) -> str:
    groups = report["groups"]
    out = [("| Group | Runs | Verified success | Claimed success | Claim = verified | Steps mean (median) "
            "| Tool-call validity | Problems: corrections / failed / no change | Recovered "
            "| Succeeded after a problem | Thinking calls |"),
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, g in groups.items():
        steps = g["steps"]
        steps_cell = f"{steps['mean']:.1f} ({steps['median']:g})" if steps["n"] else "-"
        corrections = g["invalid_calls"] + g["no_tool_call"]
        out.append("| " + " | ".join(_cell(c) for c in (
            name, _runs_cell(g), _frac(g["n_success"], g["n_verified"]), _frac(g["n_claimed"], g["n_claim_known"]),
            _frac(g["n_claim_agrees"], g["n_claim_and_verified"]), steps_cell,
            _frac(g["tool_calls"] - g["invalid_calls"], g["tool_calls"]),
            f"{g['problems']} ({corrections} / {g['failed_actions']} / {g['no_change']})", str(g["recovered"]),
            f"{g['runs_succeeded_after_problem']}/{g['runs_with_problem_verified']}"
            if g["runs_with_problem_verified"] else "-",
            f"{g['thinking_calls']}/{g['lightning_calls']}")) + " |")

    lat_groups = {n: g for n, g in groups.items() if g["latency_ms"] or g["wall_s"]["n"]}
    if lat_groups:
        out += ["", "Latency per stage, p50 / p90 (ms; wall time per run in s):", "",
                "| Group | " + " | ".join(STAGES) + " | wall (s) |", "|---" * (len(STAGES) + 2) + "|"]
        for name, g in lat_groups.items():
            out.append("| " + " | ".join([_cell(name), *(_ms(g["latency_ms"].get(s)) for s in STAGES),
                                          _ms(g["wall_s"], nd=1)]) + " |")

    tags = sorted({t for g in groups.values() for t in g["by_tag"]}, key=lambda t: (TAG_ORDER.get(t, 9), t))
    if tags:
        out += ["", "Verified success by tag:", "", "| Tag | " + " | ".join(_cell(n) for n in groups) + " |",
                "|---" * (len(groups) + 1) + "|"]
        for t in tags:
            cells = []
            for g in groups.values():
                s = g["by_tag"].get(t)
                cells.append("-" if not s else (f"{s['n_success']}/{s['n_verified']}" if s["n_verified"]
                                                else f"n/a ({s['n']} unverified)"))
            out.append(f"| {_cell(t)} | " + " | ".join(cells) + " |")

    models = sorted({m for g in groups.values() for m in g["lightning_models"]})
    sk = report["skipped"]
    notes = [f"Lightning: {', '.join(models) or 'unknown'}."]
    excluded = [f"{sk['test_runs']} test runs (oracle grounder / fake sandbox)" if sk["test_runs"] else "",
                f"{sk['before_since']} runs before --since" if sk["before_since"] else "",
                f"{sk['corrupt_lines']} corrupt lines" if sk["corrupt_lines"] else ""]
    if any(excluded):
        notes.append("Skipped: " + ", ".join(x for x in excluded if x) + ".")
    out += ["", "Runs = scored runs (+ masked / dropped / incomplete / stopped, which are excluded from every "
            "other column). Steps = Lightning calls per run. Validity = valid / total tool calls. Problems = "
            "corrections (invalid calls + replies without a tool call), failed actions and no-change actions; "
            "Recovered = problems followed by an ok, screen-changing action. " + " ".join(notes),
            f"Generated {report['generated']} by `{report['command']}`."]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    p.add_argument("--runs", type=Path, default=REPO_ROOT / "runs", help="interactive trace root (default: runs)")
    p.add_argument("--no-runs", action="store_true", help="skip interactive runs")
    p.add_argument("--summary", type=Path, nargs="+",
                   help="only the runs in these run_agent_tasks.py summaries (default: all runs, joined with "
                        "every <runs>/summary_*.json)")
    p.add_argument("--since", type=parse_since, help="interactive runs started at/after this local time, "
                                                     "e.g. 20261017-1800")
    p.add_argument("--group-by", default="thinking",
                   help="comma-separated start-event config keys for interactive groups (default: thinking; "
                        "e.g. thinking,grounder_model,lightning.model)")
    p.add_argument("--tasks", type=Path, default=REPO_ROOT / "sandbox" / "tasks.json",
                   help="tasks.json used to find task ids/tags of runs without a summary row")
    p.add_argument("--include-test-runs", action="store_true",
                   help="keep runs with the oracle grounder or the fake sandbox (never report these)")
    p.add_argument("--rollouts", type=Path, nargs="+", default=[], help="gym eval run output .jsonl file(s)")
    p.add_argument("--label", action="append", default=[],
                   help="Gym group label: once for all rollout files, or once per file in order "
                        "(default: 'gym <file stem>')")
    p.add_argument("--json", type=Path, help="write the full report (groups + per-run records) here")
    p.add_argument("--markdown", type=Path, help="write the markdown tables here (also printed)")
    args = p.parse_args(argv)

    skipped = Skipped()
    episodes: list[Episode] = []
    sources: dict[str, Any] = {"runs_dir": None, "summaries": [], "rollouts": []}
    if not args.no_runs:
        group_by = [k.strip() for k in args.group_by.split(",") if k.strip()] or ["thinking"]
        eps, summaries = collect_interactive(args.runs, args.summary, load_tasks(args.tasks), group_by, args.since,
                                             args.include_test_runs, skipped)
        episodes += eps
        sources.update(runs_dir=str(args.runs), summaries=summaries)
    if args.rollouts:
        eps, sources["rollouts"] = collect_gym(args.rollouts, args.label, skipped)
        episodes += eps
    elif args.label:
        p.error("--label needs --rollouts")

    command = "python scripts/agent_metrics.py" + "".join(" " + shlex.quote(a) for a in argv)
    report = build_report(episodes, command=command, sources=sources, skipped=skipped)
    md = render_markdown(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md)
    print(md, end="")
    if not episodes:
        print("\nno runs found", file=sys.stderr)
    if skipped.unreadable:
        print(f"\nunreadable: {'; '.join(skipped.unreadable[:10])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
