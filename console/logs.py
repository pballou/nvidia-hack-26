"""Logs page backend: run summaries and full traces for GET /logs (console/static/logs.html).

Sources, read-only:
  <root>/<run_id>/events.jsonl   the agent's StepEvents (agent/trace.py; console/mock_agent.py)
  <root>/<run_id>/console.json   written by console/runner.py for runs started from the console:
                                 request, task_id, options, voice, status, result, verdict, timestamps
roots = the console's --runs-dir plus the agent's cfg.run_dir_root (deduplicated). Sandbox state
is never read here: the verdict is the one the console recorded in console.json.

Honesty rules: a field the trace doesn't have is null (the page shows "—"); nothing is estimated.
A run is "mock" when any payload says "mock": true (or console.json says the console ran the mock).
For mock runs the config fallbacks for model names are skipped: the mock runs no Lightning, grounder
or VLM, so only names the mock itself declares are shown. Prompt identity (agent/trace.py: plan
prompt_version + prompt_sha, observation summary.prompt_sha, grounding prompt_sha) comes from the
events only; the summary's prompt_sha is keyed like its models (lightning, grounder, summary).

HTTP (binding for console/static/logs.js; see console/README.md):
  GET /logs                                  console/static/logs.html
  GET /api/logs/runs?limit=200               {"runs": [RunSummary, ...], "total": N} newest first
  GET /api/logs/runs/{run_id}                {"summary", "console", "events"} (indented: detail_json)
  GET /api/logs/runs/{run_id}/download       the same bytes as an attachment (run-<run_id>.json); the page's
                                             Copy JSON copies the detail's text, so Copy == Download byte for byte

Pure functions (tested without a server): read_events, read_console_json, summarize_run,
run_detail, find_run_dir, list_runs.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import threading
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterable
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

EVENTS_FILE = "events.jsonl"
CONSOLE_FILE = "console.json"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")             # a run dir name the logs API can address
IMAGE_RE = re.compile(r"^step_\d{2,3}(_overlay)?\.png$")      # the only files GET /runs/... serves
IMAGE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
STAGES = ("voice", "summary", "plan", "ground", "act")
DONE_REASONS = ("done", "max_steps", "stopped", "error")
DEFAULT_LIMIT = 200
MAX_LIMIT = 1000


# --- small helpers -------------------------------------------------------------------------------


def valid_run_id(run_id: Any) -> bool:
    """A run dir name: ^[A-Za-z0-9._-]{1,80}$, and not '.' or '..' (or any all-dots name)."""
    return isinstance(run_id, str) and bool(RUN_ID_RE.match(run_id)) and set(run_id) != {"."}


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def _str(v: Any) -> str | None:
    return v if isinstance(v, str) and v.strip() else None


def _dict(v: Any) -> dict:
    return v if isinstance(v, dict) else {}


def _payload(ev: dict) -> dict:
    return _dict(ev.get("payload"))


def _p50(values: list[float]) -> float | None:
    return round(statistics.median(values), 1) if values else None


def _basename(path: str) -> str:
    """Basename of a path written on any OS (traces may come from the GPU box or a Mac)."""
    return PureWindowsPath(path).name if "\\" in path and "/" not in path else PurePosixPath(path).name


# --- reading ------------------------------------------------------------------------------------


def _no_constant(_c: str) -> None:
    return None


def _finite_float(s: str) -> float | None:
    """1e999 parses to inf, which strict JSON can't carry: null, like NaN/Infinity."""
    v = float(s)
    return v if math.isfinite(v) else None


def parse_event_lines(data: bytes) -> tuple[list[dict], int]:
    """(events, n_bad_lines). Bad = not UTF-8, not JSON, or not a JSON object (e.g. a truncated last line).
    NaN/Infinity (and overflowing numbers like 1e999) become null so every event can be sent as strict JSON."""
    events: list[dict] = []
    bad = 0
    for raw in data.split(b"\n"):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw.decode("utf-8"), parse_constant=_no_constant, parse_float=_finite_float)
        except (UnicodeDecodeError, ValueError, RecursionError):
            bad += 1
            continue
        if isinstance(obj, dict):
            events.append(obj)
        else:
            bad += 1
    return events, bad


def read_events(path: Path) -> tuple[list[dict], int]:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return [], 0
    return parse_event_lines(data)


def read_console_json(path: Path) -> dict | None:
    """console.json as a dict, or None when it is missing, unreadable or not an object."""
    try:
        obj = json.loads(Path(path).read_bytes().decode("utf-8"), parse_constant=_no_constant, parse_float=_finite_float)
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        return None
    return obj if isinstance(obj, dict) else None


# --- derivation ---------------------------------------------------------------------------------


def _start_config(events: list[dict]) -> dict:
    for ev in events:
        if ev.get("kind") == "start":
            return _dict(_payload(ev).get("config"))
    return {}


def is_mock(events: list[dict], console: dict | None = None) -> bool:
    if _dict(console).get("agent") == "mock":
        return True
    return any(_payload(ev).get("mock") is True for ev in events)


def event_model(ev: dict, config: dict, mock: bool = False) -> str | None:
    """The model behind one event: plan -> Lightning, grounding -> grounder, observation -> screen summary.
    Payload first; then the start event's config, except for mock runs (they run none of these models)."""
    kind, p = ev.get("kind"), _payload(ev)
    if kind == "plan":
        own, fallback = _str(p.get("model")), _str(_dict(config.get("lightning")).get("model"))
    elif kind == "grounding":
        own, fallback = _str(p.get("model")), _str(config.get("grounder_model"))
    elif kind == "observation":
        own, fallback = _str(_dict(p.get("summary")).get("model")), _str(config.get("summary_model"))
    else:
        return None
    return own or (None if mock else fallback)


def _first_model(events: list[dict], kind: str, config: dict, mock: bool) -> str | None:
    for ev in events:
        if ev.get("kind") == kind:
            m = event_model(ev, config, mock=True)  # payload-declared only
            if m:
                return m
    return event_model({"kind": kind}, config, mock)  # the config fallback (None for mock runs)


def event_prompt_sha(ev: dict) -> str | None:
    """agent/trace.py prompt_sha of the prompt behind one event: plan and grounding payloads, an observation's
    summary. Payload only (the run config doesn't record it); the mock declares "mock"."""
    kind, p = ev.get("kind"), _payload(ev)
    if kind in ("plan", "grounding"):
        return _str(p.get("prompt_sha"))
    if kind == "observation":
        return _str(_dict(p.get("summary")).get("prompt_sha"))
    return None


def _first_prompt_sha(events: list[dict], kind: str) -> str | None:
    return next((s for s in (event_prompt_sha(ev) for ev in events if ev.get("kind") == kind) if s), None)


def _voice(console: dict | None) -> dict | None:
    v = _dict(console).get("voice")
    if not isinstance(v, dict):
        return None
    return {"text": v.get("text") if isinstance(v.get("text"), str) else None,
            "model": _str(v.get("model")),
            "latency_ms": _num(v.get("latency_ms")),
            "duration_s": _num(v.get("duration_s"))}


def _status(done: dict | None, console: dict | None) -> str:
    if done is not None:
        reason = _payload(done).get("reason")
        return reason if reason in DONE_REASONS else "unknown"
    cs = _dict(console).get("status")
    return cs if cs in ("running", "stopped", "error") else "unknown"


def summarize_run(events: list[dict], console_json: dict | None = None, *, run_id: str | None = None,
                  n_bad_lines: int = 0) -> dict:
    """RunSummary for one trace (see the module docstring and console/README.md)."""
    console = console_json if isinstance(console_json, dict) else None
    c = _dict(console)
    mock = is_mock(events, console)
    config = _start_config(events)
    start = next((ev for ev in events if ev.get("kind") == "start"), None)
    done = next((ev for ev in reversed(events) if ev.get("kind") == "done"), None)
    dp = _payload(done) if done else {}

    stamps = [t for t in (_num(ev.get("ts")) for ev in events) if t is not None]  # agent events only
    started, ended = (stamps[0], stamps[-1]) if stamps else (None, None)
    total_ms = round((ended - started) * 1000) if stamps else None

    timings: dict[str, list[float]] = {s: [] for s in STAGES}
    for ev in events:
        for stage, ms in _dict(ev.get("timings_ms")).items():
            if stage in timings and _num(ms) is not None:
                timings[stage].append(_num(ms))
    voice = _voice(console)
    if voice and voice["latency_ms"] is not None:
        timings["voice"].append(voice["latency_ms"])

    steps = dp.get("steps") if isinstance(dp.get("steps"), int) and not isinstance(dp.get("steps"), bool) else None
    if steps is None:
        seen = [ev["step"] for ev in events if isinstance(ev.get("step"), int) and not isinstance(ev.get("step"), bool)]
        steps = max(seen) if seen else None

    claimed = dp.get("success_claimed")
    if not isinstance(claimed, bool):
        claimed = _dict(c.get("result")).get("success_claimed")
        claimed = claimed if isinstance(claimed, bool) else None
    verdict = _dict(c.get("verdict"))

    request = _payload(start).get("request") if start else None
    request = request if isinstance(request, str) else (c.get("request") if isinstance(c.get("request"), str) else None)
    plan_pv = next((_str(_payload(ev).get("prompt_version")) for ev in events
                    if ev.get("kind") == "plan" and _str(_payload(ev).get("prompt_version"))), None)

    return {
        "run_id": run_id if run_id is not None else (_str(events[0].get("run_id")) if events else None),
        "console_run_id": _str(c.get("console_run_id")),
        "agent": "mock" if mock else "real",
        "started_ts": started,
        "ended_ts": ended,
        "request": request,
        "task_id": _str(c.get("task_id")),
        "models": {
            "lightning": _first_model(events, "plan", config, mock),
            "grounder": _first_model(events, "grounding", config, mock),
            "summary": _first_model(events, "observation", config, mock),
            "asr": voice["model"] if voice else None,
        },
        "status": _status(done, console),
        "steps": steps,
        "success_claimed": claimed,
        "verified": verdict.get("passed") if isinstance(verdict.get("passed"), bool) else None,
        "verdict_explanation": _str(verdict.get("explanation")),
        "total_ms": total_ms,
        "stage_p50_ms": {s: _p50(v) for s, v in timings.items()},
        "n_events": len(events),
        "n_errors": sum(1 for ev in events if ev.get("kind") == "error"),
        "n_invalid_tool_calls": sum(1 for ev in events if ev.get("kind") == "tool_call" and _payload(ev).get("valid") is False),
        "voice": voice,
        # beyond the contract, for the detail view
        "n_bad_lines": n_bad_lines,
        "prompt_version": plan_pv or (None if mock else _str(config.get("prompt_version"))),
        "prompt_sha": {  # keyed like models; from the events only
            "lightning": _first_prompt_sha(events, "plan"),
            "grounder": _first_prompt_sha(events, "grounding"),
            "summary": _first_prompt_sha(events, "observation"),
        },
        "lightning_root": None if mock else _str(_dict(config.get("lightning")).get("root")),
    }


def image_url(run_dir: Path, run_id: str, path: Any) -> str | None:
    """/runs/<run_id>/<basename> when GET /runs/... would serve it: a step image that exists in run_dir."""
    if not isinstance(path, str) or not path or not IMAGE_RUN_ID_RE.match(run_id):
        return None
    name = _basename(path)
    if not IMAGE_RE.match(name) or not (run_dir / name).is_file():
        return None
    return f"/runs/{quote(run_id)}/{quote(name)}"


def run_detail(run_dir: Path, run_id: str, *, events: list[dict] | None = None, n_bad_lines: int = 0,
               console_json: dict | None = None) -> dict:
    """{"summary", "console", "events"}: the agent's events in file order with derived t_ms, model, prompt_sha,
    source, screenshot_url/overlay_url, plus a console 'voice' event first and a console 'verdict' event last."""
    run_dir = Path(run_dir)
    if events is None:
        events, n_bad_lines = read_events(run_dir / EVENTS_FILE)
        console_json = read_console_json(run_dir / CONSOLE_FILE)
    summary = summarize_run(events, console_json, run_id=run_id, n_bad_lines=n_bad_lines)
    mock = summary["agent"] == "mock"
    config = _start_config(events)
    t0 = summary["started_ts"]

    def rel(ts: Any) -> float | None:
        ts = _num(ts)
        return round((ts - t0) * 1000, 1) if ts is not None and t0 is not None else None

    out: list[dict] = []
    if summary["voice"]:
        out.append({"run_id": run_id, "step": None, "kind": "voice", "ts": None, "t_ms": None,
                    "payload": dict(summary["voice"]), "model": summary["voice"]["model"], "source": "console"})
    for ev in events:
        d = dict(ev)
        d["t_ms"] = rel(ev.get("ts"))
        d["model"] = event_model(ev, config, mock)
        d["prompt_sha"] = event_prompt_sha(ev)
        d["source"] = "agent"
        for src, dst in (("screenshot_path", "screenshot_url"), ("overlay_path", "overlay_url")):
            d.pop(dst, None)
            url = image_url(run_dir, run_id, ev.get(src))
            if url:
                d[dst] = url
        out.append(d)
    verdict = _dict(_dict(console_json).get("verdict"))
    if verdict:
        ts = _num(_dict(console_json).get("ended_ts"))
        out.append({"run_id": run_id, "step": None, "kind": "verdict", "ts": ts, "t_ms": rel(ts),
                    "payload": {"passed": verdict.get("passed") if isinstance(verdict.get("passed"), bool) else None,
                                "explanation": verdict.get("explanation"), "label": verdict.get("label"),
                                "task_id": verdict.get("task_id")},
                    "model": None, "source": "console"})
    return {"summary": summary, "console": console_json if isinstance(console_json, dict) else None, "events": out}


# --- finding runs -------------------------------------------------------------------------------


def dedupe_roots(roots: Iterable[Path | str | None]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for r in roots:
        if r is None:
            continue
        p = Path(r).resolve()
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def find_run_dir(roots: Iterable[Path], run_id: str) -> Path | None:
    """<root>/<run_id> holding an events.jsonl, first root wins. None for invalid ids."""
    if not valid_run_id(run_id):
        return None
    for root in dedupe_roots(roots):
        d = root / run_id
        if (d / EVENTS_FILE).is_file():
            return d
    return None


def iter_run_dirs(roots: Iterable[Path]) -> dict[str, Path]:
    """{run_id: dir} for every <root>/<run_id>/events.jsonl; a run id in two roots keeps the first."""
    found: dict[str, Path] = {}
    for root in dedupe_roots(roots):
        try:
            entries = list(os.scandir(root))
        except OSError:
            continue
        for e in entries:
            if e.name in found or not valid_run_id(e.name):
                continue
            try:
                if e.is_dir() and os.path.isfile(os.path.join(e.path, EVENTS_FILE)):
                    found[e.name] = Path(e.path)
            except OSError:
                continue
    return found


def _sig(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


class RunIndex:
    """Summaries cached by (events.jsonl, console.json) (mtime_ns, size): a listing re-reads only changed runs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cache: dict[Path, tuple[tuple, dict, float]] = {}
        self.reads = 0  # summaries computed from disk (tests check the cache with it)

    def summary(self, run_dir: Path, run_id: str) -> tuple[dict, float]:
        """(summary, sort key). The sort key is started_ts, else the events file's mtime."""
        ev_sig = _sig(run_dir / EVENTS_FILE)
        sig = (ev_sig, _sig(run_dir / CONSOLE_FILE))
        with self._lock:
            hit = self._cache.get(run_dir)
        if hit and hit[0] == sig:
            return hit[1], hit[2]
        events, bad = read_events(run_dir / EVENTS_FILE)
        summary = summarize_run(events, read_console_json(run_dir / CONSOLE_FILE), run_id=run_id, n_bad_lines=bad)
        key = summary["started_ts"] if summary["started_ts"] is not None else (ev_sig[0] / 1e9 if ev_sig else 0.0)
        with self._lock:
            self.reads += 1
            self._cache[run_dir] = (sig, summary, key)
        return summary, key

    def forget_except(self, keep: Iterable[Path]) -> None:
        keep = set(keep)
        with self._lock:
            for d in [d for d in self._cache if d not in keep]:
                del self._cache[d]


def list_runs(roots: Iterable[Path], limit: int | None = DEFAULT_LIMIT, index: RunIndex | None = None) -> list[dict]:
    """RunSummaries for every run under `roots`, newest first (by the first event's ts), at most `limit` (None: all)."""
    index = index or RunIndex()
    dirs = iter_run_dirs(roots)
    rows = [index.summary(d, rid) for rid, d in dirs.items()]
    index.forget_except(dirs.values())
    rows.sort(key=lambda r: (r[1], r[0]["run_id"]), reverse=True)
    return [dict(s) for s, _ in (rows if limit is None else rows[:max(0, limit)])]


def apply_live(summary: dict, live_run_ids: set[str]) -> dict:
    """'running' only while this console process is running that agent run. console.json left at
    'running' by a console that was killed becomes 'unknown'; a live run without console.json yet is 'running'."""
    live = summary.get("run_id") in live_run_ids
    if live and summary["status"] in ("running", "unknown"):
        return {**summary, "status": "running"}
    if not live and summary["status"] == "running":
        return {**summary, "status": "unknown"}
    return summary


# --- HTTP ---------------------------------------------------------------------------------------


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def detail_json(out: dict) -> str:
    """The one serialization of a run's detail: the detail endpoint and the download send these exact bytes,
    and the page's Copy JSON copies the detail response's text, so Copy and Download are byte-identical."""
    return json.dumps(out, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def make_router(*, static_dir: Path, roots: Callable[[], list[Path]],
                live: Callable[[], set[str]] = lambda: set()) -> APIRouter:
    """roots() -> the run roots to scan; live() -> agent run ids this console is running right now."""
    router = APIRouter()
    index = RunIndex()

    def detail(run_id: str) -> dict | None:
        d = find_run_dir(roots(), run_id)
        if d is None:
            return None
        out = run_detail(d, run_id)
        out["summary"] = apply_live(out["summary"], live())
        return out

    @router.get("/logs", include_in_schema=False)
    def logs_page():
        page = Path(static_dir) / "logs.html"
        if page.is_file():
            return FileResponse(page, media_type="text/html", headers={"Cache-Control": "no-store"})
        return HTMLResponse("<p>console/static/logs.html is not built yet. The API is up: "
                            "<a href='/api/logs/runs'>/api/logs/runs</a></p>", status_code=404)

    @router.get("/api/logs/runs")
    async def runs(request: Request):
        raw = request.query_params.get("limit", str(DEFAULT_LIMIT))
        try:
            limit = int(raw)
        except ValueError:
            return _error(400, f"limit must be an integer 1..{MAX_LIMIT}")
        if not 1 <= limit <= MAX_LIMIT:
            return _error(400, f"limit must be an integer 1..{MAX_LIMIT}")

        def work() -> dict:
            all_runs = list_runs(roots(), limit=None, index=index)
            now_live = live()
            return {"runs": [apply_live(s, now_live) for s in all_runs[:limit]], "total": len(all_runs)}

        return await run_in_threadpool(work)

    @router.get("/api/logs/runs/{run_id}")
    async def run(run_id: str):
        out = await run_in_threadpool(detail, run_id) if valid_run_id(run_id) else None
        if out is None:
            return _error(404, f"unknown run {run_id!r}")
        return Response(detail_json(out), media_type="application/json", headers={"Cache-Control": "no-store"})

    @router.get("/api/logs/runs/{run_id}/download")
    async def download(run_id: str):
        out = await run_in_threadpool(detail, run_id) if valid_run_id(run_id) else None
        if out is None:
            return _error(404, f"unknown run {run_id!r}")
        return Response(detail_json(out), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="run-{run_id}.json"',
                                 "Cache-Control": "no-store"})

    return router
