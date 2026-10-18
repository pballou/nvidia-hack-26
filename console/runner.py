"""Runs agents for the console: one worker thread with one persistent asyncio loop, one run at a time.

Why a dedicated loop: T2's agent keeps one Playwright browser per process, bound to the
loop that created it, and an agent may block. Neither may happen on uvicorn's loop.

Each Run keeps its ordered list of SSE messages. A subscriber (one SSE connection)
gets a snapshot plus every later message through its own asyncio.Queue, fed with
loop.call_soon_threadsafe. Snapshot and subscribe happen under one lock, so nothing
is lost or duplicated.

Stream order: status(starting), status(running), step*, result?, verdict?,
status(done|stopped|error), end. The verdict is the console checking sandbox state
after the run; the agent never sees it.

console.json (read by the logs page, console/logs.py) goes into the agent's trace dir
(the parent of the events' screenshot paths, or cfg.run_dir_root/<agent run id>) as soon
as that dir is known, with status "running", and is rewritten atomically at the end,
before the final status and `end` messages:
  {console_run_id, agent, request, task_id, task_match, options, sid, start_url, voice, endpoints, status,
   result, verdict, started_ts, ended_ts}
`voice` ({text, model, latency_ms, duration_s} from the voice UI) is only recorded there;
it is never passed to the agent. `task_match` ({method: picked|exact|fuzzy, score, matched_from}, or null) says how
the console chose task_id: picked from the list, or matched from the request (console/app.py parse_task_match). The
agent gets the request as the user wrote it either way; the task only sets the start page, the state and the verdict. `endpoints` is the /api/config endpoints snapshot for this run's
options, taken at run start ({"vlm": {base_url, label, summary_model, served, reachable} | null,
"lightning": {base_url, model, label} | null}; both null for the mock): where its VLM and Lightning ran.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from sandbox.checks import evaluate

log = logging.getLogger("console.runner")

VERDICT_LABEL = "verified by sandbox state, not seen by the agent"
MAX_RUNS_KEPT = 200
CONSOLE_JSON = "console.json"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class Busy(Exception):
    def __init__(self, active_run_id: str):
        super().__init__(f"run {active_run_id} is in progress")
        self.active_run_id = active_run_id


class StartFailed(Exception):
    """The sandbox is unreachable or the agent can't be configured (HTTP 503)."""


@dataclass
class Message:
    seq: int
    event: str
    data: dict

    def sse(self) -> str:
        data = json.dumps(self.data, ensure_ascii=False, default=str)
        return f"id: {self.seq}\nevent: {self.event}\ndata: {data}\n\n"


@dataclass
class Run:
    run_id: str
    request: str
    task_id: str | None
    options: dict
    sid: str
    start_url: str
    status: str = "starting"
    voice: dict | None = None       # recorded in console.json only; never given to the agent
    task_match: dict | None = None  # how task_id was chosen (console.json, /api/runs/{id}); never given to the agent
    endpoints: dict | None = None   # where the VLM and Lightning were at run start (console.json, /api/runs/{id})
    started_ts: float = field(default_factory=time.time)
    ended_ts: float | None = None
    cfg: Any = None
    console_json_dir: Path | None = None  # where console.json was last written
    agent_run_id: str | None = None
    result: dict | None = None
    verdict: dict | None = None
    messages: list[Message] = field(default_factory=list)
    subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = field(default_factory=list)
    closed: bool = False            # 'end' was sent; nothing more is appended
    stop_requested: bool = False
    task: asyncio.Task | None = None
    finished: threading.Event = field(default_factory=threading.Event)


def as_dict(obj: Any) -> dict:
    """StepEvent / RunResult (dataclass with to_dict) or a plain dict -> a fresh dict."""
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "to_dict"):
        return dict(obj.to_dict())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"expected a dict or a dataclass, got {type(obj).__name__}")


def run_dir_root(cfg: Any) -> Path | None:
    """cfg.run_dir_root (AgentConfig, MockConfig) as an absolute Path, or None (e.g. a test's dict cfg)."""
    try:
        root = getattr(cfg, "run_dir_root", None)
        return Path(root).resolve() if root is not None else None
    except Exception:
        return None


def start_url_for(sandbox_url: str, path: str | None, sid: str) -> str:
    path = path or "/"
    return f"{sandbox_url}{path}{'&' if '?' in path else '?'}sid={sid}"


class Runner:
    def __init__(self, *, run_task, make_cfg, sandbox_url: str, tasks_by_id: dict[str, dict],
                 stop_timeout_s: float = 10.0, http_timeout_s: float = 5.0, agent_name: str | None = None):
        self.run_task = run_task
        self.make_cfg = make_cfg
        self.agent_name = agent_name  # "mock" | "real" (console.json's "agent")
        self.sandbox_url = sandbox_url.rstrip("/")
        self.tasks_by_id = tasks_by_id
        self.stop_timeout_s = stop_timeout_s
        self.http_timeout_s = http_timeout_s
        self.lock = threading.Lock()
        self.runs: dict[str, Run] = {}
        self.active: Run | None = None
        self.trace_dirs: dict[str, Path] = {}  # agent run_id -> its trace dir
        self.cfg_roots: list[Path] = []        # every cfg.run_dir_root a run used (logs page roots)
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop_main, name="console-agent-loop", daemon=True)
        self._thread.start()

    def _loop_main(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # --- queries -------------------------------------------------------------------

    def get(self, run_id: str) -> Run | None:
        with self.lock:
            return self.runs.get(run_id)

    def info(self, run: Run) -> dict:
        with self.lock:
            return {"run_id": run.run_id, "status": run.status, "request": run.request, "task_id": run.task_id,
                    "task_match": dict(run.task_match) if run.task_match else None, "options": dict(run.options),
                    "agent_run_id": run.agent_run_id, "result": run.result, "verdict": run.verdict,
                    "endpoints": run.endpoints, "n_events": len(run.messages)}

    def trace_dir(self, agent_run_id: str) -> Path | None:
        with self.lock:
            return self.trace_dirs.get(agent_run_id)

    def live_agent_run_ids(self) -> set[str]:
        """The agent run id of the run in progress (empty when idle or before the agent's first event)."""
        with self.lock:
            run = self.active
            return {run.agent_run_id} if run is not None and run.agent_run_id and not run.closed else set()

    def run_roots(self) -> list[Path]:
        """Every cfg.run_dir_root a run has used so far."""
        with self.lock:
            return list(self.cfg_roots)

    # --- start / stop --------------------------------------------------------------

    def start(self, request: str, task_id: str | None, options: dict, voice: dict | None = None,
              endpoints: dict | None = None, *, task_match: dict | None = None) -> Run:
        """Reset the sandbox session and schedule the agent. Raises Busy or StartFailed. Blocking (call off-loop).
        `voice`, `endpoints` and `task_match` are only recorded (console.json); the agent gets request, start_url
        and cfg."""
        task = self.tasks_by_id[task_id] if task_id else None
        run_id = "c-" + secrets.token_hex(4)
        sid = "console-" + run_id
        run = Run(run_id=run_id, request=request, task_id=task_id, options=dict(options), sid=sid,
                  start_url=start_url_for(self.sandbox_url, task["start_path"] if task else "/", sid),
                  voice=dict(voice) if voice else None,
                  task_match=dict(task_match) if task_match and task_id else None,
                  endpoints=json.loads(json.dumps(endpoints)) if endpoints is not None else None)
        with self.lock:
            if self.active is not None:
                raise Busy(self.active.run_id)
            self.active = run  # reserve the slot before the slow part
            self.runs[run_id] = run
        try:
            if self.run_task is None:
                raise StartFailed("no agent is configured")
            try:
                cfg = self.make_cfg(run.options)
            except Exception as e:
                raise StartFailed(f"agent config failed: {type(e).__name__}: {e}") from e
            self._reset_sandbox(sid, task_id)
            run.cfg = cfg
            root = run_dir_root(cfg)
            if root is not None:
                with self.lock:
                    if root not in self.cfg_roots:
                        self.cfg_roots.append(root)
        except BaseException:
            with self.lock:
                self.active = None
                self.runs.pop(run_id, None)
            raise
        self._prune()
        self.emit(run, "status", {"status": "starting"})
        asyncio.run_coroutine_threadsafe(self._drive(run, cfg), self.loop)
        return run

    def stop(self, run: Run, timeout_s: float | None = None) -> None:
        """Cancel the run on the worker loop and wait for it. Idempotent. Blocking (call off-loop)."""
        with self.lock:
            if run.closed:
                return
            run.stop_requested = True
        self.loop.call_soon_threadsafe(self._cancel, run)
        timeout_s = self.stop_timeout_s if timeout_s is None else timeout_s
        if not run.finished.wait(timeout_s):
            # The agent ignored cancellation (e.g. a blocking call). Close the run so the
            # console isn't stuck; anything it emits later is dropped.
            log.warning("run %s did not stop within %.0f s; abandoning it", run.run_id, timeout_s)
            self._finish(run, "stopped", f"the agent did not stop within {timeout_s:.0f} s")

    def shutdown(self, timeout_s: float = 3.0) -> None:
        with self.lock:
            run = self.active
        if run is not None:
            self.stop(run, timeout_s)
        self.loop.call_soon_threadsafe(self.loop.stop)

    def _cancel(self, run: Run) -> None:  # on the worker loop
        if run.task is not None and not run.task.done():
            run.task.cancel()

    # --- the run itself (worker loop) -------------------------------------------------

    async def _drive(self, run: Run, cfg: Any) -> None:
        run.task = asyncio.current_task()
        status, message = "done", None

        async def on_event(ev: Any) -> None:
            self._on_step(run, as_dict(ev))

        try:
            if run.stop_requested:
                raise asyncio.CancelledError
            result = await self.run_task(run.request, start_url=run.start_url, cfg=cfg, on_event=on_event)
            rd = as_dict(result)
            with self.lock:
                run.result = rd
                if run.agent_run_id is None and rd.get("run_id"):
                    run.agent_run_id = str(rd["run_id"])
                if run.agent_run_id and rd.get("trace_dir"):
                    self.trace_dirs.setdefault(run.agent_run_id, Path(rd["trace_dir"]).resolve())
                self._append(run, "result", rd)
                self._write_console_json(run, "running")
            if run.stop_requested:  # the agent handled the cancel itself and returned
                status, message = "stopped", "stopped by the user"
        except asyncio.CancelledError:
            status, message = "stopped", "stopped by the user"
            run.task.uncancel()
        except Exception as e:
            log.exception("run %s failed", run.run_id)
            status, message = "error", f"{type(e).__name__}: {e}"
        finally:
            self._finish(run, status, message)

    def _on_step(self, run: Run, d: dict) -> None:
        with self.lock:
            if run.agent_run_id is None and d.get("run_id"):
                run.agent_run_id = str(d["run_id"])
            for src, dst in (("screenshot_path", "screenshot_url"), ("overlay_path", "overlay_url")):
                if d.get(src):
                    p = Path(d[src]).resolve()  # the agent runs in this process, so cwd-relative is right
                    aid = run.agent_run_id or p.parent.name
                    self.trace_dirs[aid] = p.parent
                    d[dst] = f"/runs/{quote(aid)}/{quote(p.name)}"
            if run.status == "starting":
                self._append(run, "status", {"status": "running"})
            self._append(run, "step", d)
            self._write_console_json(run, "running")

    def _finish(self, run: Run, status: str, message: str | None) -> None:
        """Verdict (if a task was given), final status, end. Runs once per run, whichever path gets here first."""
        with self.lock:
            if run.closed:
                return
        verdict = self._verdict(run) if run.task_id else None
        with self.lock:
            if run.closed:
                return
            if verdict is not None:
                run.verdict = verdict
                self._append(run, "verdict", verdict)
            run.ended_ts = time.time()
            self._write_console_json(run, status, final=True)  # before `end`: a client that saw `end` can read it
            self._append(run, "status", {"status": status, **({"message": message} if message else {})})
            self._append(run, "end", {})
            run.closed = True
            if self.active is run:
                self.active = None
        run.finished.set()

    # --- console.json (the logs page reads it; console/logs.py) -------------------------

    def _console_json_dir(self, run: Run) -> Path | None:  # lock held
        aid = run.agent_run_id
        if not aid:
            return None
        d = self.trace_dirs.get(aid)
        if d is None and _SAFE_ID_RE.match(aid):
            root = run_dir_root(run.cfg)
            if root is not None and (root / aid).is_dir():
                d = (root / aid).resolve()
        return d

    def _write_console_json(self, run: Run, status: str, final: bool = False) -> None:  # lock held
        """Write console.json atomically into the run's trace dir. Mid-run only when that dir is new
        (so about once per run); nothing after the final write. Never raises: logging must not break a run."""
        if run.closed:
            return
        d = self._console_json_dir(run)
        if d is None or (not final and d == run.console_json_dir):
            return
        doc = {"console_run_id": run.run_id, "agent": self.agent_name, "request": run.request,
               "task_id": run.task_id, "task_match": run.task_match, "options": dict(run.options), "sid": run.sid,
               "start_url": run.start_url,
               "voice": run.voice, "endpoints": run.endpoints, "status": status, "result": run.result,
               "verdict": run.verdict,
               "started_ts": round(run.started_ts, 3),
               "ended_ts": round(run.ended_ts, 3) if run.ended_ts is not None else None}
        tmp = d / f".{CONSOLE_JSON}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
            os.replace(tmp, d / CONSOLE_JSON)
        except Exception as e:
            log.warning("could not write %s for run %s: %s", d / CONSOLE_JSON, run.run_id, e)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return
        old, run.console_json_dir = run.console_json_dir, d
        if old is not None and old != d:  # the trace dir turned out to be elsewhere: drop the stale copy
            try:
                if not os.path.samefile(old, d):
                    (old / CONSOLE_JSON).unlink(missing_ok=True)
            except OSError:
                pass

    # --- sandbox (verifier side; the agent never calls these) ------------------------

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.http_timeout_s, trust_env=False)

    def _reset_sandbox(self, sid: str, task_id: str | None) -> None:
        try:
            with self._client() as c:
                r = c.post(f"{self.sandbox_url}/api/reset", json={"sid": sid, "task_id": task_id})
        except httpx.HTTPError as e:
            raise StartFailed(f"sandbox unreachable at {self.sandbox_url} ({type(e).__name__})") from e
        try:
            ok = r.status_code == 200 and r.json().get("ok") is True
        except ValueError:
            ok = False
        if not ok:
            raise StartFailed(f"sandbox reset failed: HTTP {r.status_code} {r.text[:200]}")

    def _verdict(self, run: Run) -> dict:
        task = self.tasks_by_id[run.task_id]
        try:
            with self._client() as c:
                r = c.get(f"{self.sandbox_url}/api/state", params={"sid": run.sid})
            r.raise_for_status()
            passed, explanation = evaluate(r.json(), task["check"])
        except Exception as e:
            passed, explanation = False, f"could not read sandbox state: {type(e).__name__}: {e}"
        return {"task_id": run.task_id, "passed": bool(passed), "explanation": explanation, "label": VERDICT_LABEL}

    # --- messages ----------------------------------------------------------------------

    def emit(self, run: Run, event: str, data: dict) -> None:
        with self.lock:
            self._append(run, event, data)

    def _append(self, run: Run, event: str, data: dict) -> None:  # lock held
        if run.closed:
            return
        msg = Message(len(run.messages) + 1, event, data)
        run.messages.append(msg)
        if event == "status":
            run.status = data["status"]
        for sub in list(run.subscribers):
            loop, q = sub
            try:
                loop.call_soon_threadsafe(q.put_nowait, msg)
            except RuntimeError:  # that subscriber's loop is gone
                run.subscribers.remove(sub)

    def peek_done(self, run: Run, after: int) -> bool:
        """True if the run is over and has nothing after `after` (SSE answers 204: stop reconnecting)."""
        with self.lock:
            return run.closed and len(run.messages) <= after

    def subscribe(self, run: Run, after: int) -> tuple[list[Message], asyncio.Queue | None]:
        """Messages with seq > after, plus a queue for later ones (None once the run is over). Call on the subscriber's loop."""
        with self.lock:
            backlog = [m for m in run.messages if m.seq > after]
            if run.closed:
                return backlog, None
            q: asyncio.Queue = asyncio.Queue()
            run.subscribers.append((asyncio.get_running_loop(), q))
            return backlog, q

    def unsubscribe(self, run: Run, q: asyncio.Queue) -> None:
        with self.lock:
            run.subscribers = [s for s in run.subscribers if s[1] is not q]

    def _prune(self) -> None:
        with self.lock:
            old = [rid for rid, r in self.runs.items() if r.closed]
            for rid in old[: max(0, len(self.runs) - MAX_RUNS_KEPT)]:
                del self.runs[rid]
