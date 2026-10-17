"""GUI assist console backend (docs/tasks/T1-app.md Part B). Run: python -m console --agent mock|real.

HTTP API (binding for console/static; details in console/README.md):
  GET  /                          console/static/index.html (and /static/*)
  GET  /api/config                {agent, sandbox_url, headed, defaults, asr}
  GET  /api/tasks                 the 12 sandbox tasks, without their checks
  POST /api/run                   {request, task_id?, options?} -> {run_id, sid, start_url}; 409 busy, 400 bad input, 503 unavailable
  GET  /api/runs/{run_id}         run summary
  GET  /api/runs/{run_id}/events  SSE: status, step, result, verdict, end (replay + live; honors Last-Event-ID)
  POST /api/runs/{run_id}/stop    {ok: true}
  POST /api/transcribe            16 kHz mono PCM16 WAV (raw audio/wav or multipart "audio") -> {text, duration_s, latency_ms, model}
  GET  /runs/{agent_run_id}/{step_XX[_overlay].png}  trace images
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from console import asr
from console.agent_bridge import AgentUnavailable, get_agent
from console.runner import Busy, Runner, StartFailed

log = logging.getLogger("console")

CONSOLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONSOLE_DIR.parent

OPTION_CHOICES = {"grounder": ("base", "grounder"), "thinking": ("off", "retry", "on"), "lightning": ("hosted", "local")}
DEFAULT_OPTIONS = {"grounder": "base", "thinking": "retry", "lightning": "hosted"}
MAX_REQUEST_CHARS = 500
IMAGE_RE = re.compile(r"^step_\d{2,3}(_overlay)?\.png$")
AGENT_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass
class ConsoleSettings:
    agent: str = "mock"  # mock | real
    sandbox_url: str = "http://127.0.0.1:8090"
    runs_dir: Path = field(default_factory=lambda: REPO_ROOT / "runs")  # mock traces; the real agent uses its cfg.runs_dir
    headed: bool = False
    step_delay_s: float = 0.8  # mock pacing
    default_lightning: str = "local"  # the console's initial Lightning toggle (local :8000 or hosted API)
    host: str = "127.0.0.1"
    port: int = 8095
    static_dir: Path = field(default_factory=lambda: CONSOLE_DIR / "static")
    tasks_path: Path = field(default_factory=lambda: REPO_ROOT / "sandbox" / "tasks.json")
    mock_scripts: Path = field(default_factory=lambda: CONSOLE_DIR / "mock_scripts.json")
    ping_s: float = 15.0
    stop_timeout_s: float = 10.0

    def __post_init__(self):
        self.sandbox_url = self.sandbox_url.rstrip("/")
        self.runs_dir = repo_path(self.runs_dir)
        self.static_dir = repo_path(self.static_dir)
        self.tasks_path = repo_path(self.tasks_path)
        self.mock_scripts = repo_path(self.mock_scripts)
        if self.default_lightning not in OPTION_CHOICES["lightning"]:
            raise ValueError(f"default_lightning must be one of {OPTION_CHOICES['lightning']}")

    def default_options(self) -> dict:
        return {**DEFAULT_OPTIONS, "lightning": self.default_lightning}


def _error(status: int, message: str, **extra) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=status)


def parse_run_body(body, tasks_by_id: dict, defaults: dict = DEFAULT_OPTIONS) -> tuple[dict | None, str | None]:
    """Validate POST /api/run. Returns (clean, None) or (None, error message)."""
    if not isinstance(body, dict):
        return None, "body must be a JSON object"
    request = body.get("request")
    if not isinstance(request, str) or not request.strip():
        return None, "request is required"
    request = request.strip()
    if len(request) > MAX_REQUEST_CHARS:
        return None, f"request is too long ({len(request)} > {MAX_REQUEST_CHARS} characters)"
    task_id = body.get("task_id")
    if task_id is not None and (not isinstance(task_id, str) or task_id not in tasks_by_id):
        return None, f"unknown task_id {task_id!r}"
    options = body.get("options") or {}
    if not isinstance(options, dict):
        return None, "options must be an object"
    unknown = sorted(set(options) - set(OPTION_CHOICES))
    if unknown:
        return None, f"unknown option(s): {', '.join(unknown)}"
    clean = dict(defaults)
    for key, value in options.items():
        if value not in OPTION_CHOICES[key]:
            return None, f"options.{key} must be one of {', '.join(OPTION_CHOICES[key])}"
        clean[key] = value
    return {"request": request, "task_id": task_id, "options": clean}, None


def _last_event_id(request: Request) -> int:
    raw = request.headers.get("last-event-id") or request.query_params.get("last_event_id") or "0"
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def create_app(settings: ConsoleSettings | None = None, agent: tuple | None = None) -> FastAPI:
    """`agent` = (run_task, make_cfg) overrides the --agent choice (tests inject fakes here)."""
    settings = settings or ConsoleSettings()
    tasks = json.loads(settings.tasks_path.read_text())
    tasks_by_id = {t["id"]: t for t in tasks}
    public_tasks = [{k: t[k] for k in ("id", "request", "start_path", "tags", "steps_min")} for t in tasks]

    agent_error = None
    if agent is None:
        try:
            agent = get_agent(settings)
        except AgentUnavailable as e:
            agent_error = str(e)
            log.error("%s", e)
    run_task, make_cfg = agent or (None, None)
    runner = Runner(run_task=run_task, make_cfg=make_cfg, sandbox_url=settings.sandbox_url,
                    tasks_by_id=tasks_by_id, stop_timeout_s=settings.stop_timeout_s)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await run_in_threadpool(runner.shutdown)

    app = FastAPI(title="GUI assist console", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.settings = settings
    app.state.runner = runner
    app.mount("/static", StaticFiles(directory=settings.static_dir, check_dir=False), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        page = settings.static_dir / "index.html"
        if page.is_file():
            return FileResponse(page, media_type="text/html", headers={"Cache-Control": "no-store"})
        return HTMLResponse("<p>console/static/index.html is not built yet. The API is up: "
                            "<a href='/api/config'>/api/config</a></p>", status_code=404)

    @app.get("/api/config")
    def config():
        return {"agent": settings.agent, "sandbox_url": settings.sandbox_url, "headed": settings.headed,
                "defaults": settings.default_options(), "asr": asr.info()}

    @app.get("/api/tasks")
    def list_tasks():
        return public_tasks

    @app.post("/api/run")
    async def start_run(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "body must be JSON")
        clean, problem = parse_run_body(body, tasks_by_id, settings.default_options())
        if problem:
            return _error(400, problem)
        if agent_error:
            return _error(503, agent_error)
        try:
            run = await run_in_threadpool(runner.start, clean["request"], clean["task_id"], clean["options"])
        except Busy as e:
            return _error(409, "a run is already in progress; stop it first", active_run_id=e.active_run_id)
        except StartFailed as e:
            return _error(503, str(e))
        return {"run_id": run.run_id, "sid": run.sid, "start_url": run.start_url}

    @app.post("/api/transcribe")
    async def transcribe(request: Request):
        """Voice input: the browser's WAV -> local ASR engine (console/asr.py). Never leaves the machine."""
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > asr.MAX_BYTES + 4096:
            return _error(413, f"recording is too large (max {asr.MAX_SECONDS:.0f} s)")
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            form = await request.form(max_part_size=asr.MAX_BYTES + 4096)
            upload = form.get("audio")
            if upload is None or isinstance(upload, str):
                return _error(400, "multipart upload needs an 'audio' file field")
            data = await upload.read()
        else:
            data = await request.body()
        if len(data) > asr.MAX_BYTES:
            return _error(413, f"recording is too large (max {asr.MAX_SECONDS:.0f} s)")
        try:
            return await run_in_threadpool(asr.transcribe, data)  # engines are blocking; keep SSE responsive
        except asr.AudioError as e:
            return _error(400, str(e))
        except asr.EngineUnavailable as e:
            return _error(503, str(e))

    @app.get("/api/runs/{run_id}")
    def run_info(run_id: str):
        run = runner.get(run_id)
        if run is None:
            return _error(404, f"unknown run {run_id!r}")
        return runner.info(run)

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request):
        run = runner.get(run_id)
        if run is None:
            return _error(404, f"unknown run {run_id!r}")
        after = _last_event_id(request)
        if runner.peek_done(run, after):
            return Response(status_code=204)  # tells EventSource to stop reconnecting

        async def stream():
            backlog, q = runner.subscribe(run, after)
            try:
                for msg in backlog:
                    yield msg.sse()
                if q is None:
                    return
                while True:
                    try:
                        msg = await asyncio.wait_for(q.get(), settings.ping_s)
                    except TimeoutError:
                        yield ": ping\n\n"
                        continue
                    yield msg.sse()
                    if msg.event == "end":
                        return
            finally:
                if q is not None:
                    runner.unsubscribe(run, q)

        return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)

    @app.post("/api/runs/{run_id}/stop")
    async def stop_run(run_id: str):
        run = runner.get(run_id)
        if run is None:
            return _error(404, f"unknown run {run_id!r}")
        await run_in_threadpool(runner.stop, run)
        return {"ok": True}

    @app.get("/runs/{agent_run_id}/{filename}")
    def trace_image(agent_run_id: str, filename: str):
        if not IMAGE_RE.match(filename) or not AGENT_RUN_ID_RE.match(agent_run_id):
            return _error(404, "not found")
        path = (runner.trace_dir(agent_run_id) or settings.runs_dir / agent_run_id) / filename
        if not path.is_file():
            return _error(404, "not found")
        return FileResponse(path, media_type="image/png")

    return app
