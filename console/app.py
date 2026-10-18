"""GUI assist console backend (docs/tasks/T1-app.md Part B). Run: python -m console --agent mock|real.

HTTP API (binding for console/static; details in console/README.md):
  GET  /                          console/static/index.html (and /static/*, Cache-Control: no-cache)
  GET  /api/config                {agent, sandbox_url, headed, defaults, asr, endpoints}
  GET  /api/tasks                 the 12 sandbox tasks, without their checks
  POST /api/run                   {request, task_id?, options?, voice?, task_match?} -> {run_id, sid, start_url};
                                  409 busy, 400 bad input (or a grounder the VLM doesn't serve), 503 unavailable
  GET  /api/runs/{run_id}         run summary
  GET  /api/runs/{run_id}/events  SSE: status, step, result, verdict, end (replay + live; honors Last-Event-ID)
  POST /api/runs/{run_id}/stop    {ok: true}
  POST /api/transcribe            16 kHz mono PCM16 WAV (raw audio/wav or multipart "audio") -> {text, duration_s, latency_ms, model}
  GET  /runs/{agent_run_id}/{step_XX[_overlay].png}  trace images
  GET  /logs, /api/logs/runs[/{run_id}[/download]]   the logs page and its API (console/logs.py)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from console import asr, logs
from console.agent_bridge import (NO_ENDPOINTS, SERVED_NAMES, AgentUnavailable, ServedModels, endpoints_for, get_agent,
                                  unserved_grounder)
from console.runner import Busy, Runner, StartFailed, run_dir_root

log = logging.getLogger("console")

CONSOLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONSOLE_DIR.parent

OPTION_CHOICES = {"grounder": ("base", "grounder"), "thinking": ("off", "retry", "on"), "lightning": ("hosted", "local")}
DEFAULT_OPTIONS = {"grounder": "grounder", "thinking": "retry", "lightning": "local"}
MAX_REQUEST_CHARS = 500
MAX_VOICE_TEXT_CHARS = 500
MAX_VOICE_MODEL_CHARS = 120
VOICE_KEYS = ("text", "model", "latency_ms", "duration_s")
TASK_MATCH_KEYS = ("method", "score", "matched_from")
TASK_MATCH_METHODS = ("picked", "exact", "fuzzy")
IMAGE_RE = logs.IMAGE_RE                 # one definition: the logs page builds image URLs with the same rules
AGENT_RUN_ID_RE = logs.IMAGE_RUN_ID_RE
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
    default_grounder: str = "grounder"  # the console's initial Grounder choice (--grounder base|grounder)
    # Real agent only (load_config overrides; None = the agent's config and env vars decide):
    vlm_url: str | None = None          # --vlm-url -> vlm={"base_url": ...}
    summary_model: str | None = None    # --summary-model base|grounder
    vlm_label: str | None = None        # --vlm-label / CONSOLE_VLM_LABEL (default: by port, agent_bridge.vlm_label)
    served_ttl_s: float = 10.0          # GET <vlm>/models is re-checked at most this often
    served_timeout_s: float = 2.0
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
        if self.default_grounder not in OPTION_CHOICES["grounder"]:
            raise ValueError(f"default_grounder must be one of {OPTION_CHOICES['grounder']}")
        if self.summary_model is not None and self.summary_model not in SERVED_NAMES:
            raise ValueError(f"summary_model must be one of {SERVED_NAMES}")
        self.vlm_url = self.vlm_url.strip().rstrip("/") if self.vlm_url and self.vlm_url.strip() else None
        self.vlm_label = self.vlm_label.strip() if self.vlm_label and self.vlm_label.strip() else None

    def default_options(self) -> dict:
        return {**DEFAULT_OPTIONS, "grounder": self.default_grounder, "lightning": self.default_lightning}


class RevalidatedStaticFiles(StaticFiles):
    """/static/* with Cache-Control: no-cache: every page load revalidates each file by its ETag (a cheap 304 when
    it is unchanged). Without it browsers apply heuristic freshness and keep running an old console.js or voice.js
    under the always-fresh index.html after the file changes on disk, until a hard reload."""

    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)  # 200, or 304 for a matching If-None-Match
        response.headers["Cache-Control"] = "no-cache"
        return response


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
    voice, problem = parse_voice(body.get("voice"))
    if problem:
        return None, problem
    task_match, problem = parse_task_match(body.get("task_match"), task_id, request, tasks_by_id)
    if problem:
        return None, problem
    return {"request": request, "task_id": task_id, "options": clean, "voice": voice, "task_match": task_match}, None


def parse_task_match(match, task_id: str | None, request: str, tasks_by_id: dict) -> tuple[dict | None, str | None]:
    """Optional `task_match` of POST /api/run: how the console chose task_id, {method: picked|exact|fuzzy,
    score: 0..1 | null, matched_from: str}. Recorded in console.json (the logs page and the banner say "Checked as
    task: <id> (matched from your request)" or "(picked from the list)"); never passed to the agent.
    Left out with a task_id whose text the request is (request_key equal), it is recorded as an exact match: the page
    doesn't send those. Left out otherwise, nothing is recorded (the caller chose the task some other way)."""
    if match is None:
        task_text = (tasks_by_id.get(task_id) or {}).get("request") if task_id is not None else None
        if isinstance(task_text, str) and request_key(request) and request_key(request) == request_key(task_text):
            return {"method": "exact", "score": 1.0, "matched_from": request}, None
        return None, None
    if not isinstance(match, dict):
        return None, "task_match must be an object"
    unknown = sorted(set(match) - set(TASK_MATCH_KEYS))
    if unknown:
        return None, f"unknown task_match field(s): {', '.join(unknown)}"
    missing = [k for k in TASK_MATCH_KEYS if k not in match]
    if missing:
        return None, f"task_match needs {', '.join(missing)}"
    if task_id is None:
        return None, "task_match needs a task_id"
    if match["method"] not in TASK_MATCH_METHODS:
        return None, f"task_match.method must be one of {', '.join(TASK_MATCH_METHODS)}"
    score = match["score"]
    if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
                              or not 0 <= score <= 1):
        return None, "task_match.score must be a number from 0 to 1, or null"
    if not isinstance(match["matched_from"], str) or len(match["matched_from"]) > MAX_REQUEST_CHARS:
        return None, f"task_match.matched_from must be a string of at most {MAX_REQUEST_CHARS} characters"
    return {"method": match["method"], "score": None if score is None else float(score),
            "matched_from": match["matched_from"]}, None


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_WORD = "(?:" + "|".join([*_NUMBER_WORDS, "hundred"]) + ")"
_NOT_ALNUM_BEFORE, _NOT_ALNUM_AFTER = r"(?<![^\W_])", r"(?![^\W_])"  # JS (?<![\p{L}\p{N}]) / (?![\p{L}\p{N}])
_NUMBER_RUN_RE = re.compile(f"{_NOT_ALNUM_BEFORE}{_NUMBER_WORD}(?:[\\s-]+{_NUMBER_WORD})*{_NOT_ALNUM_AFTER}")


def _number_words_to_digits(run: str) -> str:
    """'fifty' -> '50', 'fifty-five' -> '55', 'two hundred five' -> '205' (console.js numberWordsToDigits)."""
    out: list[int] = []
    cur: int | None = None
    for w in re.split(r"[\s-]+", run):
        if w == "hundred":
            if cur is not None and cur % 100 == 0 and cur > 0:
                out.append(cur)
                cur = None
            cur = (1 if cur is None else cur) * 100
            continue
        v = _NUMBER_WORDS[w]
        if cur is not None and cur >= 100 and cur % 100 == 0 and v < 100:
            cur += v
        elif cur is not None and cur % 100 >= 20 and cur % 10 == 0 and v < 10:
            cur += v
        else:
            if cur is not None:
                out.append(cur)
            cur = v
    if cur is not None:
        out.append(cur)
    return " ".join(map(str, out))


def request_key(text: str) -> str:
    """console.js requestKey: case, accents, punctuation, whitespace and how an amount is written ('$50' = '50 dollars'
    = 'fifty dollars') don't matter. Two texts with the same key are an exact match."""
    s = "".join(c for c in unicodedata.normalize("NFKD", str(text)) if not unicodedata.category(c).startswith("M"))
    s = _NUMBER_RUN_RE.sub(lambda m: _number_words_to_digits(m.group(0)), s.lower())
    s = re.sub(r"\$\s*([0-9]+)(?:\.00)?(?![0-9])", r"\1", s)
    s = re.sub(f"([0-9])\\s*dollars?{_NOT_ALNUM_AFTER}", r"\1", s)
    return "".join(c for c in s if c.isalnum())


def parse_voice(voice) -> tuple[dict | None, str | None]:
    """Optional `voice` of POST /api/run: {text, model, latency_ms, duration_s} from /api/transcribe.
    Recorded in console.json for the logs page; never passed to the agent."""
    if voice is None:
        return None, None
    if not isinstance(voice, dict):
        return None, "voice must be an object"
    unknown = sorted(set(voice) - set(VOICE_KEYS))
    if unknown:
        return None, f"unknown voice field(s): {', '.join(unknown)}"
    missing = [k for k in VOICE_KEYS if k not in voice]
    if missing:
        return None, f"voice needs {', '.join(missing)}"
    for key, limit in (("text", MAX_VOICE_TEXT_CHARS), ("model", MAX_VOICE_MODEL_CHARS)):
        if not isinstance(voice[key], str) or len(voice[key]) > limit:
            return None, f"voice.{key} must be a string of at most {limit} characters"
    for key in ("latency_ms", "duration_s"):
        v = voice[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            return None, f"voice.{key} must be a non-negative number"
    return {k: voice[k] for k in VOICE_KEYS}, None


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
                    tasks_by_id=tasks_by_id, stop_timeout_s=settings.stop_timeout_s, agent_name=settings.agent)
    agent_root = _agent_run_root(make_cfg, settings.default_options())
    served = ServedModels(ttl_s=settings.served_ttl_s, timeout_s=settings.served_timeout_s)

    def endpoints(options: dict | None = None) -> dict:
        """Where the VLM and Lightning are for these options (default: the console's defaults), from the agent's
        cfg after the console's overrides. Null entries for the mock (it runs no models). Blocking (probes the VLM)."""
        if settings.agent != "real" or make_cfg is None:
            return dict(NO_ENDPOINTS)
        try:
            cfg = make_cfg(options or settings.default_options())
        except Exception as e:  # reported as a 503 when a run starts
            log.warning("endpoints: the agent config failed: %s: %s", type(e).__name__, e)
            return dict(NO_ENDPOINTS)
        return endpoints_for(cfg, served=served, label=settings.vlm_label)

    def log_roots() -> list[Path]:
        """Where the logs page looks for <run_id>/events.jsonl: --runs-dir, then the agent's cfg.run_dir_root."""
        return logs.dedupe_roots([settings.runs_dir, agent_root, *runner.run_roots()])

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await run_in_threadpool(runner.shutdown)

    app = FastAPI(title="GUI assist console", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.settings = settings
    app.state.runner = runner
    app.state.log_roots = log_roots
    app.state.served = served
    app.state.endpoints = endpoints
    app.mount("/static", RevalidatedStaticFiles(directory=settings.static_dir, check_dir=False), name="static")
    app.include_router(logs.make_router(static_dir=settings.static_dir, roots=log_roots,
                                        live=runner.live_agent_run_ids))

    @app.get("/", include_in_schema=False)
    def index():
        page = settings.static_dir / "index.html"
        if page.is_file():
            return FileResponse(page, media_type="text/html", headers={"Cache-Control": "no-store"})
        return HTMLResponse("<p>console/static/index.html is not built yet. The API is up: "
                            "<a href='/api/config'>/api/config</a></p>", status_code=404)

    @app.get("/api/config")
    def config():  # sync: FastAPI runs it in a worker thread (the VLM probe blocks for up to 2 s)
        return {"agent": settings.agent, "sandbox_url": settings.sandbox_url, "headed": settings.headed,
                "defaults": settings.default_options(), "asr": asr.info(), "endpoints": endpoints()}

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
        snapshot = await run_in_threadpool(endpoints, clean["options"])  # recorded in console.json
        unserved = unserved_grounder(snapshot, clean["options"]["grounder"])
        if unserved:
            return _error(400, unserved)
        try:
            run = await run_in_threadpool(runner.start, clean["request"], clean["task_id"], clean["options"],
                                          clean["voice"], snapshot, task_match=clean["task_match"])
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
        if not path.is_file():  # an older run (logs page): look in the run roots
            found = logs.find_run_dir(log_roots(), agent_run_id)
            path = found / filename if found is not None else path
        if not path.is_file():
            return _error(404, "not found")
        return FileResponse(path, media_type="image/png")

    return app


def _agent_run_root(make_cfg, options: dict) -> Path | None:
    """The agent's cfg.run_dir_root for the default options (the real agent writes traces there, not to
    --runs-dir). None if the agent is unavailable or its config has no run_dir_root."""
    if make_cfg is None:
        return None
    try:
        return run_dir_root(make_cfg(options))
    except Exception as e:  # a broken agent config is reported when a run starts; the logs page still works
        log.warning("logs page: could not read the agent's run_dir_root: %s: %s", type(e).__name__, e)
        return None
