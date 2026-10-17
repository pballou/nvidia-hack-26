# Session bookkeeping and ensure_chromium adapted from NeMo Gym resources_servers/interactive_browser
# (nemo-gym==0.7.0, Gym tag v0.7.0, commit 1f8afc71d45f; app.py last changed 2026-10-02 in bdc018ac9aae,
# browser/_chromium.py last changed 2026-09-22 in 6bbc05ed9d16),
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0. See README.md "Provenance".
"""NeMo Gym resources server `web_agent`: Lightning plans, the VLM grounds, Playwright clicks.

One Gym session = one Playwright context in this process's single Chromium (agent/browser.py)
plus one agent.session.AgentSession. The tool routes (/click, /type_text, /press_key, /scroll,
/done) run AgentSession.execute, the same executor the interactive agent uses, and always
answer 200 + ToolOutcome.tool_payload() ({ok, observation, error?, click?}): the body IS the
model's tool result. verify() reads sandbox state over HTTP and scores it with
sandbox.checks.evaluate. Nothing here reads the DOM: the grounder and the summarizer are the
VLM (agent/vlm.py), never the test-only DOM oracle.

Initial observation: Gym's legacy /run path discards seed_session's response, so
make_tasks.py precomputes the start screen's observation into each row's first user
message. seed_session re-takes the start screenshot, primes the AgentSession from the row
(no VLM call) and records whether the live screen still matches the row (initial_obs_stale).

Nothing in seed_session, verify or the tool routes may raise: one exception (or a non-2xx
from seed/verify) aborts the whole `gym eval run`. Failures are recorded and masked at verify.
"""

import sys
from pathlib import Path

# First: importing nemo_gym rewrites sys.path with $NEMO_GYM_EXTRA_ROOTS in front, which may name
# another checkout of this repo (SETUP.md exports it). The repo root goes in front after that.
import nemo_gym  # noqa: F401

# Gym runs this file as `<venv>/bin/python app.py` with cwd = this dir and PYTHONPATH = this dir
# only (`gym env test` also adds the repo root), so agent/, common/ and sandbox/ are not
# importable on their own. A repo-root pyproject.toml would flip Gym's venv builder into
# editable mode for every server in this repo, so put the repo root on sys.path instead,
# first, so this server always imports its own checkout's agent/, common/ and sandbox/.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path[:] = [_REPO_ROOT, *(p for p in sys.path if p != _REPO_ROOT)]

import asyncio  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union  # noqa: E402

import httpx  # noqa: E402
from fastapi import Body, FastAPI, Request, Response  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from pydantic import ConfigDict, Field, ValidationError  # noqa: E402

from nemo_gym.base_resources_server import (  # noqa: E402
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    SimpleResourcesServer,
)
from nemo_gym.episode_types import EpisodeId, TaskId  # noqa: E402
from nemo_gym.failure_kinds import PROVIDER_UNAVAILABLE, SESSION_LOST, VERIFIER_ERROR  # noqa: E402
from nemo_gym.server_utils import SESSION_ID_KEY  # noqa: E402

from agent.browser import close_pool, get_pool  # noqa: E402
from agent.config import AgentConfig, load_config  # noqa: E402
from agent.observation import Observation, build_observation  # noqa: E402
from agent.prompts import PROMPT_VERSION, TOOL_NAMES  # noqa: E402
from agent.session import ALREADY_DONE, AgentSession, Located, Summary  # noqa: E402
from agent.vlm import VLMClient  # noqa: E402
from sandbox.checks import evaluate  # noqa: E402

# One definition of the start URL and the start-screen fingerprint, shared with the row builder.
try:  # package import (tests import resources_servers.web_agent.app)
    from .make_tasks import screen_sha, start_url
except ImportError:  # `python app.py` under gym env start: this dir is the cwd and on PYTHONPATH
    from make_tasks import screen_sha, start_url

LOG = logging.getLogger(__name__)

BAD_ROW = "web_agent:bad_row"              # row lacks task_id / start_path / check
RESET_REJECTED = "web_agent:reset_rejected"  # sandbox answered /api/reset with a 4xx (unknown task_id, bad sid)
BAD_REQUEST = "web_agent:bad_request"      # /verify body is not a verify request from simple_agent

# Stand-in summary when a row carries no initial_observation (the model saw the start screen in
# its first user message; this text only shows up again if the first call fails validation).
NO_INITIAL_SUMMARY = "(the start screen is described in the first user message)"
_SID_UNSAFE = re.compile(r"[^A-Za-z0-9_.:-]")  # sandbox/app.py SID_RE is ^[A-Za-z0-9_.:-]{1,128}$
_CLOSE_TIMEOUT_S = 10.0
# ToolOutcome.timings_ms stages; per-episode totals become <stage>_ms VerifyResponse fields.
STAGES = ("summary", "ground", "act", "settle", "screenshot")


def is_vlm_infra_error(error: Optional[str]) -> bool:
    """True for agent/vlm.py transport, timeout and HTTP errors ('VLM unreachable ...', 'VLM timed out ...',
    'VLM error <code> ...', 'VLM call failed ...', '<model> returned no choices'). Model-quality failures
    ('could not parse a point ...', '... returned an empty screen description') are not infrastructure."""
    return bool(error) and (error.startswith("VLM ") or error.endswith("returned no choices"))


# Playwright errors (as agent.session._brief renders them) for a dead or hung browser, context or page:
# 'TargetClosedError: Mouse.click: Target page, context or browser has been closed',
# 'Error: Page.screenshot: Target crashed', 'TimeoutError: Page.screenshot: Timeout 15000ms exceeded.'
# (in a tool call only screenshots have a Playwright timeout; mouse and keyboard calls have none).
# Anchored to that '<ExcType>: <Class.method>: ...' shape: other errors embed model text (the target).
_BROWSER_DEAD = re.compile(r"^TargetClosedError: |^\w+: [A-Z]\w*\.\w+: (?:.*(?:Target crashed|Page crashed|"
                           r"has been closed|Browser closed|Connection closed)|Timeout \d+(?:\.\d+)?ms exceeded)")
_SCREEN_UNAVAILABLE = "(screen unavailable: "  # AgentSession's summary when even the fresh screenshot raised


def browser_infra_error(outcome: Any, prev_summary: Optional[str] = None) -> Optional[str]:
    """The browser-failure text of a tool outcome, or None. A renderer crash or hung page is the
    environment's fault, so verify masks it like a VLM outage instead of scoring a policy failure.

    `prev_summary` is the screen summary before the call: an outcome that reuses it (a grounding
    miss, ALREADY_DONE) carries no new screen failure."""
    texts = [outcome.error or ""]
    summary = getattr(outcome.observation, "screen_summary", "") or ""
    if summary.startswith(_SCREEN_UNAVAILABLE) and summary != prev_summary:  # built from an exception
        texts.append(summary[len(_SCREEN_UNAVAILABLE):].rstrip(")"))
    return next((t for t in texts if t and _BROWSER_DEAD.search(t)), None)


# --- seams: module-level so tests can monkeypatch them; production uses these defaults ---------

async def new_browser_session(cfg: AgentConfig) -> Any:
    """A fresh 1280x800 context in the process-wide Chromium (agent/browser.py)."""
    pool = await get_pool(cfg.browser)
    return await pool.new_session(cfg.viewport, navigation_timeout_s=cfg.timeouts.navigation_s)


def make_vlm(cfg: AgentConfig, browser_session: Any) -> Any:
    """Grounder + summarizer for one episode: the VLM served on cfg.vlm (`base` / `grounder`).

    Never the DOM oracle: only pytest swaps this, to exercise the loop without a GPU.
    """
    return VLMClient(cfg)


def sandbox_client(base_url: str, timeout_s: float) -> httpx.AsyncClient:
    """One client per call, so it is never bound to a dead event loop. No proxies for 127.0.0.1."""
    return httpx.AsyncClient(base_url=base_url, timeout=timeout_s, trust_env=False)


# --- Chromium install check (from interactive_browser/browser/_chromium.py @ v0.7.0) ----------
# The playwright wheel ships the driver but not the browsers. Installing on first use turns a
# launch failure deep inside a rollout into a one-off startup cost. Linux system libraries still
# need `python -m playwright install-deps chromium` (sudo) on a bare box.

_chromium_lock = threading.Lock()
_chromium_ok = False


def _browsers_dir() -> Path:
    override = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if override:
        return Path(override)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ms-playwright"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    return Path.home() / ".cache" / "ms-playwright"


def _chromium_present() -> bool:
    root = _browsers_dir()
    return root.is_dir() and any(
        any(d.rglob(name))
        for d in root.glob("chromium*")
        if d.is_dir()
        for name in ("chrome-headless-shell", "headless_shell", "chrome")
    )


def ensure_chromium() -> None:
    """Install Playwright's Chromium if it is missing; raises if the install fails."""
    global _chromium_ok
    with _chromium_lock:
        if _chromium_ok or _chromium_present():
            _chromium_ok = True
            return
        LOG.info("Playwright Chromium is missing; installing it (one-off)")
        result = subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"],
                                capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"`python -m playwright install chromium` exited {result.returncode}:\n"
                               f"{result.stderr.strip()[-2000:]}")
        _chromium_ok = True


# --- request / response models ------------------------------------------------------------------

class WebAgentConfig(BaseResourcesServerConfig):
    # Unknown keys are kept (not silently dropped) so model_post_init can reject a typo such as
    # `grounder_modle`, which would otherwise keep `base` and quietly turn base-vs-tuned into base-vs-base.
    model_config = ConfigDict(extra="allow")
    verified: bool = False                   # yaml metadata for Gym's env listing
    description: Optional[str] = None
    value: Optional[str] = None
    # The server reads configs/agent.yaml (agent/config.py: yaml < AGENT_* env vars) and applies the
    # non-null fields below as overrides, so base <-> grounder is a Gym YAML / env var change only.
    agent_config_path: Optional[str] = None  # None -> $AGENT_CONFIG or configs/agent.yaml
    grounder_model: Optional[str] = None     # base | grounder
    summary_model: Optional[str] = None      # base | grounder
    model_family: Optional[str] = None       # qwen3_vl | qwen3_5
    vlm_base_url: Optional[str] = None
    sandbox_url: Optional[str] = None
    headless: Optional[bool] = None
    sandbox_timeout_s: float = 10.0
    # Gym's aiohttp client has no timeout, so each tool call is bounded here. A slow but legitimate call
    # is ~190 s (ground + summary, each a 45 s VLM timeout plus one retry); a timeout masks the rollout.
    tool_timeout_s: float = 240.0
    # A /run that fails before verify (e.g. a model-call error) leaks its browser context: each seed
    # closes episodes older than this.
    episode_ttl_s: float = 3600.0


class WebAgentSeedSessionRequest(BaseSeedSessionRequest):
    """Legacy seed body: the whole task row plus Gym's own keys. Every field is lenient on purpose:
    a 422 here aborts the whole `gym eval run`, so bad rows are recorded and masked at verify."""
    model_config = ConfigDict(extra="allow")
    responses_create_params: Any = None
    task_id: Any = None
    start_path: Any = None
    check: Any = None
    tags: Any = None
    initial_observation: Any = None
    initial_screen_sha: Any = None
    initial_observation_source: Any = None
    prompt_version: Any = None


class WebAgentVerifyRequest(BaseVerifyRequest):
    # Optional everywhere (even the base class's required fields): a 422 on /verify aborts the run.
    responses_create_params: Any = None
    response: Any = None
    task_id: Any = None
    tags: Any = None
    check: Any = None
    initial_observation_source: Any = None
    prompt_version: Any = None


class WebAgentVerifyResponse(BaseVerifyResponse):
    """Declared fields are the only ones that reach rollouts and /aggregate_metrics; ints and
    bools there are auto-profiled (mean/steps, mean/claimed_success, ...)."""
    responses_create_params: Any = None
    response: Any = None
    task_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    steps: int = 0                    # tool calls that reached this server (valid or not)
    n_clicks: int = 0                 # pixel clicks (click, and type_text with a target)
    n_invalid_calls: int = 0          # rejected by the tool schema: unknown tool, wrong/missing args, bad enum
    n_failed_actions: int = 0         # valid calls that returned ok=false (grounding or browser failure)
    n_calls_after_done: int = 0       # valid calls refused because done was already called
    done_called: bool = False
    claimed_success: bool = False     # done(success=true)
    n_vlm_infra_errors: int = 0       # VLM transport/timeout/HTTP errors, browser crashes, tool timeouts (> 0 -> masked)
    initial_obs_stale: Optional[bool] = None  # live start screen != the row's initial_screen_sha
    initial_observation_source: Optional[str] = None  # 'vlm:<summary_model>' | 'placeholder' (from the row)
    state_source: Optional[str] = None  # done_snapshot | live
    check_explanation: Optional[str] = None
    grounder_model: Optional[str] = None
    summary_model: Optional[str] = None
    prompt_version: Optional[str] = None
    # Per-episode stage totals (ms) from ToolOutcome.timings_ms; Gym profiles them (mean/ground_ms, ...).
    summary_ms: float = 0.0
    ground_ms: float = 0.0
    act_ms: float = 0.0
    settle_ms: float = 0.0
    screenshot_ms: float = 0.0
    # Per-call values {stage: [ms, ...]} for p50/p90 in scripts/agent_metrics.py (Gym skips dicts).
    latency_ms: Dict[str, List[float]] = Field(default_factory=dict)


class _CountingVLM:
    """Per-episode grounder + summarizer proxy that records VLM infrastructure errors, so verify masks
    an outage instead of scoring it as a policy failure (which would bias base vs grounder)."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.infra_errors: List[str] = []

    async def locate(self, img: Any, target: str) -> Located:
        located = await self.inner.locate(img, target)
        self._note(located.error)
        return located

    async def summarize(self, img: Any) -> Summary:
        summary = await self.inner.summarize(img)
        self._note(summary.error)
        return summary

    def _note(self, error: Optional[str]) -> None:
        if is_vlm_infra_error(error):
            self.infra_errors.append(error[:300])

    async def aclose(self) -> None:
        close = getattr(self.inner, "aclose", None) or getattr(self.inner, "close", None)
        if close is not None:
            await close()


@dataclasses.dataclass
class _Episode:
    sid: str                                  # sandbox sid ('gym-' + Gym session id)
    task_id: Optional[str] = None
    check: Any = None
    tags: List[str] = dataclasses.field(default_factory=list)
    initial_observation_source: Optional[str] = None
    prompt_version: Optional[str] = None
    browser: Any = None
    vlm: Optional[_CountingVLM] = None
    agent: Optional[AgentSession] = None
    seed_error: Optional[str] = None
    seed_kind: Optional[str] = None
    initial_obs_stale: Optional[bool] = None
    done_state: Optional[dict] = None         # sandbox state at the first done ("first finish wins")
    n_calls: int = 0
    n_invalid: int = 0
    n_failed_actions: int = 0
    n_after_done: int = 0
    infra_errors: List[str] = dataclasses.field(default_factory=list)  # timeouts, browser crashes (+ VLM at release)
    timings: Dict[str, List[float]] = dataclasses.field(default_factory=dict)  # {stage: [ms per call]}
    created: float = dataclasses.field(default_factory=time.monotonic)
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)


class _BadRow(ValueError):
    pass


# --- server -------------------------------------------------------------------------------------

class WebAgentResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: WebAgentConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        # Browsers and episodes live in this process; a second worker would get tool calls for
        # browsers it does not hold.
        if self.config.num_workers not in (None, 1):
            raise ValueError("web_agent keeps browsers in-process and requires num_workers=1")
        unknown = sorted(self.config.model_extra or {})
        if unknown:
            raise ValueError(f"unknown web_agent config key(s) {unknown}; the known ones are WebAgentConfig's")
        self._cfg = self._agent_config()  # config errors fail at startup, not inside a rollout
        if self._cfg.use_oracle_grounder:
            raise ValueError("the DOM oracle grounder is test-only and never runs in the Gym env")
        self._episodes: Dict[str, _Episode] = {}
        # Sessions an Environment Server seeded under its own resources_session_id.
        self._typed_identity: Dict[str, Tuple[EpisodeId, TaskId]] = {}
        self._typed_locks: Dict[str, asyncio.Lock] = {}
        # Closed ids stay known, so a seed that lands after its close is rejected.
        self._typed_closed: Dict[str, EpisodeId] = {}

    def _agent_config(self) -> AgentConfig:
        c = self.config
        overrides: Dict[str, Any] = {k: getattr(c, k) for k in ("grounder_model", "summary_model", "model_family",
                                                                 "sandbox_url") if getattr(c, k) is not None}
        if c.vlm_base_url is not None:
            overrides["vlm"] = {"base_url": c.vlm_base_url}
        if c.headless is not None:
            overrides["browser"] = {"headless": c.headless}
        return load_config(c.agent_config_path, overrides=overrides)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()  # /seed_session /close_session /verify /aggregate_metrics
        for name in TOOL_NAMES:
            app.post(f"/{name}")(self._tool_route(name))
        # Last: any other POST path is a tool the model invented -> 200 + corrective message
        # (instead of FastAPI's 404 body). Built-in routes registered above still match first.
        app.post("/{tool_name}")(self.unknown_tool)

        inner = app.router.lifespan_context  # wrapped the same way as server_utils.setup_profiling

        @asynccontextmanager
        async def lifespan(a: FastAPI):
            await asyncio.to_thread(ensure_chromium)
            async with inner(a) as state:
                try:
                    yield state
                finally:
                    await self._shutdown()

        app.router.lifespan_context = lifespan
        return app

    async def _shutdown(self) -> None:
        for session_id in list(self._episodes):
            await self._release(session_id)
        try:  # the one Chromium of this process (started lazily by the first seed)
            await asyncio.wait_for(close_pool(), timeout=_CLOSE_TIMEOUT_S)
        except Exception as exc:  # usually Playwright's driver got the same SIGINT and exited first
            LOG.warning("could not close the browser pool: %s", _err(exc))

    # ----- lifecycle ---------------------------------------------------------------------------

    async def seed_session(
        self,
        request: Request,
        # Typed first, left to right: the legacy model allows extras and would swallow a typed body.
        body: Annotated[Union[ResourcesSeedSessionRequest, WebAgentSeedSessionRequest],
                        Field(union_mode="left_to_right")],
    ) -> Union[ResourcesSeedSessionResponse, BaseSeedSessionResponse]:
        await self._sweep()
        if not isinstance(body, ResourcesSeedSessionRequest):
            session_id = request.session[SESSION_ID_KEY]
            # Every /run seed carries the whole row and arrives without a cookie (Gym's DummyCookieJar),
            # so it always gets a fresh session id. A body without a row, or a seed for a session that
            # already has an episode, is the model calling a tool named seed_session: leave it alone.
            if body.responses_create_params is None or session_id in self._episodes:
                LOG.warning("ignored a seed_session that is not a fresh /run seed")
                return BaseSeedSessionResponse()
            await self._open(session_id, body)
            return BaseSeedSessionResponse()

        session_id = body.resources_session_id
        request.session[SESSION_ID_KEY] = session_id  # bind later tool/verify calls to this id
        async with self._typed_locks.setdefault(session_id, asyncio.Lock()):
            if session_id in self._typed_closed:
                raise ValueError(f"resources session is already closed: {session_id}")
            identity = self._typed_identity.get(session_id)
            if identity is not None:
                if identity != (body.episode_id, body.task_id):
                    raise ValueError("resources_session_id is already bound to another episode or task")
                return ResourcesSeedSessionResponse(resources_session_id=session_id)
            await self._open(session_id, WebAgentSeedSessionRequest.model_validate(body.task_data))
            self._typed_identity[session_id] = (body.episode_id, body.task_id)
            return ResourcesSeedSessionResponse(resources_session_id=session_id)

    async def _open(self, session_id: str, row: WebAgentSeedSessionRequest) -> None:
        """Reset the sandbox, open a context on the start page and prime the AgentSession. Never raises."""
        await self._release(session_id)  # a re-seed of the same id
        ep = _Episode(sid=("gym-" + _SID_UNSAFE.sub("_", session_id))[:128])
        self._episodes[session_id] = ep  # registered first, so verify always finds (and masks) it
        try:
            if not isinstance(row.task_id, str) or not row.task_id:
                raise _BadRow("row has no task_id")
            if not isinstance(row.start_path, str) or not row.start_path.startswith("/"):
                raise _BadRow(f"row {row.task_id} has no start_path starting with '/'")
            if not isinstance(row.check, dict):
                raise _BadRow(f"row {row.task_id} has no check object")
            ep.task_id, ep.check, ep.tags = row.task_id, row.check, _tags(row.tags)
            ep.initial_observation_source = _str_or_none(row.initial_observation_source)
            ep.prompt_version = _str_or_none(row.prompt_version)
            if ep.prompt_version not in (None, PROMPT_VERSION):
                LOG.warning("row %s was made with prompt %s; the server runs %s (regenerate the rows)",
                            ep.task_id, ep.prompt_version, PROMPT_VERSION)

            await self._sandbox_post("/api/reset", {"sid": ep.sid, "task_id": ep.task_id})
            ep.browser = await new_browser_session(self._cfg)
            await ep.browser.goto(start_url(self._cfg.sandbox_url, row.start_path, ep.sid))
            ep.vlm = _CountingVLM(make_vlm(self._cfg, ep.browser))
            agent = AgentSession(ep.browser, ep.vlm, ep.vlm)
            # Prime from the row instead of AgentSession.reset(): no VLM call, because the model
            # already got the start screen in its first user message. The first click grounds
            # on this live screenshot. Settle again first, like reset() after goto() (make_tasks.py
            # and the interactive agent), so this hashes the same frame as the row's initial_screen_sha.
            await ep.browser.settle()
            img = await ep.browser.screenshot()
            agent.prime(img, _row_observation(row.initial_observation)
                        or build_observation(step=0, url=await ep.browser.url(), title=await ep.browser.title(),
                                             screen_changed=None, last_action_result="",
                                             screen_summary=NO_INITIAL_SUMMARY))
            if isinstance(row.initial_screen_sha, str) and row.initial_screen_sha:
                ep.initial_obs_stale = screen_sha(img) != row.initial_screen_sha
                if ep.initial_obs_stale:
                    LOG.warning("start screen of %s differs from the row's initial_screen_sha", ep.task_id)
            ep.agent = agent
        except Exception as exc:
            ep.seed_error = _err(exc)
            ep.seed_kind = (BAD_ROW if isinstance(exc, _BadRow)
                            else RESET_REJECTED if isinstance(exc, httpx.HTTPStatusError)
                            and exc.response.status_code < 500 else PROVIDER_UNAVAILABLE)
            ep.agent = None
            LOG.warning("web_agent seed failed for %s: %s", ep.task_id or session_id, ep.seed_error)
            await self._close_resources(ep)

    async def close_resources_session(
        self, request: Request, body: Annotated[Optional[Dict[str, Any]], Body()] = None,
    ) -> Union[ResourcesCloseSessionResponse, Dict[str, Any]]:
        if not body:
            # Unlike interactive_browser, an empty body closes nothing: the legacy /run path never
            # calls close (verify releases), and a model calling close_session({}) must not kill
            # its own browser mid-episode.
            return {"ok": False, "error": "close_session needs a ResourcesCloseSessionRequest body; nothing closed"}
        try:
            typed = ResourcesCloseSessionRequest.model_validate(body)
        except ValidationError as error:
            raise RequestValidationError(error.errors()) from error
        session_id = typed.resources_session_id
        async with self._typed_locks.setdefault(session_id, asyncio.Lock()):
            closed_episode = self._typed_closed.get(session_id)
            if closed_episode is not None:
                if typed.episode_id != closed_episode:
                    raise ValueError("episode_id does not match the closed resources session")
                request.session.pop(SESSION_ID_KEY, None)
                return ResourcesCloseSessionResponse(resources_session_id=session_id)
            identity = self._typed_identity.get(session_id)
            if identity is not None and typed.episode_id != identity[0]:
                raise ValueError("episode_id does not match the seeded resources session")
            await self._release(session_id)  # usually verify already released it
            self._typed_identity.pop(session_id, None)
            self._typed_closed[session_id] = typed.episode_id
            request.session.pop(SESSION_ID_KEY, None)
            return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def _release(self, session_id: Optional[str]) -> None:
        ep = self._episodes.pop(session_id, None) if session_id is not None else None
        if ep is not None:
            await self._close_resources(ep)

    async def _sweep(self) -> None:
        """Close episodes whose /run never reached verify (a later verify masks them as session_lost)."""
        cutoff = time.monotonic() - self.config.episode_ttl_s
        for session_id in [k for k, ep in self._episodes.items() if ep.created < cutoff]:
            LOG.warning("closing web_agent episode %s: no verify within %g s", session_id, self.config.episode_ttl_s)
            await self._release(session_id)

    @staticmethod
    async def _close_resources(ep: _Episode) -> None:
        if ep.vlm is not None:  # verify still needs the counts once the client is gone
            ep.infra_errors.extend(ep.vlm.infra_errors)
        browser, vlm, ep.browser, ep.vlm = ep.browser, ep.vlm, None, None
        for obj in (browser, vlm):
            close = getattr(obj, "aclose", None) or getattr(obj, "close", None)
            if close is None:
                continue
            try:
                await asyncio.wait_for(close(), timeout=_CLOSE_TIMEOUT_S)
            except Exception:
                LOG.warning("could not close %s for %s", type(obj).__name__, ep.sid, exc_info=True)

    # ----- tools: always 200 + {ok, observation, error?, click?} ------------------------------

    def _tool_route(self, name: str):
        async def route(request: Request) -> Response:
            return await self._run_tool(request, name)

        route.__name__ = f"tool_{name}"  # unique OpenAPI operation ids
        return route

    async def unknown_tool(self, request: Request, tool_name: str) -> Response:
        # AgentSession answers an unknown name with the corrective "the only tools are: ..." message.
        return await self._run_tool(request, tool_name)

    async def _run_tool(self, request: Request, name: str) -> Response:
        # The loose body (never a typed pydantic model) means the model never sees a raw 422: the
        # schema check is agent.session.validate_args, with a corrective message on a mismatch.
        ep = None
        try:
            ep = self._episodes.get(request.session.get(SESSION_ID_KEY))
            if ep is None:
                return _tool_error("no active environment session (not seeded, or already verified)")
            ep.n_calls += 1
            if ep.agent is None:
                return _tool_error(f"the environment failed to start: {ep.seed_error}")
            args = await _json_body(request)
            async with ep.lock:
                prev = ep.agent.last_observation
                try:
                    outcome = await asyncio.wait_for(ep.agent.execute(name, args), self.config.tool_timeout_s)
                except asyncio.TimeoutError:
                    ep.n_failed_actions += 1
                    ep.infra_errors.append(f"{name} timed out after {self.config.tool_timeout_s:g}s")
                    last = ep.agent.last_observation
                    return _tool_error(f"{name} timed out after {self.config.tool_timeout_s:g}s; the screen may "
                                       "have changed", last.render() if last else "")
                for stage, ms in outcome.timings_ms.items():
                    ep.timings.setdefault(stage, []).append(round(ms, 1))
                invalid = outcome.invalid
                if invalid:
                    ep.n_invalid += 1
                elif outcome.error == ALREADY_DONE:
                    ep.n_after_done += 1
                elif not outcome.ok:
                    ep.n_failed_actions += 1
                dead = None if invalid else browser_infra_error(outcome, prev.screen_summary if prev else None)
                if dead:
                    ep.infra_errors.append(f"{name}: browser failure: {dead[:300]}")
                if outcome.done:  # the first done: freeze what verify will score
                    await self._snapshot_done(ep)
            return _payload_response(outcome.tool_payload())
        except Exception as exc:  # an AgentSession bug or a dead browser must not become a 500
            LOG.exception("tool %s crashed", name)
            if ep is not None:
                ep.n_failed_actions += 1
            last = ep.agent.last_observation if ep is not None and ep.agent is not None else None
            return _tool_error(f"{name} failed: {_err(exc)}", last.render() if last else "")

    async def _snapshot_done(self, ep: _Episode) -> None:
        try:
            ep.done_state = await self._sandbox_state(ep.sid)
        except Exception as exc:  # verify falls back to the live state (actions after done are refused)
            LOG.warning("could not snapshot sandbox state at done for %s: %s", ep.sid, _err(exc))

    # ----- reward --------------------------------------------------------------------------------

    async def verify(self, request: Request) -> WebAgentVerifyResponse:
        # The body is parsed here rather than by FastAPI so that no body, not even invalid JSON, can
        # turn into a 422 that aborts the run.
        session_id = request.session.get(SESSION_ID_KEY)
        ep = self._episodes.get(session_id)
        body, raw = WebAgentVerifyRequest(), None
        try:
            raw = await _json_body(request)
            body = WebAgentVerifyRequest.model_validate(raw)
        except Exception:
            pass
        if not _is_agent_verify(raw, ep):
            # Garbage, or the model calling a tool named verify (its arguments are the body, on the
            # episode's cookie). Score nothing, show no state and leave the episode running.
            return self._response(body, ep, mask=True, kind=BAD_REQUEST,
                                  reason="not simple_agent's verify request (needs response and responses_create_params "
                                         "objects and the episode's task_id); nothing scored")
        try:
            return await self._score(body, ep)
        except httpx.HTTPError as exc:
            return self._response(body, ep, mask=True, kind=PROVIDER_UNAVAILABLE,
                                  reason=f"sandbox unreachable at verify: {_err(exc)}")
        except Exception as exc:  # judge_failsafe only converts JudgeError; anything else aborts the run
            LOG.exception("verify failed")
            return self._response(body, ep, mask=True, kind=VERIFIER_ERROR, reason=_err(exc))
        finally:
            await self._release(session_id)  # the legacy /run path never calls /close_session

    async def _score(self, body: WebAgentVerifyRequest, ep: Optional[_Episode]) -> WebAgentVerifyResponse:
        if ep is None:
            return self._response(body, None, mask=True, kind=SESSION_LOST,
                                  reason="no web_agent session at verify (never seeded, already verified, "
                                         "or the server restarted)")
        if ep.seed_error is not None:
            return self._response(body, ep, mask=True, kind=ep.seed_kind, reason=f"seed failed: {ep.seed_error}")
        infra = _infra_errors(ep)
        if infra:
            # Masked whatever the reward: masking only failures would inflate success during an outage.
            return self._response(body, ep, mask=True, kind=PROVIDER_UNAVAILABLE,
                                  reason=f"{len(infra)} VLM/browser/tool infrastructure error(s); first: {infra[0]}")
        state, source = (ep.done_state, "done_snapshot") if ep.done_state is not None else (
            await self._sandbox_state(ep.sid), "live")
        # The sandbox recreates a default state for an unknown sid (restart, LRU eviction), which
        # would score a false 0. The real sandbox records meta.sid / meta.task_id at /api/reset;
        # a sandbox that doesn't expose them (agent/testing/fake_sandbox.py) skips this check.
        meta = state.get("meta") if isinstance(state.get("meta"), dict) else {}
        for key, want in (("sid", ep.sid), ("task_id", ep.task_id)):
            if key in meta and meta[key] != want:
                return self._response(body, ep, mask=True, kind=SESSION_LOST, state_source=source,
                                      reason=f"sandbox state has meta.{key}={meta[key]!r}, expected {want!r} "
                                             "(the sandbox restarted or evicted this session)")
        passed, explanation = evaluate(state, ep.check)
        if "task_id" not in meta:
            explanation += " (session check skipped: sandbox state has no meta.task_id)"
        return self._response(body, ep, reward=1.0 if passed else 0.0, explanation=explanation,
                              state_source=source)

    def _response(self, body: WebAgentVerifyRequest, ep: Optional[_Episode], *, reward: float = 0.0,
                  mask: bool = False, kind: Optional[str] = None, reason: Optional[str] = None,
                  explanation: Optional[str] = None, state_source: Optional[str] = None) -> WebAgentVerifyResponse:
        agent = ep.agent if ep is not None else None
        finished = agent.finished if agent is not None else None
        timings = ep.timings if ep is not None else {}
        totals = {f"{stage}_ms": round(sum(timings.get(stage, [])), 1) for stage in STAGES}
        return WebAgentVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reward=reward,
            mask_sample=mask,
            failure_kind=kind,
            failure_reason=reason,
            task_id=(ep.task_id if ep is not None and ep.task_id else None) or _str_or_none(body.task_id),
            tags=(ep.tags if ep is not None and ep.tags else None) or _tags(body.tags),
            steps=ep.n_calls if ep is not None else 0,
            n_clicks=agent.n_clicks if agent is not None else 0,
            n_invalid_calls=ep.n_invalid if ep is not None else 0,
            n_failed_actions=ep.n_failed_actions if ep is not None else 0,
            n_calls_after_done=ep.n_after_done if ep is not None else 0,
            n_vlm_infra_errors=len(_infra_errors(ep)) if ep is not None else 0,
            done_called=finished is not None,
            claimed_success=bool(finished and finished.get("success")),
            initial_obs_stale=ep.initial_obs_stale if ep is not None else None,
            initial_observation_source=(ep.initial_observation_source if ep is not None else None)
            or _str_or_none(body.initial_observation_source),
            state_source=state_source,
            check_explanation=explanation,
            grounder_model=self._cfg.grounder_model,
            summary_model=self._cfg.summary_model,
            prompt_version=(ep.prompt_version if ep is not None else None) or _str_or_none(body.prompt_version),
            latency_ms={stage: list(v) for stage, v in timings.items()},
            **totals,
        )

    # ----- sandbox HTTP (verifier-side state access; the agent never sees it) -------------------

    async def _sandbox_post(self, path: str, payload: dict) -> dict:
        async with sandbox_client(self._cfg.sandbox_url, self.config.sandbox_timeout_s) as client:
            resp = await client.post(path, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or not data.get("ok", False):
            raise RuntimeError(f"sandbox {path} refused: {str(data)[:200]}")
        return data

    async def _sandbox_state(self, sid: str) -> dict:
        async with sandbox_client(self._cfg.sandbox_url, self.config.sandbox_timeout_s) as client:
            resp = await client.get("/api/state", params={"sid": sid})
        resp.raise_for_status()
        state = resp.json()
        if not isinstance(state, dict):
            raise RuntimeError(f"sandbox /api/state returned {type(state).__name__}, not an object")
        return state

    # ----- metrics (masked rollouts are already excluded by Gym) --------------------------------

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Mean reward per task tag (easy / medium / hard, icon_only, ...); `tags` is a list, so
        Gym's scalar subset metrics can't slice on it."""
        rewards: Dict[str, List[float]] = {}
        for rollouts in tasks:
            for r in rollouts:
                for tag in r.get("tags") or []:
                    rewards.setdefault(str(tag), []).append(float(r.get("reward") or 0.0))
        out: Dict[str, Any] = {}
        for tag, values in sorted(rewards.items()):
            out[f"tag/{tag}/mean_reward"] = sum(values) / len(values)
            out[f"tag/{tag}/rollouts"] = len(values)
        return out

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        keys = super().get_key_metrics(agent_metrics)
        keys.update({k: v for k, v in agent_metrics.items() if k.startswith("tag/") and k.endswith("/mean_reward")})
        return keys


# --- helpers --------------------------------------------------------------------------------------

def _payload_response(payload: dict) -> Response:
    """The tool result in the same bytes the interactive planner sends (agent/planner.py:
    json.dumps(payload, ensure_ascii=False)); FastAPI's JSONResponse would use compact separators."""
    return Response(json.dumps(payload, ensure_ascii=False), media_type="application/json")


def _tool_error(error: str, observation: str = "") -> Response:
    return _payload_response({"ok": False, "observation": observation, "error": error})


def _is_agent_verify(raw: Any, ep: Optional[_Episode]) -> bool:
    """simple_agent's verify body is the whole row plus the model response (`body.model_dump() |
    {response}`), so it always carries both objects and the row's task_id, which the model never sees."""
    return (isinstance(raw, dict) and isinstance(raw.get("response"), dict)
            and isinstance(raw.get("responses_create_params"), dict)
            and (ep is None or ep.task_id is None or raw.get("task_id") == ep.task_id))


def _infra_errors(ep: _Episode) -> List[str]:
    return ep.infra_errors + (ep.vlm.infra_errors if ep.vlm is not None else [])


async def _json_body(request: Request) -> Any:
    """The parsed JSON body, None if empty, or the raw text if it isn't JSON (validation rejects it)."""
    raw = await request.body()
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode("utf-8", "replace")[:200]


def _row_observation(value: Any) -> Optional[Observation]:
    """The row's precomputed start observation (Observation.to_dict()), or None if absent/malformed."""
    if not isinstance(value, dict):
        return None
    try:
        return Observation(**{f.name: value[f.name] for f in dataclasses.fields(Observation)})
    except (KeyError, TypeError):
        return None


def _tags(value: Any) -> List[str]:
    return [str(t) for t in value] if isinstance(value, list) else []


def _str_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


if __name__ == "__main__":
    WebAgentResourcesServer.run_webserver()
