"""Picks the agent behind the console: `--agent mock` (console/mock_agent.py) or `--agent real` (T2's agent.core).

get_agent(settings) -> (run_task, make_cfg). Both agents share contracts.md §5:
    async run_task(request, *, start_url, cfg, on_event, parent_run_id=None) -> RunResult
make_cfg(options) turns the console's per-run options {grounder, thinking, lightning, clarify}
into that agent's config object. For the real agent that is agent.config.load_config with
the console's overrides: the per-run options (clarify "on"|"off" -> {"clarify": value}, the
pre-flight request check of agent/clarify.py), the sandbox URL, --headed, and (when given)
--vlm-url -> vlm={"base_url": URL} and --summary-model -> summary_model. A follow-up run also
gets run_task(..., parent_run_id=<the needs_input run it answers>) (console/runner.py). Env vars
(AGENT_VLM_BASE_URL, AGENT_SUMMARY_MODEL, ...) keep working through load_config; an
explicit console flag beats them.

Endpoints (GET /api/config "endpoints", console.json "endpoints", the logs page):
  {"vlm": {base_url, label, summary_model, served, reachable} | null,
   "lightning": {base_url, model, label} | null}
Both are null for the mock (it runs no models). They come from the agent's cfg after the
console's overrides; `served` is the ids from GET <vlm base_url>/models (ServedModels).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

import httpx

RunTask = Callable[..., Awaitable[Any]]
MakeCfg = Callable[[dict], Any]

SERVED_NAMES = ("base", "grounder")  # the VLM's served names (agent/config.py SERVED_NAMES)
VLM_PORT_LABELS = {8001: "H200 via tunnel", 8011: "Mac (vllm-metal)"}
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
HOSTED_LIGHTNING_HOST = "integrate.api.nvidia.com"
NO_ENDPOINTS = {"vlm": None, "lightning": None}


class AgentUnavailable(RuntimeError):
    """The requested agent can't be imported or configured."""


def get_agent(settings) -> tuple[RunTask, MakeCfg]:
    if settings.agent == "mock":
        from console.mock_agent import MockConfig, run_task

        def make_cfg(options: dict) -> MockConfig:
            return MockConfig(runs_dir=str(settings.runs_dir), sandbox_url=settings.sandbox_url,
                              grounder_model=options["grounder"], thinking=options["thinking"],
                              lightning=options["lightning"], clarify=options.get("clarify", "off"),
                              headed=settings.headed, step_delay_s=settings.step_delay_s,
                              scripts_path=str(settings.mock_scripts))

        return run_task, make_cfg

    if settings.agent == "real":
        try:
            from agent.core import run_task
        except ImportError as e:
            raise AgentUnavailable(f"agent.core not available yet (T2); use --agent mock ({e})") from e
        from agent.config import load_config

        def make_cfg(options: dict):
            return load_config(overrides=real_overrides(settings, options))

        return run_task, make_cfg

    raise AgentUnavailable(f"unknown agent {settings.agent!r}; use mock or real")


def real_overrides(settings, options: dict) -> dict:
    """agent.config.load_config overrides for one run of the real agent."""
    vlm_url = getattr(settings, "vlm_url", None)
    summary_model = getattr(settings, "summary_model", None)
    return {
        "grounder_model": options["grounder"],
        "thinking": options["thinking"],
        "lightning": options["lightning"],
        **({"clarify": options["clarify"]} if "clarify" in options else {}),  # "on" | "off" (agent/config.py)
        "sandbox_url": settings.sandbox_url,
        **({"browser": {"headless": False}} if settings.headed else {}),
        **({"vlm": {"base_url": vlm_url}} if vlm_url else {}),
        **({"summary_model": summary_model} if summary_model else {}),
    }


# --- endpoint labels -------------------------------------------------------------------------------


def host_port(url: Any) -> tuple[str, int | None] | None:
    """(host, port) of a URL; the port is the scheme default when the URL has none. None if it isn't a URL."""
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        u = urlsplit(url.strip())
        port = u.port
    except ValueError:
        return None
    if not u.hostname:
        return None
    if port is None:
        port = {"http": 80, "https": 443}.get(u.scheme)
    return u.hostname, port


def host_port_text(url: Any) -> str | None:
    """'127.0.0.1:8001' (IPv6 hosts in brackets), or None."""
    hp = host_port(url)
    if hp is None:
        return None
    host, port = hp
    host = f"[{host}]" if ":" in host else host
    return f"{host}:{port}" if port is not None else host


def vlm_label(base_url: Any, override: str | None = None) -> str | None:
    """--vlm-label / CONSOLE_VLM_LABEL if given; else by port: 8001 'H200 via tunnel', 8011 'Mac (vllm-metal)';
    else '<host:port>'."""
    if isinstance(override, str) and override.strip():
        return override.strip()
    hp = host_port(base_url)
    if hp is None:
        return None
    return VLM_PORT_LABELS.get(hp[1]) or host_port_text(base_url)


def lightning_label(base_url: Any) -> str | None:
    """'local BF16 (:8000)' for 127.0.0.1:8000, 'hosted API' for integrate.api.nvidia.com, else '<host:port>'."""
    hp = host_port(base_url)
    if hp is None:
        return None
    host, port = hp
    if host in LOCAL_HOSTS and port == 8000:
        return "local BF16 (:8000)"
    if host == HOSTED_LIGHTNING_HOST:
        return "hosted API"
    return host_port_text(base_url)


# --- the VLM's served names -----------------------------------------------------------------------


class ServedModels:
    """Served model ids from GET <base_url>/models (2 s timeout), re-checked at most every `ttl_s` per URL.

    get(base_url) -> (served ids | None, reachable). None + False when the server doesn't answer or the reply
    isn't an OpenAI-style model list. Blocking: call it off the event loop."""

    def __init__(self, ttl_s: float = 10.0, timeout_s: float = 2.0):
        self.ttl_s, self.timeout_s = ttl_s, timeout_s
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, list[str] | None, bool]] = {}
        self.n_checks = 0  # tests count the probes with it

    def get(self, base_url: str) -> tuple[list[str] | None, bool]:
        key = base_url.rstrip("/")
        with self._lock:  # one probe at a time; a second caller waits and then reads the fresh entry
            hit = self._cache.get(key)
            if hit is not None and time.monotonic() - hit[0] < self.ttl_s:
                return (list(hit[1]) if hit[1] is not None else None), hit[2]
            served, reachable = self._probe(key)
            self.n_checks += 1
            self._cache[key] = (time.monotonic(), served, reachable)
            return (list(served) if served is not None else None), reachable

    def _probe(self, base_url: str) -> tuple[list[str] | None, bool]:
        try:
            r = httpx.get(f"{base_url}/models", timeout=self.timeout_s, trust_env=False)
            r.raise_for_status()
            data = r.json().get("data")
            if not isinstance(data, list):
                return None, False
            return [m["id"] for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)], True
        except (httpx.HTTPError, ValueError, AttributeError):
            return None, False


# --- endpoints of one agent config -----------------------------------------------------------------


def _get(obj: Any, *path: str) -> Any:
    """obj.a.b or obj["a"]["b"]; None when any step is missing."""
    for key in path:
        if obj is None:
            return None
        obj = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return obj


def _text(v: Any) -> str | None:
    return v.strip() if isinstance(v, str) and v.strip() else None


def endpoints_for(cfg: Any, *, served: ServedModels | None = None, label: str | None = None) -> dict:
    """{"vlm", "lightning"} for an agent config (AgentConfig, or anything with the same fields). An entry the
    config doesn't describe is None. `served` probes the VLM (None: served null, reachable false)."""
    out = dict(NO_ENDPOINTS)
    vlm_url = _text(_get(cfg, "vlm", "base_url"))
    if vlm_url:
        ids, reachable = served.get(vlm_url) if served is not None else (None, False)
        summary_model = _text(_get(cfg, "summary_model"))
        out["vlm"] = {"base_url": vlm_url, "label": vlm_label(vlm_url, label),
                      "summary_model": summary_model, "served": ids, "reachable": reachable}
    l_url = _text(_get(cfg, "lightning", "base_url"))
    if l_url:
        out["lightning"] = {"base_url": l_url, "model": _text(_get(cfg, "lightning", "model")),
                            "label": lightning_label(l_url)}
    return out


def unserved_grounder(endpoints: dict | None, grounder: str) -> str | None:
    """The POST /api/run 400 message when the VLM's served names are known and don't include `grounder`."""
    vlm = (endpoints or {}).get("vlm")
    if not vlm or vlm.get("served") is None or grounder in vlm["served"]:
        return None
    serves = ", ".join(vlm["served"]) or "nothing"
    return f"grounder model '{grounder}' is not served at {vlm['base_url']} (serves: {serves})"


def endpoint_warnings(endpoints: dict | None, grounder: str) -> list[str]:
    """Startup warnings: the VLM doesn't answer, or the default grounder / the summary model isn't served there."""
    vlm = (endpoints or {}).get("vlm")
    if not vlm:
        return []
    if not vlm.get("reachable"):
        return [f"warning: VLM not reachable at {vlm['base_url']} ({vlm['label']}); real runs fail until it answers"]
    out = []
    served = vlm.get("served") or []
    for role, name in (("default grounder", grounder), ("summary model", vlm.get("summary_model"))):
        if name and name not in served:
            out.append(f"warning: {role} '{name}' is not served at {vlm['base_url']} "
                       f"(serves: {', '.join(served) or 'nothing'})")
    return out


def describe_endpoints(endpoints: dict | None) -> str:
    """One startup line, e.g. 'VLM H200 via tunnel (127.0.0.1:8001, summary base, serves base, grounder) ·
    Lightning local BF16 (:8000)'."""
    ep = endpoints or {}
    vlm, lightning = ep.get("vlm"), ep.get("lightning")
    if not vlm and not lightning:
        return "endpoints: none (the mock runs no models)"
    parts = []
    if vlm:
        served = "not reachable" if not vlm.get("reachable") else f"serves {', '.join(vlm.get('served') or []) or 'nothing'}"
        parts.append(f"VLM {vlm['label']} ({host_port_text(vlm['base_url'])}, summary {vlm.get('summary_model')}, {served})")
    if lightning:
        parts.append(f"Lightning {lightning['label']} ({lightning.get('model')})")
    return " · ".join(parts)
