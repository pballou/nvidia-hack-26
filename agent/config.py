"""Agent config: configs/agent.yaml (docs/tasks/contracts.md §4) plus env-var overrides.

Precedence: explicit `overrides` > env vars > yaml > dataclass defaults.

Env vars (all optional):
  AGENT_CONFIG              path to the yaml (default: <repo>/configs/agent.yaml)
  AGENT_SANDBOX_URL, AGENT_GROUNDER_MODEL, AGENT_SUMMARY_MODEL, AGENT_MODEL_FAMILY,
  AGENT_MAX_STEPS, AGENT_THINKING, AGENT_RUNS_DIR, AGENT_HEADLESS
  AGENT_LIGHTNING           hosted | local (picks a preset from lightning_presets)
  AGENT_LIGHTNING_BASE_URL, AGENT_LIGHTNING_MODEL, AGENT_LIGHTNING_API_KEY_ENV
  AGENT_VLM_BASE_URL, AGENT_VLM_API_KEY

Secrets never live in the yaml: `lightning.api_key_env` names the env var that holds
the key (NVIDIA_API_KEY for hosted). If it isn't exported, the repo's .env is read.

An explicit config path (argument or AGENT_CONFIG) must exist; only the implicit default
may be missing, and then the dataclass defaults apply.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agent.prompts import PROMPT_VERSION
from common.grounding import FAMILIES

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "agent.yaml"

SERVED_NAMES = ("base", "grounder")
THINKING_MODES = ("off", "on", "retry")

DEFAULT_PRESETS = {
    "hosted": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "api_key_env": "NVIDIA_API_KEY",
    },
    "local": {"base_url": "http://127.0.0.1:8000/v1", "model": "lightning", "api_key_env": None},
}


@dataclass(frozen=True)
class LightningConfig:
    base_url: str = DEFAULT_PRESETS["hosted"]["base_url"]
    model: str = DEFAULT_PRESETS["hosted"]["model"]
    api_key_env: str | None = "NVIDIA_API_KEY"

    @property
    def api_key(self) -> str:
        """The key from the environment (or .env); "EMPTY" for keyless local servers."""
        if not self.api_key_env:
            return "EMPTY"
        key = os.environ.get(self.api_key_env) or _read_dotenv().get(self.api_key_env)
        if not key:
            raise RuntimeError(f"{self.api_key_env} is not set (export it or add it to .env)")
        return key

    def __repr__(self) -> str:  # never show the key
        return f"LightningConfig(base_url={self.base_url!r}, model={self.model!r}, api_key_env={self.api_key_env!r})"


@dataclass(frozen=True)
class VLMConfig:
    base_url: str = "http://127.0.0.1:8001/v1"
    api_key: str = "EMPTY"


@dataclass(frozen=True)
class Viewport:
    width: int = 1280
    height: int = 800


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 1024
    # vLLM counts reasoning tokens against max_tokens, so thinking calls get their own cap.
    thinking_max_tokens: int = 4096


@dataclass(frozen=True)
class Timeouts:
    lightning_s: float = 180.0  # hosted Lightning takes 25-80 s per call
    vlm_s: float = 45.0
    navigation_s: float = 15.0


@dataclass(frozen=True)
class BrowserConfig:
    headless: bool = True
    locale: str = "en-US"
    timezone_id: str = "America/New_York"
    settle_ms: int = 350
    network_idle_ms: int = 2500


@dataclass(frozen=True)
class AgentConfig:
    sandbox_url: str = "http://127.0.0.1:8090"
    lightning: LightningConfig = field(default_factory=LightningConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    grounder_model: str = "base"
    summary_model: str = "base"
    model_family: str = "qwen3_vl"
    max_steps: int = 12
    thinking: str = "retry"
    viewport: Viewport = field(default_factory=Viewport)
    runs_dir: str = "runs"
    lightning_presets: dict = field(default_factory=lambda: dict(DEFAULT_PRESETS))
    sampling: Sampling = field(default_factory=Sampling)
    timeouts: Timeouts = field(default_factory=Timeouts)
    history_full_steps: int = 6
    # Interactive planner only: "required" (done() is always available to end the loop).
    # The Gym rows keep "auto", since simple_agent stops only on a reply without a tool call.
    tool_choice: str = "required"
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    # Test-only DOM oracle instead of the VLM (agent/testing/oracle_grounder.py). It
    # refuses to run outside pytest or the --use-oracle-grounder debug flag.
    use_oracle_grounder: bool = False

    def __post_init__(self):
        if isinstance(self.thinking, bool):  # YAML 1.1 reads a bare `thinking: off` as False
            object.__setattr__(self, "thinking", "on" if self.thinking else "off")
        if self.grounder_model not in SERVED_NAMES:
            raise ValueError(f"grounder_model must be one of {SERVED_NAMES}, got {self.grounder_model!r}")
        if self.summary_model not in SERVED_NAMES:
            raise ValueError(f"summary_model must be one of {SERVED_NAMES}, got {self.summary_model!r}")
        if self.thinking not in THINKING_MODES:
            raise ValueError(f"thinking must be one of {THINKING_MODES}, got {self.thinking!r}")
        if self.model_family not in FAMILIES:
            raise ValueError(f"model_family must be one of {FAMILIES}, got {self.model_family!r}")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.tool_choice not in ("auto", "required"):
            raise ValueError(f"tool_choice must be 'auto' or 'required', got {self.tool_choice!r}")

    def with_options(self, **changes: Any) -> "AgentConfig":
        """Copy with changes, e.g. with_options(grounder_model="grounder", thinking="off", lightning="local").

        `lightning` may be a preset name ("hosted" | "local"), a dict, or a LightningConfig.
        Nested fields take a dict or a dotted key: browser={"headless": False} == **{"browser.headless": False}.
        """
        for key in [k for k in changes if "." in k]:
            top, sub = key.split(".", 1)
            nested = changes.setdefault(top, {})
            if not isinstance(nested, dict):
                raise ValueError(f"can't combine {key!r} with a non-dict {top!r}")
            nested[sub] = changes.pop(key)
        if "lightning" in changes:
            changes["lightning"] = self._resolve_lightning(changes["lightning"])
        for key in ("vlm", "viewport", "sampling", "timeouts", "browser"):
            if isinstance(changes.get(key), dict):
                changes[key] = dataclasses.replace(getattr(self, key), **changes[key])
        return dataclasses.replace(self, **changes)

    def _resolve_lightning(self, value: Any) -> LightningConfig:
        if isinstance(value, LightningConfig):
            return value
        if isinstance(value, str):
            if value not in self.lightning_presets:
                raise ValueError(f"unknown lightning preset {value!r}; have {sorted(self.lightning_presets)}")
            return LightningConfig(**self.lightning_presets[value])
        return dataclasses.replace(self.lightning, **value)

    @property
    def run_dir_root(self) -> Path:
        p = Path(self.runs_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    def public_dict(self) -> dict:
        """JSON-safe summary for traces and events (no secrets). Which checkpoint the Lightning
        endpoint actually serves is not config: agent/core.py adds it to the start event."""
        return {
            "sandbox_url": self.sandbox_url,
            "lightning": {"base_url": self.lightning.base_url, "model": self.lightning.model},
            "vlm": {"base_url": self.vlm.base_url},
            "grounder_model": self.grounder_model,
            "summary_model": self.summary_model,
            "model_family": self.model_family,
            "max_steps": self.max_steps,
            "thinking": self.thinking,
            "prompt_version": PROMPT_VERSION,
            "sampling": dataclasses.asdict(self.sampling),
            "history_full_steps": self.history_full_steps,
            "tool_choice": self.tool_choice,
            "timeouts": dataclasses.asdict(self.timeouts),
            "viewport": dataclasses.asdict(self.viewport),
            "use_oracle_grounder": self.use_oracle_grounder,
        }


def _read_dotenv(path: Path = REPO_ROOT / ".env") -> dict[str, str]:
    """Minimal .env reader (`export KEY=value` or `KEY=value`, # comments). Values are not logged."""
    out: dict[str, str] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        value = value.strip()
        if value[:1] in ("'", '"') and value[0] in value[1:]:
            value = value[1:value.index(value[0], 1)]
        else:  # unquoted: like the shell, stop at whitespace (drops inline comments)
            value = value.split(None, 1)[0] if value else ""
        if value:
            out[key.strip()] = value
    return out


_ENV_SCALARS = {
    "AGENT_SANDBOX_URL": ("sandbox_url", str),
    "AGENT_GROUNDER_MODEL": ("grounder_model", str),
    "AGENT_SUMMARY_MODEL": ("summary_model", str),
    "AGENT_MODEL_FAMILY": ("model_family", str),
    "AGENT_MAX_STEPS": ("max_steps", int),
    "AGENT_THINKING": ("thinking", str),
    "AGENT_RUNS_DIR": ("runs_dir", str),
}


def _bool(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "on")


def _env_overrides(env: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for var, (key, cast) in _ENV_SCALARS.items():
        if env.get(var):
            out[key] = cast(env[var])
    if env.get("AGENT_HEADLESS"):
        out["browser"] = {"headless": _bool(env["AGENT_HEADLESS"])}
    if env.get("AGENT_LIGHTNING"):
        out["lightning"] = env["AGENT_LIGHTNING"]
    lightning_fields = {
        k: env[v] for k, v in (("base_url", "AGENT_LIGHTNING_BASE_URL"), ("model", "AGENT_LIGHTNING_MODEL"),
                               ("api_key_env", "AGENT_LIGHTNING_API_KEY_ENV")) if env.get(v)
    }
    if lightning_fields:
        out["lightning_fields"] = lightning_fields
    vlm_fields = {k: env[v] for k, v in (("base_url", "AGENT_VLM_BASE_URL"), ("api_key", "AGENT_VLM_API_KEY")) if env.get(v)}
    if vlm_fields:
        out["vlm"] = vlm_fields
    return out


def _from_dict(raw: dict) -> AgentConfig:
    raw = dict(raw or {})
    kwargs: dict[str, Any] = {}
    nested = {"lightning": LightningConfig, "vlm": VLMConfig, "viewport": Viewport, "sampling": Sampling,
              "timeouts": Timeouts, "browser": BrowserConfig}
    for key, value in raw.items():
        if key in nested and isinstance(value, dict):
            kwargs[key] = nested[key](**value)
        else:
            kwargs[key] = value
    return AgentConfig(**kwargs)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None,
                env: dict[str, str] | None = None) -> AgentConfig:
    """Build AgentConfig from yaml + env vars + explicit overrides (see module docstring)."""
    env = dict(os.environ if env is None else env)
    explicit = path or env.get("AGENT_CONFIG")
    path = Path(explicit or DEFAULT_CONFIG_PATH)
    if explicit and not path.exists():  # a typo must not silently fall back to other endpoints
        raise FileNotFoundError(f"agent config {path} does not exist")
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    cfg = _from_dict(raw)

    env_over = _env_overrides(env)
    lightning_fields = env_over.pop("lightning_fields", None)
    cfg = cfg.with_options(**env_over) if env_over else cfg
    if lightning_fields:
        cfg = cfg.with_options(lightning=lightning_fields)
    if overrides:
        cfg = cfg.with_options(**overrides)
    return cfg
