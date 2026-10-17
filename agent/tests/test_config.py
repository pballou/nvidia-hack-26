"""agent/config.py: yaml + env overrides + with_options, validation, and no secrets in output."""

from __future__ import annotations

import json
import re

import pytest

from agent.config import DEFAULT_CONFIG_PATH, REPO_ROOT, AgentConfig, LightningConfig, _read_dotenv, load_config
from agent.prompts import PROMPT_VERSION

SECRET = "nvapi-test-secret-do-not-print"


def test_yaml_defaults():
    cfg = load_config(DEFAULT_CONFIG_PATH, env={})
    assert cfg.sandbox_url == "http://127.0.0.1:8090"
    # the default endpoint is one of the presets (local on the Mac, or hosted)
    lightning = {"base_url": cfg.lightning.base_url, "model": cfg.lightning.model, "api_key_env": cfg.lightning.api_key_env}
    assert lightning in cfg.lightning_presets.values()
    assert cfg.lightning_presets["hosted"]["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert cfg.lightning_presets["hosted"]["api_key_env"] == "NVIDIA_API_KEY"
    # a local OpenAI-compatible VLM; the port moves with gate D1 (:8002 Qwen3-VL until then, :8001 after)
    assert re.fullmatch(r"http://127\.0\.0\.1:\d+/v1", cfg.vlm.base_url) and cfg.vlm.api_key == "EMPTY"
    assert (cfg.grounder_model, cfg.summary_model) == ("base", "base")
    assert cfg.model_family in ("qwen3_vl", "qwen3_5")
    assert (cfg.max_steps, cfg.thinking, cfg.history_full_steps) == (12, "retry", 6)
    assert (cfg.viewport.width, cfg.viewport.height) == (1280, 800)
    assert (cfg.sampling.temperature, cfg.sampling.top_p) == (1.0, 0.95)
    assert cfg.timeouts.lightning_s == 180
    assert cfg.browser.headless is True
    assert set(cfg.lightning_presets) >= {"hosted", "local"}
    assert cfg.use_oracle_grounder is False


def test_custom_yaml_and_agent_config_env(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text("grounder_model: grounder\nmax_steps: 4\nthinking: \"off\"\nvlm: {base_url: http://vlm:9/v1}\n")
    cfg = load_config(path, env={})
    assert (cfg.grounder_model, cfg.max_steps, cfg.thinking) == ("grounder", 4, "off")
    assert cfg.vlm.base_url == "http://vlm:9/v1" and cfg.vlm.api_key == "EMPTY"
    assert cfg.summary_model == "base"  # dataclass default for keys the yaml leaves out
    assert load_config(env={"AGENT_CONFIG": str(path)}).grounder_model == "grounder"


@pytest.mark.parametrize("value,expected", [("off", "off"), ("on", "on"), ("retry", "retry"), ('"off"', "off")])
def test_bare_yaml_thinking_values(tmp_path, value, expected):
    path = tmp_path / "agent.yaml"
    path.write_text(f"thinking: {value}\n")  # YAML 1.1: bare off/on are booleans
    assert load_config(path, env={}).thinking == expected


def test_env_overrides():
    env = {"AGENT_GROUNDER_MODEL": "grounder", "AGENT_LIGHTNING": "local", "AGENT_VLM_BASE_URL": "http://10.0.0.5:8001/v1",
           "AGENT_MAX_STEPS": "5", "AGENT_THINKING": "on", "AGENT_HEADLESS": "false", "AGENT_SUMMARY_MODEL": "grounder"}
    cfg = load_config(DEFAULT_CONFIG_PATH, env=env)
    assert cfg.grounder_model == "grounder" and cfg.summary_model == "grounder"
    assert cfg.lightning.base_url == "http://127.0.0.1:8000/v1"
    assert cfg.lightning.model == "lightning"
    assert cfg.lightning.api_key == "EMPTY"  # local preset is keyless
    assert cfg.vlm.base_url == "http://10.0.0.5:8001/v1" and cfg.vlm.api_key == "EMPTY"
    assert (cfg.max_steps, cfg.thinking, cfg.browser.headless) == (5, "on", False)
    assert cfg.browser.locale == "en-US"  # partial nested override keeps the rest


def test_env_hosted_preset():
    cfg = load_config(DEFAULT_CONFIG_PATH, env={"AGENT_LIGHTNING": "hosted"})
    assert cfg.lightning.base_url == "https://integrate.api.nvidia.com/v1"
    assert cfg.lightning.model == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert cfg.lightning.api_key_env == "NVIDIA_API_KEY"


def test_env_lightning_fields_and_precedence():
    env = {"AGENT_LIGHTNING": "local", "AGENT_LIGHTNING_MODEL": "lightning-nvfp4", "AGENT_GROUNDER_MODEL": "grounder"}
    cfg = load_config(DEFAULT_CONFIG_PATH, env=env)
    assert cfg.lightning.base_url == "http://127.0.0.1:8000/v1"
    assert cfg.lightning.model == "lightning-nvfp4"
    # explicit overrides beat env vars
    cfg = load_config(DEFAULT_CONFIG_PATH, overrides={"grounder_model": "base", "lightning": "hosted"}, env=env)
    assert cfg.grounder_model == "base"
    assert cfg.lightning.model == "nvidia/nemotron-3.5-lightning-30b-a3b"


def test_with_options():
    cfg = AgentConfig()
    c2 = cfg.with_options(grounder_model="grounder", thinking="off", lightning="local",
                          vlm={"base_url": "http://x:1/v1"}, browser={"headless": False}, runs_dir="/tmp/agent-runs")
    assert (c2.grounder_model, c2.thinking) == ("grounder", "off")
    assert c2.lightning.model == "lightning"
    assert c2.vlm.base_url == "http://x:1/v1" and c2.vlm.api_key == "EMPTY"
    assert c2.browser.headless is False and c2.browser.settle_ms == cfg.browser.settle_ms
    assert str(c2.run_dir_root) == "/tmp/agent-runs"
    assert cfg.grounder_model == "base" and cfg.browser.headless is True  # original untouched
    c3 = cfg.with_options(lightning={"model": "other"})
    assert c3.lightning.model == "other" and c3.lightning.base_url == cfg.lightning.base_url
    assert c3.with_options(lightning=LightningConfig(model="m")).lightning.model == "m"


def test_dotted_override_keys():
    """The console passes nested fields either way (load_config(overrides=...))."""
    cfg = load_config(overrides={"browser.headless": False, "vlm.base_url": "http://y:2/v1", "thinking": "off"},
                      env={})
    assert cfg.browser.headless is False and cfg.vlm.base_url == "http://y:2/v1" and cfg.thinking == "off"
    both = AgentConfig().with_options(browser={"settle_ms": 1}, **{"browser.headless": False})
    assert (both.browser.settle_ms, both.browser.headless) == (1, False)


def test_tool_choice_default_and_public():
    assert AgentConfig().tool_choice == "required"
    assert AgentConfig().public_dict()["tool_choice"] == "required"


def test_read_dotenv_follows_shell_semantics(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# comment line\n"
        "export NVIDIA_API_KEY=      # build.nvidia.com: hosted Nemotron 3.5 Lightning\n"  # .env.example as copied
        "export HF_TOKEN=hf_abc   # read-only\n"
        "QUOTED=\"two words\" # trailing\n"
        "SINGLE='x#y'\n"
        "HASHED=#notacomment\n"
        "EMPTY=\n")
    assert _read_dotenv(env) == {"HF_TOKEN": "hf_abc", "QUOTED": "two words", "SINGLE": "x#y",
                                 "HASHED": "#notacomment"}


def test_relative_runs_dir_is_under_repo():
    assert AgentConfig().run_dir_root == REPO_ROOT / "runs"


@pytest.mark.parametrize("changes", [
    {"grounder_model": "grounder-v2"}, {"summary_model": "lightning"}, {"thinking": "sometimes"},
    {"max_steps": 0}, {"lightning": "cloud"}, {"model_family": "qwen3.5"}, {"tool_choice": "any"},
])
def test_invalid_values_raise(changes):
    with pytest.raises(ValueError):
        AgentConfig().with_options(**changes)


def test_invalid_env_value_raises():
    with pytest.raises(ValueError):
        load_config(DEFAULT_CONFIG_PATH, env={"AGENT_THINKING": "maybe"})
    with pytest.raises(ValueError):
        load_config(DEFAULT_CONFIG_PATH, env={"AGENT_LIGHTNING": "nowhere"})
    with pytest.raises(ValueError, match="model_family"):
        load_config(DEFAULT_CONFIG_PATH, env={"AGENT_MODEL_FAMILY": "qwen3.5"})


def test_missing_explicit_config_raises(tmp_path, monkeypatch):
    missing = tmp_path / "agent_bf16.yaml"
    with pytest.raises(FileNotFoundError, match="agent_bf16.yaml"):
        load_config(missing, env={})
    with pytest.raises(FileNotFoundError):
        load_config(env={"AGENT_CONFIG": str(missing)})
    monkeypatch.setattr("agent.config.DEFAULT_CONFIG_PATH", missing)  # only the implicit default may be absent
    assert load_config(env={}) == AgentConfig()


def test_public_dict_has_no_key(monkeypatch):
    monkeypatch.setenv("AGENT_TEST_LIGHTNING_KEY", SECRET)
    cfg = AgentConfig().with_options(lightning={"api_key_env": "AGENT_TEST_LIGHTNING_KEY"})
    assert cfg.lightning.api_key == SECRET
    pub = cfg.public_dict()
    text = json.dumps(pub)  # JSON-safe
    assert SECRET not in text
    assert "api_key" not in text
    assert pub["lightning"] == {"base_url": cfg.lightning.base_url, "model": cfg.lightning.model}
    assert {"grounder_model", "summary_model", "thinking", "max_steps", "viewport"} <= set(pub)
    # enough to tell runs apart that share an endpoint (core.py adds the served checkpoint root)
    assert pub["prompt_version"] == PROMPT_VERSION and pub["history_full_steps"] == cfg.history_full_steps
    assert pub["sampling"] == {"temperature": 1.0, "top_p": 0.95, "max_tokens": 1024, "thinking_max_tokens": 4096}
    assert pub["timeouts"]["lightning_s"] == cfg.timeouts.lightning_s


def test_lightning_repr_hides_key(monkeypatch):
    monkeypatch.setenv("AGENT_TEST_LIGHTNING_KEY", SECRET)
    lc = LightningConfig(api_key_env="AGENT_TEST_LIGHTNING_KEY")
    assert lc.api_key == SECRET
    assert SECRET not in repr(lc) and SECRET not in str(lc)
    assert SECRET not in repr(AgentConfig(lightning=lc))


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("AGENT_TEST_UNSET_KEY_XYZ", raising=False)
    with pytest.raises(RuntimeError, match="AGENT_TEST_UNSET_KEY_XYZ"):
        _ = LightningConfig(api_key_env="AGENT_TEST_UNSET_KEY_XYZ").api_key
    assert LightningConfig(api_key_env=None).api_key == "EMPTY"
