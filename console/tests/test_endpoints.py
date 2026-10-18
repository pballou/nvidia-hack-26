"""Where the models are: /api/config "endpoints", the grounder check of POST /api/run, console.json's endpoints,
the console flags (--grounder, --vlm-url, --summary-model, --vlm-label) and scripts/demo_up.sh's checks.

No model is ever called. The real agent's config path (agent.config.load_config with the console's overrides) runs
for real; its run_task is a fake that writes a two-event trace. The VLM is a fake GET /v1/models server on a free
port, whose served names a test can change.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from console import __main__ as console_main
from console.agent_bridge import (ServedModels, describe_endpoints, endpoint_warnings, endpoints_for, get_agent,
                                  lightning_label, real_overrides, unserved_grounder, vlm_label)
from console.app import ConsoleSettings
from console.tests.conftest import free_port, messages, read_sse

REPO = Path(__file__).resolve().parents[2]
DEMO_UP = REPO / "scripts" / "demo_up.sh"
LOCAL_LIGHTNING = {"base_url": "http://127.0.0.1:8000/v1", "model": "lightning", "label": "local BF16 (:8000)"}


# --- a fake VLM: GET /v1/models only -------------------------------------------------------------------------


class FakeModels:
    """GET /v1/models -> {"data": [{"id": ...}]} with `served` (mutable); anything else 404. Counts the GETs."""

    def __init__(self, served: list[str]):
        self.served = list(served)
        self.hits = 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path.rstrip("/") != "/v1/models":
                    self.send_error(404)
                    return
                fake.hits += 1
                body = json.dumps({"object": "list", "data": [{"id": s, "object": "model"} for s in fake.served]})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_vlm():
    servers: list[FakeModels] = []

    def make(served=("base", "grounder")) -> FakeModels:
        srv = FakeModels(list(served))
        servers.append(srv)
        return srv

    yield make
    for srv in servers:
        srv.close()


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No AGENT_* / CONSOLE_VLM_LABEL from this machine; the real agent's traces go to tmp (AGENT_RUNS_DIR)."""
    for key in list(os.environ):
        if key.startswith("AGENT_") or key == "CONSOLE_VLM_LABEL":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AGENT_RUNS_DIR", str(tmp_path / "agent-runs"))
    return tmp_path / "agent-runs"


def fake_run_task(seen: list | None = None):
    """Stands in for agent.core.run_task: a start event with the real cfg.public_dict(), then done. No models."""

    async def run_task(request, *, start_url, cfg, on_event=None):
        if seen is not None:
            seen.append(cfg)
        rid = f"20261017-000000-{time.time_ns() % 10**8:08d}"
        d = Path(cfg.run_dir_root) / rid
        d.mkdir(parents=True)

        async def emit(step, kind, payload):
            e = {"run_id": rid, "step": step, "kind": kind, "ts": round(time.time(), 3), "payload": payload}
            with (d / "events.jsonl").open("a") as f:
                f.write(json.dumps(e) + "\n")
            await on_event(e)

        await emit(0, "start", {"request": request, "start_url": start_url, "config": cfg.public_dict()})
        await emit(1, "done", {"success_claimed": False, "summary": "fake", "steps": 1, "reason": "done", "stats": {}})
        return {"run_id": rid, "success_claimed": False, "summary": "fake", "steps": 1, "trace_dir": str(d)}

    return run_task


def real_console(console, sandbox_url, tmp_path, seen=None, **kw):
    """A console with agent=real: the real make_cfg (load_config + the console's overrides), a fake run_task."""
    kw = {"agent": "real", **kw}
    settings = ConsoleSettings(sandbox_url=sandbox_url, runs_dir=tmp_path / "runs", static_dir=tmp_path / "static", **kw)
    _, make_cfg = get_agent(settings)
    return console((fake_run_task(seen), make_cfg), **kw)


def post_run(base, grounder, **extra):
    return httpx.post(f"{base}/api/run", json={"request": "Make the text bigger", "options": {"grounder": grounder},
                                               **extra})


# --- labels and pure helpers ---------------------------------------------------------------------------------


@pytest.mark.parametrize("url, override, label", [
    ("http://127.0.0.1:8001/v1", None, "H200 via tunnel"),
    ("http://localhost:8001/v1", None, "H200 via tunnel"),
    ("http://127.0.0.1:8011/v1", None, "Mac (vllm-metal)"),
    ("http://127.0.0.1:18001/v1", None, "127.0.0.1:18001"),
    ("https://vlm.example.com/v1", None, "vlm.example.com:443"),
    ("http://[::1]:9000/v1", None, "[::1]:9000"),
    ("http://127.0.0.1:8011/v1", "  my Mac  ", "my Mac"),   # --vlm-label / CONSOLE_VLM_LABEL wins
    ("http://127.0.0.1:8001/v1", "", "H200 via tunnel"),
    ("not a url", None, None), (None, None, None), ("", "x", "x"),
])
def test_vlm_label(url, override, label):
    assert vlm_label(url, override) == label


@pytest.mark.parametrize("url, label", [
    ("http://127.0.0.1:8000/v1", "local BF16 (:8000)"),
    ("http://localhost:8000/v1", "local BF16 (:8000)"),
    ("https://integrate.api.nvidia.com/v1", "hosted API"),
    ("http://127.0.0.1:8100/v1", "127.0.0.1:8100"),
    ("http://10.0.0.5:8000/v1", "10.0.0.5:8000"),
    (None, None),
])
def test_lightning_label(url, label):
    assert lightning_label(url) == label


def test_served_models_probe_cache_and_failures(fake_vlm):
    vlm = fake_vlm(["base", "grounder"])
    served = ServedModels(ttl_s=60)
    assert served.get(vlm.url) == (["base", "grounder"], True)
    vlm.served = ["grounder"]
    assert served.get(vlm.url + "/") == (["base", "grounder"], True)  # cached (same URL, trailing slash aside)
    assert (served.n_checks, vlm.hits) == (1, 1)
    assert ServedModels(ttl_s=0).get(vlm.url) == (["grounder"], True)  # ttl 0: asks again
    assert ServedModels().get(f"http://127.0.0.1:{free_port()}/v1") == (None, False)  # nothing listening
    assert ServedModels().get(vlm.url.replace("/v1", "/nope")) == (None, False)      # 404


def test_endpoints_for_and_messages():
    class Cfg:  # anything shaped like AgentConfig
        class vlm:
            base_url = "http://127.0.0.1:8011/v1"
        class lightning:
            base_url, model = "http://127.0.0.1:8000/v1", "lightning"
        summary_model = "grounder"

    class Served:
        def get(self, url):
            assert url == "http://127.0.0.1:8011/v1"
            return ["grounder"], True

    ep = endpoints_for(Cfg, served=Served())
    assert ep == {"vlm": {"base_url": "http://127.0.0.1:8011/v1", "label": "Mac (vllm-metal)", "summary_model": "grounder",
                          "served": ["grounder"], "reachable": True},
                  "lightning": LOCAL_LIGHTNING}
    assert endpoints_for({"options": {}}) == {"vlm": None, "lightning": None}  # a test's dict cfg: nothing known
    assert endpoints_for(Cfg)["vlm"]["served"] is None  # no probe: unknown, not empty
    assert unserved_grounder(ep, "base") == ("grounder model 'base' is not served at http://127.0.0.1:8011/v1 "
                                             "(serves: grounder)")
    assert unserved_grounder(ep, "grounder") is None
    assert unserved_grounder({"vlm": {**ep["vlm"], "served": None, "reachable": False}}, "base") is None  # unknown
    assert unserved_grounder({"vlm": None, "lightning": None}, "base") is None
    assert endpoint_warnings(ep, "grounder") == []
    assert endpoint_warnings(ep, "base") == ["warning: default grounder 'base' is not served at "
                                             "http://127.0.0.1:8011/v1 (serves: grounder)"]
    both = {"vlm": {**ep["vlm"], "summary_model": "base"}}
    assert [w.split(" '")[0] for w in endpoint_warnings(both, "base")] == ["warning: default grounder",
                                                                            "warning: summary model"]
    assert "not reachable" in endpoint_warnings({"vlm": {**ep["vlm"], "reachable": False, "served": None}}, "base")[0]
    assert describe_endpoints(ep) == ("VLM Mac (vllm-metal) (127.0.0.1:8011, summary grounder, serves grounder) · "
                                      "Lightning local BF16 (:8000) (lightning)")
    assert describe_endpoints({"vlm": None, "lightning": None}) == "endpoints: none (the mock runs no models)"


def test_real_overrides_map_the_flags_to_load_config(clean_env):
    from agent.config import load_config

    opts = {"grounder": "grounder", "thinking": "off", "lightning": "local"}
    plain = ConsoleSettings(agent="real")
    assert real_overrides(plain, opts) == {"grounder_model": "grounder", "thinking": "off", "lightning": "local",
                                           "sandbox_url": plain.sandbox_url}
    flags = ConsoleSettings(agent="real", vlm_url="http://127.0.0.1:8011/v1/", summary_model="grounder", headed=True)
    over = real_overrides(flags, opts)
    assert over["vlm"] == {"base_url": "http://127.0.0.1:8011/v1"} and over["summary_model"] == "grounder"
    assert over["browser"] == {"headless": False}
    cfg = load_config(overrides=over)
    assert (cfg.vlm.base_url, cfg.vlm.api_key, cfg.summary_model) == ("http://127.0.0.1:8011/v1", "EMPTY", "grounder")
    with pytest.raises(ValueError):
        ConsoleSettings(summary_model="large")
    with pytest.raises(ValueError):
        ConsoleSettings(default_grounder="big")


def test_cli_flags_reach_the_settings(monkeypatch):
    seen = {}
    monkeypatch.setattr(console_main, "create_app", lambda settings: seen.setdefault("settings", settings))
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    monkeypatch.setattr(console_main, "get_agent", lambda settings: (None, lambda options: {}))
    monkeypatch.setenv("CONSOLE_VLM_LABEL", "from env")
    assert console_main.main(["--agent", "mock", "--sandbox-url", f"http://127.0.0.1:{free_port()}"]) == 0
    s = seen.pop("settings")
    assert (s.default_grounder, s.vlm_url, s.summary_model, s.vlm_label) == ("grounder", None, None, "from env")
    assert console_main.main(["--grounder", "base", "--vlm-url", "http://127.0.0.1:8011/v1", "--summary-model",
                              "grounder", "--vlm-label", "Mac", "--sandbox-url", f"http://127.0.0.1:{free_port()}"]) == 0
    s = seen.pop("settings")
    assert (s.default_grounder, s.vlm_url, s.summary_model, s.vlm_label) == ("base", "http://127.0.0.1:8011/v1",
                                                                             "grounder", "Mac")
    assert s.default_options()["grounder"] == "base"
    with pytest.raises(SystemExit):
        console_main.main(["--grounder", "large"])
    with pytest.raises(SystemExit):
        console_main.main(["--summary-model", "large"])


# --- /api/config ---------------------------------------------------------------------------------------------


def test_config_endpoints_null_for_the_mock(console, tmp_path):
    base, _ = console(None, runs_dir=tmp_path / "runs", vlm_url="http://127.0.0.1:8011/v1", summary_model="grounder")
    cfg = httpx.get(f"{base}/api/config").json()
    assert cfg["agent"] == "mock" and cfg["endpoints"] == {"vlm": None, "lightning": None}
    assert cfg["defaults"]["grounder"] == "grounder"
    base, _ = console(None, runs_dir=tmp_path / "runs", default_grounder="base")
    assert httpx.get(f"{base}/api/config").json()["defaults"]["grounder"] == "base"


def test_config_endpoints_real_served_and_labels(console, sandbox_url, tmp_path, clean_env, fake_vlm):
    vlm = fake_vlm(["base", "grounder"])
    base, app = real_console(console, sandbox_url, tmp_path, vlm_url=vlm.url, summary_model="grounder")
    cfg = httpx.get(f"{base}/api/config").json()
    assert cfg["agent"] == "real" and cfg["defaults"]["grounder"] == "grounder"
    assert cfg["endpoints"] == {
        "vlm": {"base_url": vlm.url, "label": f"127.0.0.1:{vlm.port}", "summary_model": "grounder",
                "served": ["base", "grounder"], "reachable": True},
        "lightning": LOCAL_LIGHTNING}
    # re-checked at most every 10 s: the next calls reuse the probe
    for _ in range(3):
        httpx.get(f"{base}/api/config")
    assert app.state.served.n_checks == 1 and vlm.hits == 1

    # --vlm-label names it; the hosted toggle default shows the hosted Lightning
    base, _ = real_console(console, sandbox_url, tmp_path, vlm_url=vlm.url, vlm_label="Mac (vllm-metal)",
                           default_lightning="hosted")
    ep = httpx.get(f"{base}/api/config").json()["endpoints"]
    assert ep["vlm"]["label"] == "Mac (vllm-metal)" and ep["vlm"]["summary_model"] == "base"  # configs/agent.yaml
    assert ep["lightning"] == {"base_url": "https://integrate.api.nvidia.com/v1",
                               "model": "nvidia/nemotron-3.5-lightning-30b-a3b", "label": "hosted API"}


def test_config_endpoints_follow_the_env_and_the_flag_beats_it(console, sandbox_url, tmp_path, clean_env, fake_vlm,
                                                                monkeypatch):
    env_vlm, flag_vlm = fake_vlm(["base"]), fake_vlm(["grounder"])
    monkeypatch.setenv("AGENT_VLM_BASE_URL", env_vlm.url)
    monkeypatch.setenv("AGENT_SUMMARY_MODEL", "grounder")
    base, _ = real_console(console, sandbox_url, tmp_path)
    ep = httpx.get(f"{base}/api/config").json()["endpoints"]["vlm"]
    assert (ep["base_url"], ep["summary_model"], ep["served"]) == (env_vlm.url, "grounder", ["base"])
    base, _ = real_console(console, sandbox_url, tmp_path, vlm_url=flag_vlm.url, summary_model="base")
    ep = httpx.get(f"{base}/api/config").json()["endpoints"]["vlm"]
    assert (ep["base_url"], ep["summary_model"], ep["served"]) == (flag_vlm.url, "base", ["grounder"])


def test_config_endpoints_unreachable_vlm(console, sandbox_url, tmp_path, clean_env):
    dead = f"http://127.0.0.1:{free_port()}/v1"
    base, _ = real_console(console, sandbox_url, tmp_path, vlm_url=dead)
    ep = httpx.get(f"{base}/api/config").json()["endpoints"]
    assert ep["vlm"] == {"base_url": dead, "label": dead.split("//")[1].removesuffix("/v1"), "summary_model": "base",
                         "served": None, "reachable": False}
    # served names unknown: the console doesn't guess, the run starts (here the fake agent; the real one would fail)
    r = post_run(base, "base")
    assert r.status_code == 200, r.text
    read_sse(f"{base}/api/runs/{r.json()['run_id']}/events")


def test_unserved_grounder_400_and_console_json_endpoints(console, sandbox_url, tmp_path, clean_env, fake_vlm):
    vlm = fake_vlm(["grounder"])  # the fully local Mac: one merged model, no base
    seen: list = []
    base, app = real_console(console, sandbox_url, tmp_path, seen=seen, vlm_url=vlm.url, summary_model="grounder",
                             vlm_label="Mac (vllm-metal)", served_ttl_s=0.0)
    r = post_run(base, "base")
    assert r.status_code == 400
    assert r.json() == {"error": f"grounder model 'base' is not served at {vlm.url} (serves: grounder)"}
    assert seen == []  # nothing started

    r = post_run(base, "grounder", task_id="text-size-large")
    assert r.status_code == 200, r.text
    run = r.json()
    stream = messages(read_sse(f"{base}/api/runs/{run['run_id']}/events"))
    assert [m["event"] for m in stream][-2:] == ["status", "end"]
    cfg = seen[0]
    assert (cfg.vlm.base_url, cfg.summary_model, cfg.grounder_model) == (vlm.url, "grounder", "grounder")

    want = {"vlm": {"base_url": vlm.url, "label": "Mac (vllm-metal)", "summary_model": "grounder",
                    "served": ["grounder"], "reachable": True},
            "lightning": LOCAL_LIGHTNING}
    rid = next(m["data"]["run_id"] for m in stream if m["event"] == "result")
    doc = json.loads((clean_env / rid / "console.json").read_text())
    assert doc["endpoints"] == want and doc["status"] == "done"
    assert httpx.get(f"{base}/api/runs/{run['run_id']}").json()["endpoints"] == want

    # the logs page shows where the run's models were, from console.json
    s = httpx.get(f"{base}/api/logs/runs/{rid}").json()["summary"]
    assert s["endpoints"] == {"vlm": {**want["vlm"], "host": f"127.0.0.1:{vlm.port}"},
                              "lightning": {**LOCAL_LIGHTNING, "host": "127.0.0.1:8000"}, "source": "console"}

    # the VLM now serves base too: the next run may use it (served_ttl_s=0 re-checks every time)
    vlm.served = ["base", "grounder"]
    r = post_run(base, "base")
    assert r.status_code == 200, r.text
    read_sse(f"{base}/api/runs/{r.json()['run_id']}/events")


def test_mock_console_json_has_null_endpoints(console, tmp_path):
    from console.tests.test_logs import recording_agent

    runs = tmp_path / "runs"
    gate = threading.Event()
    gate.set()
    base, _ = console(recording_agent(runs, gate, []), runs_dir=runs)  # agent=mock (the default)
    r = post_run(base, "base")  # no served-name check for the mock
    assert r.status_code == 200, r.text
    read_sse(f"{base}/api/runs/{r.json()['run_id']}/events")
    doc = json.loads(next(runs.glob("*/console.json")).read_text())
    assert doc["endpoints"] == {"vlm": None, "lightning": None} and doc["agent"] == "mock"


# --- scripts/demo_up.sh -------------------------------------------------------------------------------------------


def test_demo_up_help_lists_the_flags():
    out = subprocess.run(["bash", str(DEMO_UP), "--help"], capture_output=True, text=True, timeout=20)
    assert out.returncode == 0, out.stderr
    for flag in ("--agent", "--lightning", "--grounder", "--vlm-url", "--summary-model", "--vlm-label", "--asr",
                 "--headed", "--skip-models"):
        assert flag in out.stdout, flag
    assert ('scripts/demo_up.sh --agent real --vlm-url http://127.0.0.1:8011/v1 --summary-model grounder'
            in out.stdout)
    for bad in (["--grounder", "large"], ["--summary-model", "x"], ["--vlm-url", "ftp://x"], ["--nope"]):
        r = subprocess.run(["bash", str(DEMO_UP), *bad], capture_output=True, text=True, timeout=20)
        assert r.returncode == 2 and r.stderr, bad


def test_demo_up_checks_the_override_url_and_the_served_names(console, sandbox_url, tmp_path, fake_vlm):
    """The sandbox and a console are already up (free ports): demo_up.sh only checks the models, then exits."""
    vlm, lightning = fake_vlm(["grounder"]), fake_vlm(["lightning"])
    console_base, _ = console(None, runs_dir=tmp_path / "runs")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AGENT_", "CONSOLE_"))}
    env.update(SANDBOX_PORT=sandbox_url.rsplit(":", 1)[1], CONSOLE_PORT=console_base.rsplit(":", 1)[1],
               LIGHTNING_LOCAL_URL=lightning.url, PYTHON=str(REPO / ".venv-gym" / "bin" / "python"),
               CONSOLE_ASR_URL=f"http://127.0.0.1:{free_port()}")

    def run(*args):
        r = subprocess.run(["bash", str(DEMO_UP), "--asr", "stub", *args], capture_output=True, text=True,
                           timeout=60, env=env)
        assert r.returncode == 0, (r.stdout, r.stderr)
        return r.stdout, r.stderr

    out, err = run("--vlm-url", vlm.url + "/", "--summary-model", "grounder")  # the Mac: grounder only
    assert f"vlm      {vlm.url} up, served: grounder (grounder: grounder, summary: grounder)" in out, out
    assert "is not served" not in err, err
    assert f"lightning local {lightning.url} up, served: lightning" in out
    assert "console  already running" in out and vlm.hits == 1

    out, err = run("--vlm-url", vlm.url, "--grounder", "base")  # summary model from configs/agent.yaml: base
    assert f"warning: grounder 'base' is not served at {vlm.url} (serves: grounder)" in err, err
    assert f"warning: summary model 'base' is not served at {vlm.url} (serves: grounder)" in err, err

    env["AGENT_VLM_BASE_URL"] = vlm.url  # no flag: the agent's config (and its env vars) decide
    out, err = run()
    assert f"vlm      {vlm.url} up, served: grounder (grounder: grounder, summary: base)" in out, out
    assert "summary model 'base' is not served" in err and "grounder 'grounder'" not in err

    dead = f"http://127.0.0.1:{free_port()}/v1"
    out, err = run("--vlm-url", dead)
    assert f"warning: VLM {dead} not reachable" in err, err
