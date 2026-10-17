"""Tests for the web_agent resources server.

Most tests use in-memory fakes, wired in through app.py's module-level seams:
- SandboxSim: T1's sandbox HTTP contract (meta.sid / meta.task_id written at /api/reset, a default
  state recreated for an unknown sid, like sandbox/state.py), served through httpx.MockTransport;
- FakeBrowser: a page with one High contrast toggle whose clicks flip SandboxSim's state;
- FakeVLM: a grounder that finds "toggle" targets and a canned summarizer.

test_full_episode_* drives real Chromium against agent/testing/fake_sandbox.py with the test-only
DOM oracle injected through app.make_vlm (pytest only; the server never uses the oracle).

Run from the repo root: .venv-gym/bin/python -m pytest resources_servers/web_agent/tests -q
"""

import asyncio
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from agent.observation import build_observation
from agent.prompts import PROMPT_VERSION, SYSTEM_PROMPT, initial_user_message, responses_tools
from agent.session import ALREADY_DONE, DONE_NOTE, Located, Summary
from agent.testing.fake_sandbox import TASKS_BY_ID, FakeSandboxServer
from agent.testing.oracle_grounder import OracleGrounder
from agent.vlm import VLMClient
from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.web_agent import app as app_module
from resources_servers.web_agent.app import WebAgentConfig, WebAgentResourcesServer
from resources_servers.web_agent.make_tasks import screen_sha

EXAMPLE_ROWS = Path(app_module.__file__).parent / "data" / "example.jsonl"
TOGGLE = (600, 300, 680, 340)  # left, top, right, bottom in viewport px
TASK = {
    "id": "high-contrast-test", "request": "Turn on high contrast.", "start_path": "/settings?tab=accessibility",
    "tags": ["easy", "settings"], "steps_min": 1,
    "check": {"all": [{"path": "settings.accessibility.high_contrast", "op": "eq", "value": True}]},
}
RESPONSE = NeMoGymResponse(id="resp", created_at=0, model="lightning", object="response", output=[],
                           tool_choice="auto", parallel_tool_calls=False, tools=[]).model_dump(mode="json")
CLICK_TOGGLE = {"target_description": "the High contrast toggle"}


# --- fakes -------------------------------------------------------------------------------------

class SandboxSim:
    def __init__(self):
        self.states: dict[str, dict] = {}
        self.resets: list[dict] = []
        self.up = True
        self.reject_resets = False

    @staticmethod
    def default(sid: str, task_id: str | None = None) -> dict:
        return {"meta": {"sid": sid, "task_id": task_id}, "settings": {"accessibility": {"high_contrast": False}}}

    def state(self, sid: str) -> dict:
        return self.states.setdefault(sid, self.default(sid))

    def handler(self, request: httpx.Request) -> httpx.Response:
        if not self.up:
            raise httpx.ConnectError("sandbox down", request=request)
        if request.method == "POST" and request.url.path == "/api/reset":
            body = json.loads(request.content)
            self.resets.append(body)
            if self.reject_resets:
                return httpx.Response(404, json={"ok": False, "error": f"unknown task_id {body['task_id']!r}"})
            self.states[body["sid"]] = self.default(body["sid"], body["task_id"])
            return httpx.Response(200, json={"ok": True})
        if request.method == "GET" and request.url.path == "/api/state":
            return httpx.Response(200, json=self.state(request.url.params["sid"]))
        return httpx.Response(404, json={"ok": False})


def render(toggle_on: bool) -> Image.Image:
    img = Image.new("RGB", (1280, 800), "white")
    ImageDraw.Draw(img).rectangle(TOGGLE, fill=(30, 160, 60) if toggle_on else (150, 150, 150))
    return img


class TargetClosedError(Exception):
    """Named like Playwright's: agent.session._brief renders it as 'TargetClosedError: ...'."""


class FakeBrowser:
    def __init__(self, sim: SandboxSim):
        self.sim, self.sid, self.page_url = sim, "", ""
        self.clicks: list[tuple[float, float]] = []
        self.events: list[str] = []
        self.closed = 0
        self.failure: Exception | None = None  # raised by click and screenshot (a dead or hung page)

    def _a11y(self) -> dict:
        return self.sim.state(self.sid)["settings"]["accessibility"]

    async def goto(self, url: str) -> None:
        self.events.append("goto")
        self.page_url, self.sid = url, parse_qs(urlsplit(url).query)["sid"][0]

    async def screenshot(self) -> Image.Image:
        self.events.append("screenshot")
        if self.failure is not None:
            raise self.failure
        return render(self._a11y()["high_contrast"])

    async def click(self, x: float, y: float) -> None:
        if self.failure is not None:
            raise self.failure
        self.clicks.append((x, y))
        left, top, right, bottom = TOGGLE
        if left <= x <= right and top <= y <= bottom:  # what the real page's POST /api/action does
            self._a11y()["high_contrast"] = not self._a11y()["high_contrast"]

    async def type(self, text: str) -> None: ...
    async def press(self, key: str) -> None: ...
    async def scroll(self, direction: str, amount: str) -> None: ...
    async def select_all(self) -> None: ...

    async def settle(self) -> None:
        self.events.append("settle")

    async def url(self) -> str:
        return self.page_url

    async def title(self) -> str:
        return "Settings - Acme Utilities"

    def screen_changed(self, before: Image.Image, after: Image.Image) -> bool:
        return before.tobytes() != after.tobytes()

    async def close(self) -> None:
        self.closed += 1


class FakeVLM:
    def __init__(self):
        self.locate_calls: list[str] = []
        self.summarize_calls = 0
        self.down = False      # answer like agent/vlm.py when the server is unreachable
        self.delay_s = 0.0

    async def locate(self, img: Image.Image, target: str) -> Located:
        self.locate_calls.append(target)
        await asyncio.sleep(self.delay_s)
        if self.down:
            return Located(None, None, error="VLM unreachable at http://vlm.test/v1: ConnectError: refused",
                           model="fake")
        if "toggle" not in target.lower():
            return Located(None, None, raw="I can't find it", error="no point in reply", model="fake")
        left, top, right, bottom = TOGGLE
        return Located((left + right) / 2, (top + bottom) / 2, raw='{"point_2d": [500, 400]}',
                       point_1000=(500, 400), model="fake")

    async def summarize(self, img: Image.Image) -> Summary:
        self.summarize_calls += 1
        if self.down:
            return Summary("", error="VLM timed out after 45s at http://vlm.test/v1", model="fake")
        return Summary("PAGE: Settings\nELEMENTS:\n- toggle: High contrast @ center", model="fake")


# --- helpers and fixtures -----------------------------------------------------------------------

def make_row(task: dict = TASK, *, initial_observation: dict | None = None, initial_screen_sha: str | None = None,
             source: str = "placeholder", rcp: bool = True) -> dict:
    """A task row shaped like make_tasks.py's output."""
    row = {
        "task_id": task["id"], "start_path": task["start_path"], "check": task["check"], "tags": task["tags"],
        "steps_min": task["steps_min"], "initial_observation": initial_observation,
        "initial_screen_sha": initial_screen_sha, "initial_observation_source": source,
        "prompt_version": PROMPT_VERSION,
    }
    if rcp:
        screen = "Step 0 | URL: /settings | Title: Settings\nScreen:\n(screen description not generated)"
        row["responses_create_params"] = {
            "input": [{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": initial_user_message(task["request"], screen)}],
            "tools": responses_tools(strict=False), "parallel_tool_calls": False,
            "temperature": 1.0, "top_p": 0.95, "max_output_tokens": 1024,
        }
    return row


def make_server(**cfg) -> WebAgentResourcesServer:
    cfg.setdefault("sandbox_url", "http://sandbox.test")
    return WebAgentResourcesServer(
        config=WebAgentConfig(name="web_agent", host="127.0.0.1", port=0, entrypoint="app.py", **cfg),
        server_client=MagicMock(spec=ServerClient))


def closed_port_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"  # nothing listens once the socket closes


def seed(client: TestClient, row: dict | None = None) -> None:
    r = client.post("/seed_session", json=row or make_row())
    assert (r.status_code, r.json()) == (200, {})


def tool(client: TestClient, name: str, args) -> dict:
    r = client.post(f"/{name}", json=args)
    assert r.status_code == 200, r.text
    return r.json()


def verify(client: TestClient, row: dict | None = None) -> dict:
    r = client.post("/verify", json=(row or make_row()) | {"response": RESPONSE})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture
def sim() -> SandboxSim:
    return SandboxSim()


@pytest.fixture
def mock_sandbox(monkeypatch, sim):
    monkeypatch.setattr(app_module, "sandbox_client", lambda base_url, timeout_s: httpx.AsyncClient(
        base_url=base_url, transport=httpx.MockTransport(sim.handler)))


@pytest.fixture
def fakes(monkeypatch, sim) -> SimpleNamespace:
    ns = SimpleNamespace(browsers=[], vlm=FakeVLM())

    async def new_browser_session(cfg):
        ns.browsers.append(FakeBrowser(sim))
        return ns.browsers[-1]

    monkeypatch.setattr(app_module, "new_browser_session", new_browser_session)
    monkeypatch.setattr(app_module, "make_vlm", lambda cfg, browser_session: ns.vlm)
    return ns


@pytest.fixture
def server_cfg() -> dict:
    """WebAgentConfig overrides for the `client` fixture (tests override this fixture via parametrize)."""
    return {}


@pytest.fixture
def client(fakes, mock_sandbox, server_cfg):
    with TestClient(make_server(**server_cfg).setup_webserver()) as c:
        yield c


# --- (a) verifier ---------------------------------------------------------------------------------

def test_correct_state_scores_one(client, fakes, sim):
    seed(client)
    out = tool(client, "click", CLICK_TOGGLE)
    assert out["ok"] is True and out["click"] == {"x": 640, "y": 320}
    assert "The screen changed." in out["observation"] and "PAGE: Settings" in out["observation"]
    v = verify(client)
    assert (v["reward"], v["mask_sample"], v["failure_kind"]) == (1.0, False, None)
    assert (v["task_id"], v["tags"], v["steps"], v["n_clicks"]) == ("high-contrast-test", ["easy", "settings"], 1, 1)
    assert (v["done_called"], v["claimed_success"], v["state_source"]) == (False, False, "live")
    assert v["check_explanation"] == "all 1 conditions passed"
    assert (v["grounder_model"], v["summary_model"], v["prompt_version"]) == ("base", "base", PROMPT_VERSION)
    assert v["response"]["id"] == "resp" and v["responses_create_params"]["parallel_tool_calls"] is False
    assert sim.resets[0]["task_id"] == "high-contrast-test" and sim.resets[0]["sid"].startswith("gym-")
    assert fakes.browsers[0].page_url.startswith("http://sandbox.test/settings?tab=accessibility&sid=gym-")
    assert fakes.browsers[0].closed == 1  # released inside verify


def test_wrong_state_scores_zero(client):
    seed(client)
    v = verify(client)
    assert (v["reward"], v["mask_sample"]) == (0.0, False)
    assert "expected True, got False" in v["check_explanation"]


def test_sandbox_down_at_seed_is_masked_and_nothing_raises(fakes):
    with TestClient(make_server(sandbox_url=closed_port_url()).setup_webserver()) as client:
        seed(client)  # still 200 {}
        out = tool(client, "click", CLICK_TOGGLE)
        assert out["ok"] is False and out["error"].startswith("the environment failed to start: ConnectError")
        v = verify(client)
    assert (v["reward"], v["mask_sample"], v["failure_kind"]) == (0.0, True, "provider_unavailable")
    assert v["failure_reason"].startswith("seed failed: ConnectError") and v["steps"] == 1
    assert fakes.browsers == []  # never got as far as a browser


def test_sandbox_down_at_verify_is_masked(client, sim):
    seed(client)
    sim.up = False
    v = verify(client)
    assert (v["reward"], v["mask_sample"], v["failure_kind"]) == (0.0, True, "provider_unavailable")
    assert "sandbox unreachable at verify" in v["failure_reason"]


def test_verify_without_session_is_masked(client):
    v = verify(client)
    assert (v["reward"], v["mask_sample"], v["failure_kind"]) == (0.0, True, "session_lost")
    assert v["task_id"] == "high-contrast-test"  # from the row


def test_state_of_another_task_is_masked(client, sim):
    seed(client)
    sid = sim.resets[0]["sid"]
    sim.states.pop(sid)  # sandbox restarted: /api/state recreates a default state (task_id None)
    v = verify(client)
    assert (v["reward"], v["mask_sample"], v["failure_kind"]) == (0.0, True, "session_lost")
    assert "meta.task_id=None" in v["failure_reason"]


def test_vlm_outage_masks_the_rollout_even_when_the_state_passes(client, fakes, sim):
    seed(client)
    assert tool(client, "click", CLICK_TOGGLE)["ok"] is True
    fakes.vlm.down = True
    out = tool(client, "click", CLICK_TOGGLE)
    assert out["ok"] is False and "VLM unreachable" in out["error"]
    assert sim.states[sim.resets[0]["sid"]]["settings"]["accessibility"]["high_contrast"] is True
    v = verify(client)  # the check would pass, but an outage must not count for either grounder
    assert (v["reward"], v["mask_sample"], v["failure_kind"]) == (0.0, True, "provider_unavailable")
    assert v["n_vlm_infra_errors"] == 1 and "VLM unreachable" in v["failure_reason"]


def test_summary_outage_is_infra_but_a_bad_point_is_not(client, fakes):
    seed(client)
    out = tool(client, "click", {"target_description": "the gear icon"})  # FakeVLM: unparsable reply
    assert out["ok"] is False
    fakes.vlm.down = True
    tool(client, "scroll", {"direction": "down", "amount": "page"})  # acts, then the summary times out
    v = verify(client)
    assert (v["mask_sample"], v["n_vlm_infra_errors"]) == (True, 1)
    assert "VLM timed out" in v["failure_reason"]


@pytest.mark.parametrize("failure", [
    TargetClosedError("Mouse.click: Target page, context or browser has been closed"),
    Exception("Page.screenshot: Target crashed "),
    TimeoutError("Page.screenshot: Timeout 15000ms exceeded."),
], ids=["closed", "crashed", "hung"])
def test_dead_browser_masks_instead_of_scoring_a_policy_failure(client, fakes, failure):
    seed(client)
    fakes.browsers[0].failure = failure
    out = tool(client, "click", CLICK_TOGGLE)
    assert out["ok"] is False and "(screen unavailable: " in out["observation"]
    v = verify(client)
    assert (v["reward"], v["mask_sample"], v["failure_kind"]) == (0.0, True, "provider_unavailable")
    assert v["n_vlm_infra_errors"] == 1 and "click: browser failure: " in v["failure_reason"]


def test_browser_infra_error_reads_errors_and_fresh_screen_failures_only():
    def outcome(error=None, summary="PAGE: Settings"):
        return SimpleNamespace(error=error, observation=SimpleNamespace(screen_summary=summary))

    dead = "(screen unavailable: TargetClosedError: Page.screenshot: Target page, context or browser has been closed)"
    assert app_module.browser_infra_error(outcome("Error: Mouse.click: Target crashed")) == (
        "Error: Mouse.click: Target crashed")
    assert app_module.browser_infra_error(outcome("ValueError: bad key", dead)).startswith("TargetClosedError")
    assert app_module.browser_infra_error(outcome("could not locate 'x': no point", dead), dead) is None  # reused
    assert app_module.browser_infra_error(outcome(None, "PAGE: a dialog that has been closed")) is None  # VLM text
    assert app_module.browser_infra_error(outcome("could not locate 'x': VLM timed out after 45s")) is None
    # The model's target text can't fake a crash: only Playwright's '<Exc>: <Class.method>: ...' shape counts.
    assert app_module.browser_infra_error(outcome("could not locate 'the banner that has been closed': no point")) is None
    assert app_module.browser_infra_error(outcome("Error: Keyboard.press: Unknown key: \"Foo\"")) is None


@pytest.mark.parametrize("server_cfg", [{"tool_timeout_s": 0.05}])
def test_slow_tool_times_out_and_masks(client, fakes):
    seed(client)
    fakes.vlm.delay_s = 1.0
    out = tool(client, "click", CLICK_TOGGLE)
    assert out["ok"] is False and out["error"].startswith("click timed out after 0.05s")
    assert out["observation"].startswith("Step 0")  # the primed start observation
    v = verify(client)
    assert (v["mask_sample"], v["failure_kind"], v["n_vlm_infra_errors"]) == (True, "provider_unavailable", 1)


def test_tool_result_bytes_match_the_interactive_planner(client):
    seed(client)
    r = client.post("/click", json=CLICK_TOGGLE)
    assert r.headers["content-type"] == "application/json"
    assert r.text == json.dumps(r.json(), ensure_ascii=False)  # ", " / ": " separators, like agent/planner.py
    r = client.post("/fly", json={})
    assert r.text == json.dumps(r.json(), ensure_ascii=False)


def test_stage_latency_reaches_the_verify_response(client):
    seed(client)
    tool(client, "click", CLICK_TOGGLE)
    v = verify(client)
    assert set(v["latency_ms"]) >= {"ground", "act", "settle", "summary", "screenshot"}
    assert all(len(xs) == 1 for xs in v["latency_ms"].values())
    for stage in ("summary", "ground", "act", "settle", "screenshot"):
        assert v[f"{stage}_ms"] == pytest.approx(sum(v["latency_ms"][stage]), abs=0.2)


def test_verify_never_raises_on_garbage(client):
    seed(client)
    for kwargs in ({"content": b"{not json"}, {"json": [1, 2]}, {"json": "x"}, {"json": {}},
                   {"json": {"task_id": 7, "check": "nope", "tags": "x", "response": "not an object"}}):
        r = client.post("/verify", **kwargs)
        assert r.status_code == 200, (kwargs, r.text)
        assert (r.json()["reward"], r.json()["mask_sample"], r.json()["failure_kind"]) == (
            0.0, True, "web_agent:bad_request")
    assert tool(client, "click", CLICK_TOGGLE)["ok"] is True  # the episode kept running
    # simple_agent's shape with garbage around it is a real verify: scored from the seeded check.
    r = client.post("/verify", json={"response": {}, "responses_create_params": {}, "task_id": "high-contrast-test",
                                     "check": "nope", "_ng_rollout_id": 5})
    assert r.status_code == 200 and (r.json()["reward"], r.json()["mask_sample"]) == (1.0, False)
    assert r.json()["task_id"] == "high-contrast-test"


def test_model_calling_verify_scores_nothing_and_keeps_the_episode(client, fakes):
    """A tool call named verify posts its arguments on the episode's cookie: it must not score the
    episode, show the model any sandbox state (check_explanation) or release the browser."""
    seed(client)
    for args in ({"response": {}}, {"response": {}, "responses_create_params": {}},
                 {"response": {}, "responses_create_params": {}, "task_id": "another-task"}):
        r = client.post("/verify", json=args)
        assert r.status_code == 200, r.text
        v = r.json()
        assert (v["reward"], v["mask_sample"], v["failure_kind"]) == (0.0, True, "web_agent:bad_request"), args
        assert v["check_explanation"] is None and v["state_source"] is None
    assert fakes.browsers[0].closed == 0
    assert tool(client, "click", CLICK_TOGGLE)["ok"] is True
    v = verify(client)
    assert (v["reward"], v["mask_sample"], v["steps"]) == (1.0, False, 1)


# --- (b) seed_session --------------------------------------------------------------------------------

def test_legacy_seed_primes_from_row_without_a_vlm_call(client, fakes):
    obs = build_observation(step=0, url="http://127.0.0.1:8090/settings?sid=gen", title="Settings",
                            screen_changed=None, last_action_result="", screen_summary="PAGE: from the row")
    row = make_row(initial_observation=obs.to_dict(), initial_screen_sha=screen_sha(render(False)), source="vlm:base")
    seed(client, row)
    assert fakes.vlm.summarize_calls == 0
    out = tool(client, "click", {"target": "the toggle"})  # invalid: reuses the primed observation
    assert out["ok"] is False and "PAGE: from the row" in out["observation"]
    v = verify(client, row)
    assert (v["initial_obs_stale"], v["initial_observation_source"]) == (False, "vlm:base")


def test_seed_settles_before_the_start_screenshot(client, fakes):
    """Same frame as make_tasks.py / the interactive agent: goto() (settles) then reset() (settles again)."""
    seed(client)
    assert fakes.browsers[0].events[:3] == ["goto", "settle", "screenshot"]


def test_stale_start_screen_is_flagged(client):
    row = make_row(initial_screen_sha="0" * 64)
    seed(client, row)
    v = verify(client, row)
    assert v["initial_obs_stale"] is True and v["mask_sample"] is False  # a flag, not a mask


def test_committed_example_rows_seed_and_verify(client, fakes):
    rows = [json.loads(line) for line in EXAMPLE_ROWS.read_text().splitlines() if line.strip()]
    assert len(rows) == 5  # `gym env test +should_validate_data=true` requires exactly 5
    for row in rows:
        seed(client, row)
        out = tool(client, "click", {"target": "x"})  # invalid: re-renders the row's start observation
        assert row["initial_observation"]["screen_summary"].splitlines()[0] in out["observation"]
        v = verify(client, row)
        assert (v["mask_sample"], v["task_id"], v["n_invalid_calls"]) == (False, row["task_id"], 1)
        assert v["initial_observation_source"] == row["initial_observation_source"]
        if row["initial_screen_sha"] is None:  # placeholder rows: no fingerprint, no stale check
            assert v["initial_obs_stale"] is None
    assert fakes.vlm.summarize_calls == 0 and all(b.closed == 1 for b in fakes.browsers)


def test_typed_seed_binds_the_session(client, sim):
    row = make_row(rcp=False)
    body = ResourcesSeedSessionRequest(
        resources_session_id="rs-1", episode_id=EpisodeId(rollout_id="r1", attempt=0),
        task_id=TaskId(taskset="web_agent", task_id=TASK["id"]), task_data=row).model_dump(mode="json")
    r = client.post("/seed_session", json=body)
    assert r.status_code == 200 and r.json()["resources_session_id"] == "rs-1"
    assert sim.resets[0]["sid"] == "gym-rs-1"
    assert tool(client, "click", CLICK_TOGGLE)["ok"] is True  # cookie now carries rs-1
    v = verify(client, row | {"responses_create_params": make_row()["responses_create_params"]})
    assert (v["reward"], v["mask_sample"]) == (1.0, False)
    close = {"resources_session_id": "rs-1", "episode_id": {"rollout_id": "r1", "attempt": 0}}
    assert client.post("/close_session", json=close).json() == {"resources_session_id": "rs-1"}


def test_typed_session_contract(fakes, mock_sandbox):
    seed_body = ResourcesSeedSessionRequest(
        resources_session_id="rs-contract", episode_id=EpisodeId(rollout_id="r1", attempt=0),
        task_id=TaskId(taskset="web_agent", task_id=TASK["id"]), task_data=make_row(rcp=False))
    check_resources_session_contract(make_server().setup_webserver(), seed_body, keeps_state=True)


def test_seed_without_rcp_and_empty_close_leave_the_episode_alone(client, fakes):
    seed(client)
    # What a model calling tools named seed_session / close_session with {} would send.
    assert client.post("/seed_session", json={}).json() == {}
    assert client.post("/close_session", json={}).json()["ok"] is False
    assert tool(client, "click", CLICK_TOGGLE)["ok"] is True
    assert len(fakes.browsers) == 1 and fakes.browsers[0].closed == 0


def test_reseed_of_a_live_episode_is_a_no_op(client, fakes, sim):
    seed(client)
    assert tool(client, "click", CLICK_TOGGLE)["ok"] is True
    # A model "calling" seed_session with a whole row on its own cookie must not reset the episode.
    seed(client, make_row(TASKS_BY_ID["fake_text_large"]))
    assert len(sim.resets) == 1 and len(fakes.browsers) == 1 and fakes.browsers[0].closed == 0
    v = verify(client)
    assert (v["task_id"], v["reward"], v["steps"]) == ("high-contrast-test", 1.0, 1)


@pytest.mark.parametrize("server_cfg", [{"episode_ttl_s": 0.0}])
def test_seed_sweeps_episodes_that_never_reached_verify(client, fakes):
    seed(client)  # this /run "fails" before verify
    client.cookies.clear()  # the next /run is a new Gym session
    seed(client)
    assert fakes.browsers[0].closed == 1 and fakes.browsers[1].closed == 0
    v = verify(client)
    assert (v["mask_sample"], v["task_id"]) == (False, "high-contrast-test")


def test_bad_row_is_masked(client, fakes):
    row = make_row()
    del row["check"]
    seed(client, row)
    out = tool(client, "click", CLICK_TOGGLE)
    assert out["ok"] is False and "row high-contrast-test has no check object" in out["error"]
    v = verify(client, row)
    assert (v["mask_sample"], v["failure_kind"]) == (True, "web_agent:bad_row")


def test_rejected_reset_is_masked(client, sim, fakes):
    sim.reject_resets = True
    seed(client)
    v = verify(client)
    assert (v["mask_sample"], v["failure_kind"]) == (True, "web_agent:reset_rejected")
    assert "404" in v["failure_reason"] and fakes.browsers == []


# --- (c) tool routes ---------------------------------------------------------------------------------

def test_tool_without_session(client):
    out = tool(client, "click", CLICK_TOGGLE)
    assert out == {"ok": False, "observation": "",
                   "error": "no active environment session (not seeded, or already verified)"}


def test_wrong_argument_names_get_a_corrective_message_and_run_nothing(client, fakes):
    seed(client)
    out = tool(client, "click", {"target": "the High contrast toggle"})
    assert out["ok"] is False
    assert "unknown argument(s) target" in out["error"]
    assert "missing required argument(s) target_description" in out["error"]
    assert "Expected click(target_description: string)" in out["error"]
    assert fakes.browsers[0].clicks == [] and fakes.vlm.locate_calls == []
    v = verify(client)
    assert (v["steps"], v["n_invalid_calls"], v["n_clicks"], v["n_failed_actions"]) == (1, 1, 0, 0)


def test_unknown_tool_hits_the_catch_all(client):
    seed(client)
    out = tool(client, "fly_to_moon", {"speed": "fast"})
    assert out["ok"] is False and "unknown tool 'fly_to_moon'" in out["error"]
    assert "click(target_description: string)" in out["error"]
    assert "done(summary: string, success: boolean)" in out["error"]
    assert out["observation"].startswith("Step 1")  # the last observation, re-rendered
    assert verify(client)["n_invalid_calls"] == 1


def test_invalid_enum_and_non_object_arguments(client):
    seed(client)
    out = tool(client, "scroll", {"direction": "sideways", "amount": "page"})
    assert out["ok"] is False and "scroll.direction must be one of ['up', 'down']" in out["error"]
    out = tool(client, "press_key", "Enter")
    assert out["ok"] is False and "arguments for press_key must be a JSON object" in out["error"]
    r = client.post("/press_key")  # no body at all
    assert r.status_code == 200 and r.json()["ok"] is False
    assert verify(client)["n_invalid_calls"] == 3


def test_failed_grounding_is_a_failed_action(client, fakes):
    seed(client)
    out = tool(client, "click", {"target_description": "the gear icon"})
    assert out["ok"] is False and "could not locate 'the gear icon'" in out["error"]
    assert fakes.browsers[0].clicks == []
    v = verify(client)
    assert (v["n_failed_actions"], v["n_invalid_calls"], v["n_clicks"]) == (1, 0, 0)
    assert (v["mask_sample"], v["n_vlm_infra_errors"]) == (False, 0)  # a grounder miss is the policy's to own


def test_done_snapshots_state_and_later_actions_are_refused(client, fakes, sim):
    seed(client)
    assert tool(client, "click", CLICK_TOGGLE)["ok"] is True
    out = tool(client, "done", {"summary": "High contrast is on.", "success": True})
    assert out["ok"] is True and DONE_NOTE in out["observation"]
    sid = sim.resets[0]["sid"]
    after = tool(client, "click", CLICK_TOGGLE)
    assert (after["ok"], after["error"]) == (False, ALREADY_DONE)
    assert len(fakes.browsers[0].clicks) == 1 and sim.states[sid]["settings"]["accessibility"]["high_contrast"] is True
    sim.states[sid]["settings"]["accessibility"]["high_contrast"] = False  # live state drifts after done
    v = verify(client)
    assert (v["reward"], v["state_source"]) == (1.0, "done_snapshot")
    assert (v["done_called"], v["claimed_success"]) == (True, True)
    assert (v["steps"], v["n_clicks"], v["n_calls_after_done"], v["n_failed_actions"]) == (3, 1, 1, 0)


# --- config and metrics ------------------------------------------------------------------------------

def test_served_names_are_config_only():
    server = make_server(grounder_model="grounder", summary_model="grounder", vlm_base_url="http://vlm.test/v1",
                         headless=False)
    cfg = server._cfg
    assert (cfg.grounder_model, cfg.summary_model, cfg.vlm.base_url, cfg.browser.headless) == (
        "grounder", "grounder", "http://vlm.test/v1", False)
    assert cfg.sandbox_url == "http://sandbox.test"
    vlm = app_module.make_vlm(cfg, None)  # the production factory: the VLM, never the DOM oracle
    assert isinstance(vlm, VLMClient) and vlm.cfg.grounder_model == "grounder"


def test_bad_config_fails_at_startup():
    with pytest.raises(ValueError, match="grounder_model"):
        make_server(grounder_model="oracle")
    with pytest.raises(ValueError, match="num_workers=1"):
        make_server(num_workers=2)
    with pytest.raises(ValueError, match=r"unknown web_agent config key\(s\) \['grounder_modle'\]"):
        make_server(grounder_modle="grounder")  # a typo must not silently keep `base`
    make_server(domain="agent", verified=False, description="d", value="v")  # Gym's metadata keys are fine


def test_yaml_server_block_is_accepted(monkeypatch):
    """Every key of configs/web_agent.yaml's server block, resolved like Gym does, is a known one."""
    from omegaconf import OmegaConf

    from agent.config import load_config

    for var in ("AGENT_GROUNDER_MODEL", "AGENT_SUMMARY_MODEL", "AGENT_MODEL_FAMILY", "AGENT_VLM_BASE_URL",
                "AGENT_SANDBOX_URL", "AGENT_MAX_STEPS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AGENT_GROUNDER_MODEL", "grounder")
    conf = OmegaConf.load(Path(app_module.__file__).parent / "configs" / "web_agent.yaml")
    block = OmegaConf.to_container(conf.web_agent.resources_servers.web_agent, resolve=True)
    server = WebAgentResourcesServer(config=WebAgentConfig(name="web_agent", host="127.0.0.1", port=0, **block),
                                     server_client=MagicMock(spec=ServerClient))
    assert server.config.num_workers == 1 and server.config.model_family is None
    # Unset fields are null, so configs/agent.yaml (+ AGENT_* env) decides, exactly as for the interactive agent.
    assert (server.config.grounder_model, server.config.summary_model) == ("grounder", None)
    agent_cfg = load_config()
    assert (server._cfg.grounder_model, server._cfg.summary_model) == ("grounder", agent_cfg.summary_model)
    # Gym stays at 12 Lightning calls so reported numbers stay comparable; the interactive agent has a
    # larger cap plus the "stuck" stop (agent/progress.py), which Gym's simple_agent loop doesn't have.
    assert conf.web_agent_simple_agent.responses_api_agents.simple_agent.max_steps == 12
    assert agent_cfg.max_steps == 25


def test_compute_metrics_by_tag():
    server = make_server()
    tasks = [[{"reward": 1.0, "tags": ["easy", "settings"]}, {"reward": 0.0, "tags": ["easy", "settings"]}],
             [{"reward": 1.0, "tags": ["medium"]}]]
    metrics = server.compute_metrics(tasks)
    assert metrics["tag/easy/mean_reward"] == 0.5 and metrics["tag/easy/rollouts"] == 2
    assert metrics["tag/medium/mean_reward"] == 1.0
    assert server.get_key_metrics(metrics | {"mean/reward": 0.66}) == {
        "mean/reward": 0.66, "tag/easy/mean_reward": 0.5, "tag/medium/mean_reward": 1.0,
        "tag/settings/mean_reward": 0.5}


# --- (d) one full episode: real Chromium + fake sandbox pages + the test-only oracle ---------------------

@pytest.fixture(scope="module")
def fake_sandbox_url():
    with FakeSandboxServer() as url:
        yield url


def test_full_episode_through_the_http_routes(monkeypatch, fake_sandbox_url):
    monkeypatch.setattr(app_module, "make_vlm", lambda cfg, browser_session: OracleGrounder(browser_session))
    row = make_row(TASKS_BY_ID["fake_high_contrast"])
    with TestClient(make_server(sandbox_url=fake_sandbox_url, headless=True).setup_webserver()) as client:
        seed(client, row)
        out = tool(client, "click", CLICK_TOGGLE)
        assert out["ok"] is True and "The screen changed." in out["observation"], out
        done = tool(client, "done", {"summary": "High contrast is on.", "success": True})
        assert done["ok"] is True and DONE_NOTE in done["observation"]
        v = verify(client, row)
    assert (v["reward"], v["mask_sample"], v["claimed_success"], v["steps"], v["n_clicks"]) == (1.0, False, True, 2, 1)
    assert v["state_source"] == "done_snapshot"
    # the fake sandbox records meta.sid/task_id like the real one, so the session check ran
    assert "session check skipped" not in v["check_explanation"]
