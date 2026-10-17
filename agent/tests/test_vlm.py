"""agent/vlm.py against a fake OpenAI-compatible VLM server (httpx.MockTransport, no network)."""

from __future__ import annotations

import base64
import io
import json

import httpx
import openai
import pytest
from PIL import Image

from agent.config import AgentConfig, load_config
from agent.vlm import VLMClient, strip_thinking
from common.grounding import SUMMARY_SAMPLING, smart_resize_dims

BASE_URL = "http://vlm.test/v1"


class FakeVLM:
    """Records chat-completions requests; `replies` is a list of str (content), int (status) or Exception."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": {"message": f"fake status {reply}"}})
        body = self.requests[-1]
        return httpx.Response(200, json={
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    def image_url(self, i: int = -1) -> str:
        content = self.requests[i]["messages"][0]["content"]
        return next(part["image_url"]["url"] for part in content if part["type"] == "image_url")


def make_vlm(fake: FakeVLM, cfg: AgentConfig | None = None) -> VLMClient:
    # Default max_retries (2) on purpose: VLMClient must enforce its own single retry.
    client = openai.AsyncOpenAI(base_url=BASE_URL, api_key="x",
                                http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake)))
    return VLMClient(cfg or AgentConfig(), client, retry_backoff_s=0.0)


def screen(w: int = 1280, h: int = 800) -> Image.Image:
    return Image.new("RGB", (w, h), (240, 244, 248))


def decode(url: str) -> Image.Image:
    prefix = "data:image/png;base64,"
    assert url.startswith(prefix)
    return Image.open(io.BytesIO(base64.b64decode(url[len(prefix):])))


@pytest.mark.asyncio
async def test_locate_converts_grid_point_to_viewport_px():
    fake = FakeVLM('{"point_2d": [500, 250]}')
    loc = await make_vlm(fake).locate(screen(), "the gear icon in the top-right toolbar")
    assert (loc.x, loc.y) == (640, 200)
    assert loc.point_1000 == (500, 250) and loc.error is None
    assert loc.raw == '{"point_2d": [500, 250]}' and loc.model == "base" and loc.latency_ms >= 0
    req = fake.requests[0]
    assert req["model"] == "base" and req["temperature"] == 0 and req["max_tokens"] == 64
    assert "extra_body" not in req and "chat_template_kwargs" not in req
    text = next(p["text"] for p in req["messages"][0]["content"] if p["type"] == "text")
    assert "the gear icon in the top-right toolbar" in text and "point_2d" in text


@pytest.mark.asyncio
async def test_results_carry_prompt_sha_and_usage():
    """For trace events / the console Logs page: which prompt, and tokens when the server reports them."""
    from agent.vlm import GROUNDING_PROMPT_SHA, SUMMARY_PROMPT_SHA

    vlm = make_vlm(FakeVLM('{"point_2d": [500, 250]}', "PAGE: x"))
    loc = await vlm.locate(screen(), "the gear icon")
    s = await vlm.summarize(screen())
    assert loc.prompt_sha == GROUNDING_PROMPT_SHA["qwen3_vl"] and len(loc.prompt_sha) == 10
    assert s.prompt_sha == SUMMARY_PROMPT_SHA and s.prompt_sha != loc.prompt_sha
    assert loc.usage == s.usage == {"prompt_tokens": 1, "completion_tokens": 1}
    fam = make_vlm(FakeVLM('{"point_2d": [1, 2]}'), AgentConfig().with_options(model_family="qwen3_5"))
    assert (await fam.locate(screen(), "x")).prompt_sha == GROUNDING_PROMPT_SHA["qwen3_5"]
    failed = await make_vlm(FakeVLM(500)).summarize(screen())
    assert failed.error and failed.prompt_sha == SUMMARY_PROMPT_SHA and failed.usage is None


@pytest.mark.asyncio
async def test_image_is_preprocessed_png_and_point_uses_original_size():
    fake = FakeVLM('{"point_2d": [500, 250]}')
    big = screen(2560, 1600)  # over MAX_PIXELS -> downscaled before sending
    loc = await make_vlm(fake).locate(big, "the Save button")
    sent = decode(fake.image_url())
    assert sent.format == "PNG" and sent.mode == "RGB"
    assert sent.size == smart_resize_dims(2560, 1600) != (2560, 1600)
    assert (loc.x, loc.y) == (1280, 400)  # original pixels, not the preprocessed size


@pytest.mark.asyncio
async def test_viewport_screenshot_is_sent_at_full_size():
    fake = FakeVLM('{"point_2d": [1, 1]}')
    await make_vlm(fake).locate(screen().convert("RGBA"), "the bell icon")
    assert decode(fake.image_url()).size == (1280, 800)


@pytest.mark.asyncio
async def test_point_on_the_edge_is_clamped_into_the_image():
    loc = await make_vlm(FakeVLM('{"point_2d": [1000, 1000]}')).locate(screen(), "bottom-right corner")
    assert (loc.x, loc.y) == (1279, 799)


@pytest.mark.asyncio
async def test_unparsable_reply_is_an_error():
    fake = FakeVLM("I cannot find that element on the screen.")
    loc = await make_vlm(fake).locate(screen(), "the purple unicorn")
    assert loc.x is None and loc.y is None and loc.point_1000 is None
    assert loc.raw == "I cannot find that element on the screen."
    assert loc.error.startswith("could not parse a point from: ")
    assert len(fake.requests) == 1  # a bad answer is not retried


@pytest.mark.asyncio
async def test_think_block_is_ignored_when_parsing_the_point():
    loc = await make_vlm(FakeVLM('<think>maybe {"point_2d": [10, 10]}?</think>{"point_2d": [500, 500]}')).locate(
        screen(), "the center")
    assert (loc.x, loc.y) == (640, 400)


@pytest.mark.asyncio
async def test_model_names_follow_config_and_switch_via_with_options():
    cfg = AgentConfig().with_options(grounder_model="grounder")
    fake = FakeVLM('{"point_2d": [1, 1]}')
    vlm = make_vlm(fake, cfg)
    loc = await vlm.locate(screen(), "the gear icon")
    fake.replies = ["PAGE: Settings"]
    summary = await vlm.summarize(screen())
    assert [r["model"] for r in fake.requests] == ["grounder", "base"]
    assert loc.model == "grounder" and summary.model == "base"

    fake2 = FakeVLM("PAGE: Settings")
    await make_vlm(fake2, cfg.with_options(summary_model="grounder")).summarize(screen())
    assert fake2.requests[0]["model"] == "grounder"


@pytest.mark.asyncio
async def test_model_switch_from_env_is_config_only():
    cfg = load_config(env={"AGENT_GROUNDER_MODEL": "grounder", "AGENT_SUMMARY_MODEL": "grounder"})
    fake = FakeVLM('{"point_2d": [1, 1]}')
    await make_vlm(fake, cfg).locate(screen(), "the gear icon")
    assert fake.requests[0]["model"] == "grounder"


@pytest.mark.asyncio
async def test_summary_strips_thinking_and_is_clipped():
    body = "\n".join(["PAGE: Settings"] + [f"- button: item {i} @ left" for i in range(60)])
    fake = FakeVLM(f"<think>\nLet me look at the screen.\n</think>\n\n{body}")
    summary = await make_vlm(fake).summarize(screen())
    assert summary.error is None and summary.model == "base"
    assert "<think>" not in summary.text and "Let me look" not in summary.text
    lines = summary.text.splitlines()
    assert lines[0] == "PAGE: Settings" and len(lines) == 41 and "more lines omitted" in lines[-1]
    req = fake.requests[0]
    assert req["model"] == "base" and req["temperature"] == 0 and req["max_tokens"] >= 512
    assert any(p["type"] == "image_url" for p in req["messages"][0]["content"])


@pytest.mark.asyncio
async def test_summary_and_grounding_sampling():
    fake = FakeVLM("PAGE: Home")
    vlm = make_vlm(fake)
    await vlm.summarize(screen())
    fake.replies = ['{"point_2d": [1, 1]}']
    await vlm.locate(screen(), "the gear icon")
    summary_req, ground_req = fake.requests
    # the single owner of summary sampling (greedy alone makes Qwen3-VL loop on sparse screens)
    assert {k: summary_req[k] for k in SUMMARY_SAMPLING} == SUMMARY_SAMPLING
    assert summary_req["frequency_penalty"] == 0.3 and summary_req["max_tokens"] == 600
    assert ground_req["temperature"] == 0 and ground_req["max_tokens"] == 64 and "frequency_penalty" not in ground_req


@pytest.mark.asyncio
async def test_bad_request_inputs_are_errors_not_exceptions():
    cfg = AgentConfig()
    object.__setattr__(cfg, "model_family", "qwen9")  # past config validation (see test_config)
    fake = FakeVLM('{"point_2d": [1, 1]}')
    loc = await make_vlm(fake, cfg).locate(screen(), "the gear icon")
    assert loc.x is None and "unknown model family" in loc.error and fake.requests == []


@pytest.mark.asyncio
async def test_empty_summary_is_an_error():
    summary = await make_vlm(FakeVLM("<think>still thinking")).summarize(screen())
    assert summary.text == "" and "empty" in summary.error


@pytest.mark.asyncio
async def test_server_500_once_then_ok_is_retried():
    fake = FakeVLM(500, '{"point_2d": [500, 250]}')
    loc = await make_vlm(fake).locate(screen(), "the gear icon")
    assert len(fake.requests) == 2
    assert loc.error is None and (loc.x, loc.y) == (640, 200)
    assert fake.image_url(0) == fake.image_url(1)


@pytest.mark.asyncio
async def test_rate_limit_once_then_ok_is_retried_for_summary():
    fake = FakeVLM(429, "PAGE: Home")
    summary = await make_vlm(fake).summarize(screen())
    assert len(fake.requests) == 2 and summary.text == "PAGE: Home" and summary.error is None


@pytest.mark.asyncio
async def test_connection_error_twice_returns_error_instead_of_raising():
    fake = FakeVLM(httpx.ConnectError("connection refused"))
    vlm = make_vlm(fake)
    loc = await vlm.locate(screen(), "the gear icon")
    assert len(fake.requests) == 2  # one retry, and the client's own retries are off
    assert loc.x is None and loc.error.startswith(f"VLM unreachable at {BASE_URL}")
    assert "connection refused" in loc.error and loc.model == "base"

    summary = await vlm.summarize(screen())
    assert len(fake.requests) == 4
    assert summary.text == "" and summary.error.startswith(f"VLM unreachable at {BASE_URL}")


@pytest.mark.asyncio
async def test_timeout_is_reported():
    fake = FakeVLM(httpx.ReadTimeout("slow"))
    loc = await make_vlm(fake, AgentConfig().with_options(timeouts={"vlm_s": 7})).locate(screen(), "x")
    assert len(fake.requests) == 2 and loc.error == f"VLM timed out after 7s at {BASE_URL}"


@pytest.mark.asyncio
async def test_client_error_is_not_retried():
    fake = FakeVLM(404)
    loc = await make_vlm(fake, AgentConfig().with_options(grounder_model="grounder")).locate(screen(), "x")
    assert len(fake.requests) == 1
    assert loc.error == f"VLM error 404 from {BASE_URL} (model 'grounder'): fake status 404"


@pytest.mark.asyncio
async def test_errors_never_echo_the_image_data_url():
    def echo(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": f"bad request: {request.content.decode()}"}})

    client = openai.AsyncOpenAI(base_url=BASE_URL, api_key="x",
                                http_client=httpx.AsyncClient(transport=httpx.MockTransport(echo)))
    loc = await VLMClient(AgentConfig(), client, retry_backoff_s=0.0).locate(screen(), "x")
    assert loc.error.startswith(f"VLM error 400 from {BASE_URL}")
    assert "iVBOR" not in loc.error and len(loc.error) < 400


@pytest.mark.asyncio
async def test_qwen3_5_disables_thinking():
    fake = FakeVLM('{"point_2d": [1, 1]}')
    vlm = make_vlm(fake, AgentConfig().with_options(model_family="qwen3_5"))
    await vlm.locate(screen(), "the gear icon")
    fake.replies = ["PAGE: Home"]
    await vlm.summarize(screen())
    for req in fake.requests:
        assert req["chat_template_kwargs"] == {"enable_thinking": False}


def test_default_client_uses_config():
    cfg = AgentConfig().with_options(vlm={"base_url": "http://10.0.0.5:8001/v1"}, timeouts={"vlm_s": 12})
    vlm = VLMClient(cfg)
    assert vlm.base_url == "http://10.0.0.5:8001/v1"
    assert vlm.client.max_retries == 0 and vlm.client.timeout == 12


def test_strip_thinking():
    assert strip_thinking("<think>a\nb</think>\nanswer") == "answer"
    assert strip_thinking("reasoning only</think> answer") == "answer"  # template opened the block
    assert strip_thinking("<think>cut off") == ""
    assert strip_thinking(None) == "" and strip_thinking("plain") == "plain"
