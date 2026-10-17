"""LightningClient retry/downgrade/stats behavior and load_api_key parsing, against httpx.MockTransport (no network)."""

import asyncio
import json
import types

import httpx
import pytest

from data_prep.rewrite import client as cl
from data_prep.rewrite.client import ENDPOINTS, SAMPLING, FatalHTTPError, LightningClient, TransportError, load_api_key

MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
SCHEMA = {"type": "object", "properties": {"items": {"type": "array"}}, "required": ["items"]}
KEY = "nvapi-TEST-not-a-real-key-123"


def ok(content='{"items": []}', usage=(10, 5), finish="stop", model="served-model", headers=None):
    body = {"id": "c1", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}]}
    if usage:
        body["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}
    return httpx.Response(200, json=body, headers=headers)


class FakeSleep:
    def __init__(self):
        self.delays = []

    async def __call__(self, delay):
        self.delays.append(delay)


class Script:
    """Handler that replays `steps` (Response, exception class, or callable(request)) then returns ok() forever."""

    def __init__(self, *steps):
        self.steps, self.requests = list(steps), []

    def __call__(self, request):
        self.requests.append(request)
        step = self.steps.pop(0) if self.steps else ok()
        if isinstance(step, type) and issubclass(step, Exception):
            raise step("simulated", request=request)
        return step(request) if callable(step) else step

    def bodies(self):
        return [json.loads(r.content) for r in self.requests]


def make(handler, endpoint="local", **kw):
    sleep = FakeSleep()
    return LightningClient(ENDPOINTS[endpoint], transport=httpx.MockTransport(handler), sleep=sleep, **kw), sleep


def run(c, *calls):
    """Run complete() for each (messages, json_schema) pair sequentially; return completions or the exception."""
    async def go():
        try:
            return [await c.complete(m, json_schema=s) for m, s in calls]
        except Exception as e:  # noqa: BLE001 - returned to the test for inspection
            return e
        finally:
            await c.aclose()
    return asyncio.run(go())


def test_endpoints_and_sampling_constants():
    assert ENDPOINTS["hosted"].base_url == "https://integrate.api.nvidia.com/v1"
    assert ENDPOINTS["hosted"].model == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert ENDPOINTS["hosted"].api_key_env == "NVIDIA_API_KEY"
    assert ENDPOINTS["local"].api_key_env is None and ENDPOINTS["h200"].model == "lightning"
    assert SAMPLING == {"temperature": 0.7, "top_p": 0.95, "max_tokens": 4096, "chat_template_kwargs": {"enable_thinking": False}}


def test_request_body_and_no_auth_without_key():
    h = Script()
    c, sleep = make(h)
    [comp] = run(c, (MSGS, SCHEMA))
    [req] = h.requests
    assert str(req.url) == "http://127.0.0.1:8000/v1/chat/completions" and req.method == "POST"
    assert "authorization" not in req.headers
    body = json.loads(req.content)
    assert body["model"] == "lightning" and body["messages"] == MSGS
    assert body["temperature"] == 0.7 and body["top_p"] == 0.95 and body["max_tokens"] == 4096
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"] == {"type": "json_object"}
    assert comp.text == '{"items": []}' and comp.finish_reason == "stop" and comp.model == "served-model"
    assert comp.http_attempts == 1 and comp.response_format == "json_object" and sleep.delays == []


def test_auth_header_and_overrides_with_key():
    h = Script()
    c, _ = make(h, endpoint="hosted", api_key=KEY, base_url="https://example.test/v1/", model="other")
    run(c, (MSGS, None))
    [req] = h.requests
    assert req.headers["authorization"] == f"Bearer {KEY}"
    assert str(req.url) == "https://example.test/v1/chat/completions"
    assert json.loads(req.content)["model"] == "other"
    assert KEY not in repr(c)


def test_string_endpoint_and_hosted_defaults():
    h = Script()
    c, _ = make(h, endpoint="hosted")
    c2 = LightningClient("hosted", transport=httpx.MockTransport(h))
    assert c2.url == "https://integrate.api.nvidia.com/v1/chat/completions" and c2.model == ENDPOINTS["hosted"].model
    asyncio.run(c2.aclose())
    run(c, (MSGS, None))
    assert "authorization" not in h.requests[0].headers


@pytest.mark.parametrize("mode,schema,expected", [
    ("json_schema", SCHEMA, {"type": "json_schema", "json_schema": {"name": "rewrites", "schema": SCHEMA}}),
    ("json_schema", None, {"type": "json_object"}),
    ("json_object", SCHEMA, {"type": "json_object"}),
    ("none", SCHEMA, None),
])
def test_response_format_modes(mode, schema, expected):
    h = Script()
    c, _ = make(h, response_format=mode)
    run(c, (MSGS, schema))
    assert h.bodies()[0].get("response_format") == expected


def test_bad_response_format_rejected():
    with pytest.raises(ValueError):
        LightningClient("local", response_format="xml")


def test_429_then_200_honors_retry_after():
    h = Script(httpx.Response(429, headers={"Retry-After": "7", "x-ratelimit-remaining": "0"}, text="slow down"))
    c, sleep = make(h)
    [comp] = run(c, (MSGS, None))
    assert comp.http_attempts == 2 and len(h.requests) == 2
    assert sleep.delays == [7.0]  # jitter cap is 2s, so Retry-After wins
    assert c.stats["retries"] == 1 and c.stats["by_status"] == {"429": 1, "200": 1}
    assert c.stats["rate_limit"]["x-ratelimit-remaining"] == "0"


def test_retry_after_smaller_than_backoff_keeps_backoff():
    h = Script(*[httpx.Response(429, headers={"Retry-After": "1"})] * 3)
    c, sleep = make(h, backoff_base=4.0)
    c._rng = types.SimpleNamespace(uniform=lambda a, b: b)
    run(c, (MSGS, None))
    assert sleep.delays == [4.0, 8.0, 16.0]


def test_exponential_backoff_full_jitter_and_cap():
    h = Script(*[httpx.Response(429)] * 4)
    c, sleep = make(h, backoff_base=2.0, backoff_max=5.0)
    c._rng = types.SimpleNamespace(uniform=lambda a, b: b)  # upper bound of the jitter window
    [comp] = run(c, (MSGS, None))
    assert sleep.delays == [2.0, 4.0, 5.0, 5.0] and comp.http_attempts == 5
    h2 = Script(*[httpx.Response(429)] * 3)
    c2, sleep2 = make(h2)
    run(c2, (MSGS, None))
    assert len(sleep2.delays) == 3 and all(0 <= d <= cap for d, cap in zip(sleep2.delays, (2, 4, 8)))


@pytest.mark.parametrize("status", [500, 502, 503, 408])
def test_retryable_status_then_200(status):
    h = Script(httpx.Response(status, text="upstream error"))
    c, sleep = make(h)
    [comp] = run(c, (MSGS, None))
    assert comp.http_attempts == 2 and len(sleep.delays) == 1
    assert c.stats["by_status"] == {str(status): 1, "200": 1}


@pytest.mark.parametrize("exc", [httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError, httpx.RemoteProtocolError])
def test_transport_exceptions_then_200(exc):
    h = Script(exc, exc)
    c, sleep = make(h)
    [comp] = run(c, (MSGS, None))
    assert comp.http_attempts == 3 and len(sleep.delays) == 2 and comp.text == '{"items": []}'
    assert c.stats["by_status"] == {exc.__name__: 2, "200": 1}


def test_unparseable_200_is_retried():
    h = Script(httpx.Response(200, text="<html>gateway</html>"), httpx.Response(200, json={"choices": []}))
    c, sleep = make(h)
    [comp] = run(c, (MSGS, None))
    assert comp.http_attempts == 3 and len(sleep.delays) == 2


def test_exhausted_retries_raise_transport_error():
    h = Script(*[httpx.Response(503, text="overloaded")] * 10)
    c, sleep = make(h, max_attempts=3)
    err = run(c, (MSGS, None))
    assert isinstance(err, TransportError) and err.attempts == 3
    assert "503" in str(err) and "overloaded" in str(err)
    assert len(h.requests) == 3 and len(sleep.delays) == 2
    assert c.stats["retries"] == 2 and c.stats["transport_errors"] == 1 and c.stats["http_attempts"] == 3


def test_exhausted_timeouts_raise_transport_error():
    h = Script(*[httpx.ReadTimeout] * 6)
    c, _ = make(h)
    err = run(c, (MSGS, None))
    assert isinstance(err, TransportError) and "ReadTimeout" in str(err) and len(h.requests) == 6


def test_401_is_fatal_immediately_and_never_leaks_key():
    h = Script(httpx.Response(401, json={"error": f"invalid credentials: Bearer {KEY}"}))
    c, sleep = make(h, endpoint="hosted", api_key=KEY)
    err = run(c, (MSGS, None))
    assert isinstance(err, FatalHTTPError) and err.status == 401
    assert len(h.requests) == 1 and sleep.delays == [] and c.stats["retries"] == 0
    assert "401" in str(err) and "NVIDIA_API_KEY" in str(err)
    for text in (str(err), repr(err), err.body, repr(c), json.dumps(c.stats)):
        assert KEY not in text


@pytest.mark.parametrize("status,body", [(403, "forbidden"), (404, "model not found"), (422, "messages: field required"),
                                         (400, "maximum context length exceeded")])
def test_other_4xx_fatal_without_retry(status, body):
    h = Script(httpx.Response(status, text=body + "x" * 1000))
    c, sleep = make(h)
    err = run(c, (MSGS, None))
    assert isinstance(err, FatalHTTPError) and err.status == status and len(h.requests) == 1 and sleep.delays == []
    assert body in str(err) and len(err.body) == 300
    assert c.stats["downgrades"] == [] and c.response_format == "json_object"


def test_400_response_format_downgrades_schema_object_none():
    def reject_formats(request):
        if "response_format" in json.loads(request.content):
            return httpx.Response(400, json={"error": {"message": "response_format is not supported by this backend"}})
        return ok()

    h = Script(reject_formats, reject_formats, reject_formats)
    c, sleep = make(h, response_format="json_schema")
    [first, second] = run(c, (MSGS, SCHEMA), (MSGS, SCHEMA))
    formats = [b.get("response_format", {}).get("type") for b in h.bodies()]
    assert formats == ["json_schema", "json_object", None, None]
    assert first.response_format == "none" and first.http_attempts == 3 and second.http_attempts == 1
    assert c.response_format == "none" and sleep.delays == [] and c.stats["retries"] == 0
    assert c.stats["downgrades"] == [{"from": "json_schema", "to": "json_object", "status": 400},
                                     {"from": "json_object", "to": "none", "status": 400}]


def test_400_guided_hint_downgrades_one_level():
    h = Script(httpx.Response(400, text="guided decoding backend failed to compile schema"))
    c, _ = make(h, response_format="json_schema")
    [comp] = run(c, (MSGS, SCHEMA))
    assert [b["response_format"]["type"] for b in h.bodies()] == ["json_schema", "json_object"]
    assert comp.response_format == "json_object" and c.response_format == "json_object"


def test_400_format_hint_at_none_is_fatal():
    h = Script(httpx.Response(400, text="json_object unsupported"))
    c, _ = make(h, response_format="none")
    err = run(c, (MSGS, SCHEMA))
    assert isinstance(err, FatalHTTPError) and len(h.requests) == 1


def test_concurrent_400s_do_not_skip_a_level():
    async def handler(request):
        body = json.loads(request.content)
        await asyncio.sleep(0.005)
        if body.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(400, text="json_schema not supported")
        return ok()

    c, _ = make(handler, response_format="json_schema", concurrency=4)

    async def go():
        try:
            return await asyncio.gather(*[c.complete(MSGS, json_schema=SCHEMA) for _ in range(4)])
        finally:
            await c.aclose()

    comps = asyncio.run(go())
    assert {x.response_format for x in comps} == {"json_object"} and c.response_format == "json_object"
    assert len(c.stats["downgrades"]) == 1


def test_usage_summed_and_missing_usage_is_zero():
    h = Script(ok(usage=(100, 20)), ok(usage=None), ok(usage=(7, 3)))
    c, _ = make(h)
    comps = run(c, (MSGS, None), (MSGS, None), (MSGS, None))
    assert [(x.prompt_tokens, x.completion_tokens) for x in comps] == [(100, 20), (0, 0), (7, 3)]
    assert c.stats["prompt_tokens"] == 107 and c.stats["completion_tokens"] == 23
    assert c.stats["requests"] == 3 and c.stats["http_attempts"] == 3


def test_null_content_reasoning_ignored_and_length_finish():
    def resp(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": None, "reasoning_content": "thinking..."},
                                                      "finish_reason": "length"}]})

    h = Script(resp)
    c, _ = make(h)
    [comp] = run(c, (MSGS, None))
    assert comp.text == "" and comp.finish_reason == "length" and comp.model == "lightning"
    assert comp.prompt_tokens == 0 and comp.completion_tokens == 0


def test_concurrency_semaphore_never_exceeds_limit():
    state = {"inflight": 0, "peak": 0, "calls": 0}

    async def handler(request):
        state["inflight"] += 1
        state["calls"] += 1
        state["peak"] = max(state["peak"], state["inflight"])
        await asyncio.sleep(0.01)
        state["inflight"] -= 1
        return ok()

    c, _ = make(handler, concurrency=3)

    async def go():
        try:
            return await asyncio.gather(*[c.complete(MSGS) for _ in range(12)])
        finally:
            await c.aclose()

    comps = asyncio.run(go())
    assert len(comps) == 12 and state["calls"] == 12 and state["peak"] == 3


def test_retry_after_parsing():
    assert cl._retry_after(None) is None and cl._retry_after("soon") is None
    assert cl._retry_after("2.5") == 2.5 and cl._retry_after("-3") == 0.0
    assert cl._retry_after("99999") == cl.RETRY_AFTER_MAX
    assert 0.0 <= cl._retry_after("Wed, 21 Oct 2015 07:28:00 GMT") <= cl.RETRY_AFTER_MAX


# --- load_api_key -------------------------------------------------------------------------------------------------

ENV = "T3_TEST_LIGHTNING_KEY"


@pytest.fixture
def dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)

    def write(text):
        p = tmp_path / ".env"
        p.write_text(text, encoding="utf-8")
        return p
    return write


@pytest.mark.parametrize("line,expected", [
    (f"{ENV}=abc123", "abc123"),
    (f"export {ENV}=abc123", "abc123"),
    (f"  export   {ENV} = abc123  ", "abc123"),
    (f"{ENV}='quoted value'", "quoted value"),
    (f'{ENV}="double quoted"', "double quoted"),
    (f'{ENV}="with # hash" # comment', "with # hash"),
    (f"{ENV}=abc123 # inline comment", "abc123"),
    (f"{ENV}=abc#123", "abc#123"),
    (f"{ENV}= # just a comment", None),
    (f"{ENV}=", None),
    (f"{ENV}=''", None),
    (f"# {ENV}=commented-out", None),
    (f"OTHER_{ENV}=nope", None),
])
def test_load_api_key_dotenv_parsing(dotenv, line, expected):
    p = dotenv(f"# header\nOTHER=1\n\n{line}\nLAST=2\n")
    assert load_api_key(ENV, p) == expected


def test_load_api_key_last_definition_wins(dotenv):
    assert load_api_key(ENV, dotenv(f"{ENV}=first\n{ENV}=second\n")) == "second"


def test_load_api_key_env_precedence(dotenv, monkeypatch):
    p = dotenv(f"{ENV}=from-file\n")
    monkeypatch.setenv(ENV, "from-env")
    assert load_api_key(ENV, p) == "from-env"
    monkeypatch.setenv(ENV, "   ")  # blank env var falls back to the file
    assert load_api_key(ENV, p) == "from-file"


def test_load_api_key_missing_file(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert load_api_key(ENV, tmp_path / "nope.env") is None


def test_load_api_key_default_path_is_cwd(dotenv, tmp_path, monkeypatch):
    dotenv(f"export {ENV}=cwd-value\n")
    monkeypatch.chdir(tmp_path)
    assert load_api_key(ENV) == "cwd-value"
