"""Test-only scripted stand-in for Lightning's OpenAI-compatible chat-completions API.

No network: `FakeLightning(...).client` is a real `openai.AsyncOpenAI` whose HTTP
transport is an in-process `httpx.MockTransport`, so the planner's request building and
the SDK's response parsing both run for real. Every request body is kept in `.requests`
so tests can assert tools, sampling, `chat_template_kwargs.enable_thinking` and the
message history.

    fake = FakeLightning([Reply([click("the gear icon in the top-right toolbar")]),
                          Reply([("click", '{"target_description": ')]),  # malformed JSON, sent verbatim
                          done("Turned on high contrast.")])
    planner_client = fake.client

A bare tuple in the script is shorthand for a reply with that one tool call. When the
script runs out, the reply is done(summary="script exhausted", success=False).
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx
import openai

BASE_URL = "http://fake-lightning.test/v1"

ToolCall = tuple[str, dict | str]


@dataclass
class Reply:
    """One scripted assistant turn.

    A str argument is sent verbatim as the `arguments` string (lets tests send malformed
    JSON); a dict is JSON-encoded. `reasoning` comes back as `message.reasoning_content`.
    `finish_reason` defaults to "tool_calls" if there are tool calls, else "stop".
    `status` != 200 returns an OpenAI-style error body instead (the SDK raises APIStatusError),
    with `headers` (e.g. {"retry-after": "3"} on a 429).
    """
    tool_calls: list[ToolCall] = field(default_factory=list)
    content: str = ""
    reasoning: str | None = None
    finish_reason: str | None = None
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)


# --- terse script helpers ------------------------------------------------------------------

def click(target: str) -> ToolCall:
    return ("click", {"target_description": target})


def type_text(text: str, target: str | None = None) -> ToolCall:
    args: dict = {"text": text}
    if target is not None:
        args["target_description"] = target
    return ("type_text", args)


def press_key(key: str) -> ToolCall:
    return ("press_key", {"key": key})


def scroll(direction: str = "down", amount: str = "page") -> ToolCall:
    return ("scroll", {"direction": direction, "amount": amount})


def done(summary: str, success: bool = True) -> ToolCall:
    return ("done", {"summary": summary, "success": success})


def say(content: str, reasoning: str | None = None) -> Reply:
    """A plain text reply with no tool call (finish_reason "stop")."""
    return Reply(content=content, reasoning=reasoning)


# --- the stub ------------------------------------------------------------------------------

class FakeLightning:
    """Scripted chat-completions server. `script` is a list of replies (or tool-call tuples),
    or a callable that gets each parsed request body and returns the Reply."""

    def __init__(self, script: list[Reply | ToolCall] | Callable[[dict], Reply], model: str = "fake-lightning",
                 root: str = "fake/Lightning-test-checkpoint"):
        self.script = script
        self.model = model
        self.root = root      # GET /v1/models lists `model` with this vLLM-style checkpoint root
        self.requests: list[dict] = []   # chat-completions bodies only
        self._next = 0
        self._ids = itertools.count(1)
        # max_retries=0: one SDK call == one request == one script entry, even for error replies.
        self.client = openai.AsyncOpenAI(
            base_url=BASE_URL, api_key="fake", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self._handle)),
        )

    @property
    def last_request(self) -> dict:
        return self.requests[-1]

    @property
    def exhausted(self) -> bool:
        return not callable(self.script) and self._next >= len(self.script)

    def _reply_for(self, body: dict) -> Reply:
        if callable(self.script):
            reply = self.script(body)
        elif self._next < len(self.script):
            reply = self.script[self._next]
            self._next += 1
        else:
            reply = Reply([done("script exhausted", success=False)])
        return Reply([reply]) if isinstance(reply, tuple) else reply

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": self.model, "object": "model", "created": 0, "owned_by": "fake", "root": self.root}]})
        if request.method != "POST" or not request.url.path.endswith("/chat/completions"):
            return _error(404, f"fake lightning only serves POST /v1/chat/completions and GET /v1/models, got "
                               f"{request.method} {request.url.path}")
        try:
            body = json.loads(request.content or b"{}")
        except ValueError as e:
            return _error(400, f"request body is not JSON: {e}")
        self.requests.append(body)
        if body.get("stream"):
            return _error(400, "fake lightning does not support stream=true")
        reply = self._reply_for(body)
        if reply.status != 200:
            resp = _error(reply.status, reply.content or f"scripted error {reply.status}")
            resp.headers.update(reply.headers)
            return resp
        return httpx.Response(200, json=self._completion(body, reply))

    def _completion(self, body: dict, reply: Reply) -> dict:
        n = next(self._ids)
        tool_calls = [
            {"id": f"call_{n}_{i}", "type": "function",
             "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}
            for i, (name, args) in enumerate(reply.tool_calls)
        ]
        # Like vLLM: content is null when the turn is only tool calls.
        content = reply.content or (None if tool_calls else "")
        completion_chars = len(content or "") + sum(len(tc["function"]["arguments"]) for tc in tool_calls)
        prompt_tokens = len(json.dumps(body.get("messages", []))) // 4
        completion_tokens = completion_chars // 4 + 1
        return {
            "id": f"chatcmpl-fake-{n}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model") or self.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content, "tool_calls": tool_calls,
                            "reasoning_content": reply.reasoning},
                "logprobs": None,
                "finish_reason": reply.finish_reason or ("tool_calls" if tool_calls else "stop"),
            }],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
        }


def _error(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message, "type": "fake_lightning_error",
                                                  "code": status}})
