"""Interactive Lightning planner: one chat-completions call per step, tools from agent/prompts.py.

Lightning is text-only: it sees the request, text observations and tool results, never
images. `propose()` gets one reply and never raises; `record_step()` /
`record_correction()` append the exchange to the history.

Retries (the SDK's own are off for the client built here): a timeout is not retried, since
`timeouts.lightning_s` is the whole budget of one call. A 429 waits (Retry-After, else a
growing backoff) inside propose(), so a rate-limit burst on the shared hosted key costs the
run neither a step nor an API error. A 5xx or connection error gets one quick retry.

Context budget (vllm-metal has no prefix cache for Lightning, so prompts stay short):
system + first user message (request + initial observation) + the last
`cfg.history_full_steps` step records verbatim. Older records collapse to one-liners
folded into the first user message ("Earlier steps:"), and the stale initial screen
is then dropped. Each record is self-contained (an assistant message plus one tool
message per tool_call id), so assistant/tool pairing stays valid after collapsing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import openai

from agent import prompts
from agent.config import AgentConfig

log = logging.getLogger(__name__)

INITIAL_SCREEN_OMITTED = "(initial screen omitted; see the latest observation)"
NOT_EXECUTED = "only one action per step; this call was not executed"

RATE_LIMIT_WAIT_S = 15.0       # backoff per consecutive 429 without Retry-After (15, 30, 45, 60 s)
RATE_LIMIT_MAX_WAITS = 4
MAX_RETRY_AFTER_S = 60.0
TRANSIENT_RETRY_S = 2.0        # before the one retry of a 5xx / connection error
SERVER_INFO_TIMEOUT_S = 10.0


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None   # None when raw_arguments is not a JSON object
    raw_arguments: str
    parse_error: str | None = None
    from_content: bool = False  # recovered from text content, not native tool_calls


@dataclass
class PlanResult:
    tool_calls: list[ToolCall]
    content: str
    reasoning: str | None
    thinking: bool
    latency_ms: float
    error: str | None = None
    usage: dict | None = None
    assistant_message: dict = field(default_factory=dict)  # goes back into history (no reasoning)
    finish_reason: str | None = None
    wait_ms: float = 0.0   # rate-limit / retry waits, not included in latency_ms


@dataclass
class _Record:
    messages: list[dict]
    one_line: str
    observation: str | None = None  # the observation text this step's tool result carried


def _new_id() -> str:
    return "call_" + uuid.uuid4().hex[:16]


def parse_arguments(raw: Any) -> tuple[dict | None, str | None]:
    """Tool-call arguments (JSON string or dict) -> (dict, None) or (None, error)."""
    if isinstance(raw, dict):
        return raw, None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as e:
        return None, f"{e}; got {str(raw)[:200]!r}"
    if not isinstance(value, dict):
        return None, f"arguments must be a JSON object, got {type(value).__name__}"
    return value, None


# qwen3_coder XML: <tool_call><function=click><parameter=target_description>X</parameter></function></tool_call>
_XML_BLOCK_RE = re.compile(r"(?:<tool_call>\s*)?<function=([^>\s]+)>(.*?)(?:</function>\s*(?:</tool_call>)?|$)", re.S)
_XML_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.S)
_THINK_RE = re.compile(r"^\s*(?:<think>)?(.*?)</think>\s*", re.S)


def parse_content_tool_calls(content: str) -> tuple[list[ToolCall], str]:
    """Recover tool calls written as text (qwen3_coder XML or a JSON {"name", "arguments"} object).

    Returns (calls, remaining_content). Used only when the reply has no native tool_calls.
    """
    if not content:
        return [], content
    calls: list[ToolCall] = []
    if "<function=" in content:
        for m in _XML_BLOCK_RE.finditer(content):
            args = {k: v.strip() for k, v in _XML_PARAM_RE.findall(m.group(2))}
            calls.append(ToolCall(_new_id(), m.group(1), args, json.dumps(args), from_content=True))
        if calls:
            return calls, _XML_BLOCK_RE.sub("", content).replace("</tool_call>", "").strip()

    decoder = json.JSONDecoder()
    spans: list[tuple[int, int]] = []
    pos = content.find("{")
    while pos != -1:
        try:
            obj, end = decoder.raw_decode(content, pos)
        except ValueError:
            pos = content.find("{", pos + 1)
            continue
        if isinstance(obj, dict) and isinstance(obj.get("function"), dict):
            obj = obj["function"]
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and ("arguments" in obj or "parameters" in obj):
            raw = obj.get("arguments", obj.get("parameters"))
            args, err = parse_arguments(raw)
            raw_s = raw if isinstance(raw, str) else json.dumps(raw)
            calls.append(ToolCall(_new_id(), obj["name"], args, raw_s, parse_error=err, from_content=True))
            spans.append((pos, end))
            pos = content.find("{", end)
        else:  # not a call itself; one may be nested inside it
            pos = content.find("{", pos + 1)
    if not calls:
        return [], content
    rest = content
    for start, end in reversed(spans):
        rest = rest[:start] + rest[end:]
    return calls, re.sub(r"</?tool_call>", "", rest).strip()


def _field(obj: Any, name: str) -> Any:
    """Attribute or pydantic extra field (servers add reasoning fields the SDK doesn't declare)."""
    value = getattr(obj, name, None)
    if value is None:
        value = (getattr(obj, "model_extra", None) or {}).get(name)
    return value


def _usage(resp: Any) -> dict | None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return dict(usage) if isinstance(usage, dict) else None


def _history_call(c: ToolCall) -> dict:
    # Servers json-parse history arguments for the chat template, so never send back broken JSON.
    if c.arguments is None:
        args = "{}"
    elif c.from_content:
        args = json.dumps(c.arguments)
    else:
        args = c.raw_arguments or "{}"
    return {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": args}}


def _retry_after_s(e: openai.APIStatusError) -> float | None:
    try:
        value = e.response.headers.get("retry-after")
        return max(0.0, float(value)) if value else None
    except (AttributeError, TypeError, ValueError):  # absent, or an HTTP date
        return None


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


class LightningPlanner:
    def __init__(self, cfg: AgentConfig, client: openai.AsyncOpenAI | None = None):
        self.cfg = cfg
        self._owns_client = client is None
        self._secret: str | None = None
        if client is None:
            key = cfg.lightning.api_key
            self._secret = key if key != "EMPTY" else None
            client = openai.AsyncOpenAI(base_url=cfg.lightning.base_url, api_key=key,
                                        timeout=cfg.timeouts.lightning_s, max_retries=0)  # see propose()
        self.client = client
        self.send_parallel_tool_calls = True  # dropped for the session if the server rejects it
        self.tool_choice = cfg.tool_choice     # "required" by default; "auto" if the server rejects it
        self.request: str = ""
        self.initial_observation: str = ""
        self.records: list[_Record] = []

    def start(self, request: str, observation_text: str) -> None:
        self.request, self.initial_observation, self.records = request, observation_text, []

    # --- history -------------------------------------------------------------------------

    def messages(self) -> list[dict]:
        """System + first user + the last history_full_steps records (older ones as one-liners)."""
        keep = max(0, self.cfg.history_full_steps)
        older = self.records[:-keep] if keep else list(self.records)
        recent = self.records[len(older):]
        if not older:
            first = prompts.initial_user_message(self.request, self.initial_observation)
        else:
            screen = INITIAL_SCREEN_OMITTED
            if not any(r.observation for r in recent):  # e.g. only corrections kept: keep the latest screen
                screen = next((r.observation for r in reversed(older) if r.observation), self.initial_observation)
            lines = "\n".join(f"{i}. {r.one_line}" for i, r in enumerate(older, 1))
            first = prompts.initial_user_message(self.request, screen) + f"\n\nEarlier steps:\n{lines}"
        out = [{"role": "system", "content": prompts.SYSTEM_PROMPT}, {"role": "user", "content": first}]
        for r in recent:
            out.extend(r.messages)
        return out

    def record_step(self, plan: PlanResult, tool_results: list[tuple[str, dict]], one_line: str) -> None:
        """Append the assistant message and one tool message per tool_call id (missing ids get NOT_EXECUTED)."""
        assistant = plan.assistant_message or {"role": "assistant", "content": plan.content or ""}
        by_id = dict(tool_results)
        msgs = [assistant]
        observation = None
        for tc in assistant.get("tool_calls") or []:
            payload = by_id.get(tc["id"], {"ok": False, "error": NOT_EXECUTED})
            if observation is None and payload.get("observation"):
                observation = payload["observation"]
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(payload, ensure_ascii=False)})
        self.records.append(_Record(msgs, one_line, observation))

    def record_correction(self, plan: PlanResult, text: str) -> None:
        """A reply without a tool call: keep its text and answer with a corrective user message."""
        content = plan.content or ""
        msgs = [{"role": "assistant", "content": content or "(empty reply)"}, {"role": "user", "content": text}]
        if plan.finish_reason == "length":
            line = "(no tool call) reply cut off at the token limit -> asked to call a tool"
        else:
            line = f'(no tool call) replied "{" ".join(content.split())[:80]}" -> asked to call a tool'
        self.records.append(_Record(msgs, line))

    # --- one call ------------------------------------------------------------------------

    def _kwargs(self, thinking: bool) -> dict:
        s = self.cfg.sampling
        kwargs: dict[str, Any] = dict(
            model=self.cfg.lightning.model, messages=self.messages(), tools=prompts.chat_tools(),
            tool_choice=self.tool_choice, temperature=s.temperature, top_p=s.top_p,
            max_tokens=s.thinking_max_tokens if thinking else s.max_tokens,  # reasoning counts against it
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        if self.send_parallel_tool_calls:
            kwargs["parallel_tool_calls"] = False
        return kwargs

    async def _create(self, kwargs: dict) -> Any:
        try:
            return await self.client.chat.completions.create(**kwargs)
        except openai.APIStatusError as e:
            if e.status_code not in (400, 422):
                raise
            if self.send_parallel_tool_calls and "parallel_tool_calls" in str(e):
                self.send_parallel_tool_calls = False
                kwargs.pop("parallel_tool_calls", None)
            elif self.tool_choice != "auto" and "tool_choice" in str(e):
                log.warning("server rejected tool_choice=%r; using 'auto' for this run", self.tool_choice)
                self.tool_choice = kwargs["tool_choice"] = "auto"
            else:
                raise
            return await self.client.chat.completions.create(**kwargs)

    async def propose(self, thinking: bool) -> PlanResult:
        """One Lightning reply. Never raises: API and parsing problems come back as PlanResult.error.

        Rate-limit waits and the one retry of a transient error happen in here (module
        docstring); `latency_ms` excludes those waits, which are in `wait_ms`.
        """
        t0 = time.perf_counter()
        waited = 0.0
        n_rate_limited = n_transient = 0
        kwargs: dict | None = None
        while True:
            try:
                kwargs = kwargs or self._kwargs(thinking)
                plan = self._parse(await self._create(kwargs), thinking, _ms(t0) - waited)
                plan.wait_ms = waited
                return plan
            except openai.RateLimitError as e:
                if n_rate_limited >= RATE_LIMIT_MAX_WAITS:
                    return self._error(e, thinking, t0, waited)
                n_rate_limited += 1
                wait = _retry_after_s(e)
                wait = min(RATE_LIMIT_WAIT_S * n_rate_limited if wait is None else wait, MAX_RETRY_AFTER_S)
                wait *= random.uniform(1.0, 1.2)  # several runs may share one key
                log.warning("Lightning rate-limited (429); waiting %.0f s (%d/%d)", wait, n_rate_limited,
                            RATE_LIMIT_MAX_WAITS)
            except openai.APITimeoutError as e:  # the whole per-call budget is spent
                return self._error(e, thinking, t0, waited)
            except (openai.APIConnectionError, openai.InternalServerError) as e:
                if n_transient >= 1:
                    return self._error(e, thinking, t0, waited)
                n_transient += 1
                wait = TRANSIENT_RETRY_S
                log.warning("Lightning call failed (%s); retrying once", type(e).__name__)
            except Exception as e:
                return self._error(e, thinking, t0, waited)
            t_wait = time.perf_counter()
            await asyncio.sleep(wait)
            waited += _ms(t_wait)

    def _error(self, e: Exception, thinking: bool, t0: float, waited: float) -> PlanResult:
        return PlanResult(tool_calls=[], content="", reasoning=None, thinking=thinking, latency_ms=_ms(t0) - waited,
                          error=self._describe(e), wait_ms=waited)

    async def server_info(self) -> dict:
        """What the endpoint serves as cfg.lightning.model, for the trace: {"root": <checkpoint>}.

        vLLM's /v1/models `root` is the loaded checkpoint, so it tells a community quantization
        from the official BF16 weights when both are served as `lightning`. Endpoints without a
        root (hosted) give the model id. Never raises: {"root": None, "server_error": ...}.
        """
        model = self.cfg.lightning.model
        try:
            client = self.client.with_options(timeout=SERVER_INFO_TIMEOUT_S, max_retries=0)
            page = await asyncio.wait_for(client.models.list(), SERVER_INFO_TIMEOUT_S + 1)
            for m in page.data:
                if m.id == model:
                    return {"root": _field(m, "root") or m.id}
            return {"root": None, "server_error": f"{model!r} is not in the server's /models list"}
        except Exception as e:
            return {"root": None, "server_error": self._describe(e)[:200]}

    def _describe(self, e: Exception) -> str:
        msg = f"{type(e).__name__}: {e}"
        if self._secret:
            msg = msg.replace(self._secret, "***")
        return msg[:500]

    def _parse(self, resp: Any, thinking: bool, latency_ms: float) -> PlanResult:
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return PlanResult([], "", None, thinking, latency_ms, error="empty response (no choices)", usage=_usage(resp))
        choice = choices[0]
        msg = choice.message
        content = msg.content or ""
        reasoning = _field(msg, "reasoning_content") or _field(msg, "reasoning")
        if "</think>" in content:  # server without a reasoning parser
            m = _THINK_RE.match(content)
            if m:
                reasoning = reasoning or m.group(1).strip()
                content = content[m.end():]

        calls: list[ToolCall] = []
        seen: set[str] = set()
        for tc in msg.tool_calls or []:
            fn = tc.function
            raw = fn.arguments if isinstance(fn.arguments, str) else json.dumps(fn.arguments or {})
            args, err = parse_arguments(fn.arguments)
            cid = tc.id if tc.id and tc.id not in seen else _new_id()
            seen.add(cid)
            calls.append(ToolCall(cid, fn.name or "", args, raw or "", parse_error=err))
        if not calls:
            calls, content = parse_content_tool_calls(content)

        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            assistant["tool_calls"] = [_history_call(c) for c in calls]
        return PlanResult(tool_calls=calls, content=content, reasoning=reasoning, thinking=thinking,
                          latency_ms=latency_ms, usage=_usage(resp), assistant_message=assistant,
                          finish_reason=getattr(choice, "finish_reason", None))

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.close()
