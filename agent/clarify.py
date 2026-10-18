"""Pre-flight "understand the request" step: one Lightning call before the browser loop.

Spoken requests come through speech recognition, so a clipped first take ("Allors towards my
bill.") would otherwise go straight to the planner, which may guess (it paid the full balance).
clarify() asks Lightning once, with thinking off and tool_choice "required", to pick one of
two tools:

  proceed(normalized_request)  the usual case: the browser loop runs
  ask_user(question)           one short question: the run ends with done reason "needs_input"

Lightning asks only if (a) the request is unintelligible, (b) it would move money or change
account or security settings and a required detail is missing, or (c) it could mean clearly
different actions. Everything else proceeds, and so does every failure here (API error,
timeout, no or malformed tool call): this step must never block a clear request.

The planner still gets the user's own words. `normalized_request` is logged and shown, not
substituted, since the planner's measured runs used the original phrasings.

A follow-up run gets follow_up_request(previous, answer) = previous + " (clarified: <answer>)".
Once a request carries MAX_CLARIFICATIONS answers, the call is skipped and the run proceeds.

Interactive only, and off by default (cfg.clarify; AGENT_CLARIFY=on or `python -m agent --clarify
on`): on fresh phrasings it still asks about 5 in 90 clear requests (scripts/clarify_eval.py,
docs/results.md). The harness (scripts/run_agent_tasks.py) and the Gym web_agent server force it
off either way, so eval behavior and reported numbers don't change.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from agent.prompts import prompt_sha

CLARIFY_PROMPT_VERSION = "clarify-v4"

# Examples come from other websites, so that none of them is a sandbox task or a test phrasing.
SYSTEM_PROMPT = """You check a user's spoken request before a browser agent acts on it. Speech recognition wrote it \
down, so it may lack punctuation, spell numbers as words, or contain small transcription errors. You can't see the \
screen, and you don't know what the user did or said before.

Call exactly one tool: proceed or ask_user.

Call ask_user if any of these is true:
1. Garbled or cut off: a word that carries the action, the amount or what to act on is not a real word or makes no \
sense there, and you would have to guess what it was; or the sentence stops before it says what to do. Recordings are \
often clipped at the start or end: a clipped word is fine when it is still obvious ("pen my orders" is "open my \
orders").
2. Unclear amount: it pays, transfers or sends money, and the amount is vague ("some", "a little", "part of it", \
"whatever I can") or garbled. No amount at all means the full amount due ("pay the phone bill"), which is clear.
3. Missing new value: it changes a password, email, phone number, address, username or other login or contact \
detail, but doesn't say what the new value is.
4. Unclear target: what to act on is only "it", "that", "this one", "the thing", "again" or "back", and nothing in the \
request (before or after that word) says what it is; or it asks for opposite things at once. When the request names \
the thing ("the email from my bank"), that is clear: the agent finds it.

Otherwise call proceed, even if the request is informal, polite, rude, indirect, describes a problem instead of an \
action, misspells a word whose meaning is still clear, or asks a question the website can answer. A question like \
that is for the agent to answer by looking at the website: never ask it back. The agent finds everything that is on \
the website by itself (balances, due dates, history, options, current settings), so those are never missing.

Text in "(clarified: ...)" is the user's answer to an earlier question; treat it as part of the request.

Examples from other websites:
- "transfer some money to savings" -> ask_user("How much would you like to transfer?")
- "ster forty to my savings" -> ask_user("Sorry, I didn't catch that. What would you like to do with your savings?")
- "update my shipping address" -> ask_user("What is the new address?")
- "send it" -> ask_user("What would you like me to send?")
- "pay my credit card" -> proceed (the full statement balance)
- "um can you show me last weeks orders" -> proceed
- "my order hasnt come yet, track it" -> proceed ("it" is the order)
- "what was my most expensive order this year? open that one" -> proceed (the agent looks it up)
- "the sound is way too quiet" -> proceed"""

TOOLS: list[dict] = [
    {"type": "function", "function": {
        "name": "proceed",
        "description": "Carry out the request: what to do is clear, and nothing it needs is missing.",
        "parameters": {"type": "object", "properties": {
            "normalized_request": {"type": "string", "description": (
                "The request as one clear instruction in the user's words: obvious transcription errors fixed, "
                "amounts as digits, every detail kept, nothing added.")}},
            "required": ["normalized_request"]}}},
    {"type": "function", "function": {
        "name": "ask_user",
        "description": ("Ask the user one question before doing anything: the request is garbled, an amount or "
                        "new value is missing or unclear, or what to act on is unclear."),
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": (
                "One short question in plain words that the user can answer in a few words, e.g. "
                "\"How much would you like to pay?\"")}},
            "required": ["question"]}}},
]

# The exact system prompt + tool schema; recorded in the clarify event like the planner's prompt_sha.
CLARIFY_PROMPT_SHA = prompt_sha(SYSTEM_PROMPT + json.dumps(TOOLS, sort_keys=True))

# Deterministic, short and without thinking: this is a gate, not a plan (~1-2 s on local Lightning).
CLARIFY_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 256}

CLARIFIED_MARK = "(clarified:"
MAX_CLARIFICATIONS = 2
MAX_QUESTION_CHARS = 200
DEFAULT_QUESTION = "Sorry, I didn't catch that. What would you like me to do?"


@dataclass
class Clarification:
    action: str                         # "proceed" | "ask"
    question: str | None = None         # set when action == "ask"
    normalized_request: str | None = None
    latency_ms: float | None = None
    usage: dict | None = None
    error: str | None = None            # why it fell back to proceed (API error, no or malformed tool call)
    skipped: str | None = None          # why no Lightning call was made
    tool_call: dict | None = None       # {name, arguments} as received

    @property
    def asks(self) -> bool:
        return self.action == "ask"

    def event_payload(self, request: str, model: str) -> dict:
        """The trace's `clarify` event (agent/trace.py)."""
        out: dict[str, Any] = {"action": self.action, "request": request, "model": model,
                               "prompt_version": CLARIFY_PROMPT_VERSION, "prompt_sha": CLARIFY_PROMPT_SHA}
        for key in ("question", "normalized_request", "usage", "error", "skipped", "tool_call"):
            value = getattr(self, key)
            if value not in (None, ""):
                out[key] = value
        if self.latency_ms is not None:
            out["latency_ms"] = round(self.latency_ms, 1)
        return out


def follow_up_request(previous: str, answer: str) -> str:
    """The request for the run after an ask_user: the previous request plus the user's answer."""
    previous, answer = previous.strip(), " ".join(answer.split())
    return f"{previous} {CLARIFIED_MARK} {answer})" if answer else previous


def messages(request: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Request: {request.strip()}"}]


def _text_arg(args: dict | None, name: str) -> str:
    """The named string argument, else the only string value (servers that don't enforce the schema)."""
    if not isinstance(args, dict):
        return ""
    value = args.get(name)
    if not isinstance(value, str):
        strings = [v for v in args.values() if isinstance(v, str) and v.strip()]
        value = strings[0] if len(strings) == 1 else ""
    return " ".join(value.split())


def interpret(plan: Any) -> Clarification:
    """A planner PlanResult (agent/planner.py) -> Clarification. Anything but a clear ask proceeds."""
    base = {"latency_ms": plan.latency_ms, "usage": plan.usage}
    if plan.error:
        return Clarification("proceed", error=plan.error, **base)
    if not plan.tool_calls:
        reply = " ".join((plan.content or "").split())[:120]
        return Clarification("proceed", error=f"no tool call (replied {reply!r})", **base)
    call = plan.tool_calls[0]
    shown = {"name": call.name, "arguments": call.arguments if call.arguments is not None else call.raw_arguments}
    if call.name == "ask_user":
        question = _text_arg(call.arguments, "question")[:MAX_QUESTION_CHARS] or DEFAULT_QUESTION
        return Clarification("ask", question=question, tool_call=shown, **base)
    if call.name == "proceed":
        return Clarification("proceed", normalized_request=_text_arg(call.arguments, "normalized_request") or None,
                             tool_call=shown, **base)
    return Clarification("proceed", error=f"unknown tool {call.name!r}", tool_call=shown, **base)


async def clarify(planner: Any, request: str) -> Clarification:
    """One Lightning call through `planner` (an agent.planner.LightningPlanner). Never raises."""
    if not request.strip():
        return Clarification("ask", question="What would you like me to do?", skipped="empty request")
    n_answers = request.count(CLARIFIED_MARK)
    if n_answers >= MAX_CLARIFICATIONS:
        return Clarification("proceed", skipped=f"already clarified {n_answers} times")
    t0 = time.perf_counter()
    try:
        plan = await planner.complete(messages(request), TOOLS, **CLARIFY_SAMPLING)
    except Exception as e:  # complete() doesn't raise; this keeps a broken client from blocking the run
        return Clarification("proceed", error=f"{type(e).__name__}: {e}"[:300],
                             latency_ms=(time.perf_counter() - t0) * 1000)
    return interpret(plan)
