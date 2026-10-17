"""Fake Lightning chat-completions server for httpx.MockTransport: plausible rewrite answers plus fault knobs."""

import json
import re

import httpx

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
NUMERIC = re.compile(r"[\d\s.,:%$+\-/]+")
MALFORMED = 'Sure! Here are the rewrites: {"items": [{"k": 1, "keep": tru'


def batch_items(body: dict) -> list[dict]:
    """The {"k", "label", ...} item lines of the last user message (the batch, not the few-shot example)."""
    users = [m for m in body.get("messages", []) if m.get("role") == "user"]
    items = []
    for line in str(users[-1].get("content", "") if users else "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and "k" in item and "label" in item:
            items.append(item)
    return items


def plausible(item: dict) -> dict:
    """What a well-behaved Lightning would answer for one item."""
    label = " ".join(str(item["label"]).split())
    if NUMERIC.fullmatch(label):
        return {"k": item["k"], "keep": False, "drop_reason": "numeric"}
    if label.lower().startswith("automationid"):
        return {"k": item["k"], "keep": False, "drop_reason": "unintelligible"}
    dup = any(str(n.get("label", "")).lower() == label.lower() for n in item.get("neighbors") or ())
    if dup and item.get("style") != "positional":
        return {"k": item["k"], "keep": False, "drop_reason": "duplicate_neighbor"}
    text = " ".join(label.split()[:6])
    style = item.get("style")
    instruction = (f"the {text} button" if style == "descriptive" else
                   f"the {text} at the {item.get('position', 'center')}" if style == "positional" else f"open {text}")
    return {"k": item["k"], "keep": True, "instruction": instruction}


def completion(text: str, *, finish_reason="stop", usage=(900, 120), model=MODEL) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "chatcmpl-fake", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text, "reasoning_content": None},
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": usage[0] + usage[1]}})


def make_handler(behavior="ok", *, malformed_first=0, length_first=0, drop_k=(), fail_first=0, fail_status=429,
                 retry_after=None, invalid_labels=(), usage=(900, 120), model=MODEL):
    """Return handler(request) for httpx.MockTransport. handler.calls records every request (body, headers, items).

    behavior: "ok"; "unauthorized" (always 401); "down" (always 503); "offline" (always raises httpx.ConnectError);
      or callable(request, body, call_index) -> httpx.Response | None (None falls through to the knobs below).
    fail_first: the first N HTTP calls get `fail_status` (429 by default, with Retry-After if given).
    malformed_first / length_first: the first N answered calls return broken JSON / finish_reason="length".
    drop_k: item numbers omitted from multi-item replies (single-row retries answer normally).
    invalid_labels: labels that always get an invalid item (empty instruction), to force llm_error.
    """
    calls, answered = [], [0]
    drop_k, invalid_labels = set(drop_k if not isinstance(drop_k, int) else [drop_k]), set(invalid_labels)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        items = batch_items(body)
        n = len(calls)
        calls.append({"body": body, "headers": dict(request.headers), "items": items, "url": str(request.url)})
        if callable(behavior):
            resp = behavior(request, body, n)
            if resp is not None:
                return resp
        elif behavior == "unauthorized":
            return httpx.Response(401, json={"status": 401, "title": "Unauthorized", "detail": "Invalid API key"})
        elif behavior == "down":
            return httpx.Response(503, text="service unavailable")
        elif behavior == "offline":
            raise httpx.ConnectError("connection refused", request=request)
        if n < fail_first:
            headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
            return httpx.Response(fail_status, headers=headers, json={"status": fail_status, "detail": "slow down"})
        answered[0] += 1
        if answered[0] <= malformed_first:
            return completion(MALFORMED, usage=usage, model=model)
        if answered[0] <= malformed_first + length_first:
            return completion('{"items": [{"k": 1, "keep": true, "instruction": "open', finish_reason="length",
                              usage=usage, model=model)
        out = []
        for item in items:
            if len(items) > 1 and item["k"] in drop_k:
                continue
            if item["label"] in invalid_labels:
                out.append({"k": item["k"], "keep": True, "instruction": ""})
            else:
                out.append(plausible(item))
        return completion(json.dumps({"items": out}), usage=usage, model=model)

    handler.calls = calls
    return handler
