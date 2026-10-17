"""Fake Lightning chat-completions server for httpx.MockTransport: plausible (v3-valid) rewrite answers plus fault
knobs."""

import json
import re

import httpx

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
OFFICIAL_ROOT = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
COMMUNITY_ROOT = "mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"
SERVED_IDS = ("lightning", MODEL)
NUMERIC = re.compile(r"[\d\s.,:%$+\-/]+")
MALFORMED = 'Sure! Here are the rewrites: {"items": [{"k": 1, "keep": tru'
# A pre-v3 neighbor's "direction" (where the NEIGHBOR is) as the v3 "target_is" (where the TARGET is).
TARGET_IS = {"left": "right of", "right": "left of", "above": "below", "below": "above", "overlapping": "overlapping"}
PLAIN_POSITION = {"top-left": "at the top left", "top-center": "at the top", "top-right": "at the top right",
                  "middle-left": "on the left", "center": "in the middle", "middle-right": "on the right",
                  "bottom-left": "at the bottom left", "bottom-center": "at the bottom",
                  "bottom-right": "at the bottom right"}


def _neighbors(value) -> list[dict]:
    """Item neighbors as [{"label", "target_is"}]; tolerates junk entries and the pre-v3 "direction" key."""
    out = []
    for n in value if isinstance(value, list) else ():
        if isinstance(n, dict) and isinstance(n.get("label"), str):
            rel = n.get("target_is") or TARGET_IS.get(str(n.get("direction")), "overlapping")
            out.append({"label": n["label"], "target_is": rel})
    return out


def batch_items(body: dict) -> list[dict]:
    """The {"k", "label", ...} item lines of the last user message (the batch, not the few-shot example). Lines that
    are not item objects (prose, an inline JSON example, a non-integer k) are skipped; neighbors are normalized."""
    users = [m for m in body.get("messages", []) if isinstance(m, dict) and m.get("role") == "user"]
    content = users[-1].get("content", "") if users else ""
    if isinstance(content, list):  # OpenAI content parts
        content = "".join(p.get("text") or "" for p in content if isinstance(p, dict))
    items, seen = [], set()
    for line in str(content).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = item.get("k") if isinstance(item, dict) else None
        if isinstance(k, int) and not isinstance(k, bool) and k not in seen and isinstance(item.get("label"), str):
            seen.add(k)
            items.append({**item, "neighbors": _neighbors(item.get("neighbors"))})
    return items


def plausible(item: dict) -> dict:
    """What a well-behaved Lightning would answer for one item under the v3 rules: keeps a word of its own label,
    element nouns only from control (text/area: "text"; none when control is missing), neutral "click" verb, plain-word
    positions (never grid tokens), and identical-label neighbors told apart by position or dropped."""
    label = " ".join(str(item["label"]).split())
    if NUMERIC.fullmatch(label):
        return {"k": item["k"], "keep": False, "drop_reason": "numeric"}
    if label.lower().startswith("automationid"):
        return {"k": item["k"], "keep": False, "drop_reason": "unintelligible"}
    words = lambda s: re.findall(r"[^\W_]+", s.lower())  # noqa: E731
    others = [words(n["label"]) for n in _neighbors(item.get("neighbors"))]
    dup = words(label) in others
    lookalike = dup or any(o[:1] == words(label)[:1] for o in others if o)  # "Search" vs "Search 14"
    style = item.get("style")
    if dup and style != "positional":
        return {"k": item["k"], "keep": False, "drop_reason": "duplicate_neighbor"}
    text = " ".join(label.rstrip("…").split()[:5])
    control = item.get("control") if isinstance(item.get("control"), str) else None
    where = PLAIN_POSITION.get(item.get("position"), "in the middle")
    if style == "descriptive":
        instruction = f"the {text} {control}" if control and control != "element" else f"the {text} option"
    elif style == "positional":
        instruction = f"the {text}"
    else:
        instruction = f"click {text}"
    if style == "positional" or lookalike:  # a position tells look-alikes apart
        instruction += f" {where}"
    return {"k": item["k"], "keep": True, "instruction": instruction}


def completion(text: str, *, finish_reason="stop", usage=(900, 120), model=MODEL) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "chatcmpl-fake", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text, "reasoning_content": None},
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": usage[0] + usage[1]}})


def models_list(root=OFFICIAL_ROOT, ids=SERVED_IDS) -> dict:
    """A vLLM-style GET /models body; root=None leaves the "root" key out (as hosted APIs do)."""
    return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "vllm",
                                        **({"root": root} if root is not None else {})} for i in ids]}


def make_handler(behavior="ok", *, malformed_first=0, length_first=0, drop_k=(), fail_first=0, fail_status=429,
                 retry_after=None, invalid_labels=(), usage=(900, 120), model=MODEL, served_root=OFFICIAL_ROOT,
                 models_status=200):
    """Return handler(request) for httpx.MockTransport. handler.calls records every chat request (body, headers,
    items); GET .../models is answered separately (served_root, models_status) and recorded in handler.model_calls.

    behavior: "ok"; "unauthorized" (always 401); "down" (always 503); "offline" (always raises httpx.ConnectError);
      or callable(request, body, call_index) -> httpx.Response | None (None falls through to the knobs below).
    fail_first: the first N HTTP calls get `fail_status` (429 by default, with Retry-After if given).
    malformed_first / length_first: the first N answered calls return broken JSON / finish_reason="length".
    drop_k: item numbers omitted from multi-item replies (single-row retries answer normally).
    invalid_labels: labels that always get an invalid item (empty instruction), to force llm_error.
    """
    calls, model_calls, answered = [], [], [0]
    drop_k, invalid_labels = set(drop_k if not isinstance(drop_k, int) else [drop_k]), set(invalid_labels)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/models"):
            model_calls.append({"headers": dict(request.headers), "url": str(request.url)})
            if models_status != 200:
                return httpx.Response(models_status, json={"detail": "no models for you"})
            return httpx.Response(200, json=models_list(served_root))
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

    handler.calls, handler.model_calls = calls, model_calls
    return handler
