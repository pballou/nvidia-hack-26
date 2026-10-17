"""Parse Lightning's batched rewrite output and lint every item against the metadata it was given."""

import json
import re

STYLES = ("functional", "descriptive", "positional")
DROP_REASONS = ("numeric", "unintelligible", "needs_screen", "duplicate_neighbor")
LLM_ERROR = "llm_error"
DROP_REASON_ALIASES = {
    "number": "numeric", "numbers": "numeric", "numerical": "numeric", "digits": "numeric", "number_only": "numeric",
    "numeric_only": "numeric", "numeric_label": "numeric",
    "nonsense": "unintelligible", "gibberish": "unintelligible", "junk": "unintelligible", "unclear": "unintelligible",
    "unreadable": "unintelligible", "meaningless": "unintelligible", "garbage": "unintelligible",
    "not_intelligible": "unintelligible", "unintelligible_label": "unintelligible",
    "needs_visual": "needs_screen", "ambiguous": "needs_screen", "needs_image": "needs_screen",
    "needs_context": "needs_screen", "needs_screenshot": "needs_screen", "requires_screen": "needs_screen",
    "visual": "needs_screen", "needs_visual_context": "needs_screen",
    "duplicate": "duplicate_neighbor", "duplicate_label": "duplicate_neighbor", "duplicated": "duplicate_neighbor",
    "duplicates_neighbor": "duplicate_neighbor", "neighbor_duplicate": "duplicate_neighbor",
    "not_unique": "duplicate_neighbor",
}
COLOR_WORDS = frozenset("red green blue yellow orange purple pink black white gray grey brown cyan magenta violet teal "
                        "golden silver".split())
SHAPE_WORDS = frozenset("gear cog magnifying magnifier hamburger kebab star heart bell trash pencil arrow chevron "
                        "circle circular square triangle triangular round rounded oval plus checkmark tick".split())
MIN_WORDS, MAX_WORDS = 2, 12
TARGET_WORDS = (3, 8)

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```[\w-]*[ \t]*\n?(.*?)```", re.S)
_QUOTED = re.compile(r'"([^"\n]+)"|“([^”\n]+)”|(?<!\w)\'([^\'\n]+)\'(?!\w)|(?<!\w)‘([^’\n]+)’(?!\w)')
_DECODER = json.JSONDecoder(strict=False)


def _word_re(words) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(sorted(words, key=len, reverse=True)) + r")s?\b", re.I)


_COLOR_RE, _SHAPE_RE = _word_re(COLOR_WORDS), _word_re(SHAPE_WORDS)


def parse_json_payload(text: str) -> object:
    """JSON from a model reply, tolerating <think> blocks, markdown fences and prose around the outermost {}/[]."""
    if not isinstance(text, str):
        raise ValueError(f"expected text, got {type(text).__name__}")
    text = re.split(r"</think>", _THINK.sub("", text), flags=re.I)[-1].strip()
    if not text:
        raise ValueError("empty response")
    for chunk in (text, *(m.group(1).strip() for m in _FENCE.finditer(text))):
        try:
            return _DECODER.decode(chunk)
        except ValueError:
            pass
    found = []
    for opener, closer in ("{}", "[]"):
        start, end = text.find(opener), text.rfind(closer)
        if start < 0:
            continue
        try:
            obj, stop = _DECODER.raw_decode(text, start)
            found.append((stop - start, obj))
        except ValueError:
            pass
        if end > start:
            try:
                found.append((end + 1 - start, _DECODER.decode(text[start:end + 1])))
            except ValueError:
                pass
    if not found:
        raise ValueError("no complete JSON object or array found")
    return max(found, key=lambda t: t[0])[1]


def extract_items(payload) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("items"), list):
            return payload["items"]
        if "k" in payload and "keep" in payload:
            return [payload]
    raise ValueError('expected {"items": [...]} or a JSON array of items')


def normalize_drop_reason(value) -> str | None:
    if not isinstance(value, str):
        return None
    key = re.sub(r"[\s-]+", "_", value.strip().lower())
    key = DROP_REASON_ALIASES.get(key, key)
    return key if key in DROP_REASONS else None


def normalize_instruction(text: str) -> str:
    """Whitespace-normalize, unwrap one pair of surrounding quotes, strip trailing periods."""
    s = " ".join(text.split())
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s.rstrip(".").rstrip()


def _keep(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def _proper_tokens(label: str) -> list[str]:
    """Label tokens that look like names: internal capital (GitHub), digit+letter (MP3, 4K), ALL-CAPS (USB)."""
    out = []
    for tok in re.findall(r"[A-Za-z0-9]+", label):
        if (tok[1:] != tok[1:].lower() or (re.search(r"\d", tok) and re.search(r"[A-Za-z]", tok))) and tok not in out:
            out.append(tok)
    return out


def _lint(s: str, meta: dict) -> tuple[str | None, list[str]]:
    label = meta.get("label") or ""
    neighbor_labels = [str(n.get("label") or "") for n in meta.get("neighbors") or () if isinstance(n, dict)]
    context = " ".join([label, str(meta.get("control") or ""), *neighbor_labels])
    n_words = len(s.split())
    error = None
    if not MIN_WORDS <= n_words <= MAX_WORDS:
        error = f"word_count:{n_words}"
    if error is None:
        sources = [x.lower() for x in (label, *neighbor_labels)]
        for m in _QUOTED.finditer(s):
            quoted = " ".join(next(g for g in m.groups() if g is not None).split()).strip(" .,:;!?").lower()
            if quoted and not any(quoted in x for x in sources):
                error = "invented_quote"
                break
    for kind, pattern in (("color", _COLOR_RE), ("shape", _SHAPE_RE)):
        if error is None:
            allowed = {w.lower() for w in pattern.findall(context)}
            invented = sorted({w.lower() for w in pattern.findall(s)} - allowed)
            if invented:
                error = f"invented_{kind}:{invented[0]}"
    warnings = []
    if not TARGET_WORDS[0] <= n_words <= TARGET_WORDS[1]:
        warnings.append("length")
    if s.lower() == normalize_instruction(label).lower():
        warnings.append("unchanged")
    warnings += [f"proper_noun:{tok}" for tok in _proper_tokens(label) if tok.lower() not in s.lower()]
    return error, warnings


def validate_item(item, meta) -> tuple[dict | None, str | None, list[str]]:
    """(normalized, error, warnings); normalized = {keep, rewritten, drop_reason}. Model style fields are ignored."""
    if not isinstance(item, dict):
        return None, "not_object", []
    keep = _keep(item.get("keep"))
    if keep is None:
        return None, "bad_keep", []
    if not keep:
        reason = normalize_drop_reason(item.get("drop_reason"))
        if reason is None:
            return None, "bad_drop_reason", []
        return {"keep": False, "rewritten": None, "drop_reason": reason}, None, []
    raw = item.get("instruction", item.get("rewritten"))
    if not isinstance(raw, str) or not raw.strip():
        return None, "missing_instruction", []
    if "\n" in raw.strip() or "\r" in raw.strip():
        return None, "newline", []
    s = normalize_instruction(raw)
    if not s:
        return None, "missing_instruction", []
    error, warnings = _lint(s, meta)
    if error:
        return None, error, warnings
    return {"keep": True, "rewritten": s, "drop_reason": None}, None, warnings


def _as_k(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def validate_batch(text: str, metas: list[dict]) -> tuple[dict[int, dict], dict[int, str], dict[int, list[str]], bool]:
    """Validate one reply for metas keyed k = 1..N. Returns (ok_by_k, errors_by_k, warnings_by_k, payload_ok)."""
    n = len(metas)
    try:
        items = extract_items(parse_json_payload(text))
    except ValueError as e:
        return {}, {k: f"invalid_json: {e}" for k in range(1, n + 1)}, {}, False
    ok, errors, warnings, seen = {}, {}, {}, set()
    for item in items:
        k = _as_k(item.get("k")) if isinstance(item, dict) else None
        if k is None or not 1 <= k <= n or k in seen:
            continue
        seen.add(k)
        normalized, error, warns = validate_item(item, metas[k - 1])
        if error:
            errors[k] = error
        else:
            ok[k] = normalized
        if warns:
            warnings[k] = warns
    for k in range(1, n + 1):
        if k not in seen:
            errors[k] = "missing"
    return ok, dict(sorted(errors.items())), warnings, True
