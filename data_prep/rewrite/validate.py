"""Parse Lightning's batched rewrite output and lint every item against the metadata it was given."""

import json
import os
import re

from data_prep.rewrite import meta as metalib

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
STOPWORDS = frozenset("a an and the of to in on for at by with from or your my this that is it".split())
MIN_WORDS, MAX_WORDS = 2, 12
TARGET_WORDS = (3, 8)

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```[\w-]*[ \t]*\n?(.*?)```", re.S)
_QUOTED = re.compile(r'"([^"\n]+)"|“([^”\n]+)”|(?<!\w)\'([^\'\n]+)\'(?!\w)|(?<!\w)‘([^’\n]+)’(?!\w)')
_DECODER = json.JSONDecoder(strict=False)


def _word_re(words) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(sorted(words, key=len, reverse=True)) + r")s?\b", re.I)


_COLOR_RE, _SHAPE_RE = _word_re(COLOR_WORDS), _word_re(SHAPE_WORDS)

# Labels that are only numbers, prices, percentages, counters, sizes, durations, clock times, dates or relative times:
# one or more such atoms joined by separators. They are dropped as `numeric` before Lightning sees them (rewrite.py),
# so NUMERIC_LABEL_RE.pattern is published verbatim in METHOD.md.
_NUM = r"[+\-−]?\d+(?:[.,]\d+)*"
_CLOCK = r"\d{1,2}(?::\d{2}){1,2}(?: ?[ap]\.?m\.?)?|\d{1,2} ?[ap]\.?m\.?"
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?"
          r"|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
_WDAY = r"(?:mon|tues?|wed(?:nes)?|thu(?:rs?)?|fri|sat(?:ur)?|sun)(?:day)?\.?"
_DAY = r"\d{1,2}(?:st|nd|rd|th)?\.?"
_UNIT = r"(?:s|secs?|seconds?|m|mins?|minutes?|h|hrs?|hours?|d|days?|w|wks?|weeks?|mos?|months?|y|yrs?|years?)\.?"
_LONG_UNIT = r"(?:secs?|seconds?|mins?|minutes?|hrs?|hours?|days?|wks?|weeks?|months?|yrs?|years?)\.?"
_MEASURE = (r"(?:[kmgt]i?b|bytes?|[kmg]?hz|[kmg]bps|km|mi|miles?|ft|cm|mm|kg|lbs?|oz|pt|px|ms|mph|°[cf]?)")
_YEAR = r"(?:\d{4}|'\d{2})"
_DATE = (rf"\d{{4}}-\d{{1,2}}-\d{{1,2}}(?:t\d{{1,2}}:\d{{2}}(?::\d{{2}})?(?:z|[+\-]\d{{2}}:?\d{{2}})?)?"
         rf"|\d{{1,2}}[/.\-]\d{{1,2}}[/.\-]\d{{2,4}}|\d{{4}} {_MONTH} {_DAY}|{_DAY} {_WDAY}"
         rf"|(?:(?:today|yesterday|tomorrow|{_WDAY}),? )?(?:{_MONTH} {_DAY}|{_DAY} {_MONTH})(?:,? {_YEAR})?"
         rf"|{_MONTH},? {_YEAR}")
_ATOM = (rf"(?:{_DATE})(?:,? (?:at )?(?:{_CLOCK}))?|(?:(?:today|yesterday|tomorrow|{_WDAY}),? (?:at )?)?(?:{_CLOCK})"
         rf"|(?:{_NUM}|an?|a few) ?{_UNIT} (?:ago|left|remaining)|in {_NUM} ?{_UNIT}|(?:just |right )?now"
         rf"|(?:a )?moments? ago|{_NUM} ?{_LONG_UNIT}(?: read)?|{_NUM} ?[hmsw]|{_NUM} ?{_MEASURE}|{_NUM} ?° ?[nsew]?"
         rf"|{_NUM} ?%|[+\-−]?(?:[$€£¥₹₩]|us\$|usd|eur|gbp) ?{_NUM}|{_NUM} ?(?:[$€£¥₹₩]|usd|eur|gbp)"
         rf"|(?:\d{{2,}}(?:[.,]\d+)?|\d+[.,]\d+) ?k|{_NUM}\+?|\({_NUM}(?: ?{_MEASURE})?\)|\[{_NUM}\]|\({_WDAY}\)")
NUMERIC_LABEL_RE = re.compile(rf"(?:{_ATOM})(?:(?:,? | ?[-–—/·|] ?| of )(?:{_ATOM}))*(?: today)?", re.I)
NUMERIC_LABEL_MAX_CHARS = 64  # longer labels are never treated as numeric (and the regex never backtracks on them)
RULE_MODEL = "rule:is_numeric_label"  # `model` of rows dropped by is_numeric_label before any request


def is_numeric_label(label: str) -> bool:
    """True if the whole label (normalized, case-insensitive) is only numbers, prices, percentages, counters, sizes,
    durations, clock times, dates or relative times: "42", "$4.99", "12:45 PM", "13 Oct 2023 10:00 AM",
    "73 minutes ago", "now", "(12)". Labels with any other word ("Wi-Fi 6", "Page 2", "Today") are not."""
    s = metalib.normalize_label(label).strip().rstrip(".,;:!").strip()
    return 0 < len(s) <= NUMERIC_LABEL_MAX_CHARS and NUMERIC_LABEL_RE.fullmatch(s) is not None


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


def _content_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 2 and t not in STOPWORDS]


def _neighbor_swap(s: str, meta: dict) -> str | None:
    """Neighbor label named by a functional/descriptive instruction that shares no content word with the target's
    label: the model probably described the neighbor ("Camera" -> "open more options"). Positional styles may name
    neighbors, so they are not checked."""
    if meta.get("style") == "positional":
        return None
    low, label = s.lower(), " ".join(str(meta.get("label") or "").split())
    if any(t in low for t in _content_tokens(label)):
        return None
    for n in meta.get("neighbors") or ():
        other = " ".join(str(n.get("label") or "").split()) if isinstance(n, dict) else ""
        if (other and other.lower() != label.lower() and _content_tokens(other)
                and re.search(r"(?<!\w)" + re.escape(other.lower()) + r"(?!\w)", low)):
            return other
    return None


def _same_word(a: str, b: str) -> bool:
    """Equal tokens, or inflections of one stem (setting/settings, library/libraries); numbers must be equal."""
    if a == b:
        return True
    if a.isdigit() or b.isdigit():
        return False
    n, short, long_ = len(os.path.commonprefix([a, b])), *sorted((len(a), len(b)))
    return n >= 4 and n >= short - 2 and long_ - n <= 5


def _with_joins(tokens: list[str]) -> list[str]:
    """Tokens plus adjacent pairs joined, so "Log out" matches "logout" and "Wi-Fi" matches "WiFi"."""
    return tokens + [a + b for a, b in zip(tokens, tokens[1:])]


def _no_label_word(s: str, meta: dict) -> str | None:
    """Every rewrite (all styles) keeps a content word of its own label: catches items shifted onto another item's
    instruction ("Others" -> "click Entertainment"). Labels with no ASCII content word (CJK, Arabic) are skipped."""
    want = _content_tokens(str(meta.get("label") or ""))
    if not want:
        return None
    have = _with_joins(re.findall(r"[a-z0-9]+", s.lower()))
    return None if any(_same_word(a, b) for a in _with_joins(want) for b in have) else "no_label_word"


def _neighbors(meta: dict) -> list[tuple[str, str]]:
    """(label, target_is) per neighbor with a label; target_is = where the target lies relative to that neighbor."""
    return [(" ".join(str(n.get("label") or "").split()), str(n.get("target_is") or ""))
            for n in meta.get("neighbors") or () if isinstance(n, dict) and str(n.get("label") or "").strip()]


_DIRECTIONS = {"left of": "left of", "right of": "right of", "above": "above", "below": "below", "under": "below",
               "beneath": "below"}
_RELATIONS = (*_DIRECTIONS, "next to", "near", "beside", "between")
_REL_RE = re.compile(r"(?<!\w)(" + "|".join(_RELATIONS) + r")\s+")
NEIGHBOR_PREFIX_CHARS = 12  # neighbor labels are truncated, so a mention must match this many leading characters


def _mentions(s: str, meta: dict, relations=_RELATIONS) -> list[tuple[str, list[tuple[str, str]]]]:
    """(relation, neighbors it names) for each "<relation> [the ]<neighbor label>" in s. A neighbor is named when s
    continues with at least min(len(label), NEIGHBOR_PREFIX_CHARS) of its characters (case-insensitive, trailing
    punctuation ignored) ending at a word boundary or past that many characters; only the longest matches count."""
    low, out = s.lower(), []
    labels = [(lab, rel, lab.lower().rstrip(" .:;,!?…")) for lab, rel in _neighbors(meta)]
    for m in _REL_RE.finditer(low):
        if m.group(1) not in relations:
            continue
        rest, best, named = low[m.end():].lstrip("\"'“‘"), 0, []
        for lab, target_is, key in labels:
            for r in (rest, rest[4:] if rest.startswith("the ") else ""):
                n, need = len(os.path.commonprefix([r, key])), min(len(key), NEIGHBOR_PREFIX_CHARS)
                if not key or n < need or (n == len(key) < NEIGHBOR_PREFIX_CHARS and r[n:n + 1].isalnum()):
                    continue
                if n > best:
                    best, named = n, []
                if n == best:
                    named.append((lab, target_is))
        if named:
            out.append((m.group(1), named))
    return out


def _wrong_direction(s: str, meta: dict) -> str | None:
    """A "left of/right of/above/below/under/beneath X" phrase that contradicts X's target_is ("next to", "near" and
    "between" are not checked). Neighbor labels are fed as target_is so no inversion is needed."""
    for rel, named in _mentions(s, meta, tuple(_DIRECTIONS)):
        if not any(target_is == _DIRECTIONS[rel] for _, target_is in named):
            return f"wrong_direction:{named[0][0]}"
    return None


# Element-type nouns, each with the words in the control noun, kind, label or neighbor labels that support it.
ELEMENT_NOUNS = {
    "text field": ("field",), "check box": ("checkbox", "check box"), "drop-down": ("dropdown", "drop-down"),
    "drop down": ("dropdown", "drop-down"), "button": ("button",), "icon": ("icon",), "link": ("link", "hyperlink"),
    "tab": ("tab",), "field": ("field",), "input": ("input", "field"), "checkbox": ("checkbox", "check box"),
    "dropdown": ("dropdown", "drop-down"), "menu": ("menu",), "slider": ("slider",), "switch": ("switch", "toggle"),
    "toggle": ("toggle", "switch"), "image": ("image", "picture"), "picture": ("picture", "image"),
}
VERB_NOUNS = frozenset({"toggle", "switch", "input"})  # fine as the first word: "toggle Bluetooth", "switch to Recent"
_NOUN_RE = re.compile(r"(?<!\w)(" + "|".join(sorted(ELEMENT_NOUNS, key=len, reverse=True)) + r")(?:e?s)?(?!\w)")


def _plural_re(words) -> re.Pattern:
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(w) for w in words) + r")(?:e?s)?(?!\w)")


_SUPPORT_RE = {noun: _plural_re(words) for noun, words in ELEMENT_NOUNS.items()}


def _invented_control(s: str, meta: dict) -> str | None:
    """An element-type noun not supported by the control noun, kind, label or a neighbor label. So no button/link/icon
    for a missing control (FineWeb), and only text/label/area/item/entry/option words for control text or area."""
    allowed = " ".join([str(meta.get("control") or ""), str(meta.get("kind") or ""), str(meta.get("label") or ""),
                        *(lab for lab, _ in _neighbors(meta))]).lower()
    low = s.lower()
    for m in _NOUN_RE.finditer(low):
        noun = m.group(1)
        if m.start() == 0 and noun in VERB_NOUNS:
            continue
        if not _SUPPORT_RE[noun].search(allowed):
            return f"invented_control:{noun}"
    return None


TEXT_VERBS = frozenset("click tap select choose press find go".split())


def _text_action_verb(s: str, meta: dict) -> str | None:
    """A functional rewrite of a text/area element must use a neutral click verb ("tap Digital Wellbeing"), not an
    action the text itself doesn't do ("open Digital Wellbeing"). Noun phrases ("the ... text") are not verbs."""
    if meta.get("control") not in metalib.TEXT_CONTROLS or meta.get("style") != "functional":
        return None
    first = (re.findall(r"[\w'-]+", s.lower()) or [""])[0]
    return None if first in TEXT_VERBS or first in ("the", "a", "an") else f"text_action_verb:{first}"


def _context(meta: dict) -> str:
    return " ".join([str(meta.get("label") or ""), *(lab for lab, _ in _neighbors(meta))]).lower()


_STYLE_RE = re.compile(r"(?<!\w)(" + "|".join(STYLES) + r")(?!\w)")
_GRID_RE = re.compile(r"(?<!\w)((?:top|middle|bottom)[-‐‑–](?:left|center|centre|right))(?!\w)")


def _style_word(s: str, meta: dict) -> str | None:
    """The requested style's name copied into the instruction ("the Screen Time text positional")."""
    context = _context(meta)
    words = [w for w in _STYLE_RE.findall(s.lower()) if not re.search(rf"(?<!\w){w}(?!\w)", context)]
    return f"style_word:{words[0]}" if words else None


def _grid_token(s: str, meta: dict) -> str | None:
    """A raw position bucket ("middle-right") copied verbatim; plain words ("at the top right") are fine."""
    context = _context(meta)
    tokens = [t for t in _GRID_RE.findall(s.lower()) if t not in context]
    return f"grid_token:{tokens[0]}" if tokens else None


POSITION_WORDS = frozenset("top bottom left right middle center centre upper lower first second third last".split())


def _tokens(text: str) -> list[str]:
    """Unicode word tokens minus stopwords, single characters kept ("Page 2" vs "Page 3", "X share" vs "Share this
    story") and non-ASCII words too ("关于 Google" vs "Google 搜索的运作方式")."""
    return [t for t in re.findall(r"[^\W_]+", re.sub(r"['’]s(?!\w)", "", text.lower())) if t not in STOPWORDS]


def _phrase(text: str) -> str:
    return " ".join(str(text).lower().split()).strip(" .,:;!?…")


def _ambiguous_lookalike(s: str, meta: dict) -> str | None:
    """A neighbor looks like the target (same first content word, or the same words) and the instruction doesn't
    tell them apart: no target word missing from that neighbor's label, no relation naming a neighbor, and no
    position word. Quoting the whole label also tells them apart unless it is part of the neighbor's label ("Sign in"
    vs "Sign up" is fine; "Settings" vs "Settings Sync" is not). Identical labels need a relation or a position."""
    want, have = _tokens(str(meta.get("label") or "")), _tokens(s)
    if not want or _mentions(s, meta) or POSITION_WORDS.intersection(have) - set(want):
        return None
    verbatim = _phrase(meta.get("label") or "")
    quoted = bool(verbatim) and verbatim in " ".join(s.lower().split())
    for other, _ in _neighbors(meta):
        theirs = _tokens(other)
        if not theirs or theirs[0] != want[0] or (quoted and verbatim not in _phrase(other)):
            continue
        if not any(t not in theirs and any(_same_word(t, h) for h in have) for t in want):
            return f"ambiguous_lookalike:{other}"
    return None


# Hard errors after the word-count/quote/color/shape lints, in order; the first failing one is reported.
CHECKS = (_no_label_word, _wrong_direction, _invented_control, _text_action_verb, _style_word, _grid_token,
          _ambiguous_lookalike)


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
    for check in CHECKS:
        if error is None:
            error = check(s, meta)
    warnings = []
    if not TARGET_WORDS[0] <= n_words <= TARGET_WORDS[1]:
        warnings.append("length")
    if s.lower() == normalize_instruction(label).lower():
        warnings.append("unchanged")
    warnings += [f"proper_noun:{tok}" for tok in _proper_tokens(label) if tok.lower() not in s.lower()]
    swapped = _neighbor_swap(s, meta)
    if swapped:
        warnings.append(f"neighbor_swap:{swapped}")
    return error, warnings


_EDGE_STYLE_RE = re.compile(r"^(?:" + "|".join(STYLES) + r")\s*[:\-–]?\s+|\s*[,;:\-–(]?\s*\(?(?:"
                            + "|".join(STYLES) + r")\)?$", re.I)


def cleanup_instruction(s: str, meta: dict) -> tuple[str, list[str]]:
    """Deterministic fixes for prompt vocabulary copied into the output, logged as warnings: a style name at the very
    start or end ("the MAYO text positional" -> "the MAYO text") and hyphenated grid tokens ("top-right" -> "top
    right"). A style word elsewhere, or one the label has, is left for the style_word error."""
    fixes = []
    context = _context(meta)
    stripped = _EDGE_STYLE_RE.sub("", s).strip()
    if stripped != s and stripped and not _STYLE_RE.search(context):
        s, fixes = stripped, fixes + ["stripped_style_word"]
    dehyph = _GRID_RE.sub(lambda m: re.sub(r"[-‐‑–]", " ", m.group(1)), s)
    if dehyph != s and not _GRID_RE.search(context):
        s, fixes = dehyph, fixes + ["dehyphenated_grid"]
    return s, fixes


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
    s, fixes = cleanup_instruction(normalize_instruction(raw), meta)
    if not s:
        return None, "missing_instruction", []
    error, warnings = _lint(s, meta)
    warnings = fixes + warnings
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
