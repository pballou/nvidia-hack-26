"""Per-step text observation for Lightning: {step, url, title, screen_changed, last_action_result, screen_summary}.

The rendered text is the only thing Lightning learns about the screen, so keep it
compact. The URL is shown without its query string (the sandbox `sid` is noise).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit

MAX_SUMMARY_LINES = 40
MAX_LINE_CHARS = 240  # long message rows otherwise bloat every Lightning prompt

NO_CHANGE_NOTE = (
    "The screen did NOT change after your last action. The click probably missed or the "
    "element is not interactive: describe the target differently or try another way."
)


def display_url(url: str) -> str:
    """Path, query minus the sandbox sid, fragment: '.../settings?tab=a11y&sid=x' -> '/settings?tab=a11y'."""
    if not url:
        return ""
    parts = urlsplit(url)
    if not parts.scheme.startswith("http"):
        return url
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "sid"])
    return (parts.path or "/") + (f"?{query}" if query else "") + (f"#{parts.fragment}" if parts.fragment else "")


_JSON_KEY_RE = re.compile(r'[{,]\s*"([A-Za-z][\w /&-]{0,40})"\s*:\s*')


def _json_lines(obj: dict, prefix: str = "") -> list[str]:
    out: list[str] = []
    for key, value in obj.items():
        label = f"{prefix} - {key}" if prefix else str(key)
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, dict):
                out.extend(_json_lines(item, label))
            elif item not in (None, ""):
                out.append(f"{label}: {item}")
    return out


def flatten_json_summary(text: str) -> str | None:
    """A description written as one JSON object -> "KEY: value" lines; None if it isn't one.

    The merged grounder checkpoint sometimes answers the summary prompt with JSON
    ({"PAGE": "Billing", "NAV": ..., "OVERLAYS": ...}) on one line, which the per-line cap
    would cut to its first 240 characters, dialog and all. The base model never does
    (0 of 1,315 descriptions). A reply cut off mid-object is split at its "KEY": boundaries.
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        t = t[4:].lstrip() if t.lower().startswith("json") else t
    if not t.startswith("{"):
        return None
    try:
        obj = json.loads(t)
    except ValueError:
        obj = None
    if isinstance(obj, dict):
        return "\n".join(_json_lines(obj)) or None
    parts = _JSON_KEY_RE.split(t)  # ["{", key1, value1, key2, value2, ...]
    lines = []
    for key, value in zip(parts[1::2], parts[2::2]):
        value = value.strip().strip("{}[]").strip().strip('"').replace('", "', "; ").replace('","', "; ")
        if value:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) or None


def clip_summary(text: str, max_lines: int = MAX_SUMMARY_LINES, max_line_chars: int = MAX_LINE_CHARS) -> str:
    """Drop blank and repeated lines (small VLMs loop), cap line length and line count.

    A JSON-shaped description is flattened into lines first (flatten_json_summary)."""
    text = flatten_json_summary(text) or text
    lines, seen = [], set()
    for ln in (text or "").strip().splitlines():
        ln = ln.rstrip()
        key = ln.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        lines.append(ln if len(ln) <= max_line_chars else ln[:max_line_chars - 1].rstrip() + "…")
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"(+{len(lines) - max_lines} more lines omitted)"]
    return "\n".join(lines)


@dataclass
class Observation:
    step: int
    url: str
    title: str
    screen_changed: bool | None  # None for the first observation
    last_action_result: str
    screen_summary: str

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        lines = [f"Step {self.step} | URL: {display_url(self.url)} | Title: {self.title or '(none)'}"]
        if self.last_action_result:
            lines.append(f"Last action: {self.last_action_result}")
        if self.screen_changed is False:
            lines.append(NO_CHANGE_NOTE)
        elif self.screen_changed is True:
            lines.append("The screen changed.")
        lines.append("Screen:")
        lines.append(clip_summary(self.screen_summary) or "(no description available)")
        return "\n".join(lines)


def build_observation(*, step: int, url: str, title: str, screen_changed: bool | None,
                      last_action_result: str, screen_summary: str) -> Observation:
    return Observation(step=step, url=url, title=title or "", screen_changed=screen_changed,
                       last_action_result=last_action_result or "", screen_summary=clip_summary(screen_summary))


def one_line(name: str, args: dict, ok: bool, screen_changed: bool | None, error: str | None = None) -> str:
    """Compact record of an older step for the context budget, e.g. 'click("the gear icon") -> ok, changed'."""
    shown = ", ".join(f"{k}={v!r}" if not isinstance(v, str) else f'"{v[:60]}"' for k, v in (args or {}).items())
    status = "ok" if ok else f"failed: {(error or '')[:80]}"
    if ok and screen_changed is not None:
        status += ", screen changed" if screen_changed else ", NO change"
    return f"{name}({shown}) -> {status}"
