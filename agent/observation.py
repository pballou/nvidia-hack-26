"""Per-step text observation for Lightning: {step, url, title, screen_changed, last_action_result, screen_summary}.

The rendered text is the only thing Lightning learns about the screen, so keep it
compact. The URL is shown without its query string (the sandbox `sid` is noise).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

MAX_SUMMARY_LINES = 40

NO_CHANGE_NOTE = (
    "The screen did NOT change after your last action. The click probably missed or the "
    "element is not interactive: describe the target differently or try another way."
)


def display_url(url: str) -> str:
    """Path (+ fragment) only: 'http://127.0.0.1:8090/settings?sid=x#a11y' -> '/settings#a11y'."""
    if not url:
        return ""
    parts = urlsplit(url)
    if not parts.scheme.startswith("http"):
        return url
    return (parts.path or "/") + (f"#{parts.fragment}" if parts.fragment else "")


def clip_summary(text: str, max_lines: int = MAX_SUMMARY_LINES) -> str:
    lines = [ln.rstrip() for ln in (text or "").strip().splitlines() if ln.strip()]
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
