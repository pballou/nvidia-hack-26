"""Progress-based stop for the interactive loop (agent/core.py, done reason "stuck").

A step cap alone cuts off long, correct tasks, and a higher cap alone lets a stuck agent
click the same wrong spot for minutes. So the interactive agent gets a generous cap
(configs/agent.yaml max_steps 25) plus this two-stage stop. The run is flagged after either:
- NO_PROGRESS_STREAK actions in a row made no progress (ok=False, or the screen did not
  change), or
- the same action made no progress SAME_ACTION_REPEATS times: same tool and same target /
  text / key (case and spacing ignored), or a click within CLICK_RADIUS_PX of an earlier
  click that changed nothing (the grounder lands on the same wrong element for differently
  worded targets).
The first flag is a WARNING: Lightning is told it is stuck and to change approach (and
thinks on the next call). If STOP_GRACE more actions make no progress after that, the run
stops as "stuck" with a summary of what was tried and a question for the user. Any action
that changes the screen clears the flag and the no-progress history.

Why the grace: replaying the 47 real (non-oracle) runs from Sat 2026-10-17 with these
thresholds, an immediate stop would have cut off 2 of 29 runs that later succeeded (both
after 5-6 no-change clicks); warning first and stopping 3 actions later cut off none.

An "action" is an executed, schema-valid tool call other than done; invalid calls are
corrected separately and count toward max_steps only. The Gym env does not use this: its
loop is Gym's simple_agent with max_steps 12, so reported Gym numbers are unaffected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

NO_PROGRESS_STREAK = 4
SAME_ACTION_REPEATS = 3
CLICK_RADIUS_PX = 12.0
STOP_GRACE = 3
SUMMARY_TRIED_MAX = 5

STUCK_WARNING = (
    "You seem stuck: {reason}. Try a substantially different approach: a different element or "
    "wording, scrolling to find it, filling required fields first, or another page or tab. If the "
    "request can't be done, call done(success=false) and say what you tried.")


@dataclass
class Attempt:
    name: str
    key: str                       # normalized tool + argument identity
    label: str                     # plain words for the summary, e.g. 'click "the L option"'
    xy: tuple[float, float] | None


def _norm(s: object) -> str:
    return " ".join(str(s or "").lower().split())


def attempt_key(name: str, args: dict) -> tuple[str, str]:
    """(identity key, plain label) of a tool call."""
    args = args or {}
    if name == "click":
        return f"click|{_norm(args.get('target_description'))}", f'click "{args.get("target_description", "")}"'
    if name == "type_text":
        target = args.get("target_description")
        label = f'type "{args.get("text", "")}"' + (f' into "{target}"' if target else "")
        return f"type_text|{_norm(args.get('text'))}|{_norm(target)}", label
    if name == "press_key":
        return f"press_key|{_norm(args.get('key'))}", f"press {args.get('key', '')}"
    if name == "scroll":
        return f"scroll|{_norm(args.get('direction'))}|{_norm(args.get('amount'))}", \
            f"scroll {args.get('direction', '')}"
    return f"{name}|{_norm(args)}", name


class ProgressTracker:
    """Feed every executed action to record(): returns None, "warn" (once per stuck stretch) or "stuck"."""

    def __init__(self, streak: int = NO_PROGRESS_STREAK, repeats: int = SAME_ACTION_REPEATS,
                 radius_px: float = CLICK_RADIUS_PX, grace: int = STOP_GRACE):
        self.streak_limit, self.repeats, self.radius, self.grace = streak, repeats, radius_px, grace
        self.streak = 0
        self.no_progress: list[Attempt] = []   # actions without progress since the last screen change
        self.tried: list[str] = []             # every no-progress label this run (for the summary)
        self.reason: str | None = None         # why the run was flagged
        self.warned_at: int | None = None      # streak when the warning was given
        self.n_warnings = 0

    def record(self, name: str, args: dict, *, ok: bool, screen_changed: bool | None,
               click_xy: tuple[float, float] | None = None) -> str | None:
        if name == "done":
            return None
        if ok and screen_changed:
            self.streak, self.no_progress, self.reason, self.warned_at = 0, [], None, None
            return None
        if ok and screen_changed is None:  # the result couldn't be captured: no evidence either way
            return None
        key, label = attempt_key(name, args)
        a = Attempt(name=name, key=key, label=label, xy=click_xy if name == "click" else None)
        same = [p for p in self.no_progress if p.key == key or self._near(p, a)]
        self.no_progress.append(a)
        self.tried.append(label)
        self.streak += 1
        if self.warned_at is not None:
            return "stuck" if self.streak >= self.warned_at + self.grace else None
        if len(same) + 1 >= self.repeats:
            self.reason = f"{label} made no progress {len(same) + 1} times"
        elif self.streak >= self.streak_limit:
            self.reason = f"{self.streak} actions in a row changed nothing"
        else:
            return None
        self.warned_at = self.streak
        self.n_warnings += 1
        return "warn"

    def _near(self, p: Attempt, a: Attempt) -> bool:
        return p.xy is not None and a.xy is not None and math.dist(p.xy, a.xy) <= self.radius

    def warning(self) -> str:
        return STUCK_WARNING.format(reason=self.reason or "nothing changed")

    def summary(self) -> str:
        """The done summary for a stuck stop: what was tried, and a question for the user."""
        tried: list[str] = []
        for label in reversed(self.tried):
            if label not in tried:
                tried.append(label)
            if len(tried) == SUMMARY_TRIED_MAX:
                break
        tried.reverse()
        return (f"I'm stuck: {self.reason or 'no progress'}, and changing approach didn't help. "
                f"I tried: {'; '.join(tried)}. None of it changed the screen. What would you like me to do "
                "next? You could tell me where the control is or what it looks like, or ask me to try "
                "something else.")
