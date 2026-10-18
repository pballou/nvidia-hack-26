"""StepEvent / RunResult (docs/tasks/contracts.md §5) and the runs/<run_id>/ trace writer.

Trace layout (gitignored):
  runs/<run_id>/events.jsonl        one StepEvent per line
  runs/<run_id>/step_XX.png         the screenshot observed after step XX (step_00 = start)
  runs/<run_id>/step_XX_overlay.png the screenshot the grounder saw for step XX's click,
                                    with a crosshair at the click, the target text and the step

Event payloads by kind (all JSON-serializable; done-event stats are documented in agent/core.py):
  start        {request, start_url, config, parent_run_id?}  (config.lightning.root: the checkpoint the
               endpoint reports serving, or null + server_error; parent_run_id: the needs_input run
               this request answers)
  clarify      {action: proceed|ask, request, question?, normalized_request?, model, prompt_version,
               prompt_sha, usage?, latency_ms?, error?, skipped?, tool_call?: {name, arguments}}
               (step 0, cfg.clarify only, agent/clarify.py; error = the call failed and the run
               proceeded anyway; skipped = no call was made)
  observation  {observation: {step, url, title, screen_changed, last_action_result, screen_summary}, text,
               summary?: {model, prompt_sha?, usage?, error?, reused?}}
               (no screenshot_path when the screen could not be captured; summary.reused=true when
               the description was carried over without a new summarizer call)
  plan         {content, reasoning, thinking, tool_calls: [{name, arguments}], usage?, finish_reason?, wait_ms?,
               model, prompt_version, prompt_sha}
  tool_call    {name, arguments, valid, error?, description, from_content?}
               (valid=false: rejected by schema validation; nothing is executed, so no
               grounding/action/observation event follows for that step)
  grounding    {target, x, y, point_1000, raw, model, error?, prompt_sha?, usage?}
  prompt_sha = first 10 hex of sha256 of the exact prompt (agent.prompts.prompt_sha): the planner's
  system prompt + tool schema, the summary prompt, or the grounding prompt template per model family.
  usage = {prompt_tokens, completion_tokens} when the server reports it.
  action       {name, arguments, ok, error?, screen_changed, click?: {x, y}, done?}
  error        {message, where, recoverable, finish_reason?}
  done         {success_claimed, summary, steps, reason: done|max_steps|stuck|needs_input|stopped|error, stats,
               question?, request?}
               (stuck: agent/progress.py ended a run that made no progress; the summary asks the user.
               needs_input: the clarify step asked `question` before the browser opened; steps 0)
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

EVENT_KINDS = ("start", "clarify", "observation", "plan", "tool_call", "grounding", "action", "error", "done")


@dataclass
class StepEvent:
    run_id: str
    step: int
    kind: str
    ts: float
    payload: dict[str, Any] = field(default_factory=dict)
    screenshot_path: str | None = None
    overlay_path: str | None = None
    timings_ms: dict[str, float] | None = None

    def __post_init__(self):
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {self.kind!r}")

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


@dataclass
class RunResult:
    run_id: str
    success_claimed: bool
    summary: str
    steps: int
    trace_dir: str
    reason: str | None = None    # the done event's reason
    question: str | None = None  # reason "needs_input": what to ask the user

    def to_dict(self) -> dict:
        return asdict(self)


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def draw_overlay(img: Image.Image, x: float | None, y: float | None, target: str, step: int,
                 color: tuple[int, int, int] = (255, 0, 160)) -> Image.Image:
    """Copy of `img` with a crosshair at (x, y), the target text and the step number."""
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    font = _font(18)
    if x is not None and y is not None:
        r = 14
        d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
        d.line([x - 2.2 * r, y, x - 0.4 * r, y], fill=color, width=3)
        d.line([x + 0.4 * r, y, x + 2.2 * r, y], fill=color, width=3)
        d.line([x, y - 2.2 * r, x, y - 0.4 * r], fill=color, width=3)
        d.line([x, y + 0.4 * r, x, y + 2.2 * r], fill=color, width=3)
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
    label = f"step {step}: {target}" if target else f"step {step}"
    if x is None:
        label += "  (no click point)"
    tw = d.textlength(label, font=font)
    # label near the click, kept on screen
    lx = 8 if x is None else min(max(8, x + 20), w - tw - 16)
    ly = 8 if y is None else (y + 26 if y < h - 60 else y - 50)
    d.rectangle([lx - 6, ly - 4, lx + tw + 6, ly + 24], fill=(0, 0, 0))
    d.text((lx, ly), label, fill=(255, 255, 255), font=font)
    return out


class TraceWriter:
    """Writes events.jsonl and step images under <runs_dir>/<run_id>/."""

    def __init__(self, runs_dir: str | Path, run_id: str | None = None):
        self.run_id = run_id or new_run_id()
        self.dir = Path(runs_dir) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._events = (self.dir / "events.jsonl").open("a", encoding="utf-8")

    def event(self, step: int, kind: str, payload: dict | None = None, *, screenshot_path: str | None = None,
              overlay_path: str | None = None, timings_ms: dict | None = None) -> StepEvent:
        ev = StepEvent(run_id=self.run_id, step=step, kind=kind, ts=round(time.time(), 3), payload=payload or {},
                       screenshot_path=screenshot_path, overlay_path=overlay_path,
                       timings_ms={k: round(v, 1) for k, v in timings_ms.items()} if timings_ms else None)
        self._events.write(ev.to_json() + "\n")
        self._events.flush()
        return ev

    def save_screenshot(self, step: int, img: Image.Image) -> str:
        path = self.dir / f"step_{step:02d}.png"
        img.save(path)
        return str(path)

    def save_overlay(self, step: int, img: Image.Image, x: float | None, y: float | None, target: str) -> str:
        path = self.dir / f"step_{step:02d}_overlay.png"
        draw_overlay(img, x, y, target, step).save(path)
        return str(path)

    def close(self) -> None:
        if not self._events.closed:
            self._events.close()
