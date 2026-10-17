"""Mock GUI assist agent: the same interface, events and trace layout as T2's agent.core.

    async run_task(request, *, start_url, cfg, on_event=None) -> dict (RunResult fields)

It drives a real Playwright Chromium against the sandbox from pixels alone, like the
vision agent: page.goto, screenshots, mouse clicks and wheel, the keyboard, page.url
and page.title. It never uses locators, selectors, evaluate() or the sandbox state
API. In place of Lightning and the grounder, the plan comes from
console/mock_scripts.json: for each sandbox task, the tool calls (target descriptions
in the grounder's style) plus the pixel point to click, measured ahead of time by
console/tools/calibrate_mock.py. A request picks a script by its exact text (case and
whitespace don't matter). Anything else gets a short generic run that looks at the
screen and gives up with done(success=false).

Every payload carries "mock": true. The summary/plan/ground timings are representative
simulated numbers; act timings are measured. cfg.step_delay_s paces the steps so the
console demo looks live.

Self-contained on purpose (no agent/ imports), so it keeps working while agent/ changes.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import random
import secrets
import time
import zlib
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from PIL import Image, ImageChops, ImageDraw, ImageFont

log = logging.getLogger("console.mock_agent")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SCRIPTS_PATH = HERE / "mock_scripts.json"
VIEWPORT = {"width": 1280, "height": 800}
SCROLL_PX = {"small": 240, "page": 640}
SCROLL_ORIGIN = (760, 480)  # default mouse position for the wheel (over the main content)
TOOLS = ("click", "type_text", "press_key", "scroll", "done")
FALLBACK_PLAN = "Mock agent: no scripted plan for this request"
FALLBACK_SUMMARY = ("This is the mock agent. It only replays the scripted sandbox tasks, "
                    "so it did not do anything for this request.")
NO_SUMMARY = "(mock agent: no screen description for this step)"


@dataclass
class MockConfig:
    runs_dir: str = "runs"  # relative to the repo root
    sandbox_url: str = "http://127.0.0.1:8090"
    grounder_model: str = "base"
    thinking: str = "retry"
    lightning: str = "hosted"
    headed: bool = False
    step_delay_s: float = 0.8
    scripts_path: str = str(SCRIPTS_PATH)
    max_steps: int = 12
    settle_ms: int = 250

    @property
    def run_dir_root(self) -> Path:
        p = Path(self.runs_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    def public_dict(self) -> dict:
        return {"agent": "mock", "sandbox_url": self.sandbox_url, "grounder_model": self.grounder_model,
                "thinking": self.thinking, "lightning": self.lightning, "max_steps": self.max_steps,
                "viewport": dict(VIEWPORT), "headed": self.headed, "step_delay_s": self.step_delay_s}


@dataclass
class MockStepEvent:
    """Same fields as agent.trace.StepEvent (contracts.md §5)."""
    run_id: str
    step: int
    kind: str
    ts: float
    payload: dict[str, Any] = field(default_factory=dict)
    screenshot_path: str | None = None
    overlay_path: str | None = None
    timings_ms: dict[str, float] | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class ScriptError(Exception):
    """The mock script is missing, malformed or stale."""


# --- pixels-only browser helpers (shared with console/tools/calibrate_mock.py) ----------


@asynccontextmanager
async def open_page(headed: bool = False):
    """A fresh Chromium page at 1280x800, dsf 1, reduced motion, en-US, America/New_York."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not headed)
        try:
            ctx = await browser.new_context(viewport=dict(VIEWPORT), device_scale_factor=1, reduced_motion="reduce",
                                            locale="en-US", timezone_id="America/New_York")
            yield await ctx.new_page()
        finally:
            await browser.close()


async def screenshot(page) -> Image.Image:
    png = await page.screenshot(type="png", animations="disabled", caret="hide")
    return Image.open(io.BytesIO(png)).convert("RGB")


async def settle(page, settle_ms: int = 250) -> Image.Image:
    """Let a click-triggered fetch + reload finish, then return a screenshot that has stopped changing."""
    await asyncio.sleep(0.25)
    for state, timeout in (("load", 5000), ("networkidle", 2000)):
        try:
            await page.wait_for_load_state(state, timeout=timeout)
        except Exception:  # timeouts are fine; the screenshot loop below decides
            pass
    await asyncio.sleep(settle_ms / 1000)
    img = await screenshot(page)
    for _ in range(3):
        await asyncio.sleep(0.1)
        nxt = await screenshot(page)
        if ImageChops.difference(img, nxt).getbbox() is None:
            return nxt
        img = nxt
    return img


def screen_changed(before: Image.Image, after: Image.Image, threshold: int = 12, min_pixels: int = 3) -> bool:
    """Downsampled grayscale diff: True if at least `min_pixels` cells differ by more than `threshold`."""
    size = (160, 100)
    a = before.convert("L").resize(size, Image.BOX)
    b = after.convert("L").resize(size, Image.BOX)
    return sum(ImageChops.difference(a, b).histogram()[threshold + 1:]) >= min_pixels


def _point(step: dict, i: int) -> tuple[float, float]:
    x, y = step.get("x"), step.get("y")
    ok = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y))
    if not ok or not (0 <= x < VIEWPORT["width"] and 0 <= y < VIEWPORT["height"]):
        raise ScriptError(f"step {i} has no valid click point (x={x!r}, y={y!r})")
    return float(x), float(y)


async def perform(page, step: dict, i: int = 0) -> str:
    """Run one scripted tool call with pixels and keys only. Returns the action result text."""
    tool, args = step.get("tool"), step.get("arguments") or {}
    target = args.get("target_description")
    if tool == "click":
        x, y = _point(step, i)
        await page.mouse.click(x, y)
        return f'clicked "{target}" at ({round(x)}, {round(y)})'
    if tool == "type_text":
        if target:
            x, y = _point(step, i)
            await page.mouse.click(x, y)
            await page.keyboard.press("ControlOrMeta+a")  # replace the field's text
        await page.keyboard.type(str(args.get("text", "")))
        return f'typed "{str(args.get("text", ""))[:80]}"' + (f' into "{target}"' if target else "")
    if tool == "press_key":
        await page.keyboard.press(str(args.get("key")))
        return f"pressed {args.get('key')}"
    if tool == "scroll":
        direction, amount = args.get("direction"), args.get("amount")
        if direction not in ("up", "down") or amount not in SCROLL_PX:
            raise ScriptError(f"step {i}: bad scroll arguments {args!r}")
        ox, oy = step.get("x", SCROLL_ORIGIN[0]), step.get("y", SCROLL_ORIGIN[1])
        await page.mouse.move(ox, oy)
        await page.mouse.wheel(0, SCROLL_PX[amount] * (1 if direction == "down" else -1))
        return f"scrolled {direction} ({amount})"
    raise ScriptError(f"step {i}: unknown tool {tool!r}")


def describe(tool: str, args: dict) -> str:
    """The tool call in plain words, for the console's step log."""
    if tool == "click":
        return f"Clicking: {args.get('target_description', '')}"
    if tool == "type_text":
        where = f" into {args['target_description']}" if args.get("target_description") else ""
        return f'Typing "{args.get("text", "")}"{where}'
    if tool == "press_key":
        return f"Pressing {args.get('key', '')}"
    if tool == "scroll":
        return f"Scrolling {args.get('direction', '')} ({'a little' if args.get('amount') == 'small' else 'one page'})"
    if tool == "done":
        return f"Finishing: {args.get('summary', '')}"
    return f"{tool}({args})"


# --- trace (same layout as agent/trace.py) ------------------------------------------------


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def draw_overlay(img: Image.Image, x: float, y: float, target: str, step: int,
                 color: tuple[int, int, int] = (255, 0, 160)) -> Image.Image:
    """Copy of `img` with a crosshair at (x, y), the target text and the step number (as agent/trace.py)."""
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    font = _font(18)
    r = 14
    d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
    d.line([x - 2.2 * r, y, x - 0.4 * r, y], fill=color, width=3)
    d.line([x + 0.4 * r, y, x + 2.2 * r, y], fill=color, width=3)
    d.line([x, y - 2.2 * r, x, y - 0.4 * r], fill=color, width=3)
    d.line([x, y + 0.4 * r, x, y + 2.2 * r], fill=color, width=3)
    d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
    label = f"step {step}: {target}" if target else f"step {step}"
    tw = d.textlength(label, font=font)
    lx = min(max(8, x + 20), w - tw - 16)
    ly = y + 26 if y < h - 60 else y - 50
    d.rectangle([lx - 6, ly - 4, lx + tw + 6, ly + 24], fill=(0, 0, 0))
    d.text((lx, ly), label, fill=(255, 255, 255), font=font)
    return out


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-mock-" + secrets.token_hex(3)


def render_observation(o: dict) -> str:
    """Text observation in the shape agent/observation.py renders for Lightning."""
    path = urlsplit(o["url"]).path or "/" if o["url"] else ""
    lines = [f"Step {o['step']} | URL: {path} | Title: {o['title'] or '(none)'}"]
    if o["last_action_result"]:
        lines.append(f"Last action: {o['last_action_result']}")
    if o["screen_changed"] is False:
        lines.append("The screen did NOT change after your last action.")
    elif o["screen_changed"] is True:
        lines.append("The screen changed.")
    lines += ["Screen:", o["screen_summary"] or NO_SUMMARY]
    return "\n".join(lines)


def _norm(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def find_script(path: str | Path, request: str) -> tuple[dict | None, str | None]:
    """(script, None) on a match, (None, None) if no script matches, (None, error) if the file is unusable."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None, f"{path.name} not found; run python -m console.tools.calibrate_mock"
    except (OSError, ValueError) as e:
        return None, f"could not read {path.name}: {e}"
    tasks = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(tasks, dict):
        return None, f"{path.name} has no 'tasks' object"
    key = _norm(request)
    for task_id, script in tasks.items():
        if isinstance(script, dict) and _norm(script.get("request", "")) == key:
            return {**script, "task_id": task_id}, None
    return None, None


class _Run:
    def __init__(self, request: str, start_url: str, cfg: MockConfig,
                 on_event: Callable[[MockStepEvent], Awaitable[None]] | None):
        self.request, self.start_url, self.cfg, self.on_event = request, start_url, cfg, on_event
        self.run_id = new_run_id()
        self.dir = cfg.run_dir_root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._events = (self.dir / "events.jsonl").open("a", encoding="utf-8")
        self.rng = random.Random(zlib.crc32(_norm(request).encode()))
        self.t0 = time.perf_counter()
        self.step = 0
        self.stats = {"actions": 0, "clicks": 0, "failed": 0}
        self.finished: dict | None = None

    async def emit(self, kind: str, payload: dict, *, screenshot_path: str | None = None,
                   overlay_path: str | None = None, timings_ms: dict | None = None) -> None:
        ev = MockStepEvent(run_id=self.run_id, step=self.step, kind=kind, ts=round(time.time(), 3),
                           payload={**payload, "mock": True}, screenshot_path=screenshot_path,
                           overlay_path=overlay_path,
                           timings_ms={k: round(v, 1) for k, v in timings_ms.items()} if timings_ms else None)
        if not self._events.closed:
            self._events.write(ev.to_json() + "\n")
            self._events.flush()
        if self.on_event is not None:
            try:
                await self.on_event(ev)
            except Exception:
                log.exception("on_event failed for %s", kind)

    def sim(self, lo: float, hi: float) -> float:
        return self.rng.uniform(lo, hi)

    async def pace(self, frac: float) -> None:
        if self.cfg.step_delay_s > 0:
            await asyncio.sleep(self.cfg.step_delay_s * frac)

    def save(self, img: Image.Image) -> str:
        path = self.dir / f"step_{self.step:02d}.png"
        img.save(path)
        return str(path)

    def save_overlay(self, img: Image.Image, x: float, y: float, target: str) -> str:
        path = self.dir / f"step_{self.step:02d}_overlay.png"
        draw_overlay(img, x, y, target, self.step).save(path)
        return str(path)

    async def observe(self, page, img: Image.Image, changed: bool | None, last_result: str, summary: str) -> None:
        try:
            title = await page.title()
        except Exception as e:  # mid-navigation
            title = f"(unavailable: {type(e).__name__})"
        obs = {"step": self.step, "url": page.url, "title": title, "screen_changed": changed,
               "last_action_result": last_result, "screen_summary": summary or NO_SUMMARY}
        await self.emit("observation", {"observation": obs, "text": render_observation(obs)},
                        screenshot_path=self.save(img), timings_ms={"summary": self.sim(650, 1400)})

    async def finish(self, success: bool, summary: str, reason: str) -> dict:
        if self.finished is None:
            self.stats["elapsed_ms"] = round((time.perf_counter() - self.t0) * 1000)
            await self.emit("done", {"success_claimed": success, "summary": summary, "steps": self.step,
                                     "reason": reason, "stats": dict(self.stats)})
            self.finished = {"run_id": self.run_id, "success_claimed": success, "summary": summary,
                             "steps": self.step, "trace_dir": str(self.dir)}
        return self.finished

    async def fail(self, message: str, where: str) -> dict:
        self.stats["failed"] += 1
        await self.emit("error", {"message": message, "where": where, "recoverable": False})
        return await self.finish(False, f"Mock run failed: {message}", "error")

    async def play(self, page) -> dict:
        script, script_error = find_script(self.cfg.scripts_path, self.request)
        t = time.perf_counter()
        await page.goto(self.start_url, wait_until="load", timeout=15000)
        img = await settle(page, self.cfg.settle_ms)
        log.debug("start page ready in %.0f ms", (time.perf_counter() - t) * 1000)
        await self.observe(page, img, None, "", (script or {}).get("start_screen", ""))
        if script is None:
            return await self.fallback(script_error)

        steps = script.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ScriptError(f"script {script['task_id']!r} has no steps")
        last_changed: bool | None = None
        for step in steps:
            if self.step >= self.cfg.max_steps:
                return await self.finish(False, "I ran out of steps before finishing.", "max_steps")
            self.step += 1
            if not isinstance(step, dict) or step.get("tool") not in TOOLS:
                raise ScriptError(f"step {self.step}: unknown tool {step.get('tool') if isinstance(step, dict) else step!r}")
            tool, args = step["tool"], dict(step.get("arguments") or {})
            thinking = self.cfg.thinking == "on" or (self.cfg.thinking == "retry" and last_changed is False)
            await self.pace(0.5)
            await self.emit("plan", {"content": step.get("content", ""), "reasoning": step.get("reasoning", ""),
                                     "thinking": thinking, "tool_calls": [{"name": tool, "arguments": args}]},
                            timings_ms={"plan": self.sim(1400, 3800) * (1.8 if thinking else 1.0)})
            await self.emit("tool_call", {"name": tool, "arguments": args, "valid": True,
                                          "description": describe(tool, args)})
            if tool == "done":
                await self.emit("action", {"name": tool, "arguments": args, "ok": True, "screen_changed": None,
                                           "done": True})
                return await self.finish(bool(args.get("success")), str(args.get("summary", "")), "done")

            before, click, overlay = img, None, None
            target = args.get("target_description")
            if tool == "click" or (tool == "type_text" and target):
                x, y = _point(step, self.step)
                await self.pace(0.2)
                overlay = self.save_overlay(before, x, y, target or "")
                pt = [round(x / VIEWPORT["width"] * 1000), round(y / VIEWPORT["height"] * 1000)]
                await self.emit("grounding", {"target": target, "x": x, "y": y, "point_1000": pt,
                                              "raw": json.dumps({"point_2d": pt}),
                                              "model": f"mock ({self.cfg.grounder_model})"},
                                overlay_path=overlay, timings_ms={"ground": self.sim(220, 520)})
                click = {"x": x, "y": y}
            t = time.perf_counter()
            result = await perform(page, step, self.step)
            img = await settle(page, self.cfg.settle_ms)
            act_ms = (time.perf_counter() - t) * 1000
            changed = screen_changed(before, img)
            self.stats["actions"] += 1
            self.stats["clicks"] += 1 if click else 0
            await self.emit("action", {"name": tool, "arguments": args, "ok": True, "screen_changed": changed,
                                       **({"click": click} if click else {})},
                            overlay_path=overlay, timings_ms={"act": act_ms})
            await self.pace(0.3)
            await self.observe(page, img, changed, result, step.get("screen_after", ""))
            want = step.get("path_after")
            got = urlsplit(page.url).path
            if want and got != want:
                raise ScriptError(f"mock script is stale: after step {self.step} the page is {got}, expected {want}. "
                                  "Re-run python -m console.tools.calibrate_mock")
            last_changed = changed
        return await self.finish(False, "The script ended without calling done.", "max_steps")

    async def fallback(self, script_error: str | None) -> dict:
        if script_error:
            await self.emit("error", {"message": script_error, "where": "mock_scripts", "recoverable": True})
        self.step = 1
        args = {"summary": FALLBACK_SUMMARY, "success": False}
        await self.pace(0.5)
        await self.emit("plan", {"content": FALLBACK_PLAN, "reasoning": "", "thinking": False,
                                 "tool_calls": [{"name": "done", "arguments": args}]},
                        timings_ms={"plan": self.sim(900, 1600)})
        await self.emit("tool_call", {"name": "done", "arguments": args, "valid": True,
                                      "description": describe("done", args)})
        await self.emit("action", {"name": "done", "arguments": args, "ok": True, "screen_changed": None, "done": True})
        return await self.finish(False, FALLBACK_SUMMARY, "done")

    def close(self) -> None:
        self._events.close()


async def run_task(request: str, *, start_url: str, cfg: MockConfig | None = None,
                   on_event: Callable[[MockStepEvent], Awaitable[None]] | None = None) -> dict:
    """Same contract as agent.core.run_task (contracts.md §5). Never raises, except CancelledError."""
    cfg = cfg or MockConfig()
    run = _Run(request, start_url, cfg, on_event)
    try:
        await run.emit("start", {"request": request, "start_url": start_url, "config": cfg.public_dict()})
        try:
            async with open_page(cfg.headed) as page:  # closes the browser on the way out
                try:
                    return await run.play(page)
                except ScriptError as e:
                    return await run.fail(str(e), "mock_script")
                except asyncio.CancelledError:
                    await run.finish(False, "Stopped by the user before finishing.", "stopped")
                    raise
        except asyncio.CancelledError:
            await run.finish(False, "Stopped by the user before finishing.", "stopped")  # no-op if already sent
            raise
        except Exception as e:  # browser launch, navigation, Playwright errors
            log.exception("mock run %s failed", run.run_id)
            return await run.fail(f"{type(e).__name__}: {e}", "browser")
    finally:
        run.close()
