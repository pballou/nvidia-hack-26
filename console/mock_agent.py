"""Mock GUI Assist agent: the same interface, events and trace layout as T2's agent.core.

    async run_task(request, *, start_url, cfg, on_event=None, parent_run_id=None) -> dict (RunResult fields)

It drives a real Playwright Chromium against the sandbox from pixels alone, like the
vision agent: page.goto, screenshots, mouse clicks and wheel, the keyboard, page.url
and page.title. It never uses locators, selectors, evaluate() or the sandbox state
API. In place of Lightning and the grounder, the plan comes from
console/mock_scripts.json: for each sandbox task, the tool calls (target descriptions
in the grounder's style) plus the pixel point to click, measured ahead of time by
console/tools/calibrate_mock.py. A request picks a script by its exact text (case and
whitespace don't matter), and only when the run starts where the script was recorded
(the script's start_path: same path and query, sid aside). Its points were measured on
that page, so replaying them anywhere else would click the wrong things. Anything else
gets a short generic run that looks at the screen and gives up with done(success=false),
saying why.

Request check (cfg.clarify on, like agent/clarify.py; off: no clarify event, as before): right after
start, before the browser opens, a step-0 `clarify` event in agent/trace.py's shape. Two scripted
unclear requests ask (CLARIFY_ASKS: "Allors towards my bill." and "Pay something toward my bill"):
the run ends at once with done reason "needs_input" (no browser). Everything else proceeds, with
normalized_request = the request trimmed, except a follow-up of a bill question answered with
fifty dollars / $50 (agent.clarify.follow_up_request's "(clarified: ...)" format), which the mock
restates as "Pay $50 toward my bill.". The restatement, not the user's words, then picks the script,
so that follow-up replays pay-50. Like the real agent, a request already answered twice skips the
check. A follow-up's start event carries parent_run_id.

Every payload carries "mock": true. Plans say model "mock (no Lightning)", observations
summary.model "mock (no VLM)", and prompt_version and prompt_sha are "mock" (the logs
page shows these names). Only measured timings are reported, split as agent/session.py
splits them: an action's act (the mouse/keyboard action alone) and settle (the wait for
the page to react), and an observation's screenshot (capturing the settled screen). The
mock doesn't run Lightning or a VLM, so it reports no plan/summary/ground latency rather
than inventing one. cfg.step_delay_s paces the steps so the console demo looks live.

Self-contained on purpose (no agent/ imports), so it keeps working while agent/ changes.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qsl, urlsplit

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
# Model and prompt names the mock declares (the same optional fields as agent/trace.py: plan.model,
# plan.prompt_version, plan.prompt_sha, observation.summary.{model, prompt_sha}, grounding.prompt_sha).
# They say plainly that no model ran and no prompt was sent.
MOCK_PLANNER = "mock (no Lightning)"
MOCK_SUMMARIZER = "mock (no VLM)"
MOCK_PROMPT_VERSION = "mock"
MOCK_PROMPT_SHA = "mock"
MOCK_PROMPT = {"prompt_version": MOCK_PROMPT_VERSION, "prompt_sha": MOCK_PROMPT_SHA}
MOCK_SUMMARY_INFO = {"model": MOCK_SUMMARIZER, **MOCK_PROMPT}
MOCK_CLARIFY_PROMPT = {"prompt_version": "clarify-v4 (mock)", "prompt_sha": MOCK_PROMPT_SHA}
# The request check's scripted questions (keys: clarify_key of the request) and the follow-up it understands.
CLARIFY_ASKS = {
    "allors towards my bill": "What would you like to do with your bill?",
    "pay something toward my bill": "How much would you like to pay toward your bill?",
}
FIFTY_DOLLARS = ("fifty dollars", "$50", "50 dollars", "$50.00")  # answers to a bill question (clarify_key)
BILL_RESTATED = "Pay $50 toward my bill."
CLARIFIED_MARK = "(clarified:"  # agent/clarify.py follow_up_request: previous + " (clarified: <answer>)"
MAX_CLARIFICATIONS = 2          # agent/clarify.py: a request answered this often skips the check
NO_CHANGE_NOTE = (  # same words as agent/observation.py
    "The screen did NOT change after your last action. The click probably missed or the "
    "element is not interactive: describe the target differently or try another way."
)


@dataclass
class MockConfig:
    runs_dir: str = "runs"  # relative to the repo root
    sandbox_url: str = "http://127.0.0.1:8090"
    grounder_model: str = "base"
    thinking: str = "retry"
    lightning: str = "hosted"
    clarify: str | bool = "off"  # "on" | "off" (the console's option) or a bool: the request check
    headed: bool = False
    step_delay_s: float = 0.8
    scripts_path: str = str(SCRIPTS_PATH)
    max_steps: int = 12
    settle_ms: int = 250

    @property
    def run_dir_root(self) -> Path:
        p = Path(self.runs_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def clarify_on(self) -> bool:
        return self.clarify is True or (isinstance(self.clarify, str) and self.clarify.strip().lower() == "on")

    def public_dict(self) -> dict:
        return {"agent": "mock", "sandbox_url": self.sandbox_url, "grounder_model": self.grounder_model,
                "thinking": self.thinking, "lightning": self.lightning, "clarify": self.clarify_on,
                "max_steps": self.max_steps,
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


async def start_playwright():
    """async_playwright().start(), safe to cancel. Playwright's own start is not: a cancel while the node
    driver is starting (a Stop in a run's first ~100 ms) skips its cleanup and leaves the driver (~130 MB)
    running until the process exits. Here a cancel lets the start finish (well under a second), stops the
    driver, then re-raises. Repeated cancels (two Stops) don't cut that wait short."""
    from playwright.async_api import async_playwright

    starting = asyncio.ensure_future(async_playwright().start())
    try:
        return await asyncio.shield(starting)
    except asyncio.CancelledError:
        while not starting.done():
            try:
                await asyncio.shield(starting)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not starting.cancelled() and starting.exception() is None:
            await starting.result().stop()  # stop() closes the driver's pipe at once; it exits on its own
        raise


@asynccontextmanager
async def open_page(headed: bool = False):
    """A fresh Chromium page at 1280x800, dsf 1, reduced motion, en-US, America/New_York.
    The browser and the Playwright driver are always closed on the way out, including on a cancel."""
    pw = await start_playwright()
    try:
        browser = await pw.chromium.launch(headless=not headed)
        try:
            ctx = await browser.new_context(viewport=dict(VIEWPORT), device_scale_factor=1, reduced_motion="reduce",
                                            locale="en-US", timezone_id="America/New_York")
            yield await ctx.new_page()
        finally:
            await browser.close()
    finally:
        await pw.stop()


async def screenshot(page) -> Image.Image:
    png = await page.screenshot(type="png", animations="disabled", caret="hide")
    return Image.open(io.BytesIO(png)).convert("RGB")


async def settle(page, settle_ms: int = 250, timings: dict | None = None) -> Image.Image:
    """Let a click-triggered fetch + reload finish, then return a screenshot that has stopped changing.
    With `timings`, records settle (the waits, in ms) and screenshot (time spent capturing, in ms)."""
    t0 = time.perf_counter()
    shot_s = 0.0

    async def capture() -> Image.Image:
        nonlocal shot_s
        t = time.perf_counter()
        try:
            return await screenshot(page)
        finally:
            shot_s += time.perf_counter() - t

    try:
        await asyncio.sleep(0.25)
        for state, timeout in (("load", 5000), ("networkidle", 2000)):
            try:
                await page.wait_for_load_state(state, timeout=timeout)
            except Exception:  # timeouts are fine; the screenshot loop below decides
                pass
        await asyncio.sleep(settle_ms / 1000)
        img = await capture()
        for _ in range(3):
            await asyncio.sleep(0.1)
            nxt = await capture()
            if ImageChops.difference(img, nxt).getbbox() is None:
                return nxt
            img = nxt
        return img
    finally:
        if timings is not None:
            timings["settle"] = (time.perf_counter() - t0 - shot_s) * 1000
            timings["screenshot"] = shot_s * 1000


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
        lines.append(NO_CHANGE_NOTE)
    elif o["screen_changed"] is True:
        lines.append("The screen changed.")
    lines += ["Screen:", o["screen_summary"] or NO_SUMMARY]
    return "\n".join(lines)


def _norm(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def clarify_key(text: str) -> str:
    """_norm without trailing punctuation: 'Pay something toward my bill.' == 'pay something toward my bill'."""
    return _norm(text).rstrip(" .!?,;:")


def clarify_decision(request: str) -> dict:
    """The mock's request check as a clarify event payload (agent/trace.py; emit adds "mock": true). No model runs,
    so there is no latency or usage."""
    out: dict[str, Any] = {"request": request, "model": MOCK_PLANNER, **MOCK_CLARIFY_PROMPT}
    n_answers = request.count(CLARIFIED_MARK)
    if n_answers >= MAX_CLARIFICATIONS:
        return {"action": "proceed", **out, "skipped": f"already clarified {n_answers} times"}
    question = None if n_answers else CLARIFY_ASKS.get(clarify_key(request))
    if question:
        return {"action": "ask", **out, "question": question,
                "tool_call": {"name": "ask_user", "arguments": {"question": question}}}
    base, _, rest = request.partition(CLARIFIED_MARK)
    answer = clarify_key(rest.rpartition(")")[0]) if n_answers else ""
    if clarify_key(base) in CLARIFY_ASKS and answer in FIFTY_DOLLARS:
        restated = BILL_RESTATED
    else:
        restated = " ".join(request.split())
    return {"action": "proceed", **out, "normalized_request": restated,
            "tool_call": {"name": "proceed", "arguments": {"normalized_request": restated}}}


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


def start_key(url: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """(path, sorted query pairs without sid) of a start URL or a script's start_path ("/settings?tab=x")."""
    parts = urlsplit(url or "/")
    query = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "sid")
    return parts.path or "/", tuple(query)


def show_start(url: str) -> str:
    """A start URL as the page it opens: "/settings?tab=accessibility" (no host, no sid)."""
    path, query = start_key(url)
    return path + ("?" + "&".join(f"{k}={v}" for k, v in query) if query else "")


def recorded_elsewhere(script: dict, start_url: str) -> str | None:
    """The script's start_path when the run starts on another page (its points would hit the wrong things), else
    None. Scripts written before start_path was recorded aren't checked (their path_after checks still are)."""
    want = script.get("start_path")
    if not isinstance(want, str) or start_key(want) == start_key(start_url):
        return None
    return want


class _Run:
    def __init__(self, request: str, start_url: str, cfg: MockConfig,
                 on_event: Callable[[MockStepEvent], Awaitable[None]] | None):
        self.request, self.start_url, self.cfg, self.on_event = request, start_url, cfg, on_event
        self.run_id = new_run_id()
        self.dir = cfg.run_dir_root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._events = (self.dir / "events.jsonl").open("a", encoding="utf-8")
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

    async def observe(self, page, img: Image.Image, changed: bool | None, last_result: str, summary: str,
                      timings_ms: dict | None = None) -> None:
        try:
            title = await page.title()
        except Exception as e:  # mid-navigation
            title = f"(unavailable: {type(e).__name__})"
        obs = {"step": self.step, "url": page.url, "title": title, "screen_changed": changed,
               "last_action_result": last_result, "screen_summary": summary or NO_SUMMARY}
        await self.emit("observation", {"observation": obs, "text": render_observation(obs),
                                        "summary": dict(MOCK_SUMMARY_INFO)},
                        screenshot_path=self.save(img), timings_ms=timings_ms)

    async def finish(self, success: bool, summary: str, reason: str, question: str | None = None) -> dict:
        if self.finished is None:
            self.stats["elapsed_ms"] = round((time.perf_counter() - self.t0) * 1000)
            done = {"success_claimed": success, "summary": summary, "steps": self.step, "reason": reason,
                    "stats": dict(self.stats)}
            if reason == "needs_input":  # agent/core.py: the console asks this, then starts a run with the answer
                done.update(question=question, request=self.request)
            await self.emit("done", done)
            self.finished = {"run_id": self.run_id, "success_claimed": success, "summary": summary,
                             "steps": self.step, "trace_dir": str(self.dir), "reason": reason, "question": question}
        return self.finished

    async def check_request(self) -> dict:
        """The step-0 clarify event (before the browser opens). Returns its payload."""
        await self.pace(0.3)
        c = clarify_decision(self.request)
        await self.emit("clarify", c)
        return c

    async def fail(self, message: str, where: str) -> dict:
        self.stats["failed"] += 1
        await self.emit("error", {"message": message, "where": where, "recoverable": False})
        return await self.finish(False, f"Mock run failed: {message}", "error")

    async def play(self, page, plan_request: str | None = None) -> dict:
        """Replay the script for `plan_request` (the request check's restatement, else the request itself)."""
        script, script_error = find_script(self.cfg.scripts_path, plan_request or self.request)
        elsewhere = recorded_elsewhere(script, self.start_url) if script is not None else None
        if elsewhere is not None:
            script = None
        t = time.perf_counter()
        await page.goto(self.start_url, wait_until="load", timeout=15000)
        timings: dict[str, float] = {}
        img = await settle(page, self.cfg.settle_ms, timings)
        log.debug("start page ready in %.0f ms", (time.perf_counter() - t) * 1000)
        await self.observe(page, img, None, "", (script or {}).get("start_screen", ""),
                           {"screenshot": timings["screenshot"]})
        if script is None:
            return await self.fallback(script_error, elsewhere)

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
                                     "thinking": thinking, "tool_calls": [{"name": tool, "arguments": args}],
                                     "model": MOCK_PLANNER, **MOCK_PROMPT})
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
                                              "model": f"{self.cfg.grounder_model}, mock", **MOCK_PROMPT},
                                overlay_path=overlay)
                click = {"x": x, "y": y}
            t = time.perf_counter()
            result = await perform(page, step, self.step)
            timings = {"act": (time.perf_counter() - t) * 1000}  # the mouse/keyboard action alone
            img = await settle(page, self.cfg.settle_ms, timings)
            changed = screen_changed(before, img)
            self.stats["actions"] += 1
            self.stats["clicks"] += 1 if click else 0
            await self.emit("action", {"name": tool, "arguments": args, "ok": True, "screen_changed": changed,
                                       **({"click": click} if click else {})},
                            overlay_path=overlay, timings_ms={"act": timings["act"], "settle": timings["settle"]})
            await self.pace(0.3)
            await self.observe(page, img, changed, result, step.get("screen_after", ""),
                               {"screenshot": timings["screenshot"]})
            want = step.get("path_after")
            got = urlsplit(page.url).path
            if want and got != want:
                raise ScriptError(f"mock script is stale: after step {self.step} the page is {got}, expected {want}. "
                                  "Re-run python -m console.tools.calibrate_mock")
            last_changed = changed
        return await self.finish(False, "The script ended without calling done.", "max_steps")

    async def fallback(self, script_error: str | None, elsewhere: str | None = None) -> dict:
        """No usable script: say so with done(success=false). `elsewhere` = the start_path of the script that
        matched the request text, when this run started on another page."""
        if script_error:
            await self.emit("error", {"message": script_error, "where": "mock_scripts", "recoverable": True})
        self.step = 1
        plan, summary = FALLBACK_PLAN, FALLBACK_SUMMARY
        if elsewhere is not None:
            here = show_start(self.start_url)
            plan = f"Mock agent: the scripted plan for this request starts on {elsewhere}, but this run started on {here}"
            summary = (f"This is the mock agent. Its script for this request starts on {elsewhere}, but this run "
                       f"started on {here}, so it did not do anything. Choose this task in the Sandbox task list to replay it.")
        args = {"summary": summary, "success": False}
        await self.pace(0.5)
        await self.emit("plan", {"content": plan, "reasoning": "", "thinking": False,
                                 "tool_calls": [{"name": "done", "arguments": args}],
                                 "model": MOCK_PLANNER, **MOCK_PROMPT})
        await self.emit("tool_call", {"name": "done", "arguments": args, "valid": True,
                                      "description": describe("done", args)})
        await self.emit("action", {"name": "done", "arguments": args, "ok": True, "screen_changed": None, "done": True})
        return await self.finish(False, summary, "done")

    def close(self) -> None:
        self._events.close()


async def run_task(request: str, *, start_url: str, cfg: MockConfig | None = None,
                   on_event: Callable[[MockStepEvent], Awaitable[None]] | None = None,
                   parent_run_id: str | None = None) -> dict:
    """Same contract as agent.core.run_task (contracts.md §5). Never raises, except CancelledError.
    `parent_run_id`: the needs_input run this follow-up answers (recorded in the start event)."""
    cfg = cfg or MockConfig()
    run = _Run(request, start_url, cfg, on_event)
    try:
        start = {"request": request, "start_url": start_url, "config": cfg.public_dict()}
        if parent_run_id:
            start["parent_run_id"] = parent_run_id
        await run.emit("start", start)
        plan_request = request
        try:
            if cfg.clarify_on:  # before the browser opens, like agent/core.py
                c = await run.check_request()
                if c["action"] == "ask":
                    return await run.finish(False, c["question"], "needs_input", question=c["question"])
                plan_request = c.get("normalized_request") or request
            async with open_page(cfg.headed) as page:  # closes the browser on the way out
                try:
                    return await run.play(page, plan_request)
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
