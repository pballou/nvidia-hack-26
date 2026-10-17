"""Tool execution shared by the interactive planner (agent/core.py) and the Gym env routes.

One AgentSession = one browser context + a grounder + a summarizer. `execute(name, args)`
is the single definition of what each tool does, so the console agent and the Gym
`web_agent` env behave identically. It never raises for tool problems: bad arguments,
grounding failures and browser errors come back as `ok=False` outcomes that the model
sees as observations.

Vision-only: nothing here reads the DOM or the app state. The page is seen only through
screenshots (summarizer) and acted on only through pixel clicks and keys.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image

from agent.observation import Observation, build_observation, one_line
from agent.prompts import TOOL_NAMES, tool_parameters

DONE_NOTE = "The task is finished. Do not call any more tools; reply with one short sentence for the user."
ALREADY_DONE = "done was already called, so this action was not performed. Reply without calling a tool."


@dataclass
class Located:
    """Grounder result: a click point in viewport CSS px (None on failure)."""
    x: float | None
    y: float | None
    raw: str = ""
    point_1000: tuple[float, float] | None = None
    error: str | None = None
    model: str = ""
    latency_ms: float = 0.0


@dataclass
class Summary:
    text: str
    error: str | None = None
    model: str = ""
    latency_ms: float = 0.0


class Grounder(Protocol):
    async def locate(self, img: Image.Image, target: str) -> Located: ...


class Summarizer(Protocol):
    async def summarize(self, img: Image.Image) -> Summary: ...


class Browser(Protocol):
    """What AgentSession needs from agent/browser.py:BrowserSession."""
    async def screenshot(self) -> Image.Image: ...
    async def click(self, x: float, y: float) -> None: ...
    async def type(self, text: str) -> None: ...
    async def press(self, key: str) -> None: ...
    async def scroll(self, direction: str, amount: str) -> None: ...
    async def select_all(self) -> None: ...
    async def settle(self) -> None: ...
    async def url(self) -> str: ...
    async def title(self) -> str: ...
    def screen_changed(self, before: Image.Image, after: Image.Image) -> bool: ...


@dataclass
class ToolOutcome:
    name: str
    args: dict
    ok: bool
    observation: Observation | None = None
    error: str | None = None
    click: dict | None = None            # {x, y, target, point_1000, raw, model, error?}
    screen_changed: bool | None = None
    done: bool = False
    before: Image.Image | None = None    # what the grounder saw (for the overlay)
    after: Image.Image | None = None     # the new observation's screenshot
    timings_ms: dict[str, float] = field(default_factory=dict)

    def tool_payload(self) -> dict:
        """The tool result the model sees; identical for the interactive loop and Gym routes."""
        out: dict[str, Any] = {"ok": self.ok, "observation": self.observation.render() if self.observation else ""}
        if self.error:
            out["error"] = self.error
        if self.click and self.click.get("x") is not None:
            out["click"] = {"x": round(self.click["x"]), "y": round(self.click["y"])}
        return out

    def one_line(self) -> str:
        return one_line(self.name, self.args, self.ok, self.screen_changed, self.error)


def validate_args(name: str, args: Any) -> tuple[dict | None, str | None]:
    """Check tool arguments against agent/prompts.py's schema. Returns (clean_args, error)."""
    if name not in TOOL_NAMES:
        return None, f"unknown tool {name!r}; available tools: {', '.join(TOOL_NAMES)}"
    if not isinstance(args, dict):
        return None, f"arguments for {name} must be a JSON object"
    schema = tool_parameters(name)
    props, required = schema["properties"], schema.get("required", [])
    missing = [k for k in required if k not in args or args[k] is None]
    if missing:
        return None, f"{name} is missing required argument(s): {', '.join(missing)}"
    unknown = [k for k in args if k not in props]
    if unknown:
        return None, f"{name} got unknown argument(s): {', '.join(unknown)}; allowed: {', '.join(props)}"
    clean: dict = {}
    for key, value in args.items():
        spec = props[key]
        if value is None:
            continue
        if spec.get("type") == "string":
            if not isinstance(value, str):
                return None, f"{name}.{key} must be a string"
            if "enum" in spec and value not in spec["enum"]:
                return None, f"{name}.{key} must be one of {spec['enum']}, got {value!r}"
            if key != "text" and not value.strip():
                return None, f"{name}.{key} must not be empty"
        elif spec.get("type") == "boolean":
            if isinstance(value, str) and value.lower() in ("true", "false"):
                value = value.lower() == "true"
            if not isinstance(value, bool):
                return None, f"{name}.{key} must be true or false"
        clean[key] = value
    return clean, None


_KEY_ALIASES = {
    "enter": "Enter", "return": "Enter", "esc": "Escape", "escape": "Escape", "tab": "Tab",
    "backspace": "Backspace", "delete": "Delete", "del": "Delete", "space": "Space", "spacebar": "Space",
    "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
    "arrowup": "ArrowUp", "arrowdown": "ArrowDown", "arrowleft": "ArrowLeft", "arrowright": "ArrowRight",
    "pagedown": "PageDown", "pageup": "PageUp", "home": "Home", "end": "End",
}


def normalize_key(key: str) -> str:
    """'enter' -> 'Enter', 'ctrl+a' -> 'Control+a'; anything else passes through to Playwright."""
    k = key.strip()
    if "+" in k and len(k) > 1:
        mods = {"ctrl": "Control", "control": "Control", "cmd": "Meta", "meta": "Meta", "alt": "Alt",
                "option": "Alt", "shift": "Shift"}
        parts = [p.strip() for p in k.split("+")]
        return "+".join([mods.get(p.lower(), p) for p in parts[:-1]] + [_KEY_ALIASES.get(parts[-1].lower(), parts[-1])])
    return _KEY_ALIASES.get(k.lower(), k)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


class AgentSession:
    def __init__(self, browser: Browser, grounder: Grounder, summarizer: Summarizer):
        self.browser = browser
        self.grounder = grounder
        self.summarizer = summarizer
        self.step = 0
        self.finished: dict | None = None    # done() arguments once called
        self.last_screenshot: Image.Image | None = None
        self.last_observation: Observation | None = None
        self.n_actions = 0
        self.n_clicks = 0
        self.n_failed = 0

    async def observe(self, last_action_result: str = "", screen_changed: bool | None = None,
                      img: Image.Image | None = None) -> tuple[Observation, Image.Image, dict]:
        """Screenshot (unless given) + summary + url/title -> Observation for the current step."""
        timings: dict[str, float] = {}
        if img is None:
            t0 = time.perf_counter()
            img = await self.browser.screenshot()
            timings["screenshot"] = _ms(t0)
        t0 = time.perf_counter()
        summary = await self.summarizer.summarize(img)
        timings["summary"] = _ms(t0)
        text = summary.text if not summary.error else f"(screen description unavailable: {summary.error})"
        try:
            url, title = await self.browser.url(), await self.browser.title()
        except Exception as e:  # a page mid-navigation can't report its title
            url, title = "", f"(unavailable: {type(e).__name__})"
        obs = build_observation(step=self.step, url=url, title=title, screen_changed=screen_changed,
                                last_action_result=last_action_result, screen_summary=text)
        self.last_screenshot, self.last_observation = img, obs
        return obs, img, timings

    async def reset(self) -> tuple[Observation, Image.Image, dict]:
        """Initial observation (step 0) after the caller navigated to the start page."""
        self.step = 0
        self.finished = None
        t0 = time.perf_counter()
        await self.browser.settle()
        settle = _ms(t0)
        obs, img, timings = await self.observe()
        timings["settle"] = settle
        return obs, img, timings

    async def execute(self, name: str, args: Any) -> ToolOutcome:
        """Run one tool call. Never raises; failures are ok=False outcomes with an observation."""
        self.step += 1
        clean, err = validate_args(name, args)
        if err:
            return await self._failed(name, args if isinstance(args, dict) else {}, err)
        if self.finished is not None:
            return await self._failed(name, clean, ALREADY_DONE, observe=False)
        if name == "done":
            self.finished = {"summary": clean["summary"], "success": bool(clean["success"])}
            obs = build_observation(step=self.step, url=self.last_observation.url if self.last_observation else "",
                                    title=self.last_observation.title if self.last_observation else "",
                                    screen_changed=None, last_action_result=f"done recorded (success={self.finished['success']}). {DONE_NOTE}",
                                    screen_summary="(not described: task finished)")
            return ToolOutcome(name=name, args=clean, ok=True, observation=obs, done=True)
        try:
            return await self._act(name, clean)
        except Exception as e:  # browser died, navigation crashed, ...
            return await self._failed(name, clean, f"{name} failed: {type(e).__name__}: {e}")

    async def _act(self, name: str, args: dict) -> ToolOutcome:
        timings: dict[str, float] = {}
        before = self.last_screenshot
        if before is None:
            t0 = time.perf_counter()
            before = await self.browser.screenshot()
            timings["screenshot"] = _ms(t0)
        click = None
        target = args.get("target_description")
        if name == "click" or (name == "type_text" and target):
            t0 = time.perf_counter()
            loc = await self.grounder.locate(before, target)
            timings["ground"] = _ms(t0)
            click = {"x": loc.x, "y": loc.y, "target": target, "point_1000": loc.point_1000, "raw": loc.raw,
                     "model": loc.model}
            if loc.error or loc.x is None:
                click["error"] = loc.error or "no point"
                return await self._failed(name, args, f"could not locate {target!r}: {click['error']}",
                                          click=click, before=before, timings=timings, observe=False)

        t0 = time.perf_counter()
        if name == "click":
            await self.browser.click(click["x"], click["y"])
            self.n_clicks += 1
            result = f'clicked "{target}" at ({round(click["x"])}, {round(click["y"])})'
        elif name == "type_text":
            if click:
                await self.browser.click(click["x"], click["y"])
                self.n_clicks += 1
                await self.browser.select_all()
            await self.browser.type(args["text"])
            result = f'typed "{args["text"][:80]}"' + (f' into "{target}"' if target else "")
        elif name == "press_key":
            key = normalize_key(args["key"])
            await self.browser.press(key)
            result = f"pressed {key}"
        else:  # scroll
            await self.browser.scroll(args["direction"], args["amount"])
            result = f"scrolled {args['direction']} ({args['amount']})"
        timings["act"] = _ms(t0)
        self.n_actions += 1

        t0 = time.perf_counter()
        await self.browser.settle()
        timings["settle"] = _ms(t0)
        t0 = time.perf_counter()
        after = await self.browser.screenshot()
        timings["screenshot"] = timings.get("screenshot", 0.0) + _ms(t0)
        changed = self.browser.screen_changed(before, after)
        obs, after, obs_t = await self.observe(result, changed, img=after)
        timings.update({k: v for k, v in obs_t.items() if k != "screenshot"})
        return ToolOutcome(name=name, args=args, ok=True, observation=obs, click=click, screen_changed=changed,
                           before=before, after=after, timings_ms=timings)

    async def _failed(self, name: str, args: dict, error: str, *, click: dict | None = None,
                      before: Image.Image | None = None, timings: dict | None = None,
                      observe: bool = False) -> ToolOutcome:
        """ok=False outcome. The screen is unchanged by a failed call, so the last description is reused."""
        self.n_failed += 1
        timings = dict(timings or {})
        prev = self.last_observation
        if observe or prev is None:
            try:
                obs, img, t = await self.observe(f"{name} failed: {error}", None)
                timings.update(t)
            except Exception as e:
                obs = build_observation(step=self.step, url="", title="", screen_changed=None,
                                        last_action_result=f"{name} failed: {error}",
                                        screen_summary=f"(screen unavailable: {type(e).__name__})")
        else:
            obs = build_observation(step=self.step, url=prev.url, title=prev.title, screen_changed=None,
                                    last_action_result=f"{name} failed: {error}", screen_summary=prev.screen_summary)
            self.last_observation = obs
        return ToolOutcome(name=name, args=args, ok=False, observation=obs, error=error, click=click,
                           before=before, after=self.last_screenshot, timings_ms=timings)
