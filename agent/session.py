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
    prompt_sha: str = ""             # agent.prompts.prompt_sha of the grounding prompt template
    usage: dict | None = None        # {prompt_tokens, completion_tokens} when the server reports it


@dataclass
class Summary:
    text: str
    error: str | None = None
    model: str = ""
    latency_ms: float = 0.0
    prompt_sha: str = ""             # agent.prompts.prompt_sha of the summary prompt
    usage: dict | None = None


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
    invalid: bool = False                # rejected by validate_args (schema mismatch); nothing ran
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
    """Check tool arguments against agent/prompts.py's schema. Returns (clean_args, error).

    Errors are corrective messages for the model: vllm-metal doesn't enforce the schema, so
    Lightning sometimes invents tool or argument names; tell it exactly what to send instead.
    Keys are never renamed silently (that would hide the mistake from the validity metric).
    """
    if name not in TOOL_NAMES:
        return None, (f"unknown tool {name!r}. The only tools are: "
                      f"{'; '.join(tool_signature(n) for n in TOOL_NAMES)}. Call one of them.")
    clean, err = _check_args(name, args)
    if err:
        return None, f"{err}. Expected {tool_signature(name)}. Call {name} again with exactly these argument names."
    return clean, None


def invalid_json_error(name: str, detail: str) -> str:
    """Corrective message for tool-call arguments that are not a JSON object."""
    expected = f" Expected {tool_signature(name)}." if name in TOOL_NAMES else ""
    return f"invalid JSON arguments: {detail}.{expected} Call the tool again with a valid JSON object."


def tool_signature(name: str) -> str:
    """'click(target_description: string)'; optional args get a '?' and enums list their values."""
    schema = tool_parameters(name)
    parts = []
    for key, spec in schema["properties"].items():
        kind = "|".join(f'"{v}"' for v in spec["enum"]) if "enum" in spec else spec.get("type", "any")
        parts.append(f"{key}{'' if key in schema.get('required', []) else '?'}: {kind}")
    return f"{name}({', '.join(parts)})"


def _check_args(name: str, args: Any) -> tuple[dict | None, str | None]:
    if not isinstance(args, dict):
        return None, f"arguments for {name} must be a JSON object"
    schema = tool_parameters(name)
    props, required = schema["properties"], schema.get("required", [])
    unknown = [k for k in args if k not in props]
    missing = [k for k in required if k not in args or args[k] is None]
    if unknown or missing:
        problems = []
        if unknown:
            problems.append(f"unknown argument(s) {', '.join(unknown)}")
        if missing:
            problems.append(f"missing required argument(s) {', '.join(missing)}")
        return None, f"{name} got {' and '.join(problems)}"
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
                if key not in required:  # schema-valid: an empty optional string means "not given"
                    continue
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


_MODIFIERS = {"ctrl": "Control", "control": "Control", "cmd": "Meta", "command": "Meta", "meta": "Meta",
              "super": "Meta", "win": "Meta", "alt": "Alt", "option": "Alt", "opt": "Alt", "shift": "Shift"}
# Select all / copy / paste / cut / undo / redo: the model writes Ctrl or Cmd for the same intent, but
# Chromium on macOS binds them to Meta (Control+A moves to the line start there). Playwright's
# ControlOrMeta is Meta on macOS and Control elsewhere.
_EDIT_SHORTCUT_KEYS = frozenset("acvxyz")


def normalize_key(key: str) -> str:
    """'enter' -> 'Enter', 'ctrl+a' / 'cmd+a' -> 'ControlOrMeta+a', 'ctrl+enter' -> 'Control+Enter'.

    Anything else passes through to Playwright (unknown names raise there).
    """
    k = key.strip()
    if "+" in k and len(k) > 1:
        parts = [p.strip() for p in k.split("+")]
        mods = [_MODIFIERS.get(p.lower(), p) for p in parts[:-1]]
        last = _KEY_ALIASES.get(parts[-1].lower(), parts[-1])
        if last.lower() in _EDIT_SHORTCUT_KEYS:
            mods = ["ControlOrMeta" if m in ("Control", "Meta") else m for m in mods]
        return "+".join(mods + [last])
    return _KEY_ALIASES.get(k.lower(), k)


def _brief(e: BaseException) -> str:
    """'TimeoutError: Timeout 15000ms exceeded.' (first line only: Playwright appends a call log)."""
    lines = str(e).strip().splitlines()
    return f"{type(e).__name__}: {lines[0][:200]}" if lines else type(e).__name__


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
        self.last_summary: Summary | None = None  # the summarizer's last result (model, prompt_sha, usage)
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
        summary = self.last_summary = await self.summarizer.summarize(img)
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

    def prime(self, img: Image.Image, observation: Observation) -> None:
        """Step 0 from a precomputed observation (Gym rows bake it in), without a VLM call."""
        self.step = 0
        self.finished = None
        self.last_screenshot, self.last_observation = img, observation

    async def execute(self, name: str, args: Any) -> ToolOutcome:
        """Run one tool call. Never raises; failures are ok=False outcomes with an observation."""
        self.step += 1
        clean, err = validate_args(name, args)
        if err:
            outcome = await self._failed(name, args if isinstance(args, dict) else {}, err)
            outcome.invalid = True
            return outcome
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
        except Exception as e:  # browser died, navigation crashed, ...: describe the screen as it is now
            return await self._failed(name, clean, _brief(e), observe=True)

    async def _act(self, name: str, args: dict) -> ToolOutcome:
        timings: dict[str, float] = {}
        # The live screen, not the last observation's screenshot: the page may have moved on while
        # Lightning planned (a toast faded, content loaded), so the click point and the change check
        # both use the screen as it is now.
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
                     "model": loc.model, "prompt_sha": loc.prompt_sha, "usage": loc.usage}
            if loc.error or loc.x is None:
                click["error"] = loc.error or "no point"
                return await self._failed(name, args, f"could not locate {target!r}: {click['error']}",
                                          click=click, before=before, timings=timings, observe=False)

        baseline = before  # what the result is compared with for screen_changed
        shot_ms = 0.0
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
                # Judge the typing, not the click: when the click misses the field, select-all
                # highlights the whole page, which alone would count as a change.
                t1 = time.perf_counter()
                baseline = await self.browser.screenshot()
                shot_ms = _ms(t1)
            await self.browser.type(args["text"])
            result = f'typed "{args["text"][:80]}"' + (f' into "{target}"' if target else "")
        elif name == "press_key":
            key = normalize_key(args["key"])
            await self.browser.press(key)
            result = f"pressed {key}"
        else:  # scroll
            await self.browser.scroll(args["direction"], args["amount"])
            result = f"scrolled {args['direction']} ({args['amount']})"
        timings["act"] = _ms(t0) - shot_ms
        timings["screenshot"] += shot_ms
        self.n_actions += 1

        try:
            t0 = time.perf_counter()
            await self.browser.settle()
            timings["settle"] = _ms(t0)
            t0 = time.perf_counter()
            after = await self.browser.screenshot()
            timings["screenshot"] += _ms(t0)
            changed = self.browser.screen_changed(baseline, after)
            if name == "type_text" and click and args["text"] and not changed:
                result += ", but the text did not appear (the click probably missed the field)"
            obs, after, obs_t = await self.observe(result, changed, img=after)
        except Exception as e:  # the action ran; only capturing its result failed
            return await self._uncaptured(name, args, result, e, click=click, before=before, timings=timings)
        timings.update({k: v for k, v in obs_t.items() if k != "screenshot"})
        return ToolOutcome(name=name, args=args, ok=True, observation=obs, click=click, screen_changed=changed,
                           before=before, after=after, timings_ms=timings)

    async def _uncaptured(self, name: str, args: dict, result: str, exc: Exception, *, click: dict | None,
                          before: Image.Image, timings: dict) -> ToolOutcome:
        """The action ran, but settling or the screenshot after it raised (e.g. mid-navigation).

        Not reported as failed: the model would repeat an action that already happened (and
        flip a toggle back). The screen is described afresh if it can be captured now.
        """
        self.last_screenshot = None
        note = f"{result}; the screen could not be captured right after it ({_brief(exc)})"
        try:
            obs, _, t = await self.observe(note, None)
            timings["screenshot"] += t.pop("screenshot", 0.0)
            timings.update(t)
        except Exception as e:
            obs = build_observation(step=self.step, url="", title="", screen_changed=None, last_action_result=note,
                                    screen_summary=f"(screen unavailable: {_brief(e)})")
            self.last_observation = obs
        return ToolOutcome(name=name, args=args, ok=True, observation=obs, click=click, screen_changed=None,
                           before=before, after=self.last_screenshot, timings_ms=timings)

    async def _failed(self, name: str, args: dict, error: str, *, click: dict | None = None,
                      before: Image.Image | None = None, timings: dict | None = None,
                      observe: bool = False) -> ToolOutcome:
        """ok=False outcome. Nothing ran, so the last description is reused unless `observe` (a
        browser error, after which the screen may have changed anyway) asks for a fresh one."""
        self.n_failed += 1
        timings = dict(timings or {})
        prev = self.last_observation
        if observe or prev is None:
            try:
                obs, img, t = await self.observe(f"{name} failed: {error}", None)
                timings.update(t)
            except Exception as e:
                self.last_screenshot = None  # stale now
                obs = build_observation(step=self.step, url="", title="", screen_changed=None,
                                        last_action_result=f"{name} failed: {error}",
                                        screen_summary=f"(screen unavailable: {_brief(e)})")
                self.last_observation = obs
        else:
            obs = build_observation(step=self.step, url=prev.url, title=prev.title, screen_changed=None,
                                    last_action_result=f"{name} failed: {error}", screen_summary=prev.screen_summary)
            self.last_observation = obs
        return ToolOutcome(name=name, args=args, ok=False, observation=obs, error=error, click=click,
                           before=before, after=self.last_screenshot, timings_ms=timings)
