"""run_task (docs/tasks/contracts.md §5): the interactive Lightning loop over one browser session.

Per step: Lightning proposes one tool call (planner) -> AgentSession executes it
(grounder click / keys, settle, screenshot, summary) -> the tool result goes back to
Lightning. Every Lightning call counts as a step (like Gym simple_agent's max_steps);
`cfg.max_steps` is a hard stop. Events stream to `on_event` and runs/<run_id>/.

Tool calls are validated against agent/prompts.py's schema before anything runs
(vllm-metal doesn't enforce it). An invalid call executes nothing; its tool result is a
corrective message quoting the expected signature, and it still uses up a step.

Thinking: `on` thinks on every call; `retry` thinks on the call after a problem step
(failed or invalid call, no change, no tool call), but never on two calls in a row because
of a reply without a tool call (a thinking reply cut off at the token limit would otherwise
keep thinking on until max_steps).

The start event's config.lightning also gets `root`: the checkpoint the endpoint reports
serving (vLLM /v1/models), since the URL and served name don't identify the build.

Done-event stats:
  n_tool_calls      replies with a tool call (only the first call per reply is handled)
  n_invalid_calls   handled calls rejected by validation: malformed JSON, unknown tool,
                    wrong/missing/extra argument names, bad enum or type. Nothing ran.
                    Tool-call validity rate = 1 - n_invalid_calls / n_tool_calls.
  n_actions         valid calls executed (n_tool_calls - n_invalid_calls)
  n_clicks          pixel clicks performed (click, and type_text with a target)
  n_failed_actions  executed calls that returned ok=false (grounding or browser failures)
  n_no_tool_call    replies without any tool call
  n_truncated       replies cut off at max_tokens (finish_reason "length")
  n_no_change       ok actions after which the screen did not change
  n_recoveries      problem steps (invalid call, no tool call, failed action, no change)
                    later followed by an ok, screen-changing action
  n_thinking_calls  Lightning calls with thinking enabled
  n_api_errors      Lightning calls that failed (after the planner's retries and rate-limit waits)
  latency_ms        {stage: [ms, ...]} for plan, ground, act, settle, screenshot, summary
                    (plan excludes rate-limit waits; the plan event's wait_ms has them)

Vision-only: nothing here touches the DOM or the sandbox state.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from PIL import Image

from agent.config import AgentConfig
from agent.observation import Observation
from agent.planner import NOT_EXECUTED, LightningPlanner, PlanResult, ToolCall
from agent.prompts import TOOL_NAMES
from agent.session import AgentSession, ToolOutcome, invalid_json_error, normalize_key, validate_args
from agent.trace import RunResult, StepEvent, TraceWriter

log = logging.getLogger(__name__)

NO_TOOL_CALL_CORRECTION = ("You replied without calling a tool. Call exactly one tool: click, type_text, "
                           "press_key, scroll, or done.")
TRUNCATED_CORRECTION = ("Your reply was cut off at the length limit before it called a tool. Call exactly one "
                        "tool now (click, type_text, press_key, scroll, or done), without long deliberation.")
MAX_CONSECUTIVE_API_ERRORS = 2
REASONING_PREVIEW_CHARS = 600
CLEANUP_EMIT_TIMEOUT_S = 10.0

EventCallback = Callable[[StepEvent], Awaitable[None]]


def describe_call(name: str, args: dict | None) -> str:
    """Plain words for the console, e.g. 'Clicking: the gear icon in the top-right toolbar'."""
    a = args if isinstance(args, dict) else {}
    if name == "click":
        return f"Clicking: {a.get('target_description', '?')}"
    if name == "type_text":
        target = a.get("target_description")
        return f'Typing: "{a.get("text", "")}"' + (f" into {target}" if target else "")
    if name == "press_key":
        return f"Pressing {normalize_key(str(a.get('key', '?')))}"
    if name == "scroll":
        return f"Scrolling {a.get('direction', '?')}" + (" one page" if a.get("amount") == "page" else "")
    if name == "done":
        return f"Finishing: {a.get('summary', '')}"
    return f"Unknown action: {name}"


def _invalid_description(name: str) -> str:
    return f"Unknown action: {name}" if name not in TOOL_NAMES else f"Invalid {name} call (not executed)"


def _invalid_line(call: ToolCall) -> str:
    """One-liner for the context budget; shows the argument names the model used."""
    if call.arguments is None:
        shown = "<invalid JSON>"
    else:
        shown = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)[:60]}" for k, v in call.arguments.items())
    return f"{call.name}({shown}) -> invalid call, not executed"


def _pick(timings: dict[str, float], *keys: str) -> dict[str, float] | None:
    out = {k: timings[k] for k in keys if timings.get(k) is not None}
    return out or None


async def aclose(obj: Any) -> None:
    """Best-effort close (aclose() or close(), sync or async); never raises."""
    for name in ("aclose", "close"):
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                res = fn()
                if inspect.isawaitable(res):
                    await asyncio.wait_for(res, timeout=10)
            except Exception as e:
                log.warning("closing %s failed: %s: %s", type(obj).__name__, type(e).__name__, e)
            return


async def _shielded(aw: Awaitable[Any]) -> bool:
    """Await `aw` to the end even if this task is cancelled meanwhile; True if it was."""
    task = asyncio.ensure_future(aw)
    cancelled = False
    while True:
        try:
            await asyncio.shield(task)
            return cancelled
        except asyncio.CancelledError:
            if task.cancelled():  # `aw` itself was cancelled
                return True
            cancelled = True  # ours: keep waiting for `aw`


class _Run:
    def __init__(self, cfg: AgentConfig, trace: TraceWriter, on_event: EventCallback | None):
        self.cfg, self.trace, self.on_event = cfg, trace, on_event
        self.steps = 0
        self.reason = "max_steps"
        self.error: str | None = None
        self.finished: dict | None = None
        self.session: AgentSession | None = None
        self.to_close: list[Any] = []   # browser session, planner, VLM client (whatever we created)
        self.stats: dict[str, Any] = {k: 0 for k in (
            "n_tool_calls", "n_invalid_calls", "n_actions", "n_clicks", "n_failed_actions", "n_no_tool_call",
            "n_truncated", "n_no_change", "n_recoveries", "n_thinking_calls", "n_api_errors")}
        self.latency: dict[str, list[float]] = {}
        self._problem_pending = False

    async def emit(self, step: int, kind: str, payload: dict, *, screenshot_path: str | None = None,
                   overlay_path: str | None = None, timings_ms: dict | None = None) -> None:
        timings = {k: v for k, v in (timings_ms or {}).items() if v is not None} or None
        ev = self.trace.event(step, kind, payload, screenshot_path=screenshot_path, overlay_path=overlay_path,
                              timings_ms=timings)
        if self.on_event is not None:
            try:
                await self.on_event(ev)
            except Exception:  # a broken console must never break the run
                log.exception("on_event callback failed for %s event", kind)

    def _time(self, timings: dict[str, float]) -> None:
        for stage, ms in timings.items():
            if ms is not None:
                self.latency.setdefault(stage, []).append(round(ms, 1))

    async def _observation(self, step: int, obs: Observation, img: Image.Image | None,
                           timings: dict | None) -> None:
        shot = self.trace.save_screenshot(step, img) if img is not None else None
        await self.emit(step, "observation", {"observation": obs.to_dict(), "text": obs.render()},
                        screenshot_path=shot, timings_ms=timings)

    async def run(self, request: str, start_url: str, planner_client: Any, grounder: Any, summarizer: Any,
                  pool: Any) -> None:
        cfg = self.cfg
        start = {"request": request, "start_url": start_url, "config": cfg.public_dict()}
        try:
            planner = LightningPlanner(cfg, client=planner_client)  # raises without an API key
            self.to_close.append(planner)
            start["config"]["lightning"].update(await planner.server_info())
        finally:  # every run's events begin with start, even if it fails or is cancelled here
            await self.emit(0, "start", start)
        if pool is None:
            from agent import browser  # lazy: tests inject a fake pool

            pool = await browser.get_pool(cfg.browser)
        bs = await pool.new_session(cfg.viewport, navigation_timeout_s=cfg.timeouts.navigation_s)
        self.to_close.insert(0, bs)
        await bs.goto(start_url)
        grounder, summarizer = self._models(bs, grounder, summarizer)
        session = self.session = AgentSession(bs, grounder, summarizer)

        obs, img, timings = await session.reset()
        self._time(timings)
        await self._observation(0, obs, img, timings)
        planner.start(request, obs.render())

        thinking_next = cfg.thinking == "on"
        api_errors = 0
        for step in range(1, cfg.max_steps + 1):
            self.steps = step
            plan = await planner.propose(thinking_next)
            self.stats["n_thinking_calls"] += int(plan.thinking)
            self.stats["n_truncated"] += int(plan.finish_reason == "length")
            self._time({"plan": plan.latency_ms})
            await self.emit(step, "plan", self._plan_payload(plan), timings_ms={"plan": plan.latency_ms})

            if plan.error:
                api_errors += 1
                self.stats["n_api_errors"] += 1
                fatal = api_errors >= MAX_CONSECUTIVE_API_ERRORS
                await self.emit(step, "error", {"message": plan.error, "where": "planner", "recoverable": not fatal})
                if fatal:
                    self.reason, self.error = "error", plan.error
                    return
                continue
            api_errors = 0

            if not plan.tool_calls:
                self.stats["n_no_tool_call"] += 1
                self._problem_pending = True
                truncated = plan.finish_reason == "length"
                planner.record_correction(plan, TRUNCATED_CORRECTION if truncated else NO_TOOL_CALL_CORRECTION)
                message = ("reply was cut off at the token limit before a tool call" if truncated
                           else "reply had no tool call") + "; asked Lightning to call one"
                await self.emit(step, "error", {"message": message, "where": "planner", "recoverable": True,
                                                "finish_reason": plan.finish_reason})
                # retry: think next only if this reply didn't already think (never twice in a row here)
                thinking_next = cfg.thinking == "on" or (cfg.thinking == "retry" and not plan.thinking)
                continue

            outcome = await self._run_call(step, plan, planner)
            if outcome is not None and outcome.done:
                self.reason, self.finished = "done", session.finished
                return
            ok = outcome is not None and outcome.ok
            no_change = outcome is not None and outcome.screen_changed is False
            thinking_next = cfg.thinking == "on" or (cfg.thinking == "retry" and (not ok or no_change))
        self.reason = "max_steps"

    async def _run_call(self, step: int, plan: PlanResult, planner: LightningPlanner) -> ToolOutcome | None:
        """Validate the first tool call of `plan`, execute it if valid, and record the exchange.

        An invalid call (bad JSON, unknown tool, wrong/missing/extra argument names, bad
        enum or type) executes nothing: its tool result is the corrective message plus the
        unchanged last observation. Returns None in that case.
        """
        session = self.session
        call, extras = plan.tool_calls[0], plan.tool_calls[1:]
        extra_results = [(c.id, {"ok": False, "error": NOT_EXECUTED}) for c in extras]
        self.stats["n_tool_calls"] += 1

        if call.parse_error:
            err = invalid_json_error(call.name, call.parse_error)
        else:
            _, err = validate_args(call.name, call.arguments)
        tc_payload: dict[str, Any] = {
            "name": call.name, "arguments": call.raw_arguments if call.parse_error else call.arguments,
            "valid": err is None,
            "description": describe_call(call.name, call.arguments) if err is None else _invalid_description(call.name)}
        if err:
            tc_payload["error"] = err
        if call.from_content:
            tc_payload["from_content"] = True
        await self.emit(step, "tool_call", tc_payload)

        if err:
            self.stats["n_invalid_calls"] += 1
            self._problem_pending = True
            last = session.last_observation.render() if session.last_observation else ""
            planner.record_step(plan, [(call.id, {"ok": False, "error": err, "observation": last}), *extra_results],
                                _invalid_line(call))
            return None

        self.stats["n_actions"] += 1
        session.step = step - 1  # observation numbers follow Lightning calls, like the events
        outcome = await session.execute(call.name, call.arguments)
        t = outcome.timings_ms
        self._time(t)

        if outcome.click:
            c = outcome.click
            overlay = None
            if outcome.before is not None:
                overlay = self.trace.save_overlay(step, outcome.before, c.get("x"), c.get("y"), c.get("target") or "")
            g = {"target": c.get("target"), "x": c.get("x"), "y": c.get("y"), "point_1000": c.get("point_1000"),
                 "raw": (c.get("raw") or "")[:300], "model": c.get("model")}
            if c.get("error"):
                g["error"] = c["error"]
            await self.emit(step, "grounding", g, overlay_path=overlay, timings_ms=_pick(t, "ground"))

        action: dict[str, Any] = {"name": outcome.name, "arguments": outcome.args, "ok": outcome.ok,
                                  "screen_changed": outcome.screen_changed, "done": outcome.done}
        if outcome.error:
            action["error"] = outcome.error
        if outcome.click and outcome.click.get("x") is not None:
            action["click"] = {"x": round(outcome.click["x"]), "y": round(outcome.click["y"])}
        await self.emit(step, "action", action, timings_ms=_pick(t, "act", "settle"))

        if outcome.observation is not None and not outcome.done:  # after=None: the screen was unavailable
            await self._observation(step, outcome.observation, outcome.after, _pick(t, "screenshot", "summary"))

        planner.record_step(plan, [(call.id, outcome.tool_payload()), *extra_results], outcome.one_line())

        if not outcome.ok:
            self.stats["n_failed_actions"] += 1
            self._problem_pending = True
        elif outcome.screen_changed is False:
            self.stats["n_no_change"] += 1
            self._problem_pending = True
        elif outcome.screen_changed and self._problem_pending:
            self.stats["n_recoveries"] += 1
            self._problem_pending = False
        return outcome

    def _plan_payload(self, plan: PlanResult) -> dict:
        reasoning = plan.reasoning
        if reasoning and len(reasoning) > REASONING_PREVIEW_CHARS:
            reasoning = reasoning[:REASONING_PREVIEW_CHARS] + "..."
        out = {"content": plan.content, "reasoning": reasoning, "thinking": plan.thinking,
               "tool_calls": [{"name": c.name, "arguments": c.arguments if c.arguments is not None else c.raw_arguments}
                              for c in plan.tool_calls],
               "usage": plan.usage}
        if plan.finish_reason:
            out["finish_reason"] = plan.finish_reason
        if plan.wait_ms:
            out["wait_ms"] = round(plan.wait_ms)
        return out

    def _models(self, bs: Any, grounder: Any, summarizer: Any) -> tuple[Any, Any]:
        if grounder is not None and summarizer is not None:
            return grounder, summarizer
        if self.cfg.use_oracle_grounder:
            from agent.testing.oracle_grounder import OracleGrounder  # test/debug only; guards itself

            default = OracleGrounder(bs)
        else:
            from agent.vlm import VLMClient

            default = VLMClient(self.cfg)
        self.to_close.append(default)
        return grounder or default, summarizer or default

    async def finish(self) -> RunResult:
        """Close resources, emit the done event, close the trace.

        Never raises, except that a cancel arriving meanwhile (Stop clicked twice) is deferred
        until the cleanup is complete and then re-raised.
        """
        cancelled = False
        for obj in self.to_close:
            cancelled |= await _shielded(aclose(obj))
        if self.session is not None:
            self.stats["n_clicks"] = self.session.n_clicks
        success = bool(self.finished and self.finished.get("success")) if self.reason == "done" else False
        if self.reason == "done" and self.finished:
            summary = self.finished.get("summary", "")
        elif self.reason == "max_steps":
            summary = f"Stopped: reached the {self.cfg.max_steps}-step limit before finishing."
        elif self.reason == "stopped":
            summary = "Stopped by the user."
        else:
            summary = f"Stopped after an error: {self.error or 'unknown error'}"
        stats = dict(self.stats, latency_ms=self.latency)
        done = {"success_claimed": success, "summary": summary, "steps": self.steps, "reason": self.reason,
                "stats": stats}
        try:  # the done line is written before on_event runs, so a hung console only delays it
            cancelled |= await _shielded(asyncio.wait_for(self.emit(self.steps, "done", done), CLEANUP_EMIT_TIMEOUT_S))
        except Exception:
            log.exception("could not write the done event")
        finally:
            self.trace.close()
        result = RunResult(run_id=self.trace.run_id, success_claimed=success, summary=summary, steps=self.steps,
                           trace_dir=str(self.trace.dir))
        if cancelled:
            raise asyncio.CancelledError
        return result


async def run_task(request: str, *, start_url: str, cfg: AgentConfig,
                   on_event: EventCallback | None = None,
                   planner_client: Any = None, grounder: Any = None, summarizer: Any = None, pool: Any = None,
                   run_id: str | None = None) -> RunResult:
    """Run one request in a fresh browser session (contracts.md §5).

    Optional injection points (tests, console): `planner_client` (an AsyncOpenAI-like
    client for Lightning), `grounder` / `summarizer` (agent.session protocols), `pool`
    (anything with `async new_session(viewport, *, navigation_timeout_s)`; default
    agent.browser.get_pool) and
    `run_id`. Cancelling the task (console Stop) emits done(reason='stopped') and re-raises.
    """
    run = _Run(cfg, TraceWriter(cfg.run_dir_root, run_id), on_event)
    try:
        await run.run(request, start_url, planner_client, grounder, summarizer, pool)
    except asyncio.CancelledError:
        run.reason = "stopped"
        raise
    except Exception as e:
        run.reason, run.error = "error", f"{type(e).__name__}: {e}"[:500]
        log.exception("run %s failed", run.trace.run_id)
        await run.emit(run.steps, "error", {"message": run.error, "where": "run", "recoverable": False})
    finally:
        result = await run.finish()
    return result
