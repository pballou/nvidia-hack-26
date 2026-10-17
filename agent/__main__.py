"""CLI: run one request with the interactive agent and print one line per event.

  python -m agent "Turn on high contrast" --start-url "http://127.0.0.1:8090/settings?sid=demo"
  python -m agent "Show my notifications" --lightning local --grounder grounder --thinking off --headed

--use-oracle-grounder swaps the VLM for the test-only DOM oracle (debug of loop
mechanics only; never for evals or the demo). Nothing here reads the DOM or app state.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from agent.config import SERVED_NAMES, THINKING_MODES, AgentConfig, load_config
from agent.core import run_task
from agent.trace import StepEvent


def add_model_args(p: argparse.ArgumentParser) -> None:
    """Model/config flags shared with scripts/run_agent_tasks.py."""
    p.add_argument("--config", help="agent yaml (default: $AGENT_CONFIG or configs/agent.yaml)")
    p.add_argument("--grounder", choices=SERVED_NAMES, help="served name for clicks (grounder_model)")
    p.add_argument("--summary", choices=SERVED_NAMES, help="served name for screen summaries (summary_model)")
    p.add_argument("--thinking", choices=THINKING_MODES, help="Lightning thinking: off | on | retry")
    p.add_argument("--lightning", choices=("hosted", "local"), help="Lightning endpoint preset")
    p.add_argument("--max-steps", type=int, help="hard stop on Lightning calls per task")
    p.add_argument("--headed", action="store_true", help="show the browser window")
    p.add_argument("--use-oracle-grounder", action="store_true",
                   help="DEBUG ONLY: locate targets via the DOM oracle instead of the VLM")


def config_from_args(args: argparse.Namespace) -> AgentConfig:
    """load_config + CLI overrides. --use-oracle-grounder also unlocks the oracle's guard."""
    overrides: dict = {}
    for flag, key in (("grounder", "grounder_model"), ("summary", "summary_model"), ("thinking", "thinking"),
                      ("lightning", "lightning"), ("max_steps", "max_steps")):
        if getattr(args, flag, None) is not None:
            overrides[key] = getattr(args, flag)
    if args.headed:
        overrides["browser"] = {"headless": False}
    if args.use_oracle_grounder:
        from agent.testing.oracle_grounder import enable_debug_oracle

        enable_debug_oracle()
        overrides["use_oracle_grounder"] = True
    return load_config(args.config, overrides)


def _ms(t: dict | None) -> str:
    return " (" + ", ".join(f"{k} {v:.0f}ms" for k, v in t.items()) + ")" if t else ""


def format_event(ev: StepEvent) -> str:
    """One compact line per event."""
    p = ev.payload
    if ev.kind == "start":
        text = f"{p.get('request')!r} @ {p.get('start_url')}"
    elif ev.kind == "observation":
        o = p.get("observation", {})
        changed = {True: "changed", False: "NO change", None: ""}[o.get("screen_changed")]
        text = f"{o.get('title') or '(no title)'} | {o.get('last_action_result') or 'initial screen'} {changed}".rstrip()
    elif ev.kind == "plan":
        calls = ", ".join(f"{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)[:120]})"
                          for c in p.get("tool_calls", [])) or f"no tool call: {(p.get('content') or '')[:100]!r}"
        text = f"{calls}  thinking={'on' if p.get('thinking') else 'off'}"
    elif ev.kind == "tool_call":
        text = p.get("description", "") + ("" if p.get("valid") else f"  [invalid: {p.get('error')}]")
    elif ev.kind == "grounding":
        where = f"({p['x']:.0f}, {p['y']:.0f})" if p.get("x") is not None else "no point"
        text = f"{p.get('target')!r} -> {where} model={p.get('model')}" + (f"  error: {p['error']}" if p.get("error") else "")
    elif ev.kind == "action":
        if p.get("ok"):
            changed = {True: "screen changed", False: "NO change", None: ""}[p.get("screen_changed")]
            text = f"{p.get('name')} ok {changed}".rstrip()
        else:
            text = f"{p.get('name')} FAILED: {p.get('error')}"
    elif ev.kind == "error":
        text = f"[{p.get('where')}{'' if p.get('recoverable') else ', fatal'}] {p.get('message')}"
    else:  # done
        text = (f"reason={p.get('reason')} success_claimed={p.get('success_claimed')} steps={p.get('steps')}: "
                f"{p.get('summary')}")
    return f"[{ev.step:02d}] {ev.kind:<11} {text}{_ms(ev.timings_ms)}"


async def print_event(ev: StepEvent) -> None:
    print(format_event(ev), flush=True)


async def _amain(args: argparse.Namespace) -> int:
    from agent import browser  # Playwright; not needed for format_event / config_from_args

    cfg = config_from_args(args)

    pool = await browser.get_pool(cfg.browser)
    try:
        result = await run_task(args.request, start_url=args.start_url or cfg.sandbox_url.rstrip("/") + "/",
                                cfg=cfg, on_event=print_event, pool=pool)
    finally:
        await browser.close_pool()
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m agent", description=__doc__.split("\n\n")[0])
    p.add_argument("request", help="what the user wants, in their words")
    p.add_argument("--start-url", help="page to start on (default: the sandbox root)")
    add_model_args(p)
    args = p.parse_args(argv)
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
