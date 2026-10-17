"""Picks the agent behind the console: `--agent mock` (console/mock_agent.py) or `--agent real` (T2's agent.core).

get_agent(settings) -> (run_task, make_cfg). Both agents share contracts.md §5:
    async run_task(request, *, start_url, cfg, on_event) -> RunResult
make_cfg(options) turns the console's per-run options {grounder, thinking, lightning}
into that agent's config object.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

RunTask = Callable[..., Awaitable[Any]]
MakeCfg = Callable[[dict], Any]


class AgentUnavailable(RuntimeError):
    """The requested agent can't be imported or configured."""


def get_agent(settings) -> tuple[RunTask, MakeCfg]:
    if settings.agent == "mock":
        from console.mock_agent import MockConfig, run_task

        def make_cfg(options: dict) -> MockConfig:
            return MockConfig(runs_dir=str(settings.runs_dir), sandbox_url=settings.sandbox_url,
                              grounder_model=options["grounder"], thinking=options["thinking"],
                              lightning=options["lightning"], headed=settings.headed,
                              step_delay_s=settings.step_delay_s, scripts_path=str(settings.mock_scripts))

        return run_task, make_cfg

    if settings.agent == "real":
        try:
            from agent.core import run_task
        except ImportError as e:
            raise AgentUnavailable(f"agent.core not available yet (T2); use --agent mock ({e})") from e
        from agent.config import load_config

        def make_cfg(options: dict):
            overrides = {
                "grounder_model": options["grounder"],
                "thinking": options["thinking"],
                "lightning": options["lightning"],
                "sandbox_url": settings.sandbox_url,
                **({"browser": {"headless": False}} if settings.headed else {}),
            }
            return load_config(overrides=overrides)

        return run_task, make_cfg

    raise AgentUnavailable(f"unknown agent {settings.agent!r}; use mock or real")
