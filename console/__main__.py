"""python -m console --agent mock|real [--lightning local|hosted] [--host 127.0.0.1] [--port 8095] [--sandbox-url URL] [--runs-dir runs] [--headed] [--step-delay 0.8]"""

from __future__ import annotations

import argparse
import sys

import httpx

from console import asr
from console.agent_bridge import AgentUnavailable, get_agent
from console.app import ConsoleSettings, create_app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m console", description="GUI assist console (FastAPI + one static page).")
    ap.add_argument("--agent", choices=("mock", "real"), default="mock",
                    help="mock = console/mock_agent.py (scripted, no models); real = T2's agent.core")
    ap.add_argument("--lightning", choices=("local", "hosted"), default="local",
                    help="initial Lightning toggle: local = official BF16 on :8000 (fast), hosted = build.nvidia.com API")
    ap.add_argument("--asr", choices=sorted(asr.ENGINES), default=None,
                    help="voice input engine (local only); default $CONSOLE_ASR or stub")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--sandbox-url", default="http://127.0.0.1:8090")
    ap.add_argument("--runs-dir", default="runs",
                    help="mock agent trace dir, relative to the repo root (the real agent uses configs/agent.yaml runs_dir)")
    ap.add_argument("--headed", action="store_true", help="show the Playwright browser window (for the video)")
    ap.add_argument("--step-delay", type=float, default=0.8, help="mock agent: seconds per step, so the demo looks live")
    args = ap.parse_args(argv)

    settings = ConsoleSettings(agent=args.agent, sandbox_url=args.sandbox_url, runs_dir=args.runs_dir,
                               headed=args.headed, step_delay_s=args.step_delay, host=args.host, port=args.port,
                               default_lightning=args.lightning)
    if args.asr:
        asr.configure(args.asr)
    try:
        get_agent(settings)  # fail fast, e.g. --agent real before T2's agent.core exists
    except AgentUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        httpx.get(f"{settings.sandbox_url}/api/health", timeout=2, trust_env=False).raise_for_status()
    except httpx.HTTPError as e:
        print(f"warning: sandbox not reachable at {settings.sandbox_url} ({type(e).__name__}); runs will fail "
              f"until it is up (scripts/run_sandbox.sh)", file=sys.stderr)

    import uvicorn

    print(f"GUI assist console: http://{settings.host}:{settings.port}  "
          f"(agent={settings.agent}, lightning={settings.default_lightning}, sandbox={settings.sandbox_url}, headed={settings.headed})", flush=True)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info",
                timeout_graceful_shutdown=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
