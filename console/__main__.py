"""python -m console --agent mock|real [--lightning local|hosted] [--grounder base|grounder] [--thinking off|retry|on]
[--clarify on|off] [--asr parakeet|stub] [--vlm-url URL] [--summary-model base|grounder] [--vlm-label TEXT]
[--host 127.0.0.1] [--port 8095] [--sandbox-url URL] [--runs-dir runs] [--headed] [--step-delay 0.8]"""

from __future__ import annotations

import argparse
import os
import sys

import httpx

from console import asr
from console.agent_bridge import (SERVED_NAMES, AgentUnavailable, ServedModels, describe_endpoints, endpoint_warnings,
                                  endpoints_for, get_agent)
from console.app import ConsoleSettings, create_app


def voice_note(info: dict) -> str | None:
    """A startup line about voice input when it isn't plain speech recognition (asr.info() -> text or None)."""
    if info.get("fixed_text"):
        return ("note: voice input uses the stub (fixed text, not speech recognition); the page labels it "
                "and never runs it by itself. Use --asr parakeet for real voice input.")
    if not info.get("available"):
        return (f"warning: voice input is off: {info.get('error') or 'no speech recognizer'}. The page turns it "
                "on by itself once the recognizer answers.")
    return None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m console", description="GUI assist console (FastAPI + one static page).")
    ap.add_argument("--agent", choices=("mock", "real"), default="mock",
                    help="mock = console/mock_agent.py (scripted, no models); real = T2's agent.core")
    ap.add_argument("--lightning", choices=("local", "hosted"), default="local",
                    help="initial Lightning toggle: local = official BF16 on :8000 (fast), hosted = build.nvidia.com API")
    ap.add_argument("--grounder", choices=SERVED_NAMES, default="grounder",
                    help="initial Grounder choice: grounder = the LoRA (default), base = adapter off")
    ap.add_argument("--thinking", choices=("off", "retry", "on"), default="off",
                    help="initial Thinking choice (default off, the recording setting): retry = think on the call "
                         "after a problem step, on = every call")
    ap.add_argument("--clarify", choices=("on", "off"), default="on",
                    help="initial Request check toggle (default on): one short Lightning call before the browser "
                         "opens; an unclear request ends with a question (done reason needs_input) and the answer "
                         "starts a follow-up run (agent/clarify.py)")
    ap.add_argument("--vlm-url", default=None, metavar="URL",
                    help="real agent: the grounder/summary VLM's OpenAI base URL, e.g. http://127.0.0.1:8011/v1 "
                         "(default: configs/agent.yaml vlm.base_url or $AGENT_VLM_BASE_URL)")
    ap.add_argument("--summary-model", choices=SERVED_NAMES, default=None,
                    help="real agent: the served name that writes the screen summaries Lightning reads (default: "
                         "configs/agent.yaml summary_model or $AGENT_SUMMARY_MODEL). Use grounder when the VLM "
                         "serves only the merged grounder (fully local on the Mac)")
    ap.add_argument("--vlm-label", default=os.environ.get("CONSOLE_VLM_LABEL") or None, metavar="TEXT",
                    help="how the console names the VLM, e.g. 'Mac (vllm-metal)' (default: $CONSOLE_VLM_LABEL, else "
                         "by port: 8001 'H200 via tunnel', 8011 'Mac (vllm-metal)', else host:port)")
    ap.add_argument("--asr", choices=sorted(asr.ENGINES), default=None,
                    help="voice input engine, local only (default $CONSOLE_ASR or parakeet): parakeet = NVIDIA "
                         "Parakeet TDT 0.6B v3 via scripts/run_asr.sh (voice stays off until it answers); "
                         "stub = fixed text, not speech recognition, and never auto-runs (UI testing only)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--sandbox-url", default="http://127.0.0.1:8090")
    ap.add_argument("--runs-dir", default="runs",
                    help="mock agent trace dir, relative to the repo root (the real agent uses configs/agent.yaml runs_dir)")
    ap.add_argument("--headed", action="store_true",
                    help="show the Playwright browser window; debugging only, unmeasured: headed fonts render different pixels (pay-50 fell from 8/8 to 1/4); record headless")
    ap.add_argument("--step-delay", type=float, default=0.8, help="mock agent: seconds per step, so the demo looks live")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    settings = ConsoleSettings(agent=args.agent, sandbox_url=args.sandbox_url, runs_dir=args.runs_dir,
                               headed=args.headed, step_delay_s=args.step_delay, host=args.host, port=args.port,
                               default_lightning=args.lightning, default_grounder=args.grounder,
                               default_thinking=args.thinking, default_clarify=args.clarify,
                               vlm_url=args.vlm_url, summary_model=args.summary_model, vlm_label=args.vlm_label)
    if args.asr:
        asr.configure(args.asr)
    try:
        _, make_cfg = get_agent(settings)  # fail fast, e.g. --agent real before T2's agent.core exists
    except AgentUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        httpx.get(f"{settings.sandbox_url}/api/health", timeout=2, trust_env=False).raise_for_status()
    except httpx.HTTPError as e:
        print(f"warning: sandbox not reachable at {settings.sandbox_url} ({type(e).__name__}); runs will fail "
              f"until it is up (scripts/run_sandbox.sh)", file=sys.stderr)

    endpoints_line = None
    if settings.agent == "real":
        try:
            ep = endpoints_for(make_cfg(settings.default_options()), served=ServedModels(), label=settings.vlm_label)
        except Exception as e:  # load_config rejects it: say so now; runs answer 503 with the same reason
            print(f"warning: the agent config failed: {type(e).__name__}: {e}", file=sys.stderr)
        else:
            endpoints_line = describe_endpoints(ep)
            for w in endpoint_warnings(ep, settings.default_grounder):
                print(w, file=sys.stderr)
    elif args.vlm_url or args.summary_model:
        print("note: --vlm-url / --summary-model only affect --agent real (the mock runs no models)", file=sys.stderr)

    voice = asr.info()  # one /health check of the sidecar
    note = voice_note(voice)
    if note:
        print(note, file=sys.stderr, flush=True)

    import uvicorn

    print(f"GUI assist console: http://{settings.host}:{settings.port}  "
          f"(agent={settings.agent}, lightning={settings.default_lightning}, grounder={settings.default_grounder}, "
          f"thinking={settings.default_thinking}, clarify={settings.default_clarify}, "
          f"sandbox={settings.sandbox_url}, headed={settings.headed}, "
          f"asr={voice.get('label')}{'' if voice.get('available') else ' (off)'})",
          flush=True)
    if endpoints_line:
        print(f"  {endpoints_line}", flush=True)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info",
                timeout_graceful_shutdown=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
