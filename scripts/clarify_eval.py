#!/usr/bin/env python
"""Live pass of the clarify pre-flight (agent/clarify.py) over a labelled set of phrasings.

  .venv-gym/bin/python scripts/clarify_eval.py                  # configured Lightning (local :8000)
  .venv-gym/bin/python scripts/clarify_eval.py --record         # also store the replies for the replay test
  .venv-gym/bin/python scripts/clarify_eval.py --lightning hosted --repeats 2

Calls are sequential (the local Lightning server is shared). Prints the confusion counts
(expected proceed/ask x what Lightning did), the false asks, the missed asks and latency.
--record writes each case's reply, decision and the run metadata back into the cases file
(agent/tests/fixtures/clarify_cases.json), which agent/tests/test_clarify.py replays offline.
Exit code 1 if any expected-proceed phrasing was asked about (a false ask).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # scripts/ is not a package; import agent/ from the repo root

from agent.clarify import CLARIFY_PROMPT_SHA, CLARIFY_PROMPT_VERSION, CLARIFY_SAMPLING, clarify  # noqa: E402
from agent.config import load_config  # noqa: E402
from agent.planner import LightningPlanner  # noqa: E402

DEFAULT_CASES = REPO / "agent" / "tests" / "fixtures" / "clarify_cases.json"


def _reply(c) -> dict:
    if c.tool_call:
        return dict(c.tool_call)
    return {"error": c.error}


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * len(s)))], 1)


async def run(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.cases).read_text())
    overrides = {"lightning": args.lightning} if args.lightning else None
    cfg = load_config(args.config, overrides)
    planner = LightningPlanner(cfg)
    info = await planner.server_info()
    print(f"Lightning {cfg.lightning.base_url} model={cfg.lightning.model} root={info.get('root')} "
          f"prompt={CLARIFY_PROMPT_VERSION} sha={CLARIFY_PROMPT_SHA}", flush=True)
    counts: Counter = Counter()
    latencies: list[float] = []
    flips, errors = [], []
    try:
        for case in data["cases"]:
            actions = []
            for rep in range(args.repeats):
                c = await clarify(planner, case["text"])
                actions.append(c.action)
                counts[(case["expect"], c.action)] += 1
                if c.latency_ms is not None:
                    latencies.append(c.latency_ms)
                if c.error:
                    errors.append((case["text"], c.error))
                said = c.question if c.asks else c.normalized_request
                mark = "  " if c.action == case["expect"] else "XX"
                print(f"{mark} {case['expect']:>7} -> {c.action:<7} {c.latency_ms or 0:6.0f} ms  "
                      f"{case['text']!r}  =>  {said!r}" + (f"  [error: {c.error}]" if c.error else ""), flush=True)
                if rep == 0:
                    case.update(reply=_reply(c), live_action=c.action,
                                live_said=said, live_latency_ms=round(c.latency_ms or 0, 1))
            if len(set(actions)) > 1:
                flips.append((case["text"], actions))
    finally:
        await planner.aclose()

    n_proceed = sum(v for (e, _), v in counts.items() if e == "proceed")
    n_ask = sum(v for (e, _), v in counts.items() if e == "ask")
    false_asks = counts[("proceed", "ask")]
    missed = counts[("ask", "proceed")]
    print(f"\nexpected proceed: {n_proceed - false_asks}/{n_proceed} proceeded, {false_asks} false asks")
    print(f"expected ask:     {n_ask - missed}/{n_ask} asked, {missed} missed")
    print(f"latency p50/p90: {_pct(latencies, 0.5)} / {_pct(latencies, 0.9)} ms over {len(latencies)} calls")
    for text, acts in flips:
        print(f"unstable across repeats: {text!r} {acts}")
    for text, err in errors:
        print(f"error (proceeded): {text!r}: {err}")

    if args.record:
        if args.repeats != 1:
            print("--record keeps the first repeat of each case")
        first = Counter((c["expect"], c["live_action"]) for c in data["cases"])
        data.update(
            prompt_version=CLARIFY_PROMPT_VERSION, prompt_sha=CLARIFY_PROMPT_SHA, sampling=CLARIFY_SAMPLING,
            lightning={"base_url": cfg.lightning.base_url, "model": cfg.lightning.model, "root": info.get("root")},
            recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            command="scripts/clarify_eval.py " + " ".join(shlex.quote(a) for a in sys.argv[1:]),
            confusion={f"{e}->{a}": n for (e, a), n in sorted(first.items())},
            latency_ms={"p50": _pct(latencies, 0.5), "p90": _pct(latencies, 0.9),
                        "mean": round(statistics.mean(latencies), 1) if latencies else None},
        )
        Path(args.cases).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
        print(f"recorded {len(data['cases'])} replies -> {args.cases}")
    return 1 if false_asks else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cases", default=str(DEFAULT_CASES), help="labelled phrasings (default: %(default)s)")
    p.add_argument("--config", help="agent yaml (default: $AGENT_CONFIG or configs/agent.yaml)")
    p.add_argument("--lightning", choices=("hosted", "local"), help="Lightning endpoint preset")
    p.add_argument("--repeats", type=int, default=1, help="calls per phrasing (T=0, so mostly a stability check)")
    p.add_argument("--record", action="store_true", help="write replies + metadata back into --cases")
    return asyncio.run(run(p.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
