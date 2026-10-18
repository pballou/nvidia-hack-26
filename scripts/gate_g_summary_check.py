"""Spot-check screen summaries from the local grounder server, the way the agent asks for them.

  ~/.venv-merge/bin/python scripts/gate_g_summary_check.py --base-url http://127.0.0.1:8011/v1 \
      --models base grounder --out ~/models/gate_g/<name>/summaries.json <screenshot.png> ...

Each screenshot goes through common/grounding.py (preprocess_screenshot, the SUMMARY_PROMPT
messages, SUMMARY_SAMPLING, thinking off) and the reply through agent/observation.py's
clip_summary, which is what Lightning actually reads. Reports per model and image: the line
count, whether it's the five-section plain-text layout, whether a line hit the 240-char clip,
and the clipped text. Pass --expect "Confirm payment" to also check a phrase survives clipping.
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (no pyproject, AGENTS.md)
from agent.observation import MAX_LINE_CHARS, clip_summary  # noqa: E402
from common.grounding import SUMMARY_SAMPLING, build_summary_messages, preprocess_screenshot, to_data_url  # noqa: E402

SECTIONS = ("PAGE:", "NAV:", "MAIN:", "STATE:", "OVERLAYS:")


def summarize(base_url: str, model: str, image: Path) -> dict:
    with Image.open(image) as im:
        url = to_data_url(preprocess_screenshot(im))
    body = {"model": model, "messages": build_summary_messages(url), **SUMMARY_SAMPLING,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    raw = data["choices"][0]["message"].get("content") or ""
    clipped = clip_summary(raw)
    return {
        "model": model, "image": str(image), "latency_s": round(time.perf_counter() - t0, 2),
        "finish_reason": data["choices"][0].get("finish_reason"), "usage": data.get("usage"),
        "lines": len(clipped.splitlines()), "sections": [s for s in SECTIONS if s in raw],
        "starts_with_json": raw.lstrip().startswith(("{", "[", "```")),
        "clipped_a_line": any(len(ln) > MAX_LINE_CHARS for ln in raw.splitlines()),
        "raw": raw, "clipped": clipped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--models", nargs="+", default=["base", "grounder"])
    ap.add_argument("--expect", action="append", default=[], help="phrase that should survive clipping (any image)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    results = [summarize(args.base_url.rstrip("/"), m, img) for img in args.images for m in args.models]
    for r in results:
        found = {e: e.lower() in r["clipped"].lower() for e in args.expect}
        r["expect_found"] = found
        print(f"== {r['model']} | {Path(r['image']).name} | {r['latency_s']}s | {r['finish_reason']} | "
              f"{r['lines']} lines | sections {len(r['sections'])}/5 | json={r['starts_with_json']} | "
              f"line clipped={r['clipped_a_line']} | expect {found}")
        print(r["clipped"][:1200])
    if args.out:
        args.out.expanduser().write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
