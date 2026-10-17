"""Score a model on ScreenSpot-v2-clean rows directly against a vLLM OpenAI-compatible
endpoint, without going through a NeMo Gym server stack. Scoring matches
resources_servers/gui_grounding/app.py exactly (point-in-bbox, inclusive), so results
are comparable to a `gym eval run` aggregate_metrics.json.

Rows come from data_prep/gym_rows.py's output (Gym Responses-API row format); this
script converts each row's `responses_create_params.input` to chat-completions messages
instead of replaying it through the Responses API.

Example:
  python scripts/eval_ssv2_direct.py --rows /home/ubuntu/data/gym/ssv2_clean.jsonl \
      --base-url http://127.0.0.1:8002/v1 --model base --concurrency 32 \
      --out /home/ubuntu/data/gym/results/ssv2_clean_base_qwen3vl
"""

import argparse
import asyncio
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import httpx

from common.grounding import parse_point, to_pixels

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S | re.I)


def row_to_messages(row: dict) -> tuple[str, str]:
    content = row["responses_create_params"]["input"][0]["content"]
    image_url = next(c["image_url"] for c in content if c["type"] == "input_image")
    text = next(c["text"] for c in content if c["type"] == "input_text")
    return image_url, text


def inside(pt_px: tuple[float, float], bbox: list[float]) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= pt_px[0] <= x2 and y1 <= pt_px[1] <= y2


async def score_one(client: httpx.AsyncClient, base_url: str, model: str, row: dict, max_tokens: int) -> dict:
    image_url, text = row_to_messages(row)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": text},
        ]}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    resp = await client.post(f"{base_url}/chat/completions", json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    reply = data["choices"][0]["message"]["content"] or ""
    pt = parse_point(_THINK_RE.sub("", reply))
    hit = pt is not None and inside(to_pixels(pt, row["img_w"], row["img_h"]), row["bbox"])
    usage = data.get("usage", {})
    return {
        "id": row["id"], "source": row["source"], "platform": row["platform"], "data_type": row.get("data_type"),
        "instruction": row["instruction"], "bbox": row["bbox"], "img_w": row["img_w"], "img_h": row["img_h"],
        "reward": float(hit), "parsed": pt is not None,
        "pred_x": pt[0] if pt else None, "pred_y": pt[1] if pt else None,
        "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
        "raw_output": reply,
    }


async def run(rows: list[dict], base_url: str, model: str, concurrency: int, max_tokens: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    results: list[dict | None] = [None] * len(rows)

    async def worker(i: int, row: dict):
        async with sem:
            for attempt in range(3):
                try:
                    results[i] = await score_one(client, base_url, model, row, max_tokens)
                    return
                except Exception as e:
                    if attempt == 2:
                        results[i] = {"id": row["id"], "source": row["source"], "platform": row["platform"],
                                      "data_type": row.get("data_type"), "reward": 0.0, "parsed": False,
                                      "error": repr(e)}
                    await asyncio.sleep(1.5 * (attempt + 1))

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        await asyncio.gather(*(worker(i, r) for i, r in enumerate(rows)))
    return results


def slice_accuracy(rows: list[dict], keys: tuple[str, ...]) -> dict:
    buckets = defaultdict(lambda: [0, 0])
    for r in rows:
        vals = [r.get(k) for k in keys]
        if any(v is None for v in vals):
            continue
        name = "/".join(str(v) for v in vals) + "/pass@1/accuracy"
        buckets[name][0] += r["reward"]
        buckets[name][1] += 1
    return {name: round(100 * h / n, 4) for name, (h, n) in buckets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="base")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True, help="writes <out>.jsonl and <out>_aggregate_metrics.json")
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.rows.open()]
    results = asyncio.run(run(rows, args.base_url, args.model, args.concurrency, args.max_tokens))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")

    rewards = [r["reward"] for r in results]
    errors = sum("error" in r for r in results)
    metrics = {
        "mean/reward": statistics.mean(rewards), "n": len(results), "errors": errors,
        "mean/parsed": statistics.mean(r["parsed"] for r in results if "parsed" in r),
        **slice_accuracy(results, ("platform",)),
        **slice_accuracy(results, ("data_type",)),
        **slice_accuracy(results, ("platform", "data_type")),
        **slice_accuracy(results, ("source",)),
    }
    args.out.with_name(args.out.name + "_aggregate_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nerrors: {errors}/{len(results)}")
    print(f"wrote {args.out.with_suffix('.jsonl')}")


if __name__ == "__main__":
    main()
