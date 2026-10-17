"""Deterministic metrics for a grounding results JSONL (from base_grounding_smoke.py), joined with the manifest.

Example:
  python scripts/analyze_grounding.py --data /home/ubuntu/data/os_atlas_20k \
      --results /home/ubuntu/data/os_atlas_20k/base_smoke_seed1_n500.jsonl
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

VERB_RE = re.compile(r"^(click|tap|select|choose|open|go|press|find|tag|check|navigate|view|show|enter|type|pick|toggle|switch)\b", re.I)


def bucket(v, edges, labels):
    for e, lab in zip(edges, labels):
        if v < e:
            return lab
    return labels[-1]


def outcome(r, box):
    """Funnel stage for one row: unparsed -> off-image -> far miss -> near miss -> hit."""
    if r["pred_norm"] is None:
        return "unparsed"
    x, y = r["pred_norm"]
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return "off_image"
    if r["hit"]:
        return "hit"
    bw, bh = box[2] - box[0], box[3] - box[1]
    dx = max(box[0] - x, 0, x - box[2])
    dy = max(box[1] - y, 0, y - box[3])
    return "near_miss" if dx <= bw and dy <= bh else "far_miss"


def features(r, m):
    box = r["bbox_norm"]
    area = (box[2] - box[0]) * (box[3] - box[1])
    w_px = m["bbox"][2] - m["bbox"][0]
    h_px = m["bbox"][3] - m["bbox"][1]
    words = len(r["instruction"].split())
    cy = (box[1] + box[3]) / 2
    return {
        "subset": m["source"].split("/")[1],
        "platform": m["platform"],
        "box_area": bucket(area, [0.001, 0.005, 0.02], ["<0.1%", "0.1-0.5%", "0.5-2%", ">=2%"]),
        "box_min_side_px": bucket(min(w_px, h_px), [16, 32, 64], ["<16px", "16-32px", "32-64px", ">=64px"]),
        "words": bucket(words, [2, 4, 8], ["1", "2-3", "4-7", "8+"]),
        "verb_start": "verb" if VERB_RE.match(r["instruction"]) else "label",
        "has_digit": "digit" if re.search(r"\d", r["instruction"]) else "no_digit",
        "non_alpha_heavy": "symbol-heavy" if sum(not c.isalnum() and not c.isspace() for c in r["instruction"]) / max(len(r["instruction"]), 1) > 0.15 else "normal",
        "vertical": bucket(cy, [0.1, 0.9], ["top 10%", "middle", "bottom 10%"]),
        "img_megapixels": bucket(m["img_w"] * m["img_h"] / 1e6, [1, 2.5], ["<1MP", "1-2.5MP", ">=2.5MP"]),
        "role": m.get("role") or "none",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    manifest = {}
    for line in open(args.data / "manifest.jsonl"):
        m = json.loads(line)
        manifest[m["id"]] = m
    rows = [json.loads(line) for line in open(args.results)]

    n = len(rows)
    hits = sum(r["hit"] for r in rows)
    funnel = defaultdict(int)
    slices = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    miss_dist = []
    per_row = []
    for r in rows:
        m = manifest[r["id"]]
        stage = outcome(r, r["bbox_norm"])
        funnel[stage] += 1
        f = features(r, m)
        for k, v in f.items():
            slices[k][v][0] += r["hit"]
            slices[k][v][1] += 1
        if stage in ("near_miss", "far_miss"):
            b = r["bbox_norm"]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            miss_dist.append(math.dist(r["pred_norm"], (cx, cy)))
        per_row.append({"id": r["id"], "stage": stage, **f})

    miss_dist.sort()
    summary = {
        "model_name": "Qwen/Qwen3-VL-2B-Instruct",
        "benchmark": "os_atlas_20k_smoke",
        "results_file": str(args.results),
        "total_tasks": n,
        "total_rollouts": n,
        "repeats_per_task": 1,
        "pass_at_1": round(hits / n, 4),
        "funnel": dict(funnel),
        "miss_center_distance_norm": {
            "median": round(miss_dist[len(miss_dist) // 2], 4) if miss_dist else None,
            "p90": round(miss_dist[int(len(miss_dist) * 0.9)], 4) if miss_dist else None,
        },
        "slices": {k: {v: {"hits": h, "n": t, "acc": round(h / t, 3)} for v, (h, t) in sorted(d.items())}
                   for k, d in slices.items()},
    }
    out = args.out or args.results.with_name(args.results.stem + "_metrics.json")
    out.write_text(json.dumps(summary, indent=2))
    out.with_name(args.results.stem + "_rows.jsonl").write_text("\n".join(json.dumps(p) for p in per_row) + "\n")

    print(f"pass@1 {hits}/{n} = {hits / n:.1%}")
    print("funnel:", dict(funnel))
    print("miss distance from box center (normalized):", summary["miss_center_distance_norm"])
    for k, d in summary["slices"].items():
        if k == "role":
            d = {v: s for v, s in d.items() if s["n"] >= 8}
        print(f"\n{k}:")
        for v, s in d.items():
            print(f"  {v:28s} {s['hits']:4d}/{s['n']:<4d} {s['acc']:.0%}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
