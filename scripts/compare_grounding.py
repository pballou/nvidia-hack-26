"""Paired comparison of grounding result files (from base_grounding_smoke.py) on the same rows.

Example:
  python scripts/compare_grounding.py --data <dir> \
      --run qwen3vl=<dir>/qwen3vl_2b_h200_bf16_results.jsonl --run qwen35=<dir>/qwen35_2b_h200_bf16_results.jsonl
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--run", action="append", required=True, help="name=results.jsonl (first two are compared pairwise)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    manifest = {}
    for line in open(args.data / "manifest.jsonl"):
        m = json.loads(line)
        manifest[m["id"]] = m
    runs = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        runs[name] = {r["id"]: r for r in map(json.loads, open(path))}
    names = list(runs)
    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    assert all(len(r) == len(ids) for r in runs.values()), "runs cover different rows"

    summary = {"n": len(ids), "runs": {}}
    for name, res in runs.items():
        by = defaultdict(lambda: [0, 0])
        for i in ids:
            m = manifest[i]
            for key in ("all", m["platform"], m["source"].split("/")[1]):
                by[key][0] += res[i]["hit"]
                by[key][1] += 1
        lat = sorted(res[i]["latency_s"] for i in ids)
        summary["runs"][name] = {
            "acc": {k: {"hits": h, "n": t, "acc": round(h / t, 3)} for k, (h, t) in by.items()},
            "unparsed": sum(res[i]["pred_norm"] is None for i in ids),
            "median_latency_s": lat[len(lat) // 2],
        }

    a, b = names[0], names[1]
    both = sum(runs[a][i]["hit"] and runs[b][i]["hit"] for i in ids)
    only_a = [i for i in ids if runs[a][i]["hit"] and not runs[b][i]["hit"]]
    only_b = [i for i in ids if runs[b][i]["hit"] and not runs[a][i]["hit"]]
    summary["paired"] = {
        "a": a, "b": b, "both_hit": both, f"only_{a}": len(only_a), f"only_{b}": len(only_b),
        "neither": len(ids) - both - len(only_a) - len(only_b),
        "mcnemar_p": round(mcnemar_p(len(only_a), len(only_b)), 4),
        f"examples_only_{a}": [(i, runs[a][i]["instruction"]) for i in only_a[:12]],
        f"examples_only_{b}": [(i, runs[b][i]["instruction"]) for i in only_b[:12]],
    }

    out = args.out or args.data / f"compare_{'_vs_'.join(names)}.json"
    out.write_text(json.dumps(summary, indent=2))
    keys = ["all", "desktop", "mobile", "web", "linux", "macos", "androidworld", "uibert", "fineweb"]
    print(f"{'slice':14s}" + "".join(f"{n:>22s}" for n in names))
    for k in keys:
        cells = []
        for n in names:
            s = summary["runs"][n]["acc"].get(k)
            cells.append(f"{s['hits']:4d}/{s['n']:<4d} {s['acc']:6.1%}" if s else "-")
        print(f"{k:14s}" + "".join(f"{c:>22s}" for c in cells))
    print("unparsed:", {n: summary["runs"][n]["unparsed"] for n in names},
          " median latency s:", {n: summary["runs"][n]["median_latency_s"] for n in names})
    p = summary["paired"]
    print(f"paired {a} vs {b}: both {p['both_hit']}, only {a} {p[f'only_{a}']}, only {b} {p[f'only_{b}']}, "
          f"neither {p['neither']}, McNemar p={p['mcnemar_p']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
