"""Exact McNemar tests per ScreenSpot-v2-clean slice, from a `gym eval compare` report.

    python3 -I scripts/mcnemar_slices.py \
        --report docs/eval_results/compare_ssv2_base_vs_final/compare_report.json \
        --raw <dir with screenspot_{mobile,desktop,web}_v2.json from OS-Copilot/ScreenSpot-v2 @ 5efbb1f1b5>

- The report's flips carry only `task_index`, so slices come from the SSv2-clean row order that
  data_prep/gym_rows.py builds: mobile, desktop, web, each in file order, minus the 28 rows in
  benchmarks/screenspot_v2/excluded_rows.json. Only the annotation JSONs are needed, no images.
- Each slice's baseline pass count + fixed - broken must equal the candidate pass count implied by
  the report's own per-slice accuracies, so a wrong mapping fails loudly.
- Holm-adjusted p-values cover the 11 slice tests (3 platforms, 2 types, 6 platform x type); the
  overall test is reported on its own.
"""

import argparse
import json
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SSV2_FILES = {"mobile": "screenspot_mobile_v2.json", "desktop": "screenspot_desktop_v2.json",
              "web": "screenspot_web_v2.json"}  # same order as data_prep/gym_rows.py
SSV2_EXCLUDED = REPO / "benchmarks/screenspot_v2/excluded_rows.json"


def ssv2_clean_slices(raw: Path) -> list[tuple[str, str]]:
    """(platform, data_type) per SSv2-clean row, in gym_rows.py's order."""
    excluded = {(r["img_filename"], r["instruction"]) for r in json.loads(SSV2_EXCLUDED.read_text())["rows"]}
    loaded, kept = 0, []
    for platform, fname in SSV2_FILES.items():
        for r in json.loads((raw / fname).read_text()):
            loaded += 1
            if (r["img_filename"], r["instruction"]) not in excluded:
                kept.append((platform, r["data_type"]))
    assert loaded == 1272 and len(kept) == 1244, f"expected 1,272 -> 1,244 SSv2 rows, got {loaded} -> {len(kept)}"
    return kept


def mcnemar_exact(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar: 2 * P(X <= min) for X ~ Binomial(fixed + broken, 1/2)."""
    n, k = fixed + broken, min(fixed, broken)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=ps.__getitem__)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True, help="compare_report.json from gym eval compare")
    ap.add_argument("--raw", type=Path, required=True, help="dir with the three SSv2 annotation JSONs")
    a = ap.parse_args()

    report = json.loads(a.report.read_text())
    comp = report["comparisons"][0]
    flips = comp["flips"][0]
    assert flips["common_task_count"] == 1244 and flips["mode"] == "binary", flips["common_task_count"]
    acc = {m["metric"].removesuffix("/pass@1/accuracy"): (m["baseline"]["value"], m["candidates"][0]["value"])
           for m in comp["metrics"] if m["metric"].endswith("/pass@1/accuracy")}
    base_mean, cand_mean = next((m["baseline"]["value"], m["candidates"][0]["value"])
                                for m in comp["metrics"] if m["metric"] == "mean/reward")

    rows = ssv2_clean_slices(a.raw)
    slices = ["mobile", "desktop", "web", "text", "icon"] + [f"{p}/{t}" for p in SSV2_FILES for t in ("text", "icon")]

    def members(name: str) -> set[int]:
        parts = name.split("/")
        return {i for i, (p, t) in enumerate(rows) if all(x in (p, t) for x in parts)}

    results = []
    for name in ["all"] + slices:
        idx = set(range(len(rows))) if name == "all" else members(name)
        fixed = sum(f["task_index"] in idx and f["direction"] == "fail_to_pass" for f in flips["flips"])
        broken = sum(f["task_index"] in idx and f["direction"] == "pass_to_fail" for f in flips["flips"])
        b_acc, c_acc = (100 * base_mean, 100 * cand_mean) if name == "all" else acc[name]
        b_pass, c_pass = round(b_acc * len(idx) / 100), round(c_acc * len(idx) / 100)
        assert b_pass + fixed - broken == c_pass, f"{name}: {b_pass} + {fixed} - {broken} != {c_pass} (mapping wrong?)"
        results.append({"slice": name, "n": len(idx), "baseline_pct": round(b_acc, 2), "candidate_pct": round(c_acc, 2),
                        "both_pass": b_pass - broken, "both_fail": len(idx) - b_pass - fixed,
                        "fixed": fixed, "broken": broken, "p": mcnemar_exact(fixed, broken)})
    for r, p_holm in zip(results[1:], holm([r["p"] for r in results[1:]])):
        r["p_holm"] = p_holm

    print("| Slice (rows) | Base → final | Both pass / both fail | Fixed / broken | McNemar p | Holm p (11 slices) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in results:
        holm_cell = f"{r['p_holm']:.2g}" if "p_holm" in r else "—"
        print(f"| {r['slice']} ({r['n']:,}) | {r['baseline_pct']:.2f} → {r['candidate_pct']:.2f} | "
              f"{r['both_pass']:,} / {r['both_fail']:,} | {r['fixed']} / {r['broken']} | {r['p']:.2g} | {holm_cell} |")


if __name__ == "__main__":
    main()
