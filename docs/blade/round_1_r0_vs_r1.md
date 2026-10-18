# BLADE, night loop round 1: `sft-lightning` (r0) → GRPO round 1 (r1)

**Run:** Sat 2026-10-17. The round finished at 20:50, the analysis and knob upload happened at
20:51, and round 2 started at 20:51:51.

**Inputs:**

- **Dev rollouts (969 rows, temperature 0, through the Gym `gui_grounding` env):**
  `/home/ubuntu/night/round_{0,1}/dev_r{0,1}/rollouts.jsonl`.
- **Reference:** the base Qwen3.5-2B dev run.

**Tool:** `scripts/blade_pair.py fetch/analyze --round 1`, following the runbook in
[README.md](README.md). Template report: [round_0_base_vs_sft.md](round_0_base_vs_sft.md).

## Round 1 itself (from `round_1/summary.json`)

- **GRPO:** 170 steps (TRL, colocated vLLM) on a pool of 1,282 sometimes-pass train rows, with
  the Gym `/verify` point-in-box reward. Default knobs.
- **Dev:** 86.27 → **87.00 (+0.73)**, which clears the +0.5 keep gate, so the round was
  **kept**.
- **Outcomes:** 22 rows fixed, 15 broken. McNemar exact p = 0.32: the overall gain is
  suggestive, and the per-source shifts below are the real signal.
- **Without the 46 rows confirmed as data artifacts:** 89.82 → 90.79.

## Per source and platform

| Source | n | r0 | r1 | Δ | Fixed | Broken |
|---|---|---|---|---|---|---|
| androidworld | 139 | 78.42 | **84.89** | **+6.47** | 10 | 1 |
| linux | 176 | 87.50 | 89.77 | +2.27 | 5 | 1 |
| macos | 158 | 90.51 | 92.41 | +1.90 | 5 | 2 |
| fineweb | 362 | 85.91 | 84.25 | −1.66 | 2 | 8 |
| uibert | 134 | 88.81 | 86.57 | −2.24 | 0 | 3 |

| Platform | r0 | r1 | Δ |
|---|---|---|---|
| desktop | 88.92 | 91.02 | +2.10 |
| mobile | 83.52 | 85.71 | +2.19 |
| web | 85.91 | 84.25 | −1.66 |

- **AndroidWorld recovered most:** the `android.widget.TextView` role went from 77.5 to 86.5.
  Round 0's report found SFT clicking the middle of Android list rows (x ≈ 568) instead of the
  text. The point-in-box reward targets exactly that.
- **Web (FineWeb) and UIBert regressed.** Those are the slices the knobs now push toward.

## Failure taxonomy of r1 (137 dev failures)

| Label | Rows | Meaning |
|---|---|---|
| CONSENSUS_miss | 41 | The base model misses at the same spot. These are mostly duplicate text on FineWeb pages (ambiguous labels). |
| KG_far_miss | 34 | Real model misses, far from the target. |
| DA_confirmed | 29 | Boxes that don't match the pixels (checked by eye in round 0). |
| DA_ungroundable_instruction | 9 | Instructions nothing on screen can match: AX handles, D-Bus names. |
| KG_wrong_annotated_element | 8 | The model clicks a different element that fits the instruction. |
| DA_sub_precision_box | 3 | Boxes too thin to hit. |
| KG_near_miss | 2 | Just outside the box. |

**Broken rows by label:** KG_far_miss 7, CONSENSUS_miss 3, KG_wrong_annotated_element 2,
DA_ungroundable_instruction 2, DA_confirmed 1.

## Knob change applied for round 2 (the only knobs BLADE may touch)

| Knob | Before | After | Why |
|---|---|---|---|
| `source_weights.os_atlas/uibert` | 1.0 | **1.75** | Highest model-attributable error: 8.2% vs 4.8% overall |
| `source_weights.os_atlas/fineweb` | 1.0 | **1.25** | 5.0% vs 4.8%, plus 0.25 for its 8 broken rows |
| `source_weights.os_atlas/androidworld` | 1.0 | 0.75 | 4.0%: fixed in this round |
| `source_weights.os_atlas/macos` | 1.0 | 0.5 | 2.0%: lowest error |
| `source_weights.os_atlas/linux` | 1.0 | 1.0 | Near the overall rate |
| `exclude_ids` | 0 | **285 train ids** | By rule: 85 Linux D-Bus names, 55 macOS AX handles, 9 value literals, 10 FineWeb file-name slugs, 66 sub-precision boxes, and 60 unprofiled rows from screenshots where every profiled row never passes |

- **Unchanged:** the other five knobs, byte for byte.
- **Upload:** atomic (`knobs.json` sha256 `023999c2…`), then `round_1/BLADE_DONE`.
- **Confirmed:** `round_2/knobs_used.json` shows these values with `notes: []`.
- **Dev ids** never reach the pool and aren't in `exclude_ids`.

**Open check:** one FineWeb dev row confirmed as an artifact
(`fineweb:8a1091f8…_section_1_original:1`) is one the base model passes. The box may be right.
It affects only this diagnostic, not the knobs.
