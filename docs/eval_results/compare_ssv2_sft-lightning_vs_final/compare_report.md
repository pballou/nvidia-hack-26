# gym eval compare

|  | Baseline | Candidate |
|---|---|---|
| Run | `ssv2_clean_sft-lightning` | `ssv2_clean_final-round1-grpo` |
| Rollouts | `/home/ubuntu/night/evals/sft-lightning/ssv2_clean_sft-lightning/rollouts.jsonl` | `/home/ubuntu/night/evals/final-round1-grpo/ssv2_clean_final-round1-grpo/rollouts.jsonl` |
| Aggregate metrics | `/home/ubuntu/night/evals/sft-lightning/ssv2_clean_sft-lightning/rollouts_aggregate_metrics.json` | `/home/ubuntu/night/evals/final-round1-grpo/ssv2_clean_final-round1-grpo/rollouts_aggregate_metrics.json` |
| Agent | `gui_grounding_simple_agent` | `gui_grounding_simple_agent` |
| Tasks | 1244 | 1244 |
| Repeats | 1 | 1 |

Generated 2026-10-18T02:09:10+00:00 by nemo-gym 0.7.0.

Command: `gym eval compare '+baseline_rollouts_jsonl_fpath="/home/ubuntu/night/evals/sft-lightning/ssv2_clean_sft-lightning/rollouts.jsonl"' '+candidate_rollouts_jsonl_fpaths=["/home/ubuntu/night/evals/final-round1-grpo/ssv2_clean_final-round1-grpo/rollouts.jsonl"]' '+output_dirpath="/home/ubuntu/night/evals/compare_ssv2_sft-lightning_vs_final"' +report_format=both`

### Key metrics

| Metric | Δ (cand − base) | 95% CI for Δ (cand − base) | Baseline | Baseline 95% CI | Candidate | Candidate 95% CI |
|---|---:|---|---:|---|---:|---|
| `mean/reward` | +0.0225 (+2.6%) | — | 0.8810 | — | 0.9035 | — |
| `mean/img_w` | +0.0000 (+0.0%) | — | 1704.63 | — | 1704.63 | — |
| `mean/img_h` | +0.0000 (+0.0%) | — | 1659.91 | — | 1659.91 | — |
| `mean/parsed` | +0.0000 (+0.0%) | — | 1.0000 | — | 1.0000 | — |
| `mean/pred_x` | -11.64 (-2.2%) | — | 526.00 | — | 514.36 | — |
| `mean/pred_y` | -0.2974 (-0.1%) | — | 425.36 | — | 425.06 | — |
| `mean/input_tokens` | +0.0000 (+0.0%) | — | 1821.82 | — | 1821.82 | — |
| `mean/output_tokens` | -0.0289 (-0.2%) | — | 16.77 | — | 16.74 | — |
| `mean/total_tokens` | -0.0289 (-0.0%) | — | 1838.59 | — | 1838.56 | — |
| `mobile/pass@1/accuracy` | +1.9960 (+2.2%) | — | 91.22 | — | 93.21 | — |
| `mobile/pass@1[avg-of-1]/accuracy` | +1.9960 (+2.2%) | — | 91.22 | — | 93.21 | — |
| `desktop/pass@1/accuracy` | +1.9608 (+2.3%) | — | 86.93 | — | 88.89 | — |
| `desktop/pass@1[avg-of-1]/accuracy` | +1.9608 (+2.3%) | — | 86.93 | — | 88.89 | — |
| `web/pass@1/accuracy` | +2.7460 (+3.2%) | — | 85.35 | — | 88.10 | — |
| `web/pass@1[avg-of-1]/accuracy` | +2.7460 (+3.2%) | — | 85.35 | — | 88.10 | — |
| `icon/pass@1/accuracy` | +3.7037 (+4.5%) | — | 81.67 | — | 85.37 | — |
| `icon/pass@1[avg-of-1]/accuracy` | +3.7037 (+4.5%) | — | 81.67 | — | 85.37 | — |
| `text/pass@1/accuracy` | +1.1364 (+1.2%) | — | 93.04 | — | 94.18 | — |
| `text/pass@1[avg-of-1]/accuracy` | +1.1364 (+1.2%) | — | 93.04 | — | 94.18 | — |
| `mobile/icon/pass@1/accuracy` | +3.7915 (+4.5%) | — | 84.36 | — | 88.15 | — |
| `mobile/icon/pass@1[avg-of-1]/accuracy` | +3.7915 (+4.5%) | — | 84.36 | — | 88.15 | — |
| `mobile/text/pass@1/accuracy` | +0.6897 (+0.7%) | — | 96.21 | — | 96.90 | — |
| `mobile/text/pass@1[avg-of-1]/accuracy` | +0.6897 (+0.7%) | — | 96.21 | — | 96.90 | — |
| `desktop/icon/pass@1/accuracy` | +3.9683 (+4.9%) | — | 80.95 | — | 84.92 | — |
| `desktop/icon/pass@1[avg-of-1]/accuracy` | +3.9683 (+4.9%) | — | 80.95 | — | 84.92 | — |
| `desktop/text/pass@1/accuracy` | +0.5556 (+0.6%) | — | 91.11 | — | 91.67 | — |
| `desktop/text/pass@1[avg-of-1]/accuracy` | +0.5556 (+0.6%) | — | 91.11 | — | 91.67 | — |
| `web/text/pass@1/accuracy` | +2.1368 (+2.4%) | — | 90.60 | — | 92.74 | — |
| `web/text/pass@1[avg-of-1]/accuracy` | +2.1368 (+2.4%) | — | 90.60 | — | 92.74 | — |
| `web/icon/pass@1/accuracy` | +3.4483 (+4.3%) | — | 79.31 | — | 82.76 | — |
| `web/icon/pass@1[avg-of-1]/accuracy` | +3.4483 (+4.3%) | — | 79.31 | — | 82.76 | — |
| `screenspot_v2/mobile/pass@1/accuracy` | +1.9960 (+2.2%) | — | 91.22 | — | 93.21 | — |
| `screenspot_v2/mobile/pass@1[avg-of-1]/accuracy` | +1.9960 (+2.2%) | — | 91.22 | — | 93.21 | — |
| `screenspot_v2/desktop/pass@1/accuracy` | +1.9608 (+2.3%) | — | 86.93 | — | 88.89 | — |
| `screenspot_v2/desktop/pass@1[avg-of-1]/accuracy` | +1.9608 (+2.3%) | — | 86.93 | — | 88.89 | — |
| `screenspot_v2/web/pass@1/accuracy` | +2.7460 (+3.2%) | — | 85.35 | — | 88.10 | — |
| `screenspot_v2/web/pass@1[avg-of-1]/accuracy` | +2.7460 (+3.2%) | — | 85.35 | — | 88.10 | — |

Displayed baseline/candidate values and each delta use `mean_across_repeats/<metric>` when recorded, otherwise the metric value. Delta CI = two-sided independent-sample 95% Welch t-interval for candidate minus baseline, computed from the named metric's values in each run's `repeat_level_metrics`. `—` means the required interval data was unavailable: delta intervals require at least 2 finite repeat-level values on each side. Baseline/candidate CI = 95% t-interval of the per-repeat mean across repeats, read verbatim from `ci_{low,high}_95_across_repeats/<metric>` in `*_aggregate_metrics.json`.

### Sample flips

Field: `reward` (task mean over its repeats). A task passes when its mean is above 0.5.

**1244 common tasks · 6 pass→fail · 34 fail→pass · net +28 · 1204 unchanged · 0 tied**

| Task | Direction | Baseline | Candidate | Δ | Baseline per-repeat | Candidate per-repeat |
|---:|---|---:|---:|---:|---|---|
| 229 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 627 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 729 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 928 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 1098 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 1099 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 154 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 266 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 320 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 327 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 339 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 352 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 383 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 384 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 398 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 423 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |

… and 24 more (full list in `compare_report.json`).

Tasks are joined on `_ng_task_index`. `*_aggregate_metrics.json` carries no task identity, so flips are reported by index only — this assumes both runs used the same dataset, split, `--limit` and ordering.

### Notes

- Neither run recorded per-run cross-repeat confidence intervals, so every baseline/candidate CI cell is empty. They are written for repeat-aggregated metrics when a run has 2 or more repeats.

---
Generated by `gym eval compare` (nemo-gym 0.7.0). Machine-readable output: `compare_report.json` (schema_version 1).
