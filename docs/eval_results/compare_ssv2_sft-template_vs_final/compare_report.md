# gym eval compare

|  | Baseline | Candidate |
|---|---|---|
| Run | `ssv2_clean_sft-template` | `ssv2_clean_final-round1-grpo` |
| Rollouts | `/home/ubuntu/night/evals/sft-template/ssv2_clean_sft-template/rollouts.jsonl` | `/home/ubuntu/night/evals/final-round1-grpo/ssv2_clean_final-round1-grpo/rollouts.jsonl` |
| Aggregate metrics | `/home/ubuntu/night/evals/sft-template/ssv2_clean_sft-template/rollouts_aggregate_metrics.json` | `/home/ubuntu/night/evals/final-round1-grpo/ssv2_clean_final-round1-grpo/rollouts_aggregate_metrics.json` |
| Agent | `gui_grounding_simple_agent` | `gui_grounding_simple_agent` |
| Tasks | 1244 | 1244 |
| Repeats | 1 | 1 |

Generated 2026-10-18T02:09:13+00:00 by nemo-gym 0.7.0.

Command: `gym eval compare '+baseline_rollouts_jsonl_fpath="/home/ubuntu/night/evals/sft-template/ssv2_clean_sft-template/rollouts.jsonl"' '+candidate_rollouts_jsonl_fpaths=["/home/ubuntu/night/evals/final-round1-grpo/ssv2_clean_final-round1-grpo/rollouts.jsonl"]' '+output_dirpath="/home/ubuntu/night/evals/compare_ssv2_sft-template_vs_final"' +report_format=both`

### Key metrics

| Metric | Δ (cand − base) | 95% CI for Δ (cand − base) | Baseline | Baseline 95% CI | Candidate | Candidate 95% CI |
|---|---:|---|---:|---|---:|---|
| `mean/reward` | +0.0129 (+1.4%) | — | 0.8907 | — | 0.9035 | — |
| `mean/img_w` | +0.0000 (+0.0%) | — | 1704.63 | — | 1704.63 | — |
| `mean/img_h` | +0.0000 (+0.0%) | — | 1659.91 | — | 1659.91 | — |
| `mean/parsed` | +0.0000 (+0.0%) | — | 1.0000 | — | 1.0000 | — |
| `mean/pred_x` | -11.48 (-2.2%) | — | 525.84 | — | 514.36 | — |
| `mean/pred_y` | -1.7347 (-0.4%) | — | 426.80 | — | 425.06 | — |
| `mean/input_tokens` | +0.0000 (+0.0%) | — | 1821.82 | — | 1821.82 | — |
| `mean/output_tokens` | -0.0233 (-0.1%) | — | 16.76 | — | 16.74 | — |
| `mean/total_tokens` | -0.0233 (-0.0%) | — | 1838.58 | — | 1838.56 | — |
| `mobile/pass@1/accuracy` | +0.9980 (+1.1%) | — | 92.22 | — | 93.21 | — |
| `mobile/pass@1[avg-of-1]/accuracy` | +0.9980 (+1.1%) | — | 92.22 | — | 93.21 | — |
| `desktop/pass@1/accuracy` | -0.6536 (-0.7%) | — | 89.54 | — | 88.89 | — |
| `desktop/pass@1[avg-of-1]/accuracy` | -0.6536 (-0.7%) | — | 89.54 | — | 88.89 | — |
| `web/pass@1/accuracy` | +2.9748 (+3.5%) | — | 85.13 | — | 88.10 | — |
| `web/pass@1[avg-of-1]/accuracy` | +2.9748 (+3.5%) | — | 85.13 | — | 88.10 | — |
| `icon/pass@1/accuracy` | +0.5556 (+0.7%) | — | 84.81 | — | 85.37 | — |
| `icon/pass@1[avg-of-1]/accuracy` | +0.5556 (+0.7%) | — | 84.81 | — | 85.37 | — |
| `text/pass@1/accuracy` | +1.8466 (+2.0%) | — | 92.33 | — | 94.18 | — |
| `text/pass@1[avg-of-1]/accuracy` | +1.8466 (+2.0%) | — | 92.33 | — | 94.18 | — |
| `mobile/icon/pass@1/accuracy` | +1.4218 (+1.6%) | — | 86.73 | — | 88.15 | — |
| `mobile/icon/pass@1[avg-of-1]/accuracy` | +1.4218 (+1.6%) | — | 86.73 | — | 88.15 | — |
| `mobile/text/pass@1/accuracy` | +0.6897 (+0.7%) | — | 96.21 | — | 96.90 | — |
| `mobile/text/pass@1[avg-of-1]/accuracy` | +0.6897 (+0.7%) | — | 96.21 | — | 96.90 | — |
| `desktop/icon/pass@1/accuracy` | +0.0000 (+0.0%) | — | 84.92 | — | 84.92 | — |
| `desktop/icon/pass@1[avg-of-1]/accuracy` | +0.0000 (+0.0%) | — | 84.92 | — | 84.92 | — |
| `desktop/text/pass@1/accuracy` | -1.1111 (-1.2%) | — | 92.78 | — | 91.67 | — |
| `desktop/text/pass@1[avg-of-1]/accuracy` | -1.1111 (-1.2%) | — | 92.78 | — | 91.67 | — |
| `web/text/pass@1/accuracy` | +5.5556 (+6.4%) | — | 87.18 | — | 92.74 | — |
| `web/text/pass@1[avg-of-1]/accuracy` | +5.5556 (+6.4%) | — | 87.18 | — | 92.74 | — |
| `web/icon/pass@1/accuracy` | +0.0000 (+0.0%) | — | 82.76 | — | 82.76 | — |
| `web/icon/pass@1[avg-of-1]/accuracy` | +0.0000 (+0.0%) | — | 82.76 | — | 82.76 | — |
| `screenspot_v2/mobile/pass@1/accuracy` | +0.9980 (+1.1%) | — | 92.22 | — | 93.21 | — |
| `screenspot_v2/mobile/pass@1[avg-of-1]/accuracy` | +0.9980 (+1.1%) | — | 92.22 | — | 93.21 | — |
| `screenspot_v2/desktop/pass@1/accuracy` | -0.6536 (-0.7%) | — | 89.54 | — | 88.89 | — |
| `screenspot_v2/desktop/pass@1[avg-of-1]/accuracy` | -0.6536 (-0.7%) | — | 89.54 | — | 88.89 | — |
| `screenspot_v2/web/pass@1/accuracy` | +2.9748 (+3.5%) | — | 85.13 | — | 88.10 | — |
| `screenspot_v2/web/pass@1[avg-of-1]/accuracy` | +2.9748 (+3.5%) | — | 85.13 | — | 88.10 | — |

Displayed baseline/candidate values and each delta use `mean_across_repeats/<metric>` when recorded, otherwise the metric value. Delta CI = two-sided independent-sample 95% Welch t-interval for candidate minus baseline, computed from the named metric's values in each run's `repeat_level_metrics`. `—` means the required interval data was unavailable: delta intervals require at least 2 finite repeat-level values on each side. Baseline/candidate CI = 95% t-interval of the per-repeat mean across repeats, read verbatim from `ci_{low,high}_95_across_repeats/<metric>` in `*_aggregate_metrics.json`.

### Sample flips

Field: `reward` (task mean over its repeats). A task passes when its mean is above 0.5.

**1244 common tasks · 34 pass→fail · 50 fail→pass · net +16 · 1160 unchanged · 0 tied**

| Task | Direction | Baseline | Candidate | Δ | Baseline per-repeat | Candidate per-repeat |
|---:|---|---:|---:|---:|---|---|
| 1 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 15 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 46 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 203 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 219 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 248 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 283 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 342 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 346 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 453 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 94 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 152 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 160 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 274 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 327 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 334 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 339 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 352 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 359 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 363 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |

… and 64 more (full list in `compare_report.json`).

Tasks are joined on `_ng_task_index`. `*_aggregate_metrics.json` carries no task identity, so flips are reported by index only — this assumes both runs used the same dataset, split, `--limit` and ordering.

### Notes

- Neither run recorded per-run cross-repeat confidence intervals, so every baseline/candidate CI cell is empty. They are written for repeat-aggregated metrics when a run has 2 or more repeats.

---
Generated by `gym eval compare` (nemo-gym 0.7.0). Machine-readable output: `compare_report.json` (schema_version 1).
