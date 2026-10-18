# gym eval compare

|  | Baseline | Candidate |
|---|---|---|
| Run | `results` | `ssv2_clean_final-round1-grpo` |
| Rollouts | `/home/ubuntu/kyle/nvidia-hack-26/results/ssv2_clean_base_qwen35.jsonl` | `/home/ubuntu/night/evals/final-round1-grpo/ssv2_clean_final-round1-grpo/rollouts.jsonl` |
| Aggregate metrics | `/home/ubuntu/kyle/nvidia-hack-26/results/ssv2_clean_base_qwen35_aggregate_metrics.json` | `/home/ubuntu/night/evals/final-round1-grpo/ssv2_clean_final-round1-grpo/rollouts_aggregate_metrics.json` |
| Agent | `gui_grounding_simple_agent` | `gui_grounding_simple_agent` |
| Tasks | 1244 | 1244 |
| Repeats | 1 | 1 |

Generated 2026-10-18T13:46:03+00:00 by nemo-gym 0.7.0.

Command: `gym eval compare '+baseline_rollouts_jsonl_fpath="results/ssv2_clean_base_qwen35.jsonl"' '+candidate_rollouts_jsonl_fpaths=["/home/ubuntu/night/evals/final-round1-grpo/ssv2_clean_final-round1-grpo/rollouts.jsonl"]' '+output_dirpath="/home/ubuntu/night/evals/compare_ssv2_base_vs_final"' +report_format=both`

### Key metrics

| Metric | Δ (cand − base) | 95% CI for Δ (cand − base) | Baseline | Baseline 95% CI | Candidate | Candidate 95% CI |
|---|---:|---|---:|---|---:|---|
| `mean/reward` | +0.0410 (+4.8%) | — | 0.8625 | — | 0.9035 | — |
| `mean/img_w` | +0.0000 (+0.0%) | — | 1704.63 | — | 1704.63 | — |
| `mean/img_h` | +0.0000 (+0.0%) | — | 1659.91 | — | 1659.91 | — |
| `mean/parsed` | +0.0040 (+0.4%) | — | 0.9960 | — | 1.0000 | — |
| `mean/pred_x` | +24.72 (+5.0%) | — | 489.64 | — | 514.36 | — |
| `mean/pred_y` | +2.7140 (+0.6%) | — | 422.35 | — | 425.06 | — |
| `mean/input_tokens` | +0.0000 (+0.0%) | — | 1821.82 | — | 1821.82 | — |
| `mean/output_tokens` | -22.54 (-57.4%) | — | 39.28 | — | 16.74 | — |
| `mean/total_tokens` | -22.54 (-1.2%) | — | 1861.10 | — | 1838.56 | — |
| `mobile/pass@1/accuracy` | +1.7964 (+2.0%) | — | 91.42 | — | 93.21 | — |
| `mobile/pass@1[avg-of-1]/accuracy` | +1.7964 (+2.0%) | — | 91.42 | — | 93.21 | — |
| `desktop/pass@1/accuracy` | +4.9020 (+5.8%) | — | 83.99 | — | 88.89 | — |
| `desktop/pass@1[avg-of-1]/accuracy` | +4.9020 (+5.8%) | — | 83.99 | — | 88.89 | — |
| `web/pass@1/accuracy` | +6.1785 (+7.5%) | — | 81.92 | — | 88.10 | — |
| `web/pass@1[avg-of-1]/accuracy` | +6.1785 (+7.5%) | — | 81.92 | — | 88.10 | — |
| `icon/pass@1/accuracy` | +7.2222 (+9.2%) | — | 78.15 | — | 85.37 | — |
| `icon/pass@1[avg-of-1]/accuracy` | +7.2222 (+9.2%) | — | 78.15 | — | 85.37 | — |
| `text/pass@1/accuracy` | +1.7045 (+1.8%) | — | 92.47 | — | 94.18 | — |
| `text/pass@1[avg-of-1]/accuracy` | +1.7045 (+1.8%) | — | 92.47 | — | 94.18 | — |
| `mobile/icon/pass@1/accuracy` | +5.2133 (+6.3%) | — | 82.94 | — | 88.15 | — |
| `mobile/icon/pass@1[avg-of-1]/accuracy` | +5.2133 (+6.3%) | — | 82.94 | — | 88.15 | — |
| `mobile/text/pass@1/accuracy` | -0.6897 (-0.7%) | — | 97.59 | — | 96.90 | — |
| `mobile/text/pass@1[avg-of-1]/accuracy` | -0.6897 (-0.7%) | — | 97.59 | — | 96.90 | — |
| `desktop/icon/pass@1/accuracy` | +10.32 (+13.8%) | — | 74.60 | — | 84.92 | — |
| `desktop/icon/pass@1[avg-of-1]/accuracy` | +10.32 (+13.8%) | — | 74.60 | — | 84.92 | — |
| `desktop/text/pass@1/accuracy` | +1.1111 (+1.2%) | — | 90.56 | — | 91.67 | — |
| `desktop/text/pass@1[avg-of-1]/accuracy` | +1.1111 (+1.2%) | — | 90.56 | — | 91.67 | — |
| `web/text/pass@1/accuracy` | +5.1282 (+5.9%) | — | 87.61 | — | 92.74 | — |
| `web/text/pass@1[avg-of-1]/accuracy` | +5.1282 (+5.9%) | — | 87.61 | — | 92.74 | — |
| `web/icon/pass@1/accuracy` | +7.3892 (+9.8%) | — | 75.37 | — | 82.76 | — |
| `web/icon/pass@1[avg-of-1]/accuracy` | +7.3892 (+9.8%) | — | 75.37 | — | 82.76 | — |
| `screenspot_v2/mobile/pass@1/accuracy` | +1.7964 (+2.0%) | — | 91.42 | — | 93.21 | — |
| `screenspot_v2/mobile/pass@1[avg-of-1]/accuracy` | +1.7964 (+2.0%) | — | 91.42 | — | 93.21 | — |
| `screenspot_v2/desktop/pass@1/accuracy` | +4.9020 (+5.8%) | — | 83.99 | — | 88.89 | — |
| `screenspot_v2/desktop/pass@1[avg-of-1]/accuracy` | +4.9020 (+5.8%) | — | 83.99 | — | 88.89 | — |
| `screenspot_v2/web/pass@1/accuracy` | +6.1785 (+7.5%) | — | 81.92 | — | 88.10 | — |
| `screenspot_v2/web/pass@1[avg-of-1]/accuracy` | +6.1785 (+7.5%) | — | 81.92 | — | 88.10 | — |

Displayed baseline/candidate values and each delta use `mean_across_repeats/<metric>` when recorded, otherwise the metric value. Delta CI = two-sided independent-sample 95% Welch t-interval for candidate minus baseline, computed from the named metric's values in each run's `repeat_level_metrics`. `—` means the required interval data was unavailable: delta intervals require at least 2 finite repeat-level values on each side. Baseline/candidate CI = 95% t-interval of the per-repeat mean across repeats, read verbatim from `ci_{low,high}_95_across_repeats/<metric>` in `*_aggregate_metrics.json`.

### Sample flips

Field: `reward` (task mean over its repeats). A task passes when its mean is above 0.5.

**1244 common tasks · 38 pass→fail · 89 fail→pass · net +51 · 1117 unchanged · 0 tied**

| Task | Direction | Baseline | Candidate | Δ | Baseline per-repeat | Candidate per-repeat |
|---:|---|---:|---:|---:|---|---|
| 1 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 203 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 229 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 238 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 283 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 342 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 346 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 390 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 411 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 422 | pass→fail | 1.0000 | 0.0000 | -1.0000 | 1 | 0 |
| 65 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 87 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 94 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 129 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 144 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 160 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 228 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 258 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 320 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |
| 339 | fail→pass | 0.0000 | 1.0000 | +1.0000 | 0 | 1 |

… and 107 more (full list in `compare_report.json`).

Tasks are joined on `_ng_task_index`. `*_aggregate_metrics.json` carries no task identity, so flips are reported by index only — this assumes both runs used the same dataset, split, `--limit` and ordering.

### Notes

- Neither run recorded per-run cross-repeat confidence intervals, so every baseline/candidate CI cell is empty. They are written for repeat-aggregated metrics when a run has 2 or more repeats.

---
Generated by `gym eval compare` (nemo-gym 0.7.0). Machine-readable output: `compare_report.json` (schema_version 1).
