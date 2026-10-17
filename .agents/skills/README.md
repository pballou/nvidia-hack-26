# Agent skills (vendored from NeMo Gym)

These skills are copied unmodified from [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym),
tag `v0.7.0` (commit `1f8afc7`), path `.agents/skills/`. They are Apache-2.0
licensed; see `LICENSE-NeMo-Gym`. `.claude/skills/` contains relative symlinks to
these directories, the same layout Gym uses, so both Claude Code and Codex find them.

| Skill | Use it for | First published in Gym |
| --- | --- | --- |
| `nemo-gym-blade-analysis` | **Failure analysis after every comparable pair of `gym eval run`s** (e.g. base vs. tuned). BLADE (Benchmark Level Analysis and Diagnostics Engine) turns rollout artifacts into an evidence-backed report: task outcome buckets, failure taxonomy (KG/UK/BI/TI/IR/DA), and the intervention to try next. | 2026-06-22 |
| `nemo-gym-reward-profiling` | `gym eval profile`: per-task pass rates, used to pick GRPO prompts with 0 < pass rate < 1. | 2026-05-11 |
| `nemo-gym-debugging` | Failed or hanging Gym jobs, request-boundary issues, vLLM tool-call schema checks. | 2026-04-27 |
| `add-benchmark` | Adding a benchmark (our ScreenSpot-v2 env) in Gym's layout. | 2026-07-19 |

Path notes:
- These skills were written for a Gym checkout. Paths like
  `resources_servers/my_benchmark/` correspond to the same layout in this repo.
- The BLADE helper is `.agents/skills/nemo-gym-blade-analysis/scripts/blade_toolkit.py`.
  It uses only the Python standard library.
- **`add-benchmark`:** skip every GitLab/MLflow dataset-registry step
  (`gym dataset upload --storage gitlab`, `gitlab_identifier`, `mlflow_tracking_*`,
  `+data_source=gitlab`). We have no registry.
  - Keep train/dev JSONL in `/home/ubuntu/data` and on HF.
  - Commit only `data/example*.jsonl`.
  - Never ask for a GitLab/MLflow token.
- To update, re-copy from a newer Gym tag and update the tag above.
