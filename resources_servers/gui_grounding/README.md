# gui_grounding

GUI click grounding: a screenshot plus an instruction go in, and the model answers
with a click point (`{"point_2d": [x, y]}` on a 0–1000 grid). The reward is 1 if the
point lands inside the target element's pixel box (inclusive), otherwise 0. Prompt
text, output parsing and coordinate conversion all come from `common/grounding.py`.

Adapted from NeMo Gym v0.7.0's `resources_servers/gui_coordinate` (provenance in
`app.py`).

## Data

Rows are built by `data_prep/gym_rows.py` and stay on the box, never in git. Only
`data/example.jsonl` is committed: 5 OS-Atlas rows with `file://` images.

| Set | Command | Rows |
| --- | --- | --- |
| OS-Atlas dev (checkpoint selection) | `python -m data_prep.gym_rows manifest --manifest <slice>/manifest.jsonl --split dev --out /home/ubuntu/data/gym/os_atlas_dev.jsonl` | 969 (5% of screenshot groups per source) |
| ScreenSpot-v2-clean (held-out test) | `python -m data_prep.gym_rows ssv2 --out /home/ubuntu/data/gym/ssv2_clean.jsonl` | 1,244 |

Each row carries the canonical fields `verify()` needs (`bbox`, `img_w`, `img_h`,
`platform`, `data_type`, `source`, `id`, `instruction`). Metrics are broken down by
`platform`, `data_type`, `platform_data_type` and `source`.

## Run (on the box, from the repo root)

```bash
export NEMO_GYM_EXTRA_ROOTS=$PWD && source .venv-gym/bin/activate
# env.yaml: policy_base_url http://127.0.0.1:8001/v1, policy_api_key EMPTY,
#           policy_model_name base (or grounder for the LoRA adapter)
gym env start --resources-server gui_grounding --model-type vllm_model ++server_spinup_timeout_seconds=1800
gym eval run --no-serve --agent gui_grounding_simple_agent \
  --input /home/ubuntu/data/gym/ssv2_clean.jsonl --output results/ssv2_clean_<model>.jsonl --concurrency 64
gym env test --resources-server gui_grounding   # unit tests
```

## GRPO reward

TRL's reward function POSTs each completion to this server's `/verify`, with the
row's fields plus a minimal Responses-API `response` that wraps the completion text.
`tests/test_app.py` builds the same object with `_request()`.
