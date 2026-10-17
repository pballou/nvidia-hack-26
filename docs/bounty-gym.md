# Best Model Improvement With NVIDIA NeMo Gym: GUI Assist

> **Status:** draft of Sat 2026-10-17. **TBD** = produced by the overnight loop (Sat night, hard
> end 02:50 Sun) or by Sunday's runs. Every result number (accuracies, s/step, call counts,
> rewards) is quoted from `docs/results.md` or `docs/runs.md`. Loop and env settings are quoted
> from the code cited next to them.

## 1. Summary

GUI Assist's grounder is Qwen3.5-2B with a LoRA adapter. It takes a screenshot and a click
instruction and answers `{"point_2d": [x, y]}` on a 0-1000 grid. NeMo Gym 0.7.0 is the measuring
stick for every checkpoint and, in the overnight loop (TBD), the training signal. So far only the
20-step GRPO smoke has trained on a Gym reward.

- **Score.** The `gui_grounding` env (reward = the point lands inside the target box) scores every
  checkpoint on our OS-Atlas `dev` split, and on ScreenSpot-v2-clean for the results table only.
- **Pick the RL data (overnight loop).** `gym eval run --num-repeats 8 --temperature 1.0` and
  `gym eval profile` on training rows will give per-row pass rates. The GRPO pool is the rows with
  0 < pass rate < 1.
- **Reward (overnight loop).** TRL GRPO's reward function POSTs each completion to the same env's
  `/verify`, so the training reward's point-in-box term and the eval metric are the same code.
  GRPO adds a 0.1 bonus for a parsable point (`format_reward`, a knob).
- **Decide (overnight loop).** A round is kept only if dev accuracy (temperature 0) gains at least
  +0.5 points; otherwise the loop reverts and stops. `gym eval compare` against the previous dev
  rollouts is saved alongside as the per-task flip report. BLADE on the round's artifacts may
  retune a short list of pre-approved knobs.

**So far:** LoRA SFT (TRL, scored through Gym) took dev from 76.4% to **85.66%** (+9.3) and
ScreenSpot-v2-clean from 86.3% to **89.07%** (+2.8; icons 78.1 → 84.8, +6.7). The Gym-reward
GRPO path passed its 20-step gate: 31.9 s/step, 960 `/verify` calls, 0 failures. Whether GRPO
rounds beat SFT on dev is **TBD** (overnight). A second env, `web_agent`, measures the grounder
inside the real agent: Lightning plans in a Playwright sandbox, and the reward is the task's check
on app state.

## 2. What we built in Gym 0.7.0

Both envs live in this repo in Gym's layout (`resources_servers/<name>/{app.py,configs,tests}`),
run on `nemo-gym==0.7.0` from PyPI, and don't fork Gym. The grounder is one vLLM process: served
name `base` = adapter off, `grounder` = adapter on.

### 2a. `gui_grounding` (stream A: Charlie, Kyle)

| | |
| --- | --- |
| **Scores** | Screenshot + instruction → one click point. Prompt, parser and 0-1000 ↔ pixel conversion all come from `common/grounding.py`, the same code SFT, GRPO and the agent use. |
| **Reward** | 1.0 if `to_pixels(point)` is inside `bbox` (inclusive edges, as ScreenSpot does), else 0.0 (`app.py:score`). `<think>` blocks are stripped first. A reply with no parsable point is a scored miss (`failure_reason: "no parsable click point"`), not a mask. |
| **Row** | `responses_create_params` (an `input_image` with a `file://` URL to the pre-resized PNG, the grounding prompt, temperature 0) plus `id, source, instruction, platform, data_type, bbox [x1,y1,x2,y2] px, img_w, img_h`, all declared on `GuiGroundingRunRequest` so they reach `verify()`. Rows come from one script, `data_prep/gym_rows.py`. |
| **VerifyResponse** | `GuiGroundingVerifyResponse` adds `platform_data_type`, `parsed`, `pred_x`, `pred_y`. `verify` is annotated `-> GuiGroundingVerifyResponse`, so the extras reach rollouts and metrics. |
| **Failures** | Any exception while scoring returns `reward=0, mask_sample=True, failure_reason="verify error: …"` instead of raising. The row fields themselves are required, which is safe only because every row comes from `gym_rows.py` (`web_agent`, below, makes every field optional). |
| **Metrics** | `compute_subset_metrics` per `platform`, `data_type`, `platform_data_type` and `source` (e.g. `mobile/icon/pass@1/accuracy`); `get_key_metrics` surfaces the per-slice accuracies. |
| **Data** | `dev`: 969 OS-Atlas rows, 5% of screenshots per source, grouped so no screenshot is in both train and dev. ScreenSpot-v2-clean: 1,244 rows = SSv2 (revision `5efbb1f1b5`) minus 28 desktop rows captured on the same machines as our Linux/macOS training data (`benchmarks/screenspot_v2/excluded_rows.json`); `gym_rows.py` asserts 1,272 → 1,244. |
| **Provenance** | Adapted from Gym's `resources_servers/gui_coordinate` (tag commit `1f8afc71d4`; path last changed 2026-09-29 in `016bb6b811`). The distance reward became a binary point-in-box reward on pixel boxes, plus per-slice metrics. |
| **Tests** | `tests/test_app.py`, 7 test functions: inclusive box edges, Qwen3.5's fenced-JSON base format, think stripping, unparsed = miss, a malformed row masks instead of raising, slice metrics. `_request()` builds the same body GRPO sends. |

```bash
export NEMO_GYM_EXTRA_ROOTS=$PWD && source .venv-gym/bin/activate   # repo root, on the box
# env.yaml: policy_base_url http://127.0.0.1:8001/v1, policy_model_name base (or grounder)
gym env start --resources-server gui_grounding --model-type vllm_model ++server_spinup_timeout_seconds=1800
gym eval run --no-serve --agent gui_grounding_simple_agent \
  --input /home/ubuntu/data/gym/os_atlas_dev.jsonl --output results/dev_<model>.jsonl --concurrency 64
gym env test --resources-server gui_grounding
```

### 2b. `web_agent` (stream C: Patrick)

The agent-level env: Nemotron 3.5 Lightning (text only) gets the request and a text description of
the screen, and acts through five tools (`click`, `type_text`, `press_key`, `scroll`, `done`). A
`click` target description goes to the grounder, and Playwright clicks the returned pixel.

- **Same agent code.** Every tool route calls the interactive agent's `AgentSession.execute`, and
  the tool result is byte-identical to the interactive planner's (`json.dumps(payload,
  ensure_ascii=False)`, covered by a test). Prompt and tools come from `agent/prompts.py`, and
  `make_tasks.py --check` rejects rows whose prompt or tools drifted.
- **Reward from sandbox state.** `verify` reads `GET /api/state?sid=` and runs
  `sandbox.checks.evaluate(state, row.check)`: 1.0 or 0.0. Only the verifier side (`verify` and the
  `done` snapshot) reads state. The DOM oracle is test-only, and the server refuses to start if the
  config enables it.
- **Start screen baked into rows.** Gym's legacy `/run` drops `seed_session`'s response body, so
  `make_tasks.py` precomputes the start observation with the agent's own code path and stores it in
  the first user message, `initial_observation` and `initial_screen_sha`. `seed_session` re-takes
  the screenshot, primes the session without a VLM call, and flags `initial_obs_stale` if the
  screen differs. Rows record `prompt_version`, and a `.meta.json` sidecar records the command,
  config and prompt hashes (the start screens are model output).
- **Episode end.** `simple_agent` ends on a reply without a tool call, not on `done`. So `done`
  snapshots the sandbox state and tells the model to stop, later calls are refused with
  `ALREADY_DONE`, and `verify` scores the snapshot. `max_steps: 12` matches the interactive limit,
  and rows keep `tool_choice` unset (`auto`).
- **Hallucinated lifecycle calls are harmless.** A model "calling" `seed_session`,
  `close_session` or `verify` can't reset, end or rescore its episode: a seed without a row or for
  a live episode is ignored, an empty close closes nothing, and a verify body counts only with
  `simple_agent`'s shape and the episode's `task_id`, which the model never sees. A catch-all
  `POST /{tool_name}` answers invented tool names with a corrective `ok:false`.
- **Base vs tuned is config only.** `grounder_model` / `summary_model` are Gym config fields
  (`base` | `grounder`), recorded on every rollout. Unknown config keys make the server refuse to
  start, so a typo can't silently turn base-vs-tuned into base-vs-base.
- **VerifyResponse fields:** `task_id, tags, steps, n_clicks, n_invalid_calls, n_failed_actions,
  n_calls_after_done, n_vlm_infra_errors, done_called, claimed_success, initial_obs_stale,
  state_source, check_explanation, grounder_model, summary_model, prompt_version`, per-stage
  `*_ms` totals and per-call `latency_ms`. `verify` is annotated `-> WebAgentVerifyResponse`. Gym
  auto-profiles the ints and bools, and `compute_metrics` adds `tag/<tag>/mean_reward` (easy /
  medium / hard, …).
- **Tests:** `tests/test_app.py` (37 test functions: 36 with fakes, plus one real-Chromium
  episode against the fake sandbox) and `tests/test_agent_metrics.py` (11).
- **Provenance:** adapted from Gym's `resources_servers/interactive_browser` (tag `v0.7.0`, commit
  `1f8afc7`); per-path dates are in the env's README.

**Masking.** `verify` never raises and never returns non-2xx, and every request field is optional.
Infrastructure failures return `reward=0, mask_sample=True` with a `failure_kind`, whatever the
reward would have been:

| `failure_kind` | When |
| --- | --- |
| `provider_unavailable` | Sandbox down at seed or verify, or browser launch/navigation failure at seed; any VLM transport/timeout/HTTP error, browser crash or hang, or tool timeout during the episode |
| `session_lost` | No session at verify, or the sandbox state's `meta.sid` / `meta.task_id` belongs to another session |
| `web_agent:bad_row` / `web_agent:reset_rejected` | Row lacks `task_id`, `start_path` or `check`; sandbox answered `/api/reset` with a 4xx |
| `web_agent:bad_request` | The `/verify` body isn't `simple_agent`'s (e.g. the model "called" `verify`) |
| `verifier_error` | Anything else |

Masking only the outage rollouts that failed would inflate success during an outage, and an
unmasked outage would bias base vs grounder. Model-quality failures (unparsable point, empty
screen description) stay scored.

```bash
# On Patrick's Mac (Lightning :8000, sandbox :8090, VLM :8001 through the tunnel; configs/agent.yaml)
scripts/agent_eval.sh                       # base then grounder: gym env start → eval run → profile, then compare
# 12 tasks × 3. tasks_all.jsonl isn't committed: build it first with make_tasks.py (§6)
TASKS=resources_servers/web_agent/data/tasks_all.jsonl REPEATS=3 scripts/agent_eval.sh
.venv-gym/bin/gym env test --resources-server web_agent
```

`agent_eval.sh` refuses placeholder start screens and any local Lightning other than the official
BF16 checkpoint (`FORCE=1` allows both for plumbing, recorded as `reportable=no`), runs sequentially,
routes failed rollouts to a sidecar (`+route_failures_to_sidecar=true`), and ends with
`gym eval compare` base vs grounder. `scripts/agent_metrics.py` then re-validates every tool call
from `response.output` and reports success, claimed vs verified, validity and per-stage p50/p90.

## 3. The improvement loop

`training/night_loop.py` (`Loop`), launched by `training/night_loop.sh` in tmux. The spec is
`docs/tasks/B-night-loop.md`; Gym calls go through `training/gymctl.py`.

```text
 round 0: the SFT adapter picked on dev ── dev eval @ T=0 (Loop.baseline)
      │
      ▼
 ┌─► profile: 4,000 unused train rows × 8 @ T=1.0     gymctl.eval_run(--num-repeats 8) → gymctl.profile
 │        │                                            (gym eval profile)
 │        ▼
 │   pool: 0 < pass rate < 1, ≥ 4 rollouts, knobs     pool_from_profile → pool.jsonl, pool_stats.json
 │        │
 │        ▼
 │   TRL GRPO, reward = POST gui_grounding /verify    training/grpo.py: GymVerify + 0.1 × format_ok
 │        │   gate D2 at step 20: ≤ 35 s/step, else one rejection-sampled SFT round, then stop
 │        ▼
 │   dev eval @ T=0 (969 rows) + gym eval compare     Loop.dev_eval, gymctl.compare
 │        │
 │        ▼
 │   keep if dev gains ≥ +0.5 pts, else revert, stop  Loop.round
 │        │
 │        ▼
 └── BLADE on the round's artifacts (≤ 10 min) ──► knobs.json        Loop.wait_for_blade
```

- **Profile (`Loop.profile`).** A fresh random 4,000-row train subset each round (seed = round;
  not reused across rounds, and the code asserts none are in dev), run through
  `gym eval run --num-repeats 8 +num_repeats_add_seed=true --temperature 1.0` with
  `max_output_tokens` 32 (= GRPO's completion length) and `++route_failures_to_sidecar=true`,
  then `gym eval profile ++allow_partial_rollouts=true`.
- **Pool (`pool_from_profile`).** Rows that always or never pass tend to give every completion in
  a GRPO group the same reward, so they carry little signal. The pool keeps 0 < pass rate < 1,
  drops rows with fewer than 4 scored rollouts, applies `exclude_ids` and `source_weights`, and
  caps at `pool_max`: weighted sampling without replacement above the cap; below it, a weight
  above 1 repeats rows and a weight below 1 subsamples them. Per-source always / sometimes /
  never counts go to `pool_stats.json`.
- **GRPO (`training/grpo.py`).** TRL `GRPOTrainer`, colocated vLLM (`vllm_mode="colocate"`,
  memory 0.25), continuing the current LoRA. G = 6, 8 prompts per step, lr 1e-5, temperature 1.0,
  `max_completion_length` 32, thinking off, `beta` 0, DAPO loss, group-scaled rewards, vLLM
  importance-sampling correction. `GymVerify` posts the row's canonical fields plus a minimal
  Responses-API `response` wrapping the completion, makes up to 3 attempts per call, and raises
  (ending the run) if more than 5% of a batch's calls fail. Pool rows are Gym rows, so the prompt
  and pre-resized screenshot are exactly what Gym evals send, and `check_format` asserts the
  training prompt matches serving.
  Four TRL 1.16 / vLLM 0.31 workarounds are documented in the file's docstring (image-pad
  collapse, left padding, image caps for the colocated engine, exact weight restore after unmerge).
- **Gate D2, in process (`Control`).** After 20 steps, the median s/step (excluding the first two)
  must be ≤ 35, otherwise no adapter is saved and the exit code is 10. Then, as on a crash before
  the smoke or a pool under 32 rows, the loop runs **one rejection-sampled SFT round** on up to 2
  distinct reward=1 profile replies per row (`rs_sft_rows`), which is still Gym-driven data.
- **Keep or revert.** The candidate is evaluated on dev and compared with the previous dev
  rollouts. Kept only if dev gains ≥ +0.5 points; otherwise the loop reverts and stops.
- **Stops:** 3 rounds, no round starts after 01:30, no GRPO step after 02:40, hard end 02:50
  (every step timeout is capped there), no dev gain, a `STOP` file, or any crash (revert to the
  last kept adapter, `stop_reason` recorded; a GRPO crash after a passed smoke first dev-evaluates
  its last checkpoint, which the normal keep rule can keep). `finish()` serves the best kept adapter on `:8001` as
  `grounder` and writes `FINAL.json` last.
- **Artifacts per round:** `round_<r>/` holds the profile rollouts and profile, `pool.jsonl`,
  `pool_stats.json`, `knobs_used.json`, `grpo_command.txt`, `grpo.log`, dev rollouts, the
  `gym eval compare` report and `summary.json` (which embeds the trainer's `grpo_summary.json`;
  adapters go to `/home/ubuntu/ckpt/night/<launch>/round_<r>/`). A `docs/runs.md` row is appended
  per round.

**What BLADE may change.** Only `knobs.json`, validated by `load_knobs`: `pool_max`,
`num_generations`, `grpo_max_steps`, `round_time_budget_min`, `format_reward` (clamped to
`KNOB_LIMITS`), `source_weights` (oversample "unreliable knowledge" slices) and `exclude_ids` (drop
rows BLADE flags as data artifacts). Any other key is ignored and noted. Eval sets, prompts, the
model and the learning rate never change overnight, and ScreenSpot is never touched inside the loop.

| Automated (`night_loop.py`) | Manual |
| --- | --- |
| Profile, pool, GRPO + gate D2, RS-SFT fallback, dev eval, `gym eval compare`, keep/revert, stop rules, serving the best adapter, `FINAL.json`, `runs.md` rows | Picking round 0 on dev (template vs Lightning-rewritten SFT); the BLADE pass and `knobs.json` edits (the loop waits ≤ 10 min, `BLADE_DONE` continues at once); SSv2-clean once on the final adapter; the `web_agent` base-vs-tuned eval and its BLADE pass |

## 4. Results so far

**Grounding** (Gym `gui_grounding`, Qwen3.5-2B, 2 MP, temperature 0; checkpoints chosen on dev):

| Model | Dev (969) | SSv2-clean (1,244) | icon / text | mobile / desktop / web | Source |
| --- | --- | --- | --- | --- | --- |
| Base | 76.4% | 86.3% | 78.1 / 92.5 | 91.4 / 84.0 / 81.9 | `runs.md` Sat 15:57, 16:02 |
| Pilot SFT (2,000 raw-label rows) | 81.3% (+5.0) | — | — | — | `runs.md` Sat 16:13 |
| **SFT template** (18,652 rows, 1 epoch, LoRA r16) | **85.66%** (+9.3) | **89.07%** (+2.8) | **84.8 (+6.7)** / 92.3 | 92.2 / 89.5 / 85.1 | `runs.md` Sat 17:31 |
| SFT with Lightning-rewritten instructions | TBD | TBD | TBD | TBD | ablation, T3 |
| Night-loop best kept round (method from `FINAL.json`: `grpo` / `rejection_sampled_sft`; round 0 = SFT if none kept) | TBD | TBD (once, Sunday) | TBD | TBD | `FINAL.json` |

Dev by source, base → pilot → SFT template: Linux 68.8 → 80.7 → 86.9, macOS 73.4 → 79.1 → 91.1,
AndroidWorld 70.5 → 74.1 → 81.3, UIBert 78.4 → 82.1 → 85.1, FineWeb 82.9 → 85.1 → 84.5. The pilot
also cut answer length from 37.6 to 16.7 tokens. The SSv2 three-way report (all 1,272 / 1,128 on
v1 images / 144 new, research-notes §5) is TBD. The `gym eval compare` reports for base → SFT aren't in
`runs.md` yet (TBD).

**GRPO through Gym:**

| Run | Setup | Result | Source |
| --- | --- | --- | --- |
| Gate D2 smoke | 20 steps from the pilot adapter, 200 random train rows (not a profiled pool), G=6, 8 prompts/step, lr 1e-5, reward = `/verify` + 0.1 format, throwaway Gym env on `:11020`/`:12020` | **Pass: 31.9 s/step** median while sharing the GPU with the template SFT; 960 `/verify` calls, 0 failures; verify reward 0.75 → 0.79; vLLM-vs-trainer logp diff ~0.01-0.03, IS ratio ≈ 1; peak 45 GiB | `runs.md` Sat 17:13 |
| Eval-path check | `gymctl eval` of the pilot adapter (`:8004`, head `:11010`) | 81.32%, identical to the 16:13 pilot eval | `runs.md` Sat 17:10 |
| Loop rehearsal | — | TBD (no committed record) | — |
| Rounds 1-3 | profile 4,000 × 8 → pool → GRPO → dev | TBD: pool size, steps, s/step, reward first → last, dev prev → new, kept | `round_<r>/summary.json` |

The smoke's 0.75 → 0.79 is 20 steps and shows plumbing, not improvement.

**Agent (`web_agent`)**, official BF16 Lightning, screen descriptions by `base`:

| Run | Grounder | Rows × repeats | Reward | Masked | Claimed | Tool-call validity | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sat 16:43, Gym acceptance | `base` | 5 × 1 | 3/5 | 0 | 2/5 (1 false claim) | 42/43 | n=1, prompt `agent-v3` |
| Sun, Gym | `base` | 12 × 3 | TBD | TBD | TBD | TBD | final, `agent-v4` |
| Sun, Gym | `grounder` (final) | 12 × 3 | TBD | TBD | TBD | TBD | final, `agent-v4` |
| Sun, `gym eval compare` base vs grounder | — | — | TBD | — | — | — | — |

- **Acceptance read** (`runs.md` Sat 16:43): 0 stale start screens; `text-size-large` failed on
  grounding (the L option was asked for, M was clicked).
- **Outside Gym** (interactive harness, same `AgentSession` code, 4 easy tasks, n=1, `agent-v3`):
  base grounder 3/4 verified, Kyle's 2k-row pilot adapter 4/4 (it fixed the `text-size-large`
  miss). Not final, and not a Gym number (`results.md`).
- The Saturday reads ran on prompt `agent-v3`, whose examples overlapped some sandbox tasks, so
  their absolute numbers are optimistic. The committed rows are now `agent-v4`. Base vs tuned
  stays fair because both sides get identical rows, prompts and tools.

**BLADE findings: TBD.** No BLADE report is committed yet. Its input so far is the failure
clusters from the Saturday 12-task interactive read (`results.md`): scroll-dependent targets,
the canvas chart, a dense toggle grid with a false done claim, and one planner error (`pay-50`).
Which knobs (grounding) or data, prompt or verifier changes (agent) BLADE leads to: TBD.

## 5. Gym gotchas we hit and fixed

All from `docs/research-notes.md` §3, AGENTS.md, the env READMEs and code comments.

| Gotcha | What happens | What we did |
| --- | --- | --- |
| Row fields not declared on the VerifyRequest subclass | Silently dropped; `verify()` never sees `bbox` or `check` | Both envs declare every field they read (`GuiGroundingRunRequest`, `WebAgentSeedSessionRequest` / `WebAgentVerifyRequest`) |
| `verify` annotated `-> BaseVerifyResponse` | FastAPI keeps only `reward`; slice fields never reach rollouts or metrics | Annotated with each env's own response class |
| Non-2xx from `seed_session`, the model call or `verify` (a 422 included) | Aborts the whole `gym eval run` | `web_agent`'s verify and tool routes parse bodies themselves; every seed/verify field is optional, and nothing on the legacy `/run` path raises (only the typed session API rejects a closed or mismatched session; the env passes Gym's session-conformance check); `gui_grounding` masks scoring errors; `agent_eval.sh` and the night-loop profile pass `route_failures_to_sidecar`; the night loop retries a dev eval once. A *tool* exception doesn't abort the run in 0.7.0 (the 500 body goes back to the model; research-notes §3, AGENTS.md), but `web_agent` still turns it into a 200 `ok:false` and counts it |
| `mask_sample=True` | The rollout leaves every quality metric and shows only in `coverage/*` | Infra failures mask the rollout whatever its reward; model-quality failures stay scored |
| `simple_agent` ignores `done`; `max_steps` counts model calls (checked after that turn's tools) | Episodes run on; `tool_choice: required` would run every task to `max_steps` | `done` snapshots state and tells the model to stop; `max_steps: 12` = the interactive budget; `make_tasks.py --check` rejects rows that set `tool_choice` |
| Legacy `/run` drops `seed_session`'s response body | The policy never sees a start screen returned by the server | Start observation baked into each row, with a screenshot hash and an `initial_obs_stale` flag |
| Any tool name the model emits is POSTed to `/{name}`; bad-JSON arguments never reach the server | Hallucinated `seed_session` / `close_session` / `verify` hit lifecycle routes; unknown names get a 404; server-side counts miss bad-JSON calls | Catch-all `POST /{tool_name}` registered last; repeat seeds, empty closes and verify bodies without `simple_agent`'s shape do nothing; `agent_metrics.py` re-validates calls from `response.output` |
| Gym's HTTP client has no timeout | A hung tool call hangs the rollout | Each call bounded by `tool_timeout_s` (240 s); a timeout returns `ok:false` and masks the rollout |
| Legacy `/run` never calls `close_session`; `num_workers > 1` splits state | Leaked browser contexts; tool calls land on a worker without the session | Release in `verify`'s `finally`; each seed closes episodes older than 1 h; `num_workers: 1` enforced at startup; sandbox sid = `gym-` + Gym's session id, so `--num-repeats` runs the same task concurrently |
| Venv build and imports | A root `pyproject.toml` makes every server editable; `gym env start` doesn't put the repo root on PYTHONPATH; a cold build takes ~8-9 min; `skip_venv_if_present` ignores requirements changes | No root `pyproject.toml` or `resources_servers/__init__.py`; each `app.py` puts the repo root on `sys.path` (`web_agent` after `import nemo_gym`, which reorders it); `++server_spinup_timeout_seconds=1800`; `REINSTALL=1` in `agent_eval.sh` |
| `env.yaml` merges after every config; unknown server config keys are dropped | `policy_*` overrides any `${policy_*}`; the merged config (keys included) is served at `:11000/global_config_dict_yaml`; a typo keeps the default | `web_agent`'s planner is `lightning_policy_model` with literal URLs and `${oc.env:NVIDIA_API_KEY}`; no keys in `env.yaml` or on the CLI; `WebAgentConfig` refuses unknown keys |
| Head port fixed at `:11000`; `--model` ignored with `--no-serve`; `/readyz` waits for the policy | A second env collides; the policy can't be switched per run; the env can't restart while the eval server is down for GRPO | `gymctl` runs its own env (head `:11010`, `/verify` pinned at `:12010`; the one-off eval CLI uses `:11011`/`:12011`), sets the policy at `env start` (`++policy_model_name=grounder`) and passes `++model_endpoint_readiness_timeout_seconds=0` |

Also: screenshots are `file://` URLs to the pre-resized PNGs, since base64 images would be copied
into every materialized-input and rollout file.

## 6. Reproduce

Grounding, SFT, GRPO and the night loop run **on the Brev box**, from the repo root (stack pinned
in `docs/gpu-box.md`). Gym commands use `.venv-gym`; `training.*` runs in the training venv
(`/home/ubuntu/venvs/train`, `docs/gpu-box.md` §5; `sft_full.sh` and `night_loop.sh` activate it
themselves). Log every GPU run in `docs/runs.md`.

```bash
export NEMO_GYM_EXTRA_ROOTS=$PWD && source .venv-gym/bin/activate   # Gym venv

# Gym rows (stay on the box)
python -m data_prep.gym_rows manifest --manifest /home/ubuntu/data/os_atlas_20k/manifest.jsonl \
  --split dev --out /home/ubuntu/data/gym/os_atlas_dev.jsonl
python -m data_prep.gym_rows ssv2 --out /home/ubuntu/data/gym/ssv2_clean.jsonl

# Base baseline: served name `base` on :8001 (env.yaml policy_*)
gym env start --resources-server gui_grounding --model-type vllm_model ++server_spinup_timeout_seconds=1800
gym eval run --no-serve --agent gui_grounding_simple_agent --input /home/ubuntu/data/gym/os_atlas_dev.jsonl \
  --output results/dev_base_qwen35.jsonl --concurrency 64
gym eval run --no-serve --agent gui_grounding_simple_agent --input /home/ubuntu/data/gym/ssv2_clean.jsonl \
  --output results/ssv2_clean_base_qwen35.jsonl --concurrency 64

# SFT, then dev (+ SSv2-clean for the table) through Gym on its own ports
training/sft_full.sh template /home/ubuntu/data/os_atlas_20k/manifest.jsonl
(source /home/ubuntu/serving.env && source /home/ubuntu/venvs/train/bin/activate && \
  python -m training.gymctl eval --adapter /home/ubuntu/ckpt/sft-template/adapter --name sft-template --ssv2 --gpu-util 0.35)

# Gate D2 smoke. First a throwaway gui_grounding env with /verify on :12020, in its own tmux pane
# (Gym venv). runs.md doesn't record that env's exact command; this uses gymctl.GymEnv's port overrides.
gym env start --resources-server gui_grounding --model-type vllm_model ++server_spinup_timeout_seconds=1800 \
  ++head_server.port=11020 ++gui_grounding.resources_servers.gui_grounding.port=12020
# Then GRPO in the training venv (runs.md Sat 17:13)
(source /home/ubuntu/serving.env && source /home/ubuntu/venvs/train/bin/activate && \
  python -m training.grpo --init-adapter /home/ubuntu/ckpt/pilot-qwen3.5-2mp-raw/adapter \
  --pool /home/ubuntu/night_smoke/pool.jsonl --out /home/ubuntu/ckpt/night_smoke/grpo \
  --verify-url http://127.0.0.1:12020/verify --max-steps 20)

# Overnight loop (profile → pool → GRPO → dev → keep/revert; artifacts in /home/ubuntu/night)
tmux new-session -d -s night -n loop
tmux send-keys -t night:loop "training/night_loop.sh --init-adapter /home/ubuntu/ckpt/sft-<winner>/adapter \
  --train-file /home/ubuntu/data/os_atlas_20k/<winner's train file>.jsonl" Enter

# Any two rollout files
gym eval profile --inputs <stem>_materialized_inputs.jsonl --rollouts <stem>.jsonl
gym eval compare --baseline <a>.jsonl --candidates <b>.jsonl --output-dir <dir> --report-format both
```

`web_agent` runs **on Patrick's Mac**, from the repo root: official BF16 Lightning on `:8000`
(vllm-metal, `docs/local-mac.md`), the sandbox on `:8090` (the Saturday acceptance run used
`AGENT_SANDBOX_URL=http://127.0.0.1:8091`), and the box's VLM on `:8001` through the tunnel.

```bash
# web_agent: tests, 12-task rows (not committed), base vs grounder × 3, tables
.venv-gym/bin/gym env test --resources-server web_agent
.venv-gym/bin/python resources_servers/web_agent/make_tasks.py --tasks sandbox/tasks.json --summary-model base \
  --out resources_servers/web_agent/data/tasks_all.jsonl
TASKS=resources_servers/web_agent/data/tasks_all.jsonl REPEATS=3 scripts/agent_eval.sh
.venv-gym/bin/python scripts/agent_metrics.py --no-runs --rollouts results/agent_base.jsonl results/agent_grounder.jsonl
```

Then run BLADE (`.agents/skills/nemo-gym-blade-analysis`) on each comparable pair: base vs SFT on
dev, each kept round vs the previous one, and `agent_base` vs `agent_grounder`.
