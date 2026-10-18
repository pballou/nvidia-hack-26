# Best Model Improvement with NVIDIA NeMo Gym: GUI Assist

> **Status:** draft of Sat 2026-10-17, ~18:00. Every number below is measured and logged in
> [runs.md](runs.md), unless marked **TBD**. TBD slots are filled from the overnight loop
> (Sat ~20:30 → hard end Sun 05:40 tonight, extended from the 02:50 code default) and Sunday's final
> evals. Owners: Kyle (loop, training), Charlie (envs, eval, BLADE).

## 1. Summary

We improve a 2B vision-language model at **GUI click grounding**: given a screenshot and an
instruction, it returns a click point. That model is the hands of an accessibility agent:
Nemotron 3.5 Lightning plans, and the grounder clicks. NeMo Gym is where the model is measured,
where checkpoints are chosen, and where the reward signal for training comes from:

- **Two Gym environments, kept in this repo in Gym's layout:**
  - `gui_grounding`: reward 1 if the point lands in the target box;
  - `web_agent`: the full agent on a sandbox web app, rewarded from the app's own state.
- **Every number below comes from `gym eval run`** through those environments: baselines,
  checkpoint selection on a dev split, and the held-out test.
- **An unattended overnight loop improves the model through Gym.** Each round runs
  `gym eval profile` to find the rows the model sometimes gets right, trains on them with TRL
  GRPO using the Gym server's `/verify` as the reward, then a dev eval decides whether to keep
  the new adapter (it must gain ≥ +0.5 points). `gym eval compare` against the previous round is
  saved next to it as the per-task flip report. BLADE failure analysis can steer the next round,
  through a small set of pre-approved knobs.

So far, LoRA SFT on open OS-Atlas data, selected and measured in Gym, takes Qwen3.5-2B from
**76.4% to 85.7% on dev** and **86.3% to 89.1% on ScreenSpot-v2-clean**. Most of the held-out
gain is on icons (+6.7) and desktop (+5.5). The GRPO rounds are **TBD**. Two one-round
rehearsals of the loop ran end to end on Saturday: the GRPO path was kept (dev 86.69%) and a
forced fallback round was reverted (§4). They profiled only 400 and 300 rows (4,000 per round
overnight), so they test the loop, not the gain.

## 2. Results

**Grounding** (accuracy = the click point lands inside the target box; temperature 0; all runs
through the `gui_grounding` env):

| Checkpoint | How it was made | Dev (969 rows) | SSv2-clean (1,244 rows) |
| --- | --- | --- | --- |
| Qwen3.5-2B base | — | 76.4% | 86.3% |
| Pilot SFT | 2,000 raw-label rows, 5 min | 81.3% (+5.0) | — |
| **Template SFT** (round-0 candidate) | All 18,652 train rows, raw labels, 1 epoch, 54 min | **85.7% (+9.3)** | **89.1% (+2.8)** |
| Lightning-instruction SFT | Same config and seed on T3's 18,299 rows (353 fewer than template); 8,620 instructions rewritten by Lightning | TBD | TBD |
| GRPO round 1 | Profile → pool → GRPO with the Gym `/verify` reward | TBD | — |
| GRPO round 2 / 3 | Same, if the previous round was kept | TBD | — |
| **Final** (best kept on dev) | | TBD | TBD (evaluated once) |

SSv2-clean is only evaluated for the table. Checkpoints are never chosen on it (§5).

**Where the SFT gain is** (base → template SFT):

| Split | Base | Template SFT | Δ |
| --- | --- | --- | --- |
| Dev: Linux | 68.8 | 86.9 | +18.1 |
| Dev: macOS | 73.4 | 91.1 | +17.7 |
| Dev: AndroidWorld | 70.5 | 81.3 | +10.8 |
| Dev: UIBert | 78.4 | 85.1 | +6.7 |
| Dev: FineWeb | 82.9 | 84.5 | +1.6 |
| SSv2-clean: mobile | 91.4 | 92.2 | +0.8 |
| SSv2-clean: desktop | 84.0 | 89.5 | +5.5 |
| SSv2-clean: web | 81.9 | 85.1 | +3.2 |
| SSv2-clean: text | 92.5 | 92.3 | −0.2 |
| SSv2-clean: icon | 78.1 | 84.8 | +6.7 |

- The desktop gain holds with the 28 contaminated desktop rows removed (§5), so it isn't
  same-machine leakage.
- Text grounding was already strong and stayed flat, and icons are where the headroom was.
- Fine-tuned answers are about 2× shorter: 37.6 → 16.7 output tokens (pilot), so each grounding
  call is faster.

**Agent** (`web_agent` env, 12 sandbox tasks × 3 repeats, the same Lightning, rows, prompts and
tools; only the grounder's served name changes):

| Grounder | Verified success | Claimed success | Invalid tool calls | Status |
| --- | --- | --- | --- | --- |
| `base` | 18/36 (50%) | 24/36 (8 false) | 10/259 | Sat 18:45, `agent-v4`, reportable |
| `grounder` = sft-template | **22/36 (61%)** | 25/36 (4 false) | 6/234 | Sat 19:05, `agent-v4`, reportable |
| `grounder` (final overnight adapter, if it beats sft-template on dev) | TBD | TBD | TBD | Sunday |

- **Strict-check caveat:** with a strict `mark-all-read` check, tuned is 20/36 (+2), as the
  current check doesn't guard the bell notifications ([BLADE](blade/agent_base_vs_sft.md)). The
  check stays unchanged until after Sunday's grounder rerun, so that rerun is comparable.
- **`gym eval compare`:** `mean/reward` +0.111, 95% CI [−0.06, +0.28], so suggestive but not
  significant at n=36. Failed actions per episode −0.28, CI [−0.52, −0.04], significant
  (10 → 0). No-change actions 83 → 45, and Lightning's done claims matched the sandbox 31/36 vs
  26/36. Flips: 3 fail→pass, 1 pass→fail.
- **Per task (verified/3):** text-size-large 0→2, sms-billing-off 0→2, message-support-meter 1→2,
  high-contrast 2→1. Canvas and scroll tasks stay 0/3 in both (BLADE: TBD).
- **Setup:** official BF16 Lightning (thinking off), sandbox `68c7671`, `base` screen descriptions
  in both conditions, 0 masked ([runs.md](runs.md) Sat 18:45).

A Saturday read (n=1, 4 easy tasks, not final) had the pilot adapter at 4/4 and base at 3/4. The
task base missed was a pure grounding miss: Lightning correctly asked for the "L" text-size option
six times, and every base click landed on "M". Details: [bounty-lightning.md](bounty-lightning.md)
§5.

## 3. The environments

### 3a. `gui_grounding` (`resources_servers/gui_grounding/`)

- **Task:** screenshot + instruction → `{"point_2d": [x, y]}` on Qwen's 0–1000 grid.
- **Reward:** 1 if the point, converted to pixels, falls inside the target's pixel box
  (inclusive), else 0.
- **Origin:** adapted from Gym v0.7.0's `resources_servers/gui_coordinate` (provenance in
  `app.py`). We replaced its distance reward with a binary point-in-box reward, because that's
  the ScreenSpot metric we report, and added per-slice metrics.
- **One interface module:** prompts, output parsing and coordinate conversion live only in
  `common/grounding.py`. The Gym env, the TRL SFT and GRPO code and the agent's `click` tool all
  import it, and vLLM serving uses the same 2 MP image cap, so training and eval can't drift
  apart.
- **Metrics:** sliced by `platform`, `data_type` (text/icon), `platform_data_type` and
  `source`, via Gym's `compute_subset_metrics`.
- **Rows:** `data_prep/gym_rows.py` builds three sets, all with `file://` images (no base64):
  - OS-Atlas dev, for checkpoint selection;
  - ScreenSpot-v2-clean, the held-out test;
  - GRPO pools.
- **Three jobs, one endpoint:** the same `/verify` scores dev, SSv2-clean and GRPO completions.
  The training reward is exactly the eval metric.

### 3b. `web_agent` (`resources_servers/web_agent/`)

- **Episode:** Lightning gets a user request ("Make the text bigger, I can't read this.") and a
  text description of the screen. It acts through five tools; `click(description)` goes to the
  grounder, and Playwright clicks the pixel.
- **Reward:** the task's own check on the sandbox app's state (`sandbox/checks.py`). The agent
  never sees that state.
- **Base vs tuned from one model:** `base` and `grounder` are the same vLLM process, with the
  LoRA adapter switched per request. This keeps the system at 32.6B parameters by model-card
  counts (cap 33B).
- **Same code as the demo:** every tool route calls the interactive agent's own
  `AgentSession.execute`, so Gym measures the system we demo, not a copy of it.
- **Design details:** [`resources_servers/web_agent/README.md`](../resources_servers/web_agent/README.md)
  and [bounty-lightning.md](bounty-lightning.md) §5.

## 4. How Gym drives the improvement

```text
           ┌──────────────── knobs.json (BLADE may edit; nothing else changes overnight)
           ▼
 1. profile ── gym eval run --num-repeats 8 --temperature 1.0 on 4,000 fresh train rows
           │   then gym eval profile → pass rate per row
           ▼
 2. pool ───── rows with 0 < pass rate < 1, minus exclude_ids, weighted by source, ≤ 2,000
           ▼
 3. GRPO ───── TRL GRPOTrainer, vLLM colocated, G = 6, continuing the current LoRA
           │   reward = POST gui_grounding /verify (+ 0.1 for a parseable point)
           │   first 20 steps = gate D2 (≤ 35 s/step), else one rejection-sampled SFT round
           ▼
 4. dev ────── gym eval run on dev (temperature 0) + gym eval compare vs the previous round
           ▼
 5. keep if dev ≥ +0.5 points, else revert and stop → summary.json → wait ≤ 10 min for BLADE
```

Code: `training/night_loop.py` (the loop), `training/grpo.py` (TRL GRPO with the Gym reward),
`training/gymctl.py` (the vLLM and Gym calls). Spec: [tasks/B-night-loop.md](tasks/B-night-loop.md).

**Loop settings, from the code** (`training/night_loop.py` defaults unless noted):

- **Clock:** tonight's run uses extended limits, `--last-round-start 04:15 --no-new-step 05:25
  --hard-end 05:40`, at most 3 rounds, and the box is stopped as soon as `FINAL.json` exists (by
  06:00 at the latest). The code defaults are 01:30 / 02:40 / 02:50. Every step's timeout is capped
  at the hard end, so `finish()` (cleanup, serving the best adapter, `FINAL.json`) runs by then.
- **Keep gate:** dev must gain ≥ +0.5 points (`KEEP_MIN_GAIN`), otherwise the loop reverts to the
  last kept adapter and stops. A failed dev eval is retried once.
- **Reward:** `/verify` (weight 1) plus 0.1 × "the reply parses as a point" (`format_reward`, a
  knob clamped to [0, 1]; `format_ok` in `training/grpo.py`).
- **`gym eval compare` is a saved report, not the gate.** The loop writes it to
  `round_<r>/compare/` and carries on if it fails; only the dev accuracy delta decides.
- **Other stops:** 3 rounds, a `STOP` file, or any crash (revert to the last kept adapter,
  `stop_reason` recorded). A GRPO crash after a passed smoke first dev-evaluates its last
  checkpoint, which the normal keep rule can keep.

**Why profile first.** GRPO learns from differences between replies to the same prompt. A row
the model always gets right (or always gets wrong) gives every reply in its group the same
reward, so it contributes no gradient and only burns GPU time. `gym eval profile` over 8 sampled
replies per row finds the rows that do have signal. Those "sometimes-pass" rows are also the
"unreliable knowledge" category in BLADE's taxonomy.

**Why the Gym `/verify` as the reward.** We don't reimplement the reward in the trainer. TRL's
reward function POSTs each completion, wrapped as a Responses-API object, to the same resources
server that scores our evals. If the reward and the metric ever disagreed, that would be a bug in
one place, not two.

**Gate D2 smoke (measured Sat 17:13):**

- **Setup:** 20 GRPO steps from the pilot adapter on 200 random train rows, with the Gym server
  running alongside.
- **Speed:** **31.9 s/step** median, under the 35 s gate, while sharing the GPU with a full SFT
  run.
- **Reward calls:** 960 `/verify` calls, 0 failures. Training reward 0.75 → 0.79; that's 20
  steps on training rows, not a dev claim.
- **Rollout correctness:** vLLM-vs-trainer log-prob difference about 0.01–0.03, and importance
  ratio ≈ 1. So the rollouts the reward scores are the ones the trainer learns from.

**Fallback that is still Gym-driven.** If GRPO fails the gate (or crashes before finishing its
20-step smoke, or the pool has fewer than 32 rows), the loop runs one rejection-sampled SFT round
instead, then stops. It trains on the reward=1 replies from the round's own
`gym eval profile` rollouts, at most 2 distinct replies per row. Those replies are model-generated
training data, so they're published with the command and config that made them.

**What made TRL GRPO work on Qwen3.5-2B with images** (TRL 1.16.0 + vLLM 0.31.0, found by
reading the installed sources; details in `training/grpo.py`):

1. Colocated vLLM re-expanded image tokens that TRL had already expanded. We collapse them so
   the prompt ids match exactly, and assert that they do.
2. The processor must pad on the left, so Qwen3.5's image positions line up.
3. The colocated vLLM gets the same 2 MP image cap as the serving process.
4. The bf16 merge/unmerge weight sync drifted the base weights a little every step, so we restore
   the exact base weights after each unmerge.

**Loop rehearsals (Sat, `runs.md` 17:34 and 17:52):** one round each from the template SFT
adapter, with a small profile, knobs `pool_max 300` and `grpo_max_steps 30` (22 for rehearsal 2),
`:8001` untouched and no `runs.md` rows written by the loop.

| Rehearsal | Pool | Method | Steps | s/step | Reward first → last | Dev before → after | Kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1: GRPO path | 111 of 400 profiled rows (28% sometimes-pass); profile 400 × 8 took 49 s | GRPO | 30 | 19.6 (gate D2, GPU not shared) | 0.69 → 0.85 | 85.86 → **86.69** (+0.83) | Yes |
| 2: forced fallback (`--gate-s-per-step 1`) | 300 profiled rows | Gate D2 failed as forced (19.8 s/step > 1) → rejection-sampled SFT on 541 reward=1 replies (279 tasks), lr 5e-5 | 17 | — | — | 85.86 → 85.96 (+0.10) | No: reverted, `stop_reason` recorded |

- **Rehearsal 1's baseline:** [results.md](results.md) measures the gain against the 85.86 the
  same adapter scored in the same harness. The `runs.md` row logs it against the earlier 85.66
  (+1.03). Either way it clears the +0.5 gate.
- **Artifacts:** rehearsal 1 logged the full set (profile, pool, compare report, `summary.json`,
  `DONE`, `FINAL.json`). Rehearsal 2 logged its `stop_reason` and stopped everything but `:8001`.
- **Not a result:** the pool was 111 rows, and the same adapter's dev score moved 0.2 points
  between launches (§5), so gains near the gate are close to the noise.

**Overnight rounds (TBD):**

| Round | Pool (by source) | Method | Steps | s/step | Reward first → last | Dev before → after | Kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | TBD | GRPO | TBD | TBD | TBD | round-0 re-eval (TBD) → TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Each round leaves Gym artifacts in `/home/ubuntu/night/round_<r>/`: profile rollouts and
`profile` output, `pool.jsonl` and its pass-rate stats per source (`pool_stats.json`),
`knobs_used.json`, `grpo_command.txt` and `grpo.log`, dev rollouts, the `gym eval compare` report
and `summary.json`. Adapters go to `/home/ubuntu/ckpt/night/<launch>/round_<r>/`. A sample is
committed under TBD.

## 5. Measurement we can trust

- **Splits (decided before training):**
  - An OS-Atlas `dev` split, held out **by screenshot**, chooses checkpoints.
  - **ScreenSpot-v2-clean** is the held-out test, evaluated once on the final adapter.
  - The overnight loop never touches ScreenSpot. Its profile subsets are drawn from train only.
- **We found and removed a test-set leak.** SSv2's 28 newer desktop screenshots were captured
  on the same machines as OS-Atlas's Linux and macOS training data, 1–3 weeks later: same
  wallpaper, same desktop files.
  - At least 9 have a near-duplicate training screenshot, and some test targets are labeled in
    training (e.g. "adjust font color" vs the training element "Font Color", IoU 0.83).
  - A perceptual-hash dedup can't fix this, because the same app layouts recur across many
    training screenshots. So we exclude those 28 test rows instead, and say so in every number:
    SSv2-clean = 1,272 − 28 = 1,244 rows (`benchmarks/screenspot_v2/excluded_rows.json`).
  - We haven't seen this documented elsewhere.
- **We also found burned-in labels in the training data.** In one capture window (13 Linux
  screenshots, 98 rows), OS-Atlas's own capture tool had rendered accessibility-role labels as red
  text into the image. Training on them would teach the model to read printed labels it never
  sees in real use. Those rows are dropped.
- **Other cleaning:**
  - training images are deduplicated against SSv2 by perceptual hash;
  - invalid and zero-area boxes are dropped;
  - so are instructions that appear twice on one screenshot;
  - Wave-UI's ScreenSpot rows and GroundUI-18K are never used.
- **The eval path reproduces, but dev is not bit-exact.** Re-running the pilot adapter's dev
  eval through the night loop's own code path reproduced 81.32% exactly. The template adapter,
  though, scored 85.66% and 85.86% in two launches (vLLM batching at temperature 0), so about
  0.2 points of run-to-run noise.
- **Infra failures are masked, not scored.** A verify error (`gui_grounding`), or a VLM
  transport error, browser crash or tool timeout (`web_agent`), returns `mask_sample=True` with a
  reason, instead of a 0 reward that would look like a model failure. An unparseable reply is
  still scored 0, because that is a model failure.
- **Fair agent comparison.** Base and tuned get identical rows, start screens, prompts, tools and
  Lightning; preflight refuses non-official Lightning checkpoints for reported runs.

## 6. Failure analysis (BLADE) and what we changed

**TBD (Charlie): BLADE on base vs template SFT** (`gui_grounding` dev and SSv2-clean rollouts),
and on each overnight round's dev pair. Planned table: root-cause category → example rollouts →
the action it drove (a `knobs.json` change, a data change or a prompt change).

Read from the slices so far, before BLADE:

- **SFT fixed most of the desktop gap.** Linux and macOS dev rose about 18 points. Base Qwen3.5
  was weakest there (68.8 and 73.4), and our data mix gives desktop ~35%.
- **Icons are the remaining headroom.** On SSv2-clean, icon is 84.8 vs text 92.3, so that's
  where we expect sometimes-pass rows and GRPO gains.
- **Web gained least on dev** (FineWeb +1.6). If BLADE finds the FineWeb sometimes-pass rows
  are learnable rather than label noise, we raise its `source_weights` knob.
- **Agent failure clusters** from the Saturday reads (base grounder): scroll-dependent targets,
  a canvas chart, dense toggle grids. The grounder's part of those will be checked against the
  Sunday base-vs-tuned `web_agent` pair.

## 7. Notes for the Gym team (0.7.0)

Things we hit, each handled in our code:

- **Run-aborting errors:** an exception in `seed_session` or `verify()`, a 422 on verify, or a
  failed model call aborts the whole `gym eval run`. `web_agent`'s request fields are all
  optional. `gui_grounding`'s row fields (all but `data_type`) are required, which is safe only
  because every row comes from `gym_rows.py`. Both return `reward=0, mask_sample=True` with a `failure_reason` on
  failure. The night-loop profile and `agent_eval.sh` pass `route_failures_to_sidecar`, so a
  failed rollout goes to a sidecar file instead of aborting the run. A tool exception doesn't
  abort: its 500 body goes back to the model.
- **Silently dropped fields:**
  - row fields not declared on the `VerifyRequest` subclass are silently dropped;
  - so are rollout fields unless `verify` is annotated `-> OurVerifyResponse`;
  - and config keys a server doesn't declare. `web_agent` refuses to start on unknown keys, so
    a typo can't quietly change which grounder is measured.
- **Start observation:** `simple_agent`'s `/run` drops the `seed_session` response body, so the
  start observation is baked into each `web_agent` row, with a screenshot hash that
  `seed_session` checks for staleness.
- **`simple_agent` episode end:** it ends only on a reply without a tool call (not on a `done`
  tool), and it runs every parallel call it receives.
- **Ops:** a cold server venv build takes ~8–9 min (`++server_spinup_timeout_seconds=1800`), and
  the head server defaults to `:11000`, so a second env on the box needs `++head_server.port`.

More we hit, one row each (sources: [research-notes.md](research-notes.md) §3, the env code):

| Gotcha | What happens | What we did |
| --- | --- | --- |
| `mask_sample=True` | The rollout leaves every quality metric and shows only in `coverage/*` | Infra failures mask the rollout whatever its reward; model failures (no parsable point) stay scored |
| `max_steps` counts model calls, checked after that turn's tools | `tool_choice: required` would run every task to `max_steps` | `max_steps: 12` = the interactive budget; `make_tasks.py --check` rejects rows that set `tool_choice` |
| Any tool name the model emits is POSTed to `/{name}`; bad-JSON arguments never reach the server | Hallucinated `seed_session` / `close_session` / `verify` hit lifecycle routes; unknown names get a 404 | A catch-all `POST /{tool_name}` registered last; repeat seeds and empty closes do nothing; `scripts/agent_metrics.py` re-validates every call from `response.output` |
| Gym's HTTP client has no timeout | A hung tool call hangs the rollout | Each tool call is bounded by `tool_timeout_s` (240 s); a timeout returns `ok:false` and masks the rollout |
| Legacy `/run` never calls `close_session`; `num_workers > 1` splits state | Leaked browser contexts; tool calls land on a worker without the session | Release in `verify`; each seed closes episodes older than 1 h; `num_workers: 1` enforced at startup |
| Venv build and imports | A root `pyproject.toml` makes every server editable; `gym env start` doesn't put the repo root on PYTHONPATH; `skip_venv_if_present` ignores requirements changes | No root `pyproject.toml` or `resources_servers/__init__.py`; each `app.py` puts the repo root on `sys.path`; `REINSTALL=1` in `agent_eval.sh` |
| `env.yaml` merges after every config | Its `policy_*` overrides any `${policy_*}`; the merged config is served at `:11000/global_config_dict_yaml` | `web_agent`'s planner is `lightning_policy_model` with literal URLs and `${oc.env:NVIDIA_API_KEY}`; no keys in `env.yaml` or on the CLI |
| `--model` is ignored with `--no-serve`; `/readyz` waits for the policy | The policy can't be switched per run; the env can't restart while the eval server is down for GRPO | `gymctl` sets the policy at `env start` (`++policy_model_name=grounder`), passes `++model_endpoint_readiness_timeout_seconds=0`, and uses its own ports (head `:11010`, `/verify` `:12010`; the one-off eval CLI `:11011` / `:12011`) |

## 8. Limitations

- **Small eval sets.** On 969 dev rows, a single accuracy has a standard error of about 1.1
  points, and the loop's +0.5 keep threshold is below that. Both sides are scored on the same
  rows, which makes the comparison paired, but the keep rule is a plain accuracy delta. The same
  adapter also moved 0.2 points between two launches (85.66 vs 85.86, §5). **TBD:** a paired
  McNemar test on the final base-vs-tuned pair.
- **The SFT itself is not Gym-driven.** It's plain TRL on OS-Atlas, measured and selected in
  Gym. The Gym-driven training is the GRPO loop (or its rejection-sampled fallback), and the
  final table separates the two.
- **One sandbox app for the agent eval**, with prompts tuned on its traces. Treat absolute agent
  success as in-domain. The base-vs-tuned difference is the comparable number.
- **One H200 for the event** (~23 GPU-hours), so each GRPO round gets ≤ 75 minutes.

## 9. Reproduce

Grounding, SFT, GRPO and the night loop run **on the Brev box**, from the repo root (stack pinned
in [gpu-box.md](gpu-box.md); every run is logged in [runs.md](runs.md)). Gym commands use
`.venv-gym`. `training.grpo` needs the training venv (`/home/ubuntu/venvs/train`,
[gpu-box.md](gpu-box.md) §5); `sft_full.sh` and `night_loop.sh` activate it themselves.

```bash
export NEMO_GYM_EXTRA_ROOTS=$PWD && source .venv-gym/bin/activate

# Gym rows (images stay on the box as file:// paths)
python -m data_prep.gym_rows ssv2 --out /home/ubuntu/data/gym/ssv2_clean.jsonl
python -m data_prep.gym_rows manifest --manifest /home/ubuntu/data/os_atlas_20k/manifest.jsonl \
  --split dev --out /home/ubuntu/data/gym/os_atlas_dev.jsonl

# Baseline: served name `base` on :8001 (scripts/serve_vlm.sh qwen3.5 8001)
gym env start --resources-server gui_grounding --model-type vllm_model ++server_spinup_timeout_seconds=1800
gym eval run --no-serve --agent gui_grounding_simple_agent \
  --input /home/ubuntu/data/gym/os_atlas_dev.jsonl --output results/dev_base_qwen35.jsonl --concurrency 64
gym eval run --no-serve --agent gui_grounding_simple_agent \
  --input /home/ubuntu/data/gym/ssv2_clean.jsonl --output results/ssv2_clean_base_qwen35.jsonl --concurrency 64
gym env test --resources-server gui_grounding

# Round-0 SFT (template variant), then dev + SSv2-clean through Gym (runs.md Sat 17:31)
training/sft_full.sh template /home/ubuntu/data/os_atlas_20k/manifest.jsonl
python -m training.gymctl eval --adapter /home/ubuntu/ckpt/sft-template/adapter --name sft-template --ssv2 --gpu-util 0.35

# Gate D2 smoke (runs.md Sat 17:13). First a throwaway gui_grounding env with /verify on :12020, in its
# own tmux pane. runs.md doesn't record that env's exact command; this uses gymctl.GymEnv's port overrides.
gym env start --resources-server gui_grounding --model-type vllm_model ++server_spinup_timeout_seconds=1800 \
  ++head_server.port=11020 ++gui_grounding.resources_servers.gui_grounding.port=12020
(source /home/ubuntu/serving.env && source /home/ubuntu/venvs/train/bin/activate && \
  python -m training.grpo --init-adapter /home/ubuntu/ckpt/pilot-qwen3.5-2mp-raw/adapter \
  --pool /home/ubuntu/night_smoke/pool.jsonl --out /home/ubuntu/ckpt/night_smoke/grpo \
  --verify-url http://127.0.0.1:12020/verify --max-steps 20)

# Loop rehearsal 1 (runs.md Sat 17:34, code 937b4fb; /home/ubuntu/night_rehearsal/knobs.json:
# {"pool_max": 300, "grpo_max_steps": 30}). Rehearsal 2: --profile-rows 300 --gate-s-per-step 1, grpo_max_steps 22.
training/night_loop.sh --init-adapter /home/ubuntu/ckpt/sft-template/adapter \
  --train-file /home/ubuntu/data/os_atlas_20k/manifest.jsonl --night-dir /home/ubuntu/night_rehearsal \
  --ckpt-dir /home/ubuntu/ckpt/night_rehearsal --max-rounds 1 --profile-rows 400 --blade-wait-min 0.5 \
  --final-port 0 --stop-ports --runs-md none

# The overnight loop, in tmux: profile -> GRPO (Gym /verify reward) -> dev eval + gym eval compare -> keep/revert
tmux new-session -d -s night -n loop
tmux send-keys -t night:loop "training/night_loop.sh --init-adapter /home/ubuntu/ckpt/sft-<winner>/adapter \
  --train-file /home/ubuntu/data/os_atlas_20k/<winner's train file>.jsonl" Enter

# Any two rollout files
gym eval profile --inputs <stem>_materialized_inputs.jsonl --rollouts <stem>.jsonl
gym eval compare --baseline <a>.jsonl --candidates <b>.jsonl --output-dir <dir> --report-format both
```

`web_agent` runs **on Patrick's Mac**, from the repo root: official BF16 Lightning on `:8000`
(vllm-metal, [local-mac.md](local-mac.md)), the sandbox on `:8090` (`configs/agent.yaml`; the
Saturday acceptance run used `AGENT_SANDBOX_URL=http://127.0.0.1:8091`), and the box's VLM on
`:8001` through the tunnel.

```bash
# Agent: tests, the 12-task rows (gitignored, so build them first), base vs grounder in web_agent,
# 3 repeats, then gym eval compare; then the success / validity / latency tables
.venv-gym/bin/gym env test --resources-server web_agent
.venv-gym/bin/python resources_servers/web_agent/make_tasks.py --tasks sandbox/tasks.json --summary-model base \
  --out resources_servers/web_agent/data/tasks_all.jsonl
TASKS=resources_servers/web_agent/data/tasks_all.jsonl REPEATS=3 scripts/agent_eval.sh
.venv-gym/bin/python scripts/agent_metrics.py --no-runs --rollouts results/agent_base.jsonl results/agent_grounder.jsonl
```

`agent_eval.sh` runs `gym env start` → `gym eval run` → `gym eval profile` once per grounder, in
sequence, then `gym eval compare` base vs grounder. It refuses placeholder start screens and any
local Lightning other than the official BF16 checkpoint (`FORCE=1` allows both for plumbing,
recorded as `reportable=no`). Then run BLADE (`.agents/skills/nemo-gym-blade-analysis`) on each
comparable pair: base vs SFT on dev, each kept round vs the previous one, and `agent_base` vs
`agent_grounder`.

**Published (TBD links, public Sunday morning):**

- the adapter and merged model: `patrickballou/gui-assist-grounder`;
- the training slice manifest, the Lightning rewrites and any rejection-sampled rows, each with
  the command and config that made it: `patrickballou/gui-assist-data`.
