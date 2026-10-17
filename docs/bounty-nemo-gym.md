# Best Model Improvement with NVIDIA NeMo Gym: GUI Assist

> **Status:** draft of Sat 2026-10-17, ~18:00. Every number below is measured and logged in
> [runs.md](runs.md), unless marked **TBD**. TBD slots are filled from the overnight loop
> (Sat ~20:30 → Sun 03:00) and Sunday's final evals. Owners: Kyle (loop, training), Charlie
> (envs, eval, BLADE).

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
  GRPO using the Gym server's `/verify` as the reward, then a dev eval and `gym eval compare`
  decide whether to keep the new adapter. BLADE failure analysis can steer the next round,
  through a small set of pre-approved knobs.

So far, LoRA SFT on open OS-Atlas data, selected and measured in Gym, takes Qwen3.5-2B from
**76.4% to 85.7% on dev** and **86.3% to 89.1% on ScreenSpot-v2-clean**. Most of the held-out
gain is on icons (+6.7) and desktop (+5.5). The GRPO rounds are **TBD**.

## 2. Results

**Grounding** (accuracy = the click point lands inside the target box; temperature 0; all runs
through the `gui_grounding` env):

| Checkpoint | How it was made | Dev (969 rows) | SSv2-clean (1,244 rows) |
| --- | --- | --- | --- |
| Qwen3.5-2B base | — | 76.4% | 86.3% |
| Pilot SFT | 2,000 raw-label rows, 5 min | 81.3% (+5.0) | — |
| **Template SFT** (round 0) | All 18,652 train rows, raw labels, 1 epoch, 54 min | **85.7% (+9.3)** | **89.1% (+2.8)** |
| Lightning-instruction SFT | Same config, ~60% of instructions rewritten by Lightning (T3) | TBD | TBD |
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
| `base` | TBD | TBD | TBD | Sunday, `agent-v4` |
| `grounder` (final) | TBD | TBD | TBD | Sunday, `agent-v4` |

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

**Fallback that is still Gym-driven.** If GRPO fails the gate, the loop runs one
rejection-sampled SFT round instead. It trains on the reward=1 replies from the round's own
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

**Overnight rounds (TBD):**

| Round | Pool (by source) | Method | Steps | s/step | Reward first → last | Dev before → after | Kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | TBD | GRPO | TBD | TBD | TBD | 85.7 → TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Each round leaves Gym artifacts in `/home/ubuntu/night/round_<r>/`: profile rollouts and
`profile` output, `pool.jsonl` with pass-rate stats per source, dev rollouts, the
`gym eval compare` report and `summary.json`. A sample is committed under TBD.

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
- **The eval path is deterministic.** Re-running the pilot adapter's dev eval through the
  night loop's own code path reproduced 81.32% exactly.
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

- **Run-aborting errors:** an exception in `seed_session` or `verify()`, or a 422 on verify,
  aborts the whole `gym eval run`. Our verify fields are all optional, and failures return
  `reward=0, mask_sample=True` with a `failure_reason`.
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

## 8. Limitations

- **Small eval sets.** On 969 dev rows, a single accuracy has a standard error of about 1.1
  points, and the loop's +0.5 keep threshold is below that. Both sides are scored on the same
  rows, which makes the comparison paired, but the keep rule is a plain accuracy delta. **TBD:** a
  paired McNemar test on the final base-vs-tuned pair.
- **The SFT itself is not Gym-driven.** It's plain TRL on OS-Atlas, measured and selected in
  Gym. The Gym-driven training is the GRPO loop (or its rejection-sampled fallback), and the
  final table separates the two.
- **One sandbox app for the agent eval**, with prompts tuned on its traces. Treat absolute agent
  success as in-domain. The base-vs-tuned difference is the comparable number.
- **One H200 for the event** (~23 GPU-hours), so each GRPO round gets ≤ 75 minutes.

## 9. Reproduce

On the GPU box, from the repo root (stack pinned in [gpu-box.md](gpu-box.md); every run is logged
in [runs.md](runs.md)).

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

# Round-0 SFT (template variant), then dev + SSv2-clean through Gym
training/sft_full.sh template /home/ubuntu/data/os_atlas_20k/manifest.jsonl
python -m training.gymctl eval --adapter /home/ubuntu/ckpt/sft-template/adapter --name sft-template --ssv2

# The overnight loop: profile -> GRPO (Gym /verify reward) -> dev eval + gym eval compare -> keep/revert
training/night_loop.sh --init-adapter /home/ubuntu/ckpt/sft-<winner>/adapter \
  --train-file /home/ubuntu/data/os_atlas_20k/<winner's train file>.jsonl

# Agent: base vs grounder in web_agent, 3 repeats, then gym eval compare (Mac + tunnel to :8001)
TASKS=resources_servers/web_agent/data/tasks_all.jsonl REPEATS=3 scripts/agent_eval.sh
```

**Published (TBD links, public Sunday morning):**

- the adapter and merged model: `patrickballou/gui-assist-grounder`;
- the training slice manifest, the Lightning rewrites and any rejection-sampled rows, each with
  the command and config that made it: `patrickballou/gui-assist-data`.
