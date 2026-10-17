# B-night: automatic Gym improvement loop (overnight)

**Owner:** Kyle (stream B). **Runs:** ~8pm Sat → done whenever 3 rounds (or a stop rule) finish. **Tonight's flags (decided Sat 19:20, so a slow round isn't cut short):** `--last-round-start 04:15 --no-new-step 05:25 --hard-end 05:40`. The code defaults are 01:30 / 02:40 / 02:50. The orchestrator stops the instance as soon as `FINAL.json` exists, and by 6:00 at the latest.
**Read first:** `AGENTS.md`, `docs/research-notes.md` §3–4 (Gym 0.7.0 gotchas, TRL
notes), and `docs/gpu-box.md`.

## Goal

Keep improving the grounder overnight without anyone watching, using the NeMo Gym
loop:

> profile, then GRPO on "sometimes-pass" rows with the Gym `/verify` reward, then a
> dev check, then keep it or stop.

This is the core of the **"Best Model Improvement with NVIDIA NeMo Gym"** bounty, so
every round must leave Gym artifacts we can show: rollouts, profiles, compare
reports. The orchestrator thread (Patrick's Mac) runs **BLADE** on each round's
artifacts and may adjust a few pre-approved knobs before the next round.

## Before the loop (evening, ~5–8pm)

1. **Two full SFT runs, same config and seed:**
   - **template:** raw labels, all 18,652 train rows.
   - **lightning:** T3's `train_lightning.jsonl`, expected ~6pm at
     `/home/ubuntu/data/os_atlas_20k/`.
2. **Evaluate both on dev** (969 rows) with `gym eval run` through `gui_grounding`.
   Also evaluate on SSv2-clean (1,244 rows), but **for the results table only**.
3. **Pick the better one on dev.** That's the loop's round-0 checkpoint. Log both in
   `docs/runs.md`; this is the template-vs-Lightning ablation.
4. **Serve the round-0 adapter on `:8001`** as `grounder` (base + LoRA,
   gpu-box.md §3), so T2 can run the base-vs-tuned agent eval tonight.

## Loop (round r = 1, 2, 3; stop early per the rules below)

Work dir `/home/ubuntu/night/round_<r>/`. Each round:

1. **Read knobs.** Load `/home/ubuntu/night/knobs.json` (see below). If
   `/home/ubuntu/night/STOP` exists, stop cleanly.
2. **Profile:**
   - Take a fresh random **4,000-row train subset** (never dev, never SSv2; seed =
     r).
   - `gym eval run --num-repeats 8 --temperature 1.0` with the current checkpoint.
   - Then `gym eval profile`.
3. **Pick the GRPO pool:**
   - rows with **0 < pass rate < 1**;
   - apply the knobs: slice weights, `exclude_ids`;
   - cap at `pool_max` (default 2,000).
   - Save `pool.jsonl` plus pass-rate stats per source.
4. **GRPO:**
   - TRL `GRPOTrainer`, `vllm_mode="colocate"`, continuing from the current LoRA.
   - Reward = POST to the `gui_grounding` resources server `/verify` (point in box),
     optionally plus `format_reward` (knob, default 0.1 for a parseable point).
   - Group size `num_generations` (knob, default **6**; GuirlVG found 6 > 8),
     temperature 1.0, `max_completion_length` 32, thinking off.
   - **20-step smoke first:** go only if ≤ ~35 s/step (gate D2). Then up to
     `grpo_max_steps` (default 200) or `round_time_budget_min` (default 75).
5. **Dev eval:** `gym eval run` on dev, temperature 0, then `gym eval compare`
   against the previous round's dev rollouts.
6. **Keep or revert:**
   - If dev accuracy improves by **≥ +0.5 points**, keep the new adapter as current.
   - Otherwise revert to the previous adapter and **stop the loop** (don't keep
     training past the peak).
7. **Write** `round_<r>/summary.json`:
   `{round, start, end, checkpoint, pool_size, pool_by_source, grpo_steps, s_per_step, reward_mean_first/last, dev_acc_prev, dev_acc_new, kept, stop_reason?}`.
   Also touch `round_<r>/DONE`.
8. **Wait for BLADE, at most 10 min:** the orchestrator may update `knobs.json`
   based on its BLADE report. After 10 min, continue with the knobs as they are.

**Stop rules:** stop on whichever comes first:
- 3 rounds;
- **the clock** (tonight's flags): no round starts after 04:15, no new GRPO step after 05:25, and everything ends by 05:40;
- no dev improvement;
- the `STOP` file;
- GRPO smoke fails gate D2. In that case, run **one rejection-sampled SFT round**
  instead: SFT on the reward=1 completions from the profile rollouts. It's cheaper,
  and it's still Gym-driven.

**At the end:**
- Serve the **best kept adapter** on `:8001` as `grounder`, so the morning evals
  and the demo use it.
- Write `/home/ubuntu/night/FINAL.json` (`best_round`, `adapter path`, dev table).
- Stop the training processes. Leave only the `:8001` server.
- **Don't stop the instance.** The orchestrator copies the summaries and the best adapter to
  Patrick's Mac and runs `brev stop` as soon as `FINAL.json` exists (6:00 at the latest).
  It runs `brev start` at ~7:00 Sunday and re-serves `:8001` with `scripts/serve_vlm.sh`.

## `knobs.json` (the only thing BLADE may change)

```json
{"pool_max": 2000, "num_generations": 6, "grpo_max_steps": 200, "round_time_budget_min": 75,
 "format_reward": 0.1, "source_weights": {"os_atlas/linux": 1.0, "os_atlas/macos": 1.0,
 "os_atlas/androidworld": 1.0, "os_atlas/uibert": 1.0, "os_atlas/fineweb": 1.0},
 "exclude_ids": []}
```

- `source_weights` oversamples slices where BLADE finds "unreliable knowledge"
  (sometimes-pass).
- `exclude_ids` drops rows BLADE flags as data artifacts (box ≠ label).
- **Nothing else is automatic.** No changes to the eval sets, prompts, model or
  learning rate overnight.

## Guardrails

- **Checkpoints are chosen on dev only.** ScreenSpot-v2-clean is never used inside
  the loop. In the morning it's evaluated once on the final adapter.
- **Gym gotchas** (research-notes §3):
  - verify fields optional; annotate `verify` with `-> YourVerifyResponse`;
  - pass `++server_spinup_timeout_seconds=1800`;
  - one env per head port;
  - no keys in `env.yaml`.
- **GPU:**
  - Stop `:8002` (Qwen3-VL) and `:8003` (pilot) before the loop.
  - TRL colocate needs `vllm_gpu_memory_utilization` set explicitly; the default is
    0.3.
  - Keep `:8001` down during GRPO if memory is tight, and bring it back at the end.
- **Robustness:**
  - Run everything in tmux (`night` session).
  - Write logs to `/home/ubuntu/night/round_<r>/*.log`.
  - Any crash: revert to the last kept adapter, write `summary.json` with
    `stop_reason`, and go to the end.
- **Logging:** one row per round in `docs/runs.md` (append on the box; commit in the
  morning).

## Acceptance

- [ ] Round-0 SFT picked on dev; both SFT variants logged (the ablation).
- [ ] The loop runs at least one full round unattended, with Gym artifacts per round
      (profile, pool, rollouts, compare report, `summary.json`).
- [ ] `FINAL.json` exists, the best adapter is served on `:8001` as `grounder`, and
      training processes are stopped by the hard end (05:40 tonight).
