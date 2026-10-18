# Best Model Improvement with NVIDIA NeMo Gym: GUI Assist

> **Status:** Sun 2026-10-18, 00:30. The overnight loop ran 19:38 → 22:02 and its results are
> filled in (§2, §4). Every number below is measured and logged in [runs.md](runs.md) or the
> [BLADE reports](blade/). Owners: Kyle (loop, training), Charlie (envs, eval, BLADE).

## 1. Summary

We improve a 2B vision-language model at **GUI click grounding**: given a screenshot and an
instruction, it returns a click point. That model is the hands of an accessibility agent:
Nemotron 3.5 Lightning plans, and the grounder clicks. NeMo Gym is where the model is measured,
where checkpoints are chosen, and where the reward signal for training comes from:

- **Two Gym environments, kept in this repo in Gym's layout:**
  - `gui_grounding`: reward 1 if the point lands in the target box;
  - `web_agent`: the full agent on a sandbox web app, rewarded from the app's own state.
- **Every accuracy and success rate below comes from `gym eval run`** through those environments: baselines,
  checkpoint selection on a dev split, and the held-out test.
- **An unattended overnight loop improves the model through Gym.** Each round runs
  `gym eval profile` to find the rows the model sometimes gets right, trains on them with TRL
  GRPO using the Gym server's `/verify` as the reward, then a dev eval decides whether to keep
  the new adapter (it must gain ≥ +0.5 points). `gym eval compare` against the previous round is
  saved next to it as the per-task flip report. BLADE failure analysis can steer the next round,
  through a small set of pre-approved knobs.

LoRA SFT on open OS-Atlas data, selected and measured in Gym, takes Qwen3.5-2B from **76.4% to
86.27% on dev**. The chosen variant used 8,620 instructions rewritten by Lightning, picked on dev.
The Gym loop then adds one kept GRPO round: **dev 87.0%**.

On held-out **ScreenSpot-v2-clean**, never used to choose a checkpoint, the final adapter scores
**90.35%**. A second independent T=0 run of the same adapter gave 90.27%, one row apart; we report
the run that the paired compares use. Against the baselines:
- **+4.10 points over base** (86.25%, 1,073 of 1,244 rows; 86.3 rounded), with 89 tasks fixed
  and 38 broken on the same rows (McNemar exact p ≈ 7 × 10⁻⁶). Icons rose most (78.15 → 85.37,
  61 fixed vs 22 broken); mobile text is the only slice that dropped (97.59 → 96.90, 3 vs 5,
  p = 0.73).
- **+2.25 over the loop's own starting adapter** (sft-lightning, the dev pick, 88.10%), with 34
  rows fixed and 6 broken on the same 1,244 rows (McNemar exact p ≈ 8 × 10⁻⁶). Every split rose,
  most on icons (+3.7, 22 fixed vs 2 broken, the one split significant on its own after a Holm
  correction) and web (+2.7).
- **+1.29 over template SFT** (89.07%), the stronger SFT on SSv2-clean but not the dev pick: 50
  fixed, 34 broken, p = 0.10, not significant.

Round 2 was flat on dev (−0.10, one row) and was reverted, so the keep rule stopped the loop
there. Base → SFT is plain TRL measured in Gym; SFT → final is the Gym-driven part (§8).

## 2. Results

**Grounding** (accuracy = the click point lands inside the target box; temperature 0; all runs
through the `gui_grounding` env):

| Checkpoint | How it was made | Dev (969 rows) | SSv2-clean (1,244 rows) |
| --- | --- | --- | --- |
| Qwen3.5-2B base | — | 76.4% | 86.3% |
| Pilot SFT | 2,000 raw-label rows, 5 min | 81.3% (+5.0) | — |
| Template SFT (round-0 candidate) | All 18,652 train rows, raw labels, 1 epoch, 54 min | 85.7% (+9.3) | 89.1% (+2.8) |
| **Lightning-instruction SFT** (round 0, picked on dev) | Same config and seed on T3's 18,299 rows (353 fewer than template); 8,620 Lightning instructions (8,498 differ from the raw label); 49 min | **86.3% (+9.9)** | 88.1% (+1.8) |
| GRPO round 1: **kept** | 4,000 profiled rows → 1,282-row pool → 170 GRPO steps with the Gym `/verify` reward | **87.0% (+0.72 vs round 0; 7 net rows)** | — |
| GRPO round 2: reverted | 4,000 profiled rows → 242-row pool (BLADE knobs) → 167 steps | 86.9% (−0.10 vs round 1; 1 row) | — |
| **Final** (round 1, best kept on dev) | Served as `grounder`; merged copy passed Gate G on the Mac | **87.0% (+10.6)** | **90.35% (+4.1)**; duplicate run 90.27 |

SSv2-clean is only evaluated for the table. Checkpoints are never chosen on it (§5). The two
round-0 candidates differ by +0.41 on dev (86.27 vs template's 85.86 in the same harness). That is
about 2× the ~0.2-point run-to-run noise but well inside the ~1.1-point sampling SE (§8).
SSv2-clean happened to favor template (89.07 vs 88.10). The pick followed the dev rule. Round 3
didn't run because round 2 wasn't kept. The dev number for the final adapter is a selection
number: it was the best on dev across rounds, and re-runs gave 87.00 and 86.89.

**Base → final, by slice, with significance** (SSv2-clean, same 1,244 rows; exact McNemar on
fixed vs broken rows; Holm-adjusted over the 11 slice tests):

| Slice (rows) | Base → final | Both pass / both fail | Fixed / broken | McNemar p | Holm p |
| --- | --- | --- | --- | --- | --- |
| **All (1,244)** | 86.25 → 90.35 | 1,035 / 82 | **89 / 38** | **7 × 10⁻⁶** | — |
| Mobile (501) | 91.42 → 93.21 | 446 / 22 | 21 / 12 | 0.16 | 0.49 |
| Desktop (306) | 83.99 → 88.89 | 251 / 28 | 21 / 6 | 0.006 | **0.047** |
| Web (437) | 81.92 → 88.10 | 338 / 32 | 47 / 20 | 0.001 | **0.013** |
| Text (704) | 92.47 → 94.18 | 635 / 25 | 28 / 16 | 0.10 | 0.38 |
| Icon (540) | 78.15 → 85.37 | 400 / 57 | 61 / 22 | 2 × 10⁻⁵ | **0.0002** |
| Mobile text (290) | 97.59 → 96.90 | 278 / 4 | 3 / 5 | 0.73 | 1 |
| Mobile icon (211) | 82.94 → 88.15 | 168 / 18 | 18 / 7 | 0.043 | 0.22 |
| Desktop text (180) | 90.56 → 91.67 | 159 / 11 | 6 / 4 | 0.75 | 1 |
| Desktop icon (126) | 74.60 → 84.92 | 92 / 17 | 15 / 2 | 0.002 | **0.021** |
| Web text (234) | 87.61 → 92.74 | 198 / 10 | 19 / 7 | 0.029 | 0.19 |
| Web icon (203) | 75.37 → 82.76 | 140 / 22 | 28 / 13 | 0.028 | 0.19 |

- **McNemar uses only the rows that changed.** Rows both models get right (or both get wrong)
  say nothing about which is better. If base and final were equally good, each changed row would
  be a coin flip between "fixed" and "broken". 89 vs 38 out of 127 would happen about 7 times in
  a million.
- **What holds up after correcting for 11 slice tests:** icons (p_Holm = 0.0002), web (0.013),
  desktop icons (0.021) and desktop (0.047). Mobile (p = 0.16) and text (0.10) aren't
  significant even before the correction. The mobile-icon, web-text and web-icon gains (raw p
  0.03–0.04) are suggestive on their own but don't survive it. Mobile text is the one slice that
  fell (3 fixed vs 5 broken, p = 0.73: noise).
- **The test treats each model's T=0 answers as fixed.** Our evals move about 0.2 points between
  launches (§5), which matters for close pairs, not for 89 vs 38.
- **Reproduce:** `python3 -I scripts/mcnemar_slices.py --report
  docs/eval_results/compare_ssv2_base_vs_final/compare_report.json --raw <SSv2 annotation JSONs @
  5efbb1f1b5>`. It maps each flip's `task_index` to its slice using `gym_rows.py`'s row order, and
  checks every slice against the report's own accuracies.

**Where the Gym loop's gain is** (SSv2-clean, same 1,244 rows; the loop started from sft-lightning):

| Split (rows) | Template SFT | Loop start (sft-lightning) | Final | Δ vs loop start (fixed / broken; p, Holm p) | Δ vs template (fixed / broken; p, Holm p) |
| --- | --- | --- | --- | --- | --- |
| All (1,244) | 89.07 | 88.10 | **90.35** | **+2.25** (34 / 6; p ≈ 8 × 10⁻⁶) | +1.29 (50 / 34; p = 0.10) |
| Mobile (501) | 92.22 | 91.22 | 93.21 | +2.00 (11 / 1; 0.006, 0.06) | +1.00 (15 / 10; 0.42, 1) |
| Desktop (306) | 89.54 | 86.93 | 88.89 | +1.96 (8 / 2; 0.11, 0.44) | −0.65 (11 / 13; 0.84, 1) |
| Web (437) | 85.13 | 85.35 | 88.10 | +2.75 (15 / 3; 0.008, 0.07) | +2.97 (24 / 11; 0.041, 0.41) |
| Text (704) | 92.33 | 93.04 | 94.18 | +1.14 (12 / 4; 0.077, 0.44) | +1.85 (25 / 12; 0.047, 0.42) |
| Icon (540) | 84.81 | 81.67 | 85.37 | **+3.70** (22 / 2; 4 × 10⁻⁵, **0.0004**) | +0.56 (25 / 22; 0.77, 1) |

- **Every split rose against the loop's start, but only icons is significant on its own** after
  correcting for the 11 slice tests: 22 fixed vs 2 broken, Holm p = 0.0004. Mobile (11 vs 1) and
  web (15 vs 3) are close, at Holm p 0.06–0.07. Exact McNemar per slice, Holm-adjusted over the 11
  slices, from `scripts/mcnemar_slices.py` on the paired compare reports
  ([vs loop start](eval_results/compare_ssv2_sft-lightning_vs_final/compare_report.md),
  [vs template](eval_results/compare_ssv2_sft-template_vs_final/compare_report.md)).
- **Much of the icon gain recovers the Lightning SFT's deficit there.** Against template, icons
  are even (25 fixed vs 22 broken, p = 0.77) and desktop is −0.7 (11 vs 13). Web is the
  consistent gain on SSv2-clean: against template, web text survives the correction (16 vs 3,
  Holm p = 0.049). But FineWeb fell 1.7 on dev in the same round, so the slice story isn't
  stable.
- **On dev, round 1's own gain was suggestive, not significant:** 22 fixed, 15 broken,
  p = 0.32 ([BLADE round 1](blade/round_1_r0_vs_r1.md)). The held-out pair against the loop's
  start is the significant result.
- **By source on dev, AndroidWorld rose most:** 78.4 → 84.9 on 139 rows (10 fixed, 1 broken),
  and the Android TextView role went 77.5 → 86.5. That puts it back above template SFT's 81.3.
  BLADE on the template adapter had found a row-centre click habit on Android lists, which the
  point-in-box reward scores as a miss. We haven't checked whether round 1's fixes are those
  rows.

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
tools; only the grounder is configured differently, by served name or by the adapter behind `grounder`. The FINAL
arm ran 3 h later on a restarted `:8001`, and BLADE found summarizer and planner differences on identical
inputs, so that arm is confounded):

| Grounder | Verified success | Claimed success | Invalid tool calls | Status |
| --- | --- | --- | --- | --- |
| `base` | 18/36 (50%) | 24/36 (8 false) | 10/259 | Sat 18:45, `agent-v4`, reportable |
| `grounder` = sft-template | **22/36 (61%)** | 25/36 (4 false) | 6/234 | Sat 19:05, `agent-v4`, reportable |
| `grounder` = FINAL (sft-lightning + GRPO round 1) | 17/36 (47%) | 27/36 (11 false) | 6/256 | Sat 22:04, `agent-v4`, reportable |

- **All three are within noise on agent success.** FINAL vs base: `mean/reward` −0.028, 95% CI
  [−0.20, +0.14]; failed actions 10 → 0 (the compare CI excludes 0, but all 10 sit in 4 base
  rollouts of 2 tasks); text-size-large 0/3 → 3/3.
- **We find no evidence that the 22 → 17 drop is a grounder regression.** About 6 of the 9
  changed rollouts trace to the planner or the summarizer. FINAL's hand-labelled hit rate on
  well-posed targets is 96.5% (110/114) vs SFT's 92.6% (112/121), not significant. FINAL also ran
  on a restarted serving process. Details: [blade/agent_final_vs_sft.md](blade/agent_final_vs_sft.md).
- **So the Gym-driven gains are the grounding ones** (dev, ScreenSpot-v2-clean). Inside the agent,
  both tuned grounders remove base's failed actions, so that comes from SFT, and the loop adds no
  measurable agent change. End-to-end success at n=36 can't separate the grounders.

Base vs sft-template (Sat 18:45 / 19:05):

- **Strict-check caveat:** with a strict `mark-all-read` check, sft-template is 20/36 (+2), as the
  current check doesn't guard the bell notifications ([BLADE](blade/agent_base_vs_sft.md)). The
  check was kept unchanged for the FINAL rerun (Sat 22:04), so all three arms are comparable.
- **`gym eval compare`:** `mean/reward` +0.111, 95% CI [−0.06, +0.28], so suggestive but not
  significant at n=36. Failed actions per episode −0.28, CI [−0.52, −0.04] (10 → 0). The CI
  excludes 0, but the 10 are clustered in 4 rollouts of 2 tasks. No-change actions 83 → 45, and Lightning's done claims matched the sandbox 31/36 vs
  26/36. Flips: 3 fail→pass, 1 pass→fail.
- **Per task (verified/3):** text-size-large 0→2, sms-billing-off 0→2, message-support-meter 1→2,
  high-contrast 2→1. The scroll tasks stay 0/3 in all three arms. The canvas chart task
  (max-usage-month) is 0/3 for base and SFT and 1/3 for FINAL, planner-driven per
  [BLADE](blade/agent_final_vs_sft.md). The other canvas task (report-outage-riverside) passes
  3/3 in every arm.
- **Setup:** official BF16 Lightning (thinking off), sandbox `68c7671`, `base` screen descriptions
  in both conditions, 0 masked ([runs.md](runs.md) Sat 18:45).

An early Saturday read (interactive harness, not Gym; agent-v3; n=1, 4 easy tasks; not in
runs.md) had the pilot adapter at 4/4 and base at 3/4. The
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
  `AgentSession.execute`, so Gym runs the agent code we demo, not a copy of it. (The demo adds
  the 25-step cap with a stuck stop and the pre-flight request check; Gym keeps 12 steps.)
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

- **Clock:** the overnight run used extended limits, `--last-round-start 04:15 --no-new-step 05:25
  --hard-end 05:40`, and at most 3 rounds. The code defaults are 01:30 / 02:40 / 02:50. Every
  step's timeout is capped at the hard end, so `finish()` (cleanup, serving the best adapter,
  `FINAL.json`) runs by then. In the event the loop ended on the keep rule: `FINAL.json` was
  written at 22:02, and the box was stopped at 22:35 after the final evals.
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
`gym eval profile` rollouts, at most 2 distinct replies per row. Those replies would be
model-generated training data, so they'd be published with the command and config that made
them. The fallback didn't run overnight. Rehearsal 2's 541 rows trained no reported model.

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

**Overnight rounds** (Sat 19:38 → 22:02, launch `20261017-193859`, from sft-lightning; runs.md
Sat 19:38–22:09):

| Round | Pool (by source) | Method | Steps | s/step | Verify reward first → last | Dev before → after | Kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **1,282** of 4,000 profiled rows (32% sometimes-pass): fineweb 472, linux 230, uibert 220, androidworld 183, macos 177; default knobs | GRPO | 170 (round deadline) | 20.6 (gate D2 20.0) | 0.77 → 0.90 | 86.27 (round-0 re-eval) → **87.00** (+0.72) | **Yes** |
| 2 | **242** weighted entries (source weights repeat uibert and fineweb rows and subsample androidworld and macos) from 203 candidates: 223 of 4,000 sometimes-pass (5.6%) minus 20 in `exclude_ids`. Fineweb 81, uibert 78, linux 63, androidworld 10, macos 10; BLADE knobs | GRPO | 167 (round deadline) | 20.4 (gate D2 21.1) | 0.53 → 0.86 | 87.00 → 86.89 (−0.10) | No: reverted, loop stopped |
| 3 | — | not run (round 2 wasn't kept) | | | | | |

Dev deltas here are exact row counts: round 1 was +0.72 (7 rows), round 2 was −0.10 (1 row).
The loop's own logs show +0.73 / −0.11 because they subtract accuracies already rounded to two
decimals. The keep decisions are unchanged.

- **Both rounds were clean runs:** 0 `/verify` failures in 16,176 GRPO reward calls, peak 45 GiB,
  and no crashes. Round 1's profile (4,000 rows × 8 samples) took 7 minutes.
- **The final adapter was served automatically:** the loop's `finish()` put it on `:8001` as
  `grounder` at 22:00:59 and wrote `FINAL.json` at 22:02. Two independent SSv2-clean evals of it
  followed: Kyle's at 22:04–22:09 (90.35, with the paired compares) and the orchestrator's
  `night-r1` (90.27; dev 86.89). The H200 was stopped at 22:35.

**BLADE steered round 2, through knobs only** ([blade/round_1_r0_vs_r1.md](blade/round_1_r0_vs_r1.md)).
The orchestrator ran BLADE on the round-0 → round-1 dev pair at 20:51 and uploaded `knobs.json`.
Round 2's `knobs_used.json` shows the new values with no clamps or notes. The values come from a
fixed rule in `scripts/blade_pair.py` (plain Python, no model):
- **`source_weights`:** clip(the source's model-attributable error ÷ the overall error, rounded to
  0.25, to [0.5, 2.0]), plus 0.25 for a source with ≥ 5 broken rows. That gave uibert 1.75 (8.2%
  vs 4.8%), fineweb 1.25 (5.0%, plus 8 broken rows), androidworld 0.75, macos 0.5 (the floor) and
  linux 1.0.
- **`exclude_ids`:** 285 train rows. 225 have labels nothing on screen can match or boxes too thin
  to hit (D-Bus names, AX handles, value literals, file-name slugs, sub-precision boxes). The other
  60 are unprofiled rows from screenshots where every profiled row never passes.

**Round 2 was flat** (−0.10, one dev row, within noise). **A likely contributor: round 1 sharpened
the model.** Round 1's GRPO cut training-time entropy from 0.27 to 0.19. On a fresh 4,000-row
profile at temperature 1.0, the round-1 adapter got the same result on all 8 samples for 94% of
rows, against 68% in round 1's profile. So far fewer rows had any signal to learn from:

| Round-2 profile vs round 1's | Round 1 | Round 2 |
| --- | --- | --- |
| Always-pass rows | 2,469 | 3,448 |
| Sometimes-pass rows (the GRPO candidates) | 1,282 (32%) | 223 (5.6%) |
| Never-pass rows | 249 | 329 |

- **Round 2 ended almost fully saturated.** It drew about 1,336 prompts from at most 203 unique
  rows, about 6.6 passes. Within the round, the share of groups with no gradient signal (all 6
  replies scored the same) rose from 0.43 to 0.94. Averaged over each round, though, the two were
  similar (0.72 in round 1 vs 0.74), so saturation alone doesn't explain the difference.
- **Other changes are confounded with it.** The BLADE knobs also changed in round 2 (above), so
  we can't separate their effect from the smaller pool.
- **The keep gate did its job.** The training reward still rose, but dev didn't, so the round
  was reverted and round 1 stayed the served adapter.
- **What we'd change next:** profile more than 4,000 rows per round, and cap GRPO at about 2
  passes over the pool.
  - Profile size is a launch flag (`--profile-rows`), not a knob.
  - `grpo_max_steps` is a knob, but knobs are read at the start of a round, before the pool size
    is known. BLADE's rule also only sets `source_weights` and `exclude_ids`, so it stayed at 200.

Each round leaves Gym artifacts in `/home/ubuntu/night/round_<r>/`: profile rollouts and
`profile` output, `pool.jsonl` and its pass-rate stats per source (`pool_stats.json`),
`knobs_used.json`, `grpo_command.txt` and `grpo.log`, dev rollouts, the `gym eval compare` report
and `summary.json`. Adapters go to `/home/ubuntu/ckpt/night/<launch>/round_<r>/`. The
paired SSv2-clean compare reports are in `/home/ubuntu/night/evals/compare_ssv2_*_vs_final/`. The
box is stopped. The round artifacts (summaries, pools, rollouts, compare reports) aren't committed:
they stay on the box's disk and in the orchestrator's copy on Patrick's Mac. Every number quoted
here is logged in [runs.md](runs.md) or the BLADE reports.

## 5. Measurement we can trust

- **Splits (decided before training):**
  - An OS-Atlas `dev` split, held out **by screenshot**, chooses checkpoints.
  - **ScreenSpot-v2-clean** is the held-out test. It is evaluated after the dev pick and is
    never used for selection. The final adapter has two independent runs (90.35, 90.27).
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

BLADE ran on base → template SFT, on round 0 → round 1 (the pair that set round 2's knobs), and
on the agent arms. Round 2 was reverted and wasn't analysed. The knob values come from the fixed
rule in `scripts/blade_pair.py`; no model chooses them.

| Pair | Main findings | Action it drove |
| --- | --- | --- |
| Base → template SFT ([round_0_base_vs_sft.md](blade/round_0_base_vs_sft.md)) | SFT +9.5 dev (template's 85.86 launch; 118 fixed, 26 broken, McNemar p = 3e-15). About 29% of SFT's failures are data artifacts (misaligned boxes, ungroundable AX / D-Bus labels, sliver boxes). About 24% are points where both models agree on the same wrong spot, mostly duplicate link text on FineWeb. On AndroidWorld, SFT clicks the middle of list rows | Knob recommendation **not applied**: the loop had launched from sft-lightning on default knobs first |
| Round 0 → round 1 ([round_1_r0_vs_r1.md](blade/round_1_r0_vs_r1.md)) | +0.72 dev (22 fixed, 15 broken, p = 0.32). AndroidWorld +6.5, desktop +2.1, mobile +2.2; FineWeb −1.7, UIBert −2.2. Of round 1's 126 dev failures, 41 are consensus misses, 41 data artifacts and 44 model misses (far, wrong element or near) | Round-2 knobs: `source_weights` and 285 `exclude_ids` (§4). Round 2 was flat and reverted |
| Agent: base / SFT / FINAL ([agent_base_vs_sft.md](blade/agent_base_vs_sft.md), [agent_final_vs_sft.md](blade/agent_final_vs_sft.md)) | Success within noise for all three. Failed actions 10 → 0 with either tuned grounder. star-landlord: the planner never scrolls. stop-promotions: it scrolls late, and the grid is cut off in the summary | None applied. BLADE recommends a "not found" grounder outcome (in training and in the GRPO reward) as the next step |

**Read from the slices before BLADE** (kept for the record):

- **SFT fixed most of the desktop gap.** Linux and macOS dev rose about 18 points. Base Qwen3.5
  was weakest there (68.8 and 73.4), and our data mix gives desktop ~35%.
- **Icons were the remaining headroom.** On SSv2-clean, template's icon score was 84.8 vs text
  92.3. The final adapter's icon score is 85.4.
- **Web gained least on dev** (FineWeb +1.6). Round-1 BLADE later raised fineweb to 1.25, because
  round 1 broke 8 FineWeb rows.

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
| `max_steps` counts model calls, checked after that turn's tools | `tool_choice: required` would run every task to `max_steps` | `max_steps: 12` (the interactive budget when the rows were built; the interactive agent now allows 25 with a stuck stop); `make_tasks.py --check` rejects rows that set `tool_choice` |
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
  adapter also moved 0.2 points between two launches (85.66 vs 85.86, §5).
- **Paired tests, McNemar exact on fixed vs broken tasks:**
  - base → final on SSv2-clean: 89 vs 38, p ≈ 7 × 10⁻⁶ (computed Sun 09:43, runs.md;
    [report](eval_results/compare_ssv2_base_vs_final/compare_report.md));
  - the loop's start → final on SSv2-clean: 34 vs 6, p ≈ 8 × 10⁻⁶;
  - template SFT → final on SSv2-clean: 50 vs 34, p = 0.10;
  - round 0 → round 1 on dev: 22 vs 15, p = 0.32.
  So the final adapter's held-out gain over base is significant, and so is the loop's gain over
  its own starting adapter. Against template SFT it is not (p = 0.10), and round 1's dev gain on
  its own was not (p = 0.32).
- **One kept round.** Round 2 had 6× fewer sometimes-pass rows (223 vs 1,282),
  consistent with round 1 sharpening the model (§4). The Gym-driven gain comes from a single GRPO
  round.
- **No equal-compute non-GRPO control.** We didn't run, say, rejection-sampled or continued SFT
  on the same 1,282-row pool. So the +2.25 shows that the loop's round 1 improved its starting
  adapter, not that GRPO in particular was needed for it.
- **The SFT itself is not Gym-driven.** It's plain TRL on OS-Atlas, measured and selected in
  Gym. The Gym-driven training is the GRPO loop (or its rejection-sampled fallback), and the
  final table separates the two.
- **One sandbox app for the agent eval**, with prompts tuned on its traces. Treat absolute agent
  success as in-domain. The base-vs-tuned difference is the comparable number.
- **Rounds are time-boxed.** Each round has a 75-minute budget, and both rounds stopped on it at
  170 and 167 of 200 GRPO steps. The loop itself ended on the keep rule, not the clock.

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

# Round-0 candidate SFT (template variant), then dev + SSv2-clean through Gym (runs.md Sat 17:31)
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

# Round-0 SFT (Lightning variant, the dev winner), same config and seed (runs.md Sat 18:44, 19:35)
training/sft_full.sh lightning /home/ubuntu/data/os_atlas_20k/train_lightning.jsonl
python -m training.gymctl eval --adapter /home/ubuntu/ckpt/sft-lightning/adapter --name sft-lightning --ssv2 --gpu-util 0.35

# The overnight loop, in tmux: profile -> GRPO (Gym /verify reward) -> dev eval + gym eval compare -> keep/revert
# (runs.md Sat 19:38: default knobs, the night's extended cutoffs; BLADE edited knobs.json before round 2)
tmux new-session -d -s night -n loop
tmux send-keys -t night:loop "training/night_loop.sh --init-adapter /home/ubuntu/ckpt/sft-lightning/adapter \
  --train-file /home/ubuntu/data/os_atlas_20k/train_lightning.jsonl \
  --last-round-start 04:15 --no-new-step 05:25 --hard-end 05:40" Enter

# Final adapter: SSv2-clean eval (Kyle's run; the orchestrator's night-r1 is the second), then the paired compares (runs.md Sat 22:04)
FINAL=/home/ubuntu/ckpt/night/20261017-193859/round_1/grpo/adapter
python -m training.gymctl eval --adapter $FINAL --name final-round1-grpo --ssv2 --gpu-util 0.30
E=/home/ubuntu/night/evals
for b in sft-lightning sft-template; do
  gym eval compare --baseline $E/$b/ssv2_clean_$b/rollouts.jsonl \
    --candidates $E/final-round1-grpo/ssv2_clean_final-round1-grpo/rollouts.jsonl \
    --output-dir $E/compare_ssv2_${b}_vs_final --report-format both
done

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

**Published.** Both repos are private until the Sunday ~9:30 release check, then public.

- **The final adapter and the merged final model:**
  [patrickballou/gui-assist-grounder](https://huggingface.co/patrickballou/gui-assist-grounder).
- **The training slice manifest and the Lightning rewrites**, each with the command and config
  that made it: [patrickballou/gui-assist-data](https://huggingface.co/datasets/patrickballou/gui-assist-data).
  - Whether the 8,028 hosted-API rewrites are included depends on the NVIDIA API terms check in
    [status.md](status.md).
  - No rejection-sampled rows trained a reported model, so none are published.
