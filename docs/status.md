# Team status (living doc)

**Updated:** Sun 2026-10-18 00:35 EDT by the orchestrator thread. This is the team-facing
"where is everything" page. Numbers come from [results.md](results.md) and [runs.md](runs.md).
If this page disagrees with them, they win.

## Sat evening (final state)

| What | Who | State |
| --- | --- | --- |
| Rewritten-data SFT (`train_lightning.jsonl`, same config and seed as `sft-template`) | Kyle | **Done 19:33** (`/home/ubuntu/ckpt/sft-lightning`, 572 steps) |
| Pick the round-0 adapter on dev | Kyle | **Done:** `sft-lightning` 86.27 vs `sft-template` 85.86 (same harness) |
| Overnight NeMo Gym loop in `/home/ubuntu/night` ([B-night-loop.md](tasks/B-night-loop.md)) | Automatic | **Done 22:02.** Round 1 GRPO kept (dev 86.27 → 87.00); round 2 (BLADE knobs) reverted (86.89), so the loop stopped. **FINAL = round-1 GRPO** (`/home/ubuntu/ckpt/night/20261017-193859/round_1/grpo/adapter`): dev 87.0, **ScreenSpot-v2-clean 90.35%** (duplicate run 90.27; never used for selection). `:8001` served it as `grounder` from 22:00:59 until the H200 stopped at 22:35. Write-up: [bounty-nemo-gym.md](bounty-nemo-gym.md) (updated Sun 00:17) |
| Base-vs-tuned agent eval: Gym `web_agent`, 12 tasks × 3 repeats, `base` vs `grounder` (= `sft-template`, then FINAL) | T2 | **Done.** 19:30: `sft-template` 22/36 vs base 18/36 (20/36 with a strict check). FINAL arm, Sat 22:04–22:29: **17/36**, failed actions 0, text-size-large 3/3. All three are within noise; see below |
| Thinking off / retry / on comparison: interactive agent, 12 tasks × 3 each, `grounder` = `sft-template` | T2 | **Done 22:01** (ran from 20:34): off 20/36, retry 22/36, on 26/32 (arm on stopped at 32/36 when `:8001` switched to FINAL); see below |
| Agent step cap: the interactive agent now allows 25 steps. With no progress (4 no-change actions, or the same action 3×), it first warns Lightning and turns thinking on, then stops as `stuck` if 3 more actions make no progress. That cut off 0 of today's successes in replay. The Gym eval stays at 12 with no stuck stop | T2 (b179fcd); T1 for the console | **Done** (the console shows the `stuck` stop reason, 24d51db) |
| Console `/logs` page (325848f) + review fixes | T1 | **Done** (review fixes 4e2cd67, Sat 20:10) |
| Fully local grounder on the Mac (Gate G, [local-mac.md](local-mac.md)) | T4 | **PASS on FINAL (22:22):** 200 dev rows, local 87.0% vs H200 87.0%, 99% agreement, identical prompt tokens. Merged FINAL at `~/models/grounder-merged-night-20261017-193859-round_1-grpo`; the video recipe uses it. The fully local e2e check (sft-template) went 2/3. Needs Teams/Chrome closed |
| BLADE base vs SFT on dev, plus the overnight BLADE runbook (`docs/blade/`) | Orchestrator | **Done** (97c3d70). Round 1 was kept, so BLADE on round 0 → round 1 ([round_1_r0_vs_r1.md](blade/round_1_r0_vs_r1.md)) set round 2's knobs at 20:51 (`source_weights`, 285 `exclude_ids`). Round 2 was reverted |
| HF dataset and model cards (`docs/hf/`, 97c3d70), private data upload | Orchestrator | **Data uploaded (private) 20:55**: `patrickballou/gui-assist-data`, 16 files, sha256s match the box. The model goes up Sunday ~8:00 |
| Video script ([demo-video-script.md](demo-video-script.md)), pitch segments from [pitch.md](pitch.md) §10 | Charlie | Updating numbers; dry run after ~19:45 |
| Qwen3-VL-2B zero-shot on ScreenSpot-v2-clean (90.4%, per the video script notes) | Charlie | Needs a row in runs.md / results.md |

## Night loop (automatic; ran Sat 19:38–22:02, ended early on the keep rule)

- **Loop flags:** `--last-round-start 04:15 --no-new-step 05:25 --hard-end 05:40`,
  `--max-rounds 3`. Each round: profile, GRPO on sometimes-pass rows with the Gym `/verify`
  reward, then a dev gate (keep only if ≥ +0.5 pts).
- **Round 1** (pool 1,282, 170 steps): dev 86.27 → 87.00 (+0.72, 7 rows), **kept**.
- **After round 1,** the orchestrator ran BLADE and changed only `source_weights` and
  `exclude_ids` in `knobs.json`; round 2 used them from 20:51.
- **Round 2** (pool 242, 167 steps): dev 86.89 (−0.10, 1 row), **reverted**, so the loop
  stopped. Round 3 didn't run.
- **`FINAL.json`** (best round 1) was written at 22:02. The round summaries and the FINAL adapter
  are on Patrick's Mac, and the H200 was stopped at 22:35, after the final evals.

## Sunday (updated Sat 23:15: fully local, no H200 needed)

The H200 was stopped Sat 22:35 with every eval done. The video records **fully local** on Patrick's
Mac (Gate G passed on FINAL). `brev start` is needed only if we want a live hybrid take with the
base/tuned toggle. The base-vs-tuned moment is covered by
[docs/assets/base_vs_tuned_text_size.png](assets/base_vs_tuned_text_size.png) (real overlays: base
clicks M when asked for L; FINAL clicks L).

| Time (EDT) | What | Who |
| --- | --- | --- |
| ~7:45 | Quit Teams/Chrome. T4 starts the local VLM (:8011, **base + FINAL LoRA in one instance**); then T1 relaunches the console fully local, **headless** (`--summary-model base`, clarify on, thinking off) from a fresh export of main. **Don't use `--headed`**: its font rendering changes the pixels, and pay-50 drops from 8/8 to 1/4 | Orchestrator → T4, T1 |
| (done Sun 03:42) | **Recording config verified headless:** pay-50 ✓ ×2 (7 steps, ~71 s; the dialog cue was present), text-size ✓ (3 steps, 26 s) | T1 |
| ~8:00 | One pay-50 dry run in the console. In Logs, check that the observation after "Pay bill" names "Full balance (selected)" and the Other-amount input | Patrick |
| ~8:00 | HF model upload (private): adapter + merged FINAL + model card with final numbers | Orchestrator |
| 8:00–8:30 | Rehearse the five demo tasks (FINAL passed each 3/3): download September statement, turn on autopay, pay $50, report a Riverside outage, make text bigger | Patrick |
| 8:30–10:00 | Record with Cmd+Shift+5 (screen + mic), not Zoom; team-on-camera segment separately. Show the console's run view (step screenshot + click overlay). Base vs tuned: live toggle (base is served again) or the cutaway image | Charlie (+ everyone) |
| ~9:30 | Release check (below), then make both HF repos public | Patrick |
| by 10:15 | Upload the video (unlisted YouTube) | Charlie |
| by 10:30 | Submit the form ([submission.md](submission.md)) | Patrick |
| 11:00 | Deadline | |

## Release check before going public (Patrick, Sunday ~9:30)

1. **NVIDIA API Trial Terms vs publishing the 8,028 hosted-API rewrites.** §6.3 says we own the
   generated content. §1.2 and §1.4 limit trial use to non-production, internal testing and
   evaluation. The local BF16 rows (1,003) fall under OpenMDW-1.1, which puts no restrictions on
   outputs. Confirm we're OK to publish, or publish the local rows only.
2. **Personal data copied from OS-Atlas labels:** one macOS label is the capture machine's
   account email (`…wa@gmail.com`). The others are public or example addresses. It's already
   public in OS-Atlas. Keep it verbatim (reproducible) or redact it in our release.
3. **Model card** (`docs/hf/gui-assist-grounder.README.md`): check that it carries the final
   adapter's numbers (adapter `night/20261017-193859/round_1/grpo`; dev 87.0, SSv2-clean 90.35)
   and that `library_name` matches what was uploaded (`transformers` with the merged FINAL at the
   root, as planned; `peft` if we ship only the adapter; the orchestrator sets it at the ~8:00
   upload).

## Headline numbers (final)

- **Grounding (Qwen3.5-2B, 2 MP, LoRA r16):**

  | | Dev (969 rows) | ScreenSpot-v2-clean (1,244 rows) |
  | --- | --- | --- |
  | Base | 76.4% | 86.3% |
  | SFT template (one epoch on 18,652 raw-label rows) | 85.66% (85.86 in the loop harness) | 89.07% |
  | SFT rewritten-data (18,299 rows, Lightning-rewritten instructions) | 86.27% (loop harness) | 88.10% |
  | **FINAL: + GRPO round 1 in NeMo Gym** (170 steps, `/verify` reward) | **87.00%** | **90.35%** (icon 85.4, text 94.2; duplicate run 90.27) |

  Checkpoints are chosen on dev; ScreenSpot-v2-clean is never used for selection.

  The Lightning-rewrite ablation is mixed: +0.41 on dev, which picks the round-0 checkpoint per protocol, but −0.97
  on ScreenSpot-v2-clean (icons 81.7 vs 84.8). The night loop started from `sft-lightning`, so
  FINAL = `sft-lightning` + GRPO round 1.

  On ScreenSpot-v2-clean, icons went from 78.1% (base) to 85.4% (FINAL). FINAL vs base is +4.1.
  FINAL vs `sft-lightning` is +2.25 (34 fixed / 6 broken, McNemar p ≈ 8e-6). FINAL vs
  `sft-template` is +1.29 (50 / 34, p = 0.10, not significant).
- **Night loop, round 1 on dev:** the +0.72 is 22 fixed / 15 broken rows (p = 0.32).
- **Night-loop rehearsal:** GRPO round 1 (111-row pool, 30 steps) gave +0.83 on dev. Dev
  run-to-run noise is about 0.2 points. GRPO runs at 20–32 s/step, under the 35 s gate.
- **Lightning data engine:**
  - 9,384 rewrite targets: 8,620 kept rewritten, 411 model drops kept with raw labels, 353
    removed by the numeric rule.
  - Human audit: 42 of 43 checkable rewrites were faithful.
- **Agent, Gym `web_agent`:** 12 tasks × 3 repeats, official BF16 Lightning, real sandbox,
  thinking off, the same rows for every arm. Base and `sft-template` ran Sat 18:45–19:25; the
  FINAL arm ran Sat 22:04–22:29:

  | | Base | `sft-template` | FINAL |
  | --- | --- | --- | --- |
  | Verified success | 18/36 (50%) | 22/36 (61%) | 17/36 (47%) |
  | Claim matches the sandbox | 26/36 | 31/36 | 24/36 |
  | False success claims | 8 | 4 | 11 |
  | Failed actions | 10 | **0** | **0** |
  | No-change actions | 83 | 45 | — |
  | `text-size-large` | 0/3 | 2/3 | **3/3** |

  - **All three are within noise.** `sft-template` vs base: mean reward +0.111, 95% CI
    [-0.06, +0.28]. FINAL vs base: −0.028, CI [−0.20, +0.14]. Neither is significant at n=36.
  - **The robust gain:** failed actions go 10 → 0 with either tuned grounder. The CI excludes 0,
    but the 10 sit in 4 base rollouts of 2 tasks.
  - **22 → 17 is not a grounder regression.** About 6 of the 9 changed rollouts come from the
    planner or the summarizer, and the 2 grounder-caused ones cancel. FINAL ran 3 hours later,
    on the `:8001` process restarted at 22:00:59 ([FINAL agent BLADE](blade/agent_final_vs_sft.md)).
  - With a strict `mark-all-read` check (the current one doesn't guard the bell notifications),
    `sft-template` is 20/36 (+2). The check stayed lenient for the FINAL rerun (Sat 22:04–22:29),
    so it stays comparable with the base arm ([agent BLADE](blade/agent_base_vs_sft.md)).
  - Base → `sft-template`: gains `text-size-large` 0→2, `sms-billing-off` 0→2,
    `message-support-meter` 1→2; regression `high-contrast` 2→1. `sft-template` → FINAL:
    `text-size-large` 2→3, `max-usage-month` 0→1; `high-contrast` 1→0, `mark-all-read` 3→1,
    `message-support-meter` 2→0, `sms-billing-off` 2→0.
  - Still 0/3 in every arm: the two scroll tasks. The canvas chart is 0/3 except FINAL's 1/3.
- **Thinking off / retry / on** (interactive agent, Sat 20:34–22:01, `grounder` = `sft-template`):
  off 20/36, retry 22/36, on 26/32. Arm on stopped at 32/36 when `:8001` switched to FINAL. On the
  same 32 jobs, on vs off p = 0.039 and on vs retry p = 0.18. Thinking on is a measured option,
  not the default: `retry` stays the interactive default, and the demo recording uses thinking
  off (decided Sat night; [results.md](results.md)).

## Where things are written

- **Plan and decisions:** [plan.md](plan.md). **Rules:** [../AGENTS.md](../AGENTS.md).
- **Results:** [results.md](results.md). **GPU run log:** [runs.md](runs.md).
- **Pitch:** [pitch.md](pitch.md). **Video:** [demo-video-script.md](demo-video-script.md).
  **Submission checklist:** [submission.md](submission.md).
- **Bounty write-ups:** [bounty-lightning.md](bounty-lightning.md) and
  [bounty-nemo-gym.md](bounty-nemo-gym.md).
