# Team status (living doc)

**Updated:** Sat 2026-10-17 20:58 EDT by the orchestrator thread. This is the team-facing
"where is everything" page. Numbers come from [results.md](results.md) and [runs.md](runs.md).
If this page disagrees with them, they win.

## Now (Sat evening)

| What | Who | State |
| --- | --- | --- |
| Rewritten-data SFT (`train_lightning.jsonl`, same config and seed as `sft-template`) | Kyle | **Done 19:33** (`/home/ubuntu/ckpt/sft-lightning`, 572 steps) |
| Pick the round-0 adapter on dev | Kyle | **Done:** `sft-lightning` 86.27 vs `sft-template` 85.86 (same harness) |
| Overnight NeMo Gym loop in `/home/ubuntu/night` ([B-night-loop.md](tasks/B-night-loop.md)) | Automatic | Started 19:38 from `sft-lightning` (dev 86.27). **Round 1 GRPO kept: dev 87.00 (+0.73)**. BLADE knobs applied 20:51 ([round 1 report](blade/round_1_r0_vs_r1.md)), and round 2 is running. `:8001` `grounder` stays `sft-template` until the loop's end step |
| Base-vs-tuned agent eval: Gym `web_agent`, 12 tasks × 3 repeats, `base` vs `grounder` (= `sft-template`) | T2 | **Done 19:30:** verified 18/36 → **22/36** (20/36 with a strict check); see below |
| Thinking off / retry / on comparison: interactive agent, 12 tasks × 3 each, `grounder` | T2 | Running since 20:35 (~2 h); pauses for the dry run |
| Agent step cap: the interactive agent now allows 25 steps. With no progress (4 no-change actions, or the same action 3×), it first warns Lightning and turns thinking on, then stops as `stuck` if 3 more actions make no progress. That cut off 0 of today's successes in replay. The Gym eval stays at 12 with no stuck stop | T2 (b179fcd); T1 for the console | **Done** (console display pending) |
| Console `/logs` page (325848f) + review fixes | T1 | Fix pass in progress |
| Fully local grounder on the Mac (Gate G, [local-mac.md](local-mac.md)) | T4 | **PASS 20:22:** 200 dev rows, local vllm-metal 86.0% vs H200 85.0%; 99% of hit/miss decisions agree; prompt tokens identical; local p50 1.77 s per click (H200 0.75 s). Needs Teams/Chrome closed. **Fully local end-to-end (20:32):** 2/3 verified (pay-50 and text-size pass; download-september falsely claimed success after two wrong rows). Per step: plan p50 4.6 s, click 1.0 s, screen summary 6.2 s (the slow stage on the Mac). Recipe in local-mac.md (T4) |
| BLADE base vs SFT on dev, plus the overnight BLADE runbook (`docs/blade/`) | Orchestrator | **Done** (97c3d70). Knob changes are proposed for after round 1, if it's kept |
| HF dataset and model cards (`docs/hf/`, 97c3d70), private data upload | Orchestrator | **Data uploaded (private) 20:55**: `patrickballou/gui-assist-data`, 16 files, sha256s match the box. The model goes up Sunday ~8:00 |
| Video script ([demo-video-script.md](demo-video-script.md)), pitch segments from [pitch.md](pitch.md) §10 | Charlie | Updating numbers; dry run after ~19:45 |
| Qwen3-VL-2B zero-shot on ScreenSpot-v2-clean (90.4%, per the video script notes) | Charlie | Needs a row in runs.md / results.md |

## Overnight (automatic)

- **Loop flags tonight:** `--last-round-start 04:15 --no-new-step 05:25 --hard-end 05:40`,
  `--max-rounds 3`. Each round: profile, GRPO on sometimes-pass rows with the Gym `/verify`
  reward, then a dev gate (keep only if ≥ +0.5 pts).
- **After each round,** the orchestrator runs BLADE and may change only `source_weights` and
  `exclude_ids` in `knobs.json`, within 10 min.
- **When `FINAL.json` exists,** the orchestrator copies the summaries and adapters to Patrick's
  Mac and runs `brev stop`. That happens by 6:00 at the latest.

## Sunday

| Time (EDT) | What | Who |
| --- | --- | --- |
| ~7:00 | `brev start`, re-serve `:8001` with the final adapter | Orchestrator (Kyle's OK) |
| 7:15–8:15 | One ScreenSpot-v2-clean eval of the final adapter; **agent rerun of the `grounder` arm with the final adapter** (always: tonight's 22/36 is `sft-template`, and the loop starts from `sft-lightning`; tonight's base arm is reused). Command: `GROUNDERS=grounder OUT_DIR=results/agent_final BASELINE=results/agent_v4_sft_template/agent_base.jsonl AGENT_SANDBOX_URL=http://127.0.0.1:8091 HEAD_PORT=21017 TASKS=resources_servers/web_agent/data/tasks_all.jsonl REPEATS=3 CONCURRENCY=2 scripts/agent_eval.sh` (don't regenerate `tasks_all.jsonl` before then) | Kyle, T2 |
| ~8:00 | HF model upload (private) | Orchestrator |
| 8:30–10:00 | Record the video | Charlie (+ everyone on camera) |
| ~9:30 | Release check, then make both HF repos public | Patrick |
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
3. **Model card:** fill in the final adapter's numbers, and set `library_name` (`peft` if we
   ship only the adapter).

## Headline numbers so far

- **Grounding (Qwen3.5-2B, 2 MP, LoRA r16):**

  | | Dev (969 rows) | ScreenSpot-v2-clean (1,244 rows) |
  | --- | --- | --- |
  | Base | 76.4% | 86.3% |
  | SFT template (one epoch on 18,652 raw-label rows) | 85.66% (85.86 in the loop harness) | 89.07% |
  | SFT rewritten-data (18,299 rows, Lightning-rewritten instructions) | **86.27%** (loop harness) | 88.10% |

  The Lightning-rewrite ablation is mixed: +0.41 on dev, which picks the round-0 checkpoint per protocol, but −0.97
  on ScreenSpot-v2-clean (icons 81.7 vs 84.8). The overnight loop starts from `sft-lightning`.

  On ScreenSpot-v2-clean, icons went from 78.1% to 84.8%.
- **Night-loop rehearsal:** GRPO round 1 (111-row pool, 30 steps) gave +0.83 on dev. Dev
  run-to-run noise is about 0.2 points. GRPO runs at 20–32 s/step, under the 35 s gate.
- **Lightning data engine:**
  - 9,384 rewrite targets: 8,620 kept rewritten, 411 model drops kept with raw labels, 353
    removed by the numeric rule.
  - Human audit: 42 of 43 checkable rewrites were faithful.
- **Agent, Gym `web_agent`:** 12 tasks × 3 repeats, official BF16 Lightning, real sandbox,
  thinking off. Base vs `grounder` (= `sft-template`):

  | | Base | Tuned |
  | --- | --- | --- |
  | Verified success | 18/36 (50%) | **22/36 (61%)** |
  | Claim matches the sandbox | 26/36 | **31/36** |
  | False success claims | 8 | **4** |
  | Failed actions | 10 | **0** |
  | No-change actions | 83 | **45** |

  - Mean reward +0.111. The 95% CI is [-0.06, +0.28], so it's suggestive but not significant
    at n=36. The drop in failed actions is significant.
  - With a strict `mark-all-read` check (the current one doesn't guard the bell notifications),
    tuned is 20/36 (+2). The fix is deferred until after Sunday's rerun
    ([agent BLADE](blade/agent_base_vs_sft.md)).
  - Gains: `text-size-large` 0→2, `sms-billing-off` 0→2, `message-support-meter` 1→2.
    Regression: `high-contrast` 2→1.
  - Still 0/3: the canvas chart and the two scroll tasks.

## Where things are written

- **Plan and decisions:** [plan.md](plan.md). **Rules:** [../AGENTS.md](../AGENTS.md).
- **Results:** [results.md](results.md). **GPU run log:** [runs.md](runs.md).
- **Pitch:** [pitch.md](pitch.md). **Video:** [demo-video-script.md](demo-video-script.md).
  **Submission checklist:** [submission.md](submission.md).
- **Bounty write-ups:** [bounty-lightning.md](bounty-lightning.md) and
  [bounty-nemo-gym.md](bounty-nemo-gym.md).
