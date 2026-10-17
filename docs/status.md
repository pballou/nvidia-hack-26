# Team status (living doc)

**Updated:** Sat 2026-10-17 19:35 EDT by the orchestrator thread. This is the team-facing
"where is everything" page. Numbers come from [results.md](results.md) and [runs.md](runs.md).
If this page disagrees with them, they win.

## Now (Sat evening)

| What | Who | State |
| --- | --- | --- |
| Rewritten-data SFT (`train_lightning.jsonl`, same config and seed as `sft-template`) | Kyle | Running since ~18:50, ~55 min |
| Pick the round-0 adapter on dev (template 85.86 in the loop harness vs the rewritten-data SFT) | Kyle | After the SFT |
| Overnight NeMo Gym loop in `/home/ubuntu/night` ([B-night-loop.md](tasks/B-night-loop.md)) | Kyle starts it, then it runs on its own | Target start ~20:00 |
| Base-vs-tuned agent eval: Gym `web_agent`, 12 tasks × 3 repeats, `base` vs `grounder` (= `sft-template`) | T2 | **Done 19:30:** verified 18/36 → **22/36**; see below |
| Console `/logs` page (325848f) + review fixes | T1 | Fix pass in progress |
| Fully local grounder on the Mac (Gate G, [local-mac.md](local-mac.md)) | T4 | Cleared to load models (19:30) |
| BLADE base vs SFT on dev, plus the overnight BLADE runbook (`docs/blade/`) | Orchestrator | Running |
| HF dataset and model cards (`docs/hf/`), private data upload | Orchestrator | Tonight |
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
| 7:15–8:15 | One ScreenSpot-v2-clean eval of the final adapter; agent rerun only if GRPO won | Kyle, T2 |
| ~8:00 | HF model upload (private) | Orchestrator |
| 8:30–10:00 | Record the video | Charlie (+ everyone on camera) |
| ~9:30 | Release check, then make both HF repos public | Patrick |
| by 10:30 | Submit the form ([submission.md](submission.md)) | Patrick |
| 11:00 | Deadline | |

## Headline numbers so far

- **Grounding (Qwen3.5-2B, 2 MP, LoRA r16):**

  | | Dev (969 rows) | ScreenSpot-v2-clean (1,244 rows) |
  | --- | --- | --- |
  | Base | 76.4% | 86.3% |
  | SFT (one epoch on 18,652 raw-label rows) | 85.66% | 89.07% |

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
