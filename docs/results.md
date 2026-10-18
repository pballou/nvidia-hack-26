# Results log

> **Note:** the Saturday agent reads ran on planner prompt agent-v3, whose examples
> overlapped some sandbox tasks, so their absolute numbers are optimistic. From
> tonight on, runs use agent-v4 (examples from unrelated apps). Base-vs-tuned is fair
> either way, because both sides use the same prompts.

Numbers as they land. The newest row in each table is the most recent result.
Commands and configs are in [runs.md](runs.md). "Reportable" means official BF16
Lightning, the real sandbox, and grounder choices made on the dev split.

## Grounding (accuracy = predicted click point inside the target box)

| Model | Dev (969 rows) | SSv2-clean (1,244 rows) | Notes |
| --- | --- | --- | --- |
| Qwen3.5-2B base | 76.4% | **86.3%** (icon 78.1, text 92.5) | Kyle, `gui_grounding` Gym env |
| Pilot SFT (2k rows raw labels, 5 min) | **81.3%** (+5.0) | — | Every source improved; Linux +11.9, FineWeb +2.2. Answer length 37.6 → 16.7 tokens. |
| **SFT template** (all 18,652 raw-label train rows, 1 epoch, LoRA r16) | **85.66%** (+9.3) | **89.07%** (+2.8; mobile 92.2 / desktop 89.5 / web 85.1; text 92.3 / **icon 84.8, +6.7**) | Kyle 17:31. The Lightning variant is next (ablation). |
| **SFT rewritten-data** (`train_lightning.jsonl`, 18,299 rows: Lightning-rewritten instructions, same config and seed; 572 steps) | **86.27%** (+0.41 vs template's 85.86 in the same harness; desktop 88.9 / mobile 83.5 / web 85.9; androidworld 78.4 vs 81.3, uibert 88.8 vs 85.8, fineweb 85.9 vs 84.5) | 88.10% (−0.97 vs template; mobile 91.2 / desktop 86.9 / web 85.3; text 93.0 / **icon 81.7**) | Kyle 19:33. **Template-vs-Lightning ablation, mixed:** dev +0.41 (about 2× the ~0.2 run-to-run noise; web and uibert up, androidworld down), but ScreenSpot-v2-clean −0.97 (desktop −2.6, icon −3.1). Picked on dev per protocol; SSv2-clean is table-only. Template trained on 353 extra rows. Round-0 checkpoint of tonight's loop (started 19:38). |
| **Night loop round 1: GRPO** (from `sft-lightning`, 170 steps, pool 1,282 sometimes-pass rows, Gym `/verify` reward, default knobs) | **87.00%** (+0.73; kept). androidworld 78.4 → 84.9, linux +2.3, macos +1.9; fineweb −1.7, uibert −2.2. McNemar p = 0.32 | **90.27%** (+3.97 vs base, +2.17 vs `sft-lightning`; mobile 93.2 / desktop 88.9 / web 87.9; text 94.2 / **icon 85.2**) | **FINAL** (`FINAL.json` best_round 1). 20:50. SSv2-clean eval 22:08 (`gymctl eval --name night-r1 --ssv2`); the same run re-scored dev at 86.89 (±0.2 noise). BLADE: [docs/blade/round_1_r0_vs_r1.md](blade/round_1_r0_vs_r1.md). For round 2, the knobs upweight uibert (1.75) and fineweb (1.25) and exclude 285 artifact train rows. |
| Night loop round 2: GRPO (BLADE knobs; pool 242) | 86.89% (−0.11, **not kept**; the loop stopped) | — | 22:02. The pool shrank from 1,282 to 242 rows: round 1's adapter already passes more train rows reliably, plus the 285 exclusions and the down-weighted sources. 167 steps on a small pool didn't generalize. |
| Night-loop rehearsal, GRPO round 1 (pool 111 sometimes-pass rows, 30 steps, from SFT template) | **86.69%** (+0.83 vs 85.86 in the same harness) | — | Kyle 17:36–17:50, `/home/ubuntu/night_rehearsal`. Tiny rehearsal, but it clears the +0.5 keep gate. (runs.md says +1.03 because it compares with the earlier 85.66 eval; +0.83 against the same-harness 85.86 is the fair number.) The template adapter re-evaluated at 85.86 vs 85.66 earlier: ~0.2 pt run-to-run noise on dev. A second rehearsal forced a D2 failure to test the rejection-sampled SFT fallback; it works. |
| Gate D2 GRPO smoke (20 steps) | — | — | **Pass: 31.9 s/step**, 960 `/verify` calls with 0 failures, reward 0.75 → 0.79 |
| Gate D1 comparison (OS-Atlas sample n=500, pre-cleanup) | Qwen3-VL-2B 80.6 vs Qwen3.5-2B 79.8 | — | `eval_results/qwen3vl_vs_qwen35_osatlas500.csv`. Qwen3.5 +7.0 on desktop. |

## Agent (sandbox tasks, verified from sandbox state)

### Gym `web_agent`: base vs SFT grounder (Sat 18:45–19:25, reportable)

All 12 sandbox tasks × 3 repeats, the same rows and planner for both conditions: official BF16
Lightning (thinking off), sandbox `68c7671` (stricter checks), prompt `agent-v4`, start screens
from Qwen3.5-2B `base`, screen descriptions `base` in both. Grounder `base` vs `grounder` =
`sft-template` LoRA, one vLLM process on `:8001`. 0 masked, 0 VLM infra errors, 0 stale start
screens. `gym eval compare`: 3 fail→pass, 1 pass→fail.

| | `base` grounder | `grounder` (sft-template) |
| --- | --- | --- |
| Verified success | 18/36 (50%) | **22/36 (61%)** |
| Claim matches verified | 26/36 (72%) | **31/36 (86%)** |
| False success claims | 8 | **4** |
| No-change actions | 83 | **45** |
| Failed actions (grounding/browser) | 10 | **0** |
| Tool-call validity | 249/259 (96%) | 228/234 (97%) |
| Mean steps | 7.2 | 6.5 |

- **Strict-check caveat:** with a strict `mark-all-read` check, tuned is 20/36 (+2), as the
  current check doesn't guard the bell notifications (BLADE). The check is left unchanged until
  after Sunday's grounder rerun, so that rerun stays comparable with this base arm.
- **Significance:** `mean/reward` +0.111, 95% CI [−0.06, +0.28]: suggestive, not significant
  at n=36. `n_failed_actions` −0.28/episode, CI [−0.52, −0.04]: significant.
- **Per task (verified/3, base → grounder):** text-size-large 0→2 (the L-vs-M grounding miss),
  sms-billing-off 0→2, message-support-meter 1→2, high-contrast 2→1. Unchanged: download,
  mark-all-read, pay-50, report-outage, autopay (3→3); max-usage-month (canvas),
  star-landlord-message (scroll), stop-promotions (scroll) stay 0→0.
- **Latency:** not comparable; Kyle's SFT shared the GPU during the grounder run.
- **Artifacts (T2's Mac):** `results/agent_v4_sft_template/` (rollouts, aggregate metrics,
  run_info, `agent_compare/compare_report.md`). BLADE: [docs/blade/agent_base_vs_sft.md](blade/agent_base_vs_sft.md):
  the +4 is not significant (Fisher p = 0.48), only ~2-3 of it is the grounder, grounder output
  failures 10 → 0 is the robust gain, and the biggest drop-off is the shared observation pipeline
  (summaries cut off the notifications grid and drop toggle state). It also found a lenient
  `mark-all-read` check (2 SFT passes cleared the bell notifications too; strict: 20/36).
- **Step budget:** the Gym eval keeps `max_steps` 12 (this table and Sunday's grounder rerun,
  `BASELINE=` in `scripts/agent_eval.sh` reuses this base arm). Since Sat ~19:40 the
  *interactive* agent (console, demo) uses `max_steps` 25 plus a "stuck" stop
  (`agent/progress.py`: warn after 4 no-progress actions or 3 repeats of the same action, stop
  if 3 more make no progress). Replayed on the 47 real runs, it would have stopped none of the
  29 that succeeded.

### First reads (Sat afternoon, prompt `agent-v3`, checks before `68c7671`; absolute numbers optimistic)

First read on 2026-10-17. n=1 per task, sequential, 4 easy tasks.

| Grounder | Verified | Claimed | Notes |
| --- | --- | --- | --- |
| Qwen3.5-2B base (:8001) | 3/4 | 2/4 | `text-size-large` failed: the planner correctly asked for "the L option", but the grounder hit M every time (a pure grounding miss). 0 invalid tool calls / 27. |
| Pilot adapter (:8003, not final) | **4/4** | 3/4 | It fixed the `text-size-large` grounding miss. |

- **Per-stage latency, base run, p50/p90:**

  | Stage | p50 | p90 |
  | --- | --- | --- |
  | plan (Lightning; includes thinking-retry steps and a shared server) | 4.8 s | 16.8 s |
  | ground | 0.73 s | 0.84 s |
  | summary | 1.15 s | 1.80 s |
  | settle | 0.51 s | |
  | screenshot | 57 ms | |
  | act | 11 ms | |

- **Claimed vs verified gap:** the planner sometimes doesn't claim success when
  the sandbox shows success (e.g. `mark-all-read`, where the read state wasn't
  visible in the summary). **Still open:** with the base summary model, the
  planner sometimes can't see that its action worked. The pilot-adapter run did
  claim it (6 steps). Tracked in the claimed-vs-verified column. This is useful
  material for BLADE and the Lightning write-up.

### All 12 sandbox tasks, base grounder (2026-10-17 ~17:05, n=1 per task per mode, sequential)

| Metric | Thinking off | Thinking on retry |
| --- | --- | --- |
| Verified success | 5/12 | 6/12 |
| Claim matches verified | 6/12 | **9/12** |
| No-change actions | 45 | **27** |
| Hard tasks verified | 0/3 | 2/3 |
| Mean steps | 9.1 | 8.0 |
| Plan p50/p90 | 4.1/5.5 s | 4.3/14.5 s |
| Wall-clock p50 per task | 67 s | 80 s |
| Tool-call validity | 109/109 | 95/96 |

- **Reading:** thinking on retry mostly improves self-assessment (claims match the
  sandbox state) and cuts wasted clicks, at the cost of a slower p90. The 5 vs 6
  success gap is noise at n=1.
- **Failure clusters (input for BLADE):**
  - scroll-dependent targets (`star-landlord-message`, `stop-promotions`);
  - the canvas chart (`max-usage-month`);
  - dense toggle grids (`sms-billing-off`, with a false done claim);
  - a planner error: `pay-50` with thinking off paid the full balance instead of $50.
- **Gym `web_agent` acceptance run** (5 committed rows, base, thinking off): reward
  3/5, 0 masked, claimed 2/5 (one false claim), tool-call validity 42/43.
- **Sources:** `runs/summary_20261017-165130.json` (off),
  `runs/summary_20261017-170457.json` (retry), `results/acceptance_base/`
  (T2's Mac).

- **Next:** the Gym `web_agent` eval with `--num-repeats 3`, base vs final
  grounder (tonight, once :8001 serves `grounder`).

## Lightning data engine (T3) audit

- **Delivered 18:19 to the box:** `train_lightning.jsonl` and `train_template.jsonl`, 18,299 rows
  each, identical row sets and order (sha256 in `/home/ubuntu/data/os_atlas_20k/train_files.READY.json`).
  9,384 rewrite targets: 8,620 kept rewritten, 411 model drops kept with raw labels, 353 removed by
  the numeric rule. Rewrites by endpoint: hosted 8,028, local official BF16 (Mac) 1,003. Record:
  `data_prep/rewrite/runs/os_atlas_final_v4/` (5b9bf25).
- **Ablation caveat:** the existing `sft-template` adapter trained on 18,652 rows (353 more than
  these files).

- **Prompt v4, hosted Lightning, 50 kept + 16 dropped rows (Patrick, 2026-10-17):**
  - Rewrite faithfulness **42/43 = 97.7%** of the cards he could verify (6 unsure,
    all unverifiable by the reviewer; 1 unfaithful from a source ambiguity).
  - **13 of 16 model drops overturned** (readable labels dropped as
    "unintelligible"). Policy: model drops fall back to the raw label.
