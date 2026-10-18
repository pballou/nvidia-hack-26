# Results log

> **Note:** the Saturday afternoon agent reads ran on planner prompt agent-v3, whose
> examples overlapped some sandbox tasks, so their absolute numbers are optimistic. From
> Saturday evening on, runs use agent-v4 (examples from unrelated apps). Base-vs-tuned is
> fair either way, because both sides use the same prompts.

Numbers as they land. The newest row in each table is the most recent result.
Commands and configs are in [runs.md](runs.md). "Reportable" means official BF16
Lightning, the real sandbox, and grounder choices made on the dev split.

## Grounding (accuracy = predicted click point inside the target box)

| Model | Dev (969 rows) | SSv2-clean (1,244 rows) | Notes |
| --- | --- | --- | --- |
| Qwen3.5-2B base | 76.4% | **86.3%** (icon 78.1, text 92.5) | Kyle, `gui_grounding` Gym env |
| Pilot SFT (2k rows raw labels, 5 min) | **81.3%** (+5.0) | — | Every source improved; Linux +11.9, FineWeb +2.2. Answer length 37.6 → 16.7 tokens. |
| **SFT template** (all 18,652 raw-label train rows, 1 epoch, LoRA r16) | **85.66%** (+9.3) | **89.07%** (+2.8; mobile 92.2 / desktop 89.5 / web 85.1; text 92.3 / **icon 84.8, +6.7**) | Kyle 17:31. The Lightning variant (next row) is the ablation. |
| **SFT rewritten-data** (`train_lightning.jsonl`, 18,299 rows: Lightning-rewritten instructions, same config and seed; 572 steps) | **86.27%** (+0.41 vs template's 85.86 in the same harness; desktop 88.9 / mobile 83.5 / web 85.9; androidworld 78.4 vs 81.3, uibert 88.8 vs 85.8, fineweb 85.9 vs 84.5) | 88.10% (−0.97 vs template; mobile 91.2 / desktop 86.9 / web 85.3; text 93.0 / **icon 81.7**) | Kyle 19:33. **Template-vs-Lightning ablation, mixed:** dev +0.41 (about 2× the ~0.2 run-to-run noise; web and uibert up, androidworld down), but ScreenSpot-v2-clean −0.97 (desktop −2.6, icon −3.1). Picked on dev per protocol; SSv2-clean is table-only. Template trained on 353 extra rows. Round-0 checkpoint of the night loop (started Sat 19:38). |
| **Night loop round 1: GRPO** (from `sft-lightning`, 170 steps, pool 1,282 sometimes-pass rows, Gym `/verify` reward, default knobs) | **87.00%** (+0.72, 7 rows; kept). androidworld 78.4 → 84.9, linux +2.3, macos +1.9; fineweb −1.7, uibert −2.2. McNemar p = 0.32 | **90.35%** (+4.10 vs base, +2.25 vs `sft-lightning`, with **34 rows fixed vs 6 broken** in a paired `gym eval compare`; mobile 93.2 / desktop 88.9 / web 88.1; text 94.2 / **icon 85.4**). A second independent T=0 run gave 90.27 (one row apart) | **FINAL** (`FINAL.json` best_round 1). 20:50. SSv2-clean: Kyle's `final-round1-grpo` eval with paired compares (runs.md); the orchestrator's duplicate `night-r1` run gave 90.27 / dev 86.89. BLADE: [docs/blade/round_1_r0_vs_r1.md](blade/round_1_r0_vs_r1.md). For round 2, the knobs upweight uibert (1.75) and fineweb (1.25) and exclude 285 artifact train rows. |
| Night loop round 2: GRPO (BLADE knobs; pool 242) | 86.89% (−0.10, one row; **not kept**; the loop stopped) | — | 22:02. The pool shrank from 1,282 to 242 rows: round 1's adapter already passes more train rows reliably, plus the 285 exclusions and the down-weighted sources. 167 steps on a small pool didn't generalize. |
| Night-loop rehearsal, GRPO round 1 (pool 111 sometimes-pass rows, 30 steps, from SFT template) | **86.69%** (+0.83 vs 85.86 in the same harness) | — | Kyle 17:36–17:50, `/home/ubuntu/night_rehearsal`. Tiny rehearsal, but it clears the +0.5 keep gate. (runs.md says +1.03 because it compares with the earlier 85.66 eval; +0.83 against the same-harness 85.86 is the fair number.) The template adapter re-evaluated at 85.86 vs 85.66 earlier: ~0.2 pt run-to-run noise on dev. A second rehearsal forced a D2 failure to test the rejection-sampled SFT fallback; it works. |
| Gate D2 GRPO smoke (20 steps) | — | — | **Pass: 31.9 s/step**, 960 `/verify` calls with 0 failures, reward 0.75 → 0.79 |
| Gate D1 comparison (OS-Atlas sample n=500, pre-cleanup) | Qwen3-VL-2B 80.6 vs Qwen3.5-2B 79.8 | — | `eval_results/qwen3vl_vs_qwen35_osatlas500.csv`. Qwen3.5 +7.0 on desktop. |

## Agent (sandbox tasks, verified from sandbox state)

### Demo-config check: Lightning temperature 1.0 vs 0.6 (Sun 03:07–03:18; not a headline)

Setup: fully local Mac.

- **VLM:** `:8011` on vllm-metal, one process serving `base` (descriptions) and `grounder` (the
  FINAL LoRA, clicks).
- **Lightning:** official BF16 on `:8000`, thinking off.
- **Harness:** `scripts/run_agent_tasks.py`, concurrency 1, with the adapter checked before and
  after every run. Clarify is off in the harness. That matches the console: a proceeding clarify
  call doesn't change what the planner sees.
- **Arms:** interleaved round by round, 4 repeats each.
- **Tasks:** pay-50, using the spoken phrasing from the demo ("Pay fifty dollars towards my
  bill."), and text-size-large.

| Arm | pay-50 verified | text-size-large verified | False claims | Steps (pay / text) | Wall p50 (pay / text) |
| --- | --- | --- | --- | --- | --- |
| T = 1.0 (model card) | 4/4 | 4/4 | 0 | 5-6 / 2-4 | 58 / 24 s |
| T = 0.6 | 4/4 | 4/4 | 0 | 5-7 / 3-6 | 62 / 26 s |

- **No temperature effect at this n.** NVIDIA's card recommends temperature 1.0 and top_p 0.95 for
  every mode, including thinking off and tool calling, and gives no greedy setting. The demo keeps
  1.0.
- **What decided pay-50 was the screen description of the pay dialog.**
  - These 8 runs: it said "Full balance (selected)" and listed the empty Other-amount input every
    time. The planner then chose Other amount every time, and all 8 passed.
  - T1's four console runs ~02:50, on near-identical screenshots (one is 1 px shifted): the
    description had neither cue every time. The planner passed 1/4. The other 3 continued with the
    preselected full balance, confirmed a dialog that said $142.37, and claimed $50.
  - The `:8011` process was restarted between the two sessions; what changed is being asked of T4.
- **Before recording,** do a console dry run of pay-50 and check that the dialog's description
  names the selected option.
- **Guards (post-Sunday):** check the confirmed amount against the request, and an A/B
  oscillation rule.

### Clarify pre-flight (Sat 23:00–23:50, official BF16 Lightning, local `:8000`)

`agent/clarify.py`, prompt `clarify-v4` (sha `c1adfb0587`): one Lightning call before the browser
opens (thinking off, T=0, `tool_choice` required) picks `proceed` or `ask_user`. Live pass with
`scripts/clarify_eval.py`, one call at a time. All three labelled sets are committed with their
recorded replies (`agent/tests/fixtures/clarify_*.json`), and `agent/tests/test_clarify.py`
replays them offline.

| Set | Clear requests that proceeded | Ask cases that asked | Notes |
| --- | --- | --- | --- |
| **Acceptance** (12 task phrasings, 30 spoken variants, 20 near-boundary proceed cases, 3 spec asks, 19 asks; used to tune v1 → v4) | **62/62 (0 false asks)** | **22/22** | Identical on a second repeat. Includes "Allors towards my bill.", "Pay something toward my bill", "turn it off" |
| Held-out B (written while v2 was drafted; used to tune v3 → v4) | 44/45 | 17/19 | False ask: "port an outage in the riverside district". Missed: two contradictory requests ("turn on paperless billing and keep mailing me paper bills") |
| **Held-out C** (written after v4; run once, frozen prompt) | 41/45 | 14/17 | False asks: two clipped last words ("turn on high contra") and two object-first sentences ("the bil pay fifty dollar on it please"). Missed: a security question change, a new card on file, "make the text bigger and smaller" |

- **Latency:** p50 1.6 s, p90 1.7 s per call.
- **How the sets were made:** the non-task phrasings were written by Claude workflows whose
  writers never saw the clarify prompt. The judges agreed with the writers' labels on 196 of 198
  phrasings; the 2 borderline ones aren't counted.
- **Decision (Sat night):** off in `configs/agent.yaml` (evals, CLI), because on fresh phrasings
  it asks about 5 of 90 clear requests and misses 5 of 36 asks. The **console turns it on, with
  a visible toggle**: the demo uses acceptance-set phrasings (0 false asks), and a false ask costs
  one extra answer, while a missed ask is the pre-clarify behavior. `AGENT_CLARIFY=on` or
  `python -m agent --clarify on` turns it on elsewhere. The harness and the Gym `web_agent`
  server force it off either way.
- **Prompt iterations, all on the dev sets:**
  - v1 was too lenient: 5/22 asks, and it turned "Allors towards my bill." into "pay bill".
  - v2 added explicit ask checks and examples from other websites.
  - v3 stopped it from asking a lookup question back to the user.
  - v4 lets clipped-but-obvious words and back-references proceed.
  - A v5 (contradiction as its own rule) regressed the acceptance set and was reverted. At T=0
    without thinking, boundary cases flip on small wording changes.

### Gym `web_agent`: FINAL grounder rerun (Sat 22:04–22:29, reportable)

Same rows (sha checked against the base arm below via `BASELINE=`), planner, thinking (off) and
`max_steps` 12. `grounder` = FINAL (`sft-lightning` + GRPO round 1, dev 87.00), served on `:8001`
from start to end (root checked both times). 0 masked, 0 VLM infra errors. Latency isn't
reportable (another job shared `:8001`).

| Verified (12 tasks × 3) | `base` | `sft-template` | **FINAL** |
| --- | --- | --- | --- |
| All tasks | 18/36 | 22/36 | 17/36 |
| text-size-large (small adjacent "L" option) | 0/3 | 2/3 | **3/3** |
| Failed actions (no usable point) | 10 | 0 | **0** |
| False success claims | 8 | 4 | 11 |

- **Agent success is within noise for all three grounders.** FINAL vs base: `mean/reward`
  −0.028, 95% CI [−0.20, +0.14]; SFT vs base CI [−0.06, +0.28]; SFT vs FINAL Fisher p ≈ 0.34
  (n=3 per task, planner at T=1.0). Don't claim that SFT or GRPO improved agent success, or that
  one adapter is the better agent grounder.
- **What the grounders clearly change inside the agent:** the tuned ones always return a usable
  point (failed actions 10 → 0, significant), they fix the small adjacent-option target
  (text-size-large 0/3 → 3/3 with FINAL), and FINAL hits 96.5% of well-posed targets vs SFT's
  92.6% (hand-labelled). FINAL also makes fewer unrequested setting changes (stop-promotions: 3
  vs 14 failed conditions).
- **The 22 → 17 drop is not a grounder regression.** About 6 of the 9 changed rollouts come from
  the planner or the summarizer. For example, all 3 FINAL `mark-all-read` runs leave the path at
  the planner's first call, before any grounder call. The `base` summarizer, on the `:8001`
  process restarted at 22:00:59, dropped the high-contrast state line it had shown in 6/6 earlier
  views. Add the lenient `mark-all-read` check. A clean end-to-end grounder comparison needs both
  adapters interleaved on one vLLM process.
- **Diagnosis:** [docs/blade/agent_final_vs_sft.md](blade/agent_final_vs_sft.md). It also
  recommends demoing with FINAL, scripted on tasks it passed 3/3.
- **Artifacts (T2's Mac):** `results/agent_final/` (rollouts, run_info, `agent_compare/`).

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
  current check doesn't guard the bell notifications (BLADE). The check was left unchanged for
  the FINAL grounder rerun (Sat 22:04, above), so that rerun stays comparable with this base arm.
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
- **Step budget:** the Gym eval keeps `max_steps` 12 (this table and the FINAL grounder rerun,
  whose `BASELINE=` in `scripts/agent_eval.sh` reused this base arm). Since Sat ~19:40 the
  *interactive* agent (console, demo) uses `max_steps` 25 plus a "stuck" stop
  (`agent/progress.py`: warn after 4 no-progress actions or 3 repeats of the same action, stop
  if 3 more make no progress). Replayed on the 47 real runs, it would have stopped none of the
  29 that succeeded.

### Thinking off / retry / on (Sat 20:34–22:01, interactive agent, reportable)

Interactive agent (25-step cap + stuck stop), all 12 tasks × 3 repeats per arm, arms run one after
another, concurrency 2. Grounder `grounder` = `sft-template` (pinned, checked before and after every
run), `base` screen descriptions, official BF16 Lightning, sandbox `68c7671`, `agent-v4`. Arm *on*
has 32 clean runs of 36. At 22:01 the night loop re-served `:8001` with the final adapter: two
runs in flight across the swap (star-landlord-message r3, pay-50 r3, both failed) are excluded,
and the last two jobs never started. None were rerun on a different grounder. The table compares
the same 32 (task, repeat) jobs in all three arms.

| Same 32 jobs | thinking off | thinking retry | thinking on |
| --- | --- | --- | --- |
| Verified success | 19/32 (59%) | 21/32 (66%) | **26/32 (81%)** |
| Claim matches verified | 20/32 | 23/32 | **28/32** |
| False success claims | 9 | 9 | **4** |
| No-change actions | 72 | 54 | **38** |
| 25-step cap hits / stuck stops | 6 / 0 | 2 / 0 | 0 / 1 |
| Mean steps | 10.4 | 9.0 | **5.9** |
| Plan (one Lightning call) p50 / p90 | 6.3 / 7.7 s | 5.9 / 16.7 s | 10.9 / 24.7 s |
| Wall-clock per task p50 / p90 | 65 / 220 s | 68 / 279 s | **53** / 293 s |
| Thinking tokens per run | 0 | 616 | 1,502 |

- **Significance (paired, same jobs):** on vs off, on won 8 jobs and off won 1: exact McNemar
  p = 0.039. On vs retry: 7 vs 2, p = 0.18. Retry vs off: 4 vs 2, p = 0.69. Full arms: off 20/36,
  retry 22/36, on 26/32.
- **Where thinking on wins:** the hardest tasks (off → on, same jobs). `max-usage-month` (canvas
  chart) 0/3 → 2/3, `star-landlord-message` (needs a scroll) 0/2 → 2/2, and
  `message-support-meter` (multi-step form) 0/2 → 2/2. In the full off arm both of the latter
  were 0/3.
- **Cost:** each planning call is about 1.7× slower, but runs take about half as many steps, so
  the median task finishes sooner. The p90 is about the same as retry.
- **Measured option, not the default: thinking on** verified 5 more of the same 32 jobs than
  retry (26 vs 21), with fewer false claims, at 10.9 s per planning call vs 5.9 s (median task
  53 s vs 68 s). **The demo recording uses thinking off** (decided Sat night): it's the
  Gym-measured configuration, in which FINAL passed all 5 scripted tasks 3/3. It also avoids
  25-45 s thinking pauses on camera (a fully local plan step takes ~4 s off vs ~13 s with
  thinking), and the 26/32 was measured with `sft-template`, not FINAL. `retry` stays the
  interactive default. Gym evals stay thinking-off, so the reported base-vs-tuned numbers stay
  comparable.

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

- **Done:** the Gym `web_agent` eval with `--num-repeats 3`, base vs final
  grounder: FINAL 17/36 (Sat 22:04–22:29) vs the Sat 18:45 base arm's 18/36, within
  noise. See "FINAL grounder rerun" above.

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
