# Results log

Numbers as they land. The newest row in each table is the most recent result.
Commands and configs are in [runs.md](runs.md). "Reportable" means official BF16
Lightning, the real sandbox, and grounder choices made on the dev split.

## Grounding (accuracy = predicted click point inside the target box)

| Model | Dev (969 rows) | SSv2-clean (1,244 rows) | Notes |
| --- | --- | --- | --- |
| Qwen3.5-2B base | 76.4% | **86.3%** (icon 78.1, text 92.5) | Kyle, `gui_grounding` Gym env |
| Pilot SFT (2k rows raw labels, 5 min) | **81.3%** (+5.0) | — | Every source improved; Linux +11.9, FineWeb +2.2. Answer length 37.6 → 16.7 tokens. |
| Gate D1 comparison (OS-Atlas sample n=500, pre-cleanup) | Qwen3-VL-2B 80.6 vs Qwen3.5-2B 79.8 | — | `eval_results/qwen3vl_vs_qwen35_osatlas500.csv`. Qwen3.5 +7.0 on desktop. |

## Agent (sandbox tasks, verified from sandbox state)

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
