# web_agent BLADE Analysis Report: base vs SFT click grounder (agent-v4)

Run `agent_v4_sft_template` (docs/runs.md, Sat 18:45). NeMo Gym 0.7.0, environment
`resources_servers/web_agent`, agent `web_agent_simple_agent`. Analysis date 2026-10-17.
Rollout ids are written `<arm>/<task_id>#k`, where `k` (0, 1, 2) is the order of that task's rows in the
arm's file. `sN` means the N-th tool call. In prose, "SFT" means the `grounder` arm.

## Executive Summary

- **Setup.** Each arm ran 12 sandbox tasks × 3 repeats on the same rows: the materialized inputs are
  byte-identical, including the cached first observation. Both arms used the same planner (official BF16
  Nemotron 3.5 Lightning, thinking off, temperature 1.0) and the same screen-description model (`base`).
  **Only the click grounder differs:**
  - `base` is Qwen3.5-2B with no adapter.
  - `grounder` is the `sft-template` LoRA on the same weights, served from the same vLLM process.
- **Headline.**
  - Verified passes: **18/36 → 22/36** (pass@1 0.50 → 0.61). Tasks passed at least once (pass@3):
    7/12 → 9/12.
  - `gym eval compare`: reward **+0.111, 95% CI [−0.06, +0.28]. This is not significant**: the
    rollout-level Fisher test gives p = 0.48, and the flips are 3 fail→pass against 1 pass→fail (sign test
    p = 0.63).
  - **Verifier gap.** The mark-all-read check doesn't guard notifications, and 2 SFT passes also cleared
    the bell's notifications (grounder/mark-all-read#0 s8–s9, #1 s11–s12). With a notifications-unchanged
    condition, SFT would be 20/36 (+2 over base) and mark-all-read SFT 1/3 (Recommendation 6).
- **What the SFT grounder measurably fixed.**
  - **Grounder output failures 10 → 0** (failed actions: `ok=false`, excluding invalid-argument and
    after-done calls). All 10 are cases where the base grounder answered in prose instead of a point
    ("could not parse a point"). This is the only non-latency compare metric whose CI excludes 0:
    [−0.52, −0.04]. (`summary_ms` also excludes 0, but latency is not interpretable here because another
    job shared the GPU.)
  - **Clicks now land on something:** "screen did not change" actions 83 → 45. That is not all progress.
    In stop-promotions the extra changes are wrong-row flips, and in high-contrast they are on/off
    oscillation that ended with the switch off.
  - **Clicks on small targets next to a selected option.** Clicks meant for the L text-size button that
    landed on it went from 0/18 to 2/6. The first "Accessibility tab" click from the Profile tab reached
    that tab 0/3 → 3/3 (base's later retries from the Notifications tab hit 2/2).
- **Only about 2–3 of the +4 passes are due to the grounder.**
  - text-size-large +2: grounder.
  - sms-billing-off +2: one genuine hit on the named row's switch and one lucky ambiguous click.
  - message-support-meter +1 and high-contrast −1: planner sampling variance. Within each arm, rollouts
    split on identical observations (grounder/high-contrast#0 vs #1 share their first 6 output items;
    base/message-support-meter#0 vs #2 share an identical s1). Across arms, high-contrast reaches the same
    state-less decision observation at different steps (SFT s3; base s6/s9, after the base grounder's
    Accessibility-tab miss), and message-support's s1 summary comes in two near-identical variants.
- **The largest drop-off is from "right page reached" (34/36 in both arms) to "target element hit" (22
  base / 27 SFT).** Most of that loss happens before the grounder is involved:
  - The shared screen summarizer drops what the planner needs. The notifications grid is cut off in 15/15
    first views of that tab (12/12 in sms-billing-off and stop-promotions), and the high-contrast on/off
    state is missing in 6/6 observations.
  - The planner never scrolls to off-screen targets: 0 scroll calls in 6/6 star-landlord rollouts.
  - The planner claims success falsely (8 base → 4 SFT) and keeps acting until the 12-step cap (12 → 10
    rollouts, 6 tasks), including after a toast had already reported the result (mark-all-read).
- **8 of 12 tasks give the same result in both arms** (5 always pass, 3 never pass). As run, this eval
  is limited by the planner and the observations, not by the grounder.
- **Next, ranked by impact:**
  1. Fix the observation pipeline: split run-on lines before clipping, force on/off on toggles, quote
     toasts verbatim, add a "more below" cue. This uses the same served model, so the parameter budget
     doesn't change.
  2. Add a done-gate and a toggle-oscillation guard in `agent/session.py`.
  3. Planner prompt agent-v5: scroll when the target isn't listed, name the row and column in grids, stop
     once the goal is reached.
  4. Grounder data: unselected options next to a selected one, planner-style phrasings, "not found"
     examples for absent targets, chart bars. Add pressure toward box centres in GRPO.

## Artifact Inventory

| Artifact (under `results/agent_v4_sft_template/`, gitignored, on Patrick's Mac) | Content |
|---|---|
| `agent_base.jsonl` | 36 rollouts, `grounder_model=base`, `summary_model=base`, `prompt_version=agent-v4` |
| `agent_grounder.jsonl` | 36 rollouts, `grounder_model=grounder` (sft-template), `summary_model=base` |
| `agent_*_materialized_inputs.jsonl` | 36 rows each, byte-identical across arms (`cmp`) |
| `agent_*_aggregate_metrics.json`, `*_repeat_level_metrics.json`, `*_agent_metrics.json`, `*_reward_profiling.jsonl` (12 rows) | Gym aggregates |
| `agent_*_run_info.txt` | Commands, served names, rows sha256, `reportable=yes` |
| `agent_*_failures.jsonl` | Empty: 0 dropped rollouts |
| `agent_compare/compare_report.{md,json}` | `gym eval compare` base vs candidate |
| `quality_summary.json`, `rollout_verdicts.jsonl` | **Grounder arm only:** the base arm's copy was overwritten (mtime 19:25:20, 1 s after `agent_grounder.jsonl`; prompt tokens 893,037 = grounder `mean/input_tokens` 24,806.58 × 36). 0 issues; 36/36 `unobserved` because no model-call captures were recorded, so model-call health is **not assessed** for either arm |

| Setting | Value |
|---|---|
| Rows | `resources_servers/web_agent/data/tasks_all.jsonl`, sha256 `d6e3778c…c507882`. 12 tasks from `sandbox/tasks.json`. Start screens `vlm:base`: the first observation is stored in the row, so it is identical across arms |
| Gym | `gym eval run --num-repeats 3 --concurrency 2`; `simple_agent` `max_steps: 12` (12 Lightning calls) |
| Planner | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, local, thinking off, temperature 1.0, top_p 0.95, max_output_tokens 1024 |
| Screen summary | served `base` (Qwen/Qwen3.5-2B), `SUMMARY_SAMPLING` temperature 0, frequency_penalty 0.3 |
| Grounder | `base` = Qwen/Qwen3.5-2B; `grounder` = `/home/ubuntu/ckpt/sft-template/adapter`. One vLLM process on :8001 |
| Sandbox | `http://127.0.0.1:8091`, commit 68c7671 (docs/runs.md) |
| Time | base 22:44:56–23:05:15Z, grounder 23:05:15–23:25:27Z |

**Coverage:** 36/36 rollouts per arm, 0 masked, 0 dropped, 0 VLM infrastructure errors, 0 stale initial
observations.

**Not available or not used:**

- Latency: another job shared the GPU during the candidate arm, so it is ignored.
- Ground-truth target boxes per click: agent rollouts record click points only. Whether a click hit is
  inferred from the next observation, toasts and URLs, plus box estimates read off
  `sandbox/screenshots/*.png`.
- Model-call health checks (unobserved).

**Note:** this run evaluates `sft-template`. The round-0 pick on dev is `sft-lightning` (docs/runs.md
Sat 19:38), which has **not** been run through this agent eval.

**Rollout ids.** Wherever `k` differs from `_ng_rollout_index`, the mapping is:

| Arm/task | k → `_ng_rollout_index` |
|---|---|
| base/mark-all-read, base/pay-50, grounder/mark-all-read, grounder/stop-promotions | #0 = 1, #1 = 0, #2 = 2 |
| grounder/text-size-large, grounder/sms-billing-off | #0 = 1, #1 = 2, #2 = 0 |

## Aggregate Results

Computed by `/tmp/t2-blade/blade_agent_report.py` (see Reproducibility Notes). The shared metrics match
the committed `scripts/agent_metrics.py` and docs/runs.md.

| Metric | base | SFT (`grounder`) | Δ |
|---|---:|---:|---:|
| Rollouts (coverage) | 36/36 | 36/36 | |
| Verified (reward = 1) | 18 | 22 | +4 |
| pass@1 | 0.500 | 0.611 | +0.111 |
| pass@3 (tasks with at least 1 pass) | 0.583 (7/12) | 0.750 (9/12) | +0.167 |
| Consistency (tasks passing 3/3) | 0.417 (5/12) | 0.417 (5/12) | 0 |
| Retry value (pass@3 − pass@1) | 0.083 | 0.139 | +0.056 |
| `done` called | 24 | 26 | +2 |
| Claimed success (`done(success=true)`) | 24 | 25 | +1 |
| Claim agrees with verifier | 26/36 | 31/36 | +5 |
| **False claims** (claimed, reward 0) | **8** | **4** | −4 |
| Unclaimed passes (reward 1, no success claim) | 2 | 1 | −1 |
| Claim precision (verified / claimed) | 16/24 (0.67) | 21/25 (0.84) | |
| Ended at the 12-step cap without `done` | 12 | 10 | −2 |
| Tool calls | 259 | 234 | −25 |
| Invalid calls (wrong or missing argument names; planner) | 10 | 6 | −4 |
| Tool-call validity | 96.1% | 97.4% | +1.3 pt |
| **Failed actions** (`n_failed_actions`: `ok=false`, excluding invalid-argument and after-done calls; all `ok=false` outputs: 22 / 8) | **10** | **0** | −10 |
| of which grounder parse failures (prose, no point) | 10 | 0 | −10 |
| **No-change actions** ("The screen did NOT change") | **83** | **45** | −38 |
| Scroll calls | 26 | 13 | −13 |
| Target descriptions with the "@" notation copied from the summary | 12 (all clicks) | 37 (35 clicks, 2 `type_text`) | +25 |
| Mean steps | 7.19 | 6.50 | −0.69 |

**Significance.** The interval tests below treat calls as independent, so read their p-values as
directional.

| Metric | Result |
|---|---|
| Reward | `gym eval compare` Welch CI over 3 repeat means per arm is [−0.06, +0.28]. Fisher on 18/36 vs 22/36 gives p = 0.48. Not significant |
| Failed actions | Compare Δ −0.278, CI [−0.52, −0.04], which excludes 0. All 10 sit in 4 base rollouts of 2 tasks (rollout-level Fisher p = 0.11): real, but narrow |
| No-change actions | 83/259 vs 45/234 calls, Fisher p = 0.001 |
| False claims | 8/36 vs 4/36, Fisher p = 0.34 |

**By tag** (compare report): settings 2/12 → 5/12, easy 9/12 → 11/12, medium 5/15 → 6/15, hard 4/9 →
5/9, messages 4/9 → 5/9. Unchanged: icon_only 6/12, scroll 0/6, canvas 3/6, usage 0/3, billing 9/9,
outages 3/3.

## Workflow Funnel

Each stage has one predicate per task (in the analysis script). Stages are nested; the two claim rows
are not.

- **Right page/tab:** some observation's URL is the target page. `/settings?tab=accessibility`,
  `/settings?tab=notifications`, `/billing`, `/messages`, `/messages/compose`, `/usage` or `/outages`,
  depending on the task.
- **Target element hit:** an action visibly acted on the requested element. The evidence is a toast or
  text naming it ("Text size set to Large", "SMS for Billing alerts turned off", "… for Offers and
  promotions turned off", "Downloaded", "Report an outage in Riverside", "$50"), a URL
  (`/messages/msg-landlord`), Customer Support chosen on compose, or "July 2026" plus "968" on /usage.
  - high-contrast and mark-all-read are exceptions, because the summarizer misses their toasts. For
    these, a hit is a click inside the box estimated from screenshots 09 and 06: the switch (x 1131–1229,
    y 285–325) with a screen change, or the check-double button (x 472–516, y 269–313).
- **Goal state reached (at any step):** the same as "target element hit" for one-click tasks. Otherwise:
  - stop-promotions: all three promotions channels off at the same time, tracked from toasts.
  - pay-50: "Payment of $50" or "$92.37" shown.
  - message-support-meter: URL `/messages?tab=sent`.
  - report-outage-riverside: OUT-1001 or "your reports" shown.
  - star-landlord-message: "Starred" shown while on msg-landlord.
- **Goal held at end:** none of the check's target conditions failed.
- **Verified:** reward 1, which also requires that no collateral condition failed.

| Stage | base | SFT | Lost at this stage (base / SFT) |
|---|---:|---:|---|
| Start | 36 | 36 | |
| Right page/tab reached | 34 | 34 | max-usage-month 2 / 2 (went to /billing) |
| **Target element hit** | **22** | **27** | **12 / 7**: text-size 3/1, sms-billing 3/1, star-landlord 3/3, message-support 2/0, max-usage 1/0, stop-promotions 0/2 |
| Goal state reached (any step) | 19 | 25 | 3 / 2: stop-promotions 3/1 (one channel only), message-support 0/1 (empty body) |
| Goal held at end | 18 | 22 | 1 / 3: high-contrast 1/2 (planner clicked it off again), max-usage 0/1 (July overwritten) |
| Verified | 18 | 22 | 0 / 0 |
| `done(success=true)` claimed | 24 | 25 | |
| Claimed **and** verified | 16 | 21 | |

**How to read it:**

- The biggest drop is at **target element hit**.
- SFT recovers 5 rollouts there: text-size +2, sms +2, max-usage +1, message-support +2 (planner
  variance), stop-promotions −2.
- But SFT loses 3 rollouts after reaching the goal, against 1 for base. In each, the planner undid its
  own success.
  - high-contrast (1 base / 2 SFT): both grounders carry out the planner's re-click (base/high-contrast#1
    s7, grounder/high-contrast#1 s4, #2 s4 all produce "High contrast turned off"). The arm difference is
    the planner's re-click rate, 1/3 vs 2/3 at n = 3.
  - max-usage (0 / 1): here more accurate clicks do let the planner's mistake take effect.
    grounder/max-usage-month#0 selected July at s7 and overwrote it at s8–s11; base never selected any bar.
    The same mechanism drives the stop-promotions collateral damage (Mode 4).
- 8/24 base claims and 4/25 SFT claims are false.

## Task Outcome Buckets

| Task (tags) | base verified / claimed | SFT verified / claimed | base bucket | SFT bucket | Primary label | What triggers it |
|---|---|---|---|---|---|---|
| text-size-large (easy, settings) | 0/3 / 1 | 2/3 / 2 | never | sometimes | **KG** (base) → **UK** (SFT) | Grounder misses an unselected segment next to the selected one; SFT hits only with a bare "L" |
| download-september-statement (easy, billing, icon_only) | 3/3 / 3 | 3/3 / 3 | always | always | n/a | Saturated: the summary names the row's download icon |
| mark-all-read (easy, messages, icon_only) | 3/3 / 1 | 3/3 / 2 | always | always | n/a for reward (BI in stopping; TI: 2 SFT passes also cleared notifications) | Done at s1. Mainly a planner stopping failure: in 3/6 the summarizer later reports "Marked 4 messages as read" or "All messages are already read" and the planner keeps acting. Triggers: the step-1 toast is missed in 6/6, and s2 says "did NOT change" next to the toast (grounder#1 s2) |
| turn-on-autopay (easy, billing) | 3/3 / 3 | 3/3 / 3 | always | always | n/a | Saturated |
| high-contrast (medium, settings) | 2/3 / 2 | 1/3 / 1 | sometimes | sometimes | **BI** | Summary drops the on/off state in 6/6; planner re-clicks in 3/6 |
| sms-billing-off (medium, settings) | 0/3 / 1 | 2/3 / 2 | never | sometimes | **BI** | Notifications grid cut off; targets name no row |
| max-usage-month (medium, usage, canvas, icon_only) | 0/3 / 0 | 0/3 / 0 | never | never | **BI** | Collapsed nav, so 4/6 went to /billing; goal reached then overwritten |
| report-outage-riverside (medium, outages, canvas) | 3/3 / 3 | 3/3 / 3 | always | always | n/a | District label inside the clickable region |
| star-landlord-message (medium, messages, icon_only, scroll) | 0/3 / 3 | 0/3 / 3 | never | never | **BI** | Target below the fold, 0 scrolls; 6/6 false claims |
| pay-50 (hard, billing) | 3/3 / 3 | 3/3 / 3 | always | always | n/a | Deterministic form |
| message-support-meter (hard, messages) | 1/3 / 3 | 2/3 / 3 | sometimes | sometimes | **BI** | Recipient skipped; "sent" claimed while still on /messages/compose |
| stop-promotions (hard, settings, scroll) | 0/3 / 1 | 0/3 / 0 | never | never | **BI** | Cut-off grid, target row below the fold, targets name no row |

**Bucket counts:** base 5 always / 2 sometimes / 5 never; SFT 5 always / 4 sometimes / 3 never; 0
missing.

**No failing task gets a TI, IR or DA label.** Coverage is 36/36 with 0 infrastructure errors. The
checks were checked against the rollouts (see Never-Pass Deep Dives). One verifier gap inflates passes
rather than causing failures: the mark-all-read check doesn't guard notifications, so 2 SFT passes that
also marked every notification read count as verified (Recommendation 6).

## Dominant Failure Modes

| # | Mode | Failing or false-claim rollouts (base / SFT) | Tasks | Label |
|---|---|---|---|---|
| 1 | Screen-summary observation gaps (shared `base` summarizer) | 14 / 13 rollouts affected, of which 9 / 7 fail. Counted: grid cut off 6/6, high-contrast state missing 3/3, step-1 toast missed 3/3, failed-Send alert omitted 2/1 | sms, stop-promotions, high-contrast, message-support, mark-all-read | BI (observation trigger) |
| 2 | False success claims | 8 / 4 | star-landlord, message-support, text-size, sms, stop-promotions | BI |
| 3 | Off-screen targets (late or no scroll) | 6 / 6 failures: never scrolled 3 / 4 (star-landlord 3/3, plus grounder/stop-promotions#0); scrolled too late or after wrong flips 3 / 2 (stop-promotions) | star-landlord, stop-promotions | BI |
| 4 | Dense 8×3 switch grids | 6 / 4 failures | sms-billing-off, stop-promotions | BI |
| 5 | Thrashing until the 12-step cap | 12 / 10 capped | 6 tasks (text-size, mark-all-read, high-contrast, sms, max-usage, stop-promotions) | BI |
| 6 | Misses on small options next to a selected one | 3 / 1 failures | text-size-large | KG / UK |
| 7 | Canvas bar chart | 3 / 3 failures | max-usage-month | BI |
| 8 | The high-contrast "regression" | 1 / 2 failures | high-contrast | BI (variance) |
| 9 | Grounder output format and no "not found" answer | Parse failures 10 / 0. Clicks on absent targets: 4 in SFT, and base does it too | max-usage, stop-promotions, star-landlord, message-support, mark-all-read | KG (grounder) |

### 1. Observation gaps from the shared screen summarizer

- **Cut-off grid.** The first observation of `/settings?tab=notifications` is identical in all 15
  rollouts that opened that tab: 12/12 of sms-billing-off and stop-promotions, plus base/high-contrast#0–#2
  s2 (the Accessibility-tab misclick).
  - The summarizer wrote MAIN as one run-on line. `clip_summary` (`agent/observation.py`,
    `MAX_LINE_CHARS = 240`) then cut it at `… - row: Billing alerts; New statemen…`. So the Email/SMS/Push
    columns and every other row are missing.
  - The same line lists a `push notification toggle switch @ right`, which is actually the contact card's
    Push item.
  - The first action on the tab follows from this: "Billing alerts row" in 6/6 sms rollouts (no change),
    and "push notification toggle switch" in 6/6 stop-promotions rollouts.
- **Toggle state missing.** After high contrast turns on, the next observation has no STATE line, no
  on/off marker and `OVERLAYS: none` in 6/6 rollouts (e.g. base/high-contrast#0 s6,
  grounder/high-contrast#0 s3). Yet `sandbox/state.py` returns "High contrast turned on". 3/6 planners
  then clicked it off again.
- **Missed toasts and alerts.**
  - mark-all-read: the step-1 toast ("Marked 4 messages as read") is missing from 6/6 step-1
    observations. In 3/6 a later observation does report the result, and the planner keeps acting
    anyway (a stopping failure, Mode 5):
    - grounder/mark-all-read#1: "Marked 4 messages as read" at s2 (next to "The screen did NOT change")
      and s4, then "All messages are already read" at s7–s8. No `done` through s12.
    - base/mark-all-read#2: "All messages are already read" at s5–s6. At s6 it clicks that banner, then
      opens compose.
    - grounder/mark-all-read#0: "already read" at s4, s6 and s7, then it uses the bell's "Mark all as
      read".
    - Only base/mark-all-read#1 never gets a readable toast (degenerate overlays at s4, s8, s9).
  - The overlay degenerated into `banner/tooltip/indicator/…` in 8 observations across 4 rollouts
    (base/mark-all-read#1 s4, s8, s9; base#2 s4; grounder#0 s3, s5; grounder#1 s5, s6).
  - Only 1 of 5 observations after an empty or failed Send mentions the validation text
    (base/message-support-meter#2 s2).

### 2. False success claims (8 base / 4 SFT)

| Task | base | SFT | Pattern | Example |
|---|---:|---:|---|---|
| star-landlord-message | 3 | 3 | The "Starred" toast is accepted without checking whose message was starred; the planner invents the identity | base/star-landlord-message#0 s3 "Starred the message from the landlord (Acme Outage Team)"; grounder/star-landlord-message#1 final "landlord (Jordan Lee)" |
| message-support-meter | 2 | 1 | "Sent" claimed while the URL is still /messages/compose | base#1 s4, base#2 s5, grounder#2 s5 |
| text-size-large | 1 | 0 | Claim contradicts the toast "Text size set to Small" | base/text-size-large#0 s8–s9 |
| sms-billing-off | 1 | 0 | Claim quotes a toast for the wrong row ("SMS for Payment confirmations turned off") | base/sms-billing-off#1 s8 |
| stop-promotions | 1 | 0 | Only 1 of 3 channels turned off | base/stop-promotions#1 s11 |

The metric doesn't count one more misreport. grounder/high-contrast#2 called `done(success=false)` at s9,
but its final message to the user says "The high-contrast mode is now enabled" (the state was off).

### 3. Off-screen (scroll-dependent) targets: tag `scroll` 0/6 in both arms

- **star-landlord-message.** The landlord row is row 11 of 14 in the inbox.
  - The initial observation, the same in 6/6, lists 8 rows. The last is marked `(partial)`, next to
    "14 messages", and "landlord" doesn't appear.
  - The planner made 0 scroll calls in 6/6, though it scrolled 26 times in other base tasks.
  - In 4/6 it handed the absent target to the grounder ("the message row from the landlord"). In 2/6 it
    picked the Hillcrest row itself (base#1, grounder#0).
  - One page scroll shows `row: Dana Whitfield (Landlord)` (grounder/mark-all-read#0 s6).
- **stop-promotions.** The Offers and promotions row is below the fold.
  - Base scrolled in 3/3 rollouts (s5, s9, s9), but only after repeated no-change notes.
  - SFT scrolled in time only in grounder/stop-promotions#1 (s8); #2 made a small scroll at s12, and #0
    never scrolled.

### 4. Dense switch grids (8 rows × Email/SMS/Push)

- Only 2 of 12 rollouts passed, both SFT sms-billing-off.
- Targets that name no row resolve to the first visible grid rows (Billing alerts or Payment
  confirmations).
  - Payment confirmations: e.g. base/sms-billing-off#1 s6 (1011,587); grounder/stop-promotions#0 s4, s6,
    #1 s3 (1149,590), #2 s5.
  - Billing alerts, about as often under SFT: grounder/stop-promotions#0 s5 (1149,517), s9 (896,518),
    s10 (1149,518); #1 s4 (909,517); #2 s4 (1149,517).
  - Or the SMS column header: (1019,462), no change, 5× (grounder/stop-promotions#1 s5–s7, #2 s10–s11).
- Every flip produces a toast naming the row and channel. **No rollout deliberately undid a wrong-row
  flip after reading its toast.** The two state reversals were repeat clicks the planner took as
  progress:
  - base/sms-billing-off#1: s6 "SMS for Payment confirmations turned on", s7 at the same point "…turned
    off". At s8 it claims success, quoting that toast.
  - grounder/stop-promotions#0: s5 "Push for Billing alerts turned on" is reversed at s10; s9 "Email for
    Billing alerts turned off" is reversed at s11.
- Collateral damage grew under SFT:
  - stop-promotions failed check conditions went from 5 (base) to 14 (SFT). Conditions outside the
    promotions row went from 2 to 11.
  - grounder/sms-billing-off#2 failed 7 of 12 conditions. Its s4 `type_text` on a non-input sent a space
    that scrolled the page; the planner then "turned off SMS" in 5 visible rows that weren't Billing.

### 5. Thrashing until the 12-step cap

- 12 base / 10 SFT rollouts, from the same 6 tasks, ended at the cap without calling `done`. That
  includes 2/1 mark-all-read passes that never claimed success; 2 of those 3 had seen a result toast
  (Mode 1).
- Patterns:
  - Rephrasing the same missed click: base/text-size-large#2 made 8 L-intended clicks, all no-change.
  - Flipping a switch back and forth: grounder/high-contrast#1 made 10 toggle clicks.
  - Sweeping bars or statements: grounder/max-usage-month#0 s8–s12, grounder/max-usage-month#2 s9–s12.
- Planner schema errors: 10 base / 6 SFT (`target_profile`, `target_statement`, `target_view`,
  `target_message`, missing `key`/`text`).
- The planner copied the "@" notation into 12 / 37 target descriptions (base: 12 clicks; SFT: 35 clicks
  and 2 `type_text`), despite the agent-v4 rule against it.

### 6. Grounding misses on small options next to a selected one

- **text-size-large.** Each S/M/L segment is about 52 px wide. In screenshot 09, L spans x ≈ 1166–1218,
  centre ≈ 1192.
  - **Base: 0/18 L-intended clicks hit L.** 17 landed on the already-selected M at x 1126–1152: 16 with
    no change, and base/text-size-large#0 s2 (1152,438) only re-selected Medium (toast "Text size set to
    Medium"). 1 hit S: base/text-size-large#0 s8 (1078,438).
  - **SFT: 2/6.** A bare "L" went to (1167,439) and (1167,438), 1 px inside L's left edge
    (grounder/text-size-large#0 s2, #1 s2). The quoted `text size options "L"` went to x 1152–1162, still
    inside M: 4/4 misses (grounder/text-size-large#2 s2, s6, s8, s9).
- **Same description and screen, different grounder:**
  - "Accessibility tab", first click from the Profile tab: base (474,206) opened the Notifications tab
    3/3 (base/high-contrast#0–#2 s2). SFT (556,206) opened Accessibility 3/3. Base's later "Accessibility
    tab" clicks from the Notifications tab hit 2/2 (#0 s5, #2 s8, both (563–564,206)); counting the bare
    "Accessibility" targets too, base reached the tab 3/7.
  - "Notifications tab at top-right": base (1139,26) opened /help (base/sms-billing-off#2 s2). SFT
    (425,208) was correct (grounder/sms-billing-off#1 s2).
- **High-contrast switch:** 17/17 SFT clicks on the Accessibility tab changed the screen. Base managed
  4/9; its 5 misses were at x 1116–1126 in base/high-contrast#1 s8–s12.

### 7. Canvas targets

- **report-outage-riverside, 6/6.** The district name is drawn inside its clickable polygon. The summary
  lists `Riverside (district label)`, and the dialog title confirms the district.
- **max-usage-month, 0/6.** Four separate problems:
  - (a) The task starts with the nav collapsed. In 4/6 rollouts the planner chose "View statements",
    went to /billing and never reached /usage (base#1, base#2, grounder#1, grounder#2).
  - (b) Month labels sit below the bars' clickable area: base/max-usage-month#0 s4 clicked "Oct 2025" at
    (371,702), with no change.
  - (c) The base grounder answered the targets "Sep ~710 kWh" (base#0 s9) and "Jul ~950 kWh" (s12) in
    prose. The s9 target is also a planner error: the s8 observation had just listed "Jul ~950 kWh" as
    the largest bar.
  - (d) grounder/max-usage-month#0 selected July at s7, and the page showed "968 kWh". The planner kept
    clicking bars and ended on May: the check got `'2026-05'`.

### 8. The high-contrast "regression" (2/3 → 1/3) is planner variance

- All 6 rollouts turn high contrast on with their first switch click on the Accessibility tab.
- The next observation has no state in 6/6. With the planner at temperature 1.0, 3 rollouts call `done`
  (pass) and 3 click the switch again (fail: 1 base, 2 SFT).
- **Neither action is supported by the observation.** The 3 passes (base/high-contrast#0 s6 → s7,
  base#2 s9 → s10, grounder#0 s3 → s4) are unverified claims that happened to be right: they were made on
  an observation with no STATE line, no on/off marker and `OVERLAYS: none`. The agent-v4 rule says
  "Before done(success=true), check that the latest observation (STATE, a toast or the page) shows exactly
  the result the user asked for". So the "claim agrees with verifier" count includes 3 lucky claims here,
  and the main fix for this task is Recommendation 1(b) (on/off on every toggle plus a STATE line), with
  the oscillation guard (Recommendation 2(b)) as a backstop.
- grounder/high-contrast#0 and #1 have **identical first 6 output items** and diverge at s4: `done`
  versus another click.
- Every failure fails only `settings.accessibility.high_contrast: expected True, got False`.
- SFT actually made the passing path shorter: 4 steps against 7 and 10 for base. Base had to recover from
  the Notifications-tab misclick.

### 9. Grounder output format and no "not found" answer

- **Base output-format failures.** Base answered in prose 10 times ("could not parse a point from: 'To
  accomplish the instruction …'"): base/max-usage-month#0 (2), #1 (4), base/stop-promotions#0 (2), #2 (2).
  SFT: 0.
- **SFT never answers "not found".** It always returns a point, so absent targets become confident wrong
  clicks:
  - grounder/star-landlord-message#1 s1 and #2 s1 clicked the bottom partial row, opening msg-rebate.
  - grounder/message-support-meter#2 s2: "the Message us button", typed on the compose page, became a
    click on Send (349,651), which submitted an empty form.
  - grounder/mark-all-read#1 s9 opened msg-tree-trimming. On that page, which has no checkmark button,
    "the button: checkmark icon @ right" went to the avatar (s10), the bell (s11) and the dropdown's
    "Mark all as read" (s12).
  - Base also clicks row 1 for absent targets (base/star-landlord-message#0, #2).

## Sometimes-Pass Deep Dives

### text-size-large: base 0/3 → SFT 2/3 (KG → UK)

- **Pass** (grounder/text-size-large#0, #1), 3 steps each:
  1. "Text size options @ right" → (1137,439), which hits M.
  2. "L" → (1167,439) / (1167,438), giving "Text size set to Large".
  3. `done`.
- **Fail** (grounder/text-size-large#2):
  - s1 targets "row: Text size options @ right (M selected)", copied straight from the summary.
  - It then uses quoted descriptions, which the grounder places on M four times.
  - It hits the 12-step cap with M still selected.
- **Base** never hit L. base/text-size-large#0 hit S at s8 and then claimed Large.
- **Real or lucky?** Real but fragile. The SFT point is 1 px inside the box and about 25 px left of the
  centre. Only the bare "L" phrasing hits.

### high-contrast: base 2/3, SFT 1/3 (BI)

- The pass/fail split is the planner's next sampled action on a state-less observation (Mode 8).
- Failures look different by arm:
  - base/high-contrast#1 ended stuck off after 5 misses.
  - grounder/high-contrast#1 and #2 kept flipping the switch, because every click landed.
- **Attribution:**
  - grounder/high-contrast#1 and #2 fail on the planner's re-click alone (BI).
  - base/high-contrast#1 is mixed. The planner's re-click at s7 (1152,306) turned it off ("High contrast
    turned off"; BI). Then 5 base-grounder misses at s8–s12, at x 1116–1126 outside the switch box, all
    "did NOT change", blocked the recovery (KG).

### sms-billing-off: base 0/3 → SFT 2/3 (BI)

- **Genuine pass** (grounder/sms-billing-off#1):
  - After a no-change click at s3, the summarizer happened to return per-row states:
    `- billing alerts row: toggle switch On; toggle switch On; toggle switch Off @ right`.
  - The planner named the row and channel ("toggle switch Off for SMS billing alerts"), the grounder hit
    (1027,518), and the toast read "SMS for Billing alerts turned off".
- **Lucky pass** (grounder/sms-billing-off#0):
  - s5 targeted "row Billing alerts", which names no channel. It landed on the SMS switch at (1024,518),
    but could as easily have hit Email or Push.
  - Base's version of the same intent, "row: Billing alerts" (base/sms-billing-off#0 s11), clicked the row
    centre (640,514) with no change.
- **SFT fail** (grounder/sms-billing-off#2): `type_text` on a row (s4) scrolled the page, and the planner
  then flipped SMS in 5 wrong rows.
- **Base fails:**
  - base#0 never flipped a switch; it clicked the SMS footer note 4×.
  - base#1 turned off Billing Email (s4), flipped Payment confirmations SMS on and back off (s6–s7), then
    claimed success falsely; only the billing row fails the check.
  - base#2 went to /help first (s2), which cost 2 steps. It was back on the Notifications tab at s4 with
    8 steps left, then failed the same way as base#0/#1: a row-less "the SMS toggle switch" on the cut-off
    grid flipped switches in the Payment confirmations row (s9, s11).
- **Caveat:** grounder#0 s3 and grounder#1 s3 clicked almost the same point, (763,515) and (763,518), and
  got different summaries: cut off in one, per-row in the other. grounder/sms-billing-off#2 s3 clicked
  exactly (763,518), the same point as grounder#1 s3, and got the cut-off line again, identical to its
  s2. With the identical click point (so the same hover position), the genuine pass depended on
  screenshot or summarizer nondeterminism, not on where the click landed. **Unverified:** which of the
  two (e.g. render timing versus vLLM batch nondeterminism at temperature 0).

### message-support-meter: base 1/3 → SFT 2/3 (BI)

- All 6 rollouts open compose at s1 with the "Message us" card: base (828,598), SFT (847–870, 619–624).
- The s1 summary comes in two variants ("button: Send (green)" vs "button: Send (green background)"),
  so observations are not byte-identical across rollouts. Where they are, rollouts still split:
  base/message-support-meter#0 and #2 share an identical s1 call and output, then diverge at s2.
- The 3 passes open the To picker at s2: base#0, grounder#0, grounder#1.
- The 3 failures type or send first. base#1 and base#2 diverge at s2. grounder#2 diverges already at s1,
  with the target "the Message us button" (the others used "Message us (Write to our team)"), which it
  reuses for `type_text` at s2.
- The failures:
  - base#1 never sets a recipient.
  - base#2 presses Send on the empty form, types the body, then presses Send again.
  - grounder#2 aims `type_text` at a target from the previous page ("the Message us button"). The
    grounder puts it on Send. Focus then moves to the To button (`sandbox/static/messages.js:40`), so the
    body stays empty.
- All 3 failures call `done(success=true)` while still on `/messages/compose`.
- 1/3 vs 2/3 gives Fisher p = 1.0, so this is **not grounder evidence**.

## Never-Pass Deep Dives

### star-landlord-message: 0/3 and 0/3 (BI), 6/6 false claims

1. The landlord message is below the fold (Mode 3), and the planner makes 0 scroll calls.
2. The planner clicks a row that isn't the landlord's:
   - Base opens msg-outage-hillcrest 3/3.
   - SFT opens msg-outage-hillcrest 1/3 and msg-rebate 2/3.
3. The opened page shows the wrong sender ("Acme Outage Team", "Acme Energy Insights").
4. The planner clicks Star anyway; both grounders hit it ((1166,194) and (1170–1174,195–197)).
5. It calls `done(success=true)` on "Starred".

**Check:** valid. The target is one scroll away, and `steps_min` is 2. **Grounder signal:** none.

### max-usage-month: 0/3 and 0/3 (BI)

- **Navigation.** The start page is `/` with the nav collapsed, and the first observation says "side menu:
  none visible".
  - 4/6 rollouts never reach /usage and spend their steps on /billing statements (base#1, base#2,
    grounder#1, grounder#2).
  - 3 of those never open the menu icon; base/max-usage-month#2 opens it only at s12.
- **Chart reading.** Both rollouts that reached /usage got enough from the summarizer to answer:
  - base#0 s8 estimated per-bar values: "Jul ~950 kWh; Aug ~910 kWh" (true values 968 and 902).
  - grounder#0 s6 marked "Jul 2026 (highest)".
- **Failure.**
  - base#0 is mixed. The grounder couldn't click a bar (axis labels s4/s6, empty chart area s5/s10,
    prose answers s9/s12: KG), and the planner asked for the wrong month at s9 ("Sep ~710 kWh" right
    after an observation listing "Jul ~950 kWh" as the largest: BI).
  - SFT (grounder#0) reached the goal and then overwrote it (Mode 7d).
- All 6 rollouts end at exactly 12 steps.
- **Check:** `usage.selected_month == 2026-07`, which is sound.

### stop-promotions: 0/3 and 0/3 (BI)

A pass needs 7 calls: gear → Notifications → scroll → Email, SMS and Push off in the promotions row →
`done`.

- **Base:**
  - The planner chases the invented "push notification toggle switch", and the base grounder sends those
    clicks to empty space at (1139–1176, ~378), with no change.
  - The no-change notes eventually push all 3 rollouts to scroll. Each then turns off **one** channel:
    base#0 Email (937,512), base#1 Email (923,513), base#2 SMS (1024,512).
  - base#1 claims success after that single channel. base#0 drifts into rows nobody asked about.
  - base#0 and base#2 also hit 2 parse failures each.
- **SFT:**
  - The same vague targets now hit real switches in other rows, e.g. Push for Payment confirmations at
    (1149,590). The first such click usually lands on a real switch (#1 s3, #2 s3, #0 s4, after one
    no-change at #0 s3), so the early no-change nudge that base got is missing. Later no-change notes do
    appear: (1019,462) 3× at #1 s5–s7, which pushed grounder#1 to scroll at s8, and 2× at #2 s10–s11.
  - grounder/stop-promotions#1 scrolls at s8 and turns off promotions Email. Its next observation shows
    the row's SMS and Push still On, yet it moves on to Surveys and Payment confirmations.
- **Check:** sound. It guards all 8 rows.

The base-only never-pass tasks (text-size-large, sms-billing-off) are covered above.

## Cross-Model Comparison

| Task | base → SFT | Attribution | Evidence |
|---|---|---|---|
| text-size-large | 0/3 → 2/3 | **Grounder**, fragile | L hits 0/18 → 2/6; SFT points 1 px inside the box |
| sms-billing-off | 0/3 → 2/3 | **Partly grounder**: 1 genuine row+channel hit, 1 lucky. base#2's help-icon misclick cost 2 steps, but it failed on the shared row-less-target / cut-off-grid mechanism | grounder#1 s4 (1027,518); grounder#0 s5 |
| message-support-meter | 1/3 → 2/3 | Planner variance | Rollouts split at s1–s2 on near-identical observations; within base, #0 vs #2 split at s2 on an identical s1 |
| high-contrast | 2/3 → 1/3 | Planner variance | Identical first 6 output items, then divergence (grounder#0 vs #1) |
| stop-promotions | 0/3 → 0/3; failed conditions 5 → 14 | **SFT amplifies vague planner targets** | Same text: base (1147,378) no change vs SFT (1149,590) flips the wrong row |
| max-usage-month | 0/3 → 0/3 | Navigation and stopping, not grounding | SFT selected July once (grounder#0 s7); parse failures 6 → 0 |
| star-landlord-message | 0/3 → 0/3 | Planner never scrolls | 0 scrolls |
| 5 always-pass tasks | 15/15 → 15/15 | None (saturated) | SFT clicks land closer to centres: download (1208,546) vs (1198,543), centre ≈ (1207,544); autopay (1167,230) vs (1149,229), centre ≈ (1161,230) |

- **Fixed:**
  - Prose output: 10 failed actions → 0.
  - Small targets next to a selected option: text-size segments, settings tabs.
  - Clicks now land on something: no-change actions 83 → 45 (text-size 24 → 9, sms 13 → 6). This is not
    the same as progress. stop-promotions 14 → 7 is the same change as the collateral increase below
    (wrong-row flips). high-contrast 8 → 0 includes 16 SFT flips in grounder/high-contrast#1–#2 that ended
    with the switch off.
- **Not fixed:** anything blocked before grounding (off-screen targets, collapsed nav, cut-off grid),
  the planner's false claims, and its own undoing of a reached goal.
- **Broke or worsened:**
  - Collateral changes on vague targets (stop-promotions failed conditions 5 → 14).
  - No "not found" answer, so absent targets become wrong clicks (grounder/message-support-meter#2 s2;
    grounder/mark-all-read#1 s11–s12, which cleared the bell's notifications).
  - Unrequested notifications change in mark-all-read: 2 SFT passes, 0 base. The check doesn't see it
    (Recommendation 6).
  - The high-contrast drop is variance, not a regression.
- **Statistics:** reward +0.111 [−0.06, +0.28] is not significant at n = 36. The failed-action reduction
  is significant by `gym eval compare` but concentrated in 4 base rollouts.

## Recommendations

Ranked by expected impact on this eval (failing or false-claim rollouts addressed).

| Rank | Change | Where | Label | Targets in this run |
|---|---|---|---|---|
| 1 | **Observation pipeline.** (a) Split a run-on `MAIN: - a - b - c` into lines before the 240-char clip, and add a grid example to `SUMMARY_PROMPT` (one line per row with per-column on/off). (b) Every toggle line carries on/off and a STATE line is required; if either is missing, send one targeted re-query to the **same served model**. (c) OVERLAYS quotes toasts and inline alerts verbatim, with a repetition guard for `banner/tooltip/…` loops. (d) A "more below" cue (summary line, or browser scroll metrics like URL/title). (e) Per-bar values on charts. (f) A/B `summary_model=grounder` | `agent/observation.py`, `common/grounding.py` | BI | sms + stop-promotions (10 fails; grid cut off in 12/12); high-contrast (3 fails, plus the 3 passes that were unverified claims; 1(b) is the main fix there); message-support (3 false claims); star-landlord (6); mark-all-read (step-1 toast missed in 6/6; of the 3 unclaimed passes, only base#1 never saw a readable toast, the other 2 are a stopping failure, Rec 3) |
| 2 | **Agent-loop guards** in `agent/session.py`, so both Gym and the interactive agent get them. (a) Done-gate: on `done(success=true)`, return one non-final warning when a submit-type click left URL and title unchanged with no toast, when the latest toast or STATE contradicts the request, or when the request's key noun (e.g. "landlord") is absent from the last two observations. (b) A-B-A oscillation note for switches. (c) `type_text` sends no keys when its click lands on a non-input. (d) Strip `@ …`, `row:` and `(selected)` from `target_description` before grounding | `agent/session.py` | BI | 12 false claims (8/4); high-contrast oscillation (3); grounder/sms-billing-off#2; 49 target descriptions with "@" copied (47 clicks, 2 `type_text`) |
| 3 | **Planner prompt agent-v5.** If the named item isn't in the description, scroll (a `(partial)` row or a count above the rows listed means more below). In grids, name row and column. Read each toast and undo wrong-row flips. For multi-part requests, keep a checklist and call `done` only when every part is confirmed. Stop once the requested item is selected, and re-select it after comparing. If the nav is hidden, open the menu first. Status text ("All read") is a result, not a button, and a toast that states the result ("Marked 4 messages as read", "already read") means stop and call `done`. A switch flips on every click that changes the screen | `agent/prompts.py` (rerun **both** arms: a prompt change invalidates `BASELINE=` reuse) | BI | star-landlord (6), stop-promotions (6), sms (4), max-usage (6), mark-all-read stopping failure (2 of 3 unclaimed passes ignored a result toast; grounder/mark-all-read#0 went on to the bell) |
| 4 | **Grounder data and GRPO** (stream A). Public sources only, never sandbox screens (they are the agent eval), with the usual SSv2 dedup. Add: segmented controls, button groups and pagination where the target is an unselected option next to a selected one, phrased the way the planner phrases targets (bare "L", quoted, "@ right"); "not found" examples for absent targets (`session.py` already turns a locate error into a failed action the planner sees); chart bar bodies (box = the bar, not the axis label); settings grids described by row and column. In GRPO, the binary point-in-bbox reward exerts no pull toward centres (1167 vs an edge at 1166). Add an edge-margin dev metric, or deterministic centre-distance shaping (no judge model) | training data, `gui_grounding` verifier | KG / UK | text-size (4 fails); absent-target clicks (star-landlord 4/6, message-support#2); max-usage bars |
| 5 | **Eval design.** Use ≥ 8–10 repeats on sometimes-pass tasks before making per-task claims. 8/12 tasks don't separate grounders, so add variants: rows of near-identical icons, a download below the fold, switches the summary doesn't pre-name. Keep Gym `max_steps: 12` for comparability. Since Sat 19:30 the interactive agent uses 25 steps plus a "stuck" stop (`agent/progress.py`), which `simple_agent` doesn't run, so Gym numbers don't reflect it. Optionally report max-usage-month at 25 steps as a side metric | `sandbox/tasks.json`, `scripts/agent_eval.sh` | — | Attribution noise (high-contrast, message-support) |
| 6 | **Verifier.** mark-all-read accepts an unrequested notifications change (`sandbox/state.py:_notifications_mark_read` sets every notification to read). grounder/mark-all-read#0 s8–s9 chose the bell's "Mark all as read" deliberately. grounder/mark-all-read#1 s11–s12 aimed "the button: checkmark icon @ right" at a message page that has no such button; the grounder put it on the bell (1039,21), then on "Mark all as read" (934,96), toast "Notifications marked as read". No base rollout opened the bell. Add a notifications-unchanged condition and rerun both arms: SFT would drop from 22/36 to 20/36 and mark-all-read SFT from 3/3 to 1/3 (base unchanged at 18/36). The other checks were reviewed against the rollouts and look sound | `sandbox/tasks.json` | TI | 2 rollouts (grounder/mark-all-read#0 s8–s9, grounder/mark-all-read#1 s11–s12) |

## Reproducibility Notes

**Eval** (docs/runs.md, Sat 18:45; run from the repo root):

```bash
AGENT_SANDBOX_URL=http://127.0.0.1:8091 HEAD_PORT=21017 TASKS=resources_servers/web_agent/data/tasks_all.jsonl \
  REPEATS=3 CONCURRENCY=2 OUT_DIR=results/agent_v4_sft_template scripts/agent_eval.sh
```

Per arm, this ran (from `agent_*_run_info.txt`):

```bash
gym env start --config resources_servers/web_agent/configs/web_agent_local.yaml \
  ++web_agent.resources_servers.web_agent.grounder_model=<base|grounder> ++web_agent.resources_servers.web_agent.summary_model=base \
  ++web_agent.resources_servers.web_agent.sandbox_url=http://127.0.0.1:8091 ++web_agent.resources_servers.web_agent.vlm_base_url=http://127.0.0.1:8001/v1 \
  ++web_agent.resources_servers.web_agent.model_family=qwen3_5 ++head_server.port=21017 ++server_spinup_timeout_seconds=1800 ++skip_venv_if_present=true
gym eval run --no-serve --agent web_agent_simple_agent --input resources_servers/web_agent/data/tasks_all.jsonl \
  --output results/agent_v4_sft_template/agent_<base|grounder>.jsonl --num-repeats 3 --concurrency 2 ++head_server.port=21017 +route_failures_to_sidecar=true
```

**Compare:**

```bash
gym eval compare '+baseline_rollouts_jsonl_fpath="results/agent_v4_sft_template/agent_base.jsonl"' \
  '+candidate_rollouts_jsonl_fpaths=["results/agent_v4_sft_template/agent_grounder.jsonl"]' \
  +agent_name=web_agent_simple_agent '+output_dirpath="results/agent_v4_sft_template/agent_compare"'
```

**Aggregates** (committed script; it reproduces verified, claimed, claim = verified, false claims,
validity, failed and no-change actions, and step-cap stops):

```bash
.venv-gym/bin/python scripts/agent_metrics.py --no-runs \
  --rollouts results/agent_v4_sft_template/agent_base.jsonl results/agent_v4_sft_template/agent_grounder.jsonl
```

**Funnel, buckets, per-task counts, Fisher tests and the k ↔ `_ng_rollout_index` map** (scratch script,
standard library only):

```bash
python3 -I /tmp/t2-blade/blade_agent_report.py "$PWD" --json /tmp/t2-blade/blade_agent_report.json
python3 -I /tmp/t2-blade/brief.py "$PWD" text-size-large,high-contrast [base|grounder]   # per-step trace dump
```

**Open item:** `/tmp/t2-blade/blade_agent_report.py` is not committed. AGENTS.md requires every reported
number to come from a committed script, so the funnel and bucket numbers need it committed (for example
as `scripts/blade_agent_funnel.py`) before they go into docs/bounty-nemo-gym.md. The headline aggregates
are already reproducible with `scripts/agent_metrics.py`.

**Caveats:**

- n = 3 per task, with the planner at temperature 1.0. A one-rollout change in a task is within sampling
  noise.
- The click boxes (L segment, high-contrast switch, check-double button, chart bars) are estimates read
  off the 1280×800 gallery screenshots, not boxes from the DOM.
- The funnel's "target hit" depends on what the 2B summarizer reports. Toast-based markers undercount
  wherever it misses a toast, which is why high-contrast and mark-all-read use click boxes instead.
