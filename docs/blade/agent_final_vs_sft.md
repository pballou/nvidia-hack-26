# web_agent: final grounder vs sft-template vs base (diagnosis)

Sat 2026-10-17, ~22:40. Follow-up to [agent_base_vs_sft.md](agent_base_vs_sft.md). Rollouts only
(no re-grounding), read by three independent analyses plus a judge who spot-checked the cited
steps. Scratch scripts: `/tmp/t2-diag/` on T2's Mac.

## Setup

Three Gym `web_agent` runs, 12 sandbox tasks × 3 repeats each. They share byte-identical rows
(materialized inputs sha256 `363a601a…`), the same planner (official BF16 Lightning, thinking off,
temperature 1.0, `max_steps` 12), and the same screen-description model (`base`). Only the click
grounder differs:

| Arm | Grounder | When | Verified |
| --- | --- | --- | --- |
| BASE | Qwen3.5-2B, no adapter | Sat 18:45 | 18/36 |
| SFT | `sft-template` LoRA | Sat 19:05 | 22/36 |
| FINAL | `sft-lightning` + GRPO round 1 (dev 87.00) | Sat 22:05, on the `:8001` process the night loop restarted at 22:00:59 | 17/36 |

- **`gym eval compare`, FINAL vs BASE:** `mean/reward` −0.028, 95% CI [−0.20, +0.14]. Failed
  actions 10 → 0, CI [−0.52, −0.04].
- **SFT vs BASE:** +0.111, CI [−0.06, +0.28].
- **SFT vs FINAL:** Fisher p ≈ 0.34.

## Per task (verified out of 3)

| Task | BASE | SFT | FINAL | What moved it |
| --- | --- | --- | --- | --- |
| text-size-large | 0 | 2 | **3** | **Grounder.** BASE sent 0 of 18 L-intended clicks to L (x 1126-1152, inside the selected M). FINAL hits at x 1167-1178. SFT missed 4 times on a quoted `"L"` phrasing that FINAL never received. |
| high-contrast | 2 | 1 | 0 | **Summarizer + planner.** All 30 FINAL switch clicks landed. In every FINAL run, the click at s3 turned high contrast on. The next observation is byte-identical to the one on which SFT and BASE called `done`, but FINAL's planner clicked again and turned it off, then oscillated. FINAL's first view of the tab had no "High contrast is off" state line in 0/3 runs (BASE/SFT: 6/6). |
| mark-all-read | 3 | 3 | 1 | **Planner + a lenient check.** All 3 FINAL runs leave the path at s1, the planner's first call on a byte-identical row, before any grounder call (6/6 vs 0/3, p ≈ 0.012). FINAL hit the checkmark both times it was asked. Two SFT passes also cleared the bell notifications; with a strict check SFT would be 1/3. |
| message-support-meter | 1 | 2 | 0 | **Planner.** Choosing the recipient first passes 3/3 pooled; typing or pressing Send first fails 6/6, and FINAL did the latter 3/3. Its Send clicks hit the button, the page blocks the send ("Choose a recipient"), the summary omits the alert, and the planner claims success. |
| sms-billing-off | 0 | 2 | 0 | **Summarizer luck + an SFT habit.** One SFT pass came from a one-off per-row-state summary. The other came from SFT resolving the vague target "row Billing alerts" onto the SMS switch. |
| max-usage-month | 0 | 0 | 1 | **Planner.** Both grounders hit July. FINAL stopped there; SFT kept clicking bars. |
| download, autopay, pay-50, report-outage | 3 | 3 | 3 | — |
| star-landlord, stop-promotions | 0 | 0 | 0 | Planner never scrolls (star-landlord). On stop-promotions, SFT made 14 unrequested setting changes vs FINAL's 3. |

**SFT → FINAL: 7 rollouts lost and 2 gained.** About 6 of the 9 changes come from the planner or
the summarizer, and the 2 grounder-caused ones cancel. Net grounder effect ≈ 0.

## Grounder head-to-head (all 108 rollouts, 579 grounded actions)

| Arm | Grounded actions | Returned a point | Screen changed (of ok) | Hit on well-posed targets (hand-labelled) |
| --- | --- | --- | --- | --- |
| BASE | 191 | 94.8% (10 prose answers) | 59% | — |
| SFT | 186 | 100% | 77% | 112/121 (92.6%) |
| FINAL | 202 | 100% | 70% | 110/114 (96.5%) |

- **Change rate:** FINAL's lower screen-change rate comes from the targets the planner sent it,
  not from missing: 88 of its targets were absent, vague or non-interactive (SFT: 65).
- **Matched descriptions:** in 37 of 38 shared (page, description) groups, both grounders agree
  on whether the screen changed.
- **FINAL's weakness:** on targets that aren't on screen it anchors on literal text matches, such
  as a side-menu link (one sent `mark-all-read#0` to /billing).
- **SFT's weakness:** it resolves vague targets onto real switches, which causes collateral
  setting changes.
- **Neither adapter can answer "not found".**

## Conclusions

- **No measurable agent difference between the grounders.** End-to-end success is within noise
  for all three: 18 / 22 / 17 of 36, with every CI including 0, at n=3 per task and T=1.0.
- **What the grounders clearly change inside the agent:**
  - the tuned grounders always return a usable point (100% vs 94.8%; failed actions 10 → 0);
  - the small adjacent-option target is fixed (text-size-large 0/3 → 3/3 with FINAL);
  - FINAL makes fewer unrequested setting changes than SFT.
- **FINAL is equal to or slightly better than SFT as an in-agent grounder.** That's consistent
  with its higher ScreenSpot-v2-clean score: a 1.3-point accuracy gap is about 1.5 clicks out of
  ~115 well-posed clicks per arm, which this sandbox can't resolve.
- **Confound:** FINAL ran 3 hours later on a restarted serving process. The summarizer's output
  on identical screens changed (the high-contrast STATE line), and the planner produced a
  6/6 vs 0/3 split on byte-identical first inputs. A clean end-to-end grounder comparison needs
  both adapters on one vLLM process, with runs interleaved per task.
- **What would move agent success most:** screen summaries that always report control state and
  validation alerts; a "not found" outcome in the grounder's training and GRPO reward; and the
  agent-loop guards from the base-vs-SFT report. Another 1-2 points of ScreenSpot accuracy matters
  less.

## Demo recommendation

- **Use FINAL.** It was chosen on dev (the project rule), it is better on held-out ScreenSpot-v2,
  equal or better on in-agent clicks, and makes fewer unrequested changes, which matters for an
  accessibility agent.
- **Script the demo on tasks FINAL passed 3/3:** download-september-statement, turn-on-autopay,
  pay-50, report-outage-riverside, text-size-large.
- **Handle the weak tasks openly:** avoid high-contrast and message-support, or show them and say
  why they fail.
- **Thinking:** the demo recording uses thinking off (decided Sat night), the configuration these
  3/3 results were measured in. Thinking on's 26/32 was measured with `sft-template`, so it is
  reported as a measured option, not as FINAL's result.
