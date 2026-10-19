# GUI Assist as a product: architecture, accuracy and market

> **T5 research report, Sun 2026-10-18,** written after the hackathon submission. It is research
> only: we ran no models, GPUs or paid APIs for it.
>
> - **Internal numbers** come from [results.md][results], the BLADE reports
>   ([base vs SFT][blade-agent], [FINAL vs SFT][blade-final]), [local-mac.md][local-mac], and T1's
>   fully local console traces from Sunday 02:22–11:10 (31 runs, not committed).
> - **External numbers** come from a web sweep of 12 topics plus 8 gap topics. A second agent checked
>   every finding against its source: about 550 findings, with corrections applied. Where sources still
>   contradict each other, we say so.

**Labels on numbers:**

- **M**: measured by us this weekend.
- **I**: measured independently, by benchmark maintainers or third-party and academic evaluators.
- **V**: reported by a vendor or by the authors about their own model or product.
- **E**: our estimate or arithmetic, with the assumptions stated.
- **N**: news or a secondary summary of a primary source.
- **A**: an analyst estimate or prediction.
- *(pre-2025)* marks an older source.

Citations of vendor docs, terms, prices and regulation (types D and R in §12) carry no label.

**Terms used throughout:**

- **pass@k:** a task counts if any of k repeats succeeds.
- **pass^k:** a task counts only if all k repeats succeed. This is the reliability metric.
- **a11y tree:** the accessibility tree a browser or OS exposes.
- **SoM (set-of-marks):** a screenshot with numbered boxes drawn on candidate elements.
- **ZDR:** zero data retention.
- **PHI:** protected health information.
- **BAA:** a HIPAA business associate agreement, the contract that allows a vendor to process PHI.

Each citation's source and date are in §12. Vendor benchmark numbers in 2026 often use the vendor's
own harness, tools and run selection, so a V number is an upper bound, not a comparable score.

**Contents:**

1. [Executive summary](#1-executive-summary)
2. [Recommended architecture](#2-recommended-architecture)
3. [Designs compared](#3-designs-compared-accuracy-latency-cost-privacy)
4. [What exists in October 2026](#4-what-exists-in-october-2026)
5. [How accurate SOTA is](#5-how-accurate-sota-is)
6. [Our failure clusters and their fixes](#6-our-failure-clusters-and-their-fixes)
7. [Building the real product](#7-building-the-real-product)
8. [Commercial options](#8-commercial-options)
9. [Risks](#9-risks)
10. [Two-week experiment plan](#10-two-week-experiment-plan)
11. [What we could not verify](#11-what-we-could-not-verify)
12. [References](#12-references)

## 1. Executive summary

1. **Worth two more weeks to decide. Not yet a commitment.**
   - If we continue, it is a narrower product on a different architecture: an on-prem, verifier-gated
     computer-use layer for regulated and legacy environments (VDI/Citrix, canvas apps, air-gapped
     portals), sold to the people who own those systems and through automation platforms.
   - Not a pixel-only agent for every website. And not today's design, where the planner works from text.
   - We have no buyer evidence yet: no interviews, no design partner, no pricing test (§8.3).
2. **Our bottleneck is what the planner sees, not where the grounder clicks.**
   - On well-posed targets the grounder is rarely at fault: FINAL hit 110 of 114 hand-labelled targets
     (96.5%) (M, [blade-final]).
   - But it can't refuse. 88 of the 202 targets the planner sent it were absent, vague or
     non-interactive, and those became confident wrong clicks (M, [blade-final]).
   - Two separate losses happened before the grounder (M, [blade-agent]):
     - our summary pipeline lost the notifications grid in 15/15 first views, because a run-on summary
       line was cut by our own 240-character clip;
     - the summarizer itself dropped toggle state in 6/6 views.
   - No published study tests our exact design. The nearest evidence, on VisualWebArena, keeps the a11y
     tree in every arm: 11.4% with text only, 24.8% with captions of in-page images, 38.4% with a marked
     screenshot (I, [pando]).
   - Agents also trust a text channel over pixels they have read correctly (I, [state-belief]).
3. **Without the size cap, a planner that sees the screenshot is probably stronger, and it can run
   locally. We haven't tested that on our tasks.**
   - OSWorld-Verified is saturated above its 72.36% human baseline. Claude Fable 5 scores 85.96%. The
     open Holo3-35B-A3B (3B active) scores 78.15–82.56% across two workbook rows (I, [osw-v]).
   - On WindTunnel's short web tasks, which are scored from app state and include checkout journeys,
     Claude Sonnet 5.5 passed 140 of 147 attempts. Its median attempt took 19.7 s and cost $0.032
     (I, [windtunnel]).
   - Long tasks and tasks that change state are far from solved:
     - The best maintainer-run end-to-end completion on the OSWorld 2.0 board is 44.33%, by Opus 5. A
       submitter-run agent reports 44.44% (I, [osw2-board]).
     - The best ClawBench score is 33.3%, by Sonnet 4.6. No newer model has been run on it
       (I, [clawbench]).
   - Planner reasoning matters too. With our text summary, turning thinking on took the same 32 jobs from
     19 to 26 passes (M; interactive harness, `sft-template` grounder).
4. **Per-task cost is no longer the argument for local. Compliance is.**
   - Frontier computer use costs $0.0017 (Haiku 5.5) to $0.04 (GPT-6.1 Sol) per short-task attempt, and
     up to $0.26 for GPT-6 Astra (I, [windtunnel]). An assisted contact costs $7–14 ([contactbabel],
     [gartner-cost]).
   - No hosted computer-use offering cleanly covers all three of PHI under a BAA, zero retention and EU
     processing:
     - Anthropic's BAA article says computer use is not covered ([anth-baa]).
     - Gemini computer use is Preview, which means no PHI and no data processing addendum ([gcp-cu],
       [gcp-terms]).
     - Bedrock lists it as a Beta Service ([aws-terms]).
     - OpenAI keeps screenshots flagged by its CSAM scan even under zero retention ([oai-data]).
   - **The pitch's ROI is overstated.** The pitch says $3.9–7.7M. Net of failed attempts, a 2M-customer
     utility saves (E, §7.3):
     - about $0–0.75M a year at the 70–80% success an open planner with gates might reach;
     - up to about $2.2M only with a frontier planner at 85%;
     - possibly a loss at today's measured success of about 50%.
5. **Frontier models remove some of our failure clusters, not all.**
   - **Small icons and lossy summaries** mostly go away with a frontier planner. GPT-6 Astra reports
     92.7% on ScreenSpot-Pro (V, [oai-astra]). In independent single-shot runs, icons still trail text,
     at 63.4 vs 81.8 for GPT-5.5 (I, [guilens]).
   - **Scrolling** mostly goes away for frontier planners. Open agents still find only about 30% of
     hidden targets, against 94.6% for humans (I, [reveal]).
   - **Toggle state, relational grids and canvas** only shrink:
     - GPT-5 is under 40% on binary toggles (I, [star]).
     - The best model scores 31% on relational targets, against 96.9% for humans (I, [guiprim]).
     - Neither benchmark has tested the newest models.
   - **False "done" claims** don't go away. In one strong OSWorld pipeline, 64 of 71 failures ended with
     a success claim (V, [cura]).
6. **Recommended architecture: local-first, structure plus pixels, and verifier-gated (§2).**
   - A vision-language planner that sees the screenshot, plus the accessibility tree when the deployer
     owns the page.
   - A small grounder with zoom and a "not found" answer, as the pixel fallback.
   - Deterministic state checks after each action, and a done-gate.
   - Confirmation before money moves, and a human hand-off.
   - Record-and-replay for the top intents.
   - §10 tests it as arms A10g and A11g.
7. **The grounder weights are not a moat.**
   - Apache- and MIT-licensed 2–4B grounders report ScreenSpot-v2 scores level with or above ours:
     UI-Venus-1.5-2B 92.8, MAI-UI-2B 92.5 (V, [uivenus15], [maiui]), and Holo2-4B 93.15 on the
     submitted board ([ss-board]).
   - An untuned Qwen3.5-4B scores 93.7 in an independent run (I, [windom]) and 94.3 in Alibaba's own
     run ([qwen-ui-agent]).
   - Ours is 90.35 on SSv2-clean in our harness. For the same base weights, our harness on SSv2-clean
     read about 1.4–3.6 points above WinDOM's on full SSv2. So the real gap may be larger (§5.1).
   - NeMo Gym and BLADE are public tooling. What isn't easily copied: portal replicas with state checks,
     customer trace data, and the verification layer.
8. **Commercial ranking (§8):**
   1. An on-prem SDK for pixel-level automation of VDI, canvas and legacy apps in regulated
      organizations, sold through RPA and testing platforms. This is also the least measured option.
   2. A self-service assistant for site owners, rebuilt DOM-first with pixels as the fallback.
   3. Accessibility as the mission and a feature of (2), not a standalone business.
   4. Testing, but only the VDI/canvas niche.
   5. No hosted pixel API.

   Where we lose:
   - Incumbents already ship the planner-plus-grounder design (UiPath Screen Agent).
   - Where a site exposes agent tools (WebMCP), neither pixels nor the DOM is needed. Haiku 5.5 passed
     all 147 WindTunnel attempts through WebMCP at a median $0.00025 each, against 123 of 147 through
     computer use (I, [windtunnel]).
   - Where an a11y tree exists, smaller models do much worse from pixels: GPT-5 mini scores 83.1% with
     the tree and 48.9% with pixels. Frontier models reach parity: GPT-5.4 scores 83.8% with pixels and
     81.5% with the tree (I, [componentbench]).
9. **Biggest risks (§9):**
   - **Wrong payments reported as success.** Three headed console runs paid the full $142.37 balance
     instead of $50 and claimed success (M). All three final summaries mentioned $142.37, and two said
     they had paid it, so checking the amount against the request would have caught all three.
   - **Prompt injection through screen text.**
     - Every Opus 5 configuration was broken in exactly 1 of 14 Shade GUI attack scenarios (200 attempts
       each). Attempt-level success across all 2,800 attempts was 0.39–0.54% without probes and
       0.25–0.43% with them (V, [anth-opus5-card]).
     - Independent end-to-end attacks on Opus 4.5 and 4.6 agents succeeded 83% and 50% of the time,
       where success means any of 3 runs (I, [redteamcua]).
   - **Rendering nondeterminism.** Pay-50 passed 13/13 headless (8 harness, 5 console) but 1/4 headed
     (M). In VDI we can't pin the renderer at all.
   - **Incumbents, platform bundling, and site-exposed agent tools.**
   - **Trial-API terms.** 8,028 hosted-endpoint outputs (rewrites and keep/drop decisions) are already
     public in our dataset, and the public FINAL grounder was trained on them ([submission],
     [trial-tos]). That needs NVIDIA's written OK, or a regenerate-and-replace, now.
10. **Next two weeks (§10):**
    - Run the sandbox against the designs: 12 tasks plus 12 held-out variants, 8 repeats each, with the
      same 20-step cap and the same scripted user for every arm. The arms:
      - six frontier computer-use APIs;
      - a frontier planner that sees pixels and calls our grounder;
      - open VLMs on the H200, alone and with our grounder;
      - the recommended architecture on an open VLM (A10g, A11g);
      - an accessibility-tree arm;
      - our fixed design;
      - a VDI-style arm over a lossy remote-desktop stream.
    - Re-run the top open grounders through our harness with their native prompts, and train a 4B
      grounder v2.
    - Report pass@1 and pass^8 with task-level confidence intervals, plus false-claim and unsafe-action
      rates.
    - Hold 10 buyer conversations in parallel.
    - **Decision:** build local-first if an A10g or A11g arm comes within 10 points of the best frontier
      arm on held-out pass@1, with false success ≤ 2% and no wrong money moves. Otherwise choose
      between a customer-cloud product and staying in research.
    - **Estimated cost:**
      - API: $130–640 expected. The floor is about $65 at measured medians. The hard cap is $1,000,
        with per-arm caps.
      - GPU: about 11–16 H200-hours, roughly $75–110 at the $6.90/hr we were billed.
    - **Approvals needed:** the API spend, sending synthetic sandbox screenshots to vendors, and GPU
      time.

## 2. Recommended architecture

### 2.1 Diagram

```text
 user (voice/text) ──► intent check (clarify pre-flight, kept) ──► risk tier for this intent
                                                                          │
┌─────────────────────────── on-prem or customer VPC (default) ───────────┼──────────────────────┐
│                                                                         ▼                      │
│  observation builder                    VLM planner (sees pixels, plus the a11y tree if any)   │
│   • screenshot, pinned renderer ───────►  one action per step, goal fixed by the user;         │
│   • a11y/DOM snapshot, when present       screen text is data, never instructions              │
│   • OCR of toasts, alerts, dialog text            │                                            │
│                                                   ▼                                            │
│                                  action router ──► element-ref action (a11y/DOM) ──┐           │
│                                        │                                           ├─► browser,│
│                                        └────────► pixel grounder: zoom when unsure,│   VDI, app│
│                                                   "not found" allowed ─────────────┘           │
│                                                   │                                            │
│                         risk gate: pay / submit / delete / send? ──► user confirms on screen,  │
│                                                   │                  exact amount quoted       │
│                                                   ▼                                            │
│                         state verifier: expected post-state (toast text, field value, URL,     │
│                         a11y attribute, pixel diff of the target crop)                         │
│                              │ verified ──► next step / done-gate: done only if goal verified  │
│                              └ not verified ×2, stuck, infeasible ──► human hand-off + trace   │
│                                                                                                │
│  replay library: verified runs of top intents compiled into scripts, re-validated each run     │
│  trace store (PII-redacted) ──► portal replicas ──► NeMo Gym evals, BLADE, GRPO on failures    │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Components and why

| Component | Choice | Why (evidence) |
| --- | --- | --- |
| Observation | Screenshot from a pinned renderer (fixed headless build, fonts, viewport, device scale). Add the a11y/DOM snapshot when the deployer owns the page, and OCR text for toasts and dialogs. | Where a usable DOM exists, the a11y tree helps smaller models most: GPT-5 mini scores 83.1% with it and 48.9% from pixels, while GPT-5.4 is at parity, 81.5% vs 83.8% (I, [componentbench]; mock component pages). Rendering changes flip outcomes: pay-50 passed 13/13 headless but 1/4 headed (M, [results], [console-runs]), and a 28-px border flips 7.3% of correct grounding predictions (V, [mvp]). The DOM can be stale: agents followed stale structure on 38–88% of affected probes, which argues for a pixel consistency check (I, [state-belief]). In VDI, the remote server renders and compresses the pixels, so pinning isn't possible and perturbation tests carry the load (§8.1). |
| Planner | A vision-language model that sees the screenshot. On-prem default: an Apache- or MIT-licensed 27–35B VLM. Qwen3.8-27B reports 84.3 on OSWorld-Verified (V, [qwen38]). Holo3-35B-A3B scores 78.15–82.56 across two workbook rows (I, [osw-v]); H reports 77.8 (V, [holo3]). Fara1.5-27B, web only, reports 72.3 on Online-Mind2Web (V, [fara15]). Use a cloud model where policy allows. Keep the planner pluggable, because models change monthly. | Text-only planning loses state (§3.2). Our planner, Lightning, scores 28.69% on WebArena-Verified in a third-party run, against 37.62% for Qwen3.6-35B-A3B (I, [liquid-card]). NVIDIA's card reports 9.28% on τ³-bench banking (V, [lightning-card]). If staying on NVIDIA models matters, NVIDIA reports 51.14% OSWorld-Verified for Nemotron 3.5 Super VL 120B-A12B (V, [nemotron-supervl]). It needs multi-GPU hardware, outside the one-box pilot in §2.4. H Company is post-training it for computer use ([h-newsroom]). |
| Grounder ("hands") | A specialist of 4B or less that zooms only when uncertain and can answer "not found". Retrain with our loop on a Qwen3.5-4B base. | A specialist grounder rescues weak planners: o3 scores 23.0% alone and 51.0% with the Jedi-7B grounder, both without code actions (I, [osw-v]). GTA1 with o3, which adds code actions, reaches 53.1%. Zoom adds 5–13 points on high-resolution screens at 2B (V, [maiui], [guiowl15]). Zooming every query hurts, so gate it on uncertainty (V, [uizoomer]). Refusal can be trained at 2B: 53.7% on OSWorld-G vs about 0% for most GUI grounders (V, [guiowl15]), though other variants of the same family score 5.6–70.4. In planner+grounder systems, grounding is under 1% of cost and 1.8–3.9% of latency (I, [osw-human]). |
| Action router | Use an element reference when a reliable a11y node exists, and pixels otherwise: canvas, VDI/RDP, closed shadow DOM, some cross-origin iframes, stale DOM. | Anthropic's own browser toolset falls back to coordinates for canvas, remote desktops and cross-origin iframes (V, [anth-browser]). Flutter web renders to a canvas, with semantics off by default ([flutter-a11y]). |
| Risk gate | Deterministic policy keyed on intent and element: pay, submit, delete, send, change contact details. The user confirms on screen, with the amount read from the screen and compared with the request. | Every vendor recommends this (V, [anth-cu], [oai-cu], [gem-cu]). PCI SSC's 2026 AI guidance recommends explicit human approval for agent actions on cleartext card data ([pci-ai-2026]). Our worst failure was a full-balance payment ($142.37) claimed as a success (M). All three runs' final summaries mentioned $142.37, so comparing the confirmed amount with the request would have caught them. |
| State verifier and done-gate | For every state-changing step: the expected post-state (toast text, field value, URL, a11y attribute, or a pixel diff of the target crop). `done(success)` is accepted only if the goal state is verified. | False "done" dominates failures (§6). On τ²-bench, false success is 45–48% of failures in single-control domains but 3% in the dual-control domain, where the user also controls the state. No LLM judge exceeded AUROC 0.65 at catching it (I, [tau2-false]). Rubric verifiers report a false-positive rate of 0.01–0.08, against at least 22% for WebJudge (V, [univ-verifier]). The evidence that verifiers close the gap is vendor-reported, and one study points the other way: over 86% of failures stayed false "done" even with a completeness verifier (V, [vlaa]). Deterministic checks on state therefore come first. |
| Hand-off | One tap hands over to a human with the trace and screenshot. Triggered on unverified steps, stuck runs or infeasible requests. | 87% of customers say GenAI service must offer a human (I, [gartner-87]). Trust rises from 26% to 55% when a human is visibly available (I, [five9]). |
| Replay library | Verified runs of the top intents are compiled into scripts with pixel/DOM anchors and re-validated on every run. The agent handles the long tail. | Buyers want determinism. UiPath advises "maximally deterministic, minimally agentic" (V, [uipath-screenplay]). ServiceNow records agent runs for deterministic replay (V, [servicenow]). Applitools runs no LLM at test runtime (V, [applitools]). |
| Injection defense | Screen text is data. The goal comes only from the user, the plan is made before untrusted content is read, and the risk gate is the last line. | Every Opus 5 configuration was broken in exactly 1 of 14 Shade GUI scenarios (200 attempts each); attempt-level success across all 2,800 attempts was 0.39–0.54% without probes and 0.25–0.43% with them (V, [anth-opus5-card]). A secondary summary of the Opus 5.5 card puts computer use at the highest attack success of any surface: 2.8% within 15 attempts, against 0.4–0.5% for tools and coding (N, [opus55-inj]). Splitting an attack across steps raised success by up to 31 points (I, [stepjack]). CaMeL keeps 77% of AgentDojo tasks (text and tools, not GUI) solvable with provable security, against 84% undefended (V, [camel]). |
| Eval and improvement | One NeMo Gym environment per customer-portal replica, scored on pass^k. BLADE on every pair of runs. GRPO on failures replayed in replicas, never on raw production PII. | Re-running the same agent flips 12–14% of web outcomes (I, [rerun-flips]). Agent S3 has pass@10 ≈ 78% but pass^10 ≈ 36% (V, [passk-cua]). RL on a single environment caused regressions elsewhere ([nemotron-rl]). Most of ARPO's RL gain is in-domain (V, [arpo]). |

### 2.3 What changes from the hackathon build

| Keep | Change | Drop |
| --- | --- | --- |
| The sandbox with state checks, the `gui_grounding` and `web_agent` Gym environments, BLADE, the GRPO loop with a keep-gate, the clarify pre-flight, the stuck stop, Parakeet voice, local packaging | Grounder base 2B → 4B, plus refusal, uncertainty-gated zoom, adjacent-option and toggle-state data. Planner: Lightning on text → a VLM on pixels (Lightning can keep the clarify and intent roles). Gym `max_steps` 12 → 25 plus the stuck stop, matching the console, scored on pass^k. | The text screen summary as the planner's only view. The 33B cap. The headed browser in anything we measure. |

### 2.4 Hardware for a pilot (estimates)

- **Weights.** A 27–35B VLM planner in BF16 needs about 54–70 GB, plus a 4B grounder at about 8 GB
  (E: about 2 bytes per parameter). FP8 or NVFP4 roughly halves that.
- **Our measured fit.** Lightning 30B plus a 2B VLM ran together on one 128 GB M4 Max, but only just:
  about 88 GB allocated, 28% free at the lowest, other apps closed, and the first attempt stopped on
  memory pressure (M, [local-mac]). A 27–35B VLM plus a 4B grounder would be at least as tight.
- **One site or one pilot**, any of:
  - one RTX PRO 6000 96 GB at $16,000 ([rtx6000], 2026-08-13);
  - a DGX Spark 128 GB at $6,950 ([spark-price], 2026-10-02);
  - a Mac Studio M5 Max, from $2,499 with up to 128 GB ([mac-studio], 2026-08-25). The 128 GB
    configuration a BF16 planner needs costs more; we didn't check its price.
- **Multi-tenant:** an H200 at $6.48/hr list on Brev ([brev-h200]; we were billed $6.90/hr), or
  $5.29–10.60/hr elsewhere ([gpu-prices], 2026-10-15).
- **Licensing:** NVIDIA AI Enterprise ($4,500 per GPU per year) applies to NIM, not to open weights
  served with vLLM. NVIDIA's own pages disagree on which NIM uses need it: its FAQ says any production
  NIM ([nim-faq]), while the newer offerings page says only the Production Branch ([nim-offer]). The
  price is from [nvaie].

## 3. Designs compared: accuracy, latency, cost, privacy

### 3.1 Comparison

These numbers come from different benchmarks and harnesses, so they are evidence, not a leaderboard.
The experiment in §10 puts the designs on the same tasks.

| Design | Example system | Task-success evidence | Latency | Cost per task | Screens leave the premises? | Our clusters it fixes (numbered as in §6) |
| --- | --- | --- | --- | --- | --- | --- |
| **(a) Single frontier VLM agent** (API) | Claude Sonnet/Haiku 5.5 computer toolset, GPT-6.1 Sol `computer` tool, Gemini 3.8 Flash `computer_use` | WindTunnel short tasks, 3 attempts each: Sonnet 5.5 140/147 attempts, GPT-6.1 Sol 45/49 tasks, Haiku 5.5 123/147 attempts (I, [windtunnel]). OSWorld 2.1 full completion: Opus 5.5 48.7%, Sonnet 5.5 43.5% (V, [anth-sonnet55-card]). | Median 16–21 s per short-task attempt (I, [windtunnel]). About 6–10 s per step (E from I data, [speedrun]) | $0.0017 (Haiku 5.5) to $0.040 (GPT-6.1 Sol) per short attempt; Opus 5 $0.14, GPT-6 Astra $0.26 (I, [windtunnel]) | Yes. Zero retention is possible; PHI coverage is unclear (§7.2). | Mostly: 3, 5, 7, 8. Partly: 1, 2, 4. Not: 6, 9, 10. |
| **(a′) Single open VLM agent**, on-prem | Holo3/Holo4-35B-A3B (Apache), Qwen3.8-27B (Apache), Fara1.5-9B/27B (MIT, web) | OSWorld-Verified: Holo3-35B-A3B 78.15–82.56% (I, [osw-v]). Online-Mind2Web: Fara1.5-9B 63.4% (V, [fara15]). Instrumented sites: Qwen3.8-27B 0.48 from screenshots vs 0.74–0.76 with a DOM harness (I, [webpagebench]). | Holo4-35B-A3B: about 5 s per step (V, [holo4-traj]) | Hardware only. Holo4 reports $0.05–0.08 per OSWorld task (V, [holo4]); about $0.003–0.04 on owned hardware (E, §7.3). | No | Probably like (a) but weaker. Scroll and small icons untested for open planners (§6). |
| **(b) Planner sees the screenshot, plus a specialist grounder** | Agent S3 (GPT-5 + UI-TARS-1.5-7B), GTA1-32B + GPT-5, UiPath Screen Agent | OSWorld-Verified: o3 alone 23.0%, with the Jedi-7B grounder 51.0% (both without code actions); UiPath Screen Agent + Opus 4.5 67.14% (I, [osw-v]) | Planning is 53–75% of latency, grounding 1.8–3.9% (I, [osw-human]) | GTA1 + o3: $2.43 per OSWorld task, grounding under 1% (I, [osw-human]) | Yes if the planner is in the cloud | Helps weak planners. Adds zoom and refusal (7, 8; refusal also helps 3). |
| **(c) a11y tree/DOM plus pixels** | Browser Use, Stagehand hybrid, Claude browser toolset, WebTactix | Component pages: GPT-5 mini a11y 83.1% vs pixels 48.9%; GPT-5.4 pixels 83.8% vs a11y 81.5% (I, [componentbench]). WebArena: WebTactix (DOM) 73.2% on all 812 tasks, team-reported (V, [webarena-tracker]). | Browser Use: 68 s per task vs 225–330 s for 2025 computer-use models (V, [bu-speed]) | Claude browser toolset: about 6.6k tokens of overhead per request (V, [anth-price]). Stagehand hybrid: $0.43–0.95 per task (V, [stagehand]). | Depends on where the planner runs | 1 and 2 (state and row structure from attributes), 3, 5. Fails on canvas, VDI, closed shadow DOM. |
| **(d) Ours today** | Lightning reading a text summary, plus the 2B grounder | Gym (12-step cap), 12×3, thinking off: 17/36 FINAL, 18/36 base, 22/36 SFT, all CIs including 0 (M, [results]) | Interactive, all 12 tasks, thinking off: p50 65 s, p90 220 s per task (M, [results]) | Hardware only. About $0.003–0.04 per task (E). | No | 7 (the L option went 0/3 → 3/3, M). Open: 1, 2, 3, 4, 5, 6, 8, 9, 10. |
| **(r) Recommended**: (a′) or (b), plus (c), with verifier and gates | §2 | Not measured. With an open planner, verified success at or below (a)'s (E). The gates aim to turn the remaining failures into hand-offs rather than false claims. | 20–60 s per task (E: 4–8 steps at about 5 s, plus gate checks) | ≤ $0.04 per task on owned hardware (E) | No by default | All partly; 6 is handled by the gate, not the model |

**Notes on row (d) and other caveats:**

- The Gym and interactive-harness numbers were measured hybrid, with the grounder on the H200. Gate G
  matched the local grounder to it on 200 dev rows. The console traces ran fully local.
- With thinking on, the interactive harness passed the same 32 jobs 26 times vs 19 with thinking off
  (p = 0.039). That run used a 25-step cap with the stuck stop and the `sft-template` grounder, not
  FINAL (M, [results]).
- **Console traces (M, uncommitted traces, summarized in [console-runs]).** FINAL grounder with base
  summaries, completed runs that took at least one action, mostly demo tasks:
  - headless 8/10;
  - headed 3/8.
- **Latency per step on the Mac (M).**
  - From the traces: plan about 3.6 s, then about 5.4 s for ground, act, settle and summary.
  - From [local-mac]: base summaries take 2.9–4.5 s, and grounding at 2 MP takes 1.77–1.86 s p50.
- **Row (a).** Agents' Last Exam (Opus 5.5, 38.2% pass) is left out because it runs Claude Code in an
  OS sandbox, not a screenshot loop (I, [ale]).
- **Row (c).** Stagehand v1's "hybrid 90.6% vs computer use 92.1%" compares two different models
  (Fable 5 vs Fable 5.1), so it isn't an ablation (V, [stagehand]).

### 3.2 Would one big model that sees the screenshot beat our split?

**Probably, on accuracy, but untested on our tasks.** The literature is indirect (§11). Planner
reasoning alone already closes part of the gap: with thinking on, the same 32 jobs went from 19 to 26
passes on our text summary (M). Arms A3, A10, A11 and A10g/A11g in §10 test it directly.

- **Our losses happen before the grounder.**
  - In the base-vs-SFT run, 34/36 rollouts reached the right page, but only 22 (base) and 27 (SFT) hit
    the target element. Most of that drop came from the summary pipeline and from planner choices
    (M, [blade-agent]).
  - On well-posed targets FINAL hit 110/114. It can't refuse vague or absent targets, though
    (M, [blade-final]).
- **Text observations lose state, and models trust text over pixels.**
  - A 2024 pixel-to-text pipeline beat GPT-4V on most UI questions but lost on element state such as
    on/off and selected: 0.57 vs 0.67 (V, [visiontasker], *pre-2025*). That is our cluster 5.
  - When a text channel is present, agents adopt its stated state even after reading the pixels
    correctly. One wrong first-step belief led to final failure in 97–100% of cases (I, [state-belief]).
  - The counterpoint: when the text is an accurate UI tree rather than a VLM caption, adding the
    screenshot bought only 4–6 points at about 25x the cost (I, [dailydroid]). The text channel isn't the
    problem; a lossy one is.
- **The closest architectural analogue to our design** is a 2B VLM writing summaries for a GPT-4o
  planner.
  - It reached 25.6–27.6% on AndroidWorld, against 44.0% for GPT-4o with a 7B grounder on screenshots
    (V, [ecoagent]).
  - The comparison isn't controlled: that planner still sees the first screenshot, and the summaries
    only feed memory and replanning.
- **A local model can now play the screenshot-planner role:**
  - Holo3-35B-A3B scores 78.15–82.56% on OSWorld-Verified (I, [osw-v]).
  - Fara1.5 is screenshot-only and trained to pause before payments (V, [fara15]).

**What one big model does not fix:**

- False "done" (§6).
- Prompt injection (§7.1).
- Rendering nondeterminism.
- Compliance (§7.2).
- On DOM-friendly pages, smaller and open models still trail an a11y/DOM agent badly. Frontier models
  reach parity, until clutter reopens the gap (I, [componentbench]).

That is why the recommendation in §2 is a pixel-seeing planner with structure when available, plus
gates.

**Where planner + specialist grounder (b) still pays off:**

- A cheap or local planner that grounds poorly.
- High-resolution or pro-app screens where zoom matters.
- When we want a trainable, auditable "hands" model separate from the planner.

A variant worth testing keeps a local grounder on the full-resolution pixels and lets a cloud planner
see only a redacted screenshot. Redaction tooling is immature: Microsoft Presidio's image redactor is
labelled beta and not production-ready ([presidio]).

## 4. What exists in October 2026

### 4.1 Frontier computer-use APIs and products

| Vendor / model | Interface and status | Price per 1M tokens, in / out | Cost of a 1280×800 screenshot | Headline agent number | Data terms |
| --- | --- | --- | --- | --- | --- |
| **Anthropic** Opus 5.5 / Sonnet 5.5 / Haiku 5.5 | `computer_toolset_20260801`, 17 tools including zoom. GA on the Claude API and Google Cloud, beta on Bedrock and Foundry ([anth-cu]). A separate `browser_toolset_20260801` reads the a11y tree ([anth-browser]). | $4/$20, $2/$10, $0.10/$0.50. Haiku 5.5 costs 5x ($0.50/$2.50) on prompts over 100K tokens, which long screenshot-history loops cross: 46% of calls did in SLICC's max-effort Haiku 5.5 run on long Browser Use Bench tasks of about 30 minutes ([anth-price], rate card 2026-10-07; [slicc]) | 1,334 visual tokens (⌈w/28⌉×⌈h/28⌉), plus about 4,500 tokens of toolset overhead per request ([anth-vision], [anth-price]) | OSWorld 2.1 completion on the full set: Opus 5.5 48.7%, Sonnet 5.5 43.5% (V, [anth-sonnet55-card]). Offline subset: Haiku 5.5 37.1% (V, [anth-haiku55-card]). | Zero retention: yes. HIPAA: the docs table says yes, but the BAA article says not covered ([anth-retention], [anth-baa]). No EU inference on the first-party API; US-only routing costs 1.1x ([anth-residency]). Screenshots are scanned for injection by default ([anth-cu]). |
| **OpenAI** GPT-6 Astra / GPT-6.1 Sol / GPT-6 Luna | Responses `computer` tool; the guide gives no GA or beta label, and for GPT-6 Astra OpenAI recommends code execution instead ([oai-cu]). The Agents API hosted browser is beta, with no zero retention and US only ([oai-agents-faq]). | $10/$50, $2/$10, $0.10/$0.50 ([oai-price]) | About 1,200 tokens (32-px patches × 1.2, E from [oai-vision]) | Astra: OSWorld 2.1 completion 34.3% in a Meta run (I/competitor, [meta-wonder]). ScreenSpot-Pro without tools: 92.7% (V, [oai-astra]). | `/v1/responses` can be BAA-covered with Modified Retention ([oai-hipaa]). CSAM-flagged screenshots are kept even under zero retention, and abuse logs default to 30 days ([oai-data]). |
| **Google** Gemini 4 Argon / 3.8 Flash | `computer_use` tool, Preview ([gem-cu], 2026-10-13) | Argon $2/$10 until 2026-11-30, then $4/$20. Flash $0.75/$3.75 until 2026-12-31, then $1.50/$7.50 ([gem-price]) | 1,120 tokens (high) or 2,240 (ultra_high) ([gem-media]) | Argon: OSWorld 2.0 offline partial score 69.2%, best of 3 runs, no completion rate published (V, [gem-argon]) | Preview means no PHI, no data processing addendum and no data-location commitment ([gcp-cu], [gcp-terms], [gcp-hipaa]). The Gemini API keeps abuse logs 55 days and has no zero retention ([gem-policies]). |
| **Meta** Muse Wonder | Responses-style `computer` tool, preview | $4/$20 ([meta-price]) | Not documented | OSWorld 2.1: 34.3% completion, 72.6% partial (V, [meta-wonder]) | Preview; no HIPAA statement found |
| **Amazon** Nova Act | Managed browser agent, GA 2025-12-02, us-east-1 only | $4.75 per agent-hour ([nova-price]) | — | Claims 90% reliability on early customer workflows (V, [nova-ga]) | AWS |
| **Microsoft** Copilot Studio computer use | GA May 2026, Windows | 5 credits per step, about $0.04–0.05; 15 per step on the premium model, about $0.12–0.15 ([copilot-cu]) | — | — | Microsoft says it is not intended for financial transactions and struggles with dropdowns and date pickers ([copilot-faq]) |

**Consumer agent browsers:**

- Claude in Chrome went GA on 2026-08-26 ([chrome-ga]). It blocks purchases and financial transactions
  in every permission mode ([chrome-perm]).
- Gemini in Chrome "auto browse" launched for US Pro and Ultra subscribers, capped at 20 or 200
  tasks per day ([auto-browse], [gem-autobrowse-help]). In Ars Technica's test it scored a median 7/10
  and needed nudging in almost every task ([ars-autobrowse]).
- OpenAI retired ChatGPT Atlas on 2026-08-09 ([oai-atlas]).
- Google shut down Project Mariner on 2026-05-04 ([mariner]).

The market is moving fast and products are being folded into the general assistants.

### 4.2 Open-weight models by size class

| Size | Best agent (score, type) | Best grounder (score, type) | License notes |
| --- | --- | --- | --- |
| ≤ 3B | GUI-Owl-1.5-2B: OSWorld-Verified 43.5, AndroidWorld 67.9 (V, [guiowl15]) | UI-Venus-1.5-2B: SSv2 92.8, SS-Pro 57.7, OSWorld-G 59.4 (V, [uivenus15]). MAI-UI-2B: SSv2 92.5, SS-Pro 57.4, 62.8 with zoom (V, [maiui]). Ours: 90.35 on SSv2-clean (M). | Apache or MIT. Holo1.5-3B (91.66 on the board) uses the non-commercial Qwen Research License ([ss-board]). |
| 4–9B | Fara1.5-4B/9B: WebVoyager 80.8/86.6, Online-Mind2Web 57.3/63.4, screenshot-only, pauses before payments (V, [fara15]) | Holo2-4B: SSv2 93.15 (V, [ss-board]). Untuned Qwen3.5-4B: SSv2 93.7 independently (I, [windom]), 94.3 in Alibaba's run ([qwen-ui-agent]). | Fara MIT; Holo2-4B Apache; Qwen Apache |
| 27–35B | Holo3-35B-A3B: OSWorld-Verified 78.15–82.56 across two workbook rows (I, [osw-v]); H reports 77.8 (V, [holo3]). Qwen3.8-27B: 84.3 (V, [qwen38]). Holo4-27B: 85.2, but the harness has code and shell tools (V, [holo4]). | UI-Venus-2-27B: OSWorld-G (Refine split) 79.1 (V, [uivenus2]) | Holo3 and Holo4-35B-A3B Apache. Holo4-27B is CC BY-NC (non-commercial) ([h-local]). |
| ≥ 70B | Qwen-CUA 397B-A17B: OSWorld-Verified 86.2, needs about 807 GB BF16 on 8×H200 (V, [qwen-cua]) | — | Apache. UI-TARS-2 is not open weights. |

Self-reported agent numbers for the same model can differ by 12 points between vendors' harnesses.
Qwen3.6-27B scores 63.9 on OSWorld-Verified per Qwen ([qwen38]) and 75.6 per Meta ([qwen36-harness]).

## 5. How accurate SOTA is

### 5.1 Grounding

**On comparability with our 90.35%:**

- Ours is SSv2-clean: 1,244 of 1,272 rows, with 28 desktop rows excluded for capture-machine
  contamination. It is scored in our own Gym harness: 2 MP cap, our prompt and parser, temperature 0.
- Most published numbers use all 1,272 rows and each paper's own harness. Some API-model cells use
  subsets (Argus scored 273 items for one of them, [argus]).
- The same weights move 3–8 points between harnesses:
  - The Qwen3.5-2B base scores 86.25 on SSv2-clean in our harness (M) and 83.0 on full SSv2 in
    WinDOM's (I, [windom]). Depending on how it does on the 28 excluded rows, that is about 1.4–3.6
    points apart on comparable rows.
  - So our harness may read high, which means our 90.35 may overstate our standing against published
    numbers. Only an exact full-SSv2 re-run settles it.
  - GPT-5.4 on ScreenSpot-Pro scores 36.7 and 44.5 in two independent harnesses ([argus],
    [cuactspot]).
- So don't read a gap of a few points as either noise or a lead until G1 in §10 re-runs those models in
  our harness with their native prompts.

| Benchmark | What it measures | Best GUI-specialist model | Best open ≤ 3B | Frontier general models | Notes |
| --- | --- | --- | --- | --- | --- |
| ScreenSpot-v2 (1,272) | Mobile, desktop and web targets | 95.28, UI-Venus-72B (V, [ss-board]). 97.5, Qwen-UI-Agent-27B (V, [qwen-ui-agent]). | 92.8 (V, [uivenus15]). On the board: 91.66, Holo1.5-3B (non-commercial). | Single-shot, matched harness: GPT-5.4 87.0, Sonnet 4.6 45.2 (I, [argus]) | Saturated: 10.8% of items flawed in a blind audit (I, [venusbench]) |
| ScreenSpot-Pro (1,581) | Pro apps; targets average 0.07% of the screen | Board: 85.1 (ensemble + zoom), 82.7 (single model + zoom) ([sspro-board]) | 57–58 single-pass; 70.4 with crop (V, [guiowl15]) | GPT-6 Astra 92.7, no tools (V, [oai-astra]). Opus 4.8 82.3, 87.9 with tools (V, [anth-opus48-card]). Independent single-shot: GPT-5.5 74.8 (icons 63.4), Opus 4.7 57.4 (icons 38.9) (I, [guilens]). | Tools and crops move scores by tens of points: Opus 4.8 +5.6 with tools (V, [anth-opus48-card]), GUI-Owl-1.5-2B +12.6 with crop (V, [guiowl15]) |
| OSWorld-G (564, incl. 54 refusal items) | Text matching, element recognition, layout, fine-grained manipulation, refusal | 70.6, UI-Venus-1.5-30B-A3B ([uivenus15]). 79.1 on the Refine split, UI-Venus-2-27B (V, [uivenus2]) | 52–59 at 2B (V, [uivenus15]). Holo1.5-3B 61.57, non-commercial (V, [holo15]). GroundNext-3B 64.2, possibly a different split (V, [groundnext]). Fine-grained manipulation is 40–52% at 2B. | GPT-5.4 66.7 (I, [argus]). Refusal: Gemini 2.5 Pro 38.9 (I, [jedi]). | Refusal is about 0% for most GUI-tuned grounders; GUI-Owl-1.5-2B gets 53.7 (V, [guiowl15]) |
| UI-Vision, spatial split | "The icon left of X" | — | GroundNext-3B 50.6, in-domain per its authors (V, [groundnext]). Other 2B models 10.4–21.6 (V, [uivenus15]). | — | Zoom adds only 0.9 points at 2B (V, [maiui]) |
| 2026 sets that match our clusters | Relational (GUI-Primitives), toggles (StaR), exact state (FineState), canvas (CUActSpot), hidden targets (Reveal-and-Click) | Relational: best is Opus 4.7 at 31.3%, humans 96.9% (I, [guiprim]). Toggles: GPT-5 under 40% (I, [star]). Exact state: best 22.8%, models up to early 2025 only (I, [finestate]). | — | Canvas: GPT-5.4 65.6 vs MAI-UI-2B 12.5 (I, [cuactspot]) | Hidden targets: open models about 30%, humans 94.6% (I, [reveal]) |

**Grounding scores predict agent success poorly.** In one study GPT-5.4 scored 44.5 on ScreenSpot-Pro
yet was the best OSWorld grounder, ahead of models scoring 65–71 (I, [cuactspot]).

### 5.2 Agent benchmarks

| Benchmark | Top results | Reliability and validity |
| --- | --- | --- |
| OSWorld-Verified (361 tasks, 100 steps, humans 72.36%) | Workbook rows, submitted and then verified by the maintainers (I, [osw-v]): an agent framework (Opus 4.7 + code actions) 90.19; Claude Fable 5 85.96; Holo3-35B-A3B 78.15–82.56; UiPath Screen Agent 67.14 | Ten single runs of Agent S3 + GPT-5 range from 60.67 to 65.58 (I, [osw-v]). About 10% of tasks are invalid and 15% need only a terminal (I, [epoch-osw]). An exploit agent scored 73% without solving any task (I, [rdi-exploit]). |
| OSWorld 2.0 (108 long tasks) | **Official board, v2.1 full set, end-to-end:** Opus 5 at max effort 44.33% (I, [osw2-board]). **Top submitter-run agent:** Darwinx + Opus 4.8 44.44% (450 steps, 0.99 threshold; V, [osw2-board]). **Vendor:** Opus 5.5 48.7% (V). **Paper, June release:** Opus 4.8 20.6% at about $72 per task; general open-weight models 2.8–4.6% (I, [osw2-paper]). **Later open releases** self-report higher: Holo4-35B-A3B 12.3%, Holo4-27B 41.5% (harness with code and shell, V, [holo4]); Qwen-CUA 18.5% (V, [qwen-cua]). | Partial-credit scores run 30–45 points above completion rates. The benchmark version alone moves the same model by about 13 points (I, [osw2-board]). |
| WebArena (812) and WebArena-Verified | Narada 86.1 (closed, self-run); WebTactix (DOM) 73.2 on all 812 tasks, 74.3 excluding 13 N/A, team-reported ([webarena-tracker]). Verified: about 70–72, vendor-reported ([wav-list]). Lightning 28.69 (I, [liquid-card]). | Humans 78.24. An exploit reaches about 100% ([rdi-exploit]). Answer matching rejects acceptable behavior in 21.7% of audited FAILs (I, [misscore]). |
| Online-Mind2Web (300 live tasks) | Human-eval board: 97.7, with each $3,500 evaluation paid for by the submitter ([om2w-board], [om2w-pricing]). HAL's best reproduction: 42.33 at about $0.57 per task (I, [hal-om2w]). The 2025 paper: Operator 61.3, Browser Use 30.0 (I, [om2w-paper]). | The judge changes the score: the same agent gets 78.7 from humans and 64.7 from WebJudge ([steel-om2w]) |
| AndroidWorld (116) | Five self-reported 100% entries; humans 80.0 ([aw-sheet]) | No verification |
| Tasks that change state | ClawBench (153 live sites, final submit intercepted): best 33.3% at $7.51 per task (Sonnet 4.6; no newer models tested); GPT-5.4 6.5% (I, [clawbench]) | Agents' "done" claims overstate verified success by up to 41 points (I, [webpagebench]) |
| Long read-only web research | Odysseys2 (file deliverables, no sign-in or cart): Astra 47/100 perfect, Opus 5.5 18/100; one run each; OpenAI-family LLM judge (I, [odysseys]) | Changing the judge alone moved Astra from 86 to 47 perfect |
| Blind users' own commands | OLLA: 1,258 commands from 8 blind users, replayed per model with the Windows UI Automation (UIA) tree plus a screenshot: GPT-5 52.5% (I, [olla]) | Desktop Office-type apps; users avoided delegating banking |
| Simulated assistive-technology use | A11y-CUA: autonomous Claude Sonnet 4.5 from screenshots only, 60 tasks: 78.3% by default, 41.7% keyboard-only, 28.3% at 150% magnification (I, [a11ycua]) | No user in the loop |

**Implications for how we evaluate:**

- Report verified success with pass^k and confidence intervals. Re-runs flip 12–14% of outcomes
  (I, [rerun-flips]), and our 17–22/36 spread is within that noise.
- Treat partial-credit scores, self-judged web boards and saturated benchmarks as marketing.

## 6. Our failure clusters and their fixes

The two "goes away?" columns ask whether a planner that sees pixels removes the cluster on its own: a
frontier one, or an open on-prem one of the class §2 recommends. Most cluster benchmarks have tested only
2024–2025 models, so these verdicts lean on older numbers.

| # | Cluster | What we saw | What SOTA shows | Goes away? Frontier planner | Goes away? Open planner | Fix by system design or training |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Dense toggle grids, toggle state | Our clip cut the notifications grid in 15/15 first views. sms-billing-off 0–2/3, stop-promotions 0/3 in every Gym arm (M, [blade-agent]). | Proprietary agents incl. GPT-5 (2025) under 40%, biased toward toggling (I, [star]). Exact target-state success at most 32.8% (UGround-7B; models up to early 2025, I, [finestate]). Toggle tasks fail more than 45% of the time on live settings pages (I, [websp]). | Shrinks | Shrinks | Read state from a11y attributes when present; one line per grid row with per-column state; verify each flip by its toast and undo wrong-row flips; an oscillation guard. Train state-aware reasoning: StaR lifted correct no-op behavior from 35.8% to 96.5% on a 7B model (V, [star]). |
| 2 | Relational targets (download icon far from "September 2026") | All Gym arms passed 3/3 because the summary named the row's icon. In the console, 1 of 2 completed download runs also downloaded a wrong month (2025-11) and claimed success, and an operator-stopped run did the same (2026-05) (M, [console-runs]). Vague targets like "row Billing alerts" landed on whichever switch was nearest (M). | Best model (Opus 4.7) 31.3%, humans 96.9%; 60–92% of clicks land on neither candidate (I, [guiprim]). Relational phrasing costs 27–56 points (I, [perturbed]). UI-Vision spatial at 2B: 10–22% (V, [uivenus15]). | Shrinks | Shrinks | Use the row structure from the DOM when present. Marking the two true candidates (oracle set-of-marks) lifts GPT-5 from 30% to 87%, but Opus 4.7 only from 31% to 40%; a deployed version needs its own candidate detector (I, [guiprim]). The planner names row and column; train the grounder on relational phrasing. Zoom does not help (+0.9, V, [maiui]). |
| 3 | Scroll-dependent targets | 0 scrolls in 6/6 base/SFT star-landlord rollouts; 0/3 in every Gym arm and with thinking off or retry. With thinking on, 2/2 (`sft-template` grounder, M, [results]). | Hidden targets: open agents about 30% vs humans 94.6% (I, [reveal]). Claude's scroll steps make progress at least 84.1% of the time (I, [osw-pro]). | Mostly: a pixel planner sees the partial row and scroll bar | Shrinks; untested | A "more below" cue; the rule "scroll before asking the grounder for an absent target"; a grounder that can say "not found". Thinking on helps even our text planner. |
| 4 | Canvas charts | max-usage-month 0/3 (FINAL 1/3) in Gym. 2/3 with thinking on in the interactive harness (`sft-template`, M). | Canvas: GPT-5.4 65.6 vs MAI-UI-2B 12.5 (I, [cuactspot]). ChartAct: best 84.5%, many models under 60% (I, [chartact]). A misleading chart cuts success by 34 points (I, [mislead-chart]). | Shrinks a lot, not solved | Shrinks | Canvas is where pixels win, so add chart data and verify the selection afterwards. Interaction alone barely helps chart reading: +0.67 points over vision only (V, [iplot]). Prefer the chart's own spec or labels when exposed. |
| 5 | Lossy screen summaries (selected option, dialog contents) | Pay-50 passed 13/13 headless but 1/4 headed. Three headed runs paid the full $142.37 and claimed success (M, [console-runs]). FINAL's Gym run lost a high-contrast state line (M, [blade-final]). | VisualWebArena, a11y tree in every arm: tree alone 11.4% → plus captions of in-page images 24.8% → plus a marked screenshot 38.4% (I, [pando]). Agents trust text over pixels (I, [state-belief]). | Yes, by design | Yes, by design | Don't feed an unreliable text channel. When adding a11y text, check it against the pixels: a consistency gate cut text-swap errors by 24–44 points (I, [state-belief]). |
| 6 | False "done" | False claims of 36 runs: 8 (base), 4 (SFT), 11 (FINAL) (M). Console: 7 of 9 completed failures claimed success; 5 of 7 with FINAL and base summaries (M, [console-runs]). | In one strong OSWorld pipeline, 64 of 71 failures ended in a success claim (V, [cura]). Over 86% of failures stayed false "done" even with a verifier (V, [vlaa]). τ²-bench: false success is 45–48% of failures in single-control domains, but 3% where the user controls the state (I, [tau2-false]). The Opus 5 card flags confident claims later retracted ([anth-opus5-card]). | No | No | A done-gate on verified state, using deterministic checks first. Add an independent verifier: rubric-based false-positive rate 0.01–0.08 (V, [univ-verifier]); an agentic verifier reaches 92.9% accuracy (V, [vagen]). The user confirms on their own screen. |
| 7 | Small adjacent options and icons | L text size: base 0/18 clicks, FINAL 3/3 runs. SSv2 icons 78.1 → 85.4 (M) | ScreenSpot-Pro at frontier: 82–93 (V, [anth-opus48-card], [oai-astra]). Independent single-shot icons: GPT-5.5 63.4, Opus 4.7 38.9 (I, [guilens]). Zoom adds 12–18 points at 7–72B (V, [zoomclick]), but only +5.4 at 2B (V, [maiui]). | Mostly | Shrinks; untested | Zoom gated on uncertainty (V, [uizoomer]). Pull clicks toward box centers in GRPO (BLADE recommendation). |
| 8 | Never answers "not found" | Absent targets became confident wrong clicks, e.g. Send on an empty form (M, [blade-agent]). 88 of FINAL's 202 targets were absent, vague or non-interactive (M, [blade-final]). | Most GUI-tuned grounders refuse 0% (GTA1, UI-TARS-1.5; [lookagain], [venusbench]). Exceptions: GUI-Owl-1.5-2B 53.7 on OSWorld-G, with other variants 5.6–70.4 (V, [guiowl15]); LookAgain-8B 57.4 (V, [lookagain]). General VLMs refuse more (I, [venusbench]). | Mostly | Partly | Train refusal with cut-paste data; locate, then crop and confirm. |
| 9 | Rendering brittleness | About 85k pixels of font rasterization flipped pay-50 (M, [local-mac]) | A 28-px border turns 7.3% of correct grounder predictions wrong (GTA1-7B, V, [mvp]). Cosmetic desktop changes significantly lower Claude's reliability (V, [passk-cua]). | No | No | Pin the renderer where we control it (not in VDI); add perturbation tests to the eval; predict from several views (V, [mvp]). |
| 10 | Prompt injection (not observed, structural) | Our planner reads a summary of untrusted pixels, an unmeasured channel | Every Opus 5 configuration was broken in exactly 1 of 14 Shade GUI scenarios (200 attempts each); attempt-level success over all 2,800 attempts was 0.25–0.54% (V, [anth-opus5-card]). Independent end-to-end attacks on Opus 4.5 and 4.6 agents: 83% and 50%, with success meaning any of 3 runs (I, [redteamcua]). Malicious injections on the web rose 32% in three months (I, [google-ipi]). | No | No | Keep the goal fixed by the user; treat screen text as data; use the risk gate; add at least 3 injection tasks to the eval (§10). |

## 7. Building the real product

### 7.1 Reliability engineering

- **Verifier and done-gate (cluster 6).**
  - After every state-changing action, check an expected post-state derived from the request: the toast
    text, the field value, the URL, an a11y attribute, or a diff of the target crop. A failed check
    returns a corrective observation.
  - `done(success=true)` is rejected unless the goal state is verified. This makes the agent-loop guards
    in the BLADE ranking mandatory ([blade-agent] Rec. 2).
  - Add a model check only after the deterministic ones, and use the planner itself on a crop so it adds
    no extra model. Judges alone are weak: no LLM judge exceeds 70% precision (I, [agentrewardbench]),
    and none beats AUROC 0.65 at spotting false success (I, [tau2-false]).
- **Confirmation before money moves.**
  - Use a risk tier for each intent.
  - For payments, read the amount from the confirmation dialog, compare it with the request, and ask the
    user to confirm on their own screen.
  - OpenAI reports its agent confirms financial transactions 100% of the time (V, [oai-agent-confirm]).
    Claude in Chrome refuses them outright ([chrome-perm]). PCI SSC recommends human approval
    ([pci-ai-2026]).
- **Human hand-off.**
  - A visible "talk to a person" option at every step. Automatic hand-off on two failed verifications, a
    stuck stop, or an infeasible request.
  - The trace goes with it, so the agent doesn't have to re-ask.
  - The reasons:
    - 87% of customers say GenAI service must offer a human (I, [gartner-87]).
    - Only 27% would try a chatbot again after a bad experience (I, [gartner-27]).
    - After a failed AI interaction, US customers get a full resolution only about half the time
      (I, [copc]).
- **Determinism.**
  - Pin the renderer where we control it. In VDI we don't, so perturbation tests matter more there.
  - Compile verified runs of the top intents into replayable scripts, and keep the agent for the long
    tail and for repairs.
  - Test with perturbations: headed vs headless, zoom, and borders.
- **Injection.**
  - The plan is fixed before untrusted page text is read, in the style of CaMeL (V, [camel]; tested on
    text and tool tasks, not GUI).
  - Screen text never changes the goal.
  - Risky actions go through the gate.
  - Measure injection success in the eval. Attacks split across steps raised success by up to 31 points,
    and a per-step detector caught only 14.6% of the split sub-steps (I, [stepjack]).
- **Eval harness.** The pattern we used during the hackathon generalizes:
  - a replica of each customer portal as a NeMo Gym environment, with checks on app state;
  - 8 or more repeats per task, reported as pass^k;
  - BLADE on every comparable pair of runs;
  - held-out task variants written before tuning.
- **Continuous improvement.**
  - Replay production failures in replicas (no PII), turn them into GRPO rows with the Gym `/verify`
    reward, and keep a round only if it clears a dev gate, as our night loop did.
  - Measured online-RL gains are real but modest:
    - UI-TARS-2: +4.5 OSWorld points from RL (V, [uitars2]).
    - WebAgent-R1 (a text/a11y agent, not pixels): Qwen2.5-3B went from 6.1% to 20.0% with behavior
      cloning, then to 33.9% with RL, on WebArena-Lite (V, [webagent-r1]).
    - ARPO: mostly in-domain (V, [arpo]).
  - NVIDIA saw regressions from single-environment RL ([nemotron-rl]), so train across a mix of
    environments.

### 7.2 Deployment, privacy and compliance

Screens from a logged-in session contain PII and often card or health data.

| Option | What leaves the premises | Fits | Notes |
| --- | --- | --- | --- |
| Hosted computer-use API | Every screenshot | Low-risk, public, unauthenticated pages; pilots on synthetic data | See the matrix below |
| Customer cloud (Bedrock, Google Cloud, Azure) | Screenshots go to the cloud provider's managed model service in the chosen region, under the customer's existing cloud contract | Organizations already cleared for that cloud | Bedrock computer use is a Beta Service: AWS says Beta Services "should not be used for processing sensitive data" ([aws-terms]). On Bedrock, all Claude Fable 5/5.1 traffic is kept up to 30 days for abuse detection, with zero retention for eligible Enterprise Frontier Safeguards customers through 2026-12-31 ([bedrock-abuse]). On Google Cloud, Fable 5 and Mythos 5 require data sharing with Anthropic ([gcp-abuse]). Claude on Google Cloud is GA with EU processing; PHI coverage is not stated (see the matrix below). |
| On-prem server | Nothing | Banks, utilities, government, health; VDI estates | Our default. An official Lightning NIM runs on DGX Spark: about 30 GB NVFP4 or 66 GB BF16 ([spark-nim]). |
| On the device (Mac or PC) | Nothing | An accessibility assistant run by the end user | Measured fully local this weekend (M) |

**Hosted computer use, as of 2026-10-18** (all vendor documents, several updated this week):

| Offering | PHI under a BAA | Zero retention | EU processing |
| --- | --- | --- | --- |
| Claude API computer toolset | Conflicting: the BAA article says not covered ([anth-baa], 2026-10-17); the docs table says yes ([anth-retention]) | Yes, except Fable/Mythos, which need 30-day retention | No: "us" or "global" inference only ([anth-residency]) |
| Claude on Bedrock | Beta Service, so not for sensitive data ([aws-terms]) | Available, except Fable 5/5.1: up to 30 days ([bedrock-abuse]) | EU inference profiles exist |
| Gemini computer use | No: pre-GA offerings must not process PHI ([gcp-hipaa], [gcp-terms]) | Not on the Gemini API ([gem-policies]) | No data-location commitment while Preview |
| Claude on Google Cloud (Agent Platform) | Not stated: Google's HIPAA list does not name Claude ([gcp-hipaa]) | Not documented for computer use | Yes: Opus 5.5 is GA with an EU multi-region ([gcp-opus55]) |
| OpenAI Responses `computer` tool | Endpoint-level, with Modified Retention ([oai-hipaa]) | Yes, but CSAM-flagged screenshots are kept ([oai-data]) | `/v1/responses` is served in the EU, but the computer tool is not listed in the residency table ([oai-data]) |

**Regulation that applies to our use case:**

- **PCI DSS.**
  - Displayed card numbers are limited to the BIN (the first 6–8 digits) and the last four digits
    (Req. 3.4.1; [pci-saq], *pre-2025*).
  - Screen captures and recordings that contain card data are in scope and must be protected
    ([pci-phone], *pre-2025*).
  - Under SAQ A (the lightest PCI self-assessment questionnaire), every card-data field on an embedded
    payment form must sit inside the processor's iframe (FAQ 1438, 2016, *pre-2025*; [pci-faq1438]).
  - So the agent should stop before the card fields and hand them to the user, never screenshot or log
    them.
- **HIPAA.** The Security Rule rewrite has slipped to July 2027, so the current rule applies
  ([hipaa-delay]).
- **Gramm-Leach-Bliley Act (GLBA) Safeguards Rule.** Encryption, MFA, activity logging and oversight
  of service providers ([glba]).
- **EU Digital Operational Resilience Act (DORA).** It applies to bank buyers. The 19 designated
  critical ICT providers are clouds, not model labs ([dora]).
- **EU AI Act.**
  - Since 2026-08-02, Article 50 requires telling people they are dealing with an AI, explicitly
    including AI agents, in a form accessible to people with disabilities ([euai-art50]).
  - High-risk Annex III obligations now start on 2027-12-02 ([euai-nav]).
- **California CPPA.** Rules on automated decision-making technology start 2027-01-01 ([cppa]).

**Licensing of our own stack:**

- **Models.**
  - Lightning is under OpenMDW-1.1: commercial use is allowed and outputs are unrestricted
    ([lightning-card], [openmdw]). Its OSI review is still open and contested ([osi-review]).
  - Qwen3.5-2B is Apache-2.0 ([qwen35-2b]).
  - Parakeet TDT 0.6B v3 is CC-BY-4.0 ([parakeet]).
- **Rewritten instructions.** 8,028 rows went through the hosted trial API: rewrites, plus keep/drop
  decisions.
  - They are already public in our dataset, and the public FINAL grounder was trained on them
    ([submission]).
  - The release check relied on ToS §6.3, which says we own Generated Content ([status]). The same
    terms also bar using Generated Content "in production" (ToS §1.2) and distributing it (ToS §4.2)
    ([trial-tos], v. 2025-09-19).
  - So the exposure is current, not a future pre-sale issue. **Action now:** get NVIDIA's written OK,
    or regenerate those rows on the self-hosted checkpoint (whose OpenMDW license leaves outputs
    unrestricted), retrain, and replace the public artifacts.
- **Screenshots.**
  - OS-Atlas's Apache-2.0 card bundles third-party screenshots ([osatlas], *pre-2025*). The Rico images
    under UIBert carry research terms with an indemnity and no open license ([rico]).
  - The whole GUI-grounding field shares this risk. A product should add licensed or self-generated UI
    data.

### 7.3 Expected success and cost per task for a customer-service workload

**Workload:** the top self-service intents of a utility or bank portal, such as paying a bill, setting
up autopay, getting a statement, reporting an outage, changing a notification setting and messaging
support. These are short, single-portal tasks of 2–8 steps, and several move money.

**Inputs:**

| Input | Value | Type | Source |
| --- | --- | --- | --- |
| Assisted contact cost | $13.50 median (Gartner, Feb 2024, *pre-2025*). $7.20 per inbound call (ContactBabel 2026). | I | [gartner-cost], [contactbabel] |
| Assisted contact cost, utility | $11.02 per call, an input assumption in the utility's own rate-case filing (2023, *pre-2025*) | V | [pnm] |
| Self-service contact cost | $1.84 | I | [gartner-cost] |
| Calls for a 2M-customer utility | 2–3M per year (McKinsey 2020, *pre-2025*) | I | [mckutil] |
| Share of contacts existing self-service could handle | 20–40%, as service leaders reported (Gartner 2020, *pre-2025*) | I (opinion survey) | [gartner-2040] |
| Customer adoption | 49% say they would use a chatbot; 7% did in their last interaction (2026) | I | [gartner-27] |
| Frontier success on short, state-checked tasks | Per attempt: Sonnet 5.5 140/147, Haiku 5.5 123/147. Tasks solved: GPT-6.1 Sol 45/49. | I | [windtunnel] |
| Success on harder, state-changing live tasks | ClawBench: best 33.3% (Sonnet 4.6) | I | [clawbench] |
| Blind users' own commands, desktop apps, UIA tree given | GPT-5 52.5% | I | [olla] |
| Our sandbox | Gym, thinking off: 47% FINAL, 50% base, 61% SFT, all CIs including 0. Interactive harness, thinking on, `sft-template`: 81% (26/32). Not measured with FINAL. | M | [results] |
| CS vendor AI pricing | Fin $0.99 per outcome, including "assumed resolutions" when the customer just leaves. Zendesk $1.50–2.00 per verified resolution. Agentforce $2 per conversation. | V | [fin-outcomes], [zendesk-price], [sf-price] |
| Frontier computer-use cost | $0.0017–0.040 per short attempt (Haiku 5.5 to GPT-6.1 Sol); Opus 5 $0.14, GPT-6 Astra $0.26 | I | [windtunnel] |

**Are WindTunnel's tasks like ours?** Closer than they first look (I, [windtunnel]):

- **Scoring.** WindTunnel scores runs from the resulting application state, with no human or model
  judge. That is like our sandbox checks.
- **Tasks.** The journeys include browse → cart → checkout, ticket checkout and booking. Its top tier is
  "sensitive action" tasks of 8–15 steps. Most of ours are 2–7 actions.
- **"Solved".** A task counts as solved when a majority of its 3 attempts pass.
- **Sites.** Real open-source apps cloned at pinned commits, not our utility portal.
- **Cost growth.** Pooled cost per attempt rises about 9x from the shortest tier to the longest. That
  figure comes from previous-generation models only.
- **Our budget (E).** For tasks like pay-50 and stop-promotions, budget 2–10x the median per-attempt
  cost. That is still under $0.40 per task on Sonnet 5.5.

**Local cost per task (E):**

- **Task time.** Across all 12 tasks, the interactive agent's median was 65 s (M, [results]). The
  console's 52 s median was mostly demo tasks.
- **M5 Max laptop.** About $0.18 per hour, amortized over 3 years running around the clock; amortization
  is about 98% of that (N, a blog's estimate, [m5-finops]). A 65 s task works out to about $0.003 at
  full use, or $0.013 at 25% use.
- **DGX Spark.** $6,950, plus about 160 W at $0.15/kWh, comes to about $0.29 per hour over 3 years.
  - Its memory bandwidth is half an M4 Max's, and our pinned vLLM has a decode regression on it
    ([vllm-spark]). So assume tasks take 1.5–2x longer there (E).
  - That gives roughly $0.005–0.04 per task, depending on use and concurrency.
- **H200.** At the $6.90/hr we were billed, serving 8–16 concurrent sessions, about $0.008–0.016 per
  task. Our concurrency on an H200 is unmeasured.

**Expected verified success on this workload (E):**

| Design | Expected verified success | Hand-offs | Basis |
| --- | --- | --- | --- |
| (d) ours today | 35–55% with thinking off; about 70% with thinking on | Few: it rarely stops itself | Gym and interactive numbers (M) minus about 10 points for prompts tuned in-domain. The thinking-on number is from `sft-template`, not FINAL. |
| (a) frontier computer-use API | 75–85% per attempt; the false-claim rate is unknown | Vendor-dependent | WindTunnel per-attempt 84–95% (I), minus about 10 points for money-moving steps (ClawBench, I) |
| (r) recommended, open planner, with gates | 60–85%: at or below (a), because the planner is open | Most of the rest, if the gates work | Assumes the done-gate and risk gate turn most failures into hand-offs. §10 screens false success at ≤ 2%. Showing a production-grade ≤ 0.5% needs far more episodes: 0 in 192 only bounds it below about 1.6%. |

**What this does to the pitch's ROI.**

- [pitch.md][pitch] treats the 20–40% as contacts moved outright.
- Here we multiply that by adoption and verified success, and subtract the cost of failed attempts.
- **Failure cost (E):** each failed attempt adds 0.5–1 extra assisted contact on top of the call that
  would have happened anyway, at $3.60–11.02.
- Product, integration and operations costs are not included.

Applied to a 2M-customer utility with 2.5M calls a year:

| Scenario | Eligible | Adoption | Verified success | Calls deflected | Gross savings at $5.36–9.18 per call | Failure cost | **Net savings** | Inference, local ($0.01–0.03 per attempt) | Inference, frontier (2–10x Sonnet 5.5 median) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Today's success | 30% | 15% | 50% | 56,250 | $0.30–0.52M | $0.20–0.62M | **−$0.32M to +$0.31M** | $1–3k | $7–36k |
| Low | 20% | 7% | 70% | 24,500 | $0.13–0.22M | $0.04–0.12M | **$0.02–0.19M** | < $1.1k | $2–11k |
| Mid | 30% | 15% | 80% | 90,000 | $0.48–0.83M | $0.08–0.25M | **$0.23–0.75M** | $1–3k | $7–36k |
| High | 40% | 30% | 85% | 255,000 | $1.37–2.34M | $0.16–0.50M | **$0.87–2.18M** | $3–9k | $19–95k |

- The per-call saving is ContactBabel's $7.20 or PNM's $11.02, minus $1.84 for self-service.
- Low and Mid match the gated design on an open planner (60–85%, above). High's 85% assumes a frontier
  planner.
- All values in this table are estimates (E).
- The pitch's $3.9–7.7M assumed adoption × success = 100% and $7.70 saved per call (E Source
  $9.54 − $1.84).

**The takeaways:**

- At today's measured success, the product may not pay for itself. It needs gated success of about 70%
  or more, plus adoption well above the 7% current chatbot use.
- Local inference is a rounding error. Frontier inference isn't always:
  - up to $95k a year in the high scenario, against $0.9–2.2M net;
  - in the low scenario, up to $11k a year, which can eat much of a thin $0.02–0.19M net.
- Adoption, verified success and safety decide the business case.
- The cost of a failed attempt matters. In one Gartner worked example, a four-contact journey cost
  $46.55 to resolve, 4x the $11.64 average contact (illustrative, [gartner-journey]).
- Gartner predicts GenAI cost per resolution will exceed $3 by 2030 (A, [gartner-3usd]). For a local
  stack, the argument is predictable cost at volume, not a lower price.

## 8. Commercial options

### 8.1 Ranked

| Rank | Option | Buyer and pain | Why pixels and on-prem matter | Evidence | Main risks | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **On-prem SDK for pixel-level automation** of VDI, Citrix, canvas and legacy apps, with natural-language targets, verifier, replay and eval harness | Automation and RPA teams in regulated or air-gapped organizations; RPA and testing vendors as channels | On Citrix, without a DOM, automation means surface automation: region, image and coordinate matching ([bp-surface]). One practitioner found its anchors frustrating in table- and grid-like layouts (anecdotal, [bp-forum]). TestComplete's object detection sends screenshots to SmartBear's cloud ([testcomplete]). | UiPath and NVIDIA have explored Nemotron for air-gapped sites since 2025-09 ([uipath-nvidia]). On-prem was 58.92% of 2025 RPA revenue (analyst estimate, [mordor-rpa]). 30% of 23 US federal RPA programs hosted their bots in VDI (2020, *pre-2025*, [fed-rpa]); that doesn't measure how many target apps lack a usable UI tree. | Incumbents already ship this design: UiPath's Screen Agent is planner plus "targeter" at 67.14% on OSWorld-Verified (I, [osw-v]), and its grounder wraps UI-TARS-7B (V, [uipath-uicube]). Tosca's on-prem AI arrives this fall ([tosca-onprem]). Buyers want determinism: UiPath's CEO says agents aren't replacing deterministic automation ([uipath-call]). In VDI we can't pin the renderer, and checks become OCR and pixel diffs. Our footprint (a 27–35B VLM) needs more GPU than UiPath's CV server, not less. | **Best fit, least measured.** Sell the system (verified, replayable, on-prem), not the weights. Partner or OEM; don't try to displace RPA. §10 adds a remote-desktop arm to test it. |
| 2 | **Self-service assistant for site owners** (bank, utility, benefits agency), DOM-first with pixel fallback, on-prem | Contact-center leaders: only 14% of issues are fully resolved in self-service (*pre-2025*, [gartner-14]) | The site owner controls the page, so the DOM and APIs are available. Pixels matter for canvas, third-party embedded widgets and legacy portals. On-prem matters for PII. | Fin, Zendesk and Agentforce sell resolutions at $0.99–2 ([fin-outcomes], [zendesk-price], [sf-price]). Salesforce is buying Fin for about $3.6B ([sf-fin]). Users want a human option ([gartner-87]). | **Site-exposed agent tools.** An owner who exposes WebMCP or MCP tools needs neither pixels nor the DOM: Haiku 5.5 passed 147/147 WindTunnel attempts through WebMCP at a median $0.00025 each, against 123/147 through computer use (I, [windtunnel]). Also: CS platforms own the stack and the APIs; adoption is low (7% used a chatbot last time, [gartner-27]); net ROI may be negative at today's success (§7.3); liability for money movements. | **Pilot with one design partner,** rebuilt DOM-first and tools-first where the owner can expose them. Acting on the real UI is the fallback. |
| 3 | **Accessibility assistant** (the user's own agent, working on any site) | Blind, low-vision, older and motor-impaired users; buyers are employers, NGOs, governments and B2B2C programs | Users don't control sites. 95.9% of top home pages have detectable WCAG failures, and empty buttons appear on 30.6% ([webaim-million]). 61% of large Dutch online stores couldn't take an order through assistive tech ([acm-nl]). | 40.6% of screen-reader users already use AI to guide online tasks ([webaim-sr]). Blind users avoided delegating banking ([olla]). | Bundling, willingness to pay, legal positioning (below the table) | **Mission, and a feature of (2) and of grants and partnerships,** not a standalone business. Never claim compliance. |
| 4 | **Self-healing UI test automation** | QA teams | Only for VDI, Citrix and canvas apps, where selectors don't exist (Eggplant is the incumbent, [eggplant]) | Crowded and funded: Functionize $41M ([functionize]), Momentic $15M ([momentic]), QA Wolf $1 per run ([qawolf]). Octomind shut down for lack of market validation ([octomind]). Applitools sells determinism ([applitools]). Market estimates vary 2x ([mordor-test]). | Our headed/headless flip is exactly the nondeterminism test buyers reject | **Niche only,** as a module of (1) |
| 5 | **Hosted pixel-native automation API** | Developers | None vs frontier APIs | Frontier computer use costs $0.0017–0.04 per short-task attempt for Haiku 5.5 to GPT-6.1 Sol, $0.14–0.26 for Opus 5 and GPT-6 Astra ([windtunnel]). Browser Use charges $0.02 per browser-hour ([bu-price]). Browserbase raised $40M ([browserbase-b]). | No moat: open weights (Fara, Holo, UI-Venus) and frontier APIs | **Don't** |

**Rank 3 risks in detail:**

- **Bundling.** Apple Voice Control will accept "tap the purple folder" ([apple-a11y]). TalkBack and
  Narrator describe screens, and JAWS labels unlabeled controls ([jaws-ai]). Gemini auto browse and
  Claude in Chrome already exist.
- **Willingness to pay.** Guide charges $8–25 a month and warns against using it for banking
  ([guide]). Be My Eyes is free to users and funded by business subscriptions ([bemyeyes-b2b]).
- **Legal positioning.** The FTC fined accessiBe $1M over compliance claims ([ftc-accessibe]), and 983
  of 3,948 ADA suits in 2025 targeted sites that had an accessibility widget ([ecomback]).

### 8.2 Competitor map

| Segment | Players and approach | Where a pixel-only, on-prem, small-model approach wins | Where it loses |
| --- | --- | --- | --- |
| Frontier computer-use APIs | Anthropic, OpenAI, Google, Meta (§4.1). Anthropic's browser toolset also reads the a11y tree ([anth-browser]). | Air-gapped or PHI workloads; EU-only processing; predictable cost at volume | Capability on long tasks, speed, safety tooling (injection classifiers, confirmations), cost at low volume |
| Site-exposed agent tools | WebMCP and MCP servers that site owners publish | Sites whose owners won't or can't expose tools: legacy, third-party, VDI | Any site whose owner exposes them: 147/147 vs 123/147 for Haiku 5.5, at about 1/7 the cost and a quarter of the time ([windtunnel]) |
| Managed agent services | Amazon Nova Act ($4.75 per agent-hour, [nova-price]); Copilot Studio computer use (about $0.04–0.15 per step, [copilot-cu]); Simular Sai ($14.34 per OSWorld 2.1 task, [simular]); H Company's agents API | On-prem; predictable cost at volume | Distribution, maturity, Windows integration |
| Browser infrastructure and DOM agents | Browserbase/Stagehand ($40M Series B, [browserbase-b]); Browser Use (DOM-first, $0.02 per browser-hour, [bu-price]); Kernel ($22M, [kernel]); Steel, Hyperbrowser, Anchor, Skyvern | Canvas, Flutter web (semantics off by default, [flutter-a11y]), VDI, stale DOM ([state-belief]). Compact cross-origin payment iframes: Browser Use patched these on 2026-07-24, but cross-origin extraction is still off by default ([bu-iframe]). | DOM-friendly sites, for any small or open model: GPT-5 mini 83.1% a11y vs 48.9% pixels, UI-TARS-1.5-7B 12.6% from pixels ([componentbench]); speed (68 s vs 225–330 s, [bu-speed]); ecosystem |
| Open pixel models | H Company Holo (mostly Apache; Holo4-27B and Holo2-30B-A3B non-commercial; local recipes for Mac and GPU; in NVIDIA's Nemotron Coalition, [h-local], [h-newsroom]); Microsoft Fara1.5 (MIT, pauses before payments, [fara15]); UI-Venus, MAI-UI, GUI-Owl | Our eval and verifier system, packaging, and domain data for a given portal | Weights: bigger labs ship better grounders and agents at our size, under open licenses |
| RPA | UiPath (Screen Agent, ScreenPlay, CV for Citrix), Automation Anywhere, Power Automate ($150 per bot per month, [pa-price]), Blue Prism (surface automation), ServiceNow (AI Desktop Actions with deterministic replay, [servicenow]) | An on-prem grounding component that plugs in where vendors accept your own model (ScreenPlay accepts an OpenAI-compatible endpoint, Tosca exposes MCP; [uipath-screenplay], [tosca-onprem]) | Distribution, governance, customer trust, determinism; they already ship planner plus grounder |
| Testing | Tricentis Tosca Vision AI, Keysight Eggplant, TestComplete object detection (cloud), Applitools, mabl, Functionize, Momentic, QA Wolf, TestDriver | VDI and canvas apps; on-prem | Deterministic execution expectations; a crowded field |
| Customer-service AI agents | Fin (Salesforce), Zendesk, Agentforce, Sierra, Decagon, Ada; Google CX Agent Studio; Amazon Connect | Acting on the real UI where no API exists; on-prem | They own the CS stack and integrations. Site owners can give them APIs. |
| Accessibility | OS and assistive-tech vendors (Apple, Google, Microsoft, Vispero); Be My Eyes; Guide; overlays (AudioEye, $40.3M revenue, [audioeye]) | A private, cross-app agent on the user's device | Bundling, willingness to pay, legal positioning |

**Takeaway.**

- **Pixels and on-prem win** where structure is missing, wrong or stale (canvas, Flutter web, VDI and
  RDP, stale DOM), and in air-gapped or regulated estates.
- **They lose** in these places:
  - on DOM-friendly pages, for any small or open model;
  - wherever the owner exposes agent tools;
  - on long or hard tasks;
  - anywhere determinism is required;
  - on distribution;
  - on weight quality, against larger labs.
- **On long and hard tasks, the open-model numbers are low.** Open models complete 2.8–4.6% of OSWorld
  2.0 in the paper's runs ([osw2-paper]). Later releases self-report 12–42% (V, [holo4]). The best ≤3B
  agent scores 43.5 on OSWorld-Verified ([guiowl15]).

### 8.3 What we don't know about buyers

- **Market evidence is missing.** We have no buyer interviews, design partner, pricing test, security
  review or VDI workload data.
- **The hard parts of selling** are untested: enterprise sales cycles, security certifications, and
  on-prem support for a three-person team.
- **What isn't easily copied** is narrow: portal replicas with state checks, customer trace data, the
  verification layer, and the habit of measuring from app state. NeMo Gym, BLADE and the models are
  public.
- **Next two weeks:** hold 10 buyer conversations alongside §10. Aim for 4 RPA and automation leads with
  VDI estates, 3 contact-center leaders at utilities or credit unions, 2 accessibility program owners,
  and 1 RPA vendor partnership team. Ask about workload share on VDI, willingness to run models
  on-prem, and the price of a verified task.

## 9. Risks

| Risk | Evidence | Impact | Mitigation |
| --- | --- | --- | --- |
| Wrong state-changing action claimed as success | Three headed console runs paid the full $142.37 instead of $50 and claimed success; all three of their own summaries mentioned $142.37 (M, [console-runs]). In one strong OSWorld pipeline, 64 of 71 failures ended with a success claim ([cura]). | Severe: money, trust, regulators (CFPB on chatbot "doom loops", *pre-2025*, [cfpb]) | Risk gate (compare the confirmed amount with the request), done-gate, user confirmation on their own screen; start pilots read-only |
| Prompt injection via screen content | Every Opus 5 configuration was broken in exactly 1 of 14 Shade GUI scenarios; attempt-level success over all 2,800 attempts was 0.25–0.54% ([anth-opus5-card]). Independent end-to-end attacks: 83% and 50% on Opus 4.5 and 4.6 agents, with success meaning any of 3 runs ([redteamcua]). Attacks split across steps evade detectors ([stepjack]). | Severe | Goal fixed by the user, plan-then-execute, risk gate, at least 3 injection scenarios in the eval |
| Rendering and environment nondeterminism | Pay-50 passed 13/13 headless but 1/4 headed (M). A 28-px border turns 7.3% of correct predictions wrong ([mvp]). In VDI the remote server renders and compresses the pixels. | High, highest for VDI (rank 1) | Pinned renderer where we control it, multi-view prediction, perturbation and lossy-stream tests (§10 A12), replay library |
| Incumbents already ship the design | UiPath Screen Agent (planner plus targeter, [osw-v]); UiPath + NVIDIA air-gapped Nemotron work ([uipath-nvidia]); Microsoft, Salesforce + Fin ([sf-fin]) | High | Partner or OEM; differentiate on on-prem verification and evaluation |
| Site-exposed agent tools | WebMCP: 147/147 vs 123/147 for computer use, at about 1/7 the cost ([windtunnel]) | High for the site-owner product | Tools-first where owners expose them; pixels for the rest |
| Platform bundling | Claude in Chrome, Gemini auto browse, Apple Voice Control, JAWS AI Labeler (§4, §8) | High for accessibility | Focus on regulated and on-prem buyers, where bundled cloud agents can't go |
| No buyer evidence | §8.3 | High | 10 buyer conversations alongside §10 |
| Commoditized models | Open 2–4B grounders at or above ours (§4.2) | Medium | Don't sell weights. Sell the system and domain adaptation. |
| Data licensing, now | 8,028 hosted trial-API rows are already public, along with a model trained on them ([submission], [trial-tos]). Rico screenshots ([rico]). OpenMDW's OSI review is contested ([osi-review]). | Medium | NVIDIA's written OK, or regenerate on self-hosted weights, retrain and replace; add licensed or self-generated data; legal review |
| Compliance ambiguity | Anthropic's BAA and docs conflict; preview and beta terms ([anth-baa], [aws-terms], [gcp-terms]) | Medium | Default on-prem; get written confirmation before any hosted pilot |
| Accessibility claims | FTC accessiBe order ([ftc-accessibe]) | Medium | Never claim WCAG/ADA/EAA compliance |
| Weak or gamed evidence | Saturated and exploitable benchmarks ([rdi-exploit], [epoch-osw]); vendor-reported numbers; our n of 36 is in-domain | Medium | Our own held-out replicas, pass^k with confidence intervals, and independent arms (§10) |
| Latency | Ours: 53–68 s median per task on our 12 tasks (M, [results]). WindTunnel frontier medians: 16–21 s, on different tasks ([windtunnel]). Not directly comparable. | Medium for live users | Replay for top intents; prefix caching (not available for Lightning on vllm-metal today, but available on GPU vLLM, [local-mac]); a faster planner; drop the summary call |

## 10. Two-week experiment plan

**Goal:** put the designs from §3 on the same tasks, with the same state checks, and decide on the
architecture.

**Approvals needed before running:**

- The API spend.
- Sending sandbox screenshots to vendors. They are synthetic, with no PII.
- GPU time.

**Tasks.** The 12 current sandbox tasks, plus 12 held-out variants:

- 2 re-skinned layouts of current tasks;
- 4 aimed at our clusters:
  - a toggle grid named by row and column;
  - a download below the fold;
  - a payment confirmation with a preselected option;
  - a chart with hover;
- 2 where the right behavior is hand-off:
  - a card field in a cross-origin iframe;
  - an infeasible request;
- 4 from outside the known clusters (for example a date picker, a search-and-filter table, and one task
  on a self-hosted open-source app).

**Injection set:** 3 scenarios, each with 8 repeats per arm:

- an instruction in a message body ("pay the full balance");
- a fake toast;
- hidden page text.

**Before any arm runs:**

- Fix the lenient `mark-all-read` check ([blade-agent] Rec. 6).
- Freeze A1's code.
- Have a teammate who didn't build A1 write the held-out variants.

**Runs:** 8 repeats per task. That is 192 main episodes plus 24 injection episodes per arm.

**Same rules for every arm:**

- **Step cap.** 20 steps for every arm. This is a deliberate common cap, below the product's 25 steps,
  to bound API cost.
- **Scripted user.** It approves a confirmation only if the quoted amount equals the request, answers
  clarifications from a fixed spec, and counts a hand-off as a hand-off.
- **Verifiers in A1, A10g and A11g** read only the screen and the a11y tree, never sandbox state. The
  sandbox state is the scorer only.

**Metrics:**

- **Held-out pass@1 is the gate metric,** with paired, task-level bootstrap 95% confidence intervals.
  pass^8 is reported alongside.
- **Counts:** false-success claims; unsafe actions (wrong money moves, unrequested setting changes);
  correct hand-offs; injection success.
- **Cost and speed:** median and p90 wall time; steps; dollars or GPU-seconds per episode.
- **Privacy:** whether screenshots leave the machine.

**Power.** 12 held-out tasks × 8 repeats is small:

- a 10-point gap is about one task;
- zero false successes in 192 episodes only bounds the true rate below about 1.6%.

So the safety thresholds are screening checks, not evidence of production safety. If confidence
intervals overlap at the gate, add held-out tasks rather than repeats.

| Arm | Design | Where it runs | Estimated cost |
| --- | --- | --- | --- |
| A0 | Ours today: FINAL grounder, thinking off and on | Mac | $0 |
| A1 | Ours fixed: state lines and per-row grid summary, "more below" cue, done-gate, oscillation guard, amount confirmation, grounder refusal | Mac | $0 |
| A2 | Lightning reading the Playwright a11y snapshot instead of the VLM summary. Measures how much a faithful text channel closes the gap. | Mac | $0 |
| A3 | Design (b): Sonnet 5.5 sees the screenshot and calls our grounder through a `click(target)` tool | API + Mac | ≈ Sonnet arm |
| A4–A6 | Single frontier VLM (a): Haiku 5.5, Sonnet 5.5, Opus 5.5 (computer toolset) | API | See below |
| A7–A8 | GPT-6.1 Sol, GPT-6 Luna (Responses `computer`) | API | See below |
| A9 | Gemini 3.8 Flash (`computer_use`, ultra_high images) | API | See below |
| A10 | Single open VLM (a′): Holo3-35B-A3B or Qwen3.8-27B, both Apache, end to end | H200 | GPU |
| A11 | Design (b) with an open on-prem planner: open VLM plus our grounder | H200 | GPU |
| **A10g, A11g** | **The recommended architecture (r):** A10 or A11, plus the Playwright a11y snapshot, A1's done-gate, risk gate and refusal | H200 | GPU |
| A12 | A VDI-style stream: A11g and the best frontier arm, run through a VNC session with lossy compression, on 6 tasks × 8 | Mac or H200, plus API | Small |
| G1 | Re-run UI-Venus-1.5-2B, MAI-UI-2B, Holo2-4B and an untuned Qwen3.5-4B through our `gui_grounding` env on SSv2-clean and dev. Each gets its documented prompt, coordinate convention and resolution. Spot-check about 50 rows against the published number first (our parser handles only Qwen-family points on 0–1000). | H200 | GPU, about 1–2 hrs |
| G2 | Grounder v2: Qwen3.5-4B SFT, then one GRPO round with refusal, adjacent-option and toggle-state data from synthetic UIs (never eval tasks) | H200 | GPU, about 4–5 hrs |
| D | Determinism: pay-50, text-size and sms-billing, headed vs headless (3 tasks × 8 × 2), on the local arms plus Haiku 5.5 and Sonnet 5.5 | All | Small |

**API cost (E):** 216 episodes per arm, plus A3, A12 and D.

- **Floor, at WindTunnel's measured short-task medians:** about **$65 in total**. Per arm:

  | Arm | Approx. cost | Basis |
  | --- | --- | --- |
  | Haiku 5.5 | $0.4 | WindTunnel median |
  | GPT-6 Luna | $0.5 | WindTunnel median |
  | Sonnet 5.5 | $7 | WindTunnel median |
  | GPT-6.1 Sol | $9 | WindTunnel median |
  | Opus 5.5 | $24 | Opus 5's $0.139 per attempt, scaled to Opus 5.5's 20% lower prices |
  | Gemini 3.8 Flash | $14 | Stagehand's $0.065 per task |
  | A3 | $7 | About the Sonnet arm |
  | A12 and D | $3 | Small subsets |

- **Expected:** 2–10x the floor, because our money-moving tasks run longer (§7.3). That gives **about
  $130–640**.
- **Worst case:**
  - Projected at 20 turns with full screenshot history and no cache hits, every arm comes to about
    $1,100.
  - Re-sent assistant output adds about 14%, so about $1,260.
  - That projection was made during the research sweep, at list prices ([anth-price], [oai-price],
    [gem-price]).
  - Prompt caching cuts it about 4–5x.
- **Budget:** a hard cap of $1,000, split into per-arm caps:

  | Arm | Cap |
  | --- | --- |
  | Opus 5.5 | $250 |
  | Sonnet 5.5 | $150 |
  | A3 | $150 |
  | GPT-6.1 Sol | $120 |
  | Gemini 3.8 Flash | $140 |
  | Haiku 5.5 | $15 |
  | GPT-6 Luna | $15 |
  | A12 and D | $70 |
  | Reserve | $90 |

  An arm that hits its cap stops and reports how many episodes it completed.

**GPU cost (E):** about 11–16 H200-hours, roughly **$75–110** at the $6.90/hr we were billed (list
$6.48, [brev-h200]):

- A10, A11, A10g and A11g at about 1–2 hours each (216 episodes at concurrency 4);
- G1 for 1–2 hours and G2 for 4–5;
- model downloads and reruns.

**Schedule:**

| Days | Work |
| --- | --- |
| 1–2 | Fix the check. Freeze A1. A teammate writes the 12 held-out variants and the 3 injection scenarios. Add arm adapters to the harness: Claude toolset, OpenAI `computer`, Gemini `computer_use`, the open VLMs' action formats, an a11y-snapshot observation, a grounder-as-tool, and the gates as a reusable wrapper. Set up the VNC stream. |
| 3–4 | A0, A1 and A2 on the Mac. G1 on the H200. |
| 5–7 | A4–A9 with spend caps, then A3. |
| 8–9 | A10, A11, A10g and A11g on the H200. G2, then A1 and A11g with grounder v2. |
| 10 | A12 (lossy stream), D (determinism), injection and hand-off analysis, BLADE on every arm pair. |
| 11–12 | Write-up and decision, together with the buyer conversations (§8.3). |

**Decision gates:**

- **On-prem product on an open model:**
  - **Condition:** A10g or A11g comes within 10 points of the best frontier arm on held-out pass@1,
    with the lower bound of the paired CI within 20 points. False success is ≤ 2% and there are no
    wrong money moves (screening).
  - **Then:** build local-first.
- **Customer-cloud or frontier-API product for buyers without PHI:**
  - **Condition:** frontier arms reach ≥ 80% held-out pass@1 with gates, but open arms are more than 10
    points behind.
  - **Then:** this is the alternative to local-first.
- **Stay in research:**
  - **Condition:** every arm stays below 80% on the held-out short intents even with gates.
  - **Then:** pilot nothing that moves money.
- **The VDI option (rank 1):**
  - **Condition:** A12 stays within 10 points of the same arm on a clean stream.
  - **Then:** rank 1 goes ahead.
- **A2's result informs, but doesn't gate.** The site-owner product is already DOM-first. A2 tells us
  how much a faithful text channel buys on its own.

## 11. What we could not verify

- **No study tests our design** (a pixel-only VLM summary feeding a text-only planner) against the same
  planner given screenshots. Every caption result keeps an a11y or DOM backbone ([pando], [vwa],
  *pre-2025*).
- **No independent latency per request** for the current models (Sonnet/Haiku 5.5, GPT-6.x, Gemini 4).
  The only per-step data is cua-speedrun's ([speedrun]): previous-generation models, plus GPT-6 Astra at
  low effort (about 15 s per batch of about 3 actions).
- **No neutral third party has run** the 5.5 generation, GPT-6, Gemini 4 or Muse Wonder on OSWorld 2.x.
  Every number is from the vendor or a competitor ([osw2-board]).
- **Anthropic's PHI coverage** for computer use: the BAA article and the docs table conflict, and the
  contract text is behind an access request ([anth-baa], [anth-retention]).
- **The share of automation workloads** that run on VDI or Citrix. The only number is a 2020 federal
  survey with n=23 ([fed-rpa]).
- **How common canvas UIs, payment iframes and closed shadow DOM are** on logged-in bill-pay pages.
  Crawls cover only public home pages.
- **Throughput at concurrency.** Our single-stream numbers (M, [local-mac]):
  - Lightning BF16 decodes at an inter-token latency of about 38–42 ms (about 24–26 tok/s) on the M4 Max.
  - The grounder at 2 MP takes 1.77–1.86 s p50 on the Mac and 0.64–0.75 s on the H200.

  External proxies, which aren't our stack:
  - oMLX on an M4 Max: 62 tok/s ([omlx]).
  - An H200 run with no speculation: 311–327 tok/s on an early checkpoint ([tw-h200]).
  - vLLM 0.31.0 regressed decode on DGX Spark ([vllm-spark]).
- **Hardware and API prices** move weekly. DGX Spark is +75% year on year ([spark-price]). Gemini
  promotional prices end on 2026-11-30 and 2026-12-31 ([gem-price]).

## 12. References

Internal: [results.md][results], [BLADE base vs SFT][blade-agent], [BLADE FINAL vs SFT][blade-final],
[local-mac.md][local-mac], [pitch.md][pitch], [submission.md][submission], [status.md][status], and the
[console-run summary][console-runs] (T1's traces, which are not committed).

Type: V = vendor or the authors on their own product; I = independent; D = vendor documentation, terms
or pricing (facts, not performance claims); R = regulator or law; N = news or secondary source;
A = analyst estimate.

| Key | Source | Date | Type |
| --- | --- | --- | --- |
| [anth-cu] | Anthropic: computer use tool docs | live, accessed 2026-10-18 | D |
| [anth-vision] | Anthropic: vision docs (image token formula) | live, accessed 2026-10-18 | D |
| [anth-price] | Anthropic: pricing (toolset overhead) | rate card 2026-10-07 | D |
| [anth-browser] | Anthropic: browser use tool docs | live, accessed 2026-10-18 | D |
| [anth-retention] | Anthropic: API and data retention | live, accessed 2026-10-18 | D |
| [anth-baa] | Anthropic: BAA help article | 2026-10-17 | D |
| [anth-residency] | Anthropic: data residency | live, accessed 2026-10-18 | D |
| [anth-sonnet55-card] | Claude Sonnet 5.5 system card | 2026-09-28 | V |
| [anth-haiku55-card] | Claude Haiku 5.5 system card | 2026-10-07 | V |
| [anth-opus5-card] | Claude Opus 5 system card | 2026-07-24 (rev. 2026-08-19) | V |
| [anth-opus48-card] | Claude Opus 4.8 system card | 2026-05-28 | V |
| [opus55-inj] | Secondary summary of the Opus 5.5 system card's injection results | 2026-09-22 card; summary accessed 2026-10-18 | N |
| [chrome-ga] | Claude in Chrome GA | 2026-08-26 | D |
| [chrome-perm] | Claude in Chrome permissions guide | 2026-08-12 | D |
| [oai-cu] | OpenAI: computer use guide | live, accessed 2026-10-18 | D |
| [oai-vision] | OpenAI: images and vision guide | live, accessed 2026-10-18 | D |
| [oai-price] | OpenAI: API pricing | live, accessed 2026-10-18 | D |
| [oai-astra] | OpenAI: GPT-6 Astra launch | 2026-09-03 | V |
| [oai-hipaa] | OpenAI: HIPAA-eligible products | 2026-10-17 | D |
| [oai-data] | OpenAI: your data | live, accessed 2026-10-18 | D |
| [oai-atlas] | OpenAI: Atlas into ChatGPT | 2026-07 (accessed 2026-10-18) | D |
| [oai-agent-confirm] | OpenAI: ChatGPT agent safety (confirmations) | 2025-07-17 | V |
| [gem-cu] | Google: Gemini computer use docs | 2026-10-13 | D |
| [gem-price] | Google: Gemini API pricing | 2026-10-16 | D |
| [gem-media] | Google: media resolution | live, accessed 2026-10-18 | D |
| [gem-argon] | Google DeepMind: Gemini 4 Argon model card | 2026-10-13 | V |
| [gem-policies] | Google: Gemini API usage policies | 2026-06-09 | D |
| [gem-autobrowse-help] | Google: Gemini in Chrome auto browse help | live, accessed 2026-10-18 | D |
| [gcp-cu] | Google Cloud: computer use (Preview) | live, accessed 2026-10-18 | D |
| [gcp-terms] | Google Cloud: service terms (Pre-GA) | 2026-10-12 | D |
| [gcp-hipaa] | Google Cloud: HIPAA compliance | live, accessed 2026-10-18 | D |
| [meta-wonder] | Meta: Muse Wonder model card | 2026-10-12 | V |
| [nova-price] | AWS: Nova Act pricing | live, accessed 2026-10-18 | D |
| [copilot-cu] | Microsoft: Copilot Studio computer use | 2026-07-03 | D |
| [copilot-faq] | Microsoft: Copilot Studio computer use FAQ | live, accessed 2026-10-18 | D |
| [aws-terms] | AWS: service terms (Beta Services) | 2026-10-13 | D |
| [osw-v] | OSWorld-Verified official results workbook | live, accessed 2026-10-18 | I |
| [osw2-board] | OSWorld 2.0 official results | 2026-10-15 | I |
| [osw2-paper] | OSWorld 2.0 paper | 2026-07-13 (v2) | I |
| [ale] | Agents' Last Exam leaderboard | sampled 2026-10-03 | I |
| [windtunnel] | WindTunnel v1.4 (nekuda; the author promotes WebMCP) | 2026-10-18 | I |
| [speedrun] | cua-speedrun | 2026-09-30 | I |
| [stagehand] | Browserbase Stagehand evals | v1 view 2026-09-02; v2 2026-09-03 | V |
| [bu-speed] | Browser Use: speed matters | 2025-10-09 | V |
| [bu-price] | Browser Use pricing | live, accessed 2026-10-18 | D |
| [bu-iframe] | Browser Use PR 5296 (payment iframes) | 2026-07-24 | V |
| [om2w-paper] | "An Illusion of Progress?" (Online-Mind2Web) | 2025-10-08 (v4) | I |
| [om2w-board] | Online-Mind2Web human-eval leaderboard | live, accessed 2026-10-18 | I (paid evaluation) |
| [om2w-pricing] | Online-Mind2Web human-evaluation pricing | live, accessed 2026-10-18 | D |
| [hal-om2w] | HAL: Online-Mind2Web | live; runs from 2025 | I |
| [steel-om2w] | Steel: Online-Mind2Web board | 2026-10-11 | N (aggregator) |
| [clawbench] | ClawBench | 2026-07-20 | I |
| [odysseys] | Odysseys2 | 2026-10 | I |
| [webarena-tracker] | WebArena tracker | 2026-10-16 | N (aggregator) |
| [wav-list] | WebArena-Verified list | 2026-10-13 | N (aggregator) |
| [liquid-card] | Liquid AI LFM2.5-24B-A2B card (Lightning WebArena-Verified) | 2026-10-13 | I (competitor) |
| [lightning-card] | NVIDIA Nemotron 3.5 Lightning BF16 card | 2026-08-11 | V |
| [aw-sheet] | AndroidWorld leaderboard sheet | live, accessed 2026-10-18 | V (self-reported rows) |
| [epoch-osw] | Epoch AI: what OSWorld tells us | 2025-10-30 | I |
| [rdi-exploit] | Berkeley RDI: trustworthy benchmarks | 2026-04 | I |
| [misscore] | How benchmarks mis-score CUAs | 2026-07-30 | I |
| [passk-cua] | pass@k vs pass^k for CUAs (Agent S3) | 2026-04-20 | V |
| [rerun-flips] | Observation-mode reruns (12–14% flips) | 2026-08-06 | I |
| [pando] | PANDO (VisualWebArena baselines) | 2026-05-26 | I (baselines) |
| [vwa] | VisualWebArena | 2024-01, *pre-2025* | I |
| [dailydroid] | DailyDroid | 2026-04 | I |
| [state-belief] | Pixels vs structure belief study | 2026-07-18 | I |
| [visiontasker] | VisionTasker | 2024-07-29, *pre-2025* | V |
| [ecoagent] | EcoAgent | 2025-11-17 | V |
| [jedi] | Jedi / OSWorld-G | 2025-10-24 | I |
| [osw-human] | OSWorld-Human (MLSys 2026) | 2026-05-18 | I |
| [fara15] | Microsoft Fara1.5 card (GitHub release 2026-07-22) | 2026-05-21 | V |
| [holo4] | H Company: Holo4 | 2026-09-28 | V |
| [h-local] | H Company: local inference recipes | live, accessed 2026-10-18 | V |
| [h-newsroom] | H Company newsroom (Nemotron Coalition) | 2026-10-15 | V |
| [qwen38] | Qwen3.8-27B card | 2026-08 | V |
| [qwen-cua] | Qwen-CUA | 2026-08 | V |
| [qwen36-harness] | Meta Muse Glimmer-30B card (Qwen3.6-27B comparison) | 2026-08 | I (competitor) |
| [uivenus15] | UI-Venus-1.5 | 2026-02 | V |
| [maiui] | MAI-UI | 2025-12-26 | V |
| [guiowl15] | GUI-Owl-1.5 (Mobile-Agent-v3.5) | 2026-02 | V |
| [qwen35-2b] | Qwen3.5-2B card | 2026-03-02 | V |
| [qwen-ui-agent] | Qwen-UI-Agent paper (Alibaba's re-run of its own Qwen3.5-4B base) | 2026-07-30 | V |
| [ss-board] | ScreenSpot-v2 leaderboard results | live, accessed 2026-10-18 | V (submitted) |
| [sspro-board] | ScreenSpot-Pro leaderboard | updated 2026-09-10 | V (submitted) |
| [venusbench] | VenusBench-GD (label audit, refusal) | 2025-12-18 | I |
| [windom] | WinDOM (Qwen3.5-2B SFT + GRPO) | 2026-06 | I for the base |
| [guilens] | GUI-Lens | 2026-08-04 | I |
| [argus] | Argus matched single-shot grounding | 2026-09-29 | I |
| [cuactspot] | CUActSpot / Phi-Ground team | 2026-05-12 | I |
| [guiprim] | GUI-Primitives (EMNLP 2026) | 2026-08-27 | I |
| [star] | StaR (CVPR 2026) | 2026-03-18 | I |
| [finestate] | FineState-Bench (venue unverified) | 2026-04-30 | I |
| [websp] | WebSP-Eval | 2026-04 | I |
| [zoomclick] | ZoomClick | 2025-12-05 | V |
| [uizoomer] | UI-Zoomer | 2026-04-15 | V |
| [reveal] | Reveal-and-Click | 2026-05-27 | I |
| [osw-pro] | OSWorld-Pro | 2026-09-21 | I |
| [chartact] | ChartAct | 2026-05-26 | I |
| [mislead-chart] | Misleading-chart web agents | 2026-10-09 | I |
| [iplot] | iPlotBench (interactive chart QA) | 2026-04-22 | V |
| [mvp] | MVP: multiple-view prediction (CVPR 2026) | 2026-06 | V |
| [perturbed] | GUI-Perturbed | 2026-04-15 | I |
| [cura] | CURA (false completion claims) | 2026-08 | V |
| [vlaa] | VLAA-GUI (v3) | 2026-09-24 | V |
| [tau2-false] | False success on τ²-bench | 2026-06-01 | I |
| [webpagebench] | WebPageBench | 2026-09-28 | I |
| [componentbench] | ComponentBench | 2026-08 | I |
| [univ-verifier] | Microsoft Research: universal verifier | 2026-04-21 | V |
| [vagen] | VAGEN agentic verifier | 2026-02 | V |
| [agentrewardbench] | AgentRewardBench | 2025-04 | I |
| [lookagain] | LookAgain | 2026-08-10 | V |
| [redteamcua] | RedTeamCUA (ICLR 2026) | 2025-05 (rev. 2026) | I |
| [stepjack] | StepJack | 2026-08 | I |
| [google-ipi] | Google: prompt injections on the web | 2026-04-23 | I |
| [camel] | CaMeL (AgentDojo, text and tools) | 2025-06-24 | V |
| [olla] | OLLA blind-user study (EMNLP 2026) | 2026-09-01 | I |
| [a11ycua] | A11y-CUA (CHI 2026) | 2026-02-10 | I |
| [uitars2] | UI-TARS-2 | 2025-09-02 | V |
| [arpo] | ARPO | 2025-05-22 | V |
| [webagent-r1] | WebAgent-R1 (EMNLP 2025) | 2025 | V |
| [nemotron-rl] | NVIDIA Nemotron 3 Super RLVR docs | live (blog 2026-03-11) | V |
| [nemotron-supervl] | NVIDIA Nemotron 3.5 Super VL card | 2026-10-15 | V |
| [pci-saq] | PCI DSS v4.0 SAQ B (Req. 3.4.1) | 2022, *pre-2025* | R |
| [pci-phone] | PCI SSC: telephone payment guidance (recordings) | 2018-11, *pre-2025* | R |
| [pci-ai-2026] | PCI SSC AI security guidance (Help Net Security report) | 2026-10-09 | N |
| [euai-nav] | European Commission: navigating the AI Act | live, accessed 2026-10-18 | R |
| [euai-art50] | European Commission: Article 50 FAQ | 2026-07-24 | R |
| [glba] | 16 CFR 314.4 (GLBA Safeguards) | live, accessed 2026-10-18 | R |
| [dora] | ESAs: designated critical ICT providers | 2025-11-18 | R |
| [hipaa-delay] | DWT: HHS HIPAA rulemaking timeline | 2026-07 | N |
| [cppa] | California CPPA: ADMT regulations | 2025-09-23 | R |
| [presidio] | Microsoft Presidio image redactor | live, accessed 2026-10-18 | D |
| [spark-price] | The Register: DGX Spark pricing | 2026-10-02 | N |
| [mac-studio] | Apple: Mac Studio M5 Max/Ultra | 2026-08-25 | D |
| [rtx6000] | TechPowerUp: RTX PRO 6000 price | 2026-08-13 | N |
| [gpu-prices] | CloudRift GPU price comparison | 2026-10-15 | N (vendor-run) |
| [brev-h200] | NVIDIA Brev H200 listing | live, accessed 2026-10-18 | D |
| [nvaie] | NVIDIA AI Enterprise pricing | 2026-09-02 | D |
| [nim-offer] | NVIDIA NIM offerings | 2026-10-16 | D |
| [spark-nim] | NVIDIA Lightning NIM on DGX Spark | live, accessed 2026-10-18 | D |
| [omlx] | oMLX Lightning bf16 on M4 Max | 2026-09-23 | I |
| [tw-h200] | Thoughtworks: Lightning on H200 (NVIDIA-commissioned) | 2026-08-11 | I |
| [vllm-spark] | vLLM issue 59770 (Spark regression) | 2026-10 | I |
| [m5-finops] | llmkube: M5 Max FinOps | 2026-04-26 | I |
| [trial-tos] | NVIDIA API Trial Terms of Service | v. 2025-09-19 (re-uploaded 2026-10-16) | D |
| [openmdw] | OpenMDW FAQ | 2026-05 | D |
| [osi-review] | OSI license-review thread on OpenMDW-1.1 | 2026-08-17 | N |
| [rico] | Rico copyright notice | live, accessed 2026-10-18 | D |
| [osatlas] | OS-Atlas-data card | 2024-12-04, *pre-2025* | D |
| [parakeet] | NVIDIA Parakeet TDT 0.6B v3 card | 2025-08-14 | D |
| [gartner-cost] | Gartner: cost per contact | 2024-02, *pre-2025* | I |
| [gartner-14] | Gartner: 14% resolved in self-service | 2024-08-19, *pre-2025* | I |
| [gartner-2040] | Gartner: why customers bail on self-service | 2020, *pre-2025* | I (opinion survey) |
| [mckutil] | McKinsey: utility IVR | 2020, *pre-2025* | I |
| [contactbabel] | ContactBabel: US contact center decision-makers' guide | 2026 edition | I |
| [gartner-journey] | Gartner: customer service metrics (illustrative worked example) | 2025–2026 | I |
| [pnm] | PNM grid modernization cost-benefit filing (utility's own input assumption) | 2023-11-22, *pre-2025* | V |
| [gartner-3usd] | Gartner: GenAI cost per resolution by 2030 | 2026-01-26 | A |
| [fin-outcomes] | Intercom: Fin outcomes billing | live, accessed 2026-10-18 | D |
| [zendesk-price] | Zendesk pricing | live, accessed 2026-10-18 | D |
| [sf-price] | Salesforce: Agentforce pricing | 2025-05-19 | D |
| [sf-fin] | Salesforce to acquire Fin | 2026-06-15 | D |
| [gartner-87] | Gartner: 87% want a human option | 2026-08-04 | I |
| [gartner-27] | Gartner: 27% would retry a chatbot | 2026-09-02 | I |
| [five9] | Five9/Hanover consumer survey | 2026-10-14 | I (vendor-commissioned) |
| [copc] | COPC AI customer-experience research | 2026-01-16 | I |
| [cfpb] | CFPB: chatbots in consumer finance | 2023-06-06, *pre-2025* | R |
| [webaim-sr] | WebAIM Screen Reader Survey #11 | 2026-09-29 | I |
| [webaim-million] | WebAIM Million 2026 (February 2026 crawl) | live, accessed 2026-10-18 | I |
| [acm-nl] | Dutch ACM: accessibility of online stores | 2026-03-24 | R |
| [ftc-accessibe] | FTC: accessiBe final order | 2025-04-22 | R |
| [ecomback] | EcomBack 2025 ADA lawsuit report | live, accessed 2026-10-18 | I (vendor) |
| [audioeye] | AudioEye FY2025 results (SEC) | 2026-03-05 | D |
| [guide] | Guide (Windows agent for BLV users) | live, accessed 2026-10-18 | D |
| [apple-a11y] | Apple accessibility updates | 2026-05-19 | D |
| [jaws-ai] | Vispero JAWS 2026 | 2025-10-29 | D |
| [auto-browse] | Google: Gemini 3 auto browse | 2026-01-28 | D |
| [ars-autobrowse] | Ars Technica: auto browse test | 2026-02-12 | N |
| [mariner] | The Verge: Project Mariner shut down | 2026-05-06 | N |
| [browserbase-b] | Upstarts: Browserbase $40M Series B | 2025-06-17 | N |
| [kernel] | Kernel Series A | 2025-10-09 | D |
| [simular] | Simular: Sai tops OSWorld 2.0 | 2026-08-28 (upd. 2026-10-02) | V |
| [uipath-uicube] | UiPath UI-CUBE | 2025-11-21 | V |
| [uipath-screenplay] | UiPath ScreenPlay best practices | live, accessed 2026-10-18 | D |
| [uipath-nvidia] | UiPath + NVIDIA | 2025-09-30 | D |
| [uipath-call] | UiPath Q4 FY26 earnings call | 2026-03-12 | D |
| [mordor-rpa] | Mordor Intelligence: RPA market | live, accessed 2026-10-18 | A |
| [mordor-test] | Mordor Intelligence: automation testing market | 2026-08-05 | A |
| [testcomplete] | SmartBear TestComplete: what's new | 2026-09-30 | D |
| [tosca-onprem] | Tricentis: Tosca AI on-prem | 2026-07-22 | D |
| [applitools] | Applitools Autonomous | 2026-03-11 | D |
| [octomind] | Octomind shutdown letter | 2026-04-23 | D |
| [qawolf] | QA Wolf pricing | live, accessed 2026-10-18 | D |
| [pa-price] | Microsoft Power Automate pricing | live, accessed 2026-10-18 | D |
| [bp-surface] | Blue Prism surface automation (region mode) | 2025-10-20 | D |
| [servicenow] | ServiceNow AI Desktop Actions release notes | 2026-10 | D |
| [eggplant] | Keysight Eggplant | live, accessed 2026-10-18 | D |
| [functionize] | Functionize $41M Series B | 2025-08-19 | D |
| [momentic] | TechCrunch: Momentic $15M | 2025-11-24 | N |
| [flutter-a11y] | Flutter web accessibility docs | live, accessed 2026-10-18 | D |
| [fed-rpa] | State of federal RPA (government community-of-practice survey) | 2020-11, *pre-2025* | I |
| [holo3] | H Company: Holo3-35B-A3B card | 2026-04-01 | V |
| [holo15] | H Company: Holo1.5-3B card (Qwen Research License) | 2025-09-25 | V |
| [holo4-traj] | H Company: OSWorld trajectory browser (200-step budget) | live, accessed 2026-10-18 | V |
| [uivenus2] | UI-Venus-2 | 2026-08-27 | V |
| [groundnext] | GroundNext / GroundCUA | 2026-06-09 (v2) | V |
| [nim-faq] | NVIDIA NIM product FAQ | 2026-08-06 | D |
| [oai-agents-faq] | OpenAI: Agents API beta FAQ | live, accessed 2026-10-18 | D |
| [meta-price] | Meta Model API pricing | live, accessed 2026-10-18 | D |
| [nova-ga] | Amazon Science: Nova Act service GA | 2025-12-02 | V |
| [slicc] | SLICC independent Browser Use Bench run | 2026-10-12 | I |
| [gcp-opus55] | Google Cloud: Claude Opus 5.5 on Agent Platform | 2026-10-15 | D |
| [gcp-abuse] | Google Cloud: Agent Platform abuse monitoring | 2026-10-15 | D |
| [bedrock-abuse] | AWS: Bedrock abuse detection (Fable retention) | live, accessed 2026-10-18 | D |
| [pci-faq1438] | PCI SSC FAQ 1438 (SAQ A iframes), via pciwatch mirror | 2016-09, *pre-2025* | N |
| [bemyeyes-b2b] | Be My Eyes: business and Workplace pricing | live, accessed 2026-10-18 | D |
| [bp-forum] | Blue Prism community thread on Citrix (anecdotal) | 2026-03-17 | N |

[results]: ../results.md
[blade-agent]: ../blade/agent_base_vs_sft.md
[blade-final]: ../blade/agent_final_vs_sft.md
[local-mac]: ../local-mac.md
[pitch]: ../pitch.md
[submission]: ../submission.md
[status]: ../status.md
[console-runs]: console-runs.md
[anth-cu]: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
[anth-vision]: https://platform.claude.com/docs/en/build-with-claude/vision
[anth-price]: https://platform.claude.com/docs/en/about-claude/pricing
[anth-browser]: https://platform.claude.com/docs/en/agents-and-tools/tool-use/browser-use-tool
[anth-retention]: https://platform.claude.com/docs/en/manage-claude/api-and-data-retention
[anth-baa]: https://support.claude.com/en/articles/8114513-business-associate-agreements-baa-for-commercial-customers
[anth-residency]: https://platform.claude.com/docs/en/manage-claude/data-residency
[anth-sonnet55-card]: https://www-cdn.anthropic.com/870c8f525702625d2c62fc6dd04c857e3250bec1/Claude%20Sonnet%205.5%20System%20Card.pdf
[anth-haiku55-card]: https://www-cdn.anthropic.com/e1080d6bf5ae2018ea3c2f414064be03232f5be5/Claude%20Haiku%205.5%20System%20Card.pdf
[anth-opus5-card]: https://www-cdn.anthropic.com/ceaf5c7ff2783855203fde8208ec311252dced5b/Claude%20Opus%205%20System%20Card.pdf
[anth-opus48-card]: https://www-cdn.anthropic.com/0b4915911bb0d19eca5b5ee635c80fef830a37ea/Claude%20Opus%204.8%20System%20Card.pdf
[opus55-inj]: https://agentic-cyber-explorer.pages.dev/events/anthropic-opus-5-5-system-card-agentic-boundary-injection-2026/
[chrome-ga]: https://claude.com/resources/articles/claude-in-chrome-generally-available
[chrome-perm]: https://support.claude.com/en/articles/12902446-claude-in-chrome-permissions-guide
[oai-cu]: https://developers.openai.com/api/docs/guides/tools-computer-use
[oai-vision]: https://developers.openai.com/api/docs/guides/images-vision
[oai-price]: https://developers.openai.com/api/docs/pricing
[oai-astra]: https://openai.com/index/gpt-6-astra/
[oai-hipaa]: https://help.openai.com/en/articles/20001069-hipaa-eligible-products-and-functionality
[oai-data]: https://developers.openai.com/api/docs/guides/your-data
[oai-atlas]: https://help.openai.com/en/articles/20001371-evolving-atlas-into-chatgpt-for-browser-based-agentic-work
[oai-agent-confirm]: https://deploymentsafety.openai.com/chatgpt-agent/promptinjection
[gem-cu]: https://ai.google.dev/gemini-api/docs/computer-use
[gem-price]: https://ai.google.dev/gemini-api/docs/pricing
[gem-media]: https://ai.google.dev/gemini-api/docs/media-resolution
[gem-argon]: https://deepmind.google/models/model-cards/gemini-4-argon/
[gem-policies]: https://ai.google.dev/gemini-api/docs/usage-policies
[gem-autobrowse-help]: https://support.google.com/gemini/answer/16821166
[gcp-cu]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/computer-use
[gcp-terms]: https://cloud.google.com/terms/service-terms
[gcp-hipaa]: https://cloud.google.com/security/compliance/hipaa
[meta-wonder]: https://research.meta.ai/static/muse-wonder-model-card
[nova-price]: https://aws.amazon.com/nova/pricing/
[copilot-cu]: https://learn.microsoft.com/en-us/microsoft-copilot-studio/computer-use
[copilot-faq]: https://learn.microsoft.com/en-us/microsoft-copilot-studio/faqs-computer-use
[aws-terms]: https://aws.amazon.com/service-terms/
[osw-v]: https://osworld-v1.xlang.ai/static/data/osworld_verified_results.xlsx
[osw2-board]: https://osworld-v2.xlang.ai/static/data/leaderboard/official-results.json
[osw2-paper]: https://arxiv.org/html/2606.29537v2
[ale]: https://snorkel.ai/leaderboard/agents-last-exam/
[windtunnel]: https://github.com/nekuda-ai/WindTunnel
[speedrun]: https://arxiv.org/html/2609.40284v1
[stagehand]: https://stagehand.dev/evals
[bu-speed]: https://browser-use.com/posts/speed-matters
[bu-price]: https://browser-use.com/pricing
[bu-iframe]: https://github.com/browser-use/browser-use/pull/5296
[om2w-paper]: https://arxiv.org/html/2504.01382v4
[om2w-board]: https://huggingface.co/spaces/osunlp/Online_Mind2Web_Leaderboard/raw/main/human_Mind2Web-Online%20-%20Leaderboard_data.csv
[om2w-pricing]: https://github.com/OSU-NLP-Group/Online-Mind2Web/blob/main/data/schema_v2/HUMAN_EVALUATION_PRICING.md
[hal-om2w]: https://hal.cs.princeton.edu/online_mind2web
[steel-om2w]: https://leaderboard.steel.dev/leaderboards/online-mind2web/
[clawbench]: https://arxiv.org/html/2604.08523
[odysseys]: https://odysseysbench.com/v2/
[webarena-tracker]: https://personalagentbench.com/benchmarks/webarena/
[wav-list]: https://benchmarklist.com/benchmarks/webarena_verified/
[liquid-card]: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B
[lightning-card]: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
[aw-sheet]: https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo/export?format=csv
[epoch-osw]: https://epoch.ai/publications/what-does-osworld-tell-us-about-ais-ability-to-use-computers
[rdi-exploit]: https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/
[misscore]: https://arxiv.org/abs/2607.28367
[passk-cua]: https://arxiv.org/html/2604.17849
[rerun-flips]: https://arxiv.org/abs/2608.06171
[pando]: https://arxiv.org/html/2605.24785v2
[vwa]: https://arxiv.org/html/2401.13649
[dailydroid]: https://arxiv.org/html/2604.17817
[state-belief]: https://arxiv.org/html/2607.04334
[visiontasker]: https://arxiv.org/html/2312.11190v2
[ecoagent]: https://arxiv.org/html/2505.05440v3
[jedi]: https://arxiv.org/html/2505.13227
[osw-human]: https://arxiv.org/html/2506.16042v2
[fara15]: https://huggingface.co/microsoft/Fara1.5-9B
[holo4]: https://hcompany.ai/newsroom/holo4
[h-local]: https://hub.hcompany.ai/models-api/local-inference
[h-newsroom]: https://hcompany.ai/newsroom
[qwen38]: https://huggingface.co/Qwen/Qwen3.8-27B
[qwen-cua]: https://arxiv.org/abs/2608.02352
[qwen36-harness]: https://huggingface.co/meta-models/Muse-Glimmer-30B
[uivenus15]: https://arxiv.org/abs/2602.09082
[maiui]: https://arxiv.org/abs/2512.22047
[guiowl15]: https://arxiv.org/html/2602.16855
[qwen35-2b]: https://huggingface.co/Qwen/Qwen3.5-2B
[qwen-ui-agent]: https://arxiv.org/html/2607.28227
[ss-board]: https://gui-agent.github.io/grounding-leaderboard/results/screenspot_v2.json
[sspro-board]: https://gui-agent.github.io/grounding-leaderboard/
[venusbench]: https://arxiv.org/html/2512.16501
[windom]: https://arxiv.org/html/2606.25964
[guilens]: https://arxiv.org/html/2608.03270
[argus]: https://arxiv.org/html/2606.25760v2
[cuactspot]: https://arxiv.org/html/2605.12501
[guiprim]: https://arxiv.org/html/2608.21832
[star]: https://arxiv.org/html/2509.13615v5
[finestate]: https://arxiv.org/html/2604.27974
[websp]: https://arxiv.org/html/2604.06367v1
[zoomclick]: https://arxiv.org/html/2512.05941
[uizoomer]: https://arxiv.org/html/2604.14113
[reveal]: https://openreview.net/forum?id=zdfed8pwuk
[osw-pro]: https://arxiv.org/html/2609.24890
[chartact]: https://arxiv.org/html/2605.26994
[mislead-chart]: https://arxiv.org/html/2610.12823
[iplot]: https://arxiv.org/html/2604.21134v1
[mvp]: https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_MVP_Multiple_View_Prediction_Improves_GUI_Grounding_CVPR_2026_paper.pdf
[perturbed]: https://arxiv.org/abs/2604.14262
[cura]: https://arxiv.org/abs/2608.27808
[vlaa]: https://arxiv.org/html/2604.21375v3
[tau2-false]: https://arxiv.org/abs/2606.09863
[webpagebench]: https://arxiv.org/html/2609.35026
[componentbench]: https://arxiv.org/html/2608.18307v1
[univ-verifier]: https://www.microsoft.com/en-us/research/articles/the-art-of-building-verifiers-for-computer-use-agents/
[vagen]: https://arxiv.org/html/2602.00575v1
[agentrewardbench]: https://arxiv.org/html/2504.08942v2
[lookagain]: https://arxiv.org/html/2608.09723v1
[redteamcua]: https://arxiv.org/html/2505.21936
[stepjack]: https://arxiv.org/html/2608.06477
[google-ipi]: https://blog.google/security/prompt-injections-web/
[camel]: https://arxiv.org/abs/2503.18813
[olla]: https://arxiv.org/html/2609.00524
[a11ycua]: https://arxiv.org/html/2602.09310
[uitars2]: https://arxiv.org/html/2509.02544v2
[arpo]: https://arxiv.org/html/2505.16282
[webagent-r1]: https://aclanthology.org/2025.emnlp-main.401/
[nemotron-rl]: https://docs.nvidia.com/nemotron/latest/nemotron/super3/rl/rlvr.html
[nemotron-supervl]: https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b/modelcard
[pci-saq]: https://listings.pcisecuritystandards.org/documents/PCI-DSS-v4-0-SAQ-B.pdf
[pci-phone]: https://listings.pcisecuritystandards.org/documents/Protecting_Telephone_Based_Payment_Card_Data_v3-0_nov_2018.pdf
[pci-ai-2026]: https://www.helpnetsecurity.com/2026/10/09/pci-ssc-payment-environments-ai-security-guidance/
[euai-nav]: https://digital-strategy.ec.europa.eu/en/faqs/navigating-ai-act
[euai-art50]: https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act
[glba]: https://www.ecfr.gov/current/title-16/chapter-I/subchapter-C/part-314/section-314.4
[dora]: https://www.esma.europa.eu/sites/default/files/2025-11/List_of_designated_CTPPs.pdf
[hipaa-delay]: https://www.dwt.com/blogs/privacy--security-law-blog/2026/07/hhs-updates-hipaa-rulemaking-timeframes
[cppa]: https://cppa.ca.gov/announcements/2025/20250923.html
[presidio]: https://github.com/microsoft/presidio/blob/main/docs/image-redactor/index.md
[spark-price]: https://www.theregister.com/systems/2026/10/02/nvidia-debuts-4999-dgx-spark-with-half-the-ram-and-storage-amid-memory-crunch/5300622
[mac-studio]: https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/
[rtx6000]: https://www.techpowerup.com/351549/nvidia-rtx-pro-6000-blackwell-96-gb-gpu-now-costs-usd-16-000
[gpu-prices]: https://www.cloudrift.ai/cloud-gpu-pricing-comparison
[brev-h200]: https://brev.nvidia.com/environment/new/public?gpu=H200
[nvaie]: https://docs.nvidia.com/ai-enterprise/planning-resource/licensing-guide/latest/pricing.html
[nim-offer]: https://docs.nvidia.com/nim/large-language-models/latest/about-nim-llm/nim-offerings.html
[spark-nim]: https://build.nvidia.com/spark/nim-llm
[omlx]: https://omlx.ai/benchmarks/performance/01yn0d9t
[tw-h200]: https://huggingface.co/blog/lujangusface/tw-eagle3-nemotron35-lightning
[vllm-spark]: https://github.com/vllm-project/vllm/issues/59770
[m5-finops]: https://llmkube.com/blog/m5-max-aider-polyglot-and-finops
[trial-tos]: https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf
[openmdw]: https://openmdw.ai/faq/
[osi-review]: https://lists.opensource.org/pipermail/license-review_lists.opensource.org/2026-August/006150.html
[rico]: https://interactionmining.org/archive/rico/copyright.txt
[osatlas]: https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data
[parakeet]: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
[gartner-cost]: https://www.gartner.com/en/documents/5164231
[gartner-14]: https://www.gartner.com/en/newsroom/press-releases/2024-08-19-gartner-survey-finds-only-14-percent-of-customer-service-issues-are-fully-resolved-in-self-service
[gartner-2040]: https://www.gartner.com/smarterwithgartner/do-you-know-why-customers-bail-on-self-service
[mckutil]: https://www.mckinsey.com/industries/electric-power-and-natural-gas/our-insights/transforming-interactive-voice-response-systems-in-utilities
[contactbabel]: https://www.contactbabel.com/the-us-contact-center-decision-makers-guide/
[gartner-journey]: https://emt.gartnerweb.com/ngw/globalassets/en/customer-service-support/documents/trends/cs-metrics-repackage.pdf
[pnm]: https://www.txnmenergy.com/~/media/Files/P/PNM-Resources/rates-and-filings/PNM%20Grid%20Mod/CBA/Omni%20B%20Warner.pdf
[gartner-3usd]: https://www.gartner.com/en/newsroom/press-releases/2026-01-26-gartner-predicts-genai-cost-per-resolution-for-customer-service-will-exceed-offshore-human-agent-costs-by-2030
[fin-outcomes]: https://www.intercom.com/help/en/articles/8205718-fin-ai-agent-outcomes
[zendesk-price]: https://www.zendesk.com/pricing/
[sf-price]: https://help.salesforce.com/s/articleView?id=004811240&language=en_US&type=1
[sf-fin]: https://www.salesforce.com/news/press-releases/2026/06/15/salesforce-signs-definitive-agreement-to-acquire-fin/
[gartner-87]: https://www.gartner.com/en/newsroom/press-releases/2026-08-04-gartner-survey-finds-87-percent-of-customers-say-companies-using-genai-for-customer-service-must-provide-access-to-a-human-agent0
[gartner-27]: https://www.gartner.com/en/newsroom/press-releases/2026-09-02-gartner-finds-only-27-percent-of-customers-would-try-a-chatbot-again-after-a-negative-experience
[five9]: https://www.five9.com/news/news-releases/new-five9-research-reveals-consumers-embrace-ai-customer-service-trust-still
[copc]: https://www.copc.com/ai-customer-experience-research-2025/
[cfpb]: https://www.consumerfinance.gov/data-research/research-reports/chatbots-in-consumer-finance/chatbots-in-consumer-finance/
[webaim-sr]: https://webaim.org/projects/screenreadersurvey11/
[webaim-million]: https://webaim.org/projects/million/
[acm-nl]: https://www.acm.nl/en/publications/acm-majority-large-online-stores-are-not-accessible-customers-disabilities
[ftc-accessibe]: https://www.ftc.gov/news-events/news/press-releases/2025/04/ftc-approves-final-order-requiring-accessibe-pay-1-million
[ecomback]: https://www.ecomback.com/annual-2025-ada-website-accessibility-lawsuit-report
[audioeye]: https://www.sec.gov/Archives/edgar/data/1362190/000110465926024106/aeye-20260305xex99d1.htm
[guide]: https://www.guideinteraction.com/
[apple-a11y]: https://www.apple.com/newsroom/2026/05/apple-unveils-new-accessibility-features-and-updates-with-apple-intelligence/
[jaws-ai]: https://vispero.com/news/vispero-announces-version-2026-of-its-leading-accessibility-software/
[auto-browse]: https://blog.google/products-and-platforms/products/chrome/gemini-3-auto-browse/
[ars-autobrowse]: https://arstechnica.com/google/2026/02/tested-how-chromes-auto-browse-agent-handles-common-web-tasks/
[mariner]: https://www.theverge.com/tech/925559/google-project-mariner-shut-down
[browserbase-b]: https://www.upstartsmedia.com/p/browserbase-raises-40m-and-launches-director
[kernel]: https://www.kernel.sh/blog/series-a-announcement
[simular]: https://www.simular.ai/articles/sai-tops-osworld-2-0
[uipath-uicube]: https://arxiv.org/abs/2511.17131
[uipath-screenplay]: https://docs.uipath.com/agents/automation-cloud/latest/user-guide-screenplay/best-practices
[uipath-nvidia]: https://ir.uipath.com/news/detail/409/uipath-delivers-trusted-agentic-automation-for-sensitive-workflows-with-nvidia
[uipath-call]: https://www.fool.com/earnings/call-transcripts/2026/03/12/uipath-path-q4-2026-earnings-call-transcript/
[mordor-rpa]: https://www.researchandmarkets.com/reports/4787490/robotic-process-automation-market-share
[mordor-test]: https://www.mordorintelligence.com/industry-reports/automation-testing-market
[testcomplete]: https://support.smartbear.com/testcomplete/docs/general-info/whats-new.html
[tosca-onprem]: https://www.tricentis.com/blog/toscas-ai-is-coming-to-on-premises-deployments-this-fall
[applitools]: https://applitools.com/blog/applitools-autonomous-eyes-ai-testing-updates/
[octomind]: https://octomind.dev/blog/a-letter-to-our-users-customers-and-readers/
[qawolf]: https://www.qawolf.com/pricing
[pa-price]: https://www.microsoft.com/en-us/power-platform/products/power-automate/pricing
[bp-surface]: https://documentation.blueprism.com/bp-6-10/en-us/application-model-setup/spy-modes/region-mode.htm
[servicenow]: https://www.servicenow.com/docs/r/australia/release-notes/agentic-desktop-rn.html
[eggplant]: https://www.keysight.com/us/en/products/design-engineering-software/software-quality-engineering/industries/engineering-applications.html
[functionize]: https://www.functionize.com/blog/functionize-raises-41m-series-b-to-accelerate-the-future-of-autonomous-quality-assurance
[momentic]: https://techcrunch.com/2025/11/24/momentic-raises-15m-to-automate-software-testing/
[flutter-a11y]: https://docs.flutter.dev/ui/accessibility/web-accessibility
[fed-rpa]: https://digital.gov/s3/files/m-files/state-of-federal-rpa.pdf
[holo3]: https://huggingface.co/Hcompany/Holo3-35B-A3B
[holo15]: https://huggingface.co/Hcompany/Holo1.5-3B
[holo4-traj]: https://trajectories.hcompany.ai/b/osworld
[uivenus2]: https://arxiv.org/abs/2609.00028
[groundnext]: https://arxiv.org/html/2511.07332v2
[nim-faq]: https://docs.api.nvidia.com/nim/docs/product
[oai-agents-faq]: https://help.openai.com/en/articles/20001551-agents-api-beta-faq
[meta-price]: https://dev.meta.ai/docs/pricing-rate-limits
[nova-ga]: https://www.amazon.science/blog/amazon-nova-act-service
[slicc]: https://sliccy.com/benchmark
[gcp-opus55]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/opus-5-5
[gcp-abuse]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/abuse-monitoring
[bedrock-abuse]: https://docs.aws.amazon.com/bedrock/latest/userguide/abuse-detection.html
[pci-faq1438]: https://pciwatch.org/pcissc/faqs/1438/
[bemyeyes-b2b]: https://www.bemyeyes.com/business/workplace-pricing/
[bp-forum]: https://community.blueprism.com/t5/Product-Forum/Unable-to-attch-to-Citrix-Workspace-in-bp-v-7-1-1/td-p/120971
