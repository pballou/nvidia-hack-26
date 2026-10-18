# GUI assist: the pitch

Team **sherpas** (Patrick, Kyle, Charlie). Fine-Tuned & Specialized track.

Market figures link to their sources. **Illustrative** means our own arithmetic on sourced inputs, with the assumptions stated; it is not a published figure. Our results come from [results.md](results.md) as of Sun 2026-10-18; the final checkpoint (SFT + GRPO round 1) was chosen on dev.

## 1. One line

**Say what you want to do on a website, and GUI assist does it on the real page.** It works from screenshots alone, with small models that run on your own hardware. Companies have already built self-service. Customers can't find it. We connect the two.

## 2. The problem

- **Self-service fails, so people call.**
  - Only **14%** of service issues are fully resolved in self-service, and only 36% of "very simple" ones. The top reason is that customers **couldn't find the content they needed (43%)** ([Gartner 2024][g14]).
  - An agent contact costs a median **$13.50**, against **$1.84** for self-service ([Gartner 2024][gcost]).
  - Bain finds that up to **70%** of calls to bank contact centers are avoidable and better handled digitally. At one major bank, more than 40% of callers were at a screen or had a device at hand when they called ([Bain 2017][bain17]).
  - So each contact moved from an agent to self-service saves about **$11.66** ($13.50 − $1.84; illustrative, from Gartner's medians).
- **There is an accessibility gap.**
  - **28.7%** of US adults have a disability ([CDC][cdc]).
  - **61.2M** Americans are 65 or older ([Census][census]).
  - **95.9%** of the top 1M home pages have detectable WCAG failures ([WebAIM 2026][webaim]).
- **The legal exposure is growing.**
  - **3,117** federal website-accessibility suits were filed in 2025, up 27% ([Seyfarth][seyfarth]).
  - The European Accessibility Act has applied to consumer banking and e-commerce since **28 Jun 2025** ([EUR-Lex][eaa]).
  - US state and local governments, city-run utilities included, must meet WCAG 2.1 AA on their sites and apps from **26 Apr 2027**. Smaller entities and special districts have until 26 Apr 2028 ([DOJ][doj]).

## 3. The product

- **Voice in.** NVIDIA Parakeet 0.6B transcribes on the device, so audio never leaves the machine.
- **Lightning plans, a 2B vision model clicks.** NVIDIA Nemotron 3.5 Lightning (official BF16) breaks the request into steps. Our fine-tuned Qwen3.5-2B finds each target in the screenshot, and Playwright clicks it.
- **No integration, and private.**
  - It needs no API, no DOM and no site changes, so it can target unlabeled icons and canvas screens (canvas is still a weak spot, see §7).
  - All 32.6B parameters are sized for one on-prem GPU.
  - Users watch every click on the real site and can say "stop" at any time.

It is an **assistive layer alongside real WCAG fixes, not a compliance product.** accessiBe agreed to pay the FTC $1M to settle charges that it falsely claimed its AI tool could make any site WCAG-compliant ([FTC 2025][ftc]).

## 4. Demo moment: the Acme Utilities portal

1. Say **"Pay fifty dollars toward my bill."**
2. The screen shows the transcript, then the plan, then a click overlay on each step: *Pay bill → Other amount → 50 → Continue*.
3. The portal's own dialog shows **Amount $50.00**, and the agent clicks *Confirm payment*.
4. A **✓** appears, checked against the portal's state, not the agent's claim.

Cutaways:
- **"Download my September statement."** The download button has no label. A screen reader can't name it, but the grounder can see the icon.
- **"Make the text bigger."** The base model clicks M, and the tuned model clicks L.

## 5. Customers and ROI (illustrative, assumptions stated)

**Savings** = contacts per year × share moved × (assisted cost − self-service cost).

**Assumption:** we move **20–40%** of contacts. That is the share of live volume service leaders *report* their existing self-service could easily handle ([Gartner 2020][g2040]). It is opinion, not a measurement. Our own rate needs a pilot.

| Customer | Contacts per year | Net saving per contact | Savings per year |
|---|---|---|---|
| **Retail bank, Wells Fargo scale** | ~43M (see the bank note) | $13.50 − $1.84 = $11.66 | **~$100–200M** (upper bound) |
| **Electric utility, 2M customers** | 2.5M, the midpoint of 2–3M ([McKinsey 2020][mckutil]) | $9.54 ([E Source][esrc]) − $1.84 = $7.70 | **~$3.9–7.7M** |
| **State benefits portal** | **1M (assumption, unsourced)** | $11.66 (cross-industry) | **~$2.3–4.7M** |

- **Bank:**
  - The ~43M assumes service spend at Gartner's median of 0.7% of revenue ([Gartner][gcost]). On $83.7B of revenue ([Wells Fargo][wfq4]), that is $586M, all assumed to go to assisted contacts at $13.50.
  - This is not a Wells Fargo disclosure.
  - Fargo passed 1 billion interactions in under three years ([Wells Fargo, Mar 2026][wfnews]), up from 245M in 2024 alone ([VentureBeat][vb]). That proves demand. Fargo is built per task on the bank's own APIs, so our sweet spot is the banks without a Fargo.
  - Rule of thumb: 0.12–0.24% of revenue.
- **Utility:**
  - McKinsey puts a typical 2M-customer utility at 2–3M calls a year costing **$20–40M**, so our $9.54-per-call input is conservative. Only a handful of utilities offer self-service for bill-inquiry calls in their phone menus (IVR), while a typical telecom self-serves 60–70% of billing calls ([McKinsey 2020][mckutil]).
  - Across the US, utility call handling costs roughly $1.6–2.3B a year (illustrative): 163.7M electricity customers ([EIA 2024][eia]) × 1–1.5 calls each (McKinsey's ratio) × $9.54.
- **Government:** much of the value is access for residents. The 2027 deadline still requires fixing the site itself.

## 6. Why now, and why small local models

- **A logged-in screen is personal data.**
  - A US data breach averages **$11.5M**, and IBM calls AI sovereignty "a critical security requirement" ([IBM 2026][ibm]).
  - Wells Fargo built Fargo so that no customer personal data reaches the outside LLM ([VentureBeat][vb]).
  - HIPAA and EU DORA add contracts and oversight for cloud processors ([HHS][hhs], [ESMA][dora]).
- **Cloud cost scales with every task.**
  - A 1080p screenshot is 2,691 tokens on Claude's high-resolution tier. At Claude Opus 5.5's $4 per million input tokens ([Anthropic][anth], [vision][anthv]), a 10-step task is ~$0.11 in screenshots alone, or ~$108k a month at 1M tasks (illustrative).
  - The model choice matters: a small cloud model such as Claude Haiku 5.5 ($0.10 per million) is ~40x cheaper ([Anthropic][anth]). So privacy and control, not price alone, is the main case for local.
  - Local inference is mostly hardware bought once. We found no sourced per-token figure, so we quote none.
  - Cloud vendors do offer regional data routing ([Anthropic][anth]).
- **The market is moving this way.**
  - Gartner predicts small task-specific models will be used **3x** more than general LLMs by 2027 ([Gartner][gslm]).
  - It also predicts agentic AI will resolve 80% of common service issues without a human by 2029, cutting operational costs 30%. Its own example is AI agents "navigating websites" to complete tasks ([Gartner][g2029]).
- **NVIDIA and Red Hat stack:** Lightning plans and also rewrote our training instructions. We built our `gui_grounding` and `web_agent` environments, rewards and evals in NeMo Gym. Parakeet handles voice, vLLM serves the models (vllm-metal on the Mac), and Brev provides the H200.
- **Where it runs today (measured):**
  - Lightning, Parakeet and the grounder (base + final LoRA in one vllm-metal instance) all run on an M4 Max Mac (128 GB). Training and the Gym evals ran on one H200.
  - Grounding takes 0.73 s per click on the H200 and 1.86 s on the Mac (p50). Fully local, planning takes ~4.3 s per step with thinking off (Saturday's hybrid reads: 4.1–4.8 s p50, p90 ~15–17 s when thinking retries kick in).

## 7. Proof it works (measured this weekend)

- **Grounding:** the base 2B model scores **86.3%** on ScreenSpot-v2-clean (1,244 rows). We found and excluded 28 contaminated rows. The final tuned model scores **90.35%**.
- **Fine-tuning:**
  - One LoRA epoch on all 18,652 training rows (54 min on one H200) raised dev accuracy from 76.4% to **85.66% (+9.3)**. The same epoch on 18,299 rows with Lightning-rewritten instructions reached **86.27%** (vs 85.86% for raw labels in the same harness), so we kept that one (picked on dev). The rewrite ablation is mixed: +0.41 on dev, −0.97 on ScreenSpot-v2-clean.
  - Then GRPO in NeMo Gym (TRL, rewarded by our `gui_grounding` env's `/verify` point-in-box check; 170 steps on 1,282 rows the model sometimes got right) took dev to **87.00%**. That is the final model. A second round, with its data mix set by BLADE failure analysis, was flat (86.89%), so the loop stopped.
  - On the held-out test, ScreenSpot-v2-clean went from 86.3% to **90.35% (+4.1)**, and +2.25 over the SFT model GRPO started from (34 rows fixed, 6 broken). **Icons went from 78.1% to 85.4% (+7.3)**. Icon-only controls are often the ones a screen reader can't name: WebAIM finds empty buttons on 30.6% of top home pages and empty links on 46.3% ([WebAIM 2026][webaim]).
  - A 5-minute pilot (2,000 rows) had already given +5.0 on dev, and answers got 2x shorter.
  - In a human audit, 97.7% (42/43) of the Lightning rewrites that could be checked were faithful, or 85.7% if "unsure" counts as a failure.
- **Agent on Acme, verified from the app's state** (NeMo Gym `web_agent`, 12 tasks × 3 repeats, thinking off):
  - Verified success: base 18/36, raw-label SFT 22/36, final 17/36. All three are within noise (final vs base 95% CI [−0.20, +0.14]), so we claim no end-to-end task gain.
  - What tuning does change: failed actions (no usable click point) went from 10 to **0** (significant), the small adjacent "L" text-size option went from 0/3 to **3/3**, and the final grounder hit **96.5%** (110/114) of hand-labelled well-posed targets vs SFT's 92.6% (not significant). BLADE traces most remaining failures to the planner and the screen summary, not the grounder.
  - 96–97% of tool calls were valid (base 249/259, SFT 228/234).
  - Thinking on lifted the interactive agent (SFT grounder) from 19/32 to **26/32** on the same jobs (p = 0.039), at ~11 s per planning call. The demo keeps thinking off.
  - A pre-flight Lightning check asks when a request is unclear: 0 false asks on 62 clear requests and 22/22 asks on our acceptance set, 5 false asks in 90 on held-out phrasings. It is on in the console and off for evals.
  - Caveats: 3 repeats per task, with the planner sampling at temperature 1.0, on our own test app.
- **Known misses:** targets that need scrolling, the canvas chart, dense toggle grids, and one run that paid the full balance instead of $50.
- **Fully local (measured):** Lightning (official BF16), Parakeet and the grounder (base + final LoRA in one vllm-metal instance; 32.6B by model-card counts) run on one M4 Max Mac (128 GB). The local grounder matches the H200: 87.0% vs 87.0% on 200 dev rows, 99% agreement.
- **Not measured:** real call deflection. That needs a pilot.

## 8. Business model and competition

**Pricing:** per resolved task, priced well under the $11.66 that each deflected contact saves (illustrative). Regulated buyers can take an on-prem annual license instead. Price points are not yet validated.

| Alternative | Its gap | GUI assist |
|---|---|---|
| FAQ chatbots and in-app assistants | Most answer but don't act on the page, and the ones that act need API work per task. Only 28% of bank and card app customers use the virtual assistant, and satisfaction drops on complex problems ([J.D. Power 2026][jdp]). | It acts on the site the customer already uses. |
| RPA | Scripted per process, tied to a fixed UI, built for the back office. | It takes free-form requests and reads the screen again at every step. |
| DOM and accessibility-tree agents | They need clean markup or site integration. Canvas has no DOM. | It works from pixels only. |
| Accessibility overlays | They claim compliance, yet 983 ADA web suits in 2025 (25% of the total) targeted sites that use widgets ([EcomBack][ecom]). | It makes no compliance claim. |
| Cloud computer-use agents | Screens leave the building, and the cost is per task. | It runs locally. |

## 9. The ask

- **One design partner:** a utility, credit union or benefits agency. We would pilot on their real portal, on-prem, for their top call reasons, and measure verified task success, contacts avoided and hand-offs.
- **What we build next:**
  - **A one-tap hand-off to a human.** 64% of customers would prefer companies not use AI for service ([Gartner][g64]), and the CFPB warns that blocking access to humans can break the law ([CFPB][cfpb]).
  - **Confirmation before any money moves.**
  - **GRPO in NeMo Gym** on our failure clusters.
  - **A packaged fully local install** (today's Mac setup is manual, [local-mac.md](local-mac.md)).
- **Open:** the model and data go public on Hugging Face Sunday ~9:30, after the release check.

## 10. 60-second video script

**Segment 2, pitch (~35 s).** Show each source as a caption.
> "Only fourteen percent of customer-service problems get solved through self-service, and the top reason is that people can't find what they need. So they call. An agent contact costs about thirteen fifty, against under two dollars for self-service. More than one in four US adults has a disability, and ninety-six percent of the top million home pages fail automated accessibility checks. GUI assist closes that gap. You say what you want, and it does it on the real website, from the screen alone."

**Segment 5, so what (~25 s).** Caption: *illustrative, assumptions in docs/pitch.md*.
> "For a two-million-customer utility, moving twenty to forty percent of calls to self-service is worth about four to eight million dollars a year. GUI assist needs no API, and about two hours of fine-tuning took icon clicks from seventy-eight to eighty-five percent. It runs on this Mac, so screens never leave the building. We're looking for one utility, credit union or benefits agency to pilot it."

**Don't say:**
- "ADA compliant" or "EAA compliant".
- That 20–40% is measured.
- That fine-tuning raised end-to-end task success (agent success is within noise). Say what it fixed: failed clicks 10 → 0, the small "L" option 0/3 → 3/3.
- Wells Fargo "$1B". Use the sourced, illustrative ~$100–200M upper bound.

[g14]: https://www.gartner.com/en/newsroom/press-releases/2024-08-19-gartner-survey-finds-only-14-percent-of-customer-service-issues-are-fully-resolved-in-self-service
[gcost]: https://www.gartner.com/en/documents/5164231
[g2040]: https://www.gartner.com/smarterwithgartner/do-you-know-why-customers-bail-on-self-service
[bain17]: https://www.bain.com/insights/bank-contact-centers-dial-t-for-transformation/
[cdc]: https://www.cdc.gov/disability-and-health/about/index.html
[census]: https://www.census.gov/newsroom/press-releases/2025/older-adults-outnumber-children.html
[webaim]: https://webaim.org/projects/million/
[seyfarth]: https://www.adatitleiii.com/2026/03/federal-court-website-accessibility-lawsuit-filings-bounce-back-in-2025/
[eaa]: https://eur-lex.europa.eu/EN/legal-content/summary/accessibility-of-products-and-services.html
[doj]: https://www.ada.gov/resources/2024-03-08-web-rule/
[ftc]: https://www.ftc.gov/news-events/news/press-releases/2025/01/ftc-order-requires-online-marketer-pay-1-million-deceptive-claims-its-ai-product-could-make-websites
[mckutil]: https://www.mckinsey.com/industries/electric-power-and-natural-gas/our-insights/transforming-interactive-voice-response-systems-in-utilities
[esrc]: https://www.esource.com/blog/427221kzsn/why-fcr-valuable-metric-utility-contact-centers
[eia]: https://www.eia.gov/electricity/annual/html/epa_02_11.html
[wfq4]: https://www.wellsfargo.com/assets/pdf/about/investor-relations/earnings/fourth-quarter-2025-earnings.pdf
[wfnews]: https://newsroom.wf.com/news-releases/news-details/2026/Wells-Fargo-Reaches-Major-Digital-Milestones/default.aspx
[ibm]: https://assets.ibm.com/adobe/assets/urn:aaid:aem:21111142-1251-4369-86fb-57b82f5bb108/original/as/Cost%20of%20a%20Data%20Breach%20Report%202026.pdf
[vb]: https://venturebeat.com/business/wells-fargos-ai-assistant-just-crossed-245-million-interactions-with-zero-humans-in-the-loop-and-zero-pii-to-the-llm
[hhs]: https://www.hhs.gov/hipaa/for-professionals/special-topics/health-information-technology/cloud-computing/index.html
[dora]: https://www.esma.europa.eu/esmas-activities/digital-finance-and-innovation/digital-operational-resilience-act-dora
[anth]: https://platform.claude.com/docs/en/about-claude/pricing
[anthv]: https://platform.claude.com/docs/en/build-with-claude/vision
[gslm]: https://www.gartner.com/en/newsroom/press-releases/2025-04-09-gartner-predicts-by-2027-organizations-will-use-small-task-specific-ai-models-three-times-more-than-general-purpose-large-language-models
[g2029]: https://www.gartner.com/en/newsroom/press-releases/2025-03-05-gartner-predicts-agentic-ai-will-autonomously-resolve-80-percent-of-common-customer-service-issues-without-human-intervention-by-20290
[jdp]: https://www.jdpower.com/business/press-releases/2026-us-digital-banking-and-credit-card-mobile-app-satisfaction-studies/
[ecom]: https://www.ecomback.com/annual-2025-ada-website-accessibility-lawsuit-report
[g64]: https://www.gartner.com/en/newsroom/press-releases/2024-07-09-gartner-survey-finds-64-percent-of-customers-would-prefer-that-companies-didnt-use-ai-for-customer-service
[cfpb]: https://www.consumerfinance.gov/data-research/research-reports/chatbots-in-consumer-finance/chatbots-in-consumer-finance/
