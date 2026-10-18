# Submission

**Due Sunday 2026-10-18, 11:00am EDT.** One submission per team, via the
[Airtable form](https://airtable.com/appm7NS9oN8K3friP/pagqXe6ElIlXx6oa3/form).
Target: everything finished by 10:30.

## Form fields → what we need → owner

| Form field | What we put there | Owner | Source |
| --- | --- | --- | --- |
| Project Name / Team Name | **GUI Assist** / **sherpas** | — | — |
| Team Member Names & Email Addresses | Patrick, Kyle, Charlie + emails | Patrick | — |
| Submission Track | Fine-Tuned & Specialized | — | — |
| Submission Description | 1 paragraph: problem → approach → results | Patrick | README pitch |
| Demo Video URL | Unlisted **YouTube** (accepted by both the form and the checklist), 3–5 min | Charlie (records and uploads) | §Video |
| GitHub Repo | `https://github.com/pballou/nvidia-hack-26` (public) | — | — |
| AI Models Used | Nemotron 3.5 Lightning (30B-A3B, official NVIDIA BF16 checkpoint); our fine-tuned grounder (Qwen3.5-2B + LoRA); NVIDIA Parakeet TDT 0.6B v3 (local voice input) | Kyle | — |
| Tools Used | NeMo Gym (envs, rollouts, profiling, compare, BLADE), vLLM (+ vllm-metal on the Mac), NVIDIA Brev (H200), NeMo-Speech.cpp (Parakeet voice input), TRL, Playwright | all | — |
| ☑ Lightning bounty + explanation | What Lightning does (planner/tool-caller; instruction rewriting), why, how we optimize it (thinking budget, tool schema, Gym-measured success/latency) | Patrick | `docs/bounty-lightning.md` (final, Sun). Form text: §Form text |
| ☑ NeMo Gym bounty + explanation | Gym setup (`gui_grounding`, `web_agent`), **baseline vs. final** numbers from `gym eval compare`, what we changed and why (BLADE findings → data/GRPO) | Charlie + Kyle | `docs/bounty-nemo-gym.md` (final, Sun 00:30). Form text: §Form text |
| ☑ Red Hat vLLM bounty + explanation | How vLLM fits (one process for `base` + `grounder`, Gym evals, colocated GRPO rollouts, vllm-metal on the Mac), how we configured it, measured results | Kyle | `docs/bounty-vllm.md` (draft). Form text: §Form text |

## Bounty asks → where we answer them

Condensed from the organizers' bounty slides. "Open" is what a judge could push on.

**Best Use of NVIDIA Nemotron Lightning** (RTX 5070, NVIDIA Developer Livestream feature):

| Ask | Where we answer it | Open |
| --- | --- | --- |
| **Challenge:** Lightning does meaningful real work, alone or with other models; show what it enables | Planner and tool-caller for every agent step, plus the data engine that rewrote 8,620 SFT instructions ([bounty-lightning.md](bounty-lightning.md) §2) | — |
| **Required:** a core part of the project | It's the agent's only planner: nothing happens on screen without a Lightning tool call | — |
| **Demonstrate:** its contribution (quality, speed, efficiency, usefulness, creativity) | Thinking ablation: on vs off 26 vs 19 of 32 jobs, p = 0.039. Rewrite audit 42/43 faithful. Self-hosted ~1.2 s per step vs ~26 s hosted (§5, §6; [results.md](results.md)) | The rewrite ablation is mixed: dev +0.41 (inside the ~1.1-point SE), SSv2-clean −0.97. Present it as an ablation, not a win |
| **Submit:** what it does, why we chose it, how we optimize it | §2, §3, §4; form text below | — |

**Best Model Improvement with NeMo Gym** (RTX 5070, NVIDIA Developer Livestream feature):

| Ask | Where we answer it | Open |
| --- | --- | --- |
| **Challenge:** one task a small model fails at today; make the score better with Gym | 2B GUI click grounding: base 76.4% dev / 86.25% SSv2-clean → final 87.0% / 90.35% ([bounty-nemo-gym.md](bounty-nemo-gym.md) §1–§2) | — |
| **Required:** a Gym env with tasks, tools the agent can use, and code that scores each attempt | `gui_grounding` (rows + point-in-box `/verify`) and `web_agent` (12 sandbox tasks, 5 tools, app-state checks) (§3) | The score that improved is on `gui_grounding`, which is single-turn with no tools. The env with tools, `web_agent`, shows no significant change (18 / 22 / 17 of 36). Name both envs and say which one moved |
| **Demonstrate:** baseline vs final on the same held-out tasks and eval config; isolate what made the improvement | SSv2-clean, same 1,244 rows and harness, T=0: 86.25 → 90.35 (+4.10). Separated into SFT (+9.9 dev), GRPO round 1 (+2.25 SSv2-clean over its start, p ≈ 8e-6) and the template-vs-Lightning SFT ablation (§2, §8) | No equal-compute non-GRPO control. (The base → final pair is now tested: 89 fixed vs 38 broken, p ≈ 7e-6) |
| **Submit:** Gym setup, baseline vs final, how to run scoring, how we made the improvement | §3, §2, §9, §4; form text below | The §9 scoring commands need the Brev box, which is stopped |

**Best Use of Red Hat vLLM** (JBL speaker per member, Red Hat Office Hours feature):

| Ask | Where we answer it | Open |
| --- | --- | --- |
| **Challenge:** genuinely served on vLLM; vLLM plays a meaningful role in how the app runs | vLLM serves every generative model at every stage: H200 grounder, Gym evals, GRPO rollouts, and Lightning plus the local grounder on the Mac ([bounty-vllm.md](bounty-vllm.md) §2) | — |
| **Required:** vLLM as the serving layer for a core model; functional and central | One process serves `base` + `grounder`, on the H200 and on the Mac, and the parameter-cap design depends on it (§2) | — |
| **Demonstrate:** performance, efficiency, throughput, latency, scalability or reliability | 32,000 Gym replies in ~7 min; grounding p50 0.64–0.75 s; Gate G parity between CUDA and Metal runtime LoRA (87.0 vs 87.0); GRPO log-probs match (§4) | No controlled benchmark (`vllm bench serve`). Latencies come from real runs, several on a shared GPU |
| **Submit:** how vLLM fits, how we configured and optimized it, measurable results; depth over breadth | bounty-vllm.md §2–§4; form text below | We'd planned it as checkbox-only. Check whether the form has a text box |

## Final submission checklist (Sun 2026-10-18; submit by 10:30, deadline 11:00)

- [ ] **Video:** recorded, uploaded to YouTube as **Unlisted**, link pasted into the README and the form (Charlie).
- [x] **GitHub repo** public: https://github.com/pballou/nvidia-hack-26
- [x] **HF model** public: https://huggingface.co/patrickballou/gui-assist-grounder
- [x] **HF data** public: https://huggingface.co/datasets/patrickballou/gui-assist-data (10:27)
- [ ] **Airtable form**, using the values below. Paste the description, then the three bounty boxes from §Form text.
  - Project / team: **GUI Assist** / **sherpas**
  - Members: Patrick (pballou@sherpa6.com), Kyle (kodonnell@sherpa6.com), Charlie (carleth@sherpa6.com)
  - Track: **Fine-Tuned & Specialized**
  - Bounties: ☑ Nemotron Lightning ☑ NeMo Gym ☑ Red Hat vLLM
  - AI models: NVIDIA Nemotron 3.5 Lightning 30B-A3B (official BF16); our grounder `patrickballou/gui-assist-grounder` (Qwen3.5-2B + LoRA, SFT + GRPO); NVIDIA Parakeet TDT 0.6B v3. 32.6B total by model-card counts
  - Tools: NeMo Gym 0.7.0 (envs, rollouts, profile/compare, BLADE), vLLM 0.31 (+ vllm-metal on the Mac), TRL 1.16 (SFT, GRPO), NVIDIA Brev (H200), NeMo-Speech.cpp, Playwright
- [ ] After submitting: confirm the form receipt, and revoke any HF tokens you don't need.

**Submission description** (paste-ready):

> GUI Assist lets people do things on any website just by saying what they want. It's an
> accessibility and self-service layer that works from pixels alone, with no API, DOM or site
> changes. Only 14% of customer-service issues get resolved in self-service today, and a call costs
> $13.50 against $1.84. NVIDIA Nemotron 3.5 Lightning plans each step from a text description of
> the screen and calls our fine-tuned 2B grounder (Qwen3.5-2B + LoRA), which turns "the download
> icon in the September row" into a click. NVIDIA Parakeet handles voice, and Lightning asks a
> short question when a request is unclear. We fine-tuned the grounder on OS-Atlas data, with
> instructions rewritten by Lightning, then improved it with GRPO in NeMo Gym. On the held-out
> ScreenSpot-v2-clean test, it went from 86.25% to 90.35% (icons 78.1% to 85.4%), and in the agent
> it cut failed actions from 10 to 0. All three models (32.6B by model-card counts) run fully
> locally on one MacBook through vllm-metal, so screens never leave the device. The model, the
> data (with the exact rewrite prompt) and the code are all open.

## Form text (paste-ready drafts)

Short versions of the bounty write-ups for the form's explanation boxes. Keep them consistent
with the full docs; each one links to its full doc for judges who want more.
- **NeMo Gym text: final** (Sun 00:30). It matches bounty-nemo-gym.md: the SFT number is
  sft-lightning's (the dev pick), the agent numbers name each arm, and the final adapter's numbers
  are filled in.
- **Lightning text: final** (Sun 00:35). It matches bounty-lightning.md and results.md: the agent
  numbers name each arm (base 18/36, sft-template 22/36, final 17/36, all within noise), and the
  thinking result is the 12 × 3 off/retry/on ablation, not Saturday's n=1 read.
- **vLLM text: draft** (Sun). It matches bounty-vllm.md.

**Lightning bounty:**

> Nemotron 3.5 Lightning is the brain of GUI Assist, a vision-only accessibility agent. Lightning
> never sees the screen. After each action it reads a text observation (the URL, whether the
> screen changed, what its last action did, and a screen description from a 2B VLM), then calls
> one of five schema-checked tools. A call like `click("the L option in the Text size row")` goes
> to our fine-tuned 2B grounder, which turns the description into a pixel. Offline, Lightning
> also rewrites raw accessibility labels into user-style click instructions for the final
> grounder's SFT data (8,620 rewrites kept). In a human audit, 42 of 43 checkable rewrites were
> faithful (85.7% if "unsure" counts as a failure), and the data is published with its prompt. We
> self-host NVIDIA's official BF16 checkpoint: about 1.2 s per planner step on an idle server,
> against about 26 s on the hosted free tier. We tune it with corrective schema validation,
> compact observations and per-step thinking control. On the interactive agent (12 tasks × 3,
> raw-label SFT grounder `sft-template`), thinking on every call verified 26 of the 32 jobs all
> three arms completed, against 21 with thinking only after a problem step (the config default;
> the demo runs thinking off, the Gym-measured setting) and 19 with it off (paired McNemar on vs off p = 0.039), with 4 false success claims
> instead of 9 and about half the steps of thinking off, at a median 10.9 s per planning call vs
> 5.9 s with retry. In the NeMo Gym `web_agent` env (12 tasks × 3, thinking off), verified
> success is within noise for all three grounders: base 18/36, `sft-template` 22/36 (20/36 with a
> stricter mark-all-read check), and the final grounder (SFT on the Lightning rewrites, then
> GRPO) 17/36. Both tuned grounders cut failed actions from 10 to 0 (the CI excludes 0, but the 10
> sit in 4 base rollouts of 2 tasks), and BLADE traces about 6 of the 9 rollouts that changed
> between the `sft-template` and final arms to the planner or the screen descriptions, not the
> clicks. Full write-up:
> `https://github.com/pballou/nvidia-hack-26/blob/main/docs/bounty-lightning.md`

**NeMo Gym bounty:**

> We used NeMo Gym to measure and improve a 2B grounding VLM (Qwen3.5-2B + LoRA), the model that
> turns "the gear icon in the top-right" into a click. Two Gym environments live in our repo:
> `gui_grounding` (reward = the click lands in the target box; adapted from Gym's
> `gui_coordinate`) and `web_agent` (Nemotron Lightning plus our grounder on a sandbox app,
> rewarded from the app's own state). Every number comes from `gym eval run`. LoRA SFT, with
> Lightning-rewritten instructions and picked on dev, took dev accuracy from 76.4% to 86.27%. The
> improvement loop runs through Gym: `gym eval profile` finds rows the model sometimes
> gets right, TRL GRPO trains on them with the Gym server's `/verify` as the reward, a dev eval
> through Gym decides keep or revert (gain ≥ +0.5 points), and `gym eval compare` against the
> previous round is saved as the per-task flip report. BLADE failure analysis steers the next
> round through pre-approved knobs. GRPO round 1 raised dev from 86.27% to 87.00% (22 fixed vs 15
> broken, not significant on its own); round 2 was flat and reverted. Final adapter: 87.0% dev and 90.35% on held-out ScreenSpot-v2-clean (a second run gave 90.27). On ScreenSpot-v2-clean that is +4.10 over base (86.25%; 89 rows fixed vs 38 broken, McNemar p ≈ 7e-6), and +2.25 over the loop's starting adapter (34 rows fixed vs 6 broken on the same rows, McNemar p ≈ 8e-6). Against the raw-label (template) SFT, stronger on ScreenSpot-v2-clean but not the dev pick, it is +1.29 (p = 0.10). In the `web_agent` env (12 tasks × 3), agent success is within noise for all grounders (base 18/36, template SFT 22/36, final 17/36), and both tuned grounders cut failed actions from 10 to 0. Along the way we found and excluded a ScreenSpot-v2 leak: 28 desktop test screenshots
> were captured on the same machines as OS-Atlas training data. Full write-up:
> `https://github.com/pballou/nvidia-hack-26/blob/main/docs/bounty-nemo-gym.md`

**vLLM bounty:**

> vLLM is the serving layer for every generative model in GUI Assist, a vision-only accessibility
> agent (only voice input runs elsewhere, on NeMo-Speech.cpp). On the H200, one vLLM 0.31.0 process
> serves Qwen3.5-2B under two names: `base` (screen descriptions) and `grounder` (our LoRA adapter,
> switched per request). That keeps base and tuned as one model (32.6B system total, under the 33B
> cap), and the same server script answers the agent's clicks and every NeMo Gym eval. We pin the
> 2 MP image cap in `--mm-processor-kwargs` to match training, pass screenshots as `file://` paths
> instead of base64, and run at 0.30 GPU memory, so SFT and GRPO train beside it. GRPO rollouts
> come from vLLM colocated with TRL. After four fixes to TRL's colocate path, rollout log-probs
> match the trainer to within 0.01–0.03. Gym's profile pass drew 32,000 sampled replies (4,000
> screenshots × 8) in about 7 minutes. The demo runs fully local on a MacBook through vllm-metal,
> the vLLM project's plugin for Apple GPUs: official BF16 Lightning (about 1.2 s per idle step,
> against a p50 of about 26 s on the hosted free tier) plus one Qwen3.5-2B instance with the same
> per-request LoRA as the H200. On 200 dev rows, Metal and CUDA runtime LoRA agree: 87.0% vs 87.0%,
> identical prompt tokens on every row, and 99% of decisions agreeing. Full write-up:
> `https://github.com/pballou/nvidia-hack-26/blob/main/docs/bounty-vllm.md`

## README must include (organizer checklist)

The README skeleton has a section for each:

- [x] **Quick start**: commands to run (Patrick: localhost app plus model serving)
- [x] **Tech stack & architecture diagram** (simple is fine)
- [x] **How to reproduce the demo**: env vars, API keys, sample `.env` (`.env.example`, `env.yaml.example`)
- [x] **Datasets / synthetic data + provenance**: OS-Atlas subsets used and excluded (and why), Lightning-rewritten instructions (published plus method), ScreenSpot-v2 (eval only)
- [x] **Known limitations & next steps**
- [ ] **Deployed URL or a short screen capture** of the working app
- [x] **Team roster**: names, roles and contacts (README, Sat 20:40). The README is public, so use contacts you're happy to publish (e.g. GitHub handles or work emails).

## Publishing (required for this track: "publish it openly so others can reuse it")

Patrick publishes under the Hugging Face account **`patrickballou`**.
- **As done:** both uploads ran **from Patrick's Mac** with his token from `.env` (it never went to the box). The
  checkpoints were copied to the Mac before the H200 was stopped, and the sha256s match the box copies. The data was
  staged in `~/hf-release/gui-assist-data` (no images); the model release (merged root, `adapter/`, `training/`) in
  `~/hf-release/gui-assist-grounder`.
- The original plan, uploading from the Brev box with a fine-grained token scoped to the two repos, wasn't needed.
  Never run `hf auth login` on the shared box.

Both repos are **public** (model Sun 10:15, data Sun 10:27), after Patrick's release check: model Apache-2.0,
data CC BY 4.0, hosted-API rewrites published.

Upload commands as run (on the Mac, with `set -a; source .env; set +a`):

```bash
hf upload patrickballou/gui-assist-data ~/hf-release/gui-assist-data . --repo-type dataset
hf upload patrickballou/gui-assist-grounder ~/hf-release/gui-assist-grounder . --repo-type model
```

| Artifact | Where | Contents |
| --- | --- | --- |
| Model | `patrickballou/gui-assist-grounder` (HF model repo) | Merged final model at the root, the PEFT LoRA in `adapter/`, GRPO records in `training/`, plus a model card: base model, data and licenses, training config, SSv2 results (3 ways), intended use (accessibility), limitations |
| Data | `patrickballou/gui-assist-data` (HF dataset repo) | Our slice manifest (source + image filename + bbox + split), the **Lightning-rewritten instructions**, and any other model-generated training data (Lightning keep/drop filter decisions; the rejection-sampled SFT fallback wasn't used), each with the exact prompt, command and config that made it. Images stay referenced by source filename, not re-hosted. |
| Code + envs | This repo | `resources_servers/gui_grounding`, `resources_servers/web_agent`, data prep, training configs, eval results |

## Video (3–5 min, camera on, show the team)

Follow the organizers' recommended flow:

| Segment | Time | Content |
| --- | --- | --- |
| 1. Team | 20–30 s | Team name; who did what |
| 2. Pitch | 30–40 s | Commercial framing (judge advice): self-service fails (14% resolved), so customers call ($13.50 vs $1.84); the accessibility gap. Then "say what you want, and it does it on the real site, from the screen alone". **Script: [pitch.md §10](pitch.md#10-60-second-video-script).** |
| 3. Live demo | 45–60 s | Core loop with minimal cuts: user request → Lightning plans → grounder clicks (show the click overlay) → page changes → task done. Show one base-vs-tuned difference. |
| 4. How we built it | 60–90 s | Architecture diagram; NeMo Gym envs and the base → SFT → GRPO results; BLADE failure analysis; vLLM on the H200; official Lightning BF16 on vLLM (vllm-metal on the Mac) plus thinking control. Challenges: the ScreenSpot contamination we found and excluded, coordinate conventions, the one-GPU budget. |
| 5. So what | 20–30 s | ROI (illustrative): a 2M-customer utility saves ~$4–8M/yr at 20–40% call deflection; no API needed; measured improvement (icons 78 → 85%); "runs on this Mac" (fully local, 32.6B; Gate G passed); the ask: one design partner. **Script: [pitch.md §10](pitch.md#10-60-second-video-script); "don't say" list there too.** |

**Owner:** Charlie records and uploads; full script: [demo-video-script.md](demo-video-script.md). **Recording plan:** fully local on Patrick's Mac, headless (no `--headed`),
thinking off, request check on; config verified Sun 03:42 (pay-50 ✓ ×2, text-size ✓). Script the
five tasks FINAL passed 3/3; avoid high-contrast and message-support. Record Sunday 8:30–10:00,
upload unlisted by 10:15, submit the form by 10:30.

## Judging criteria → our evidence (use in the video and write-ups)

| Criterion | What we show | Source |
| --- | --- | --- |
| **Completeness:** works end to end; uses open data | Voice → Lightning plans → grounder clicks → task verified from app state, live on the Acme portal. Trained on open OS-Atlas data, published with provenance. | console demo, `docs/results.md`, README data table |
| **Fine-Tuned track:** measurable improvement over a baseline on eval data | Base vs. SFT vs. final (SFT + GRPO) on dev and **ScreenSpot-v2-clean** (86.25 → 90.35%, +4.10; icons 78.1 → 85.4), plus base-vs-tuned agent runs in the Gym `web_agent` env (task success within noise; failed actions 10 → 0) | `docs/results.md`, `gym eval compare` reports |
| **Technical depth** | LoRA SFT + GRPO with **NeMo Gym rewards**; an automated overnight profile→GRPO→dev-gate loop; BLADE failure analysis; contamination found and removed; Lightning data engine with a human audit | `docs/tasks/B-night-loop.md`, `docs/blade/`, `data_prep/rewrite/runs/os_atlas_final_v4/METHOD.md` |
| **NVIDIA/Red Hat use, and why** | Nemotron Lightning (planner + data engine), NeMo Gym (envs, rewards, profiling, BLADE), Parakeet (voice), vLLM on H200 + vllm-metal on the Mac, Brev. **Why:** private, local inference; 32.6B total runs on one GPU or a MacBook | README stack table, `docs/bounty-lightning.md` |
| **Problem fit / usability** | Customer self-service on existing websites, with no integration. Fewer support calls; accessibility. Usable tomorrow by an everyday person, by voice. | `docs/pitch.md` |
| **Creativity** | Voice + vision-only grounding + a text-only planner. It reads pixels, so it applies to canvas and legacy UIs that DOM agents can't read (on the canvas-chart task both tuned grounders hit the right bar; the task passes 1/3 with FINAL). | `docs/blade/agent_final_vs_sft.md` |
| **Performance** (measured) | Local Lightning **~1.2 s/step (idle server) vs 25–80 s/call** on the hosted free tier (~20–60×); ~4 s/step inside the fully local agent. Fine-tuned answers ~2× shorter (37.6 → 16.7 tokens, pilot SFT), so grounding is faster. Grounding p50 0.73 s (H200; 1.86 s on the Mac). | `docs/results.md`, `docs/local-mac.md` |
