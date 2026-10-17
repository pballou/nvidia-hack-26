# Submission

**Due Sunday 2026-10-18, 11:00am EDT.** One submission per team, via the
[Airtable form](https://airtable.com/appm7NS9oN8K3friP/pagqXe6ElIlXx6oa3/form).
Target: everything finished by 10:30.

## Form fields → what we need → owner

| Form field | What we put there | Owner | Source |
| --- | --- | --- | --- |
| Project Name / Team Name | **GUI assist** / **sherpas** | — | — |
| Team Member Names & Email Addresses | Patrick, Kyle, Charlie + emails | Patrick | — |
| Submission Track | Fine-Tuned & Specialized | — | — |
| Submission Description | 1 paragraph: problem → approach → results | Patrick | README pitch |
| Demo Video URL | Unlisted **YouTube** (accepted by both the form and the checklist), 3–5 min | Charlie (records and uploads) | §Video |
| GitHub Repo | `https://github.com/pballou/nvidia-hack-26` (public) | — | — |
| AI Models Used | Nemotron 3.5 Lightning (30B-A3B, official NVIDIA BF16 checkpoint); our fine-tuned grounder (Qwen3.5-2B + LoRA); NVIDIA Parakeet TDT 0.6B v3 (local voice input) | Kyle | — |
| Tools Used | NeMo Gym (envs, rollouts, profiling, compare, BLADE), vLLM (+ vllm-metal on the Mac), NVIDIA Brev (H200), NeMo-Speech.cpp (Parakeet voice input), TRL, Playwright | all | — |
| ☑ Lightning bounty + explanation | What Lightning does (planner/tool-caller; instruction rewriting), why, how we optimize it (thinking budget, tool schema, Gym-measured success/latency) | Patrick | `docs/bounty-lightning.md` (draft; final Sunday). Form text: §Form text |
| ☑ NeMo Gym bounty + explanation | Gym setup (`gui_grounding`, `web_agent`), **baseline vs. final** numbers from `gym eval compare`, what we changed and why (BLADE findings → data/GRPO) | Charlie + Kyle | `docs/bounty-nemo-gym.md` (draft; final Sunday). Form text: §Form text |
| ☑ Red Hat vLLM bounty | (checkbox only) vLLM serves both models; GRPO rollouts via vLLM colocate | Kyle | — |

## Form text (paste-ready drafts)

Short versions of the bounty write-ups for the form's explanation boxes. Fill in the **TBD**s from
`docs/results.md` on Sunday, and keep them consistent with the full docs. The SFT numbers and
the 18/36 → 22/36 agent result below are `sft-template`'s. The overnight loop starts from
`sft-lightning` (dev 86.27%), so add its numbers, and update the agent numbers if the agent eval
is rerun with the final adapter. Each one links to its
full doc for judges who want more.

**Lightning bounty:**

> Nemotron 3.5 Lightning is the brain of GUI Assist, a vision-only accessibility agent. Lightning
> never sees the screen. After each action it reads a text observation (the URL, whether the
> screen changed, what its last action did, and a screen description from a 2B VLM), then calls
> one of five schema-checked tools. A call like `click("the L option in the Text size row")` goes
> to our fine-tuned 2B grounder, which turns the description into a pixel. Offline, Lightning
> also rewrites raw accessibility labels into user-style click instructions for the grounder's
> training data. In a human audit, 42 of 43 checkable rewrites were faithful (85.7% if "unsure"
> counts as a failure), and the data is published with its prompt. We self-host NVIDIA's official BF16 checkpoint: about 1.2 s per planner step on an
> idle server, against about 26 s on the hosted free tier. We tune it with corrective schema
> validation, compact observations and per-step thinking control. Turning thinking on only after
> a problem step raised claimed-equals-verified from 6/12 to 9/12 tasks and cut actions that
> changed nothing on screen from 45 to 27 (n=1 per task). In the NeMo Gym `web_agent` env (12 tasks × 3),
> swapping the base grounder for the fine-tuned one raised verified success from 18/36 to 22/36 and cut
> failed actions from 10 to 0. Full write-up:
> `https://github.com/pballou/nvidia-hack-26/blob/main/docs/bounty-lightning.md`

**NeMo Gym bounty:**

> We used NeMo Gym to measure and improve a 2B grounding VLM (Qwen3.5-2B + LoRA), the model that
> turns "the gear icon in the top-right" into a click. Two Gym environments live in our repo:
> `gui_grounding` (reward = the click lands in the target box; adapted from Gym's
> `gui_coordinate`) and `web_agent` (Nemotron Lightning plus our grounder on a sandbox app,
> rewarded from the app's own state). Every number comes from `gym eval run`. SFT took dev
> accuracy from 76.4% to 85.7% and held-out ScreenSpot-v2-clean from 86.3% to 89.1% (icons
> +6.7). The improvement loop runs through Gym: `gym eval profile` finds rows the model sometimes
> gets right, TRL GRPO trains on them with the Gym server's `/verify` as the reward, and a dev
> eval plus `gym eval compare` decides keep or revert. BLADE failure analysis steers the next
> round. In the `web_agent` env, the tuned grounder raised verified task success from 18/36 to
> 22/36 over the base. After GRPO: TBD dev and TBD ScreenSpot-v2-clean. Along the way we found and excluded a ScreenSpot-v2 leak: 28 desktop test screenshots
> were captured on the same machines as OS-Atlas training data. Full write-up:
> `https://github.com/pballou/nvidia-hack-26/blob/main/docs/bounty-nemo-gym.md`

**vLLM bounty** (checkbox only; use this if the form asks for a sentence):

> vLLM serves every model in the system. One Qwen3.5-2B process on the H200 serves both `base`
> (screen descriptions) and `grounder` (our LoRA adapter, switched per request), so the base and
> tuned models count once toward the parameter cap. GRPO rollouts run on vLLM colocated with TRL,
> and Lightning runs on a MacBook through vllm-metal, the vLLM project's plugin for Apple GPUs.

## README must include (organizer checklist)

The README skeleton has a section for each:

- [ ] **Quick start**: commands to run (Patrick: localhost app plus model serving)
- [ ] **Tech stack & architecture diagram** (simple is fine)
- [ ] **How to reproduce the demo**: env vars, API keys, sample `.env` (`.env.example`, `env.yaml.example`)
- [ ] **Datasets / synthetic data + provenance**: OS-Atlas subsets used and excluded (and why), Lightning-rewritten instructions (published plus method), ScreenSpot-v2 (eval only)
- [ ] **Known limitations & next steps**
- [ ] **Deployed URL or a short screen capture** of the working app
- [ ] **Team roster**: names, roles and contacts. The README is public, so use contacts you're happy to publish (e.g. GitHub handles or work emails).

## Publishing (required for this track: "publish it openly so others can reuse it")

Patrick publishes under the Hugging Face account **`patrickballou`**.
- Uploads run **from the Brev box**, where the checkpoints are, using a fine-grained token scoped to write only `patrickballou/gui-assist-grounder` and `patrickballou/gui-assist-data`.
- That token is exported only in the upload shell (`export HF_TOKEN=...`), never
  saved with `hf auth login`, never committed, and revoked after the event.
- Patrick's full write token stays in `.env` on his laptop.

The repos already exist and are **private**. Uploads go in private (data Saturday night, model
~8:00 Sunday). Patrick makes both public at release, Sun ~9:30, after his release check.

Upload commands (on the box):

```bash
hf upload patrickballou/gui-assist-grounder /home/ubuntu/ckpt/<run>/adapter adapter --repo-type model
hf upload patrickballou/gui-assist-grounder /home/ubuntu/ckpt/<run>/merged . --repo-type model
hf upload patrickballou/gui-assist-data /home/ubuntu/data/release . --repo-type dataset
```

| Artifact | Where | Contents |
| --- | --- | --- |
| Model | `patrickballou/gui-assist-grounder` (HF model repo) | Merged model and/or LoRA adapter, plus a model card: base model, data and licenses, training config, SSv2 results (3 ways), intended use (accessibility), limitations |
| Data | `patrickballou/gui-assist-data` (HF dataset repo) | Our slice manifest (source + image filename + bbox + split), the **Lightning-rewritten instructions**, and any other model-generated training data (Lightning keep/drop filter decisions, rejection-sampled reward=1 rollouts), each with the exact prompt, command and config that made it. Images stay referenced by source filename, not re-hosted. |
| Code + envs | This repo | `resources_servers/gui_grounding`, `resources_servers/web_agent`, data prep, training configs, eval results |

## Video (3–5 min, camera on, show the team)

Follow the organizers' recommended flow:

| Segment | Time | Content |
| --- | --- | --- |
| 1. Team | 20–30 s | Team name; who did what |
| 2. Pitch | 30–40 s | Commercial framing (judge advice): self-service fails (14% resolved), so customers call ($13.50 vs $1.84); the accessibility gap. Then "say what you want, and it does it on the real site, from the screen alone". **Script: [pitch.md §10](pitch.md#10-60-second-video-script).** |
| 3. Live demo | 45–60 s | Core loop with minimal cuts: user request → Lightning plans → grounder clicks (show the click overlay) → page changes → task done. Show one base-vs-tuned difference. |
| 4. How we built it | 60–90 s | Architecture diagram; NeMo Gym envs and the base → SFT → GRPO results; BLADE failure analysis; vLLM on the H200; official Lightning BF16 on vLLM (vllm-metal on the Mac) plus thinking control. Challenges: the ScreenSpot contamination we found and excluded, coordinate conventions, the one-GPU budget. |
| 5. So what | 20–30 s | ROI (illustrative): a 2M-customer utility saves ~$4–8M/yr at 20–40% call deflection; no API needed; measured improvement (icons 78 → 85%); "fits on one GPU" (or "runs on this Mac" only if that check passed); the ask: one design partner. **Script: [pitch.md §10](pitch.md#10-60-second-video-script); "don't say" list there too.** |

**Owner:** Charlie records and uploads; full script: [demo-video-script.md](demo-video-script.md). **Recording plan:** dry run Saturday after ~19:45. Record Sunday 8:30–10:00,
upload unlisted by 10:15, submit the form by 10:30.

## Judging criteria → our evidence (use in the video and write-ups)

| Criterion | What we show | Source |
| --- | --- | --- |
| **Completeness:** works end to end; uses open data | Voice → Lightning plans → grounder clicks → task verified from app state, live on the Acme portal. Trained on open OS-Atlas data, published with provenance. | console demo, `docs/results.md`, README data table |
| **Fine-Tuned track:** measurable improvement over a baseline on eval data | Base vs. SFT (vs. GRPO) on dev and **ScreenSpot-v2-clean**, plus base-vs-tuned **agent task success** in the Gym `web_agent` env | `docs/results.md`, `gym eval compare` reports |
| **Technical depth** | LoRA SFT + GRPO with **NeMo Gym rewards**; an automated overnight profile→GRPO→dev-gate loop; BLADE failure analysis; contamination found and removed; Lightning data engine with a human audit | `docs/tasks/B-night-loop.md`, `docs/blade/`, `data_prep/rewrite/runs/os_atlas_final_v4/METHOD.md` |
| **NVIDIA/Red Hat use, and why** | Nemotron Lightning (planner + data engine), NeMo Gym (envs, rewards, profiling, BLADE), Parakeet (voice), vLLM on H200 + vllm-metal on the Mac, Brev. **Why:** private, local inference; 32.6B total runs on one GPU or a MacBook | README stack table, `docs/bounty-lightning.md` |
| **Problem fit / usability** | Customer self-service on existing websites, with no integration. Fewer support calls; accessibility. Usable tomorrow by an everyday person, by voice. | `docs/pitch.md` |
| **Creativity** | Voice + vision-only grounding + a text-only planner. Works on canvas and legacy UIs that DOM agents can't operate. | demo canvas tasks |
| **Performance** (measured) | Local Lightning **~1.2 s/step vs 25–80 s/call** on the hosted free tier (~20–60×). Fine-tuned answers ~2× shorter (37.6 → 16.7 tokens), so grounding is faster. ASR ~0.1 s per clip. Grounding p50 0.73 s. | `docs/results.md`, `docs/local-mac.md` |
