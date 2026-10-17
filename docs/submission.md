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
| AI Models Used | Nemotron 3.5 Lightning (30B-A3B, official NVIDIA BF16 checkpoint); our fine-tuned grounder (Qwen3.5-2B or Qwen3-VL-2B + LoRA) | Kyle | — |
| Tools Used | NeMo Gym (envs, rollouts, profiling, compare, BLADE), vLLM, NVIDIA Brev (H200), TRL, Playwright | all | — |
| ☑ Lightning bounty + explanation | What Lightning does (planner/tool-caller; instruction rewriting), why, how we optimize it (thinking budget, tool schema, Gym-measured success/latency) | Patrick | `docs/bounty-lightning.md` (write Sunday) |
| ☑ NeMo Gym bounty + explanation | Gym setup (`gui_grounding`, `web_agent`), **baseline vs. final** numbers from `gym eval compare`, what we changed and why (BLADE findings → data/GRPO) | Charlie + Kyle | `docs/bounty-nemo-gym.md` (write Sunday) |
| ☑ Red Hat vLLM bounty | (checkbox only) vLLM serves both models; GRPO rollouts via vLLM colocate | Kyle | — |

## README must include (organizer checklist)

The README skeleton has a section for each:

- [ ] **Quick start**: commands to run (Patrick: localhost app plus model serving)
- [ ] **Tech stack & architecture diagram** (simple is fine)
- [ ] **How to reproduce the demo**: env vars, API keys, sample `.env` (`.env.example`, `env.yaml.example`)
- [ ] **Datasets / synthetic data + provenance**: OS-Atlas subsets used and excluded (and why), GroundCUA, Lightning-rewritten instructions (published plus method), ScreenSpot-v2 (eval only)
- [ ] **Known limitations & next steps**
- [ ] **Deployed URL or a short screen capture** of the working app
- [ ] **Team roster**: names, roles and contacts. The README is public, so use contacts you're happy to publish (e.g. GitHub handles or work emails).

## Publishing (required for this track: "publish it openly so others can reuse it")

Patrick publishes under the Hugging Face account **`patrickballou`**.
- Uploads run **from the Brev box**, where the checkpoints are, using a fine-grained token scoped to write only `patrickballou/gui-assist-grounder` and `patrickballou/gui-assist-data`.
- That token is exported only in the upload shell (`export HF_TOKEN=...`), never
  saved with `hf auth login`, never committed, and revoked after the event.
- Patrick's full write token stays in `.env` on his laptop.

The repos already exist and are **private**. Make them public at release (Sun morning).

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

**Owner:** Charlie records and uploads. **Recording plan:** do a dry run Saturday night if possible. Record Sunday 8:30–10:00,
upload unlisted by 10:15, submit the form by 10:30.

## Judging criteria → our evidence (use in the video and write-ups)

| Criterion | What we show | Source |
| --- | --- | --- |
| **Completeness:** works end to end; uses open data | Voice → Lightning plans → grounder clicks → task verified from app state, live on the Acme portal. Trained on open OS-Atlas data, published with provenance. | console demo, `docs/results.md`, README data table |
| **Fine-Tuned track:** measurable improvement over a baseline on eval data | Base vs. SFT (vs. GRPO) on dev and **ScreenSpot-v2-clean**, plus base-vs-tuned **agent task success** in the Gym `web_agent` env | `docs/results.md`, `gym eval compare` reports |
| **Technical depth** | LoRA SFT + GRPO with **NeMo Gym rewards**; an automated overnight profile→GRPO→dev-gate loop; BLADE failure analysis; contamination found and removed; Lightning data engine with a human audit | `docs/tasks/B-night-loop.md`, `docs/blade/`, METHOD.md |
| **NVIDIA/Red Hat use, and why** | Nemotron Lightning (planner + data engine), NeMo Gym (envs, rewards, profiling, BLADE), Parakeet (voice), vLLM on H200 + vllm-metal on the Mac, Brev. **Why:** private, local inference; 32.6B total runs on one GPU or a MacBook | README stack table, `docs/bounty-lightning.md` |
| **Problem fit / usability** | Customer self-service on existing websites, with no integration. Fewer support calls; accessibility. Usable tomorrow by an everyday person, by voice. | `docs/pitch.md` |
| **Creativity** | Voice + vision-only grounding + a text-only planner. Works on canvas and legacy UIs that DOM agents can't operate. | demo canvas tasks |
| **Performance** (measured) | Local Lightning **~1.2 s/step vs 25–80 s/call** on the hosted free tier (~20–60×). Fine-tuned answers ~2× shorter (37.6 → 16.7 tokens), so grounding is faster. ASR ~0.1 s per clip. Grounding p50 0.73 s. | `docs/results.md`, `docs/local-mac.md` |
