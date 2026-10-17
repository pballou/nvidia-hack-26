# Submission

**Due Sunday 2026-10-18, 11:00am EDT.** One submission per team, via the
[Airtable form](https://airtable.com/appm7NS9oN8K3friP/pagqXe6ElIlXx6oa3/form).
Target: everything finished by 10:30.

## Form fields → what we need → owner

| Form field | What we put there | Owner | Source |
| --- | --- | --- | --- |
| Project Name / Team Name | **TBD** (decide today) | all | — |
| Team Member Names & Email Addresses | Patrick, Kyle, Charlie + emails (in the form only, not the public repo) | Patrick | — |
| Submission Track | Fine-Tuned & Specialized | — | — |
| Submission Description | 1 paragraph: problem → approach → results | Patrick | README pitch |
| Demo Video URL | Unlisted **YouTube** (accepted by both the form and the checklist), 3–5 min | Patrick | §Video |
| GitHub Repo | `https://github.com/pballou/nvidia-hack-26` (public) | — | — |
| AI Models Used | Nemotron 3.5 Lightning (30B-A3B, NVFP4); our fine-tuned grounder (Qwen3.5-2B or Qwen3-VL-2B + LoRA) | Kyle | — |
| Tools Used | NeMo Gym (envs, rollouts, profiling, compare, BLADE), vLLM, NVIDIA Brev (H200), TRL, Playwright | all | — |
| ☑ Lightning bounty + explanation | What Lightning does (planner/tool-caller; instruction rewriting), why, how we optimize it (thinking budget, tool schema, Gym-measured success/latency) | Patrick | `docs/bounty-lightning.md` (write Sunday) |
| ☑ NeMo Gym bounty + explanation | Gym setup (`gui_grounding`, `web_agent`), **baseline vs. final** numbers from `gym eval compare`, what we changed and why (BLADE findings → data/GRPO) | Charlie + Kyle | `docs/bounty-nemo-gym.md` (write Sunday) |
| ☑ Red Hat vLLM bounty | (checkbox only) vLLM serves both models; GRPO rollouts via vLLM colocate | Kyle | — |

## README must include (organizer checklist)

The README skeleton has a section for each:

- [ ] **Quick start**: commands to run (Patrick: localhost app plus model serving)
- [ ] **Tech stack & architecture diagram** (simple is fine)
- [ ] **How to reproduce the demo**: env vars, API keys, sample `.env` (`.env.example`, `env.yaml.example`)
- [ ] **Datasets / synthetic data + provenance**: OS-Atlas subsets used and excluded (and why), GroundCUA if used, Lightning-rewritten instructions (published plus method), ScreenSpot-v2 (eval only)
- [ ] **Known limitations & next steps**
- [ ] **Deployed URL or a short screen capture** of the working app
- [ ] **Team roster**: names and roles. Put contacts in the form; in the public README, use GitHub handles at most.

## Publishing (required for this track: "publish it openly so others can reuse it")

Approval is granted. Patrick sets up the Hugging Face account and a **write** token,
kept in `.env` on the machine that uploads and never committed.

| Artifact | Where | Contents |
| --- | --- | --- |
| Model | HF model repo | Merged model and/or LoRA adapter, plus a model card: base model, data and licenses, training config, SSv2 results (3 ways), intended use (accessibility), limitations |
| Data | HF dataset repo | Our slice manifest (source + image filename + bbox + split), the **Lightning-rewritten instructions** with the exact prompt and script that made them. Images stay referenced by source filename, not re-hosted. |
| Code + envs | This repo | `resources_servers/gui_grounding`, `resources_servers/web_agent`, data prep, training configs, eval results |

## Video (3–5 min, camera on, show the team)

Follow the organizers' recommended flow:

| Segment | Time | Content |
| --- | --- | --- |
| 1. Team | 20–30 s | Team name; who did what |
| 2. Pitch | 30–40 s | Accessibility: "say what you want, the agent does it in any app, from pixels alone", using small local models |
| 3. Live demo | 45–60 s | Core loop with minimal cuts: user request → Lightning plans → grounder clicks (show the click overlay) → page changes → task done. Show one base-vs-tuned difference. |
| 4. How we built it | 60–90 s | Architecture diagram; NeMo Gym envs and the base → SFT → GRPO results; BLADE failure analysis; vLLM on the H200; Lightning NVFP4 plus thinking control. Challenges: the ScreenSpot contamination we found and excluded, coordinate conventions, the one-GPU budget. |
| 5. So what | 20–30 s | Works on apps without accessible code; small enough to run locally; measured improvement |

**Recording plan:** do a dry run Saturday night if possible. Record Sunday 8:30–10:00,
upload unlisted by 10:15, submit the form by 10:30.
