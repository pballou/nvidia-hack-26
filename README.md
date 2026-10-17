# GUI assist

Team **sherpas**: Patrick, Kyle, Charlie.

**Say what you want, and the agent does it, in any app, from pixels alone.**

This is an accessibility agent built for the
[NVIDIA x Red Hat Small Models Hack](https://luma.com/nvidia-redhat-small-models-hack)
(Raleigh, NC, 17–18 October 2026), **Fine-Tuned & Specialized** track. A user describes
a task in plain language. NVIDIA Nemotron 3.5 Lightning plans the steps, and our small
fine-tuned vision model finds what to click on the screen. Because the agent works
from screenshots instead of the app's code or DOM, it applies to apps whose internals
aren't accessible.

> Status: work in progress during the hackathon weekend. Sections marked *TBD* are
> filled in as results land.

## Quick start

*TBD (Patrick): commands to run the localhost app and the models.*

## Tech stack & architecture

```text
user request ─► Nemotron 3.5 Lightning (planner, tool calls) ──► click("the gear icon …")
                        ▲                                              │
     text observation   │                                              ▼
 (screen summary, URL,  │          screenshot ─► fine-tuned grounder VLM ─► click point (0–1000)
  "screen changed?")    │                                              │
                        └──────────── Playwright browser ◄─────────────┘
```

| Component | Role |
| --- | --- |
| Small open VLM (≤ 3B): Qwen3.5-2B or Qwen3-VL-2B + LoRA, picked after baselines | Grounding model we fine-tune |
| Nemotron 3.5 Lightning (30B-A3B, NVFP4) | Text planner and tool-calling agent; also rewrites training instructions |
| NVIDIA NeMo Gym | Environments (`gui_grounding`, `web_agent`), rollouts, rewards, profiling, eval compare, BLADE failure analysis |
| TRL | SFT and GRPO; the GRPO reward comes from the Gym `/verify` endpoint |
| vLLM | Serving both models and GRPO rollouts |
| NVIDIA Brev (1× H200) | Training and eval |
| Playwright | Sandbox browser automation |

Total generative parameters stay within the event's 33B cap: Lightning 30B plus the
grounder ≤ 3B. The screen-summary role uses the same base weights with the LoRA
adapter turned off (one 2B model plus an adapter).

## How to reproduce

- **Setup:** [SETUP.md](SETUP.md) covers tools, access and keys. Copy `.env.example`
  to `.env` and `env.yaml.example` to `env.yaml`.
- **GPU serving and training:** [docs/gpu-box.md](docs/gpu-box.md).
- *TBD:* data prep, training and eval commands.

## Data & provenance

| Data | Use | License / notes |
| --- | --- | --- |
| [OS-Atlas-data](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data): Windows, AndroidWorld, UIBert, Widget Captioning, FineWeb subsets | Training | Apache-2.0 (dataset card). **Linux and macOS desktop subsets excluded:** they were captured on the same machines as ScreenSpot-v2's desktop test screenshots (details in [docs/research-notes.md](docs/research-notes.md) §5). |
| [GroundCUA](https://huggingface.co/datasets/ServiceNow/GroundCUA) | Training (desktop) | MIT |
| Lightning-rewritten instructions | Training | Synthetic; published with its prompt and script (*TBD link*) |
| [ScreenSpot-v2](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2) (revision `5efbb1f1b5`) | **Held-out test only** | Apache-2.0 |

## Results

*TBD.* ScreenSpot-v2 for base vs. SFT vs. SFT+GRPO, reported three ways (all 1,272
rows / 1,128 rows on v1 images / 144 new rows), split by mobile/desktop/web × text/icon.
Plus browser-task completion with the base grounder vs. the tuned grounder, all
produced with `gym eval run` and `gym eval compare`.

## Known limitations & next steps

*TBD.*

## Team

| Name | Role | Contact |
| --- | --- | --- |
| Patrick | Agent and Lightning integration, `web_agent` env, sandbox app, demo | TBD |
| Kyle | Training and serving (TRL, vLLM, GPU budget) | TBD |
| Charlie | Data and evaluation (data prep, `gui_grounding` env, ScreenSpot eval, BLADE) | TBD |

## Docs

- [SETUP.md](SETUP.md): tool setup (GitHub, Brev CLI, API keys, Python envs)
- [docs/gpu-box.md](docs/gpu-box.md): GPU serving and training commands (pinned stack)
- [docs/plan.md](docs/plan.md): architecture, decisions, milestones, workstreams
- [docs/submission.md](docs/submission.md): submission checklist, publishing, video plan
- [docs/research-notes.md](docs/research-notes.md): verified facts and sources
- [AGENTS.md](AGENTS.md): rules for teammates and coding agents

## License

TBD. Model and dataset licenses are recorded per component before publication.
