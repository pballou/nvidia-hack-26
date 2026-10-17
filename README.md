# nvidia-hack-26

A small, specialized GUI visual-grounding model, built for the
[NVIDIA x Red Hat Small Models Hack](https://luma.com/nvidia-redhat-small-models-hack)
(Raleigh, NC, 17–18 October 2026) in the **Fine-Tuned & Specialized** track.

Given a screenshot and an instruction such as "open the settings menu", the model
predicts where to click. At demo time, NVIDIA Nemotron 3.5 Lightning (text-only)
plans browser tasks and calls the grounding model as a tool. Playwright executes the
actions in a sandbox web app, and checks on the sandbox's state decide whether each
task succeeded.

> Status: work in progress during the hackathon weekend.

## Components

| Component | Role |
| --- | --- |
| Small open VLM (≤ 3B): Qwen3.5-2B or Qwen3-VL-2B, picked after baselines | Grounding model we fine-tune |
| Nemotron 3.5 Lightning (30B-A3B) | Text planner and tool-calling agent |
| NVIDIA NeMo Gym | Environments, rollouts, and rewards for eval and training |
| vLLM | Model serving and RL rollout generation |
| Playwright | Sandbox browser automation |
| [OS-Atlas data](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data) | Training data (reviewed subset) |
| ScreenSpot-v2 | Held-out benchmark (never trained on) |

## Docs

- [SETUP.md](SETUP.md): tool setup (GitHub, Brev CLI, API keys, Python envs)
- [docs/plan.md](docs/plan.md): architecture, milestones, workstreams
- [docs/research-notes.md](docs/research-notes.md): verified facts and sources
- [AGENTS.md](AGENTS.md): rules for teammates and coding agents

## Results

Planned comparisons: base model, SFT, and SFT+GRPO (if completed) on ScreenSpot-v2,
reported overall and split into text and icon targets, plus browser-task completion
with the base grounder vs. the tuned grounder. We'll report limitations and negative
results too.

## License

TBD. Model and dataset licenses are recorded per component before publication.
