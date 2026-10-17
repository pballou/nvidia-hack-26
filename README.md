# GUI assist

Team **sherpas**: Patrick, Kyle, Charlie.

**Say what you want, and the agent does it, in any app, from pixels alone.**

Only 14% of customer-service issues are fully resolved in self-service, mostly because
people can't find what they need, so they call: a median $13.50 per agent contact
against $1.84 for self-service ([Gartner](https://www.gartner.com/en/newsroom/press-releases/2024-08-19-gartner-survey-finds-only-14-percent-of-customer-service-issues-are-fully-resolved-in-self-service)).
GUI assist lets a bank, utility or benefits portal's customers say what they want and
watch it happen on the real site, with small models that run on the company's own
hardware. Market sizing, ROI assumptions and sources: [docs/pitch.md](docs/pitch.md).

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

Tested on macOS (Apple silicon) with [uv](https://docs.astral.sh/uv/). It needs
Python 3.13.14, which uv downloads for you.

**1. Setup (once).**

```bash
git clone https://github.com/pballou/nvidia-hack-26.git && cd nvidia-hack-26
uv venv --python 3.13.14 .venv-gym
uv pip install --python .venv-gym/bin/python "nemo-gym==0.7.0" -r sandbox/requirements.txt -r console/requirements.txt
.venv-gym/bin/playwright install chromium
cp .env.example .env        # then set NVIDIA_API_KEY (build.nvidia.com) for hosted Lightning
```

**2. Try it without any models.** This starts the sandbox app (:8090) and the GUI
assist console (:8095) with a scripted mock agent that clicks by pixel coordinates:

```bash
scripts/demo_up.sh --agent mock
```

Open <http://127.0.0.1:8095>, pick **"Turn on high contrast."** in the task list and
press Enter. You'll see each step, where the agent clicked, and a ✓/✗ checked against
the sandbox's own state (the agent never sees that state).

**3. Bring up the models** (details: [docs/local-mac.md](docs/local-mac.md)).

- **Grounder VLM on :8001.** Tunnel to the GPU box (served names `base` and
  `grounder`, [docs/gpu-box.md](docs/gpu-box.md) §3):
  `ssh -N -o ControlPath=none -o ExitOnForwardFailure=yes -L 8001:localhost:8001 permanent-tan-marmot`.
  Or serve it on the Mac (docs/local-mac.md §3b).
- **Lightning:** local on :8000, served as `lightning` (the official NVIDIA BF16
  checkpoint, docs/local-mac.md §2; ~1.2 s per planner step). Or hosted, which needs
  only `NVIDIA_API_KEY` in `.env` but is slow (25–80 s per step).

**4. Run the real agent.**

```bash
scripts/demo_up.sh --agent real                        # local Lightning; --lightning hosted to use the API; --headed to watch the browser
```

The script checks that the VLM (`vlm.base_url` in `configs/agent.yaml`) and Lightning
respond, then prints the console URL. In the console, choose Grounder `base` (before
SFT) or `grounder`, then run an easy task such as "Make the text bigger, I can't read
this."

**5. Voice input (optional).** NVIDIA Parakeet TDT 0.6B v3 runs locally through
NeMo-Speech.cpp 0.2.0. Audio never leaves the machine, and there's no cloud speech.

- **Install** the pinned release binary and the official weights
  (`nvidia/parakeet-tdt-0.6b-v3`, q8_0 GGUF, 714 MB; the tool checks both SHAs):

  ```bash
  curl -fsSLO https://raw.githubusercontent.com/NVIDIA/NeMo-Speech.cpp/v0.2.0/scripts/install.sh
  shasum -a 256 install.sh   # c364c26ba45fb61b772dc75e9d9d2cb7373cddcda20d6f4a3a1347823c31fc9b
  sh install.sh --version 0.2.0 --backend metal --binary-only --no-modify-path
  ~/.local/bin/nemo-speech pull parakeet-tdt
  ```

- **Serve** with `scripts/run_asr.sh` (127.0.0.1:8096). `demo_up.sh` uses it when
  `/health` answers; otherwise voice falls back to a stub that returns fixed text.
- **Use it in the console:**
  - Click the mic, or hold Right Option (or Space when no text field has focus), and
    say the request.
  - It runs after a 1.5 s countdown, which Esc cancels.
  - Saying "stop" during a run stops it.

Run the pieces separately with `scripts/run_sandbox.sh`, `scripts/run_asr.sh` and
`scripts/run_console.sh --agent mock|real [--asr parakeet] [--headed]`.

Tests:
`.venv-gym/bin/python -m pytest sandbox/tests console/tests -q`. They cover the
scripted oracle that solves all 12 tasks, wrong-action checks, session isolation,
pixel determinism, the no-egress check, and the console API.

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
| Nemotron 3.5 Lightning (30B-A3B, official NVIDIA BF16) | Text planner and tool-calling agent; also rewrites training instructions |
| NVIDIA NeMo Gym | Environments (`gui_grounding`, `web_agent`), rollouts, rewards, profiling, eval compare, BLADE failure analysis |
| TRL | SFT and GRPO; the GRPO reward comes from the Gym `/verify` endpoint |
| vLLM | Serving both models and GRPO rollouts |
| NVIDIA Brev (1× H200) | Training and eval |
| Playwright | Sandbox browser automation |

Total parameters stay within the event's 33B cap: Lightning 30B + grounder 2B
(Qwen3.5-2B + LoRA) + NVIDIA Parakeet 0.6B for voice input = 32.6B (model-card counts). The screen-summary role uses the same base weights with the LoRA
adapter turned off (one 2B model plus an adapter).

## How to reproduce

- **Setup:** [SETUP.md](SETUP.md) covers tools, access and keys. Copy `.env.example`
  to `.env` and `env.yaml.example` to `env.yaml`.
- **GPU serving and training:** [docs/gpu-box.md](docs/gpu-box.md).
- *TBD:* data prep, training and eval commands.

## Data & provenance

| Data | Use | License / notes |
| --- | --- | --- |
| [OS-Atlas-data](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data): Linux, macOS, AndroidWorld, UIBert, FineWeb subsets | Training | Apache-2.0 (dataset card). The Linux/macOS desktop data was captured on the same machines as 28 ScreenSpot-v2 desktop screenshots, so those 28 rows are excluded from our test set (see below). |
| Lightning-rewritten instructions | Training | Synthetic; published with its prompt and script (*TBD link*) |
| [ScreenSpot-v2](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2) (revision `5efbb1f1b5`) | **Held-out test only**, as **SSv2-clean**: 1,244 of 1,272 rows. The excluded 28 are listed in `benchmarks/screenspot_v2/excluded_rows.json`. | Apache-2.0 |

## Results

*TBD.* ScreenSpot-v2-clean (1,244 rows) for base vs. SFT vs. SFT+GRPO, split by
mobile/desktop/web × text/icon, plus the OS-Atlas dev split.
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
- [docs/local-mac.md](docs/local-mac.md): running the demo on a Mac (vllm-metal), gates and fallbacks
- [docs/plan.md](docs/plan.md): architecture, decisions, milestones, workstreams
- [docs/submission.md](docs/submission.md): submission checklist, publishing, video plan
- [docs/bounty-nemo-gym.md](docs/bounty-nemo-gym.md): NeMo Gym bounty write-up (envs, the Gym-driven improvement loop, results)
- [docs/bounty-lightning.md](docs/bounty-lightning.md): Nemotron Lightning bounty write-up (planner, data engine, optimizations)
- [docs/research-notes.md](docs/research-notes.md): verified facts and sources
- [AGENTS.md](AGENTS.md): rules for teammates and coding agents

## License

TBD. Model and dataset licenses are recorded per component before publication.
