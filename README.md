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

**Fully local (Mac):** the merged grounder on :8011 also writes the screen summaries
(bring it up with [docs/local-mac.md](docs/local-mac.md)):

```bash
scripts/demo_up.sh --agent real --vlm-url http://127.0.0.1:8011/v1 --summary-model grounder --vlm-label "Mac (vllm-metal)"
```

The script checks that the VLM (`--vlm-url`, else `vlm.base_url` in `configs/agent.yaml`)
and Lightning respond. It also checks that the VLM serves the chosen grounder and summary
model, then prints the console URL. The console starts with Grounder `grounder` (the LoRA;
`--grounder base` for the adapter off). It refuses a run whose grounder the VLM doesn't
serve, and shows where each model runs (the Logs page does this per run). Then run an easy
task such as "Make the text bigger, I can't read this."

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

- **Serve** with `scripts/run_asr.sh` (127.0.0.1:8096). The console uses it by default
  (`demo_up.sh` and `run_console.sh` alike). Until `/health` answers, the mic shows as
  off, and the page turns voice on by itself once the sidecar is up. `--asr stub` is
  fixed text for UI testing only: the page labels it and never runs it by itself.
- **Use it in the console:**
  - Click the mic, or hold Right Option (or Space when no text field has focus), and
    say the request.
  - It runs after a 1.5 s countdown, which Esc cancels. In Options you can choose
    5 s or no auto-run instead.
  - Saying "stop" (or "stop it", "cancel that") during a run stops it.

Run the pieces separately with `scripts/run_sandbox.sh`, `scripts/run_asr.sh` and
`scripts/run_console.sh --agent mock|real [--headed]` (voice uses Parakeet; `--asr stub` for fixed text).

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
| Small open VLM (≤ 3B): Qwen3.5-2B + LoRA, chosen over Qwen3-VL-2B after baselines (gate D1) | Grounding model we fine-tune |
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
- **Data prep, training and eval:** the exact command behind every GPU job is in
  [docs/runs.md](docs/runs.md); the end-to-end sequence is under "Reproduce" in
  [docs/bounty-nemo-gym.md](docs/bounty-nemo-gym.md).

## Data & provenance

| Data | Use | License / notes |
| --- | --- | --- |
| [OS-Atlas-data](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data): Linux, macOS, AndroidWorld, UIBert, FineWeb subsets | Training | Apache-2.0 (dataset card). The Linux/macOS desktop data was captured on the same machines as 28 ScreenSpot-v2 desktop screenshots, so those 28 rows are excluded from our test set (see below). |
| Lightning-rewritten instructions | Training (`sft-lightning`, the overnight loop's starting point) | Synthetic; prompt, script and method in [data_prep/rewrite/](data_prep/rewrite/) (run record `data_prep/rewrite/runs/os_atlas_final_v4/`); data in [patrickballou/gui-assist-data](https://huggingface.co/datasets/patrickballou/gui-assist-data) (public from Sunday ~9:30) |
| [ScreenSpot-v2](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2) (revision `5efbb1f1b5`) | **Held-out test only**, as **SSv2-clean**: 1,244 of 1,272 rows. The excluded 28 are listed in `benchmarks/screenspot_v2/excluded_rows.json`. | Apache-2.0 |

## Results

So far (Saturday evening), all produced with `gym eval run` and `gym eval compare`. Full
log with per-platform splits: [docs/results.md](docs/results.md); commands: [docs/runs.md](docs/runs.md).

**Grounding** (Qwen3.5-2B, 2 MP; accuracy = the click point lands inside the target box):

| Grounder | OS-Atlas dev (969 rows) | ScreenSpot-v2-clean (1,244 rows) |
| --- | --- | --- |
| Base | 76.4% | 86.3% (text 92.5, icon 78.1) |
| SFT, raw labels (`sft-template`: LoRA r16, 1 epoch on 18,652 rows) | 85.66% | 89.07% (text 92.3, icon 84.8) |
| SFT, Lightning-rewritten instructions (`sft-lightning`: 18,299 rows, same config and seed) | 86.27% | 88.10% (text 93.0, icon 81.7) |
| **Final: `sft-lightning` + GRPO round 1 in NeMo Gym** (170 steps, Gym `/verify` point-in-box reward) | **87.00%** | **90.35%** (text 94.2, icon 85.4; vs `sft-lightning`: 34 rows fixed, 6 broken) |

`sft-lightning` won the round-0 pick on dev (the template scored 85.86% in the same harness;
dev run-to-run noise is about 0.2 points), so the overnight loop starts from it. On the held-out
ScreenSpot-v2-clean it scores 0.97 points below the template, so the rewrite ablation is mixed. We
choose checkpoints on dev only and report both.

**Browser tasks** (Gym `web_agent`, 12 sandbox tasks × 3 repeats, official BF16 Lightning;
only the grounder changes): verified success 18/36 with `base` vs **22/36** with
`grounder` = `sft-template`. Failed actions went from 10 to 0, and false success claims
from 8 to 4. With a strict `mark-all-read` check, tuned is 20/36 (+2): the current check
doesn't guard the bell notifications ([BLADE report](docs/blade/agent_base_vs_sft.md)). The
FINAL grounder (SFT + GRPO) scored 17/36. All three are within noise at n=36. The robust in-agent
gains are that the tuned grounders always return a usable point (failed actions 10 → 0) and fix
the small adjacent-option target (text-size-large 0/3 → 3/3). End-to-end success depends mostly
on the planner's choices and on what the screen description shows
([diagnosis](docs/blade/agent_final_vs_sft.md)).

The overnight loop kept a round only if it gained ≥ 0.5 points on dev: round 1 (GRPO) was kept,
round 2 (−0.11) was not, so round 1 is the final adapter above.

## Known limitations & next steps

- **One sandbox app** for the agent eval, and the prompts were tuned on its traces, so treat
  absolute success as in-domain. Base vs tuned (same rows, prompts and tools) is the
  comparable number.
- **Small n:** agent success with the three grounders (18, 22 and 17 of 36) is within noise:
  SFT vs base +0.11 mean reward, 95% CI [-0.06, +0.28]; FINAL vs base −0.03, CI [-0.20, +0.14].
  The drop in failed actions is significant.
- **Still 0/3:** the canvas-chart task and the two scroll-dependent tasks.

Next steps, ranked by the [BLADE report](docs/blade/agent_base_vs_sft.md). The agent pipeline
is frozen until Sunday's grounder rerun, so the comparison with Saturday's base arm holds:

1. **Observation pipeline.** The 2B screen describer drops toggle state and cuts off long grids
   (high-contrast state missing in 6/6 rollouts; the notifications grid cut in 15/15 first
   views). Report STATE reliably and add "more below" scroll hints.
2. **Agent-loop guards.** Refuse `done(success=true)` when the latest toast or STATE contradicts
   the request, and catch on/off oscillation. The interactive stuck stop also needs fixing: an
   occasional screen change resets its count, so 7 of 36 thinking-off runs still hit the
   25-step cap without a stuck stop.
3. **Planner prompt `agent-v5`.** Scroll when the target isn't listed, and stop once a toast
   confirms the result.
4. **Grounder data and RL.** Add segmented controls and adjacent-option targets (the tuned
   grounder's "L" hits land 1 px inside the box), and add centre-distance pressure to the GRPO
   reward.
5. **Verifier.** Make the `mark-all-read` check guard the bell notifications, then rerun both
   arms.
6. **Local latency.** Reuse the screen description when the screen didn't change, cap thinking
   tokens, and stop a no-change click from triggering retry thinking (a 31.9 s step in a local
   smoke run; [bounty-lightning.md](docs/bounty-lightning.md) §7).

## Team

| Name | Role | Contact |
| --- | --- | --- |
| Patrick | Agent and Lightning integration, `web_agent` env, sandbox app, demo | pballou@sherpa6.com |
| Kyle | Training and serving (TRL, vLLM, GPU budget) | kodonnell@sherpa6.com |
| Charlie | Data and evaluation (data prep, `gui_grounding` env, ScreenSpot eval, BLADE) | carleth@sherpa6.com |

## Docs

- [docs/status.md](docs/status.md): team status, where everything is right now
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

Code in this repo: [Apache License 2.0](LICENSE). Vendored NeMo Gym skills under `.agents/skills/` keep
their own license ([LICENSE-NeMo-Gym](.agents/skills/LICENSE-NeMo-Gym)).

Models and data carry their own licenses, which are recorded on each Hugging Face card: the base
model [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), NVIDIA Nemotron 3.5 Lightning and
Parakeet (NVIDIA's model licenses), and the OS-Atlas source data.
