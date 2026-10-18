# GUI Assist

Team **sherpas**: Patrick, Kyle, Charlie.

**Say what you want, and the agent does it, in any app, from pixels alone.**

Only 14% of customer-service issues are fully resolved in self-service, mostly because
people can't find what they need, so they call: a median $13.50 per agent contact
against $1.84 for self-service ([Gartner](https://www.gartner.com/en/newsroom/press-releases/2024-08-19-gartner-survey-finds-only-14-percent-of-customer-service-issues-are-fully-resolved-in-self-service)).
GUI Assist lets a bank, utility or benefits portal's customers say what they want and
watch it happen on the real site, with small models that run on the company's own
hardware. Market sizing, ROI assumptions and sources: [docs/pitch.md](docs/pitch.md).

**On Hugging Face:** model [patrickballou/gui-assist-grounder](https://huggingface.co/patrickballou/gui-assist-grounder) ·
data [patrickballou/gui-assist-data](https://huggingface.co/datasets/patrickballou/gui-assist-data). Run it yourself: [Quick start](#quick-start).

This is an accessibility agent built for the
[NVIDIA x Red Hat Small Models Hack](https://luma.com/nvidia-redhat-small-models-hack)
(Raleigh, NC, 17–18 October 2026), **Fine-Tuned & Specialized** track. A user describes
a task in plain language. NVIDIA Nemotron 3.5 Lightning plans the steps, and our small
fine-tuned vision model finds what to click on the screen. Because the agent works
from screenshots instead of the app's code or DOM, it applies to apps whose internals
aren't accessible.

> Status: final hackathon submission (Sun 2026-10-18). Results are logged in
> [docs/results.md](docs/results.md), with the command behind each in [docs/runs.md](docs/runs.md).

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

**3. Get the models from Hugging Face.**

| What | Where | Notes |
| --- | --- | --- |
| Our grounder (final: SFT + GRPO in NeMo Gym) | [patrickballou/gui-assist-grounder](https://huggingface.co/patrickballou/gui-assist-grounder) | Merged weights at the root; PEFT LoRA in `adapter/`; model card with all results |
| Our training data | [patrickballou/gui-assist-data](https://huggingface.co/datasets/patrickballou/gui-assist-data) | Slice manifest, dev/train files, Lightning-rewritten instructions and method (no images) |
| Base model | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | Needed to serve the LoRA (`base` = adapter off) |
| Planner | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) | Official BF16. Or skip the download and use hosted Lightning with `NVIDIA_API_KEY` |

```bash
hf download patrickballou/gui-assist-grounder --local-dir models/gui-assist-grounder
hf download Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16      # ~60 GB; skip for hosted Lightning
```

**4. Serve the grounder and run the real agent.** One vLLM process serves `base` (screen
summaries) and the LoRA as `grounder` (clicks), so it counts as one 2B model.

*On an NVIDIA GPU (vLLM 0.31.0):*

```bash
vllm serve Qwen/Qwen3.5-2B --served-model-name base --port 8001 \
  --enable-lora --max-lora-rank 64 --lora-modules grounder=models/gui-assist-grounder/adapter \
  --max-model-len 16384 --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --mm-processor-kwargs '{"size": {"longest_edge": 2073600, "shortest_edge": 65536}}' \
  --default-chat-template-kwargs '{"enable_thinking": false}'
scripts/demo_up.sh --agent real --lightning hosted --vlm-url http://127.0.0.1:8001/v1 --summary-model base
```

Self-hosting Lightning on a Hopper GPU needs the vLLM #61603 patch
([docs/gpu-box.md](docs/gpu-box.md) §2). Never add `--speculative-config`: it would add
models beyond the 33B budget.

*Fully local on a Mac (Apple silicon, 128 GB; what the demo video uses):* install vllm-metal
0.31.0 ([docs/local-mac.md](docs/local-mac.md) §0), start Lightning on :8000 (§2; ~1.2 s per
planner step on an idle server, ~4 s inside the agent), then:

```bash
# vllm-metal names the language-model layers differently, so re-key the adapter (same tensors):
python3 -I scripts/gate_g_lora_keys.py --adapter models/gui-assist-grounder/adapter --out models/gui-assist-mlx-keys
# A memory watchdog (required by the launcher) and the grounder on :8011, in two terminals:
python3 -I scripts/gate_g_watch.py --pidfile ~/models/logs/local-grounder-8011.pid --port 8011 --probe-margin-s 1.5
LORA=models/gui-assist-mlx-keys GPU_UTIL=0.08 MAX_BATCHED=2560 MIN_FREE_PCT=26 scripts/run_local_grounder.sh
scripts/demo_up.sh --agent real --vlm-url http://127.0.0.1:8011/v1 --summary-model base --vlm-label "Mac (vllm-metal)"
```

Quit memory-heavy apps first (the launcher refuses below 26% free). Keep the browser
headless: `--headed` renders different pixels (unmeasured; pay-50 fell from 8/8 to 1/4).

`demo_up.sh` checks that the VLM serves both names and that Lightning answers, then prints
the console URL (<http://127.0.0.1:8095>). The console starts with the tuned `grounder`
(the Grounder toggle switches to `base`), thinking off and the "ask when unclear" check
on (`--thinking retry|on`, `--clarify off`). Try "Make the text bigger, I can't read
this." or "Pay fifty dollars toward my bill." The ✓/✗ comes from the sandbox's own
state, which the agent never sees.

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
`scripts/run_console.sh --agent mock|real` (voice uses Parakeet; `--asr stub` for fixed text).

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
| Lightning-rewritten instructions | Training (`sft-lightning`, the overnight loop's starting point) | Synthetic; prompt, script and method in [data_prep/rewrite/](data_prep/rewrite/) (run record `data_prep/rewrite/runs/os_atlas_final_v4/`); data in [patrickballou/gui-assist-data](https://huggingface.co/datasets/patrickballou/gui-assist-data) |
| [ScreenSpot-v2](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2) (revision `5efbb1f1b5`) | **Held-out test only**, as **SSv2-clean**: 1,244 of 1,272 rows. The excluded 28 are listed in `benchmarks/screenspot_v2/excluded_rows.json`. | Apache-2.0 |

## Results

Final numbers, all produced with `gym eval run` and `gym eval compare`. Full
log with per-platform splits: [docs/results.md](docs/results.md); commands: [docs/runs.md](docs/runs.md).

**Grounding** (Qwen3.5-2B, 2 MP; accuracy = the click point lands inside the target box):

| Grounder | OS-Atlas dev (969 rows) | ScreenSpot-v2-clean (1,244 rows) |
| --- | --- | --- |
| Base | 76.4% | 86.25% (text 92.5, icon 78.1) |
| SFT, raw labels (`sft-template`: LoRA r16, 1 epoch on 18,652 rows) | 85.66% | 89.07% (text 92.3, icon 84.8) |
| SFT, Lightning-rewritten instructions (`sft-lightning`: 18,299 rows, same config and seed) | 86.27% | 88.10% (text 93.0, icon 81.7) |
| **Final: `sft-lightning` + GRPO round 1 in NeMo Gym** (170 steps, Gym `/verify` point-in-box reward) | **87.00%** | **90.35%** (text 94.2, icon 85.4; vs `sft-lightning`: 34 rows fixed, 6 broken) |

`sft-lightning` won the round-0 pick on dev (the template scored 85.86% in the same harness;
dev run-to-run noise is about 0.2 points), so the overnight loop starts from it. On the held-out
ScreenSpot-v2-clean it scores 0.97 points below the template, so the rewrite ablation is mixed. We
choose checkpoints on dev only and report both.

**Browser tasks** (Gym `web_agent`, 12 sandbox tasks × 3 repeats, official BF16 Lightning;
only the grounder changes): verified success 18/36 with `base` vs 22/36 with
`grounder` = `sft-template`. Failed actions went from 10 to 0, and false success claims
from 8 to 4 (FINAL: 11). With a strict `mark-all-read` check, tuned is 20/36 (+2): the current check
doesn't guard the bell notifications ([BLADE report](docs/blade/agent_base_vs_sft.md)). The
FINAL grounder (SFT + GRPO) scored 17/36. All three are within noise at n=36. The robust in-agent
gains are that the tuned grounders always return a usable point (failed actions 10 → 0) and fix
the small adjacent-option target (text-size-large 0/3 → 3/3). End-to-end success depends mostly
on the planner's choices and on what the screen description shows
([diagnosis](docs/blade/agent_final_vs_sft.md)).

The overnight loop kept a round only if it gained ≥ 0.5 points on dev: round 1 (GRPO) was kept,
round 2 (−0.10) was not, so round 1 is the final adapter above.

## Known limitations & next steps

- **One sandbox app** for the agent eval, and the prompts were tuned on its traces, so treat
  absolute success as in-domain. Base vs tuned (same rows, prompts and tools) is the
  comparable number.
- **Small n:** agent success with the three grounders (18, 22 and 17 of 36) is within noise:
  SFT vs base +0.11 mean reward, 95% CI [-0.06, +0.28]; FINAL vs base −0.03, CI [-0.20, +0.14].
  The drop in failed actions is significant.
- **Still 0/3 in all three Gym arms:** the two scroll-dependent tasks. The canvas-chart task is 0/3
  except FINAL's 1/3.

Next steps, ranked by the [BLADE report](docs/blade/agent_base_vs_sft.md). The agent pipeline
stayed frozen through the FINAL grounder rerun (Sat 22:04), so all three arms compare:

1. **Observation pipeline.** The 2B screen describer drops toggle state and cuts off long grids
   (high-contrast state missing in 6/6 rollouts; the notifications grid cut in 15/15 first
   views). Report STATE reliably and add "more below" scroll hints.
2. **Agent-loop guards.** Refuse `done(success=true)` when the latest toast or STATE contradicts
   the request, and catch on/off oscillation. The interactive stuck stop also needs fixing: an
   occasional screen change resets its count, so 7 of 36 thinking-off runs still hit the
   25-step cap without a stuck stop.
3. **Planner prompt `agent-v5`.** Scroll when the target isn't listed, and stop once a toast
   confirms the result.
4. **Grounder data and RL.** Add segmented controls and adjacent-option targets (`sft-template`'s
   "L" hits land 1 px inside the box), and add centre-distance pressure to the GRPO
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
- [docs/bounty-vllm.md](docs/bounty-vllm.md): Red Hat vLLM bounty write-up (serving architecture, configuration, measured results)
- [docs/research-notes.md](docs/research-notes.md): verified facts and sources
- [AGENTS.md](AGENTS.md): rules for teammates and coding agents

## License

Code in this repo: [Apache License 2.0](LICENSE). Copyright 2026 Patrick Ballou, Kyle O'Donnell and Charlie Arleth ([NOTICE](NOTICE)). Vendored NeMo Gym skills under `.agents/skills/` keep
their own license ([LICENSE-NeMo-Gym](.agents/skills/LICENSE-NeMo-Gym)).

Models and data carry their own licenses, which are recorded on each Hugging Face card: the base
model [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), NVIDIA Nemotron 3.5 Lightning and
Parakeet (NVIDIA's model licenses), and the OS-Atlas source data.
