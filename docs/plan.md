# Plan

Proposed plan for the hackathon weekend; adjust it at kickoff. Background facts are
in [research-notes.md](research-notes.md), and tool setup is in [SETUP.md](../SETUP.md).

## Goal

A ≤3B VLM fine-tuned for GUI click grounding, improved through NeMo Gym, and used as
a tool by Nemotron 3.5 Lightning in a browser agent.

Headline results we want:

1. **ScreenSpot-v2:** base vs. SFT vs. SFT+GRPO, overall and split by
   mobile/desktop/web × text/icon, produced with `gym eval run` and compared with
   `gym eval compare`.
2. **Agent task completion:** the same Lightning planner with the base grounder vs.
   the tuned grounder, measured in our Gym `web_agent` env.
3. **A BLADE failure analysis** that explains what improved and why.
4. **Published artifacts:** the model (or adapter), the data slice and synthetic
   instructions with their methodology, the Gym envs, and eval results.

## Architecture

```text
            OS-Atlas slice ──► data_prep ──► canonical JSONL ──► Gym rows ─┐
  Lightning (rewrites a11y labels into natural instructions) ──┘           │
                                                                           ▼
  ScreenSpot-v2 (eval only) ──► Gym rows ──► resources_servers/gui_grounding (verify = point in bbox)
                                                │            ▲
                    gym eval run / profile /    │            │ POST /verify (reward)
                    compare + BLADE             ▼            │
                                       TRL SFT ──► TRL GRPO ─┘ ──► tuned grounder (vLLM :8001)
                                                                           │
  resources_servers/web_agent:  Lightning (planner, :8000 or hosted) ── click(target) ──► grounder
                                 Playwright sandbox app ──► verify() reads app state
```

## Planned repo layout

```text
common/                  shared code: grounding prompt(s), output parsing, coordinate conversion
data_prep/               OS-Atlas slicing/cleaning/dedup, ScreenSpot-v2 → JSONL, Lightning rewriting
resources_servers/
  gui_grounding/         Gym env A (from Gym's gui_coordinate): screenshot + instruction → click
  web_agent/             Gym env B (from Gym's interactive_browser): Lightning drives the sandbox
benchmarks/screenspot_v2/  BLADE package: skill/, rollouts/, golden_reports/
training/                TRL SFT + GRPO configs and scripts (reward = Gym /verify)
sandbox/                 demo web app exposing per-session state
scripts/                 serving and launch helpers for the Brev box
docs/                    plan, research notes, results
```

## Shared contracts (agree on these before splitting up)

- **Canonical grounding sample** (one JSON object per target):
  `{id, source, image_path, img_w, img_h, instruction, bbox: [x1, y1, x2, y2] (pixels), platform: mobile|desktop|web, data_type: text|icon|null, split: train|dev|test}`.
- **One module owns the model interface:** `common/grounding.py` holds the prompt
  template per base model, the output parser, and the 0–1000 ↔ pixel conversion. The
  Gym envs, TRL rewards, the agent's `click` tool, and the eval scripts all import it,
  so training and eval can't drift apart.
- **Splits:**
  - `dev` is carved from OS-Atlas **by screenshot** (never by row), and is used to
    choose checkpoints.
  - `test` is ScreenSpot-v2 only.
- **Endpoints on the Brev box:**
  - grounder vLLM at `:8001`, served-model-name `grounder`;
  - Lightning vLLM at `:8000`, served-model-name `lightning`.
  - During development, use hosted Lightning instead of the `:8000` server.
- **Box paths:** see [SETUP.md](../SETUP.md#5-shared-paths-on-the-brev-box).

## Milestones

**Hard deadline: everything (including demo videos) is due Sun 11:00am EDT.** Aim to
be finished by 10:30. The fine-tune (SFT, plus GRPO if it passes its gate) should be
finished by the end of Saturday, so Sunday morning is only evals, the demo, and
publishing.

| # | When (target) | Done when |
| --- | --- | --- |
| M0 Setup | Sat 1pm | Everyone passes the SETUP.md checks; the grounder is served on the box; the Gym hello world works. |
| M1 Baseline | Sat 3pm | ScreenSpot-v2 runs through `gui_grounding`. Baselines measured for Qwen3.5-2B and Qwen3-VL-2B, each with its native prompt; first BLADE report. **Gate D1: pick the base model.** |
| M2 Data + SFT | Sat 3pm → 8pm | OS-Atlas slice cleaned, deduped against SSv2, and split (start in parallel with M1); Lightning instructions generated; SFT pilot (~2k samples, ~15 min) by ~5pm; main SFT run done by ~8pm; dev and SSv2 evals; `gym eval compare`. |
| M3 Gym loop / GRPO | Sat 8pm → midnight | `gym eval profile` on training rows gives a GRPO set (0 < pass rate < 1). A 20-step TRL GRPO smoke test passes, then the main GRPO run. Hard stop at ~1am. **Gate D2: keep GRPO only if dev improves.** |
| M4 Agent | Sat afternoon → Sat night (parallel) | Sandbox app plus 8–15 tasks, including icon-only and canvas targets where DOM-based agents fail; `web_agent` env with Lightning; base vs. tuned grounder runs; click-overlay traces. |
| M5 Ship | Sun 7am → 10:30am | Final evals, BLADE report, HF publish (model, data, methodology), README results, demo video recorded, everything submitted by 11:00. |

**Budget vs. clock:** running the H200 nonstop from now until 11am Sunday would
use almost the whole $160 at ~$6.90/hr. Stop it whenever no job is running,
including overnight once GRPO and the evals finish.

## Workstreams (for dividing the work)

| Stream | Owns | First deliverable |
| --- | --- | --- |
| **A: Data & eval** | `data_prep/`, `resources_servers/gui_grounding/`, `benchmarks/screenspot_v2/`, BLADE reports | ScreenSpot-v2 Gym JSONL plus the `gui_grounding` env producing baseline numbers (M1) |
| **B: Training & serving** | `training/`, `scripts/` serving, the GPU schedule and budget, checkpoints → HF | vLLM grounder serving, then the SFT pilot (M2); GRPO with the Gym `/verify` reward (M3) |
| **C: Agent & Lightning** | `sandbox/`, `resources_servers/web_agent/`, Lightning planner and instruction rewriting, demo | `web_agent` env with hosted Lightning against the base grounder (M4); Lightning data engine for M2 |

`common/grounding.py` is shared: stream A writes the first version, and B and C review
it.

## Decision gates

- **D1 base model:** use the higher dev/SSv2 score that still leaves headroom (the
  icon splits). Prefer the model whose TRL GRPO smoke test passes. If it's a tie, take
  Qwen3.5-2B (best-documented single-H200 recipe).
- **D2 GRPO:** go only if SFT finished by ~9pm Saturday and the smoke test runs at
  ≤ ~35 s/step, so the run can finish by ~1am. Keep GRPO only if dev accuracy beats
  SFT. Otherwise ship SFT and say so.
- **Demo hardware:** run Lightning NVFP4 and the grounder co-resident on the H200.
  Stop all training first.

## GPU etiquette (one shared H200, ~23 GPU-hours total)

- Say what you're running in team chat before starting a GPU job. Use one tmux
  session per person, named after you.
- Training has priority for GPU memory; eval servers yield to it.
- Nobody leaves the box idle: whoever finishes last runs `brev stop permanent-tan-marmot`.
  The instance is stoppable and keeps its disk.
- Log each GPU run (command, config, start/end, result) in `docs/runs.md`.
