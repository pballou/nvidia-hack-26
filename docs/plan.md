# Plan

Agreed plan (as of Sat 2026-10-17). Background facts are
in [research-notes.md](research-notes.md), and tool setup is in [SETUP.md](../SETUP.md).

## Goal

A ≤3B VLM fine-tuned for GUI click grounding, improved through NeMo Gym, and used as
a tool by Nemotron 3.5 Lightning in a browser agent.

Headline results we want:

1. **ScreenSpot-v2-clean (1,244 rows):** base vs. SFT vs. SFT+GRPO, overall and split by
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
  - VLM vLLM at `:8001`, with served names `base` (base weights: baselines and
    screen summaries) and `grounder` (the LoRA adapter, after SFT);
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
| M2 Data + SFT | Sat 3pm → 8pm | Training slice built with `data_prep/sample_os_atlas.py` (Linux/macOS **included**; their 28 overlapping SSv2 rows are excluded from eval instead), sized from the pilot, per the Training data mix table below. Cleaned, deduped against SSv2, split by screenshot. Start in parallel with M1. Lightning instructions generated; SFT pilot (~2k samples, ~15 min) by ~5pm; main SFT run done by ~8pm; dev and SSv2 evals; `gym eval compare`. |
| M3 Gym loop / GRPO | Sat 8pm → midnight | `gym eval profile` on training rows gives a GRPO set (0 < pass rate < 1). A 20-step TRL GRPO smoke test passes, then the main GRPO run. Hard stop at ~1am. **Gate D2: keep GRPO only if dev improves.** |
| M4 Agent | Sat afternoon → night (parallel, mostly without the GPU) | Sandbox app plus 8–15 tasks, including icon-only and canvas targets where DOM-based agents fail; `web_agent` env with hosted Lightning. Run it against the `base` grounder before ~5pm and in the gaps between training runs; click-overlay traces. **Base-vs-tuned agent runs move to M5.** |
| M5 Ship | Sun 7am → 10:30am | Base-vs-tuned agent runs, final evals, BLADE report, HF publish (model, data, methodology), README results, demo video recorded, everything submitted by 11:00. |

**Budget vs. clock:** running the H200 nonstop from Sat 1pm to Sun 11am (~22 h)
would cost ≈ $152 at ~$6.90/hr, almost the whole $160. Stop it whenever no job is running,
including overnight once GRPO and the evals finish.

## Workstreams (for dividing the work)

| Stream | Owns | First deliverable |
| --- | --- | --- |
| **A: Data & eval** (Charlie) | `data_prep/`, `resources_servers/gui_grounding/`, `benchmarks/screenspot_v2/`, BLADE reports | ScreenSpot-v2 Gym JSONL plus the `gui_grounding` env producing baseline numbers (M1) |
| **B: Training & serving** (Kyle) | `training/`, `scripts/` serving, the GPU schedule and budget, checkpoints → HF | vLLM grounder serving, then the SFT pilot (M2); GRPO with the Gym `/verify` reward (M3) |
| **C: Agent & Lightning** (Patrick) | `sandbox/`, `resources_servers/web_agent/`, Lightning planner and instruction rewriting, demo | `web_agent` env with hosted Lightning against the base grounder (M4); Lightning data engine for M2 |

`common/grounding.py` is shared: stream A writes the first version, and B and C review
it.

## Decisions so far

- **Owners:** A = Charlie, B = Kyle, C = Patrick.
- **Problem:** an accessibility feature. A user describes what they want in plain
  language, and the agent navigates the app to the right page and does it.
  - It's **vision-only**, so it works on apps whose code or DOM isn't available,
    which is most real apps.
  - In our sandbox, the DOM and app state are used **only by `verify()` for
    scoring**, never by the agent.
- **Agent loop:** multi-step, built on Gym's `simple_agent` tool loop with
  `max_steps`. Tasks are short (2–4 steps, e.g. open a menu and then pick an item).
  Single-screen click tasks are the fallback demo.
- **How Lightning "sees" (confirmed):**
  - Lightning is text-only. After every action, the env returns a text observation:
    a structured screen summary (visible headings and clickable elements) written by
    the **base VLM**, plus the URL and whether the screen changed.
  - The base VLM is the same weights as the grounder with the LoRA adapter turned
    off.
  - **Separate instances count as two models** (card counts: 30 + 2 + 2 = 34 > 33),
    so never serve the base and the tuned model separately.
  - **Agreed design: one loaded 2B model, with the LoRA adapter switched per
    request.** It counts as one model (32B total).
    - vLLM does this natively: `--enable-lora --lora-modules grounder=<adapter>`.
      Requests to `base` skip the adapter; requests to `grounder` apply it.
    - There's no reload between requests, and mixed requests batch together.
    - Stream B verifies vLLM LoRA support for the chosen model at gate D1.
  - The env reads the screen-summary model name from config (`summary_model: base`).
    If the organizers count base + adapter as two models, switch it to `grounder`
    (the tuned model for both roles) and spot-check summary quality.
- **Data and training flow:** set everything up, then a small POC run end to end,
  then size the full run from the POC's measured throughput.
- **Training data mix (A), by share of targets:**

| Platform | Source | Share | Instructions |
| --- | --- | --- | --- |
| Desktop ~35% | OS-Atlas **Linux** | ~17.5% | Raw a11y labels; Lightning rewrites ~60% |
| | OS-Atlas **macOS** | ~17.5% | Raw a11y labels; Lightning rewrites ~60% |
| | *(optional)* OS-Atlas Windows / GroundCUA | — | Only if time allows |
| Mobile ~30% | OS-Atlas AndroidWorld | ~15% | Raw labels; Lightning rewrites ~60% |
| | OS-Atlas UIBert | ~15% | Natural (keep) |
| Web ~35% | OS-Atlas FineWeb (range-extracted) | ~35% | HTML text; Lightning rewrites ~60% |

  - This matches the defaults in Charlie's `sample_os_atlas.py`. RICOSCA and Widget
    Captioning are left out to keep it simple.
  - Aim for ≥ 40% icon-like targets per platform. OS-Atlas has no text/icon label;
    its `data_type` is the control type. So use a proxy: small, near-square boxes,
    or icon-like control types.
  - **FineWeb must pass a ScreenSpot-overlap check before use** (research-notes §5).
    The fallback is SeeClick web.
  - Pilot: about 2k targets in these proportions. Full run: sized from the pilot,
    probably 20–30k.
  - `dev`: 5% of **base screenshots** per source, grouped so no screenshot is in
    both train and dev:
    - by Rico ID across UIBert and Widget Captioning, which share images;
    - by page for FineWeb;
    - by base screenshot (strip `_subN`) for Windows and AndroidWorld;
    - by app for GroundCUA.
    The first dev slice must exist **before 3pm** (gate D1 uses it).
  - One target per training example for the pilot. Packing (several targets per
    screenshot) is faster, but it needs either a custom collator (`assistant_only_loss`
    isn't available on our stack) or a multi-target prompt/completion format that
    differs from eval. Only do it if the pilot shows the unpacked run is too slow.
- **Image preprocessing contract (A owns it; everyone imports it):**
  - **One function** in `common/grounding.py`, used by data prep, Gym eval rows, TRL
    and the MVP app: `preprocess_screenshot(img) -> img_resized`.
    1. Convert to RGB (composite any alpha on white).
    2. Resize **preserving aspect ratio** so `MIN_PIXELS ≤ W·H ≤ MAX_PIXELS`.
    3. Round W and H to multiples of 32 (Qwen3-VL/3.5: 16 px patch × 2×2 merge).
       This is the same math as the processor's `smart_resize`.
    4. Save PNG (or JPEG q≥95).
  - **Never** pad/letterbox, crop, stretch, or apply mean/std yourself; the
    processor normalizes pixels.
  - **Phones and desktops:** aspect-preserving resize handles both tall and wide
    screens. Don't force a fixed size.
  - **Targets:** the bbox center in **relative 0–1000** coordinates, which are
    resize-invariant. Stream A confirms that Qwen3.5 uses the same 0–1000 convention
    at baseline.
  - **The same `MAX_PIXELS` everywhere.** Use the identical cap in vLLM
    `--mm-processor-kwargs size` and the TRL processor, so nothing gets resized
    twice. Choose ~1 MP vs ~2 MP during the M1 baselines, then freeze it as a
    constant.
  - **MVP:** Playwright with a fixed viewport (e.g. 1280×800) and
    `device_scale_factor=1`. Screenshot → the same preprocess → model → 0–1000 →
    viewport CSS pixels.
- **Fine-tuning method (B):** LoRA on the language model, with the vision encoder
  frozen.
  - It matched full fine-tuning in published GUI-grounding results at a fraction of
    the cost.
  - Keep the adapter, for the base/LoRA serving above. Also export a merged copy for
    publishing.
- **Gates D1 and D2:** agreed as written below.
- **Gate D1 decided (Sat ~4:30pm): fine-tune `Qwen/Qwen3.5-2B` at 2 MP.**
  - Baselines on Charlie's OS-Atlas sample: Qwen3-VL-2B 80.6% (6% near-miss),
    Qwen3.5-2B 79.8%. That's a statistical tie, so the tie-break rule applies.
  - Both passed Kyle's D1 checks: LoRA loads in vLLM, including Qwen3.5's
    linear-attention layers, and the 20-step SFT smoke runs (Qwen3.5 1.75 s/step
    at 2 MP).
  - Qwen3.5 is covered by TRL's VLM GRPO tests and has a published single-H200 TRL
    recipe, which lowers the risk of the GRPO step the Gym bounty depends on.
  - Pixel cap frozen at **2 MP** (`MAX_PIXELS = 2_073_600`) everywhere:
    `common/grounding.py`, `serving.env`, TRL.
  - Charlie's per-split breakdown is still pending. We'd reconsider only if Qwen3-VL
    leads by more than ~5 points on icons or desktop.
- **Eval (option C):** OS-Atlas `dev` split (by screenshot) for choosing checkpoints.
  The final held-out test is **ScreenSpot-v2-clean**: 1,244 rows, all of SSv2 minus
  the 28 desktop rows captured on the same machines as our Linux/macOS training data
  (`benchmarks/screenspot_v2/excluded_rows.json`).
- **Hosting:**
  - Training and eval run on Brev (vLLM).
  - **Demo: hybrid first** (details in [local-mac.md](local-mac.md)).
    - Lightning runs on Patrick's Mac via **vllm-metal**, the official vLLM plugin
      for Apple GPUs. Gate L by ~5pm.
    - The grounder runs on the H200 (base + LoRA) through a tunnel.
    - **Fully local** (adding a merged grounder on the Mac, used for both roles)
      only if Gate G passes at ~8pm.
    - **Fallback:** everything on the H200.
  - The video says "runs locally" only for what actually ran on the Mac. Otherwise
    say "fits on one GPU (~32B total)".
- **Lightning data generation (instruction rewriting):** hosted Lightning by default.
  Batch several labels per request and check the rate limit early. Self-host only if
  the rate limit blocks it.
- **BLADE packages:** commit only small sampled rollouts that use `file://` paths
  (no base64) under `benchmarks/screenspot_v2/rollouts/`. Full rollouts go to HF.
- **Publishing:** required by the track. Patrick owns the HF account. Details are in
  [submission.md](submission.md).
- **Submission:** the form has bounty checkboxes for Lightning, NeMo Gym and vLLM;
  the first two need written explanations. See [submission.md](submission.md).
- **Brev:** the instance stays stopped while we code; start it for GPU work only.
  Use `brev login` (browser flow).

## Decision gates

- **D1 base model (by 3pm):** choose on the **dev** score. Report SSv2 baselines,
  but don't select on them. Prefer headroom on the icon splits. Both checks must
  pass for the chosen model:
  1. vLLM `--enable-lora` loads a dummy adapter built from our PEFT config.
  2. A 20-step TRL SFT smoke test runs.
  If it's a tie, take Qwen3.5-2B (best-documented single-H200 recipe).
- **D2 GRPO:** go only if SFT finished by ~9pm Saturday and the smoke test runs at
  ≤ ~35 s/step, so the run can finish by ~1am. Keep GRPO only if dev accuracy beats
  SFT. Otherwise ship SFT and say so.
- **Demo hardware:** see "Hosting" above. Hybrid if Gate L passes; fully local only
  if Gate G also passes; otherwise all on the H200 (official Lightning BF16 plus the
  grounder co-resident, all training stopped).

## GPU etiquette (one shared H200, ~23 GPU-hours total)

- Say what you're running in team chat before starting a GPU job. Use one tmux
  session per person, named after you.
- Training has priority for GPU memory; eval servers yield to it.
- Nobody leaves the box idle: whoever finishes last runs `brev stop permanent-tan-marmot`.
  The instance is stoppable and keeps its disk.
- Log each GPU run (command, config, start/end, result) in `docs/runs.md`.
