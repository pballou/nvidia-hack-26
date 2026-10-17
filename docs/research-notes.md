# Research notes

Verified facts behind our decisions, checked on 2026-10-17. Each item was
fact-checked against primary sources: model cards, the HF API, papers, and code.
Re-check anything you're about to rely on if it's marked *estimate* or *unverified*.

## 1. Event rules

Source: [event Notion](https://concrete-panther-c83.notion.site/nv-rh-small-models-hack)
and the official rules on the [Luma page](https://luma.com/nvidia-redhat-small-models-hack).

- **Parameter cap:** ≤ 33B total across all generative language models in the
  system, "including any used for verification, critique, routing, or speculative
  decoding". Embedding and reranking models are excluded. Counts use "the official
  parameter count published in the model developer's model card". MoE models count
  total parameters.
  - Lightning's card says **30B total / 3B active**, which leaves **≤ 3B for the VLM**.
  - For reference only: the HF API reports 31.58B for Lightning, and 32.91B including
    its MTP (multi-token prediction) head. The rule uses the card number.
  - Don't enable Lightning's external draft models, DSpark (~0.97B) or DFlash
    (~0.83B). Serve without speculative decoding.
- **Code:** all code must be written during the hackathon. Rule text: "Open source
  code is allowed only if it's been open-sourced for 2+ weeks."
  - **Confirmed (2026-10-17): it applies per project, not per release.**
  - So the pinned stack is fine: nemo-gym 0.7.0, vllm 0.31.0, trl 1.16.0. So is the
    vLLM #61718 Lightning patch, a file from the vLLM project, which has been public
    for years.
- **Base + LoRA counted once:** our reading is that the screen-summary role (base
  weights, adapter off) and the grounder (adapter on) are one model. That's
  **unconfirmed**; ask alongside the 2-week question.
- **Data:** must meaningfully use open data and cite sources. Synthetic data must be
  published together with its methodology.
- **Judging:** "systems engineering", meaning a working end-to-end system. Criteria:
  - technical execution and completeness;
  - meaningful use of NVIDIA and Red Hat technology;
  - value and impact;
  - innovation and performance.
- **Bounties we target:**
  - "Best Use of NVIDIA Nemotron Lightning": Lightning must be central to the value,
    the demo must be reliable, and output quality should be improved through eval or
    feedback loops.
  - "**Best Model Improvement** With NVIDIA NeMo Gym": the model has to get *better*
    through Gym, not just be evaluated in it.
- **Deadline:** the team's deadline for everything is **Sunday 2026-10-18,
  11:00am EDT**, when demo videos are due. The event itself ends at 4:00pm EDT (Luma
  `endDate`). Submit through the Airtable form linked from the Notion page. Teams
  have at most 3 members.

## 2. Grounding-model candidates (≤ 3B)

Parameter counts are HF `safetensors.total`. ScreenSpot-v2 (SSv2) scores vary a lot
with the prompt and harness, so **measure our own baselines** before choosing.

| Model | Params | SSv2 overall | Notes |
| --- | --- | --- | --- |
| `Qwen/Qwen3.5-2B` | 2.27B | 66.5 ([Liquid harness](https://www.liquid.ai/blog/lfm2-5-vl-3b)) / 83.0 ([WinDOM](https://arxiv.org/abs/2606.25964), official tool-call prompt) | Best-documented recipe. WinDOM: 1×H200, TRL, 54k Playwright DOM-labelled records; SFT 85.9, SFD+RL 88.8–89.5. Covered by TRL's VLM SFT+GRPO tests. Needs `fla` + `causal-conv1d` kernels (otherwise slow torch fallback). Its chat template puts an empty `<think></think>` in the generation prompt. |
| `Qwen/Qwen3-VL-2B-Instruct` | 2.13B | 86.7 ([MAI-UI eval](https://arxiv.org/abs/2512.22047)) | Mobile 95.5/82.0, desktop 95.4/**73.6**, web 89.7/**76.4** (text/icon); icons are the headroom. Used in Gym's committed `circle_click` rollouts. Not in TRL's GRPO VLM unit tests. |
| `Qwen/Qwen3.5-0.8B` | 0.87B | not measured on the full set (mobile subset 71.1) | SS-Pro 46.5. |
| `OpenGVLab/InternVL3_5-2B` | 2.35B | ~82.0 (Liquid) | `<box>[[x1,y1,x2,y2]]</box>` on 0–1000. |

- **Avoid `Qwen/Qwen2.5-VL-3B-Instruct`.** Its card says 3B, so it's legal only at
  exactly 33B, with zero headroom (HF counts 3.75B). Its `qwen-research` license
  also conflicts with "publish openly". Many "3B" GUI models are built on it.
- **Card counts vs HF counts:** eligibility uses card counts. Qwen3.5-2B and
  Qwen3-VL-2B are both "2B" on their cards; the HF totals in the table are for
  reference.
- **Comparators to cite:**
  - Qwen3-VL-4B base 93.1 ([KV-Ground](https://huggingface.co/vocaela/KV-Ground-4B-BaseQw3vl)).
  - MAI-UI-2B 92.5, UI-Venus-1.5-2B 92.8, UI-TARS-2B-SFT 84.7.
- **Coordinates:** Qwen3-VL emits points on a relative **0–1000 grid**, so resizing
  doesn't change the target (confirmed by the Qwen maintainers). Qwen3.5 shares the
  processor, but its convention is inferred; verify it with a baseline.
  Qwen2.5-VL emits absolute pixels.
- **Baseline trap** ([NeMo RL #2260](https://github.com/NVIDIA-NeMo/RL/issues/2260)):
  Qwen3-VL-2B scored 41.9% on a click task with the wrong output format, and 95% with
  the right one. Use the model's native format and prompt.

## 3. NeMo Gym (v0.7.0, Python ≥ 3.13.14, `gym` CLI)

Checked at code level in [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym).

**Architecture.** There are three server types. Gym itself is CPU-only and talks
HTTP to vLLM or a hosted API.

| Server | What it does |
| --- | --- |
| Agent | Runs the rollout loop. |
| Model | Proxies Responses API ↔ chat completions. |
| Resources | Owns the tools and `verify()`, which computes the reward. |

**Image inputs work end to end.** Put `{"type":"input_image","image_url":"data:...|file://...|http://..."}`
in `responses_create_params.input`; `vllm_model` converts it to chat `image_url`.
Images inside **tool outputs** are not supported. That's fine for us, since Lightning
is text-only.

**Templates to copy** (all first published more than 2 weeks ago):

| Template | First published | What it gives us |
| --- | --- | --- |
| `resources_servers/gui_coordinate` | 2026-08-05 | Image + instruction rows, `<point>(x,y)</point>` on 0–1000, distance-based reward. This becomes our click-in-bbox ScreenSpot env. |
| `resources_servers/circle_click` | 2026-03-07 | Image click via a tool, binary reward, Qwen3-VL-2B example rollouts. |
| `resources_servers/interactive_browser` | 2026-09-22 | Stateful per-session Playwright, tools, `verify()` from live state. This becomes our `web_agent` env. |
| `resources_servers/osworld` + `osworld_agent` | — | Screenshot plus pyautogui reference (heavy). |

There's no ScreenSpot env in Gym, so ours could be an upstream contribution.

**`verify()` contract:**

- It receives the original JSONL row plus `response`; `response.output` is the whole
  trajectory.
- Custom row fields (e.g. `bbox`) survive only if they're declared on your
  `VerifyRequest` subclass. Undeclared fields are silently dropped.
- Fields declared on your `VerifyResponse` are what show up in the rollouts and in
  metrics.

**`simple_agent` loop:**

- It stops when the model replies without a tool call, or when `max_steps` is hit
  (the default is unlimited).
- A `done` tool does **not** end the loop.
- Each tool is a FastAPI route whose name equals the tool name. Return errors as 200
  with an `error` field.

**Session state:**

- State is keyed by `request.session["session_id"]`.
- Stateful servers need `num_workers: 1` and must release resources inside `verify()`.

**Eval tooling:**

- `gym eval run` writes rollouts, materialized inputs, `_aggregate_metrics.json` and
  failures.
- `gym eval profile` gives per-task pass rates (needs `--num-repeats` > 1).
- `gym eval compare` gives baseline-vs-candidate per-task flips.
- `gym eval reverify` re-scores stored rollouts. It needs `REVERIFY_MODE=STATELESS`.
- Use `compute_subset_metrics` for per-platform and per-text/icon breakdowns.

**Gotchas:**

- **One exception in a tool or `verify()` aborts the whole `gym eval run`.** Catch
  everything; return `reward=0, mask_sample=True`.
- Base64 screenshots get copied into the materialized-input and rollout files. Prefer
  `file://` (start vLLM with `--allowed-local-media-path`) or http URLs when Gym and
  vLLM share a box.
- `vllm_model`'s `uses_reasoning_parser`:
  - The default `true` is harmless for a VLM served without a reasoning parser.
  - Setting it to `false` while vLLM *has* a reasoning parser crashes.
  - Lightning needs `true`.
- Use `vllm_model` (or `inference_provider`) for build.nvidia.com and OpenRouter.
  `openai_model` requires a Responses-API upstream.
- Pin one stack: the PyPI 0.7.0 wheel and its bundled configs. Gym `main` has moved
  past it; for example, `drop_unrepresentable_request_fields` exists only on `main`.

**Failure analysis with BLADE** ([tutorial](https://docs.nvidia.com/nemo/gym/tutorials/evaluation-tutorials/blade-analysis-skill)):

- BLADE is an agent skill, vendored in `.agents/skills/nemo-gym-blade-analysis`.
- Point it at the artifacts from a `gym eval run` (rollouts JSONL,
  `_aggregate_metrics.json`, reward profile, config).
- It buckets tasks into always-pass, sometimes-pass and never-pass.
- It labels root causes: KG knowledge gap, UK unreliable knowledge, BI behavioral,
  TI task/verifier, IR infra, DA data artifact.
- It maps each cause to an intervention (e.g. UK → RL on sometimes-pass tasks;
  DA/TI → fix the data or verifier before comparing models).
- It needs at least two comparable scored runs; base vs. tuned works.
- A BLADE-ready package lives at `benchmarks/<name>/{skill,rollouts,golden_reports}`
  and is validated with `blade_toolkit.py validate`.

## 4. Training framework constraints

| Path | Works for ≤3B Qwen VLMs? | Why |
| --- | --- | --- |
| NeMo RL + Gym GRPO | **No** | `nemo_rl/environments/nemo_gym.py` asserts a Nemotron VL/Omni placeholder processor (NemotronNanoVLV2 and NemotronH Omni). Unmerged PR #2092 ran Qwen3-VL-2B on 4×H100; it's stale. |
| VeRL + Gym | **No** | Text-only (`multi_modal_inputs={}`). |
| **TRL `GRPOTrainer` + reward fn → Gym `/verify`** | **Yes (chosen)** | The same pattern as NVIDIA/Unsloth's [official Gym notebook](https://github.com/unslothai/notebooks/blob/main/nb/NeMo-Gym-Sudoku.ipynb): start the resources server alone and POST each completion to `/verify`. |
| TRL `SFTTrainer` | Yes | Tests cover Qwen3-VL and Qwen3.5. |
| NeMo Automodel VLM SFT | Mostly ≥ 4B recipes | — |

TRL notes:

- TRL's VLM + vLLM GRPO test is skipped upstream (no CI), so **run a 20-step smoke test
  before relying on it**. The fallback is `use_vllm=False`, which is fine for outputs
  of ≤ 48 tokens.
- `gradient_checkpointing` defaults to True.
- Set `vllm_max_model_length`; the default allocates the full 262k context.

**Verified stack** ([gpu-box.md](gpu-box.md)): vLLM v0.31.0 image, torch 2.13.0+cu130,
transformers 5.17.0 (vllm caps it below 5.18), TRL 1.16.0.
- On one GPU, TRL GRPO must use `vllm_mode="colocate"`. Server mode needs NCCL across
  two GPUs.
- `AsyncGRPOTrainer` refuses Qwen3.5.
- VLM SFT can't use `packing`, `padding_free` or `assistant_only_loss` on this stack.

**Gym-native improvement loop** (the "model improvement with Gym" story):

1. Run `gym eval run --num-repeats 8 --temperature 1.0` on training rows (never on
   ScreenSpot-v2).
2. Run `gym eval profile`.
3. Use the reward=1 samples as rejection-sampled SFT data, and the tasks with
   0 < pass rate < 1 as the GRPO set.
4. Run `gym eval compare`, then BLADE.

## 5. Data

**ScreenSpot-v2** ([HF](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2), Apache-2.0):

- 1,272 samples on 730 screenshots: mobile 501, desktop 334, web 437; text 718, icon 554.
- `bbox` is **[x, y, w, h] in pixels**. Convert to xyxy; a hit is inclusive.
- Files are `screenspot_{mobile,desktop,web}_v2.json` and `screenspotv2_image.zip` (1.33 GB).
- **Eval only. Never train on it.** ScreenSpot v1 shares 586 of SSv2's 730 images,
  so training on v1 contaminates v2, and v1 isn't an independent test set.
- **Pin the dataset revision to `5efbb1f1b5`.** The desktop JSON was re-uploaded in
  Dec 2024 and Jan 2025.
- **SSv2 is not one clean set.** SSv2 replaced 144 v1 questions with 144 new
  screenshots: 52 mobile (iOS), 28 desktop, 64 web. New files have 6-character stems
  like `pc_8g2jf6.png`. **Report three numbers:** all 1,272 rows; the 1,128 rows on
  v1 images; the 144 new rows.
- ScreenSpot-Pro (1,581 samples, xyxy pixels, MIT) is an optional harder benchmark.

**OS-Atlas-data** ([HF](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data), Apache-2.0 card; 816 GB): our training pool.

- Rows are `{img_filename, instruction, bbox}`. `bbox` is `[left, top, right, bottom]`
  normalized to [0, 1]. Desktop and AndroidWorld rows also have `data_type`.
- **⚠ Contamination: do not use OS-Atlas Linux or macOS desktop data**
  (`linux_splited.json`, `macos_splited.json`).
  - SSv2's 28 new desktop screenshots (20 Ubuntu at 2560×1440, 8 macOS at 2880×1800)
    were captured on the same machines and desktop setups, 1–3 weeks after the
    OS-Atlas captures. Same wallpaper, same desktop files.
  - At least 9 of those 28 rows have a near-duplicate training screenshot.
  - The test targets themselves are labeled in training. For example, the test row
    "adjust font color" matches the training element "Font Color" at IoU 0.83.
  - The same app layouts recur across many training screenshots, so a perceptual-hash
    dedup **cannot** remove the leak. Exclude both files entirely (about 2.5k images).
  - Neither OS-Atlas nor any later paper documents this; it's our own finding,
    verified by two independent checks.
- **Other sources look low-risk** (sampled, not exhaustive):
  - AndroidWorld, RICO, Widget Captioning, UIBert and SeeClick web.
  - Windows desktop (different resolution, Aug 2024 captures).
  - FineWeb and AMEX were not checked. Still run a pHash dedup against SSv2 for
    every source.
- **Never use:** Wave-UI rows with `source == "screenspot"` (they're in every split)
  or GroundUI-18K (it contains all of ScreenSpot).
- Usable subsets:

| Subset | Screenshots | Targets | How to get images | Instruction style |
| --- | --- | --- | --- | --- |
| ~~Linux / macOS desktop~~ | — | — | **Excluded (contaminated)** | — |
| Windows desktop | 50k unique | many | Range-extract 3–4k images from the split archive (71 GB) | Raw a11y labels |
| AndroidWorld | 15.9k | 90k | `mobile_images.zip` (2.3 GB) | Raw labels |
| UIBert | 5.7k | 17k | `UIBert.zip` (0.5 GB) | Natural |
| Widget Captioning + RICOSCA | 32k | 275k | Range-extract from `rico_imgs.zip` (6.5 GB) | Natural |
| FineWeb web | 1.6M+ | millions | Range-extract 3–5k images from `fineweb_3m_s11.zip` | HTML text |

- **Web and Windows:** these archives are 46–71 GB. HF supports HTTP range requests
  (verified: 206), so you can pull individual members without downloading the whole
  archive.
- **Quality filters** (from our stats and the recipe review). Drop:
  - degenerate boxes, boxes outside [0, 1], or boxes covering more than 15% of the
    screen (0.3–0.7% of boxes are invalid);
  - icon-font glyphs, numeric-only text, `AutomationID:` / URL / "running window"
    junk;
  - instructions over 12 words;
  - any instruction that appears twice on the same screenshot (14% of Widget
    Captioning rows).
- **Team's choice:** 40% desktop / 30% mobile / 30% web, RICOSCA dropped. See the
  docs/plan.md data-mix table.
- **`data_type` isn't text/icon:** where OS-Atlas has `data_type`, it's the control
  type (e.g. `android.widget.Button`), not ScreenSpot's text/icon label. Use a proxy
  for icon-ness.
- **⚠ FineWeb wasn't checked for SSv2 overlap,** and it supplies all of our web
  share. Before training on it, stream A runs the same check used for Linux/macOS
  on the sampled FineWeb images, against the 437 SSv2 web images:
  - pHash distance;
  - instruction text plus bbox IoU against SSv2 targets.
  Pay particular attention to the 64 new SSv2 web screenshots of popular sites
  (GitHub, Steam, PyTorch, YouTube, Amazon…). Drop anything that matches. If FineWeb
  looks risky, switch the web share to SeeClick web, which was checked as low-risk.
- **Recipe from the review:**
  - About 20–30k element queries, packed 3–6 per screenshot.
  - Mix ~40–45% desktop, ~25% mobile, ~30% web, with 40–50% icon-like targets.
  - Rewrite 50–70% of the raw a11y/HTML names into 3–6 word functional instructions
    in ScreenSpot style; keep the rest raw.
  - Split train/dev by base screenshot (strip `_subN`), by Rico ID across
    UIBert/WC/RICOSCA, and by page for web.
  - Use the same `max_pixels` (~1–2.1 MP) in training and eval.
- **Desktop replacement for Linux/macOS:** OS-Atlas Windows, plus
  [GroundCUA](https://huggingface.co/datasets/ServiceNow/GroundCUA) (MIT, desktop, ~1.15M
  LLM-written instructions; download per-app folders with `allow_patterns`).

**Other sources:**

- `ritzzai/GUI-R1`: `train.parquet` (482 MB) is GUI-R1-3K, a difficulty-filtered RL
  pool built from OS-Atlas-type data; UI-R1-E-3B reached SSv2 89.5 on 2k of its rows.
  It could seed GRPO, but **check its sources for Linux/macOS OS-Atlas images first**.
- MolmoPoint-GUISyn and RICO-WidgetCaptioning are license-clean.
- Avoid UGround (CC-BY-NC-SA, gated).

**Training-recipe evidence:**

- GuirlVG (RL): LoRA matched full fine-tuning (87.4 vs. 87.5) at 25× lower cost per
  iteration, and group size 6 beat 8.
- ZonUI: LoRA r=8 on 16k samples ≈ the full 120k.
- What matters most for RL is a point-in-box (or dense) reward plus difficulty
  filtering.

## 6. Nemotron 3.5 Lightning

- **Model:** text-only, hybrid Mamba-2 + MoE, OpenMDW-1.1 license.
  [HF card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16).
- **Hosted:** model `nvidia/nemotron-3.5-lightning-30b-a3b` at
  `https://integrate.api.nvidia.com/v1`. Use it during development so the GPU stays
  free.
- **Self-hosted on the H200:**
  - Use the **NVFP4** checkpoint (~19.7 GB weights). It runs as W4A16 on Hopper.
    BF16 is ~66 GB.
  - KV cache is tiny (~0.2 GB per 64K-token sequence in FP8), so it co-resides easily
    with the VLM.
- **Known Hopper bug:** vLLM [#61603](https://github.com/vllm-project/vllm/issues/61603)
  gives NaN or empty output on H200. The fix (PR #61718) isn't released yet;
  gpu-box.md §1 backports it.
- **vLLM flags:** `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3`
  (plus `--kv-cache-dtype fp8 --mamba-cache-mode align --enable-prefix-caching`; `--trust-remote-code` isn't needed). Full command: gpu-box.md §2.
- **Thinking:** toggle per request with `chat_template_kwargs: {enable_thinking: true|false}`.
  The hosted API also accepts `reasoning_budget`.
- **Sampling:** temperature 1.0, top_p 0.95.
- **Bounty ideas** (ranked):
  1. Lightning as planner and tool-caller in the browser agent, measured in a Gym env
     with base vs. tuned grounder.
  2. Lightning as the **data engine**: rewrite raw a11y labels into natural
     instructions, then filter out ambiguous ones. Ablate template vs. Lightning
     instructions on SSv2. Publish the data.
  3. Reasoning control as a measured feature: thinking off for routine steps, a
     budget on retries; plot success vs. latency.

**Demo hosting (being evaluated):**
- Patrick wants to run everything on an M4 Max (128 GB) for the demo. Local
  runtimes (vLLM on Apple Silicon, MLX, llama.cpp) are being checked for Lightning
  tool-calling support, Qwen3.5/Qwen3-VL vision support, and matching image
  preprocessing.
- The fallback is vLLM on the Brev H200, with the app and browser on the laptop
  through an SSH tunnel.

## 7. Compute estimates (*estimates*: confirm with a 20-step pilot)

- **Budget:** $160 at **$6.90/hr** for the running H200 (console price; the list price
  is $6.48) ≈ **23 GPU-hours** total.
- **Image tokens:** Qwen3-VL is 32 px per token.
  - 1280×800 ≈ 1,000 tokens; 1920×1080 ≈ 2,040.
  - Capping at ~1 MP gives ~940 tokens.
  - The full SSv2 set is ~3.4M tokens at default resolution, or ~1.15M capped.
- **Throughput:**
  - SSv2 eval: ~2–5 min per model.
  - SFT: ~25–50 min per 10k **packed** samples (several targets per screenshot).
    Our pilot is unpacked (one target per example, so each target re-encodes the
    image), so measure it rather than using this estimate.
  - GRPO: ~20–35 s/step on 1×H200, ≈ 1–2 h for 200 steps.

**Planned H200-hours:**

| Phase | H200-h |
| --- | --- |
| Session overhead | 1 |
| Base evals | 1 |
| Lightning data generation (hosted by default, so ~0) | 0–1 |
| SFT | 4 |
| GRPO (gated) | 6 |
| Re-evals | 1 |
| Agent eval | 1.5 |
| Demo | 3 |
| Buffer | 4.5 |
| **Total** | **≈ 23** |
