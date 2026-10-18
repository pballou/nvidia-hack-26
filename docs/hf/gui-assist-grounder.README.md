---
license: apache-2.0  # same as the base model (Qwen/Qwen3.5-2B); training-data attributions in "Licenses"
base_model: Qwen/Qwen3.5-2B
library_name: transformers  # merged bf16 weights at the repo root; the PEFT adapter is in adapter/
pipeline_tag: image-text-to-text
language:
- en
datasets:
- patrickballou/gui-assist-data
- OS-Copilot/OS-Atlas-data
tags:
- gui-grounding
- gui-agent
- computer-use
- accessibility
- lora
- peft
- trl
- grpo
- nemo-gym
- screenspot
- qwen3.5
---

# GUI assist grounder (Qwen3.5-2B + LoRA)

A 2B vision-language model fine-tuned for **GUI click grounding**. It takes a screenshot and a
plain-language target ("the gear icon at the top right", "open more options") and answers with
one click point: `{"point_2d": [x, y]}` on a 0–1000 grid.

It is the grounding tool of **GUI assist**, a vision-only accessibility agent built at the
NVIDIA x Red Hat Small Models Hack (October 2026). NVIDIA Nemotron 3.5 Lightning plans each step
from a text description of the screen and calls this model to turn the target description into
pixels. The agent never reads the app's DOM or code. Code, envs and eval logs:
<https://github.com/pballou/nvidia-hack-26>.

The model is a LoRA fine-tune of `Qwen/Qwen3.5-2B`'s language model; the vision encoder is
frozen. This repo ships it two ways: merged into full bf16 weights at the repo root, and as the
PEFT adapter in `adapter/`. Training was LoRA SFT on an OS-Atlas slice with
Nemotron-Lightning-rewritten instructions (`sft-lightning`). An overnight loop then ran GRPO
rounds (up to 3), with a NeMo Gym environment's `/verify` endpoint as the reward. A round was kept
only if it improved dev by at least 0.5 points. Round 1 was kept (dev 86.27 → 87.00). Round 2 was
flat (86.89), so it was reverted and the loop stopped. **The released model is `sft-lightning` + GRPO round 1.**

## At a glance

| | |
| --- | --- |
| Base model | [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B), revision `15852e8c16`, Apache-2.0 |
| Parameters | 2B (model-card count; HF safetensors total 2.27B). The LoRA adapter adds about 16.8M (16,819,200, computed for r = 16 from the base weight shapes; PEFT loads an adapter of this config as ~16.8M trainable parameters in 372 tensors) |
| Released checkpoint | `sft-lightning` (round 0, picked on dev: 86.27 vs `sft-template`'s 85.86 in the same harness) + **GRPO round 1** (TRL, rewarded by the NeMo Gym env's `/verify` point-in-box check; 170 steps, 1,282-row pool; kept on dev: 86.27 → 87.00). Round 2 was reverted (86.89). Loop launch `20261017-193859`, adapter `night/20261017-193859/round_1/grpo/adapter` (`FINAL.json`: best round 1) |
| Formats | Merged bf16 weights at the repo root (Transformers, vLLM, vllm-metal); PEFT LoRA adapter in `adapter/` (r = 16, alpha = 32, 186 modules) |
| Input | One screenshot (aspect ratio up to 200:1, resized to ≤ 2,073,600 px) + one instruction |
| Output | `{"point_2d": [x, y]}`, integers on a 0–1000 grid relative to the image |
| Held-out test | ScreenSpot-v2-clean: 1,244 of ScreenSpot-v2's 1,272 rows. **28 rows excluded**, see [below](#the-28-row-screenspot-v2-exclusion) |
| Headline (base → released) | SSv2-clean **86.25% → 90.35%** (+4.10; never used for selection); icons **78.1% → 85.4%**; dev 76.4% → 87.0%. Agent tasks verified (12 × 3): base 18/36, released 17/36, `sft-template` 22/36, all within noise |

## Files in this repo

| Path | What |
| --- | --- |
| `/` (root) | **The released model, merged:** the LoRA folded into the base weights as one bf16 `Qwen3_5ForConditionalGeneration` checkpoint in Transformers format (as in the base, the 36 GDN `A_log` / norm tensors stay float32; weights sha256 `cd120bc0cd6f642f85537c503922fa9c3c6a9348defd643136da81ed784cef2f`). For Transformers, for vLLM without `--enable-lora`, and for vllm-metal. Made with [`scripts/gate_g_merge.py`](https://github.com/pballou/nvidia-hack-26/blob/main/scripts/gate_g_merge.py) (peft 0.21.2, transformers 5.17.0, on CPU): each of the 186 targeted weights becomes bf16(W + 2·B·A), merged in float32 by peft (identical to a manual merge, max difference 0.0). Every other tensor and file (vision tower, embeddings, tokenizer, chat template, processor configs) is the base at revision `15852e8c16`, bit for bit. It passed Gate G (see [Running on a Mac](#running-on-a-mac-vllm-metal)) |
| `adapter/` | **The same model as a PEFT LoRA adapter**, with its original PEFT keys (`model.language_model.layers.*`): `adapter_config.json` and `adapter_model.safetensors` (sha256 `54ce9389bec28af4278fadb527824e01d8d75132c7cb35fe8a9935e655bf41e8`). r = 16, alpha = 32, 186 modules. For vLLM `--lora-modules` (base and tuned from one process, below) or `PeftModel`. It is the adapter the H200 served as `grounder` in the released model's Gym evals |
| `training/` | The release round's record from the overnight loop, five files. Round 1: `grpo_command.txt` (the exact GRPO command), `knobs_used.json`, `pool_stats.json` (profile and pool by source) and `round_1_summary.json` (pool, steps, rewards, gate D2, `/verify` calls, dev before and after). `round_0_summary.json`: the starting `sft-lightning` adapter's dev eval. Paths inside are the training box's (`/home/ubuntu/...`) |

Eval metrics, the paired comparisons and every run's command and config are in
[docs/results.md](https://github.com/pballou/nvidia-hack-26/blob/main/docs/results.md) and
[docs/runs.md](https://github.com/pballou/nvidia-hack-26/blob/main/docs/runs.md) on GitHub.

Transformers 5.17 loads the root with `AutoModelForImageTextToText`. PEFT 0.21 loads the adapter
with `PeftModel.from_pretrained(base, "patrickballou/gui-assist-grounder", subfolder="adapter")`.
Every accuracy and agent number on this card was measured with vLLM. The H200 runs applied the
adapter at runtime (`--lora-modules`). The merged root was checked against that adapter only
through Gate G (200 dev rows on vllm-metal: 87.0% vs 87.0%, 99.0% agreement).

## Results

All numbers measure **accuracy**: the predicted point lands inside the target's pixel box, with
edges counting as inside. Every run goes through the NeMo Gym `gui_grounding` env
(`gym eval run`), at temperature 0 with max 128 output tokens. Images are capped at 2 MP for
both models. Checkpoints are chosen on **dev only**. SSv2-clean is only reported and was never
used for selection: both round-0 candidates were scored on it for the table (it favored
`sft-template`, 89.07 vs 88.10, and the pick followed dev anyway), and the released checkpoint
was evaluated on it after the loop had picked it.

| Model | Dev (969) | SSv2-clean (1,244) | Mobile | Desktop | Web | Text | Icon |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-2B base | 76.4 | 86.25 | 91.4 | 84.0 | 81.9 | 92.5 | 78.1 |
| SFT, raw labels (`sft-template`, round-0 candidate) | 85.66 | 89.07 | 92.22 | 89.54 | 85.13 | 92.33 | 84.81 |
| SFT, Lightning instructions (`sft-lightning`, round 0, picked on dev) | 86.27 | 88.10 | 91.22 | 86.93 | 85.35 | 93.04 | 81.67 |
| **Released: `sft-lightning` + GRPO round 1** (kept on dev; round 2 reverted) | **87.00** | **90.35** | **93.21** | **88.89** | **88.10** | **94.18** | **85.37** |

- **Dev numbers:** `sft-lightning`'s 86.27 is from the loop harness, where `sft-template` scored
  85.86 (85.66 in its first launch). The released 87.00 is a selection number, the best on dev
  across rounds. Re-runs gave 87.00 and 86.89.
- **SSv2-clean, two runs of the released adapter:** 90.35 is Kyle's one-time eval
  (`final-round1-grpo`), the canonical number, which the paired comparisons below use. It rounds
  to mobile 93.2 / desktop 88.9 / web 88.1 and text 94.2 / icon 85.4. A second, independent T=0
  run (`night-r1`) gave **90.27**, one row apart. Neither was used for any selection.
- **Paired comparisons on SSv2-clean** (same 1,244 rows, McNemar exact on fixed vs broken rows):
  - vs `sft-lightning`, the loop's starting adapter: **+2.25**, 34 fixed / 6 broken,
    p ≈ 8 × 10⁻⁶;
  - vs `sft-template`: +1.29, 50 fixed / 34 broken, p = 0.10, **not significant**;
  - vs base (86.25; 86.3 rounded): **+4.10**, 89 fixed / 38 broken, p ≈ 7 × 10⁻⁶.
- **Released, by platform and type** (text / icon): mobile 96.90 / 88.15, desktop 91.67 / 84.92,
  web 92.74 / 82.76. Base: mobile 97.6 / 82.9, desktop 90.6 / 74.6, web 87.6 / 75.4.

Dev by source (OS-Atlas, held out by screenshot):

| Model | Linux | macOS | AndroidWorld | UIBert | FineWeb |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-2B base | 68.8 | 73.4 | 70.5 | 78.4 | 82.9 |
| `sft-template` | 86.9 | 91.1 | 81.3 | 85.1 | 84.5 |
| `sft-lightning` | 87.5 | 91.1 (90.51) | 77.7 (78.42) | 88.8 | 85.9 |
| **Released** (round 1) | **89.77** | **92.41** | **84.89** | **86.57** | **84.25** |

`sft-lightning` has two dev runs, both 86.27 overall: its own eval, and the loop's round-0
re-eval (in parentheses where they differ). Round 1's deltas are against the re-eval.

- **Where SFT helped (`sft-template`):** the gain is mostly desktop (Linux +18.1, macOS +17.7 on dev; SSv2-clean
  desktop +5.5) and icons (SSv2-clean +6.7). Text grounding was already strong and stayed flat
  (92.5 → 92.3).
- **Where GRPO round 1 helped** (against its start, `sft-lightning`):
  - Every SSv2-clean split rose, most on icons (+3.70) and web (+2.75). The per-split moves
    aren't tested on their own.
  - Much of the icon and desktop gain recovers `sft-lightning`'s deficit there: against
    `sft-template`, icon is only +0.56 and desktop is −0.65.
  - On dev, AndroidWorld rose most (78.4 → 84.9, 10 fixed and 1 broken), while FineWeb (−1.7)
    and UIBert (−2.2) fell. Round 1's dev gain alone is suggestive, not significant: 22 fixed,
    15 broken, p = 0.32.
  - One kept round, and no equal-compute non-GRPO control (e.g. SFT on the same 1,282-row
    pool). So the +2.25 shows the loop's round improved its starting adapter, not that GRPO in
    particular was needed for it.
- **Dev noise:** re-evaluating the same adapter in a new launch gave 85.66 vs 85.86. That's
  about 0.2 points from vLLM batch nondeterminism at temperature 0. The GRPO keep gate is
  +0.5 points.
- **GRPO rehearsal (not the released model):**
  - one round from `sft-template` on a 111-row pool, 30 steps;
  - dev went from 85.86 to 86.69 (+0.83) in the same harness.
- **Agent task success** (NeMo Gym `web_agent` env, 12 sandbox tasks × 3 repeats, official
  BF16 Lightning planner, thinking off; the same rows, prompts and tools, and only the grounder
  changes: served name `base`, or the adapter behind `grounder`). `base` and `sft-template` ran
  Sat 18:45–19:25; the released adapter ran Sat 22:04–22:29, on the serving process the loop had
  restarted at 22:00.

  | | `base` | `sft-template` | **Released** |
  | --- | --- | --- | --- |
  | Verified success | 18/36 (50%) | 22/36 (61%) | 17/36 (47%) |
  | Claim matches the sandbox | 26/36 | 31/36 | 24/36 |
  | False success claims | 8 | 4 | 11 |
  | Failed actions | 10 | 0 | 0 |
  | No-change actions | 83 | 45 | — (not tallied) |

  - **All three are within noise on agent success.** Released vs base: mean reward −0.028,
    95% CI [−0.20, +0.14]. `sft-template` vs base: +0.111, CI [−0.06, +0.28]. With a strict
    `mark-all-read` check, `sft-template` is 20/36.
  - **What the tuned grounders do change:** failed actions 10 → 0 with either one. The CI
    excludes 0, but the 10 sit in 4 base rollouts of 2 tasks. `text-size-large` (a small "L"
    next to the selected "M") went 0/3 → 3/3 with the released adapter (2/3 with
    `sft-template`).
  - **The 22 → 17 drop isn't evidence of a grounder regression.** A BLADE review traces about 6
    of the 9 rollouts that changed between `sft-template` and the released adapter to the
    planner or the screen summarizer. The released adapter's hand-labelled hit rate on
    well-posed targets is 96.5% (110/114), vs 92.6% (112/121) for `sft-template`, which isn't
    significant either. At n = 36, end-to-end success can't separate the grounders.
  - **Per task vs base** (verified/3):
    - released: `text-size-large` 0→3 and `max-usage-month` 0→1; `high-contrast` 2→0,
      `mark-all-read` 3→1 and `message-support-meter` 1→0, all three traced to the planner or
      the summarizer;
    - `sft-template`: `text-size-large` 0→2, `sms-billing-off` 0→2 and `message-support-meter`
      1→2; `high-contrast` 2→1.
- **Not comparable with published ScreenSpot-v2 numbers:**
  - SSv2-clean drops 28 desktop rows;
  - we use our own prompt and harness (see "Prompt" below).

Sources: [docs/results.md](https://github.com/pballou/nvidia-hack-26/blob/main/docs/results.md)
and [docs/runs.md](https://github.com/pballou/nvidia-hack-26/blob/main/docs/runs.md), which has
the command, config and output path of every run. The released adapter's SSv2-clean eval and
paired compares are runs.md's "Final adapter, one-time SSv2-clean eval" row (Kyle, Sat 22:04),
and the duplicate run is the Sat 22:08 row. The agent table is in results.md ("Agent") and in
runs.md (the Sat 18:45 row, and Patrick's Sat 22:04 row). Failure analyses are in
[docs/blade/](https://github.com/pballou/nvidia-hack-26/tree/main/docs/blade):
`round_1_r0_vs_r1.md` for the GRPO round, `agent_final_vs_sft.md` for the agent arms.

## The 28-row ScreenSpot-v2 exclusion

We train on OS-Atlas Linux and macOS desktop data. ScreenSpot-v2 added 28 new desktop
screenshots (20 Ubuntu, 8 macOS) that were captured **on the same machines and desktop
setups**, 1–3 weeks after the OS-Atlas captures: same wallpaper, same desktop files.

- At least 9 of the 28 have a near-duplicate training screenshot.
- Some test targets are labeled in training. For example, the test row "adjust font color"
  matches the training element "Font Color" at IoU 0.83.
- A dedup on the training side can't remove this, because the same app layouts recur across
  many training screenshots.

So every ScreenSpot-v2 number on this card excludes those 28 rows: **SSv2-clean** = revision
`5efbb1f1b5`, 1,272 − 28 = 1,244 rows. The exclusion is the only filtering of ScreenSpot-v2. The
list (file name and instruction, with the reason) is
[`benchmarks/screenspot_v2/excluded_rows.json`](https://github.com/pballou/nvidia-hack-26/blob/main/benchmarks/screenspot_v2/excluded_rows.json),
also shipped in the dataset repo.

The desktop gain holds on SSv2-clean, so it isn't explained by same-machine leakage.
ScreenSpot v1/v2 were never trained on. Training screenshots were also deduplicated against
ScreenSpot-v2 by perceptual hash.

## How to use

### Prompt and output format (verbatim)

This is the exact prompt used in training, evals and the agent, from
[`common/grounding.py`](https://github.com/pballou/nvidia-hack-26/blob/main/common/grounding.py)
(family `qwen3_5`). It is a single user turn: the image first, then this text.

```text
This is a screenshot. Where should I click to accomplish this instruction: "{instruction}"?
Output the click point in JSON format as {"point_2d": [x, y]}.
```

- **Chat template:** use Qwen3.5's template with thinking off
  (`chat_template_kwargs={"enable_thinking": false}`). Training asserts that its rendered prompt
  equals vLLM's generation prompt.
- **Output:** one JSON object, `{"point_2d": [x, y]}`.
  - The training target was the box center, rounded and clamped to integers in [0, 1000].
  - The tuned model's answers are short. The pilot SFT cut the average answer from 37.6 to 16.7
    output tokens.
  - `grounding.parse_point` also accepts `bbox_2d` (it returns the center),
    `<point>(x, y)</point>` and a bare `x, y`. It returns None for anything outside [0, 1000].

### Coordinate convention

- `x` runs left to right and `y` top to bottom, both on a **relative 0–1000 grid** over the
  image the model saw.
- The grid is relative, so resizing doesn't move the target. Convert with the **original**
  screenshot's size: `x_px = x / 1000 * W`, `y_px = y / 1000 * H`
  (`grounding.to_pixels`).
- For a browser viewport, use CSS pixels with `device_scale_factor=1`.

**Image preprocessing** (`grounding.preprocess_screenshot`, identical in training, Gym evals and
serving):

1. Convert to RGB, compositing any alpha onto white.
2. Resize, keeping the aspect ratio, so that 65,536 ≤ W·H ≤ **2,073,600** (`MAX_PIXELS`), with W
   and H multiples of 32 (bicubic). This is the same math as the Qwen processor's
   `smart_resize`.
3. No padding, cropping or manual normalization.

At 2 MP an image is about 2,025 visual tokens. Serve with the same pixel cap (below), so nothing
is resized twice.

### Serve with vLLM: base and grounder in one process

This is adapted from the command we serve with on the H200
([docs/gpu-box.md](https://github.com/pballou/nvidia-hack-26/blob/main/docs/gpu-box.md) §3,
`scripts/serve_vlm.sh`). Only the volume mounts, the adapter path and two box-only settings
(`HF_HUB_OFFLINE=1`, and `--allowed-local-media-path` for `file://` images) differ. It uses the pinned
image vLLM v0.31.0. One process serves two names:

- `base`: Qwen3.5-2B with the adapter off;
- `grounder`: this LoRA, applied per request.

Mixed requests batch together, and there's no reload between them.

```bash
hf download patrickballou/gui-assist-grounder --include "adapter/*" --local-dir ./gui-assist-grounder

docker run -d --name vllm-vlm --gpus all --ipc=host -p 127.0.0.1:8001:8001 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$PWD/gui-assist-grounder:/adapters/gui-assist-grounder:ro" \
  vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3 \
  --model Qwen/Qwen3.5-2B --served-model-name base Qwen/Qwen3.5-2B \
  --port 8001 --gpu-memory-utilization 0.30 \
  --max-model-len 16384 --max-num-seqs 128 --max-num-batched-tokens 16384 \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --mm-processor-kwargs '{"size": {"longest_edge": 2073600, "shortest_edge": 65536}}' \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-lora --max-lora-rank 64 --lora-modules grounder=/adapters/gui-assist-grounder/adapter
curl -s localhost:8001/v1/models   # expect "base" and "grounder"
```

- **Pixel cap:** in `--mm-processor-kwargs`, `size.longest_edge` / `shortest_edge` are the
  max and min **total pixels**. Keep them equal to `MAX_PIXELS` / `MIN_PIXELS` above.
- **No reasoning parser:** don't pass `--reasoning-parser` for Qwen3.5. vLLM would return the
  answer as `reasoning` with `content: null` (vLLM #53284).
- **Pinning the base:** our evals served `Qwen/Qwen3.5-2B` at revision `15852e8c16`. Add
  `--revision 15852e8c16360a2fea060d615a32b45270f8a8fc` to pin it.
- **Without LoRA:** serve the merged weights at the repo root
  (`--model patrickballou/gui-assist-grounder --served-model-name grounder`, the other flags as
  above minus the three LoRA ones). That process serves only the tuned model, with no
  adapter-off `base`.
- **On a Mac:** see [Running on a Mac](#running-on-a-mac-vllm-metal) below.

### Query it

```python
# needs: openai, pillow, and common/grounding.py from github.com/pballou/nvidia-hack-26
from openai import OpenAI
from PIL import Image
from common import grounding

shot = Image.open("screenshot.png")
img = grounding.preprocess_screenshot(shot)   # RGB, aspect kept, <= 2,073,600 px, sides multiple of 32
messages = grounding.build_grounding_messages(grounding.to_data_url(img), "open the settings", family="qwen3_5")

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="EMPTY")
reply = client.chat.completions.create(model="grounder", messages=messages, temperature=0, max_tokens=128)
pt = grounding.parse_point(reply.choices[0].message.content)   # (x, y) on 0-1000, or None
if pt is not None:
    x_px, y_px = grounding.to_pixels(pt, *shot.size)            # pixels of the ORIGINAL screenshot
```

Send `model="base"` to the same server to get the untuned model. The agent uses `base` for its
screen descriptions.

### Running on a Mac (vllm-metal)

The model also runs on Apple silicon under vllm-metal 0.31.0 (which tracks vLLM 0.31.0, like the
H200). It passed our **Gate G** check in both forms. On the same 200 dev rows, the grounder on
vllm-metal was compared with the H200's runtime-LoRA `grounder` serving the same adapter. To pass,
`prompt_tokens` must match on every row, hit rates within 2 points, and hit/miss agreement
≥ 90%:

| Form on the Mac | `prompt_tokens` identical | Hit rate, Mac vs H200 | Hit/miss agreement |
| --- | --- | --- | --- |
| Merged checkpoint (repo root), serves `grounder` only | 200/200 | 87.0% vs 87.0% | 99.0% |
| Base + LoRA in **one** vllm-metal instance (`base` and `grounder`, adapter per request) | 200/200 | 87.0% vs 87.0% | 99.0% (local p50 1.86 s) |

- **vllm-metal needs the adapter's keys renamed.** vllm-metal 0.31 wraps the language model's
  layers under mlx-vlm's names (`model.layers.N.…`), while PEFT names them
  `model.language_model.layers.N.…`. The adapter as published fails at startup with "matched 0
  wrapped modules".
  - `adapter/` keeps the original PEFT keys, which vLLM on CUDA and `PeftModel` expect.
  - [`scripts/gate_g_lora_keys.py`](https://github.com/pballou/nvidia-hack-26/blob/main/scripts/gate_g_lora_keys.py)
    writes a re-keyed copy for vllm-metal (`model.language_model.layers.*` → `model.layers.*`).
    The tensors are identical, and the 186 modules map one to one.
- **For an agent, prefer base + LoRA.** The merged checkpoint has no `base`. When the tuned model
  also wrote the screen descriptions, it put them on one long JSON line, which the agent's
  summary clipping cut; that hid a payment-confirmation dialog from the planner. Use `base` for
  descriptions and `grounder` for clicks, as on the H200.
- **Render screenshots headless.** Every measured agent run (the Gym `web_agent` evals and the
  agent harness) rendered the sandbox in headless Chromium. Headed Chromium rasterizes fonts differently: about 85k pixels of the
  sandbox's pay dialog changed, and on those pixels the describer dropped the selected-option
  cue (pay-50 passed 1/4 headed vs 8/8 headless). A headed browser is an unmeasured input for
  both the describer and the grounder.
- **Memory:** next to the BF16 Lightning planner on the same Mac (a 128 GB M4 Max), the grounder
  only fit at `--gpu-memory-utilization 0.08 --max-num-batched-tokens 2560`, with other large
  apps closed.
- Setup, the serve script (`scripts/run_local_grounder.sh`) and the full Gate G record:
  [docs/local-mac.md](https://github.com/pballou/nvidia-hack-26/blob/main/docs/local-mac.md)
  ("Gate G results").

## Training

### Data

Training data is [`patrickballou/gui-assist-data`](https://huggingface.co/datasets/patrickballou/gui-assist-data):
a cleaned OS-Atlas slice (Linux, macOS, AndroidWorld, UIBert, FineWeb).

- **Rows:** 19,621 targets, train 18,652 / dev 969, split by screenshot.
- **Cleaning:** burned-in-overlay screenshots dropped, and a perceptual-hash dedup against
  ScreenSpot-v2.
- **Instructions:** the dataset card covers the sources, licenses and the
  Nemotron-Lightning-rewritten instructions.
- **The two round-0 SFT variants, trained with the same config and seed:**
  - `sft-template` trained on all 18,652 train rows with raw labels.
  - `sft-lightning` trained on `train_lightning.jsonl`: 18,299 rows, 8,620 of them with a
    Lightning instruction (8,498 differ from the raw label).
  - The two also differ by the 353 numeric-label rows that the Lightning files drop.
- **Released variant:** `sft-lightning` won the round-0 pick on dev (86.27 vs 85.86), so the
  released model starts from it.
- **GRPO pools** are drawn from the round-0 run's training file (`train_lightning.jsonl`),
  never from dev or ScreenSpot.

### SFT (LoRA)

`training/sft.py` with TRL 1.16.0 and the config in `training/lora.py`:

| Setting | Value |
| --- | --- |
| LoRA | r = 16, alpha = 32, dropout 0.05, bias none |
| Target modules (language model only; vision tower frozen) | `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`, and Qwen3.5's Gated DeltaNet `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}` (18 of the 24 layers), matched under `model.language_model.layers.*` |
| Epochs / batch | 1 epoch; 4 per device × 8 gradient accumulation = 32 examples per step (583 steps for 18,652 rows; 572 for 18,299) |
| Optimizer schedule | lr 2e-4, cosine, warmup 3% of steps; bf16; gradient checkpointing; seed 0 |
| Examples | One target per example, no packing. Prompt-completion format, so loss is on the completion only; `max_length=None` (image tokens are never truncated) |
| Images | Pre-resized with `preprocess_screenshot`; processor capped at the same `MAX_PIXELS = 2_073_600` / `MIN_PIXELS = 65_536` |
| Run (`sft-template`) | 1× H200, 583 steps, 54 min, peak 8.6 GiB, loss 0.99 → 0.16 |
| Run (`sft-lightning`, released round 0) | 1× H200, 572 steps, Sat 18:44 → 19:33 (49 min, 4.93 s/step median), peak 8.6 GiB, loss 0.98 → 0.21 |

### GRPO (overnight NeMo Gym loop)

`training/night_loop.py` runs up to 3 rounds from the round-0 SFT adapter:

1. **Profile.** Take a fresh random 4,000-row train subset and run it through the
   `gui_grounding` env: `gym eval run --num-repeats 8 --temperature 1.0`, then
   `gym eval profile`.
2. **Pool.** Keep rows the model sometimes gets right (0 < pass rate < 1), capped at
   `pool_max`.
3. **GRPO.** Run `training/grpo.py` (TRL `GRPOTrainer`, vLLM colocated), continuing the LoRA.
   The reward is the Gym server's `/verify` (1 if the point is in the box) plus 0.1 × a format
   reward (the reply parses as a point).
4. **Gate.** Evaluate on dev at temperature 0. Keep the adapter only if dev improves by
   **≥ 0.5 points**; otherwise revert and stop. A `gym eval compare` against the previous round
   is saved next to it as a per-row flip report; only the dev delta decides.

| GRPO setting (code defaults) | Value |
| --- | --- |
| Group size / prompts per step | 6 / 8 |
| lr | 1e-5, constant with 5 warmup steps; max grad norm 1.0 |
| Loss | `dapo`, group-scaled rewards, `beta = 0` (no KL), 1 iteration |
| Sampling | temperature 1.0, top_p 1.0, max 32 completion tokens, thinking off |
| Limits | ≤ 200 steps and a 75-minute budget per round (knobs); gate D2 needs ≤ 35 s/step over the first 20 steps, or the round falls back to rejection-sampled SFT on reward=1 replies |

**The overnight run** (Sat 19:38 → 22:02 EDT, launch `20261017-193859`, from `sft-lightning`):

| Round | Pool | Steps | s/step | Verify reward first → last | Dev before → after | Kept |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **1,282** of 4,000 profiled rows were sometimes-pass (32%): FineWeb 472, Linux 230, UIBert 220, AndroidWorld 183, macOS 177. Default knobs | 170 (round budget) | 20.6 (gate D2 20.0) | 0.77 → 0.90 | 86.27 → **87.00** (+0.72; 22 fixed, 15 broken) | **Yes: the released adapter** |
| 2 | **242** weighted entries from 203 candidates (223 sometimes-pass of 4,000, 5.6%, minus 20 excluded ids). Knobs from the round-1 BLADE review: source weights UIBert 1.75, FineWeb 1.25, AndroidWorld 0.75, macOS 0.5, and 285 excluded train ids | 167 (round budget) | 20.4 (gate D2 21.1) | 0.53 → 0.86 | 87.00 → 86.89 (−0.10) | No: reverted, and the loop stopped |
| 3 | — | not run (round 2 wasn't kept) | | | | |

- **The released adapter used default knobs only.** The BLADE knob change went into round 2,
  which was reverted. The knob values come from a fixed rule in `scripts/blade_pair.py`; no
  model chooses them.
- **The rejection-sampled SFT fallback didn't run.** Gate D2 passed in both rounds, so no
  reward=1 rollouts were used as SFT data.
- **Clean runs:** 0 `/verify` failures in 16,176 GRPO reward calls, peak 45 GiB, no crashes.
- **Why round 2 was flat, likely:** round 1 sharpened the model, so a fresh profile found far
  fewer rows with any signal (223 sometimes-pass vs 1,282). The knob change is a confound, so
  its effect can't be separated from the smaller pool.
- **Dev deltas are exact row counts** (+0.72 = 7 rows, −0.10 = 1 row). The loop's own logs
  show +0.73 / −0.11, because they subtract accuracies already rounded to two decimals. The keep
  decisions are the same.

Sources: round 1's `grpo_command.txt`, `knobs_used.json`, `pool_stats.json` and
`round_1_summary.json`, and round 0's `round_0_summary.json`, ship in this repo's `training/`.
Round 2 and the loop's `FINAL.json` (best round 1, dev 87.0) are logged in runs.md (Sat
19:38–22:02 rows).

### Software and hardware

- **Hardware:** 1× NVIDIA H200 on NVIDIA Brev. The merge for the root weights ran on a Mac's
  CPU (torch 2.13.0, transformers 5.17.0, peft 0.21.2).
- **Software:** vLLM 0.31.0 (image pinned by digest), torch 2.13.0+cu130, transformers 5.17.0,
  TRL 1.16.0, peft 0.21.2, NeMo Gym 0.7.0. On the Mac: vllm-metal 0.31.0.

## NeMo Gym environments

Both environments are in the GitHub repo, in Gym's layout (`resources_servers/`):

- **`gui_grounding`** (eval and GRPO reward):
  - Adapted from NeMo Gym v0.7.0's `gui_coordinate`.
  - The input is a screenshot plus an instruction, and the model answers with `point_2d`.
  - Reward 1 if the point is inside the target's pixel box, with edges counting as inside.
  - Metrics are split by platform, text/icon and source.
  - It produced every grounding accuracy in Results, ran the GRPO profiles, and serves
    `/verify` as the GRPO reward.
- **`web_agent`** (agent eval):
  - The planner is Nemotron 3.5 Lightning (official BF16). It gets a text observation and
    acts only through five tools.
  - A `click` description goes to this grounder (served name `grounder`, or `base` for the
    baseline). Playwright clicks the returned pixels.
  - The reward comes from the sandbox app's own state.
  - It gives the base-vs-tuned agent comparison above.

## Intended use

- **The grounding tool of a GUI agent,** especially an accessibility agent. A planner or a
  person describes one target in words, and this model returns where to click on the current
  screenshot.
  - It covers desktop (Linux, macOS), mobile (Android) and web screenshots.
  - It works from pixels alone, so it applies to apps whose DOM or accessibility tree isn't
    available.
- **A small, self-hostable baseline** for GUI-grounding research. It fits next to a 30B planner
  on one GPU.

**Out of scope:**

- Acting on its own: it returns a point and has no notion of task success.
- Detection of several targets, box output and OCR or question answering.
- Unattended high-stakes actions (payments, deletions, account changes) without confirmation.

## Limitations

- **Agent failure clusters** from the `web_agent` runs:
  - **Scroll-dependent targets:** the model only sees the current screenshot and can't point at
    something off-screen. Both scroll tasks stayed at 0/3 with all three grounders.
  - **A canvas chart** (picking a bar in a drawn chart): 0/3 with base and `sft-template`, 1/3
    with the released adapter. The review attributes that pass to the planner: both tuned
    grounders hit the same bar, and only the released run's planner stopped there.
  - **Dense toggle grids,** where neighboring switches look alike: `sms-billing-off` was 0/3 with
    base, 2/3 with `sft-template` and 0/3 with the released adapter. The review traces
    `sft-template`'s passes to a one-off screen summary and to its habit of resolving a vague
    target onto a real switch.
- **Icons are still weaker than text:** SSv2-clean 85.37 vs 94.18 for the released adapter
  (`sft-lightning` 81.67 vs 93.04, `sft-template` 84.81 vs 92.33).
- **Web is the released adapter's weakest dev source:** FineWeb went 82.9 (base) → 85.9
  (`sft-lightning`) → 84.25 (released; round 1 broke 8 FineWeb rows and fixed 2). On SSv2-clean,
  though, web gained the most of the three platforms (81.9 → 88.10).
- **The training labels are mostly accessibility-tree and HTML labels,** some of them rewritten
  by an LLM that never saw the screen.
  - Some source boxes are on the wrong element.
  - Phrasings far from ScreenSpot style may ground worse.
- **The training data has limited coverage:**
  - desktop is Linux and macOS only (no Windows);
  - the labels are mostly English;
  - there's no high-DPI or multi-monitor handling beyond the 2 MP cap. Very small targets on
    large screens are harder after downscaling.
- **One target per call** and no reasoning: there is no "I can't find it" signal. A wrong but
  well-formed point looks like a right one. In the agent runs, when the target wasn't on screen,
  the released adapter tended to anchor on a literal text match, such as a side-menu link.
- **Desktop evaluation shares machines with the training data:** see the
  [28-row exclusion](#the-28-row-screenspot-v2-exclusion).

## Parameter count

- **This model:** `Qwen/Qwen3.5-2B` is **2B** on its model card (HF safetensors total 2.27B).
  The LoRA adapter (r = 16 on 186 modules) adds about 16.8M parameters. That number is computed
  from the base weight shapes, and PEFT loads an adapter of this config as ~16.8M trainable
  parameters in 372 tensors (A and B for each module).
- **The whole agent:** Nemotron 3.5 Lightning (30B by card) + this grounder (2B) + NVIDIA
  Parakeet TDT 0.6B for voice input = **32.6B by model-card counts**. The event's cap is 33B.
- **Why the grounder counts once:** the base and tuned roles are one loaded model with the
  adapter switched per request, not two instances.

## Licenses

- **This model:** Apache-2.0, the same as the base model
  ([`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B): Apache-2.0 per its card). The weights
  derive from it, so its license and NOTICE carry over (LICENSE at the repo root). Confirmed by the team, Sun 2026-10-18.
- **Training data** (released separately as `patrickballou/gui-assist-data` under CC BY 4.0, its most
  restrictive source). Attribution for the data this model learned from:
  - OS-Atlas-data is Apache-2.0 per its card.
  - The UIBert expressions are CC BY 4.0 per the UIBert repo, and are attributed here and in
    the dataset card.
  - The FineWeb-derived web screenshots show third-party pages.
  - Lightning outputs:
    - The 1,003 rows from the downloaded BF16 checkpoint fall under its OpenMDW-1.1 license,
      which places no restrictions on outputs.
    - The 8,028 rows from NVIDIA's hosted API fall under the NVIDIA API Trial Terms of Service,
      which limit trial use to non-production. **TBD (open question for the dataset card):**
      Patrick's ~9:30 release check decides whether these rows are published or only the local
      rows are.
  - Details and the open questions are in the dataset card.

## Citation

```bibtex
@article{wu2024atlas,
  title   = {OS-ATLAS: A Foundation Action Model for Generalist GUI Agents},
  author  = {Wu, Zhiyong and others},
  journal = {arXiv preprint arXiv:2410.23218},
  year    = {2024}
}
```

GUI assist (team sherpas: Patrick, Kyle, Charlie), NVIDIA x Red Hat Small Models Hack,
Raleigh NC, 17–18 October 2026. <https://github.com/pballou/nvidia-hack-26>
