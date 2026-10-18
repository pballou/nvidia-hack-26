---
license: apache-2.0  # base model is Apache-2.0; team to confirm (see "Licenses")
base_model: Qwen/Qwen3.5-2B
library_name: transformers  # TBD: change to peft if only the adapter/ folder is uploaded (no merged weights at the root)
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

The model is a LoRA adapter on `Qwen/Qwen3.5-2B`'s language model; the vision encoder is frozen.
Training was LoRA SFT on an OS-Atlas slice with Nemotron-Lightning-rewritten instructions
(`sft-lightning`). Up to 3 overnight GRPO rounds followed, with a NeMo Gym environment's
`/verify` endpoint as the reward. A round was kept only if it improved dev (**TBD:** which
rounds were kept).

> Draft card. Items marked **TBD** are filled in once the overnight GRPO loop and the morning
> evals finish.

## At a glance

| | |
| --- | --- |
| Base model | [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B), revision `15852e8c16`, Apache-2.0 |
| Parameters | 2B (model-card count; HF safetensors total 2.27B). The LoRA adapter adds about 16.8M (16,819,200, computed for r = 16 from the base weight shapes; **TBD:** confirm against `adapter_model.safetensors`) |
| Released checkpoint | Round 0: `sft-lightning`, picked on dev (86.27 vs `sft-template`'s 85.86 in the same harness). **TBD:** plus the GRPO rounds kept on dev (from `/home/ubuntu/night/FINAL.json`) |
| Input | One screenshot (any aspect ratio, resized to ≤ 2,073,600 px) + one instruction |
| Output | `{"point_2d": [x, y]}`, integers on a 0–1000 grid relative to the image |
| Held-out test | ScreenSpot-v2-clean: 1,244 of ScreenSpot-v2's 1,272 rows. **28 rows excluded**, see [below](#the-28-row-screenspot-v2-exclusion) |
| Headline so far (SFT, before GRPO) | `sft-template` (raw labels; not the released adapter): SSv2-clean **86.3% → 89.07%**; icons **78.1% → 84.8%**; dev 76.4% → 85.66%; agent tasks verified 18/36 → 22/36. `sft-lightning` (round 0 of the released model): dev 86.27%, SSv2-clean TBD |

## Files in this repo

| Path | What | Source on the GPU box |
| --- | --- | --- |
| `adapter/` | PEFT LoRA adapter (`adapter_config.json`, weights), for vLLM `--lora-modules` or `PeftModel` | `/home/ubuntu/ckpt/<run>/adapter`. **TBD:** `<run>` from `/home/ubuntu/night/FINAL.json` |
| `/` (root) | The same model with the adapter merged into the base weights (for runtimes without LoRA, e.g. vllm-metal on a Mac) | `/home/ubuntu/ckpt/<run>/merged`. **TBD:** no merge/export script is in the repo yet |
| `training/` (TBD) | `args.json`, `run.json`, `summary.json` and `train.log` for SFT; `grpo_summary.json` for each kept GRPO round; the loop's `FINAL.json` | `/home/ubuntu/ckpt/sft-<variant>/`, `/home/ubuntu/ckpt/night/round_<r>/grpo/`, `/home/ubuntu/night/` |
| `eval/` (TBD) | Gym eval metrics for dev and SSv2-clean, and `gym eval compare` reports | `/home/ubuntu/night/evals/<name>/` |

## Results

All numbers measure **accuracy**: the predicted point lands inside the target's pixel box, with
edges counting as inside. Every run goes through the NeMo Gym `gui_grounding` env
(`gym eval run`), at temperature 0 with max 128 output tokens. Images are capped at 2 MP for
both models. Checkpoints are chosen on **dev only**. SSv2-clean is only reported, and the final
checkpoint is evaluated on it once.

| Model | Dev (969) | SSv2-clean (1,244) | Mobile | Desktop | Web | Text | Icon |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-2B base | 76.4 | 86.3 | 91.4 | 84.0 | 81.9 | 92.5 | 78.1 |
| SFT, raw labels (`sft-template`, round 0) | 85.66 | 89.07 | 92.2 | 89.5 | 85.1 | 92.3 | 84.8 |
| SFT, Lightning instructions (`sft-lightning`, round 0) | 86.27 | TBD | TBD | TBD | TBD | TBD | TBD |
| **Released: `sft-lightning` + kept GRPO rounds** (best on dev) | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

The `sft-lightning` dev number is from the loop harness, where `sft-template` scored 85.86.

Dev by source (OS-Atlas, held out by screenshot):

| Model | Linux | macOS | AndroidWorld | UIBert | FineWeb |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-2B base | 68.8 | 73.4 | 70.5 | 78.4 | 82.9 |
| `sft-template` | 86.9 | 91.1 | 81.3 | 85.1 | 84.5 |
| `sft-lightning` | TBD (desktop 88.9) | TBD (desktop 88.9) | 78.4 | 88.8 | 85.9 |
| **Released** | TBD | TBD | TBD | TBD | TBD |

- **Where SFT helped (`sft-template`):** the gain is mostly desktop (Linux +18.1, macOS +17.7 on dev; SSv2-clean
  desktop +5.5) and icons (SSv2-clean +6.7). Text grounding was already strong and stayed flat
  (92.5 → 92.3).
- **Dev noise:** re-evaluating the same adapter in a new launch gave 85.66 vs 85.86. That's
  about 0.2 points from vLLM batch nondeterminism at temperature 0. The GRPO keep gate is
  +0.5 points.
- **GRPO rehearsal (not the released model):**
  - one round from `sft-template` on a 111-row pool, 30 steps;
  - dev went from 85.86 to 86.69 (+0.83) in the same harness.
- **Agent task success** (NeMo Gym `web_agent` env, 12 sandbox tasks × 3 repeats, official
  BF16 Lightning planner, thinking off; only the grounder's served name changes). Measured with
  `grounder` = `sft-template`, before GRPO. That isn't the released adapter, which starts from
  `sft-lightning`. **TBD:** rerun with the released checkpoint, or keep this label.

  | | `base` | `sft-template` |
  | --- | --- | --- |
  | Verified success | 18/36 (50%) | **22/36 (61%)** |
  | Claim matches the sandbox | 26/36 | 31/36 |
  | False success claims | 8 | 4 |
  | Failed actions | 10 | 0 |
  | No-change actions | 83 | 45 |

  - The mean reward rose by 0.111, but the 95% CI is [-0.06, +0.28]: suggestive, not
    significant at n=36. The drop in failed actions is significant.
  - Gains: `text-size-large` 0→2, `sms-billing-off` 0→2, `message-support-meter` 1→2.
  - Regression: `high-contrast` 2→1.
- **Not comparable with published ScreenSpot-v2 numbers:**
  - SSv2-clean drops 28 desktop rows;
  - we use our own prompt and harness (see "Prompt" below).

Sources: [docs/results.md](https://github.com/pballou/nvidia-hack-26/blob/main/docs/results.md)
and [docs/runs.md](https://github.com/pballou/nvidia-hack-26/blob/main/docs/runs.md), which has
the command, config and output path of every run. The agent table is in results.md ("Agent")
and in runs.md (the Sat 18:45 row).

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
`scripts/serve_vlm.sh`). Only the volume mounts and the adapter path differ. It uses the pinned
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
- **Without LoRA support** (e.g. vllm-metal): serve the merged weights at the repo root
  instead (**TBD**, see Files).

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
| Run (`sft-lightning`, released round 0) | 1× H200, 572 steps, started Sat 18:44, done 19:33. **TBD:** peak memory and loss (no closed runs.md row yet) |

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
4. **Gate.** Evaluate on dev at temperature 0, then run `gym eval compare`. Keep the adapter
   only if dev improves by **≥ 0.5 points**; otherwise revert and stop.

| GRPO setting (code defaults) | Value |
| --- | --- |
| Group size / prompts per step | 6 / 8 |
| lr | 1e-5, constant with 5 warmup steps; max grad norm 1.0 |
| Loss | `dapo`, group-scaled rewards, `beta = 0` (no KL), 1 iteration |
| Sampling | temperature 1.0, top_p 1.0, max 32 completion tokens, thinking off |
| Limits | ≤ 200 steps per round (knob); gate D2 needs ≤ 35 s/step over the first 20 steps, or the round falls back to rejection-sampled SFT on reward=1 replies |

**TBD for the released model:**

- rounds run and kept, steps per round, pool sizes;
- the knob values after the BLADE review;
- whether the rejection-sampled SFT fallback ran;
- the per-round dev numbers.

These come from `round_<r>/summary.json`, `knobs.json` and `FINAL.json`.

### Software and hardware

- **Hardware:** 1× NVIDIA H200 on NVIDIA Brev.
- **Software:** vLLM 0.31.0 (image pinned by digest), torch 2.13.0+cu130, transformers 5.17.0,
  TRL 1.16.0, peft 0.21.2, NeMo Gym 0.7.0.

## NeMo Gym environments

Both environments are in the GitHub repo, in Gym's layout (`resources_servers/`):

- **`gui_grounding`** (eval and GRPO reward):
  - Adapted from NeMo Gym v0.7.0's `gui_coordinate`.
  - The input is a screenshot plus an instruction, and the model answers with `point_2d`.
  - Reward 1 if the point is inside the target's pixel box, with edges counting as inside.
  - Metrics are split by platform, text/icon and source.
  - It produced every number above, ran the GRPO profiles, and serves `/verify` as the GRPO
    reward.
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
    something off-screen. Both scroll tasks stayed at 0/3 with `sft-template`.
  - **A canvas chart** (picking a bar in a drawn chart): 0/3 with `sft-template`.
  - **Dense toggle grids,** where neighboring switches look alike. The base grounder failed
    `sms-billing-off`; `sft-template` passed it 2 of 3 times.
- **Icons are still weaker than text** (SSv2-clean 84.8 vs 92.3 after `sft-template`;
  `sft-lightning` TBD).
- **Web gained least on dev** (FineWeb +1.6 with `sft-template`, +3.0 with `sft-lightning`).
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
  well-formed point looks like a right one.
- **Desktop evaluation shares machines with the training data:** see the
  [28-row exclusion](#the-28-row-screenspot-v2-exclusion).

## Parameter count

- **This model:** `Qwen/Qwen3.5-2B` is **2B** on its model card (HF safetensors total 2.27B).
  The LoRA adapter (r = 16 on 186 modules) adds about 16.8M parameters. That number is computed
  from the base weight shapes; **TBD:** confirm it against the adapter file.
- **The whole agent:** Nemotron 3.5 Lightning (30B by card) + this grounder (2B) + NVIDIA
  Parakeet TDT 0.6B for voice input = **32.6B by model-card counts**. The event's cap is 33B.
- **Why the grounder counts once:** the base and tuned roles are one loaded model with the
  adapter switched per request, not two instances.

## Licenses

- **This model:** Apache-2.0, the same as the base model
  ([`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B): Apache-2.0 per its card).
  **TBD:** team to confirm.
- **Training data:**
  - OS-Atlas-data is Apache-2.0 per its card.
  - The UIBert expressions are CC BY 4.0 per the UIBert repo, and are attributed here and in
    the dataset card.
  - The FineWeb-derived web screenshots show third-party pages.
  - Lightning outputs:
    - The 1,003 rows from the downloaded BF16 checkpoint fall under its OpenMDW-1.1 license,
      which places no restrictions on outputs.
    - The 8,028 rows from NVIDIA's hosted API fall under the NVIDIA API Trial Terms of Service,
      which limit trial use to non-production. **Team to confirm**; see the dataset card.
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
