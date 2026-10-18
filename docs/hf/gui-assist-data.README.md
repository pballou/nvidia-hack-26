---
license: cc-by-4.0  # the most restrictive source license we include (UIBert expressions, CC BY 4.0); OS-Atlas-derived parts are also Apache-2.0. See "Sources and licenses"
pretty_name: "GUI Assist grounding data: OS-Atlas slice + Nemotron Lightning instructions"
language:
- en
task_categories:
- image-text-to-text
size_categories:
- 10K<n<100K
tags:
- gui-grounding
- gui-agent
- screenshots
- accessibility
- synthetic-instructions
- nemotron
- os-atlas
configs:
- config_name: lightning
  default: true
  data_files:
  - split: train
    path: train_lightning.jsonl
  - split: dev
    path: dev.jsonl
- config_name: template
  data_files:
  - split: train
    path: train_template.jsonl
  - split: dev
    path: dev.jsonl
- config_name: rewrites
  data_files:
  - split: train
    path: lightning_rewrites/rewrites.jsonl
- config_name: manifest
  data_files:
  - split: full
    path: manifest.jsonl
---

# GUI Assist grounding data

This is the training and dev data for
[`patrickballou/gui-assist-grounder`](https://huggingface.co/patrickballou/gui-assist-grounder),
a 2B GUI click-grounding model: given a screenshot and an instruction, it returns where to
click. We built the data for the GUI Assist accessibility agent at the NVIDIA x Red Hat Small
Models Hack (October 2026). Code: <https://github.com/pballou/nvidia-hack-26>.

The repo contains four things:

1. **A cleaned slice of [OS-Atlas-data](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data):**
   19,621 click targets from Linux, macOS, AndroidWorld, UIBert and FineWeb screenshots.
   Each target has a pixel box, a platform and a train/dev split. Dev is split by screenshot.
2. **Instructions rewritten by NVIDIA Nemotron 3.5 Lightning.** Raw accessibility and HTML
   labels ("More options") became ScreenSpot-style click instructions ("open more options").
   The repo has every keep/drop decision, the verbatim prompt, the commands and a human audit.
3. **Two training files with the same rows in the same order (18,299 rows each).**
   `train_lightning.jsonl` mixes Lightning rewrites with raw labels; `train_template.jsonl`
   has raw labels only. They are for an ablation.
4. **The GRPO prompt pool of the released model** (`grpo/round_1_pool.jsonl`, 1,282 train
   rows). See [GRPO prompt pool](#grpo-prompt-pool).

The released grounder was trained with LoRA SFT on `train_lightning.jsonl`, then one round of
GRPO (TRL, rewarded by the NeMo Gym `gui_grounding` environment's point-in-box check) on
`grpo/round_1_pool.jsonl`. It scores **90.35%** on ScreenSpot-v2-clean
(base model: 86.25%) and 87.00% on `dev`. Details are on the
[model card](https://huggingface.co/patrickballou/gui-assist-grounder).

**Images are not re-hosted.** Each row refers to its OS-Atlas screenshot by source file name.
See [Getting the images](#getting-the-images).

## At a glance

| | |
| --- | --- |
| Targets in the manifest | 19,621 (train 18,652 / dev 969) |
| Rows in each release training file | 18,299 (the 18,652 train rows minus 353 numeric-label rows) |
| Platforms (manifest) | web 6,986 (FineWeb) / desktop 6,635 (Linux 3,271, macOS 3,364) / mobile 6,000 (AndroidWorld 3,000, UIBert 3,000) |
| Lightning rewrites | 9,384 train targets selected. 8,620 rewritten and kept (8,498 differ from the raw label; for 122, Lightning returned the label unchanged), 411 model drops kept with the raw label, 353 removed by a numeric-label rule |
| Lightning endpoints | NVIDIA-hosted API `nvidia/nemotron-3.5-lightning-30b-a3b`: 8,028 rows. Official `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` on vllm-metal (Mac): 1,003 rows |
| Human audit (random sheet) | 42/43 = 97.7% faithful among cards that could be checked; 42/49 = 85.7% if "unsure" counts as not faithful |
| GRPO prompt pool (released model) | 1,282 train rows, from round 1 (the only GRPO round in the released model) |
| Held-out test it was built for | ScreenSpot-v2-clean: ScreenSpot-v2 (revision `5efbb1f1b5`) minus 28 desktop rows, 1,244 rows. **No ScreenSpot-v2 images or boxes are in this repo**, only the file names and instructions of the 28 excluded rows |

## Files

| Path in this repo | Rows | sha256 (prefix) | What it is |
| --- | --- | --- | --- |
| `README.md` | — | — | This card |
| `manifest.jsonl` | 19,621 | `4ceb7ea3d7ac8e85` | The slice: every target, raw label, pixel box, split |
| `dev.jsonl` | 969 | `a63416c399422811` | Dev rows (raw labels), same columns as the training files. Made from `manifest.jsonl` with the command in [Splits](#splits) |
| `train_lightning.jsonl` | 18,299 | `7039f73114f1e35e` | Train rows; 8,620 Lightning instructions, of which 8,498 differ from the raw label |
| `train_template.jsonl` | 18,299 | `93bee7ca56a266c4` | Same rows and order, raw labels |
| `train_files.READY.json` | — | — | sha256 of the two training files at delivery |
| `screenspot_v2_excluded_rows.json` | 28 | — | The ScreenSpot-v2 rows excluded from our test set, and why |
| `lightning_rewrites/rewrites.jsonl` | 9,384 | `390c46db50b13fb1` | One record per selected target: original, rewrite or drop reason, style, model, time |
| `lightning_rewrites/METHOD.md` | — | — | Full method: verbatim prompt, few-shot examples, validation rules, every invocation's command, counts |
| `lightning_rewrites/stats.json` | — | — | All rewrite counts, sampling settings, per-invocation commands and token usage |
| `lightning_rewrites/apply_stats.json` | — | — | How the rewrites were applied to make the two training files |
| `lightning_rewrites/review_verdicts_v4_seed0.json` | 66 cards | — | Human audit verdicts |
| `build/sampler_stats.json` | — | — | Sampler arguments and per-source filter counts |
| `build/overlay_flagged.json` | 13 images | — | Linux screenshots with burned-in annotation overlays (dropped) |
| `build/manifest.stats.json` | — | — | Overlay drop, ScreenSpot-v2 pHash dedup (every match and distance), dev/train split counts; 20,000 input rows → 19,621 |
| `build/blank_dedup_recheck.json` | 79 images | — | Second check of the 79 near-blank ScreenSpot-v2 images that the pHash step skips, against the manifest's 4,339 screenshots: exact MD5, thumbnail pixel difference and SSIM, with each image's 3 nearest manifest screenshots. Results in [How it was built](#how-it-was-built), step 3 |
| `grpo/round_1_pool.jsonl` | 1,282 | `cc4458e6d8a6357d` | The GRPO prompt pool of the released model's round (round 1) |
| `grpo/round_1_pool_stats.json` | — | — | Per-source profile of the 4,000 profiled rows (always / sometimes / never solved) and the pool's per-source counts |

`lightning_rewrites/METHOD.md`, `stats.json`, `apply_stats.json` and
`review_verdicts_v4_seed0.json` are also in the GitHub repo under
`data_prep/rewrite/runs/os_atlas_final_v4/`, and `screenspot_v2_excluded_rows.json` is
`benchmarks/screenspot_v2/excluded_rows.json` there.

**Not included:**

- `cache.sqlite`, the Lightning request cache.
- `review.html`, the audit contact sheet. It embeds 180 images cut from the screenshots (for
  each of the 66 cards, a crop and a thumbnail of the whole screenshot, plus 48 zooms), and we
  don't re-host images.
- Any images.
- `rs_sft/`. The rejection-sampled SFT fallback of the GRPO loop wasn't used.

## Splits

| Split | Rows | Instructions | Use |
| --- | --- | --- | --- |
| `train` (manifest) | 18,652 | Raw labels (UIBert: natural) | The `sft-template` adapter (raw-label SFT, a round-0 candidate not used in the released model) trained on all 18,652 |
| `train` (release training files) | 18,299 | `lightning`: 8,620 Lightning instructions (8,498 changed, 122 identical to the label) + 6,813 raw + 2,866 natural. `template`: all raw/natural | SFT; the GRPO pool was drawn from `train_lightning.jsonl` |
| `dev` | 969 | Raw labels (UIBert: natural); **never rewritten** | Checkpoint selection |

Train rows by source (manifest → training files): AndroidWorld 2,861 → 2,676; FineWeb
6,624 → 6,533; Linux 3,095 → 3,093; macOS 3,206 → 3,131; UIBert 2,866 → 2,866. Dev rows by
source: AndroidWorld 139, FineWeb 362, Linux 176, macOS 158, UIBert 134.

`dev` is 5% of screenshot **groups** per source (`data_prep/finalize_manifest.py`, seed 0), so
no screenshot is in both splits. A group is the base screenshot with OS-Atlas's `_subN` crop
suffix removed (Linux, macOS, AndroidWorld), the page (FineWeb) or the Rico id (UIBert).

There is no test split. Our held-out test is ScreenSpot-v2-clean, which isn't redistributed
here (see [Contamination](#contamination-handling-screenspot-v2)).

`dev.jsonl` has the manifest's dev rows, with `instruction_style` filled in and the two extra
columns the training files carry (`instruction_original`, `lightning_drop_reason`), so all
splits in a config share one schema. It was made with this command (Python
3.12; sha256 prefix `a63416c399422811`):

```bash
python3 - <<'EOF'
import json
with open("manifest.jsonl") as f, open("dev.jsonl", "w") as out:
    for line in f:
        r = json.loads(line)
        if r["split"] != "dev":
            continue
        r["instruction_original"] = r["instruction"]
        r["instruction_style"] = "natural" if r["source"] == "os_atlas/uibert" else "raw"
        r["lightning_drop_reason"] = None
        out.write(json.dumps(r) + "\n")
EOF
test "$(wc -l < dev.jsonl)" -eq 969
```

## Row schema

`manifest.jsonl`, `dev.jsonl`, `train_lightning.jsonl` and `train_template.jsonl` hold one
JSON object per click target:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | `<subset>:<screenshot stem>:<k>`, e.g. `linux:output_20240912_143552_original_screenshot:0` |
| `source` | string | `os_atlas/linux`, `os_atlas/macos`, `os_atlas/androidworld`, `os_atlas/uibert`, `os_atlas/fineweb` |
| `image_path` | string | `images/<subset>/<OS-Atlas img_filename, "/" replaced by "_">`. Not included: see below |
| `img_w`, `img_h` | int | Size of the original screenshot in pixels |
| `instruction` | string | The click instruction to train on: raw label in the manifest, template file and dev; rewritten or raw in `train_lightning.jsonl` |
| `bbox` | [float ×4] | `[x1, y1, x2, y2]` in **pixels** of the original screenshot (OS-Atlas's normalized `[left, top, right, bottom]` × width/height, rounded to 0.1 px) |
| `platform` | string | `desktop`, `mobile` or `web` |
| `data_type` | string or null | OS-Atlas's **control type** from the accessibility tree (`push-button`, `AXStaticText`, `android.widget.TextView`, ...). Null for UIBert and FineWeb. **Not** ScreenSpot's text/icon label |
| `instruction_style` | string or null | Training files: `functional`, `descriptive`, `positional` (Lightning rewrite), `raw` or `natural`. Dev: `raw` or `natural`. Null in the manifest |
| `group` | string | Screenshot group used for the split |
| `split` | string | `train` or `dev` |
| `instruction_original` | string | Training files and dev: the raw manifest label |
| `lightning_drop_reason` | string or null | Training files: set on the 411 rows Lightning dropped and that we kept with the raw label: `unintelligible` (264), `numeric` (68), `needs_screen` (29), `duplicate_neighbor` (26), `llm_error` (24); absent on the other rows (null when loaded). Dev: null on every row |

`lightning_rewrites/rewrites.jsonl` fields are `id`, `source`, `original`, `rewritten` (null
when dropped), `style`, `keep`, `drop_reason`, `prompt_version` (`v4`), `model` (the model
that produced the row, or `rule:is_numeric_label`) and `ts`. METHOD.md documents each one.

`grpo/round_1_pool.jsonl` rows have `id`, `source`, `instruction`, `platform`, `data_type`,
`bbox`, `img_w` and `img_h` with the same meaning as above, plus `responses_create_params`,
the NeMo Gym request (see [GRPO prompt pool](#grpo-prompt-pool)).

## Getting the images

The screenshots come from
[OS-Copilot/OS-Atlas-data](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data). Download
the archives below. A file's `image_path` is `images/<subset>/` plus its archive member name,
with `/` replaced by `_`:

| Subset | OS-Atlas annotation file | OS-Atlas image archive |
| --- | --- | --- |
| Linux | `desktop_domain/linux_splited.json` | `desktop_domain/linux_images.zip` |
| macOS | `desktop_domain/macos_splited.json` | `desktop_domain/macos_images.zip` |
| AndroidWorld | `mobile_domain/aw_mobile.json` | `mobile_domain/mobile_images.zip` |
| UIBert | `mobile_domain/uibert_raw.json` | `mobile_domain/UIBert.zip` |
| FineWeb | `web_domain/fineweb_3m.json` | `web_domain/fineweb_3m_s11.zip` (only this one of the ten FineWeb zips) |

`data_prep/sample_os_atlas.py` in the GitHub repo reads exactly these files. It reads them
from `--raw` when they are there; otherwise it fetches them from the Hub, and for the FineWeb
zip it reads only the needed members over HTTP range requests.

- **Revision:** `scripts/download.py` downloaded the raw OS-Atlas files at the pinned commit
  `e3a4c90c5f6129c25efdaa0671b08a11f7cb8f3f`.
- The sampler itself doesn't pin a revision, so anything it fetched came from `main`. On
  2026-10-17, `main` was that same commit (last modified 2024-12-04).

Training resizes every screenshot with `common/grounding.py:preprocess_screenshot`. It converts
to RGB and does an aspect-preserving resize to at most 2,073,600 pixels, with sides that are
multiples of 32. Boxes here are in original pixels, so they don't depend on that resize.

## Sources and licenses

**This dataset is released under CC BY 4.0**, the most restrictive license among the sources
we include: the UIBert expressions are CC BY 4.0, and everything else is Apache-2.0
(OS-Atlas-data) or has no restrictions on outputs (Lightning, OpenMDW-1.1). CC BY 4.0 requires
attribution: cite OS-Atlas, UIBert and this dataset (citations below). Images aren't included;
they stay with their sources (Rico screenshots are under research terms, and FineWeb pages
belong to their sites).

| Source | What we use | License |
| --- | --- | --- |
| [OS-Atlas-data](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data) (Wu et al., 2024) | Screenshots (by reference), labels, boxes, control types for all five subsets | **Apache-2.0** (dataset card). The card lists no per-subset licenses |
| OS-Atlas Linux and macOS desktop | Captured by the OS-Atlas authors | Apache-2.0 via OS-Atlas-data |
| OS-Atlas AndroidWorld | Sampled by OS-Atlas from the [AndroidWorld](https://github.com/google-research/android_world) environment | Released in OS-Atlas-data (Apache-2.0). The AndroidWorld code repo is Apache-2.0. For the screenshots' app content, see source |
| [UIBert](https://github.com/google-research-datasets/uibert) (Bai et al., 2021), on [Rico](https://interactionmining.org/rico) screenshots | Natural referring expressions (UIBert rows: 2,866 train, 134 dev). We filtered them and normalized them (Unicode NFKC, collapsed whitespace); otherwise they are unchanged | The UIBert README says its datasets are licensed under [**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/). Credit: Bai et al., UIBert (citation below). Rico images: the [Rico copyright notice](http://interactionmining.org/archive/rico/copyright.txt) says the screenshots "may contain copyrighted work" and gives them out under research terms, not an open license. We don't re-host them |
| OS-Atlas FineWeb web | Pages crawled by OS-Atlas from URLs in [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | FineWeb (the URL and text corpus) is ODC-By 1.0, and its card adds CommonCrawl's Terms of Use. That covers the corpus, not the pages. For the rendered page content and its labels, **see source**: the rights belong to the respective sites |
| NVIDIA Nemotron 3.5 Lightning outputs (`lightning_rewrites/`, `instruction` in `train_lightning.jsonl`) | Rewritten instructions, keep/drop decisions | **Local BF16 checkpoint (1,003 rows):** the `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` card gives OpenMDW-1.1. The license text says it "does not impose any restrictions or obligations with respect to any use, modification, or sharing of any outputs". **NVIDIA-hosted API (8,028 rows):** the build.nvidia.com page says "The trial service is governed by the NVIDIA API Trial Terms of Service", and the model by OpenMDW-1.1. Those terms say you own the Generated Content unless the model's license says otherwise (§6.3). They also grant access "for limited trial purposes only and without use of the API Service or Generated Content in production" (§1.2), and, without a paid subscription, "only for internal testing and evaluation purposes" (§1.4). The team decided on Sun 2026-10-18 to publish these rows: we own the generated content (§6.3), the model license puts no restrictions on outputs, and the hackathon requires publishing model-generated training data. Each row records which endpoint produced it |
| [ScreenSpot-v2](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2) | **Not included.** Used only to dedup our training images, and as the held-out test | Apache-2.0 (dataset card) |

Not used: ScreenSpot v1/v2 data for training, GroundUI-18K and Wave-UI (both contain
ScreenSpot), and OS-Atlas Windows, SeeClick web, AMEX, Widget Captioning, RICOSCA and
GroundCUA.

## How it was built

All code is in the GitHub repo. The Lightning step's exact commands are in
`lightning_rewrites/METHOD.md`, generated by `rewrite.py` from the cached records of each run.

1. **Sample** (`data_prep/sample_os_atlas.py`, seed 0). Quotas: Linux 3,500, macOS 3,500,
   AndroidWorld 3,000, UIBert 3,000, FineWeb 7,000 (20,000 targets). At most 8 targets per
   screenshot. FineWeb rows are kept with probability 0.25 while streaming. The sampler drops:
   - boxes that are invalid, zero-area or over 15% of the screen;
   - labels that are icon-font glyphs or have no letters;
   - `AutomationID:`, URL and "running window" labels;
   - labels over 12 words;
   - labels that point at two or more different boxes on one screenshot.

   Per-source drop counts are in `build/sampler_stats.json`.
2. **Drop burned-in overlays** (`data_prep/rewrite/overlay_scan.py`). In one capture window,
   OS-Atlas's capture tool rendered accessibility-role labels as red text into Linux
   screenshots. We drop 13 images (98 rows), found by pure-red pixel fraction > 0.2%, scoped to
   Linux.
3. **Dedup against ScreenSpot-v2** (`data_prep/finalize_manifest.py`). We compute a 64-bit
   perceptual hash (`imagehash.phash`) of every training screenshot and compare it with every
   ScreenSpot-v2 image. We drop any screenshot within Hamming distance 8. Result: 36 training
   screenshots (281 rows) dropped: Linux 17, macOS 17, FineWeb 2. Every match and its distance
   is in `build/manifest.stats.json`.

   Near-blank ScreenSpot-v2 images (grayscale std < 25; 79 images) are skipped as match
   targets, because pHash collides on them. `data_prep/recheck_blank_dedup.py` checks those 79
   separately against all 4,339 screenshots of the final manifest (train and dev). Its output
   is `build/blank_dedup_recheck.json`:
   - **Exact MD5** of the file bytes and of the decoded pixels: no matches.
   - **Similarity flags:** mean absolute difference of 32×32 grayscale thumbnails (flag
     ≤ 0.03) and a single-window SSIM (flag ≥ 0.9), over each image's 3 nearest manifest
     screenshots. 65 of the 79 images have at least one flagged neighbor: 190 flagged pairs
     over 72 manifest screenshots. All 190 are flagged on pixel difference; 6 of them also
     have SSIM ≥ 0.9 (highest 0.959).
   - The file records flags only. The 72 flagged screenshots are still in this release:
     278 manifest rows (273 train, 5 dev), 264 of them in each training file.
4. **Split** by screenshot group, 5% dev per source → 19,621 rows (train 18,652 / dev 969).
5. **Rewrite with Lightning** (train split only; `data_prep/rewrite/rewrite.py`, prompt `v4`,
   code `d0d9d625`). Details are in the next section.
6. **Apply** (`data_prep/rewrite/apply.py`). This writes `train_lightning.jsonl` and
   `train_template.jsonl` with the same 18,299 rows in the same order. Only the deterministic
   numeric-label rule removes rows (353). Rows Lightning dropped stay in both files with
   their raw label.
7. **GRPO prompt pool** (`training/night_loop.py`, round 1). See
   [GRPO prompt pool](#grpo-prompt-pool).

## Lightning-rewritten instructions

OS-Atlas Linux, macOS, AndroidWorld and FineWeb "instructions" are raw accessibility or HTML
labels. ScreenSpot-v2 and real users phrase targets as short instructions. NVIDIA Nemotron 3.5
Lightning (30B-A3B) rewrites a deterministic 60% of those labels. UIBert's natural expressions
pass through unchanged.

- **Model and endpoints:**
  - **NVIDIA-hosted API** (`https://integrate.api.nvidia.com/v1`, model
    `nvidia/nemotron-3.5-lightning-30b-a3b`): 8,028 rows. The hosted API doesn't report which
    checkpoint serves it. Its build.nvidia.com model card describes the official NVFP4
    checkpoint (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`).
  - **The official checkpoint `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`** on
    vllm-metal on a Mac: 1,003 rows. We added it after the hosted API rate-limited us. For
    the last ~13 minutes it ran alongside the final hosted invocation, working from the
    other end of the queue (`--reverse`). Before sending anything, `rewrite.py` checks the
    local server's `GET /models` and refuses community or quantized checkpoints.
  - **No model** for 353 rows, removed by `rule:is_numeric_label`.
- **Lightning never sees the screenshot.** Each target goes to it as one JSON line built from
  manifest metadata:
  - the label and the platform;
  - a control noun mapped from `data_type`;
  - a 3×3 grid position and a size bucket;
  - up to 8 neighboring labeled targets with their relative direction;
  - the requested style.
- **Selection and style, both by sha256 of the row id.**
  - Selected: `int(sha256(id)) % 100 < 60`.
  - Style, from `int(sha256("style:" + id)) % 100`: functional ~50% ("open more options"),
    descriptive ~30% ("the Settings button"), positional ~20% ("the Share button at the top
    right").
- **Request settings:** temperature 0.7, top_p 0.95, max_tokens 4096, thinking off
  (`enable_thinking: false`), `response_format: json_object`, up to 20 rows per request.
- **Validation** (`data_prep/rewrite/validate.py`):
  - Each reply is checked against a JSON shape, then linted for:
    - invented colors, icon shapes, quotes or element nouns;
    - contradicted neighbor directions;
    - answers shifted onto another item;
    - copied style words and grid tokens;
    - look-alike ambiguity.
  - Failed rows are re-asked, up to 3 attempts per invocation. Rows that still fail become
    `llm_error`.
- **Counts** (from `lightning_rewrites/stats.json` and `apply_stats.json`):

| | Rows |
| --- | --- |
| Selected train targets | 9,384 (AndroidWorld 1,694, FineWeb 3,920, Linux 1,819, macOS 1,951) |
| Rewritten and kept | 8,620 (functional 4,272, descriptive 2,601, positional 1,747). 8,498 differ from the raw label; for 122, Lightning returned the label itself |
| Dropped by the numeric rule, never sent (removed from both training files) | 353 |
| Dropped by Lightning or failed validation (kept with the raw label) | 411: unintelligible 264, numeric 68, needs_screen 29, duplicate_neighbor 26, llm_error 24 |
| Valid on the first try (of 9,031 rows sent) | 89.5% (our gate was ≥ 95%: **not met**; failing rows were re-asked) |
| Drop rate, all drops except llm_error | 7.9% (gate < 25%: met) |
| Requests / total tokens, all invocations | 1,015 / 3,544,638 |

**Reproduce:** run the `rewrite.py` command in METHOD.md "Reproduce". Its request cache
(`cache.sqlite`) is not published, so a rerun sends new requests, and sampling at temperature
0.7 gives different rewrites. `rewrites.jsonl` is therefore the artifact. Then run `apply.py`,
which is deterministic.

**One stale line in METHOD.md:** its "Validation, retries and drops" section still says
`apply.py` removes every `keep=false` row. That was the rule before the audit. The released
files follow METHOD.md's "Human audit" section and `apply_stats.json`: only the 353
numeric-rule rows are removed, and model drops keep their raw label.

## Human audit

One person reviewed a random sheet (`review_sample.py`, seed 0) drawn from the first hosted
invocation (`20261017T204115Z-ef71ce`). Each card showed the screenshot crop with the target
box, the raw label and the rewrite. Verdicts are in
`lightning_rewrites/review_verdicts_v4_seed0.json`.

| Kept rewrites (50 cards) | Cards |
| --- | --- |
| Faithful | 42 |
| Unfaithful (a source ambiguity) | 1 |
| Unsure (couldn't be verified from the crop: hidden text, look-alikes, non-English) | 6 |
| Not judged | 1 |

- **Faithfulness:** 42/43 = **97.7%** of the cards that could be verified (gate ≥ 90%), or
  42/49 = **85.7%** if unsure cards count as not faithful.
- **Model drops** (16 cards): 3 were correct and 13 were wrong. Lightning dropped readable
  labels and file names as "unintelligible" or "numeric". So we don't trust model drops:
  those rows keep their raw label instead of being removed.
- **Scope:** the audit covered hosted-API rows only. The 1,003 rows from the local BF16
  checkpoint weren't audited separately.

## GRPO prompt pool

`grpo/round_1_pool.jsonl` holds the prompts of the released model's only GRPO round (round 1
of `training/night_loop.py`). A second round ran but wasn't kept, so its pool isn't
published.

- **Profile:** a random 4,000-row sample of the train rows of `train_lightning.jsonl` (seed 1;
  never dev or ScreenSpot). The starting model, the LoRA SFT adapter trained on
  `train_lightning.jsonl`, answered each row 8 times at temperature 1.0 through the
  `gui_grounding` NeMo Gym environment (reward 1 when the predicted point is inside the box),
  followed by `gym eval profile`.
- **Pool:** the rows it solved sometimes but not always (0 < pass rate < 1): 1,282 of the
  4,000. FineWeb 472, Linux 230, UIBert 220, AndroidWorld 183, macOS 177. The per-source
  always / sometimes / never counts are in `grpo/round_1_pool_stats.json`.
- **Instructions** are the rows' instructions in `train_lightning.jsonl`; 566 of the 1,282
  differ from the raw label. The Lightning terms in [Sources and licenses](#sources-and-licenses)
  apply to them.
- **`responses_create_params`** is the Gym request as the loop wrote it: the grounding prompt,
  temperature 0, at most 32 output tokens. Its `image_url` is a `file://` path to a
  pre-resized copy of the screenshot (at most 2,073,600 pixels) on our training machine.
  Point it at your own copy. `bbox`, `img_w` and `img_h` are in original-screenshot pixels,
  as in the manifest.

## Contamination handling (ScreenSpot-v2)

- **Never trained on:** ScreenSpot v1 or v2, GroundUI-18K, or Wave-UI's ScreenSpot rows.
  `rewrite.py` refuses any row whose source, id or image path mentions ScreenSpot or GroundUI,
  and `training/sft.py` refuses ScreenSpot rows.
- **Perceptual-hash dedup:** every training screenshot is checked against ScreenSpot-v2
  (step 3 above).
- **Near-blank ScreenSpot-v2 images:** the pHash step skips the 79 near-blank images (pHash
  collides on them), so `build/blank_dedup_recheck.json` checks them separately (step 3 above). It found no exact
  copies. Its similarity flags weren't used to drop anything.
- **28 ScreenSpot-v2 rows excluded from the test set instead.** ScreenSpot-v2 added 28 new
  desktop screenshots (20 Ubuntu at 2560×1440, 8 macOS at 2880×1800). They were captured on
  the same machines and desktop setups as OS-Atlas's Linux and macOS data, 1–3 weeks later:
  same wallpaper, same desktop files.
  - At least 9 of the 28 have a near-duplicate training screenshot.
  - Some test targets are labeled in training. For example, the test row "adjust font color"
    matches the training element "Font Color" at IoU 0.83.
  - The same app layouts recur across many training screenshots, so a dedup on the training
    side can't remove the leak.
  - We keep the Linux/macOS training data and drop those 28 rows from every number we
    report. **ScreenSpot-v2-clean** = 1,272 − 28 = 1,244 rows of revision `5efbb1f1b5`.
  - The 28 rows (file name and instruction) and the reason are in
    `screenspot_v2_excluded_rows.json`.
  - We haven't seen this overlap documented elsewhere. Anyone who trains on OS-Atlas
    Linux/macOS and reports ScreenSpot-v2 desktop numbers is affected.
- **FineWeb:** the web share is covered only by the same pHash dedup, which dropped 2 FineWeb
  screenshots. We didn't run the planned instruction-text plus box-IoU check against
  ScreenSpot-v2 web targets (`docs/research-notes.md` §5 in the GitHub repo).

## Intended use

- Training and evaluating small GUI click-grounding models. Each row is one target: a
  screenshot plus a click instruction, with the target's box.
- Ablations of synthetic and raw instructions: `lightning` vs `template` have the same rows,
  boxes and order.
- Checkpoint selection on `dev`.

**Not intended for:**

- Testing anything on ScreenSpot-v2 without the 28-row exclusion. Our desktop data and those
  rows share machines.
- Training models that identify or profile people. Screenshots can show names, usernames and
  page content from public websites. The labels are copied verbatim from OS-Atlas and can
  include personal data that was on screen. For example, some macOS rows' label is the
  capture machine's account email address.

## Limitations

- **Labels come from accessibility trees and HTML.**
  - Some boxes are on the wrong element, and the rewrite follows the label, so those rows
    stay wrong.
  - `data_type` is a control type, not text/icon, so this data has no icon/text split.
- **Lightning works from metadata only.**
  - If a label is ambiguous without the pixels, the rewrite can fit the label but not the
    boxed element.
  - Position is a coarse 3×3 bucket.
  - The neighbors are only the targets we sampled, at most 8 per screenshot.
- **The quality gate was missed.** 89.5% of rows were valid on the first try, against our
  95% gate. Rows that failed were re-asked and linted, but the lints are heuristics.
- **The audit is small** (50 kept rewrites and 16 drops, hosted rows only).
- **The mix is uneven.**
  - Mostly English. The rewrite prompt keeps non-English labels verbatim inside English
    instructions.
  - Desktop is Linux and macOS only (no Windows).
  - Web is one FineWeb archive.
  - The numeric-label rule shifts the training mix slightly away from numeric targets.
- **Same-machine overlap with ScreenSpot-v2 desktop** (above). The 28-row exclusion handles it
  on the test side.

## Reproduce

`RAW` holds the OS-Atlas files from [Getting the images](#getting-the-images) and the
ScreenSpot-v2 images; `WORK` is a scratch directory. The paths are placeholders.

```bash
# 1. sample (the archives are range-read from HF if they aren't under --raw)
python data_prep/sample_os_atlas.py --raw $RAW/os_atlas --out $WORK/os_atlas_20k
# 2. overlay scan
python data_prep/rewrite/overlay_scan.py --manifest $WORK/os_atlas_20k/manifest.jsonl --workers 14
# 3-4. overlay drop + pHash dedup vs ScreenSpot-v2 + split (rewrites --manifest in place, after a .bak copy)
python data_prep/finalize_manifest.py --manifest $WORK/os_atlas_20k/manifest.jsonl \
    --screenspot-images $RAW/screenspot_v2/screenspotv2_image \
    --overlay-flagged $WORK/os_atlas_20k/overlay_flagged.json
# 3b. blank-image recheck (the script's documented usage) -> blank_dedup_recheck.json
python data_prep/recheck_blank_dedup.py --manifest $WORK/os_atlas_20k/manifest.jsonl \
    --screenspot-images $RAW/screenspot_v2/screenspotv2_image \
    --stats $WORK/os_atlas_20k/manifest.stats.json
# 5. Lightning rewrites: the exact command of each invocation is in lightning_rewrites/METHOD.md
#    (its data/os_atlas_final/ is the manifest's directory, here $WORK/os_atlas_20k)
# 6. apply. Reconstructed from the inputs and settings in lightning_rewrites/apply_stats.json;
#    the command line itself wasn't recorded. Writes the training files next to the manifest
python data_prep/rewrite/apply.py --manifest $WORK/os_atlas_20k/manifest.jsonl \
    --rewrites $WORK/os_atlas_20k/rewrite_v4/rewrites.jsonl --splits train
# 7. GRPO pool: the profile and pool steps of round 1 of training/night_loop.py
```

**Code versions:**

- **Sampler (step 1).** The manifest's `group` values keep the file extension
  (`linux:<file>.png`), which matches `sample_os_atlas.py` at commit `63c3e5f`. Commit
  `87cb960` (current) drops the extension. Rerunning the current sampler therefore writes
  different `group` strings and a different sha256.
- **Overlay scan and finalize (steps 2–4).** `overlay_scan.py` and `finalize_manifest.py`
  haven't changed since `39e1eca`.
- **Rewrite (step 5).** The code is `d0d9d625`.

**Checking a rerun:** compare it with `manifest.jsonl` (sha256 `4ceb7ea3d7ac8e85…`). The
sampler arguments of the original run are recorded in `build/sampler_stats.json`, and the
settings and results of steps 2–4 in `build/manifest.stats.json`.

## Citation

If you use this data, please cite OS-Atlas and the datasets it builds on (UIBert, Rico,
AndroidWorld, FineWeb), and this dataset:

```bibtex
@article{wu2024atlas,
  title   = {OS-ATLAS: A Foundation Action Model for Generalist GUI Agents},
  author  = {Wu, Zhiyong and others},
  journal = {arXiv preprint arXiv:2410.23218},
  year    = {2024}
}
@misc{bai2021uibert,
  title         = {UIBert: Learning Generic Multimodal Representations for UI Understanding},
  author        = {Chongyang Bai and Xiaoxue Zang and Ying Xu and Srinivas Sunkara and Abhinav Rastogi and Jindong Chen and Blaise Aguera y Arcas},
  year          = {2021},
  eprint        = {2107.13731},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
@misc{guiassist2026data,
  title        = {GUI Assist grounding data: OS-Atlas slice + Nemotron Lightning instructions},
  author       = {{GUI Assist team (sherpas)}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/patrickballou/gui-assist-data}}
}
```

GUI Assist (team sherpas: Patrick, Kyle, Charlie), NVIDIA x Red Hat Small Models Hack,
Raleigh NC, 17–18 October 2026. <https://github.com/pballou/nvidia-hack-26>
