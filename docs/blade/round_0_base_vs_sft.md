# gui_grounding BLADE analysis, round 0: base Qwen3.5-2B vs SFT template adapter

OS-Atlas `dev` split, 969 rows, NeMo Gym `gui_grounding` env, written Sat 2026-10-17 ~19:40 EDT.
Produced with `scripts/blade_pair.py` (runbook: [README.md](README.md)) plus 30 screenshots checked by eye.
Checked again ~20:20 EDT: every number was recomputed from the raw rollouts. Two rows that had been flagged as
artifacts are boxed correctly and now count as model misses (see §1). The knob values are unchanged.

> **Status:** this recommendation was **not applied**, and it no longer can be for round 1. The night loop
> started Sat 19:38 from **sft-lightning**, which won dev 86.27 vs 85.86, using `DEFAULT_KNOBS` and tonight's
> extended limits (docs/runs.md). Round 1 read `knobs.json` when it started, minutes after launch, so it runs
> on the defaults.
> The weights below were derived from the template adapter, so treat them as reference only. The first knobs the
> loop can still take are for round 2, and they come from the round-1 BLADE pass ([README.md](README.md) §3).
> That pass re-derives the same 225 `exclude_ids` from `train_lightning.jsonl`. If round 1 isn't kept, the loop
> stops and knobs no longer matter.

## Executive summary

- **SFT adds +9.49 dev points: 76.37 → 85.86.** 118 rows fixed and 26 broken (exact McNemar p = 3e-15).
  Both runs parse 100% of replies, and no rows are masked or missing.
- **The gain is on desktop (+18.3) and mobile (+9.2). Web gains only +1.7** (fineweb 82.87 → 84.53).
- **About 29% of the SFT model's 137 failures are data artifacts, not model errors** (40 rows):
  - boxes that don't line up with the screenshot on 6 screenshots, confirmed by eye (28 failures);
  - instructions nothing on screen can ground, such as AX handles, D-Bus names and file-name alt text (9);
  - sliver boxes (3): two clipped at a section-crop edge, one at the top of a Chrome tab card.
  - Another 33 failures (24%) are points where the base and SFT models agree on the same wrong spot. These are
    mostly duplicate visible targets (the same link text twice on one page), plus some spatial-language errors
    both models share.
  - If we drop the 46 artifact rows, dev reads **79.96 → 89.49**. This is a diagnostic only: checkpoints are
    still chosen on the raw number.
- **Three real model weaknesses:**
  - uibert spatial and ordinal instructions: 10.4% model error, 7 regressions;
  - a "row-centre" click habit that SFT introduced on androidworld. 9 SFT misses sit at x = 568 on the 0–1000
    grid, but 6 of them are on artifact rows. 6 of androidworld's 7 regressions click mid-row (x 499–574);
  - linux icon-only toolbar buttons.
- **Knob recommendation (for the template adapter; superseded, see Status):**
  - `source_weights`: macos 1.0 → 0.5, androidworld 1.0 → 1.5, uibert 1.0 → 1.75. Linux and fineweb stay at
    1.0.
  - `exclude_ids`: [] → 225 train rows whose labels can't be grounded. All 225 are `split=train` rows of both
    `train_template.jsonl` and `train_lightning.jsonl`, and none is a dev id.
  - The loop doesn't wait for BLADE between round 0 and round 1, so round-1 knobs had to be in place
    before launch. Tonight's loop launched first (see "When to apply").

## Artifact inventory

| | BEFORE (base) | AFTER (SFT template) |
|---|---|---|
| Rollouts on the box | `/home/ubuntu/kyle/nvidia-hack-26/results/dev_base_qwen35.jsonl` | `/home/ubuntu/night_rehearsal/round_0/dev_r0/rollouts.jsonl` |
| Local copy (gitignored) | `runs/blade/base/dev_base_qwen35.jsonl` | `runs/blade/r0/night/round_0/dev_r0/rollouts.jsonl` |
| Model | served `base`, no adapter | served `grounder` = `/home/ubuntu/ckpt/sft-template/adapter` (rehearsal launch 20261017-175222) |
| Run | `gym eval run --agent gui_grounding_simple_agent --input /home/ubuntu/data/gym/os_atlas_dev.jsonl`, 1 repeat, T = 0, max_output_tokens 128 | same, `--concurrency 64 ++head_server.port=11010` (`round_0/dev_r0/command.txt`) |
| Rows / parsed / masked / failures sidecar | 969 / 969 / 0 / 0 | 969 / 969 / 0 / 0 |

- **Prompts:** identical in both runs (0 of 969 `input` blocks differ).
- **Reply format:** the base model answers with a JSON list (`[{"point_2d": [x, y], "label": ...}]`). SFT
  answers with a bare `{"point_2d": [x, y]}`. `common/grounding.py` parses both.
- **Supporting evidence used:**
  - the rehearsal's round-1 train profile, run with the same adapter (`round_1/profile_r1/`, 300 train rows,
    8 repeats at T = 1.0, 32 tokens) and its `pool_stats.json`;
  - the train files `/home/ubuntu/data/os_atlas_20k/train_{template,lightning}.jsonl`. They have the same
    18,299 ids in the same order.
- **Not available:** `gym eval compare` for this pair (the base run happened outside the loop), and repeats
  on dev. pass@k therefore equals pass@1, and the "sometimes-pass" evidence below comes from the train profile.

## Aggregate results

| Slice | n | base | SFT | Δ | fixed | broken | artifact rows | base, artifacts removed | SFT, artifacts removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **overall** | 969 | **76.37** | **85.86** | **+9.49** | 118 | 26 | 46 | 79.96 | 89.49 |
| os_atlas/linux | 176 | 68.75 | 87.50 | +18.75 | 37 | 4 | 4 | 70.35 | 88.37 |
| os_atlas/macos | 158 | 73.42 | 91.14 | +17.72 | 29 | 1 | 6 | 76.32 | 94.08 |
| os_atlas/androidworld | 139 | 70.50 | 81.29 | +10.79 | 22 | 7 | 14 | 78.40 | 90.40 |
| os_atlas/uibert | 134 | 78.36 | 85.82 | +7.46 | 17 | 7 | 0 | 78.36 | 85.82 |
| os_atlas/fineweb | 362 | 82.87 | 84.53 | +1.66 | 13 | 7 | 22 | 87.65 | 89.12 |
| desktop (linux + macos) | 334 | 70.96 | 89.22 | +18.26 | | | | | |
| mobile (androidworld + uibert) | 273 | 74.36 | 83.52 | +9.16 | | | | | |
| web (fineweb) | 362 | 82.87 | 84.53 | +1.66 | | | | | |

**Element roles with n ≥ 10:**
- push-button: 58.7 → 87.3
- toggle-button: 46.7 → 73.3
- AXStaticText: 73.4 → 93.0
- android TextView: 67.4 → 80.9
- android ImageView: 73.3 → 66.7, the only role that got worse

**By resolution:** the largest gains are on high-resolution desktop screenshots:
- 2560×1440: 50.0 → 77.1
- 2880×1800: 65.8 → 86.8
- 1440×900: 75.8 → 92.5
- 1920×1080 (fineweb): 82.9 → 84.5

**Where the 118 fixes come from:** of the base model's misses that SFT fixed:
- 22 of 29 macos fixes were near misses (within 2% of the image diagonal), so a precision gain;
- 26 of 37 linux fixes were far misses, so the base model didn't recognise the target.

## Workflow funnel (single-step grounding)

969 rows → reply parsed (base 969, SFT 969) → point lands in the box (base 740, SFT 832).

The SFT model's 137 misses break down as:
- 7 near misses (within 2% of the diagonal);
- 13 points inside a different annotated element;
- 40 rows whose label is an artifact;
- 33 consensus misses (base and SFT agree on the same wrong spot);
- 44 other far misses.

Formatting and parsing have no losses at all. Every loss is where the click lands.

## Task outcome buckets (base → SFT)

| Bucket | Rows | Note |
|---|---:|---|
| still pass | 714 | |
| fixed (0 → 1) | 118 | linux 37, macos 29, androidworld 22, uibert 17, fineweb 13 |
| broken (1 → 0) | 26 | androidworld 7, uibert 7, fineweb 7, linux 4, macos 1. By label: 20 far miss, 4 wrong element, 2 near miss. None is an artifact (the base model hit each box) |
| still fail | 111 | 57 of the 137 SFT failures sit on screenshots with 3 or more failing rows |

**Sometimes-pass evidence (train profile, same adapter, 8 samples at T = 1):**

| Source | Tasks | Mean pass rate | Always | Sometimes | Never |
|---|---:|---:|---:|---:|---:|
| androidworld | 45 | 0.94 | 37 | 8 | 0 |
| fineweb | 113 | 0.76 | 64 | 35 | 14 |
| linux | 52 | 0.81 | 32 | 18 | 2 |
| macos | 48 | 0.91 | 39 | 7 | 2 |
| uibert | 42 | 0.85 | 26 | 13 | 3 |

- 81 of the 300 rows (27%) are GRPO candidates (0 < pass rate < 1).
- These are rows the adapter was trained on, so the pass rates are higher than on dev. The gap is widest on
  androidworld: 0.94 on train vs 81.3% on dev.

## Failure taxonomy (SFT model, 137 failed rows)

One primary label per row (rules in the `scripts/blade_pair.py` docstring). The BLADE code is in brackets.

| Label | Rows | By source | BLADE code |
|---|---:|---|---|
| KG_far_miss | 44 | fineweb 15, linux 9, uibert 9, androidworld 7, macos 4 | KG (some UK: 20 are regressions) |
| CONSENSUS_miss | 33 | fineweb 17, linux 7, uibert 5, macos 3, androidworld 1 | mostly TI (ambiguous target), some shared KG |
| DA_confirmed | 28 | fineweb 15, androidworld 13 | DA |
| KG_wrong_annotated_element | 13 | uibert 4, androidworld 3, fineweb 3, linux 2, macos 1 | KG |
| DA_ungroundable_instruction | 9 | macos 5, linux 2, fineweb 2 | DA |
| KG_near_miss | 7 | linux 2, fineweb 2, androidworld 1, macos 1, uibert 1 | KG (precision) |
| DA_sub_precision_box | 3 | fineweb 2, androidworld 1 | DA / TI |
| FMT_unparsed | 0 | | |

Examples below use points on the 0–1000 grid as (x, y), and boxes as [x1, y1, x2, y2] on the same grid
unless marked px.

### 1. DA: boxes don't match the pixels (28 failures, 6 screenshots, all checked by eye)

- **`androidworld:2304de99-36bb-4054-a5a6-a5d75c3ad95b`** (8/8 rows fail): 7 of the 8 boxes sit about one list
  row below their text.
  - Row `:1` "net.gsantner.markor Version v2.10.9 (146)" is the exception. Its box [175, 156, 504, 195] covers
    that text, and the base model hits it (337, 167). SFT misses at (568, 175), so this row is a real
    regression (§4), not an artifact.
  - Example `:2` "Rate this app": the box [175, 198, 402, 221] covers empty space under the
    "net.gsantner.markor" line.
  - The base model clicks the real "Rate this app" text (290, 242). SFT clicks the same row at (568, 242).
- **`fineweb:0d9a46919c83619f0454a8943fff361d_section_2_original`** (6/6 fail): 5 of the 6 boxes come from a
  different page layout.
  - Row `:1` "Contact CLAS Dean's Office" is the exception. Its box is the matching footer link "Dean's Office
    Administration", which the base model hits (552, 543). SFT clicks the "Contact" menu (484, 84), so this
    row counts as a model miss (a broken row), not an artifact.
  - Example `:4` "SEARCH": the box [257, 11, 525, 41] is on the site title. Both models click the real
    SEARCH button (804, 22).
  - Example `:5` "CLAS Undergraduate Programs Office": the box is the top-right corner. Both models click the
    real footer link (230, 640).
- **`fineweb:1c6eaa5b…_section_1_original`** (5 failures): the archive sidebar boxes are shifted one line down.
  - Example `:1` "June 2024": the box sits on "May 2024". Both models click "June 2024" (675, 812).
- **`fineweb:8a1091f8…_section_1_original`** (5 failures): six photo thumbnails labelled only by alt text
  ("2018 USA Seminar and Workshop", …). Both models stamp the same thumbnail for every row; SFT clicks
  (732, 801) each time. This also produces the fineweb x = 732 / y = 800 attractor.
- **`androidworld:1bfd2d97…`** rows `:0 :1 :2 :5`: the instruction is a contributor's name, but the box is the
  flag icon next to it.
  - Example `:0` "Gozzwip": the box is [58, 453, 136, 488]. The base model clicks the name (277, 482); SFT
    clicks (568, 483).
- **`androidworld:c70161b7…`** rows `:2 :3 :4`, a Chrome tab switcher:
  - `:2` ", tab" is an empty tab title.
  - `:3` "Clock error" and `:4` "Close Clock error tab" have 13 px and 24 px strips (5 and 10 units on the
    0–1000 grid of a 2,400 px tall screen), clipped at the top of a card.

### 2. DA: the instruction or box can't be grounded (12 failures)

- **Ungroundable instructions:**
  - `macos:20240905_133328_screenshot:5` "<AXUIElement 0x2876f1d40> {pid=1697}". The instruction is an
    accessibility handle.
  - `linux:output_20240912_143239_original_screenshot:0` ":1.21/StatusNotifierItem", a tray icon named by its
    D-Bus address. SFT clicks the tray icon next to it (934, 9).
  - `fineweb:477d0785…_section_2_original:2` "20220112-SMA-0862", an image file name used as alt text.
  - **Some of these rows pass by memorised position,** for example the "fixed" row
    `linux:…143239_original_screenshot:4` ":1.72/StatusNotifierItem". That is exactly the signal GRPO should
    not reinforce.
- **Sub-precision boxes:**
  - `fineweb:1d9c5449…_section_3_original:1` "turf farm Winchester", box [230, 0, 386, 3] px: a 3 px strip
    where the section crop cut the element off.
  - `fineweb:ab075a8c…_section_3_original:6` "ENTERPRISE RESOURCE PLANNING SOFTWARE", box
    [1348, 0, 1617, 4] px.

### 3. CONSENSUS_miss: both models click the same wrong spot (33)

This bucket is mixed: mostly a label/verifier ambiguity (TI), sometimes a weakness both models share (KG).

- **Duplicate visible target (TI):**
  - `linux:output_20240912_145746_original_screenshot:2` "Hyperlink": the box is the Insert-Hyperlink
    toolbar icon. Both models click the visible "Hyperlink" entry in the Styles sidebar.
  - `fineweb:730f486f…_section_1_original:2` "Search": the box is the sidebar GO button. Both models click the
    header search control.
  - On the same page, `:1` "Company News" appears three times (heading, breadcrumb, post category). That row is
    labelled `KG_far_miss`, not consensus, because the two models picked different copies.
- **Shared spatial-language error (KG):** `uibert:54082:0` "select the third option from the down". The box is
  SEATED BARBELL MILITARY PRESS, the third from the bottom. Both models click BARBELL SHRUG BEHIND THE BACK,
  the third from the top (568, 300).

### 4. KG: far misses and wrong elements (57), including most regressions

- **Androidworld "row-centre" habit, introduced by SFT (UK):** on list rows SFT clicks the centre of the text
  column instead of the text itself.
  - x ≈ 568 is (175 + 961) / 2. On these screens, list-row text starts at x = 175, and 961 is the most common
    right box edge in androidworld train.
  - 11 SFT points fall at x = 568 and 9 of them miss. 6 of those 9 misses are on confirmed-artifact rows, so
    only 3 are model errors: `1bfd2d97…:3`, `2304de99…:1` and `71b3f7c1…:2`. The base model never clicks
    x = 568 on any of the 139 androidworld rows.
  - 6 of androidworld's 7 regressions click mid-row (SFT x 499–574) where the base model clicked the text:
    `1bfd2d97…:3`, `2304de99…:1`, `71b3f7c1…:1/:2`, `c70161b7…:0` and `f6afdcf0…:2`. The 7th is the
    "Close tab" case below.
  - `androidworld:71b3f7c1-e347-4aa3-af7f-1e6f7b1a24cc:1` "Does not repeat" (broken): the box is the tight
    text [175, 381, 469, 407]. The base model clicks the text (319, 392); SFT clicks empty row space (574, 393).
    `:2` "Add reminder" is the same case (568, 825).
  - `androidworld:f6afdcf0…:2` "Device • 2 contacts" (broken, near miss): SFT clicks (500, 206), just right of
    the box [0, 177, 456, 234].
  - On real Android the row is often clickable, but the verifier only rewards the box. Since the base model
    already did this right, it's a reliability gap that GRPO with the point-in-box reward should fix directly.
- **uibert spatial and ordinal language (KG + UK):**
  - `uibert:62394:0` "click on first icon at the top right corner" (broken): the box is the first of two
    top-right icons. The base model hits it (843, 69); SFT clicks the rightmost icon (952, 71).
  - `uibert:39605:1` "click on icon on the left side of live" (broken): SFT clicks (841, 62), right of the box
    [664, 33, 781, 98].
  - `uibert:28014:0` "click the profile icon" (broken): SFT clicks a user's avatar photo (56, 452) instead of
    the bottom-bar profile tab (base 898, 902).
- **Linux icon-only toolbar buttons (KG):**
  - `linux:…145746_original_screenshot:7` "Fill Format Mode": SFT clicks a different toolbar icon (532, 102).
  - `linux:…145746_original_screenshot:4` "Sort Ascending" (broken, near miss): SFT clicks the neighbouring
    icon (316, 75), box [321, 63, 334, 88].
  - `linux:…150300_original_screenshot_sub1:4` "Animation" (broken, wrong element): SFT lands on the adjacent
    sidebar deck button (984, 363).
  - Linux "Off"/"On" status labels (`…163241_original_screenshot_sub1:2/3`, broken) show the same row-centre
    habit as androidworld (SFT x = 175 for both).
- **Ambiguous "close" target:** `androidworld:c70161b7…:5` "Close tab" (broken). SFT clicks the X on the
  "Privacy error" card (432, 410) instead of the untitled tab's X (432, 697, which the base model hits).

## Which clusters are data artifacts and which are model weaknesses

| Cluster | Rows (SFT failures) | Verdict | Evidence |
|---|---:|---|---|
| Misaligned boxes on 6 screenshots | 28 | **Data artifact (box ≠ label)** | All 6 checked by eye. Both models click the right visible text. Listed by row id in `runs/blade/da_confirmed_dev.txt`. Two correctly boxed rows on these screens (`2304de99…:1`, `0d9a…:1`) count as model misses |
| Ungroundable instruction (AX handle, D-Bus name, value literal, file-name alt text) | 9 | **Data artifact** | Rule-based. Any passes come from memorised positions |
| Sub-precision box (< 6 px or < 0.6% of the image side) | 3 | **Data artifact / task issue** | Rule-based. Two clipped at a section-crop edge, one a strip at the top of a tab card |
| Consensus miss, duplicate visible target (fineweb 17, linux 7, macos 3, androidworld 1) | 28 | **Task/label ambiguity, not something to train toward** | Base and SFT agree within 1.5% of the diagonal. 3 checked by eye, all another instance of the same text |
| Consensus miss, uibert | 5 | **Model weakness** | Spatial instructions. 1 checked by eye: label correct, both models wrong in the same way |
| Androidworld row-centre habit (x ≈ 568) | 9 misses in the x = 568 bucket, 3 of them on correctly labelled rows. Plus regressions at x = 574 / 500 / 499: 6 of androidworld's 7 regressions in all | **Model weakness introduced by SFT (UK)** | The base model clicked the text and passed |
| uibert spatial/ordinal language | 14 (far miss 9, wrong element 4, near miss 1) | **Model weakness (KG)** | Labels correct in the 4 checked by eye. 7 regressions vs base |
| Linux model errors, mostly icon-only toolbar buttons | 13 | **Model weakness (KG)** | Labels correct in the 4 checked by eye. Neighbouring-icon confusion |

## Recommendation for the overnight loop (knobs.json only)

**Not applied. Superseded for tonight:** see the Status note at the top. The loop runs from sft-lightning, and
round 1 already uses `DEFAULT_KNOBS`. Only `source_weights` and `exclude_ids` change. Every other knob keeps
its `DEFAULT_KNOBS` value from `training/night_loop.py`.

```diff
 {
   "pool_max": 2000, "num_generations": 6, "grpo_max_steps": 200, "round_time_budget_min": 75, "format_reward": 0.1,
   "source_weights": {
     "os_atlas/linux": 1.0,
-    "os_atlas/macos": 1.0,
+    "os_atlas/macos": 0.5,
-    "os_atlas/androidworld": 1.0,
+    "os_atlas/androidworld": 1.5,
-    "os_atlas/uibert": 1.0,
+    "os_atlas/uibert": 1.75,
     "os_atlas/fineweb": 1.0
   },
-  "exclude_ids": []
+  "exclude_ids": [ 225 train ids, full list at the end of this file ]
 }
```

The file is generated as `runs/blade/r0/knobs_proposed.json` by
`python3 -I scripts/blade_pair.py analyze --round 0 --knobs default`. **Don't upload it tonight** (see
Status). The script picks up `runs/blade/da_confirmed_dev.txt` by itself. It also picks up the train file
named in `runs/blade/r0/night/loop.json`, which is the rehearsal's `manifest.jsonl`; that yields the same 225
ids.

**Weight rule** (from the script):

    weight = clip(round_to_0.25(err_source / err_all), 0.5, 2.0), plus 0.25 if the source has >= 5 regressions

Here `err` is the SFT model's model-attributable dev error (FMT and KG labels only), counted on rows that aren't
artifacts. The overall rate is 6.9% (64 errors on 923 rows that aren't artifacts).

### source_weights

- **macos 0.5:**
  - Model error 4.0%, the lowest at 0.57× the overall rate. Dev is 91.1 raw and 94.1 with artifacts removed.
  - Only 1 regression.
  - Only 7 of 48 train rows are sometimes-pass.
  - A macos prompt buys the least in a GRPO step budget of about 1,600 prompt visits per round (200 steps × 8
    prompts).
- **androidworld 1.5:**
  - The raw dev number is the worst (81.3), but 14 of its 26 failures are artifacts (13 confirmed by eye, 1
    sub-precision box).
  - The rest is 8.8% model error (1.27×) plus 7 regressions: 1.25 by the rule, +0.25 for the regressions.
  - 6 of the 7 regressions are the SFT row-centre habit, a reliability gap that the point-in-box reward
    penalises directly. The 7th is the "Close tab" wrong-card X.
  - Androidworld has few GRPO candidates: 8 of 45 rows are sometimes-pass, and only macos has fewer (7 of 48).
    At weight 1.0 it would be about 10% of the pool.
- **uibert 1.75:**
  - The highest model error, 10.4% (1.51×), on clean labels (no artifact rows): 1.5 by the rule, +0.25 for 7
    regressions.
  - The failures are spatial and ordinal language. 13 of 42 train rows are sometimes-pass, so GRPO has
    signal to work with.
- **linux 1.0 (no change):**
  - 7.6% model error (1.09×) rounds to 1.0. It would take 1.125× to round up to 1.25.
  - Icon-recognition errors are real, but linux already has the highest sometimes-pass share (18 of 52), so it
    keeps a large pool share without an upweight.
- **fineweb 1.0 (no change):**
  - 5.9% model error (0.85×) rounds to 0.75, then +0.25 for 7 regressions.
  - Its biggest bucket, 17 consensus misses on duplicate link text, is label ambiguity. We shouldn't upweight
    noise there, and web is the demo surface, so we don't downweight it either.

**Expected pool composition:** from the rehearsal profile's sometimes-pass rates scaled to the 4,000-row
profile, 1,074 candidates become a pool of about 1,222 (still under `pool_max` 2,000, so the weights act as
duplication and subsampling).

| Source | Default share | Proposed share |
|---|---:|---:|
| fineweb | 41% | 36% |
| linux | 22% | 19% |
| uibert | 18% | 28% |
| androidworld | 10% | 13% |
| macos | 9% | 4% |

### exclude_ids

225 train ids. `train_template.jsonl` and `train_lightning.jsonl` share ids, so the list works whichever SFT
wins. Matches are tested on `instruction_original`.

The list was checked against both files. All 225 ids are `split=train` rows in each, and none is among the 969
dev ids. The same rules run on `train_lightning.jsonl` (the live loop's train file) or on the 18,652-row
`manifest.jsonl` produce exactly this set.

| Rule | Rows | Example |
|---|---:|---|
| linux D-Bus name | 85 | ":1.21/StatusNotifierItem" |
| macos AX handle | 55 | "<AXUIElement 0x299dcc570> {pid=1697}" |
| value literal: macos 7, fineweb 2 | 9 | "True", "False", "nan" |
| fineweb file-name slug | 10 | "bulleteers-dam-run-3249" (8 near-identical gallery thumbnails), "catnip-724539-640.jpg" |
| sub-precision box: fineweb 57, androidworld 5, linux 3, uibert 1 | 66 | 43 of the fineweb rows are strips at most 5.4 px tall (0.5% of 1,080 px) at the top or bottom crop edge |

**Why exclude them:**
- These rows can only be learned by memorising a position. On dev the same rules cover 17 rows, and 12 of
  them fail with SFT.
- In the rehearsal profile one of them, `fineweb:5bcc75ec…_section_7_original:0` "Divas Half Marathon" (box
  [505, 0, 641, 5] px, text clipped off), was sometimes-pass (0.75) and **did enter the GRPO pool**. That's
  the case this knob prevents.
- The cost is about 1.2% of train rows.

**Dev ids are deliberately not in exclude_ids:**
- `night_loop.load_train_rows` refuses dev ids, so dev rows never reach the pool and listing them would do
  nothing.
- The 46 dev artifact rows feed only the artifact-removed numbers and the weight rule, where they're left out
  of each source's error rate.

**No profile-based exclusions this round:** the 300-row rehearsal profile has no screenshot whose 2 or more
profiled rows are all never-pass.

### When to apply

- **The loop doesn't wait for BLADE between round 0 and round 1.** `run()` calls `baseline()` and then
  `round(1)` straight away, and `round(1)` reads `knobs.json` at its start. So round-1 knobs have to be in
  `/home/ubuntu/night/knobs.json` **before the loop launches**. In the rehearsal, round 0 lasted only 2 min
  (17:52:22 → 17:54:24).
- **Tonight that window has passed.** The loop launched Sat 19:38 from sft-lightning with `DEFAULT_KNOBS`
  (docs/runs.md). Round 1 runs on those defaults whatever `knobs.json` says now.
- **Next chance: the round-1 wait.** It happens only if round 1 is kept (dev +0.5 or more) and sets no
  `stop_reason`. Run the README §3 procedure with `R=1`. BEFORE is `round_0/dev_r0` (sft-lightning) and AFTER
  is `round_1/dev_r1`, so the weights are recomputed for the adapter that is actually training. The same 225
  rule-based `exclude_ids` get added from `train_lightning.jsonl`. Don't upload this file's template-based
  weights.
- **The loop won't overwrite `knobs.json`.** It writes defaults only if the file is missing, and
  `archive_previous()` doesn't move it.
- **How to write it:** atomically, as in the README.
- **How to check it took:** `round_<r>/knobs_used.json` of the next round should show the new values and
  `"notes": []`.

**The Lightning SFT did win dev** (86.27 vs 85.86) and became the round-0 adapter. The loop started before the
README's round-0 re-run on its dev eval could be done, and the round-1 pass above replaces that re-run.

### Not for tonight (eval sets, prompts and the verifier are frozen overnight)

- Report the dev artifacts in the final BLADE write-up. Keep SSv2-clean as the headline.
- For the next SFT data build:
  - add the ungroundable-instruction and sub-precision filters to `data_prep` (training only);
  - consider dropping fineweb thumbnails that have only alt text.

## Reproducibility

```bash
cd /Users/pballou/projects/nvidia-hack-26
python3 -I scripts/blade_pair.py fetch --round 0 --night-dir /home/ubuntu/night_rehearsal   # read-only; also fetches the base run
python3 -I scripts/blade_pair.py analyze --round 0 --knobs default \
    --train-file runs/blade/data/train_template.jsonl \
    --profile runs/blade/rehearsal/rehearsal_r1/profile_r1/rollouts_reward_profiling.jsonl \
    --pool-stats runs/blade/rehearsal/rehearsal_r1/pool_stats.json
python3 -I scripts/blade_pair.py images --round 0 --n 24        # 24 screenshots (6 more via --only --screens …)
```

- **Outputs** (all gitignored): `runs/blade/r0/{analysis.md, analysis.json, rows.jsonl, knobs_proposed.json,
  knobs_diff.json, images/}`.
- **Confirmed artifact rows/screenshots:** `runs/blade/da_confirmed_dev.txt`, listed below so it can be
  recreated.
- **Runbook check:** the same commands on the rehearsal's round 1 (`--round 1 --night-dir
  /home/ubuntu/night_rehearsal --out runs/blade/rehearsal/r1`) report dev 85.86 → 85.96 (5 fixed, 4 broken,
  not kept). That matches its `summary.json`.

Confirmed dev artifacts, as row ids. The two whole-screen entries were replaced with their bad rows, leaving out
`2304de99…:1` and `0d9a…:1`, whose boxes are correct. `analyze` now prints a `CHECK:` line if a confirmed row
is one the BEFORE or reference model passes.

```text
androidworld:2304de99-36bb-4054-a5a6-a5d75c3ad95b:0
androidworld:2304de99-36bb-4054-a5a6-a5d75c3ad95b:2
androidworld:2304de99-36bb-4054-a5a6-a5d75c3ad95b:3
androidworld:2304de99-36bb-4054-a5a6-a5d75c3ad95b:4
androidworld:2304de99-36bb-4054-a5a6-a5d75c3ad95b:5
androidworld:2304de99-36bb-4054-a5a6-a5d75c3ad95b:6
androidworld:2304de99-36bb-4054-a5a6-a5d75c3ad95b:7
fineweb:0d9a46919c83619f0454a8943fff361d_section_2_original:0
fineweb:0d9a46919c83619f0454a8943fff361d_section_2_original:2
fineweb:0d9a46919c83619f0454a8943fff361d_section_2_original:3
fineweb:0d9a46919c83619f0454a8943fff361d_section_2_original:4
fineweb:0d9a46919c83619f0454a8943fff361d_section_2_original:5
androidworld:1bfd2d97-e82f-4928-a1a2-3cfd6d1e3a54:0
androidworld:1bfd2d97-e82f-4928-a1a2-3cfd6d1e3a54:1
androidworld:1bfd2d97-e82f-4928-a1a2-3cfd6d1e3a54:2
androidworld:1bfd2d97-e82f-4928-a1a2-3cfd6d1e3a54:5
androidworld:c70161b7-b9e0-451d-9ca4-8e672e975858:2
androidworld:c70161b7-b9e0-451d-9ca4-8e672e975858:3
androidworld:c70161b7-b9e0-451d-9ca4-8e672e975858:4
fineweb:1c6eaa5b111446852f77e165b72fd551_section_1_original:0
fineweb:1c6eaa5b111446852f77e165b72fd551_section_1_original:1
fineweb:1c6eaa5b111446852f77e165b72fd551_section_1_original:2
fineweb:1c6eaa5b111446852f77e165b72fd551_section_1_original:5
fineweb:1c6eaa5b111446852f77e165b72fd551_section_1_original:7
fineweb:8a1091f826da3fd97fbfe18df573851a_section_1_original:0
fineweb:8a1091f826da3fd97fbfe18df573851a_section_1_original:1
fineweb:8a1091f826da3fd97fbfe18df573851a_section_1_original:2
fineweb:8a1091f826da3fd97fbfe18df573851a_section_1_original:4
fineweb:8a1091f826da3fd97fbfe18df573851a_section_1_original:5
fineweb:8a1091f826da3fd97fbfe18df573851a_section_1_original:6
```

<details><summary>exclude_ids (225 train ids)</summary>

```json
["androidworld:307df8e9-d3ec-46e5-bf78-9122a4a8c0c2:3", "androidworld:56930fde-4d36-4c0a-809e-97a5349ed9f5:6", "androidworld:7cb6e323-843b-4470-8c7f-e6908903c143:7", "androidworld:9bf4ba1d-3cd8-4971-9344-66e29681bf1c:1", "androidworld:f68d5c3a-cc29-438b-9789-d435d2375bf0:4", "fineweb:072dd6b857788bfe5296298833d4d273_section_3_original:0", "fineweb:0777a8e591c9ae172405e8bec5c623a6_section_5_original:1", "fineweb:09e9a8eeb59def87ab763e763d45dc42_section_2_original:4", "fineweb:1282cb89dc28af2cf052aedd7a286e76_section_2_original:0", "fineweb:1282cb89dc28af2cf052aedd7a286e76_section_2_original:1", "fineweb:1282cb89dc28af2cf052aedd7a286e76_section_2_original:2", "fineweb:1282cb89dc28af2cf052aedd7a286e76_section_2_original:3", "fineweb:1282cb89dc28af2cf052aedd7a286e76_section_2_original:4", "fineweb:1b252c0e774f0e2262ad906f0f68fc6c_section_2_original:3", "fineweb:1b252c0e774f0e2262ad906f0f68fc6c_section_2_original:6", "fineweb:1d70b9d3adbc7aaa8a432c074eaca409_section_4_original:0", "fineweb:1ed3b4a46b917b4fa304fd2dd780de4c_section_2_original:0", "fineweb:1ed3b4a46b917b4fa304fd2dd780de4c_section_2_original:4", "fineweb:1f557dd179f7a61b6a4a0a2d10709473_section_1_original:6", "fineweb:2582cd0949abe6b37634b37f5531b9ed_section_4_original:2", "fineweb:2582cd0949abe6b37634b37f5531b9ed_section_4_original:3", "fineweb:2582cd0949abe6b37634b37f5531b9ed_section_4_original:5", "fineweb:25c72c62d9b65e1bcb84031527b5357e_section_6_original:1", "fineweb:25c72c62d9b65e1bcb84031527b5357e_section_6_original:4", "fineweb:3580cb3c52980a9a01e636a6d29f3c09_section_3_original:0", "fineweb:3580cb3c52980a9a01e636a6d29f3c09_section_3_original:1", "fineweb:3580cb3c52980a9a01e636a6d29f3c09_section_3_original:2", "fineweb:3580cb3c52980a9a01e636a6d29f3c09_section_3_original:3", "fineweb:3580cb3c52980a9a01e636a6d29f3c09_section_3_original:4", "fineweb:3580cb3c52980a9a01e636a6d29f3c09_section_3_original:5", "fineweb:3580cb3c52980a9a01e636a6d29f3c09_section_3_original:6", "fineweb:3580cb3c52980a9a01e636a6d29f3c09_section_3_original:7", "fineweb:378b772ec094be440876960db4b73329_section_3_original:1", "fineweb:3a3637c3356f572c07ef220a839a6b6e_section_3_original:6", "fineweb:3f46cc12d92e2a6b84c8108c01220c5e_section_3_original:4", "fineweb:49a55bcb65e2d600c9fe1abcfe5ab8b8_section_3_original:0", "fineweb:57873a9328eddf57a1559f4a12d642c1_section_2_original:3", "fineweb:5baf52ff492310f48ef534d98c64e481_section_1_original:1", "fineweb:5bcc75ec36f20e445f5c406539cc5626_section_7_original:0", "fineweb:6424c798725416d41ba717b93a1a3c28_section_3_original:0", "fineweb:6d1ebbf26fa403b11f8e3ec8a5702226_section_4_original:3", "fineweb:755c32435fca54a7b3b4e291f188afb9_section_2_original:0", "fineweb:76c4e5542291e6a66f0f8030ca2bad96_section_3_original:0", "fineweb:7e20c1b23783e333180e64a2078c97e5_section_3_original:2", "fineweb:8394a31013badecd7905ab33db5bc592_section_2_original:1", "fineweb:87ea49adebe43d438405ba5475dcf3be_section_1_original:1", "fineweb:8fe97e9e7681d8d2a2f96eca3af2e9a8_section_3_original:0", "fineweb:a44c0910e30ba34741c0e9973a25ffeb_section_4_original:4", "fineweb:b2a70bd054572f6f25e45aaa7472a41f_section_1_original:2", "fineweb:b2cc735af1f4369391d07dd9457c51b7_section_2_original:0", "fineweb:b8c600f65133e9f0ad2ba860d842375e_section_2_original:1", "fineweb:be2ace9ce27dce41ba7b468ab5e6ac15_section_4_original:0", "fineweb:be7b47d2e5c06968e8b2f6339cf3510f_section_2_original:0", "fineweb:be7b47d2e5c06968e8b2f6339cf3510f_section_2_original:1", "fineweb:be7b47d2e5c06968e8b2f6339cf3510f_section_2_original:2", "fineweb:be7b47d2e5c06968e8b2f6339cf3510f_section_2_original:3", "fineweb:be7b47d2e5c06968e8b2f6339cf3510f_section_2_original:4", "fineweb:be7b47d2e5c06968e8b2f6339cf3510f_section_2_original:5", "fineweb:be7b47d2e5c06968e8b2f6339cf3510f_section_2_original:6", "fineweb:c0d13d14d6303f639cab93ce224b8075_section_1_original:3", "fineweb:c53f96cdc09810a75c12b633a65897c3_section_2_original:2", "fineweb:d05d93dce4260880f403dcfb65aa4d71_section_2_original:0", "fineweb:d05d93dce4260880f403dcfb65aa4d71_section_2_original:1", "fineweb:d05d93dce4260880f403dcfb65aa4d71_section_2_original:2", "fineweb:d05d93dce4260880f403dcfb65aa4d71_section_2_original:3", "fineweb:d05d93dce4260880f403dcfb65aa4d71_section_2_original:4", "fineweb:d9919a6ac58ed2ade5be8c1206915717_section_1_original:5", "fineweb:e53fa1245673434bf2e65e99af26425d_section_1_original:0", "fineweb:e53fa1245673434bf2e65e99af26425d_section_1_original:1", "fineweb:e53fa1245673434bf2e65e99af26425d_section_1_original:3", "fineweb:e6e3134b4749578e88a8ab14c21e1a50_section_1_original:1", "fineweb:e6e3134b4749578e88a8ab14c21e1a50_section_1_original:4", "fineweb:e6e3134b4749578e88a8ab14c21e1a50_section_1_original:5", "fineweb:e6e3134b4749578e88a8ab14c21e1a50_section_1_original:7", "linux:output_20240911_171026_original_screenshot_sub1:1", "linux:output_20240911_171026_original_screenshot_sub1:5", "linux:output_20240911_171822_original_screenshot:2", "linux:output_20240911_171822_original_screenshot:6", "linux:output_20240911_171822_original_screenshot_sub1:1", "linux:output_20240911_171822_original_screenshot_sub1:5", "linux:output_20240912_113625_original_screenshot_sub1:1", "linux:output_20240912_113714_original_screenshot_sub1:4", "linux:output_20240912_113743_original_screenshot_sub1:4", "linux:output_20240912_113814_original_screenshot_sub1:5", "linux:output_20240912_114221_original_screenshot_sub1:1", "linux:output_20240912_135805_original_screenshot_sub1:1", "linux:output_20240912_135805_original_screenshot_sub1:2", "linux:output_20240912_143552_original_screenshot:2", "linux:output_20240912_143552_original_screenshot:7", "linux:output_20240912_143552_original_screenshot_sub1:1", "linux:output_20240912_143552_original_screenshot_sub1:2", "linux:output_20240912_143643_original_screenshot:5", "linux:output_20240912_143800_original_screenshot_sub1:3", "linux:output_20240912_143805_original_screenshot:0", "linux:output_20240912_143810_original_screenshot:2", "linux:output_20240912_143815_original_screenshot_sub1:4", "linux:output_20240912_143830_original_screenshot:3", "linux:output_20240912_143901_original_screenshot_sub1:3", "linux:output_20240912_144001_original_screenshot:1", "linux:output_20240912_144001_original_screenshot:7", "linux:output_20240912_145659_original_screenshot_sub1:2", "linux:output_20240912_145808_original_screenshot_sub1:5", "linux:output_20240912_145808_original_screenshot_sub1:6", "linux:output_20240912_145821_original_screenshot_sub1:7", "linux:output_20240912_145931_original_screenshot:2", "linux:output_20240912_145931_original_screenshot:3", "linux:output_20240912_145948_original_screenshot_sub1:7", "linux:output_20240912_150010_original_screenshot:4", "linux:output_20240912_150010_original_screenshot_sub1:3", "linux:output_20240912_150017_original_screenshot:4", "linux:output_20240912_150027_original_screenshot_sub1:5", "linux:output_20240912_150027_original_screenshot_sub1:7", "linux:output_20240912_150035_original_screenshot_sub1:4", "linux:output_20240912_150046_original_screenshot_sub1:4", "linux:output_20240912_150046_original_screenshot_sub1:5", "linux:output_20240912_150131_original_screenshot_sub1:4", "linux:output_20240912_150155_original_screenshot:6", "linux:output_20240912_150217_original_screenshot_sub1:2", "linux:output_20240912_150217_original_screenshot_sub1:3", "linux:output_20240912_150254_original_screenshot_sub1:5", "linux:output_20240912_150306_original_screenshot_sub1:0", "linux:output_20240912_150316_original_screenshot:6", "linux:output_20240912_150316_original_screenshot_sub1:4", "linux:output_20240912_152917_original_screenshot_sub1:0", "linux:output_20240912_152924_original_screenshot_sub1:0", "linux:output_20240912_152924_original_screenshot_sub1:2", "linux:output_20240912_153004_original_screenshot_sub1:4", "linux:output_20240912_153004_original_screenshot_sub1:6", "linux:output_20240912_153021_original_screenshot_sub1:0", "linux:output_20240912_153021_original_screenshot_sub1:5", "linux:output_20240912_153042_original_screenshot:5", "linux:output_20240912_153042_original_screenshot_sub1:2", "linux:output_20240912_153042_original_screenshot_sub1:4", "linux:output_20240912_153123_original_screenshot:7", "linux:output_20240912_153123_original_screenshot_sub1:5", "linux:output_20240912_153123_original_screenshot_sub1:6", "linux:output_20240912_153136_original_screenshot_sub1:1", "linux:output_20240912_153136_original_screenshot_sub1:3", "linux:output_20240912_153240_original_screenshot_sub1:4", "linux:output_20240912_153306_original_screenshot_sub1:1", "linux:output_20240912_153343_original_screenshot_sub1:1", "linux:output_20240912_153435_original_screenshot_sub1:0", "linux:output_20240912_153505_original_screenshot_sub1:4", "linux:output_20240912_153622_original_screenshot_sub1:2", "linux:output_20240912_153627_original_screenshot_sub1:7", "linux:output_20240912_153634_original_screenshot_sub1:0", "linux:output_20240912_153645_original_screenshot:5", "linux:output_20240912_153802_original_screenshot:3", "linux:output_20240912_153802_original_screenshot_sub1:4", "linux:output_20240912_153836_original_screenshot_sub1:0", "linux:output_20240912_153855_original_screenshot:3", "linux:output_20240912_154152_original_screenshot:2", "linux:output_20240912_154152_original_screenshot_sub1:1", "linux:output_20240912_154152_original_screenshot_sub1:6", "linux:output_20240912_155614_original_screenshot_sub1:0", "linux:output_20240912_160101_original_screenshot_sub1:4", "linux:output_20240912_160107_original_screenshot_sub1:2", "linux:output_20240912_160324_original_screenshot:5", "linux:output_20240912_160324_original_screenshot_sub1:4", "linux:output_20240912_160324_original_screenshot_sub1:5", "linux:output_20240912_160532_original_screenshot:2", "linux:output_20240912_161219_original_screenshot_sub1:7", "macos:20240904_150524_screenshot:6", "macos:20240904_154059_screenshot:1", "macos:20240904_154402_screenshot:5", "macos:20240904_154416_screenshot:3", "macos:20240904_184011_screenshot:3", "macos:20240904_184023_screenshot:5", "macos:20240904_184329_screenshot:6", "macos:20240904_184608_screenshot:6", "macos:20240905_110001_screenshot:3", "macos:20240905_110134_screenshot:1", "macos:20240905_110224_screenshot:0", "macos:20240905_110256_screenshot:5", "macos:20240905_110256_screenshot_sub0:4", "macos:20240905_110833_screenshot:4", "macos:20240905_110833_screenshot_sub0:1", "macos:20240905_111036_screenshot:4", "macos:20240905_112152_screenshot:1", "macos:20240905_112152_screenshot:5", "macos:20240905_112218_screenshot_sub1:0", "macos:20240905_112218_screenshot_sub1:2", "macos:20240905_112218_screenshot_sub1:5", "macos:20240905_112343_screenshot:0", "macos:20240905_112343_screenshot:3", "macos:20240905_112400_screenshot_sub1:1", "macos:20240905_112400_screenshot_sub1:6", "macos:20240905_112613_screenshot_sub1:0", "macos:20240905_112624_screenshot:1", "macos:20240905_112624_screenshot:6", "macos:20240905_112637_screenshot:0", "macos:20240905_112637_screenshot:6", "macos:20240905_112647_screenshot:6", "macos:20240905_113118_screenshot_sub1:1", "macos:20240905_113118_screenshot_sub1:5", "macos:20240905_113118_screenshot_sub1:7", "macos:20240905_113129_screenshot:7", "macos:20240905_113129_screenshot_sub1:0", "macos:20240905_113129_screenshot_sub1:4", "macos:20240905_113138_screenshot_sub1:1", "macos:20240905_113138_screenshot_sub1:3", "macos:20240905_113138_screenshot_sub1:6", "macos:20240905_113214_screenshot:3", "macos:20240905_113214_screenshot_sub1:1", "macos:20240905_113214_screenshot_sub1:2", "macos:20240905_113435_screenshot_sub1:1", "macos:20240905_113435_screenshot_sub1:6", "macos:20240905_113459_screenshot:2", "macos:20240905_114025_screenshot:0", "macos:20240905_114332_screenshot:0", "macos:20240905_114332_screenshot_sub2:0", "macos:20240905_114631_screenshot:2", "macos:20240905_133312_screenshot:0", "macos:20240905_133312_screenshot_sub1:7", "macos:20240905_135146_screenshot:0", "macos:20240905_135244_screenshot:0", "macos:20240905_135620_screenshot:4", "macos:20240905_144542_screenshot_sub1:5", "macos:20240905_144601_screenshot_sub1:4", "macos:20240905_155801_screenshot_sub0:2", "macos:20240905_155945_screenshot:3", "macos:20240905_155945_screenshot_sub0:2", "macos:20240905_161033_screenshot:5", "macos:20240905_161039_screenshot:0", "uibert:40956:3"
]
```

</details>
