# T3: Lightning data engine (instruction rewriting for training data)

**Owner:** Patrick (stream C). **Time box:** ~2.5 h. Time-critical, because main SFT
starts ~6:30pm Saturday:
- **~4pm:** prompt and a 200-row dry run, reviewed.
- **~6pm:** the full run on Charlie's slice.
- The SFT pilot (~5pm) uses raw labels, so it doesn't wait on this.

**Read first:** `AGENTS.md` (especially the open-data rule),
`docs/research-notes.md` §5 (data quality and the recipe), `docs/plan.md` (the
data-mix table), `docs/tasks/contracts.md`.

## Why this exists

OS-Atlas Linux/macOS desktop, AndroidWorld and FineWeb "instructions" are raw accessibility or
HTML labels, often a single word ("System", "More options menu"). ScreenSpot-v2, and
real users, phrase requests as instructions ("open the settings", "download the
September statement").

Lightning rewrites the labels into natural, faithful instructions. That makes
Lightning causally responsible for a measured part of the model's improvement, which
is our strongest "Best Use of Nemotron Lightning" story. Everything we generate must
be **published with its method**.

## Input

- Charlie's canonical manifest (from `data_prep/sample_os_atlas.py`):
  `manifest.jsonl` rows with
  `{id, source, image_path, img_w, img_h, instruction, bbox[x1,y1,x2,y2 px], platform, data_type, split}`.
- The path is on the box under `/home/ubuntu/data/<slice>/`. Copy it locally for
  development.
- **Ask Charlie to populate `data_type`** (the control type) where the source has it.
  He currently writes `None`.

## What to rewrite

- **Raw-label sources:** `os_atlas/linux`, `os_atlas/macos`, `os_atlas/androidworld`,
  `os_atlas/fineweb` (and `os_atlas/windows` if added), plus any source whose
  instructions are bare labels.
  - Select **~60%** of their rows deterministically: hash of `id` mod 100 < 60.
  - The rest keep the raw label.
- **Natural-instruction sources** (`uibert`, `widget_captioning`, `groundcua`) pass
  through unchanged.

## How

- **Lightning can't see the image,** so give it only metadata it can trust:
  - the raw label;
  - `data_type` (if present);
  - `platform`;
  - a coarse **position bucket** from the bbox (3×3 grid: top-left … bottom-right);
  - a size bucket (small/medium/large);
  - up to ~8 **neighboring labels** from the same screenshot, sorted by distance, for
    disambiguation.
- **Styles,** about 50/30/20:
  - functional: "open the settings";
  - descriptive: "the Settings button";
  - positional: "the button at the top right, next to Search".
  - Imperative, 3–8 words, ScreenSpot-like.
- **Hard rules in the prompt:**
  - Never invent visual attributes (colors, icon shapes) that aren't in the metadata.
  - Never invent text.
  - Keep any proper noun from the label.
- **Return `keep=false` with a `drop_reason`** when the label is numeric or
  unintelligible, would need seeing the screen to disambiguate, or duplicates a
  neighbor.
- **Batching:** ~20 rows per request, with **JSON array output** (try `response_format`
  json and validate the shape). Rows that fail validation are re-queued individually,
  up to 2 retries, then marked `keep=false, drop_reason="llm_error"`.
- **Settings:**
  - Thinking **off** (`chat_template_kwargs.enable_thinking=false`).
  - temperature 0.7, top_p 0.95.
  - Record the model id and `prompt_version`.
- **Endpoint order:**
  1. Hosted (`NVIDIA_API_KEY`): **check the rate limit in the first 10 minutes.**
  2. Patrick's Mac via vllm-metal (`docs/local-mac.md`; the Gate L setup doubles as
     this).
  3. The H200, only if the GPU is idle and Kyle agrees.
  - Concurrency 4–8, with exponential backoff on 429/5xx.
- **Resume:** a cache keyed by `(prompt_version, id)` (sqlite or jsonl), so reruns and
  crashes don't re-spend.

## Outputs (`data_prep/rewrite/`)

| File | Contents |
| --- | --- |
| `rewrite.py` | CLI: `--manifest`, `--out`, `--endpoint hosted\|local\|h200`, `--limit`, `--dry-run` |
| `prompts.py` | System prompt and few-shot examples (5–8, covering each style and the drop cases), `PROMPT_VERSION` |
| `apply.py` | `apply_rewrites(manifest, rewrites) -> train manifest`. Replaces `instruction` where `keep && rewritten`, adds `instruction_original` and `instruction_style`, drops rows marked `keep=false` only if they were selected for rewriting. Deterministic. Also writes the **two ablation variants** Kyle can train on: `train_template.jsonl` (raw labels) and `train_lightning.jsonl` (mixed). |
| `review_sample.py` | Writes an HTML contact sheet of 50 random rewritten rows: image crop with the target box drawn, original label, rewrite, style. A human can audit it in ~5 minutes. |
| `<out>/rewrites.jsonl` | `{id, source, original, rewritten \| null, style, keep, drop_reason, prompt_version, model, ts}` |
| `<out>/stats.json` | Counts by source, style and drop reason; tokens; wall time; requests; errors |
| `<out>/METHOD.md` | **For publication:** prompt verbatim, model and parameters, selection rule, metadata given, filters, counts, known limitations |

**Publication:** Patrick uploads `rewrites.jsonl`, `METHOD.md` and `stats.json` to
`patrickballou/gui-assist-data` (see `docs/submission.md`). Image filenames reference
OS-Atlas; images aren't re-hosted.

## Quality gate (before the full run)

- 200-row dry run, then `review_sample.py` on 50 rows.
- **Pass if:**
  - at least 90% are faithful (they describe the boxed element and invent nothing);
  - at least 95% are valid JSON on the first try;
  - the drop rate is under 25% on raw-label sources.
- Iterate on the prompt (bump `PROMPT_VERSION`) at most twice, then run the full
  slice.
- Tell Charlie and Kyle when `train_lightning.jsonl` is ready, and put the paths in
  chat and in `docs/runs.md`.

## Tests

- Unit-test the position and size bucketing, neighbor selection, batch JSON
  validation and re-queue, deterministic selection, `apply_rewrites`, and cache
  resume.
- Use a fake Lightning stub for the error paths (malformed JSON, 429, timeout).

## Acceptance criteria

- [ ] The dry run and review pass the quality gate. The full run finishes on
      Charlie's slice by ~6pm.
- [ ] `rewrites.jsonl`, `stats.json`, `METHOD.md`, `train_template.jsonl` and
      `train_lightning.jsonl` are produced. They're deterministic given the same
      cache.
- [ ] No ScreenSpot data anywhere in the inputs. Assert that the manifest has no
      `screenspot` source. OS-Atlas Linux/macOS are allowed under option C; see
      AGENTS.md.
- [ ] The numbers for the Lightning bounty write-up are logged: rows rewritten,
      throughput, tokens, drop reasons.

## Out of scope

Sampling and cleaning (Charlie's `sample_os_atlas.py`), training (Kyle), and the
agent (T2).
