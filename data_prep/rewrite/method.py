"""Write OUT/METHOD.md, the publication record of how the Lightning rewrites were made.

A pure function of stats.json plus the prompt and lint constants in code, so the same stats give the same bytes.
"""

import json
import re
from pathlib import Path

from data_prep.rewrite import meta, prompts, validate

EXAMPLE_META = {"label": "More options", "platform": "mobile", "control": "ImageButton", "position": "top-right",
                "size": "small", "neighbors": [{"label": "Search", "direction": "left"},
                                               {"label": "Inbox", "direction": "below"}], "style": "functional"}
FIELD_DOCS = {
    "k": "item number within the request (1..N); the reply is matched back by k.",
    "label": "the raw manifest instruction (accessibility name, HTML text or link text), whitespace-normalized.",
    "platform": "`mobile`, `web` or `desktop`, from the manifest.",
    "control": "control type from the row's `role`/`data_type`, last dotted segment "
               "(`android.widget.ImageButton` -> `ImageButton`); omitted when absent or only `text`/`icon`.",
    "kind": "`text` or `icon` when the source labels it (ScreenSpot-style); omitted otherwise.",
    "position": "3x3 grid cell (thirds of width and height) holding the bbox center: `top-left`, `top-center`, "
                "`top-right`, `middle-left`, `center`, `middle-right`, `bottom-left`, `bottom-center`, `bottom-right`. "
                "A center exactly on a 1/3 line goes to the later cell.",
    "size": f"bbox area / screenshot area: `small` < {meta.SIZE_SMALL_MAX:g}, `medium` < {meta.SIZE_MEDIUM_MAX:g}, "
            "else `large`.",
    "neighbors": f"up to {meta.MAX_NEIGHBORS} other manifest targets on the same screenshot, nearest bbox center first "
                 f"(ties by label, id). Each has its original label (first {meta.LABEL_MAX_CHARS} characters) and the "
                 "direction of the neighbor relative to the target: `overlapping` if either center lies inside the "
                 "other box, else `left`/`right`/`above`/`below` by the dominant axis of the center offset (ties go "
                 "horizontal).",
    "style": "the style to write, assigned by id (below).",
}
STYLE_DOCS = {"functional": 'an imperative action ("open the settings")',
              "descriptive": 'a noun phrase naming the element ("the Settings button")',
              "positional": 'locates the element by position and neighbors '
                            '("the button at the top right, next to Search")'}
RECORD_DOCS = [
    ("id", "manifest row id (`<subset>:<screenshot stem>:<k>`), which names the OS-Atlas screenshot"),
    ("source", "manifest source, e.g. `os_atlas/androidworld`"),
    ("original", "the raw label exactly as in the manifest"),
    ("rewritten", "Lightning's instruction after validation, or null when dropped"),
    ("style", "requested style (functional / descriptive / positional)"),
    ("keep", "false if Lightning dropped the row or it failed validation 3 times"),
    ("drop_reason", f"null, one of {', '.join(validate.DROP_REASONS)}, or {validate.LLM_ERROR}"),
    ("prompt_version", "prompt version (cache key together with id)"),
    ("model", "model id that produced the row"),
    ("ts", "UTC time the row was generated"),
]


def _fence(text: str, lang: str = "text") -> str:
    """Fenced block that keeps `text` verbatim (fence longer than any backtick run inside)."""
    ticks = "`" * max(3, 1 + max((len(m) for m in re.findall(r"`+", text)), default=0))
    return f"{ticks}{lang}\n{text if text.endswith(chr(10)) else text + chr(10)}{ticks}"


def _cell(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v).replace("|", "\\|").replace("\n", " ")


def _table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(_cell(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def _style_rule(split) -> str:
    edges, edge, parts = [], 0, []
    for name, share in split:
        edge += share
        edges.append((name, edge))
    for name, edge in edges[:-1]:
        parts.append(f'"{name}" if p < {edge} else ')
    return "".join(parts) + f'"{edges[-1][0]}"' if edges else '""'


def _section_results(stats: dict) -> list[str]:
    rw, q, g = stats.get("rewrites", {}), stats.get("quality", {}), stats.get("gate", {})
    reqs, tp, run = stats.get("requests", {}), stats.get("throughput", {}), stats.get("run", {})
    status = "complete" if not stats.get("pending_rows") else f"**incomplete: {stats.get('pending_rows')} rows pending**"
    out = ["## Results", "",
           f"Status: {status}" + (f"; aborted: {stats['aborted']}" if stats.get("aborted") else "") + ". "
           f"{rw.get('total', 0)} rows rewritten or dropped: {rw.get('kept', 0)} kept, {rw.get('dropped', 0)} dropped.",
           "", "By source:", "",
           _table(["source", "targets", "kept", "dropped", "drop rate"],
                  [(s, v["targets"], v["kept"], v["dropped"], _pct(v["drop_rate"]))
                   for s, v in rw.get("by_source", {}).items()]),
           "", "By style:", "",
           _table(["style", "targets", "kept"],
                  [(s, v["targets"], v["kept"]) for s, v in rw.get("by_style", {}).items()]),
           "", "Drop reasons:", "",
           _table(["drop_reason", "rows"], list(rw.get("drop_reasons", {}).items()) or [("(none)", 0)]),
           "", "Validation warnings on kept rows (counted, not dropped):", "",
           _table(["warning", "count"], list(rw.get("warnings", {}).items()) or [("(none)", 0)]),
           "", "Validation errors on failed attempts (those rows were re-asked):", "",
           _table(["error", "count"], list(q.get("validation_errors", {}).items()) or [("(none)", 0)]),
           "", "Quality gate (the faithfulness check is manual, on the `review_sample.py` contact sheet):", "",
           _table(["metric", "value", "gate", "pass"], [
               ("valid on first try (rows)", _pct(q.get("first_try_valid_rate")),
                f">= {g.get('first_try_valid_min', 0.95):.0%}", g.get("first_try_valid_ok")),
               ("drop rate, excluding llm_error", _pct(q.get("drop_rate_raw")), f"< {g.get('drop_rate_max', 0.25):.0%}",
                g.get("drop_rate_ok")),
               ("batch replies that parsed as JSON", _pct(q.get("batch_json_ok_rate")), "", None),
               ("rows re-asked individually", q.get("rows_retried"), "", None),
               ("rows dropped as llm_error", q.get("llm_error_rows"), "", None),
               ("faithful on manual review", "see review.html", f">= {g.get('faithful_min', 0.9):.0%}", None)]),
           "", "Requests and tokens (`all runs` sums every invocation for this prompt version on this cache):", ""]
    this, allr = reqs.get("this_run", {}), reqs.get("all_runs", {})
    keys = [("requests", "n"), ("HTTP attempts", "http_attempts"), ("HTTP retries", "retries"),
            ("prompt tokens", "prompt_tokens"), ("completion tokens", "completion_tokens"),
            ("total tokens", "total_tokens"),
            ("latency p50 (s)", "latency_p50_s"), ("latency p95 (s)", "latency_p95_s")]
    out.append(_table(["", "this run", "all runs"], [(label, this.get(k), allr.get(k)) for label, k in keys]
                      + [("by status", json.dumps(this.get("by_status", {})), json.dumps(allr.get("by_status", {}))),
                         ("by kind", json.dumps(this.get("by_kind", {})), json.dumps(allr.get("by_kind", {})))]))
    out += ["", "Throughput of the last invocation (a resumed job reports only its last invocation here):", "",
            _table(["wall time (s)", "rows/min", "requests/min", "completion tokens/s"],
                   [(run.get("wall_s"), tp.get("rows_per_min"), tp.get("requests_per_min"),
                     tp.get("completion_tokens_per_s"))])]
    if this.get("downgrades"):
        out += ["", f"response_format downgrades: `{json.dumps(this['downgrades'])}`"]
    return out


def render_method(stats: dict) -> str:
    run, man = stats.get("run", {}), stats.get("manifest", {})
    pv = run.get("prompt_version", prompts.PROMPT_VERSION)
    pct = run.get("select_pct", meta.SELECT_PCT)
    split = run.get("style_split") or [list(s) for s in meta.STYLE_SPLIT]
    example_line = prompts.format_item(1, EXAMPLE_META)
    fields = list(json.loads(example_line))
    rf_final = run.get("response_format_final") or run.get("response_format_requested")
    lines = [
        f"# Lightning instruction rewrites: method (prompt {pv})", "",
        "Written by `data_prep/rewrite/rewrite.py` from `stats.json` and the prompt in `data_prep/rewrite/prompts.py`, "
        "so it always matches the run it describes. Files: `rewrites.jsonl` (one row per target), `stats.json` "
        "(all counts), this file.", "",
        "## Purpose", "",
        "OS-Atlas AndroidWorld, FineWeb and Windows \"instructions\" are raw accessibility or HTML labels "
        "(\"More options\", \"System\"). ScreenSpot-v2 and real users phrase targets as short instructions "
        "(\"open the settings\"). NVIDIA Nemotron 3.5 Lightning rewrites a deterministic subset of those raw labels "
        "into short, faithful click instructions in three styles, to train a small (<= 3B) GUI grounding VLM. The other "
        "raw rows keep their label, and natural-instruction sources pass through unchanged. `apply.py` turns this file's "
        "`rewrites.jsonl` into `train_lightning.jsonl` (mixed) and `train_template.jsonl` (the same rows and order with "
        "raw labels) for the ablation.", "",
        "## Model and request settings", "",
        _table(["setting", "value"], [
            ("model (recorded in rewrites.jsonl)", f"`{run.get('model_label')}`"),
            ("model name sent", f"`{run.get('model')}`"),
            ("model reported by the server", ", ".join(f"`{m}`" for m in run.get("models_reported") or []) or "n/a"),
            ("endpoint", f"{run.get('endpoint')} (`{run.get('base_url')}`), OpenAI-compatible chat completions"),
            ("prompt_version", f"`{pv}`"),
            ("response_format", f"requested `{run.get('response_format_requested')}`, final `{rf_final}`"),
            ("rows per request / concurrent requests", f"{run.get('batch_size')} / {run.get('concurrency')}")]),
        "", "Sampling parameters sent with every request. Reasoning (\"thinking\") is off via "
            "`chat_template_kwargs.enable_thinking=false`:", "",
        _fence(json.dumps(run.get("sampling", {}), indent=2), "json"), "",
        "## Which rows are rewritten", "",
        f"Raw-label sources: {', '.join(f'`{s}`' for s in run.get('raw_sources', []))}. Natural-instruction sources "
        f"(passed through unchanged): {', '.join(f'`{s}`' for s in run.get('natural_sources', []))}. A raw-label row is "
        "selected for rewriting if and only if:", "",
        _fence(f'int(hashlib.sha256(row["id"].encode("utf-8")).hexdigest(), 16) % 100 < {pct}', "python"), "",
        "Unselected raw-label rows keep their raw label (`instruction_style` = `raw` after `apply.py`).",
    ]
    if run.get("limit") is not None:
        dry = " (dry run)" if run.get("dry_run") else ""
        lines += ["", f"This run was limited to {run['limit']} selected rows{dry}"
                  + ": the first ones ordered by `hashlib.sha256((\"sample:\" + row[\"id\"]).encode(\"utf-8\"))"
                  ".hexdigest()`, written in manifest order."]
    lines += [
        "", "## Style assignment", "",
        "Each selected row gets one style, by id:", "",
        _fence(f'p = int(hashlib.sha256(("style:" + row["id"]).encode("utf-8")).hexdigest(), 16) % 100\n'
               f"style = {_style_rule(split)}", "python"), "",
        *[f"- **{name}** (~{share}%): {STYLE_DOCS.get(name, name)}." for name, share in split],
        "", "## What Lightning sees", "",
        "Lightning is text-only and **never sees the screenshot**. Each target is one compact JSON line built only from "
        "manifest metadata:", "",
        *[f"- `{f}`: {FIELD_DOCS.get(f, 'see `data_prep/rewrite/prompts.py`.')}" for f in fields],
        "", "Neighbors are other *sampled* targets of the screenshot (the manifest keeps at most ~8 per screenshot), "
            "not every element on screen. Example user message for a one-item request, exactly as sent:", "",
        _fence(prompts.format_batch([EXAMPLE_META])), "",
        "Each request is: the system prompt, the few-shot pair below, then one user message with N item lines.", "",
        "## Prompt (verbatim)", "", "### System prompt", "", _fence(prompts.SYSTEM_PROMPT), "",
        "### Few-shot examples", "",
    ]
    for msg in prompts.FEW_SHOT:
        lines += [f"**{msg['role']}:**", "", _fence(str(msg["content"])), ""]
    lines += [
        "## Response format", "",
        f"`response_format` requested: `{run.get('response_format_requested')}` (`json_schema` sends the schema below; "
        "`json_object` asks for any JSON object; `none` sends nothing). If the server rejects the mode with HTTP 400, "
        "the client steps down once (json_schema -> json_object -> none) and keeps the lower mode for the rest of the run. "
        "Every reply is validated locally against this shape whatever the mode:", "",
        _fence(json.dumps(prompts.RESPONSE_SCHEMA, indent=2), "json"), "",
        "## Validation, retries and drops", "",
        "- Parsing tolerates `<think>` blocks, markdown fences and prose around the outermost JSON object or array; "
        "items are matched to inputs by `k` (first item wins; unknown `k` ignored). Any style the model echoes is "
        "ignored: the style comes from the id.",
        f"- `keep=true`: `instruction` must be one line of {validate.MIN_WORDS}-{validate.MAX_WORDS} words "
        f"({validate.TARGET_WORDS[0]}-{validate.TARGET_WORDS[1]} targeted; outside that is a `length` warning), "
        "whitespace-normalized, trailing period removed. It is rejected if it has a color or icon-shape word (lists "
        "below, simple plurals included) that is not in the label, a neighbor label or the control type, or if it "
        "quotes text that is not in the label or a neighbor label.",
        "- Warnings, counted but kept: `unchanged` (same as the label), `length`, `proper_noun:<token>` (a label token "
        "with an internal capital, digits and letters, or all caps is missing from the instruction).",
        f"- `keep=false` must give a `drop_reason` in {', '.join(f'`{r}`' for r in validate.DROP_REASONS)} "
        "(common synonyms are mapped onto these).",
        f"- Retries: one request carries up to {run.get('batch_size')} rows. Rows whose item is missing or invalid, or "
        "every row when the reply is not valid JSON or was cut off (`finish_reason=length`), are re-asked one row per "
        "request, up to 2 more times (3 attempts in all). Rows still invalid are dropped as "
        f"`drop_reason={validate.LLM_ERROR}`.",
        "- Transport failures (HTTP 408/429/5xx, timeouts, connection errors) are retried with exponential backoff and "
        "full jitter, honoring `Retry-After`. If they persist, the rows are not cached and are re-sent on the next run. "
        "Other HTTP 4xx errors stop the run.",
        "- `apply.py` removes selected rows with `keep=false` from both training variants, so the ablation compares "
        "identical rows.",
        f"- Color words: {', '.join(sorted(validate.COLOR_WORDS))}.",
        f"- Shape words: {', '.join(sorted(validate.SHAPE_WORDS))}.",
        "- Before any request the manifest load aborts if a row's source, id or image path mentions ScreenSpot or "
        "GroundUI, or if it comes from OS-Atlas Linux/macOS desktop data (captured on the same machines as "
        "ScreenSpot-v2's new desktop screenshots).", "",
        *_section_results(stats), "",
        "## Reproduce", "",
        _fence(str(run.get("command", "")), "bash"), "",
        _table(["", ""], [
            ("git commit", f"`{run.get('git_sha')}`" + (" (uncommitted changes in data_prep/rewrite)"
                                                        if run.get("git_dirty") else "")),
            ("manifest", f"`{man.get('path')}`"), ("manifest sha256", f"`{man.get('sha256')}`"),
            ("manifest rows", man.get("rows")), ("rows by source", json.dumps(man.get("rows_by_source", {}))),
            ("selected by source", json.dumps(man.get("selected_by_source", {}))),
            ("targets this run", man.get("targets")), ("run id", f"`{run.get('run_id')}`"),
            ("started / finished (UTC)", f"{run.get('started')} / {run.get('finished')}")]),
        "", "Sampling at temperature 0.7 is not deterministic, so `rewrites.jsonl` is the artifact. Rerunning the command "
            "against the same `cache.sqlite` (keyed by prompt_version and id) sends no requests and rewrites "
            "`rewrites.jsonl` byte for byte; `apply.py` is deterministic given it.", "",
        "## rewrites.jsonl fields", "",
        _table(["field", "meaning"], [(f"`{k}`", v) for k, v in RECORD_DOCS]), "",
        "## Known limitations", "",
        "- Lightning works from metadata only. A label that is ambiguous without the pixels can still produce an "
        "instruction that fits the label but not the boxed element. The `needs_screen` and `duplicate_neighbor` drops "
        "and the manual review reduce this but do not remove it.",
        "- Position is a coarse 3x3 bucket and size is three buckets, so positional instructions are approximate.",
        "- Neighbors are only the targets sampled from the same screenshot (at most ~8), so \"between A and B\" may skip "
        "unlabeled elements in between.",
        "- Lints are word lists and heuristics: they catch invented colors, icon shapes and quoted text, but not every "
        "hallucination. Faithfulness was audited by hand on a random sample (review_sample.py), not on every row.",
        "- The hosted endpoint's weights and serving stack are controlled by NVIDIA and may change between runs; "
        "`models_reported` records what the server said.",
        "- Model drops (`keep=false`) and `llm_error` rows remove those rows from both training variants, which shifts "
        "the training mix slightly away from numeric and generic labels.", "",
        "## Data sources and license", "",
        "- **OS-Atlas** ([OS-Copilot/OS-Atlas-data](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data), "
        "Apache-2.0 per the dataset card). Z. Wu et al., \"OS-ATLAS: A Foundation Action Model for Generalist GUI "
        "Agents\", arXiv:2410.23218, 2024. Labels, boxes and screenshots come from it.",
        "- Screenshots are **not re-hosted**: `id` and the manifest's `image_path` refer to the OS-Atlas files by name.",
        f"- Rewrites were generated by `{run.get('model_label')}` (NVIDIA Nemotron 3.5 Lightning).",
        "- No ScreenSpot (v1/v2), GroundUI-18K or OS-Atlas Linux/macOS desktop data is used (asserted at load time).",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_method(out_dir: Path, stats: dict) -> Path:
    path = Path(out_dir) / "METHOD.md"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(render_method(stats), encoding="utf-8")
    tmp.replace(path)
    return path
