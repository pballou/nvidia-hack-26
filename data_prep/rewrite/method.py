"""Write OUT/METHOD.md, the publication record of how the Lightning rewrites were made.

A pure function of stats.json plus the prompt, metadata and lint code, so the same stats give the same bytes. Who
generated the rows comes from the cached records (stats.json `rewrites`), not from the invocation that wrote the file.
"""

import inspect
import json
import re
from pathlib import Path

from data_prep.rewrite import client, meta, prompts, validate

# A synthetic screenshot: the "exactly as sent" example is built from it with meta.row_meta + prompts.format_batch.
_EX = {"source": "os_atlas/androidworld", "image_path": "images/androidworld/example.png", "img_w": 1080,
       "img_h": 2400, "platform": "mobile", "data_type": None, "split": None}
EXAMPLE_ROWS = [
    {**_EX, "id": "androidworld:example:0", "instruction": "More options", "bbox": [960, 100, 1060, 200],
     "role": "android.widget.ImageButton"},
    {**_EX, "id": "androidworld:example:1", "instruction": "Search", "bbox": [820, 100, 920, 200],
     "role": "android.widget.ImageButton"},
    {**_EX, "id": "androidworld:example:2", "instruction": "Inbox", "bbox": [960, 400, 1060, 500],
     "role": "android.widget.TextView"},
]
FIELD_DOCS = {
    "k": "item number within the request (1..N); the reply is matched back by k.",
    "label": "the raw manifest instruction (accessibility name, HTML text or link text) with invisible format "
             "characters removed (Unicode category Cf: bidi marks such as U+200E/U+2068, zero-width characters) and "
             f"whitespace collapsed. Labels longer than {prompts.LABEL_MAX_CHARS} characters are cut there and end "
             "in `…`.",
    "platform": "`mobile`, `web` or `desktop`, from the manifest.",
    "control": "a plain noun for the row's `role` (else `data_type`), via the table below. The raw value is reduced to "
               "a key: last dotted segment, a macOS `AX` prefix dropped, lowercased, non-alphanumerics removed "
               "(`android.widget.ImageButton` -> `imagebutton`, `AXPopUpButton` -> `popupbutton`). Unknown keys "
               "become `element`; omitted when both are absent or only `text`/`icon` (those go to `kind`).",
    "kind": "`text` or `icon` when the source labels it (ScreenSpot-style); omitted otherwise.",
    "position": "3x3 grid cell (thirds of width and height) holding the bbox center: `top-left`, `top-center`, "
                "`top-right`, `middle-left`, `center`, `middle-right`, `bottom-left`, `bottom-center`, `bottom-right`. "
                "A center exactly on a 1/3 line goes to the later cell. The prompt asks for plain words (\"at the top "
                "right\"); a copied grid token is a validation error.",
    "size": f"bbox area / screenshot area: `small` < {meta.SIZE_SMALL_MAX:g}, `medium` < {meta.SIZE_MEDIUM_MAX:g}, "
            "else `large`.",
    "neighbors": f"up to {meta.MAX_NEIGHBORS} other manifest targets on the same screenshot, nearest bbox center first "
                 f"(ties by label, id). Each has its label (normalized like `label`, first {meta.LABEL_MAX_CHARS} "
                 "characters) and `target_is`: where the TARGET lies relative to that neighbor, phrased to be copied "
                 "(`right of` Search -> \"right of Search\"): `overlapping` if either center lies inside the other box, "
                 "else `left of`/`right of`/`above`/`below` by the dominant axis of the center offset (ties go "
                 "horizontal).",
    "style": "the style to write, assigned by id (below).",
}
STYLE_DOCS = {"functional": 'an imperative action ("open more options"); for a text or area control only a neutral '
                            'verb plus the label ("tap Digital Wellbeing")',
              "descriptive": 'a noun phrase naming the element ("the Settings button")',
              "positional": 'locates the element by a plain-word position and/or a neighbor relation, keeping a word '
                            'of its own label ("the Share button at the top right", "the Save button below Title")'}
RECORD_DOCS = [
    ("id", "manifest row id (`<subset>:<screenshot stem>:<k>`), which names the OS-Atlas screenshot"),
    ("source", "manifest source, e.g. `os_atlas/androidworld`"),
    ("original", "the raw label exactly as in the manifest"),
    ("rewritten", "Lightning's instruction after validation, or null when dropped"),
    ("style", "requested style (functional / descriptive / positional)"),
    ("keep", "false if the row was dropped by rule, by Lightning, or failed validation 3 times"),
    ("drop_reason", f"null, one of {', '.join(validate.DROP_REASONS)}, or {validate.LLM_ERROR}"),
    ("prompt_version", "prompt version (cache key together with id)"),
    ("model", "model id that produced the row, or `rule:is_numeric_label` for numeric labels dropped before any "
              "request"),
    ("ts", "UTC time the row was generated"),
]
_ELEMENT_NOUNS = list(getattr(validate, "ELEMENT_NOUNS", None) or (
    "button", "icon", "link", "tab", "field", "text field", "input", "checkbox", "check box", "dropdown", "menu",
    "slider", "switch", "toggle", "image", "picture"))
_VERB_NOUNS = sorted(getattr(validate, "VERB_NOUNS", None) or ("toggle",))
_TEXT_VERBS = sorted(getattr(validate, "TEXT_VERBS", None) or ("click", "tap", "select", "choose", "press", "find",
                                                                  "go"))
# Every validation error code (a failing row is re-asked alone, up to 3 attempts per invocation). One line each.
ERROR_DOCS = [
    ("invalid_json", "the reply had no parseable JSON object or array; every row of the request is re-asked"),
    ("truncated", "the reply was cut off (`finish_reason=length`); every row of the request is re-asked"),
    ("missing", "the reply has no item with this `k`"),
    ("not_object", "the item is not a JSON object"),
    ("bad_keep", "`keep` is not true or false"),
    ("bad_drop_reason", "`keep=false` without a known `drop_reason`"),
    ("missing_instruction", "`keep=true` without a non-empty `instruction`"),
    ("newline", "the instruction spans more than one line"),
    ("word_count:<n>", f"fewer than {validate.MIN_WORDS} or more than {validate.MAX_WORDS} words"),
    ("invented_quote", "quotes text that is not in the label or a neighbor label"),
    ("invented_color:<word>", "a color word (list below) not in the label, a neighbor label or the control"),
    ("invented_shape:<word>", "an icon-shape word (list below) not in the label, a neighbor label or the control"),
    ("no_label_word", "no content word of the row's own label, in any style (catches an answer shifted to another "
                      "item); labels without ASCII words are exempt"),
    ("wrong_direction:<neighbor>", "\"left of/right of/above/below/under/beneath <neighbor>\" contradicts that "
                                   "neighbor's `target_is` (\"next to\", \"near\", \"between\" are not checked)"),
    ("invented_control:<noun>", f"an element-type noun ({', '.join(_ELEMENT_NOUNS)}; {'/'.join(_VERB_NOUNS)} as the "
                                "first word are verbs) not backed by the control, the label or a neighbor label: without "
                                "a control none at all, except button or link on web pages (generic clickable nouns), "
                                "and for a text/area control only words such as text, label, area, item, entry, option"),
    ("text_action_verb:<verb>", "a functional instruction for a text/area control that starts with a verb other than "
                                f"{', '.join(_TEXT_VERBS)} (no \"open X\" on a text)"),
    ("style_word:<word>", "the word functional, descriptive or positional, unless a label has it"),
    ("grid_token:<token>", "a hyphenated grid token (top-left, middle-right, ...) copied from `position`"),
    ("ambiguous_lookalike:<neighbor>", "a neighbor label is identical or starts with the same word, and the "
                                       "instruction has none of the target's distinguishing words, no relation to a "
                                       "neighbor and no position word (identical labels need a relation or position)"),
]
WARNING_DOCS = [
    ("stripped_style_word", "a style name (functional/descriptive/positional) copied at the very start or end of the "
                            "instruction was removed deterministically (validate.cleanup_instruction)"),
    ("dehyphenated_grid", "a hyphenated grid token (top-right) was rewritten as plain words (top right)"),
    ("unchanged", "the instruction equals the label"),
    ("length", f"outside the targeted {validate.TARGET_WORDS[0]}-{validate.TARGET_WORDS[1]} words"),
    ("neighbor_swap:<label>", "a functional/descriptive instruction names a neighbor's label and no word of the "
                              "target's label"),
    ("proper_noun:<token>", "a label token with an internal capital, digits and letters, or all caps is missing"),
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


def example_meta() -> dict:
    """What Lightning is told about EXAMPLE_ROWS[0], built by the same code as every real row."""
    return meta.row_meta(EXAMPLE_ROWS[0], meta.index_by_image(EXAMPLE_ROWS), style="functional")


def control_table() -> str:
    by_noun = {}
    for key, noun in meta.CONTROL_NOUNS.items():
        by_noun.setdefault(noun, []).append(key)
    rows = [(f"`{noun}`", ", ".join(f"`{k}`" for k in keys)) for noun, keys in by_noun.items()]
    return _table(["`control` sent", "role/data_type keys"], rows + [("`element`", "any other key")])


def _flags(pattern: re.Pattern) -> str:
    names = [n for n in ("IGNORECASE", "MULTILINE", "DOTALL", "VERBOSE", "ASCII") if pattern.flags & getattr(re, n)]
    return ", ".join(f"`re.{n}`" for n in names) or "none"


def numeric_rule_blocks() -> list[str]:
    """validate.is_numeric_label verbatim: its source, the plain constants it names, and every compiled pattern it
    names as the full regex text."""
    fn = validate.is_numeric_label
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = repr(fn)
    names = [n for n in fn.__code__.co_names if hasattr(validate, n)]
    consts = [f"{n} = {getattr(validate, n)!r}" for n in names
              if isinstance(getattr(validate, n), (int, float, str)) and not isinstance(getattr(validate, n), bool)]
    out = [_fence(source.rstrip() + ("\n\n" + "\n".join(consts) if consts else ""), "python")]
    for name in names:
        value = getattr(validate, name)
        if isinstance(value, re.Pattern):
            out += ["", f"`validate.{name}` (flags: {_flags(value)}), the regex text verbatim:", "",
                    _fence(value.pattern)]
    return out


def _rows(n) -> str:
    return f"{n} row" + ("" if n == 1 else "s")


def _by_count(counts: dict) -> list[str]:
    return [f"`{k}` ({_rows(v)})" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def _mixed(counts: dict, none: str = "none") -> str:
    """`a` (N rows); `b` (M rows): every value when the records are mixed."""
    return "; ".join(_by_count(counts)) or none


def _section_provenance(stats: dict) -> list[str]:
    run, rw = stats.get("run", {}), stats.get("rewrites", {})
    models = rw.get("llm_models", rw.get("models", {}))
    served = "not reported" + (" (NVIDIA's hosted API reports no root)" if run.get("endpoint") == "hosted" else "")
    return [
        "Who generated the rows, from the cached records (one value per row; every value is listed when rows were "
        "generated in several invocations):", "",
        _table(["", "rows"], [
            ("model (recorded in rewrites.jsonl)", _mixed(models, "none: no row was generated by a model")),
            ("served checkpoint (`GET /models` root; `(not reported)` = NVIDIA's hosted API, which reports none)",
             _mixed(rw.get("served_model_roots", {}))),
            ("endpoint", _mixed(rw.get("endpoints", {}))),
            ("dropped by rule, no model", f"{_rows(rw.get('rule_drops', 0))} (`model` = "
                                          f"`{rw.get('rule_model', 'rule:is_numeric_label')}`)")]),
        "", "The invocation that wrote this file (it may have sent nothing, e.g. a resume):", "",
        _table(["setting", "value"], [
            ("model label", f"`{run.get('model_label')}`"),
            ("model name sent", f"`{run.get('model')}`"),
            ("model reported by the server", ", ".join(f"`{m}`" for m in run.get("models_reported") or []) or "n/a"),
            ("served checkpoint (`GET /models` root for that name)",
             f"`{run.get('served_model_root')}`" if run.get("served_model_root") else served),
            ("endpoint", f"{run.get('endpoint')} (`{run.get('base_url')}`), OpenAI-compatible chat completions"),
            ("requests sent", (stats.get("requests", {}).get("this_run") or {}).get("n")),
            ("response_format", f"requested `{run.get('response_format_requested')}`, final "
                                f"`{run.get('response_format_final') or run.get('response_format_requested')}`"),
            ("rows per request / concurrent requests", f"{run.get('batch_size')} / {run.get('concurrency')}"),
            ("started / finished (UTC)", f"{run.get('started')} / {run.get('finished')}"),
            ("splits rewritten", ", ".join(run.get("splits") or []) or "all")]),
        *(["", f"Note: {run['note']}"] if run.get("note") else []),
    ]


def _section_results(stats: dict) -> list[str]:
    rw, q, g = stats.get("rewrites", {}), stats.get("quality", {}), stats.get("gate", {})
    reqs, tp, runs = stats.get("requests", {}), stats.get("throughput", {}), stats.get("runs", [])
    status = "complete" if not stats.get("pending_rows") else f"**incomplete: {stats.get('pending_rows')} rows pending**"
    out = ["## Results", "",
           f"Status: {status}" + (f"; aborted: {stats['aborted']}" if stats.get("aborted") else "") + ". "
           f"{rw.get('total', 0)} rows rewritten or dropped: {rw.get('kept', 0)} kept, {rw.get('dropped', 0)} dropped "
           f"({rw.get('rule_drops', 0)} of them by the numeric rule, never sent).",
           "", "By source:", "",
           _table(["source", "targets", "kept", "dropped", "of which by rule", "drop rate"],
                  [(s, v["targets"], v["kept"], v["dropped"], v.get("rule_dropped", 0), _pct(v["drop_rate"]))
                   for s, v in rw.get("by_source", {}).items()]),
           "", "By style:", "",
           _table(["style", "targets", "kept"],
                  [(s, v["targets"], v["kept"]) for s, v in rw.get("by_style", {}).items()]),
           "", "Drop reasons (`numeric` includes the rule drops):", "",
           _table(["drop_reason", "rows"], list(rw.get("drop_reasons", {}).items()) or [("(none)", 0)]),
           "", "Validation warnings on kept rows (counted, not dropped):", "",
           _table(["warning", "count"], list(rw.get("warnings", {}).items()) or [("(none)", 0)]),
           "", "Validation errors on failed attempts (those rows were re-asked):", "",
           _table(["error", "count"], list(q.get("validation_errors", {}).items()) or [("(none)", 0)]),
           "", "Quality gate (the faithfulness check is manual, on the `review_sample.py` contact sheet):", "",
           _table(["metric", "value", "gate", "pass"], [
               (f"valid on first try (of {q.get('llm_rows')} rows sent to the model; rule drops excluded)",
                _pct(q.get("first_try_valid_rate")), f">= {g.get('first_try_valid_min', 0.95):.0%}",
                g.get("first_try_valid_ok")),
               ("drop rate, all drops except llm_error (rule drops included)", _pct(q.get("drop_rate_raw")),
                f"< {g.get('drop_rate_max', 0.25):.0%}", g.get("drop_rate_ok")),
               ("drop rate of the model (rows sent; rule drops excluded)", _pct(q.get("drop_rate_model")), "", None),
               ("rows dropped by the numeric rule (never sent)", q.get("rule_drops"), "", None),
               ("batch replies that parsed as JSON", _pct(q.get("batch_json_ok_rate")), "", None),
               ("rows that needed more than one attempt (across invocations)", q.get("rows_retried"), "", None),
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
    out += ["", f"Throughput over all {tp.get('invocations', 0)} invocations that sent at least one request (sum of "
                "rows / sum of wall time; an invocation that sends nothing, such as a resume, does not change it):", "",
            _table(["invocations", "rows", "requests", "wall time (s)", "rows/min", "requests/min",
                    "completion tokens/s"],
                   [(tp.get("invocations"), tp.get("rows"), tp.get("requests"), tp.get("wall_s"),
                     tp.get("rows_per_min"), tp.get("requests_per_min"), tp.get("completion_tokens_per_s"))]),
            "", "Those invocations:", "",
            _table(["run id", "started / finished (UTC)", "wall (s)", "rows", "rule drops", "requests",
                    "prompt / completion tokens", "endpoint", "model label", "served checkpoint", "command"],
                   [(f"`{r.get('run_id')}`", f"{r.get('started')} / {r.get('finished')}", r.get("wall_s"),
                     r.get("rows_done"), r.get("rule_drops"), r.get("requests"),
                     f"{r.get('prompt_tokens')} / {r.get('completion_tokens')}",
                     f"{r.get('endpoint')} (`{r.get('base_url')}`)", f"`{r.get('model_label')}`",
                     f"`{r.get('served_model_root')}`" if r.get("served_model_root") else "not reported",
                     f"`{r.get('command')}`") for r in runs] or [("(none)", *[""] * 10)])]
    if this.get("downgrades"):
        out += ["", f"response_format downgrades: `{json.dumps(this['downgrades'])}`"]
    return out


def _banner(run: dict) -> list[str]:
    if not run.get("unofficial_model"):
        return []
    roots = ", ".join(run.get("unofficial_roots") or [str(run.get("served_model_root") or "unverified")])
    return [f"**NOT FOR PUBLICATION: generated with an unofficial/community checkpoint ({roots}); plumbing only**", "",
            f"Only rows generated by the official `{client.OFFICIAL_CHECKPOINT}` checkpoint or NVIDIA's hosted endpoint "
            "may be published or reported. Rerun without `--allow-unofficial-model` against the official checkpoint; "
            "rows from this checkpoint are then regenerated, not reused.", ""]


def _generated_by(rw: dict) -> str:
    who = ", ".join(_by_count(rw.get("llm_models", rw.get("models", {}))))
    text = f"Rewrites were generated by {who} (NVIDIA Nemotron 3.5 Lightning)" if who else \
        "No row has been generated by a model yet"
    return text + (f"; {rw['rule_drops']} numeric labels were dropped by rule, without a model." if
                   rw.get("rule_drops") else ".")


def render_method(stats: dict) -> str:
    run, man, rw = stats.get("run", {}), stats.get("manifest", {}), stats.get("rewrites", {})
    pv = run.get("prompt_version", prompts.PROMPT_VERSION)
    pct = run.get("select_pct", meta.SELECT_PCT)
    split = run.get("style_split") or [list(s) for s in meta.STYLE_SPLIT]
    fields = ["k", *prompts.ITEM_KEYS]
    lines = [
        *_banner(run),
        f"# Lightning instruction rewrites: method (prompt {pv})", "",
        "Written by `data_prep/rewrite/rewrite.py` from `stats.json` and the prompt in `data_prep/rewrite/prompts.py`, "
        "so it always matches the run it describes. Files: `rewrites.jsonl` (one row per target), `stats.json` "
        "(all counts), this file.", "",
        "## Purpose", "",
        "OS-Atlas Linux, macOS, AndroidWorld, FineWeb and Windows \"instructions\" are raw accessibility or HTML labels "
        "(\"More options\", \"System\"). ScreenSpot-v2 and real users phrase targets as short instructions "
        "(\"open the settings\"). NVIDIA Nemotron 3.5 Lightning rewrites a deterministic subset of those raw labels "
        "into short, faithful click instructions in three styles, to train a small (<= 3B) GUI grounding VLM. The other "
        "raw rows keep their label, and natural-instruction sources pass through unchanged. `apply.py` turns this file's "
        "`rewrites.jsonl` into `train_lightning.jsonl` (mixed) and `train_template.jsonl` (the same rows and order with "
        "raw labels) for the ablation, written next to the manifest.", "",
        "## Model and request settings", "",
        *_section_provenance(stats),
        "", f"Provenance check: before sending anything to a local server, `rewrite.py` reads `GET /models` and refuses "
            f"to run unless the served checkpoint is an `nvidia/` id with no community or quantization marker "
            f"({', '.join(f'`{m}`' for m in client.UNOFFICIAL_MARKERS)}); the reported data must come from "
            f"`{client.OFFICIAL_CHECKPOINT}` or NVIDIA's hosted endpoint. Each cached row records the checkpoint "
            "that produced it.",
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
        "", "## Numeric labels: dropped by rule, never sent", "",
        f"Before batching, a selected row whose label (normalized as in `label` below) satisfies "
        f"`validate.is_numeric_label` is recorded as `keep=false`, `drop_reason=numeric`, "
        f"`model={rw.get('rule_model', 'rule:is_numeric_label')}`, `attempts=0` and is never sent to Lightning "
        f"(numbers, prices, percentages, counters, clock times, dates, relative times). "
        f"{rw.get('rule_drops', 0)} rows in this file's counts were dropped this way. The rule, verbatim:", "",
        *numeric_rule_blocks(),
        "", "## Style assignment", "",
        "Each selected row gets one style, by id:", "",
        _fence(f'p = int(hashlib.sha256(("style:" + row["id"]).encode("utf-8")).hexdigest(), 16) % 100\n'
               f"style = {_style_rule(split)}", "python"), "",
        *[f"- **{name}** (~{share}%): {STYLE_DOCS.get(name, name)}." for name, share in split],
        "", "## What Lightning sees", "",
        "Lightning is text-only and **never sees the screenshot**. Each target is one compact JSON line built only from "
        "manifest metadata (optional fields are left out when unknown):", "",
        *[f"- `{f}`: {FIELD_DOCS.get(f, 'see `data_prep/rewrite/prompts.py`.')}" for f in fields],
        "", "`control` nouns (`meta.CONTROL_NOUNS`; `text` and `area` mark the label or region itself, never a "
            "button):", "",
        control_table(),
        "", "Neighbors are other *sampled* targets of the screenshot (the manifest keeps at most ~8 per screenshot), "
            "not every element on screen. Example user message for a one-item request, built from a synthetic "
            "screenshot by the same code (`meta.row_meta`, `prompts.format_batch`), exactly as sent:", "",
        _fence(prompts.format_batch([example_meta()])), "",
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
        f"- `keep=true`: `instruction` must be one line of {validate.MIN_WORDS}-{validate.MAX_WORDS} words, "
        "whitespace-normalized, trailing period removed, and pass every check below.",
        f"- `keep=false` must give a `drop_reason` in {', '.join(f'`{r}`' for r in validate.DROP_REASONS)} "
        "(common synonyms are mapped onto these).",
        f"- Retries: one request carries up to {run.get('batch_size')} rows. Rows whose item is missing or invalid, or "
        "every row when the reply is not valid JSON or was cut off (`finish_reason=length`), are re-asked up to 2 "
        "more times (3 attempts per invocation): "
        + ("together, in one request per failed reply (its failed rows only, renumbered from k=1; `--group-retries`)"
           if run.get("group_retries") else "one row per request")
        + f". Rows still invalid are dropped as `drop_reason={validate.LLM_ERROR}`."
        + (f" Requests were paced client-side at {run.get('rpm')} per minute (`--rpm`)." if run.get("rpm") else ""),
        "- Attempt accounting: every failed attempt is stored per row, so a row's `attempts` counts all its LLM "
        "attempts across invocations (an abort mid-retry, `--retry-errors`), and a row that ever failed is never "
        "counted as valid on the first try.",
        "- Output breaker: replies with no valid item extend a streak (a single-row retry whose one item failed only "
        "extends a running streak); any valid item resets it. While a streak runs, rows that use up their attempts "
        f"are held back instead of becoming `llm_error`. After {run.get('max_consecutive_empty_replies', 5)} such "
        "replies in a row the run stops with exit code 2 and those rows stay pending.",
        "- Transport failures (HTTP 408/429/5xx, timeouts, connection errors, undecodable bodies) are retried with "
        "exponential backoff and full jitter, honoring `Retry-After`. If they persist, the rows are not cached and are "
        "re-sent on the next run. Other HTTP 4xx errors stop the run.",
        "- `apply.py` removes selected rows with `keep=false` from both training variants, so the ablation compares "
        "identical rows.",
        "", "Validation errors (the attempt fails and the row is re-asked):", "",
        _table(["error", "meaning"], [(f"`{k}`", v) for k, v in ERROR_DOCS]), "",
        "Warnings (counted, the row is kept):", "",
        _table(["warning", "meaning"], [(f"`{k}`", v) for k, v in WARNING_DOCS]), "",
        f"- Color words: {', '.join(sorted(validate.COLOR_WORDS))}.",
        f"- Shape words: {', '.join(sorted(validate.SHAPE_WORDS))}.",
        "- Before any request the manifest load aborts if a row's source, id or image path mentions ScreenSpot or "
        "GroundUI, so no ScreenSpot rows are in the inputs (asserted). ScreenSpot-v2-clean is the held-out test. "
        "OS-Atlas Linux/macOS desktop data is training data; the 28 ScreenSpot-v2 desktop rows captured on the same "
        "machines are excluded from the test set instead (`benchmarks/screenspot_v2/excluded_rows.json`).", "",
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
            "`rewrites.jsonl` byte for byte; `apply.py` is deterministic given it. The commands of the invocations "
            "that sent requests are in the table above.", "",
        "## rewrites.jsonl fields", "",
        _table(["field", "meaning"], [(f"`{k}`", v) for k, v in RECORD_DOCS]), "",
        "## Known limitations", "",
        "- Lightning works from metadata only. A label that is ambiguous without the pixels can still produce an "
        "instruction that fits the label but not the boxed element. The `needs_screen` and `duplicate_neighbor` drops "
        "and the manual review reduce this but do not remove it.",
        "- Some source boxes are on the wrong element (a label boxed on a neighboring button); the rewrite follows the "
        "label, so such rows stay wrong. The manual review counts them separately (source error), not as rewrite "
        "errors.",
        "- Position is a coarse 3x3 bucket and size is three buckets, so positional instructions are approximate.",
        "- Neighbors are only the targets sampled from the same screenshot (at most ~8), so \"between A and B\" may skip "
        "unlabeled elements in between.",
        "- Lints are word lists and heuristics: they catch invented colors, icon shapes, quoted text, element nouns, "
        "contradicted directions and shifted answers, but not every hallucination. Faithfulness was audited by hand on "
        "a random sample (review_sample.py), not on every row.",
        "- The hosted endpoint's weights and serving stack are controlled by NVIDIA and may change between runs; "
        "`models_reported` records what the server said.",
        "- Rule drops, model drops (`keep=false`) and `llm_error` rows remove those rows from both training variants, "
        "which shifts the training mix slightly away from numeric and generic labels.", "",
        "## Data sources and license", "",
        "- **OS-Atlas** ([OS-Copilot/OS-Atlas-data](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data), "
        "Apache-2.0 per the dataset card). Z. Wu et al., \"OS-ATLAS: A Foundation Action Model for Generalist GUI "
        "Agents\", arXiv:2410.23218, 2024. Labels, boxes and screenshots come from it.",
        "- Screenshots are **not re-hosted**: `id` and the manifest's `image_path` refer to the OS-Atlas files by name.",
        f"- {_generated_by(rw)}",
        "- No ScreenSpot (v1/v2) or GroundUI-18K data is used (asserted at load time). ScreenSpot-v2-clean (revision "
        "`5efbb1f1b5` minus the 28 desktop rows in `benchmarks/screenspot_v2/excluded_rows.json`) is the held-out test.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_method(out_dir: Path, stats: dict) -> Path:
    path = Path(out_dir) / "METHOD.md"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(render_method(stats), encoding="utf-8")
    tmp.replace(path)
    return path
