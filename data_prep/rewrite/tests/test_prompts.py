"""prompts: message structure, item/batch format, few-shot validity and coverage, response schema, version pin."""

import hashlib
import json
import re

import pytest

from data_prep.rewrite import prompts
from data_prep.rewrite.prompts import (DROP_REASONS, FEW_SHOT, FEW_SHOT_ANSWER, FEW_SHOT_METAS, PROMPT_VERSION,
                                       RESPONSE_SCHEMA, STYLES, SYSTEM_PROMPT, build_messages, format_batch,
                                       format_item)

# Bump PROMPT_VERSION whenever the prompt text changes (the rewrite cache is keyed by it), then add the new pin here.
PROMPT_FINGERPRINTS = {"v1": "ca30d2812a875f50", "v2": "ba789479023ffcd4", "v3": "73fbcdcb90446c55", "v4": "bce5b7ac328ea6e5"}

POSITIONS = {"top-left", "top-center", "top-right", "middle-left", "center", "middle-right", "bottom-left",
             "bottom-center", "bottom-right"}
TARGET_IS = {"left of", "right of", "above", "below", "overlapping"}
GRID_TOKENS = POSITIONS - {"center"}
CONTRACT_KEYS = ["k", "label", "platform", "control", "kind", "position", "size", "neighbors", "style"]

META = {"label": "More options", "platform": "mobile", "control": "ImageButton", "position": "top-right",
        "size": "small", "neighbors": [{"label": "Search", "target_is": "right of"}], "style": "functional"}


def _is_type(value, name) -> bool:
    return {"object": isinstance(value, dict), "array": isinstance(value, list), "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool), "boolean": isinstance(value, bool),
            "null": value is None}[name]


def schema_errors(schema, value, path="$") -> list[str]:
    """Minimal JSON Schema check (type, enum, required, properties, items): enough for RESPONSE_SCHEMA."""
    errors = []
    types = schema.get("type")
    if types is not None and not any(_is_type(value, t) for t in (types if isinstance(types, list) else [types])):
        return [f"{path}: expected {types}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum")
    if isinstance(value, dict):
        errors += [f"{path}: missing {key}" for key in schema.get("required", ()) if key not in value]
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                errors += schema_errors(sub, value[key], f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            errors += schema_errors(schema["items"], v, f"{path}[{i}]")
    return errors


def fingerprint() -> str:
    blob = json.dumps([SYSTEM_PROMPT, FEW_SHOT, format_batch([META]), RESPONSE_SCHEMA], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def test_prompt_version():
    assert isinstance(PROMPT_VERSION, str) and PROMPT_VERSION.strip()


def test_prompt_version_pinned_to_prompt_text():
    assert PROMPT_FINGERPRINTS.get(PROMPT_VERSION) == fingerprint(), (
        f"prompt text changed (fingerprint {fingerprint()}): bump PROMPT_VERSION and add "
        f"'<new version>': '{fingerprint()}' to PROMPT_FINGERPRINTS in this test")


def test_build_messages_structure():
    metas = [META, {**META, "label": "Search"}]
    msgs = build_messages(metas)
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1:-1] == FEW_SHOT
    assert msgs[-1] == {"role": "user", "content": format_batch(metas)}
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant"] * (len(FEW_SHOT) // 2) + ["user"]
    assert all(set(m) == {"role", "content"} and isinstance(m["content"], str) and m["content"] for m in msgs)
    msgs[1]["content"] = "mutated"
    assert FEW_SHOT[0]["content"] != "mutated"


def test_format_item_single_line_contract_keys():
    meta = {**META, "kind": "icon", "id": "androidworld:x:0", "bbox": [1, 2, 3, 4], "extra": "leak?",
            "neighbors": [{"label": "Search", "target_is": "right of", "id": "secret", "direction": "left"}]}
    line = format_item(3, meta)
    assert "\n" not in line
    item = json.loads(line)
    assert list(item) == CONTRACT_KEYS
    assert item["k"] == 3 and item["neighbors"] == [{"label": "Search", "target_is": "right of"}]
    assert {k: item[k] for k in ("label", "platform", "control", "kind", "size", "style")} == \
        {k: meta[k] for k in ("label", "platform", "control", "kind", "size", "style")}
    assert item["position"] == "top right"  # v4: plain words, not the top-right token


def test_format_item_omits_missing_optional_keys():
    meta = {k: v for k, v in META.items() if k != "control"}
    assert "control" not in json.loads(format_item(1, meta))
    assert "control" not in json.loads(format_item(1, {**META, "control": None}))
    assert "kind" not in json.loads(format_item(1, META))
    assert json.loads(format_item(1, {**META, "neighbors": []}))["neighbors"] == []


def test_format_item_text_handling():
    meta = {**META, "label": 'Öffnen 设置 "now"\nplease  \t ok'}
    line = format_item(1, meta)
    assert "Öffnen 设置" in line and "\n" not in line
    assert json.loads(line)["label"] == 'Öffnen 设置 "now" please ok'
    long = json.loads(format_item(1, {**META, "label": "word " * 200}))["label"]
    assert len(long) <= prompts.LABEL_MAX_CHARS + 1 and long.endswith("…")


def test_format_batch_numbering():
    metas = [{**META, "label": f"Item label {i}"} for i in range(5)]
    lines = format_batch(metas).split("\n")
    head, items = lines[0], [json.loads(x) for x in lines[2:]]
    assert lines[1] == "" and "5 items" in head and "JSON" in head
    assert [it["k"] for it in items] == [1, 2, 3, 4, 5]
    assert [it["label"] for it in items] == [m["label"] for m in metas]
    single = format_batch([META])
    assert "1 item," in single and "1 items" not in single


def test_format_batch_on_real_metas():
    meta_mod = pytest.importorskip("data_prep.rewrite.meta", reason="meta.py not available")
    from data_prep.rewrite.tests import synth
    rows = synth.make_rows()
    by_image = meta_mod.index_by_image(rows)
    metas = [meta_mod.row_meta(r, by_image) for r in rows[:20]]
    for k, line in enumerate(format_batch(metas).split("\n")[2:], 1):
        item = json.loads(line)
        assert item["k"] == k and set(item) <= set(CONTRACT_KEYS)
        assert item["style"] in STYLES and item["position"] in {p.replace("-", " ") for p in POSITIONS}
        assert all(set(n) == {"label", "target_is"} and n["target_is"] in TARGET_IS for n in item["neighbors"])


def test_format_batch_reminds_target_is_and_own_label():
    head = format_batch([META]).split("\n")[0]
    assert "target_is" in head and "copy it as is" in head and "own label" in head
    assert "direction" not in head


def test_few_shot_metas_are_well_formed():
    assert FEW_SHOT[0] == {"role": "user", "content": format_batch(FEW_SHOT_METAS)}
    for m in FEW_SHOT_METAS:
        assert set(m) <= set(CONTRACT_KEYS) - {"k"}
        assert m["platform"] in {"mobile", "web", "desktop"} and m["position"] in POSITIONS
        assert m["size"] in {"small", "medium", "large"} and m["style"] in STYLES
        assert 1 <= len(m["neighbors"]) <= 8
        assert all(set(n) == {"label", "target_is"} and n["target_is"] in TARGET_IS and len(n["label"]) <= 60
                   for n in m["neighbors"])


def test_few_shot_answer_matches_schema_and_covers_cases():
    assert len(FEW_SHOT) % 2 == 0
    total = kept = 0
    kept_styles, reasons = set(), []
    for user, assistant in zip(FEW_SHOT[::2], FEW_SHOT[1::2]):
        assert user["role"] == "user" and assistant["role"] == "assistant"
        answer = json.loads(assistant["content"])
        assert schema_errors(RESPONSE_SCHEMA, answer) == []
        lines = [json.loads(x) for x in user["content"].split("\n")[2:]]
        assert [it["k"] for it in answer["items"]] == [it["k"] for it in lines]
        for it, inp in zip(answer["items"], lines):
            if it["keep"]:
                kept += 1
                kept_styles.add(inp["style"])
                assert 3 <= len(it["instruction"].split()) <= 8, it
                assert "drop_reason" not in it and "\n" not in it["instruction"]
                assert not it["instruction"].endswith(".") and '"' not in it["instruction"]
            else:
                reasons.append(it["drop_reason"])
                assert "instruction" not in it
        total += len(answer["items"])
    assert 8 <= total <= 9 and 5 <= kept <= 6
    assert kept_styles == set(STYLES)
    assert sorted(reasons) == ["needs_screen", "numeric", "unintelligible"]
    assert json.loads(FEW_SHOT[1]["content"]) == FEW_SHOT_ANSWER


# Local lints for the v2 audit fixes, so the few-shot teaches them whatever validate.py checks.
STOP = set("a an and the of to in on for at by with from or your my this that is it".split())
ELEMENT_NOUNS = {"button", "icon", "link", "tab", "field", "input", "checkbox", "dropdown", "menu", "slider",
                 "switch", "toggle", "image", "picture"}
NEUTRAL_VERBS = {"click", "tap", "select"}
RELATION = re.compile(r"\b(left of|right of|above|below) ([A-Z][\w&' ]*)")


def _words(text) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 2 and w not in STOP]


def _keeps():
    return [(it["instruction"], m) for it, m in zip(FEW_SHOT_ANSWER["items"], FEW_SHOT_METAS) if it["keep"]]


def test_few_shot_keeps_follow_v3_rules():
    for instruction, m in _keeps():
        low, words = instruction.lower(), set(_words(instruction))
        assert words & set(_words(m["label"])), (instruction, "keeps a word of its own label")
        assert not any(tok in low for tok in GRID_TOKENS) and not words & set(STYLES), instruction
        context = set(_words(" ".join([m["label"], m.get("control") or "", *(n["label"] for n in m["neighbors"])])))
        assert not (words & ELEMENT_NOUNS) - context, (instruction, "element noun not backed by control or labels")
        if m.get("control") in ("text", "area") and m["style"] == "functional":
            assert low.split()[0] in NEUTRAL_VERBS, instruction
        for rel, name in RELATION.findall(instruction):
            assert any(n["label"].startswith(name.strip()) and n["target_is"] == rel for n in m["neighbors"])


def test_few_shot_covers_v2_audit_failures():
    keeps = _keeps()
    # a) a text control in functional style gets a neutral verb, not "open" or a button noun
    assert any(m.get("control") == "text" and m["style"] == "functional" for _, m in keeps)
    # c) a long accessibility description becomes a short phrase, not the copied sentence
    long = [(i, m) for i, m in keeps if len(m["label"].split()) >= 8]
    assert long and all(" ".join(m["label"].split()[:3]).lower() not in i.lower() for i, m in long)
    # d) a look-alike neighbor (same first word) is told apart by a word only the target has
    pairs = [(i, m, n) for i, m in keeps for n in m["neighbors"] if _words(n["label"])[:1] == _words(m["label"])[:1]]
    assert pairs and all(set(_words(i)) & (set(_words(m["label"])) - set(_words(n["label"]))) for i, m, n in pairs)
    # f) a positional keep copies a neighbor's target_is verbatim
    assert any(f"{n['target_is']} {n['label']}" in i for i, m in keeps if m["style"] == "positional"
               for n in m["neighbors"])
    # g) a positional keep says its position in plain words
    assert any(re.search(r"\b(at|on|in) the (top|bottom|left|right|middle)\b", i) for i, m in keeps
               if m["style"] == "positional")
    # h) web rows without a control get no element noun, in more than one style
    assert len({m["style"] for _, m in keeps if "control" not in m}) >= 2


def test_few_shot_numeric_example_reaches_the_model():
    """Labels the numeric rule drops never reach Lightning, so the few-shot numeric drop must be one it lets through."""
    validate = pytest.importorskip("data_prep.rewrite.validate", reason="validate.py not available")
    if not hasattr(validate, "is_numeric_label"):
        pytest.skip("validate.is_numeric_label not available yet")
    assert not any(validate.is_numeric_label(m["label"]) for m in FEW_SHOT_METAS)


def test_few_shot_duplicate_example_has_duplicate_neighbor():
    for it, meta in zip(FEW_SHOT_ANSWER["items"], FEW_SHOT_METAS):
        if it.get("drop_reason") == "duplicate_neighbor":
            assert any(n["label"].lower() == meta["label"].lower() for n in meta["neighbors"])


def test_few_shot_passes_validator_cleanly():
    validate = pytest.importorskip("data_prep.rewrite.validate", reason="validate.py not available")
    ok, errors, warnings, payload_ok = validate.validate_batch(FEW_SHOT[1]["content"], FEW_SHOT_METAS)
    assert payload_ok and errors == {} and warnings == {}
    assert sorted(ok) == list(range(1, len(FEW_SHOT_METAS) + 1))
    for it in FEW_SHOT_ANSWER["items"]:
        assert ok[it["k"]]["keep"] == it["keep"]
        assert ok[it["k"]]["rewritten"] == it.get("instruction")
        assert ok[it["k"]]["drop_reason"] == it.get("drop_reason")


def test_response_schema_shape():
    item = RESPONSE_SCHEMA["properties"]["items"]["items"]
    assert RESPONSE_SCHEMA["required"] == ["items"] and item["required"] == ["k", "keep"]
    assert set(item["properties"]) == {"k", "keep", "instruction", "drop_reason"}
    good = {"items": [{"k": 1, "keep": True, "instruction": "open more options"},
                      {"k": 2, "keep": False, "drop_reason": "numeric", "instruction": None}]}
    assert schema_errors(RESPONSE_SCHEMA, good) == []
    assert schema_errors(RESPONSE_SCHEMA, {"items": [{"k": 1, "keep": False, "drop_reason": "llm_error"}]})
    assert schema_errors(RESPONSE_SCHEMA, {"items": [{"k": "1", "keep": True}]})
    assert schema_errors(RESPONSE_SCHEMA, {"items": [{"k": 1}]})
    json.dumps(RESPONSE_SCHEMA)


def test_response_schema_enum_matches_validate():
    validate = pytest.importorskip("data_prep.rewrite.validate", reason="validate.py not available")
    enum = RESPONSE_SCHEMA["properties"]["items"]["items"]["properties"]["drop_reason"]["enum"]
    assert enum == [*validate.DROP_REASONS, None]
    assert DROP_REASONS == validate.DROP_REASONS and STYLES == validate.STYLES


def test_system_prompt_names_vocabulary_and_stays_tight():
    for word in (*STYLES, *DROP_REASONS, "JSON", '"items"', "keep", "drop_reason", "instruction"):
        assert word in SYSTEM_PROMPT, word
    assert 'target_is right of Search -> "right of Search"' in SYSTEM_PROMPT and "copy it as is" in SYSTEM_PROMPT
    assert "direction" not in SYSTEM_PROMPT and "search for hotels" not in SYSTEM_PROMPT
    assert "the app usage bar chart" in SYSTEM_PROMPT and "tap Digital Wellbeing" in SYSTEM_PROMPT
    assert len(SYSTEM_PROMPT.split()) <= 800
