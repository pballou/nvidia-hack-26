"""validate.py: tolerant JSON parsing, batch keying, item normalization and the faithfulness lints."""

import json

import pytest

from data_prep.rewrite import validate as v


def meta(label="Settings", neighbors=("Search", "Red alert"), control="ImageButton", style="functional"):
    m = {"label": label, "platform": "mobile", "position": "top-right", "size": "small",
         "neighbors": [{"label": n, "direction": "left"} for n in neighbors], "style": style}
    if control:
        m["control"] = control
    return m


def item(k=1, instruction="open the settings", keep=True, **kw):
    return {"k": k, "keep": keep, "instruction": instruction, **kw}


def check(instruction, m=None):
    """validate_item for one keep=True rewrite -> (rewritten or None, error, warnings)."""
    normalized, error, warnings = v.validate_item(item(instruction=instruction), m or meta())
    return (normalized or {}).get("rewritten"), error, warnings


GOOD = json.dumps({"items": [item(1, "open the settings."), {"k": 2, "keep": False, "drop_reason": "numeric"},
                             item(3, "the Search button")]})
METAS3 = [meta(), meta(label="42"), meta(label="Search", neighbors=("Settings",))]


# --- payload parsing -----------------------------------------------------------------------------------------------

def test_valid_batch():
    ok, errors, warnings, payload_ok = v.validate_batch(GOOD, METAS3)
    assert payload_ok and errors == {} and warnings == {}
    assert ok == {1: {"keep": True, "rewritten": "open the settings", "drop_reason": None},
                  2: {"keep": False, "rewritten": None, "drop_reason": "numeric"},
                  3: {"keep": True, "rewritten": "the Search button", "drop_reason": None}}


@pytest.mark.parametrize("text", [
    f"```json\n{GOOD}\n```",
    f"Here are the rewrites:\n```\n{GOOD}\n```\nLet me know if you need more.",
    f"<think>Row 2 is a number {{so drop}} it.</think>\n{GOOD}",
    f"<THINK>\nreasoning\n</THINK>```json\n{GOOD}```",
    f"reasoning without an opening tag</think>{GOOD}",
    f"Sure! {GOOD} Hope this helps [1].",
    f"Note [see format]: {GOOD}",
    json.dumps(json.loads(GOOD)["items"]),
    json.dumps(json.loads(GOOD), indent=2),
])
def test_tolerant_payloads(text):
    ok, errors, _, payload_ok = v.validate_batch(text, METAS3)
    assert payload_ok and errors == {} and sorted(ok) == [1, 2, 3]


def test_parse_json_payload_returns_object():
    assert v.parse_json_payload('<think>{"x": 1}</think> {"items": []}') == {"items": []}
    assert v.parse_json_payload('prefix [1, 2] suffix') == [1, 2]
    assert v.parse_json_payload('{"a": "line\nbreak"}') == {"a": "line\nbreak"}


@pytest.mark.parametrize("text", [
    "", "   ", "not json at all", "<think>only thoughts</think>", GOOD[:-20], '{"items": [{"k": 1, "keep": tru',
    '{"result": "ok"}', '{"items": "nope"}', "42", '"a string"', "```json\n```",
])
def test_malformed_payload(text):
    ok, errors, warnings, payload_ok = v.validate_batch(text, METAS3)
    assert not payload_ok and ok == {} and warnings == {}
    assert sorted(errors) == [1, 2, 3] and all(e.startswith("invalid_json: ") for e in errors.values())


def test_parse_rejects_non_text():
    with pytest.raises(ValueError):
        v.parse_json_payload(None)
    with pytest.raises(ValueError):
        v.parse_json_payload("no json here")


def test_extract_items():
    assert v.extract_items({"items": [1]}) == [1]
    assert v.extract_items([{"k": 1}]) == [{"k": 1}]
    single = {"k": 1, "keep": False, "drop_reason": "numeric"}
    assert v.extract_items(single) == [single]
    for bad in ({"items": {"k": 1}}, {"rows": []}, "x", 3, None):
        with pytest.raises(ValueError):
            v.extract_items(bad)


# --- batch keying --------------------------------------------------------------------------------------------------

def test_missing_item():
    ok, errors, _, payload_ok = v.validate_batch(json.dumps({"items": [item(1)]}), METAS3[:2])
    assert payload_ok and list(ok) == [1] and errors == {2: "missing"}


def test_duplicate_k_first_wins():
    text = json.dumps({"items": [item(1, "open the settings"), item(1, "tap the settings entry")]})
    ok, errors, _, _ = v.validate_batch(text, [meta()])
    assert ok[1]["rewritten"] == "open the settings" and errors == {}
    text = json.dumps({"items": [item(1, "settings"), item(1, "open the settings")]})
    ok, errors, _, _ = v.validate_batch(text, [meta()])
    assert ok == {} and errors == {1: "word_count:1"}


def test_string_float_and_bad_k():
    items = [item("2", "open search"), item(1.0), item(True, "tap something"), item(0), item(9), {"keep": True},
             "junk", None]
    ok, errors, _, payload_ok = v.validate_batch(json.dumps(items), [meta(), meta(label="Search"), meta()])
    assert payload_ok and sorted(ok) == [1, 2] and errors == {3: "missing"}
    assert ok[2]["rewritten"] == "open search"


def test_errors_and_warnings_keyed_by_k():
    items = [item(1, "settings"), item(2, "open GitHub settings now please")]
    ok, errors, warnings, _ = v.validate_batch(json.dumps(items), [meta(), meta(label="GitHub settings")])
    assert errors == {1: "word_count:1"} and list(ok) == [2] and 2 not in warnings


# --- item normalization --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("keep,expected", [
    (True, True), (False, False), ("true", True), ("FALSE", False), (" True ", True),
    ("yes", None), (1, None), (0, None), (None, None), ("", None),
])
def test_keep_types(keep, expected):
    normalized, error, _ = v.validate_item(item(keep=keep, drop_reason="numeric"), meta())
    if expected is None:
        assert normalized is None and error == "bad_keep"
    else:
        assert error is None and normalized["keep"] is expected


@pytest.mark.parametrize("reason,expected", [
    ("numeric", "numeric"), ("unintelligible", "unintelligible"), ("needs_screen", "needs_screen"),
    ("duplicate_neighbor", "duplicate_neighbor"), ("needs_visual", "needs_screen"), ("ambiguous", "needs_screen"),
    ("duplicate", "duplicate_neighbor"), ("nonsense", "unintelligible"), ("Needs Screen", "needs_screen"),
    ("needs-visual", "needs_screen"), ("NUMERIC ", "numeric"),
])
def test_drop_reason_aliases(reason, expected):
    normalized, error, warnings = v.validate_item({"k": 1, "keep": False, "drop_reason": reason}, meta())
    assert error is None and warnings == []
    assert normalized == {"keep": False, "rewritten": None, "drop_reason": expected}


@pytest.mark.parametrize("reason", ["llm_error", "other", "", None, 3, "too_long"])
def test_bad_drop_reason(reason):
    assert v.validate_item({"k": 1, "keep": False, "drop_reason": reason}, meta()) == (None, "bad_drop_reason", [])


def test_all_aliases_target_known_reasons():
    assert set(v.DROP_REASON_ALIASES.values()) <= set(v.DROP_REASONS)
    assert v.LLM_ERROR not in v.DROP_REASONS and v.STYLES == ("functional", "descriptive", "positional")


def test_model_style_ignored_and_shape():
    normalized, error, _ = v.validate_item(item(style="positional", drop_reason="numeric"), meta())
    assert error is None and normalized == {"keep": True, "rewritten": "open the settings", "drop_reason": None}


@pytest.mark.parametrize("raw,expected", [
    ("  open   the\tsettings  ", "open the settings"),
    ("open the settings.", "open the settings"),
    ("open the settings...", "open the settings"),
    ('"open the settings"', "open the settings"),
    ("open the settings\n", "open the settings"),
])
def test_instruction_normalization(raw, expected):
    assert check(raw)[:2] == (expected, None)


@pytest.mark.parametrize("raw", [None, "", "   ", 5, ["open settings"], "."])
def test_missing_instruction(raw):
    assert check(raw)[1] == "missing_instruction"


def test_rewritten_key_fallback():
    normalized, error, _ = v.validate_item({"k": 1, "keep": True, "rewritten": "open the settings"}, meta())
    assert error is None and normalized["rewritten"] == "open the settings"


def test_newline_rejected():
    assert check("open the\nsettings")[1] == "newline"


def test_not_object():
    assert v.validate_item("open settings", meta()) == (None, "not_object", [])


# --- lints ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n,error,length_warning", [
    (1, "word_count:1", True), (2, None, True), (3, None, False), (8, None, False), (9, None, True),
    (12, None, True), (13, "word_count:13", True),
])
def test_word_count_bounds(n, error, length_warning):
    text = " ".join(["open", "the", "settings"] + ["now"] * 20)
    rewritten, err, warnings = check(" ".join(text.split()[:n]))
    assert err == error and ("length" in warnings) is length_warning
    assert (rewritten is None) is (error is not None)


@pytest.mark.parametrize("text,error", [
    ("tap the blue settings icon", "invented_color:blue"),
    ("tap the GREY settings icon", "invented_color:grey"),
    ("open the settings gear", "invented_shape:gear"),
    ("tap the magnifying glass", "invented_shape:magnifying"),
    ("tap the three stars", "invented_shape:star"),
    ("tap the circular settings button", "invented_shape:circular"),
    ("tap the rounded settings button", "invented_shape:rounded"),
    ("tap the red and blue settings", "invented_color:blue"),   # red is in a neighbor label; blue is not
    ("open the settings around here", None),                    # word boundaries: around != round
    ("start the settings wizard", None),                        # start != star
    ("redo the settings change", None),                         # redo != red
])
def test_invented_color_shape(text, error):
    assert check(text)[1] == error


def test_color_shape_allowed_from_metadata():
    assert check("tap the blue theme toggle", meta(label="Blue theme"))[1] is None
    assert check("tap the button by red alert", meta())[1] is None              # neighbor label has "Red"
    assert check("tap the star next to search", meta(label="Stars rating"))[1] is None   # simple plural
    assert check("tap the stars next to search", meta(label="Star"))[1] is None
    assert check("tap the arrow at the top", meta(control="arrow"))[1] is None   # from control
    assert check("tap the arrow at the top", meta(control=None))[1] == "invented_shape:arrow"
    assert check("open the pink settings", meta(neighbors=("Pink Floyd",)))[1] is None


@pytest.mark.parametrize("text,error", [
    ("type in the 'Username' field", "invented_quote"),
    ('tap "Sign out" in the menu', "invented_quote"),
    ("tap “Log in” at the top", "invented_quote"),
    ("tap ‘Log in’ at the top", "invented_quote"),
    ("tap 'settings' at the top", None),         # case-insensitive substring of the label
    ('tap the button left of "Search"', None),   # neighbor label
    ("tap 'Red' near the top", None),            # substring of neighbor "Red alert"
    ("open the user's settings page", None),     # apostrophe, not a quote
    ("don't open the settings", None),
])
def test_invented_quote(text, error):
    assert check(text)[1] == error


def test_unchanged_warning():
    rewritten, error, warnings = check("open settings.", meta(label="Open settings"))
    assert error is None and rewritten == "open settings" and "unchanged" in warnings
    assert "unchanged" not in check("open the settings page")[2]


@pytest.mark.parametrize("label,text,warning", [
    ("GitHub repo", "open the repository page", "proper_noun:GitHub"),
    ("iPhone 15 Pro", "view the phone details", "proper_noun:iPhone"),
    ("MP3 player", "open the music player", "proper_noun:MP3"),
    ("USB-C cable", "buy the charging cable", "proper_noun:USB"),
    ("AutomationID:x", "open the automation panel", "proper_noun:AutomationID"),
])
def test_proper_noun_warning(label, text, warning):
    rewritten, error, warnings = check(text, meta(label=label))
    assert error is None and rewritten == text and warning in warnings


@pytest.mark.parametrize("label,text", [
    ("GitHub repo", "open the GitHub repo"),
    ("GitHub repo", "open the github repo"),
    ("USB-C cable", "buy the usb-c cable"),
    ("Settings", "open the settings page"),
    ("42 items", "show all 42 items"),
])
def test_no_proper_noun_warning(label, text):
    assert not [w for w in check(text, meta(label=label))[2] if w.startswith("proper_noun")]


def test_warnings_returned_with_valid_item_only_in_batch():
    text = json.dumps({"items": [item(1, "open the repository page")]})
    ok, errors, warnings, _ = v.validate_batch(text, [meta(label="GitHub repo")])
    assert list(ok) == [1] and errors == {} and warnings == {1: ["proper_noun:GitHub"]}
