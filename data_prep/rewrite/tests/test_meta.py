"""meta.py: selection, styles, ScreenSpot/GroundUI contamination and manifest checks, buckets, directions, neighbors,
row_meta; synth."""

import hashlib
import json
from collections import Counter

import pytest

from data_prep.rewrite import meta
from data_prep.rewrite.tests import synth


def row(rid="androidworld:img:0", label="Settings", bbox=(10, 10, 20, 20), image="images/androidworld/img.png", **kw):
    r = {"id": rid, "source": "os_atlas/androidworld", "image_path": image, "img_w": 300, "img_h": 300,
         "instruction": label, "bbox": None if bbox is None else list(bbox), "platform": "mobile", "data_type": None,
         "role": None, "split": None}
    r.update(kw)
    return r


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# --- selection and styles ------------------------------------------------------------------------------------------

def test_stable_pct_matches_formula_and_salt():
    key = "androidworld:abc:3"
    assert meta.stable_pct(key) == int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100
    assert meta.stable_pct(key, "style:") == int(hashlib.sha256(("style:" + key).encode()).hexdigest(), 16) % 100
    assert all(0 <= meta.stable_pct(f"x{i}") < 100 for i in range(200))


def test_selection_rate_and_determinism():
    rows = [row(rid=f"androidworld:img{i // 8}:{i % 8}") for i in range(10_000)]
    picked = [meta.is_selected(r) for r in rows]
    assert 0.55 <= sum(picked) / len(rows) <= 0.65
    assert picked == [meta.is_selected(r) for r in rows]
    assert all(meta.is_selected(r) == (meta.stable_pct(r["id"]) < meta.SELECT_PCT) for r in rows[:500])


def test_natural_sources_never_selected():
    for source in ("os_atlas/uibert", "os_atlas/widget_captioning", "groundcua", "groundcua/vscode"):
        assert not any(meta.is_selected(row(rid=f"x:{i}:0", source=source)) for i in range(300))


def test_extra_sources_override():
    r = row(source="seeclick/web")
    with pytest.raises(meta.ManifestError):
        meta.is_selected(r)
    assert meta.classify_source("seeclick/web", extra_raw=("seeclick/web",)) == "raw"
    assert meta.classify_source("seeclick/web", extra_raw="a,seeclick/web") == "raw"
    assert meta.classify_source("os_atlas/fineweb", extra_natural=["os_atlas/fineweb"]) == "natural"
    assert meta.classify_source("x", extra_natural=("x",)) == "natural"
    with pytest.raises(meta.ManifestError):
        meta.classify_source("x", extra_raw=("x",), extra_natural=("x",))


def test_classify_known_sources():
    for s in ("os_atlas/linux", "os_atlas/macos", "os_atlas/windows", "os_atlas/androidworld", "os_atlas/fineweb"):
        assert meta.classify_source(s) == "raw"
    assert meta.RAW_LABEL_SOURCES == {"os_atlas/linux", "os_atlas/macos", "os_atlas/androidworld", "os_atlas/fineweb",
                                      "os_atlas/windows"}
    for s in ("os_atlas/uibert", "os_atlas/widget_captioning", "groundcua", "groundcua/gimp"):
        assert meta.classify_source(s) == "natural"


def test_unknown_source_raises_with_hint():
    with pytest.raises(meta.ManifestError, match="--raw-sources") as e:
        meta.classify_source("os_atlas/ricosca")
    assert "--natural-sources" in str(e.value)
    with pytest.raises(meta.ManifestError):
        meta.classify_source("groundcuax")


def test_style_split():
    styles = Counter(meta.assign_style(f"fineweb:page{i}:{i % 7}") for i in range(10_000))
    assert set(styles) == {"functional", "descriptive", "positional"}
    for style, share in meta.STYLE_SPLIT:
        assert abs(100 * styles[style] / 10_000 - share) <= 4, (style, styles)
    assert meta.assign_style("a:b:0") == meta.assign_style("a:b:0")


def test_style_matches_pct_thresholds():
    for i in range(500):
        rid = f"r{i}"
        p = meta.stable_pct(rid, "style:")
        assert meta.assign_style(rid) == ("functional" if p < 50 else "descriptive" if p < 80 else "positional")


# --- contamination and manifest loading ----------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    {"source": "screenspot"},
    {"source": "wave_ui/screenspot"},
    {"image_path": "images/ScreenSpot_v2/x.png"},
    {"id": "screenspot_v2:web_3:0"},
    {"source": "groundui"},
    {"id": "GroundUI-18K:7"},
    {"image_path": "/home/ubuntu/data/groundui_18k/images/a.png"},
])
def test_contamination_raises(bad, tmp_path):
    rows = [row(rid="androidworld:ok:0"), row(rid="androidworld:ok:1", **{"id": "tmp:1", **bad})]
    with pytest.raises(meta.ContaminationError, match="AGENTS.md"):
        meta.assert_clean_manifest(rows)
    with pytest.raises(meta.ContaminationError):
        meta.load_manifest(write_jsonl(tmp_path / "m.jsonl", rows))


def test_contamination_message_counts_reasons():
    rows = [row(rid=f"screenspot:a:{i}") for i in range(3)] + [row(rid="b:0", source="groundui")]
    with pytest.raises(meta.ContaminationError) as e:
        meta.assert_clean_manifest(rows)
    assert "screenspot=3" in str(e.value) and "groundui=1" in str(e.value)


def test_clean_rows_pass():
    meta.assert_clean_manifest([row(), row(rid="fineweb:p:0", source="os_atlas/fineweb",
                                           image_path="images/fineweb/linux_kernel_docs.png")])


@pytest.mark.parametrize("subset", ["linux", "macos"])
def test_os_atlas_linux_macos_are_training_data(subset, tmp_path):
    """Eval option C: Linux/macOS load, are raw-label sources, and ~60% are selected (the 28 overlapping SSv2 rows
    are excluded from the test set instead)."""
    rows = [row(rid=f"{subset}:shot{i // 8}:{i % 8}", source=f"os_atlas/{subset}", platform="desktop",
                image=f"images/{subset}/shot{i // 8}.png") for i in range(400)]
    meta.assert_clean_manifest(rows)
    assert meta.load_manifest(write_jsonl(tmp_path / "m.jsonl", rows)) == rows
    assert meta.classify_source(f"os_atlas/{subset}") == "raw"
    picked = [r for r in rows if meta.is_selected(r)]
    assert 0.5 <= len(picked) / len(rows) <= 0.7
    assert all(meta.is_selected(r) == (meta.stable_pct(r["id"]) < meta.SELECT_PCT) for r in rows)


def test_load_manifest_reads_without_filtering(manifest_path):
    rows = meta.load_manifest(manifest_path)
    lines = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert rows == lines and len(rows) == 48


def test_load_manifest_duplicate_ids(tmp_path):
    with pytest.raises(meta.ManifestError, match="duplicate id"):
        meta.load_manifest(write_jsonl(tmp_path / "m.jsonl", [row(), row()]))


@pytest.mark.parametrize("change", [{"bbox": [1, 2, 3]}, {"bbox": [1, 2, "3", 4]}, {"bbox": None}, {"img_w": 0},
                                    {"bbox": [1, 2, True, 4]}, {"id": 5}, {"id": ""}, {"source": None},
                                    {"instruction": None}])
def test_load_manifest_bad_fields(tmp_path, change):
    with pytest.raises(meta.ManifestError):
        meta.load_manifest(write_jsonl(tmp_path / "m.jsonl", [row(**change)]))


def test_load_manifest_missing_key_and_bad_json(tmp_path):
    r = row()
    del r["platform"]
    with pytest.raises(meta.ManifestError, match="platform"):
        meta.load_manifest(write_jsonl(tmp_path / "a.jsonl", [r]))
    (tmp_path / "b.jsonl").write_text(json.dumps(row()) + "\n{not json\n")
    with pytest.raises(meta.ManifestError, match=":2"):
        meta.load_manifest(tmp_path / "b.jsonl")


def test_load_manifest_unknown_source(tmp_path):
    p = write_jsonl(tmp_path / "m.jsonl", [row(source="os_atlas/ricosca")])
    with pytest.raises(meta.ManifestError, match="unknown source"):
        meta.load_manifest(p)
    assert len(meta.load_manifest(p, extra_natural=("os_atlas/ricosca",))) == 1


def test_index_by_image_keeps_order():
    rows = [row(rid="a:0", image="i1"), row(rid="b:0", image="i2"), row(rid="a:1", image="i1")]
    by = meta.index_by_image(rows)
    assert list(by) == ["i1", "i2"] and [r["id"] for r in by["i1"]] == ["a:0", "a:1"]


# --- control type / kind -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("role,data_type,expected", [
    ("android.widget.ImageButton", None, "ImageButton"),
    (None, "Button", "Button"),
    ("", "ControlType.Edit", "Edit"),
    (None, "text", None),
    ("icon", None, None),
    ("icon", "android.widget.CheckBox", "CheckBox"),
    (None, None, None),
    ("link", "text", "link"),
])
def test_control_type(role, data_type, expected):
    assert meta.control_type(row(role=role, data_type=data_type)) == expected


def test_control_type_missing_role_key():
    r = row(data_type="android.widget.Switch")
    del r["role"]
    assert meta.control_type(r) == "Switch"


def test_element_kind():
    assert meta.element_kind(row(data_type="icon")) == "icon"
    assert meta.element_kind(row(data_type="Text")) == "text"
    assert meta.element_kind(row(data_type="android.widget.Button")) is None
    assert meta.element_kind(row(data_type=None, role="icon")) == "icon"


# --- geometry ------------------------------------------------------------------------------------------------------

def box_at(cx, cy, half=5):
    return [cx - half, cy - half, cx + half, cy + half]


@pytest.mark.parametrize("cx,cy,expected", [
    (50, 50, "top-left"), (150, 50, "top-center"), (250, 50, "top-right"),
    (50, 150, "middle-left"), (150, 150, "center"), (250, 150, "middle-right"),
    (50, 250, "bottom-left"), (150, 250, "bottom-center"), (250, 250, "bottom-right"),
])
def test_position_bucket_grid(cx, cy, expected):
    assert meta.position_bucket(box_at(cx, cy), 300, 300) == expected


@pytest.mark.parametrize("cx,cy,expected", [
    (100, 50, "top-center"),        # x == W/3 -> middle column
    (99.9, 50, "top-left"),
    (200, 50, "top-right"),         # x == 2W/3 -> right column
    (50, 100, "middle-left"),       # y == H/3 -> middle row
    (50, 200, "bottom-left"),
    (300, 300, "bottom-right"),     # center on the far edge clamps into the last cell
    (0, 0, "top-left"),
    (-20, 400, "bottom-left"),      # out-of-image centers clamp
])
def test_position_bucket_boundaries(cx, cy, expected):
    assert meta.position_bucket(box_at(cx, cy), 300, 300) == expected


def test_position_bucket_non_square_boundary():
    assert meta.position_bucket(box_at(640, 540), 1920, 1080) == "center"
    assert meta.position_bucket(box_at(1280, 720), 1920, 1080) == "bottom-right"
    assert meta.position_bucket([0, 0, 1280 / 3, 10], 640, 400) == "top-center"     # center x == W/3


@pytest.mark.parametrize("w,h,expected", [
    (70, 70, "small"), (50, 100, "medium"), (0, 100, "small"), (100, 299, "medium"), (100, 300, "large"),
    (1000, 1000, "large"),
])
def test_size_bucket_thresholds(w, h, expected):
    assert meta.size_bucket([10, 10, 10 + w, 10 + h], 1000, 1000) == expected


def test_size_bucket_inverted_box_is_small():
    assert meta.size_bucket([50, 50, 10, 10], 100, 100) == "small"


@pytest.mark.parametrize("other,expected", [
    ([300, 120, 400, 180], "right"),
    ([0, 120, 50, 180], "left"),
    ([120, 0, 180, 50], "above"),
    ([120, 300, 180, 400], "below"),
    ([140, 140, 160, 160], "overlapping"),     # other center inside target
    ([0, 0, 1000, 1000], "overlapping"),       # target center inside other
    ([250, 250, 350, 350], "right"),           # |dx| == |dy| -> horizontal
    ([-50, -50, 50, 50], "left"),
    ([110, 210, 190, 260], "below"),
])
def test_relative_direction(other, expected):
    assert meta.relative_direction([100, 100, 200, 200], other) == expected


def test_neighbors_sort_k_self_and_duplicates():
    img = "images/androidworld/s.png"
    target = row(rid="a:s:0", label="Share", bbox=box_at(100, 100), image=img)
    others = [
        row(rid="a:s:1", label="Far away", bbox=box_at(290, 290), image=img),
        row(rid="a:s:2", label="Share", bbox=box_at(130, 100), image=img),        # same label as target: kept
        row(rid="a:s:3", label="beta", bbox=box_at(100, 60), image=img),         # dist 40
        row(rid="a:s:4", label="alpha", bbox=box_at(140, 100), image=img),       # dist 40, label sorts first
        row(rid="a:s:6", label="alpha", bbox=box_at(60, 100), image=img),        # dist 40, same label, later id
        row(rid="a:s:5", label="  Two\n  words  ", bbox=box_at(100, 150), image=img),
        row(rid="a:s:7", label="x" * 80, bbox=box_at(100, 200), image=img),
    ]
    image_rows = [target] + others
    got = meta.neighbors(target, image_rows)
    assert [n["label"] for n in got] == ["Share", "alpha", "alpha", "beta", "Two words", "x" * 60, "Far away"]
    assert [n["direction"] for n in got] == ["right", "right", "left", "above", "below", "below", "right"]
    assert meta.neighbors(target, image_rows, k=2) == got[:2]
    assert meta.neighbors(target, image_rows, k=0) == []
    assert meta.neighbors(target, [target]) == []


def test_neighbors_cap_and_self_by_id():
    img = "i"
    rows = [row(rid=f"a:i:{k}", label="Same", bbox=box_at(10 + 20 * k, 10), image=img) for k in range(12)]
    got = meta.neighbors(rows[0], rows)
    assert len(got) == meta.MAX_NEIGHBORS == 8
    assert all(n["label"] == "Same" and n["direction"] == "right" for n in got)


def test_neighbors_use_original_label():
    a = row(rid="a:0", bbox=box_at(10, 10))
    b = row(rid="a:1", label="tap the settings gear", instruction_original="Settings", bbox=box_at(50, 10))
    assert meta.neighbors(a, [a, b])[0]["label"] == "Settings"


def test_normalize_label():
    assert meta.normalize_label("  a\tb \n c ") == "a b c"
    assert meta.normalize_label(42) == "42"
    assert meta.normalize_label("\u200e\u2068Classic\u2069") == "Classic"  # macOS bidi marks
    assert meta.normalize_label("Wi\u200b-Fi\ufeff") == "Wi-Fi"


@pytest.mark.parametrize("raw,noun", [
    ("push-button", "button"), ("link", "link"), ("menu", "menu"), ("tree-item", "tree item"), ("label", "text"),
    ("table-cell", "table cell"), ("page-tab", "tab"), ("check-box", "checkbox"), ("toggle-button", "toggle button"),
    ("combo-box", "dropdown"), ("menu-item", "menu item"), ("AXStaticText", "text"), ("AXTextField", "text field"),
    ("AXRadioButton", "radio button"), ("AXButton", "button"), ("AXGroup", "area"), ("AXCheckBox", "checkbox"),
    ("AXLink", "link"), ("AXPopUpButton", "dropdown"), ("AXTabGroup", "tab"), ("AXMenuItem", "menu item"),
    ("android.widget.TextView", "text"), ("android.widget.EditText", "text field"),
    ("android.widget.CheckBox", "checkbox"), ("android.widget.Button", "button"),
    ("android.widget.FrameLayout", "area"), ("android.widget.LinearLayout", "area"),
    ("android.widget.ImageButton", "button"), ("Hyperlink", "link"), ("something-new", "element"), (None, None),
])
def test_control_noun(raw, noun):
    assert meta.control_noun(raw if raw is None else meta.control_type({"role": raw})) == noun


def test_row_meta_shape_and_style():
    img = "images/androidworld/s.png"
    target = row(rid="a:s:0", label=" More  options ", bbox=box_at(250, 40), image=img,
                 role="android.widget.ImageButton")
    other = row(rid="a:s:1", label="Search", bbox=box_at(200, 40, 20), image=img)
    by = meta.index_by_image([target, other])
    m = meta.row_meta(target, by)
    assert list(m) == ["label", "platform", "control", "position", "size", "neighbors", "style"]
    assert m == {"label": "More options", "platform": "mobile", "control": "button", "position": "top-right",
                 "size": "small", "neighbors": [{"label": "Search", "direction": "left"}],
                 "style": meta.assign_style("a:s:0")}
    assert meta.row_meta(target, by, style="positional")["style"] == "positional"
    m2 = meta.row_meta(row(data_type="icon"), {})
    assert "control" not in m2 and m2["kind"] == "icon" and m2["neighbors"] == []


# --- synthetic fixtures --------------------------------------------------------------------------------------------

def test_synth_manifest_is_realistic(manifest_path):
    rows = meta.load_manifest(manifest_path)
    assert len({r["id"] for r in rows}) == len(rows) == 48
    assert all((manifest_path.parent / r["image_path"]).is_file() for r in rows)
    labels = [r["instruction"] for r in rows]
    assert {"42", "AutomationID:x", "GitHub"} <= set(labels)
    by = meta.index_by_image(rows)
    assert any(len({r["instruction"] for r in rs}) < len(rs) for rs in by.values())   # a duplicate label on a shot
    assert {r["platform"] for r in rows} == {"mobile", "web"}
    assert {meta.classify_source(r["source"]) for r in rows} == {"raw", "natural"}
    assert all(0 <= r["bbox"][0] < r["bbox"][2] <= r["img_w"] and 0 <= r["bbox"][1] < r["bbox"][3] <= r["img_h"]
               for r in rows)


def test_synth_deterministic_and_rows_variant(tmp_path):
    a = synth.make_rows(seed=3)
    assert a == synth.make_rows(seed=3) and a != synth.make_rows(seed=4)
    p = synth.make_manifest(tmp_path, n_images=4, per_image=3, seed=3, sources=("os_atlas/windows",),
                            write_images=False)
    rows = meta.load_manifest(p)
    assert len(rows) == 12 and not (tmp_path / "images").exists()
    assert {r["platform"] for r in rows} == {"desktop"}
