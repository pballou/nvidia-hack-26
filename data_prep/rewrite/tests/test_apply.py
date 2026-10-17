"""Tests for apply.py: rewrite application, the template ablation variant, and the CLI outputs."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from data_prep.rewrite import apply, meta, validate
from data_prep.rewrite.apply import MissingRewriteError, apply_rewrites, template_variant

APPLY_PY = Path(apply.__file__).resolve()
NEW_KEYS = ["instruction_original", "instruction_style"]


def selected_by_hash(rid: str) -> bool:
    """The contract's selection rule, computed independently of meta.py."""
    return int(hashlib.sha256(rid.encode("utf-8")).hexdigest(), 16) % 100 < 60


def pick_ids(prefix: str, selected: bool, n: int) -> list[str]:
    out, i = [], 0
    while len(out) < n:
        rid = f"{prefix}:shot{i // 4}:{i % 4}"
        if selected_by_hash(rid) == selected:
            out.append(rid)
        i += 1
    return out


def make_row(rid: str, source: str, instruction: str, platform: str = "mobile") -> dict:
    subset = source.split("/")[-1]
    return {"id": rid, "source": source, "image_path": f"images/{subset}/{rid.split(':')[1]}.png", "img_w": 1080,
            "img_h": 2400, "instruction": instruction, "bbox": [10.0, 20.0, 110.0, 120.0], "platform": platform,
            "data_type": None, "role": "android.widget.ImageButton", "split": None}


def make_rec(row: dict, keep: bool = True, rewritten: str | None = None, style: str = "functional",
             drop_reason: str | None = None, model: str | None = None) -> dict:
    """keep=False records default to rule drops (removed); pass a model name for a model drop (raw fallback)."""
    if keep and rewritten is None:
        rewritten = "open " + row["instruction"].lower()
    model = model or (validate.RULE_MODEL if not keep else "nvidia/nemotron-3.5-lightning-30b-a3b")
    return {"id": row["id"], "source": row["source"], "original": row["instruction"],
            "rewritten": rewritten if keep else None, "style": style, "keep": keep,
            "drop_reason": None if keep else drop_reason, "prompt_version": "v1",
            "model": model, "ts": "2026-10-17T18:00:00+00:00"}


@pytest.fixture
def ids():
    return {"aw_sel": pick_ids("androidworld", True, 3), "aw_raw": pick_ids("androidworld", False, 2),
            "fw_sel": pick_ids("fineweb", True, 2), "fw_raw": pick_ids("fineweb", False, 1),
            "ui_sel": pick_ids("uibert", True, 1), "ui_raw": pick_ids("uibert", False, 1)}


@pytest.fixture
def manifest(ids):
    aw, fw, ui = "os_atlas/androidworld", "os_atlas/fineweb", "os_atlas/uibert"
    return [make_row(ids["aw_sel"][0], aw, "42"),
            make_row(ids["fw_raw"][0], fw, "Pricing", "web"),
            make_row(ids["aw_sel"][1], aw, "More options"),
            make_row(ids["ui_sel"][0], ui, "tap the search icon"),
            make_row(ids["aw_raw"][0], aw, "Navigate up"),
            make_row(ids["fw_sel"][0], fw, "GitHub", "web"),
            make_row(ids["aw_sel"][2], aw, "Settings"),
            make_row(ids["ui_raw"][0], ui, "open the menu"),
            make_row(ids["fw_sel"][1], fw, "Sign in", "web"),
            make_row(ids["aw_raw"][1], aw, "Back")]


@pytest.fixture
def rewrites(manifest, ids):
    """keep=false for the numeric label, a descriptive rewrite for GitHub, functional for the rest."""
    by_id = {r["id"]: r for r in manifest}
    recs = [make_rec(by_id[ids["aw_sel"][0]], keep=False, drop_reason="numeric"),
            make_rec(by_id[ids["aw_sel"][1]], rewritten="open more options"),
            make_rec(by_id[ids["fw_sel"][0]], rewritten="the GitHub link", style="descriptive"),
            make_rec(by_id[ids["aw_sel"][2]], rewritten="the settings button at the top right", style="positional"),
            make_rec(by_id[ids["fw_sel"][1]], rewritten="sign in to your account")]
    return recs[::-1]  # record order must not matter


def test_helper_selection_matches_meta(manifest, ids):
    for r in manifest:
        assert meta.is_selected(r) == (r["id"] in ids["aw_sel"] + ids["fw_sel"])


def test_keep_rewritten_replaces_instruction(manifest, rewrites, ids):
    out = {r["id"]: r for r in apply_rewrites(manifest, rewrites)}
    r = out[ids["aw_sel"][1]]
    assert r["instruction"] == "open more options"
    assert r["instruction_original"] == "More options"
    assert r["instruction_style"] == "functional"
    assert list(r) == list(manifest[2]) + NEW_KEYS
    assert {k: v for k, v in r.items() if k not in NEW_KEYS and k != "instruction"} == \
        {k: v for k, v in manifest[2].items() if k != "instruction"}
    assert out[ids["fw_sel"][0]]["instruction"] == "the GitHub link"
    assert out[ids["fw_sel"][0]]["instruction_style"] == "descriptive"
    assert out[ids["aw_sel"][2]]["instruction_style"] == "positional"


def test_keep_false_selected_rows_dropped(manifest, rewrites, ids):
    train = apply_rewrites(manifest, rewrites)
    assert ids["aw_sel"][0] not in {r["id"] for r in train}
    assert [r["id"] for r in train] == [r["id"] for r in manifest if r["id"] != ids["aw_sel"][0]]


def test_natural_rows_pass_through(manifest, rewrites, ids):
    out = {r["id"]: r for r in apply_rewrites(manifest, rewrites)}
    for rid, text in ((ids["ui_sel"][0], "tap the search icon"), (ids["ui_raw"][0], "open the menu")):
        assert out[rid]["instruction"] == out[rid]["instruction_original"] == text
        assert out[rid]["instruction_style"] == "natural"


def test_unselected_raw_rows_pass_through(manifest, rewrites, ids):
    out = {r["id"]: r for r in apply_rewrites(manifest, rewrites)}
    for rid in ids["aw_raw"] + ids["fw_raw"]:
        assert out[rid]["instruction"] == out[rid]["instruction_original"]
        assert out[rid]["instruction_style"] == "raw"


def test_inputs_not_mutated_and_dict_input_equivalent(manifest, rewrites):
    m0, r0 = copy.deepcopy(manifest), copy.deepcopy(rewrites)
    a = apply_rewrites(manifest, rewrites)
    b = apply_rewrites(manifest, {r["id"]: r for r in rewrites})
    assert a == b
    assert manifest == m0 and rewrites == r0
    assert a == apply_rewrites(manifest, rewrites)


def test_missing_rewrite_raises_unless_allowed(manifest, rewrites, ids):
    partial = [r for r in rewrites if r["id"] != ids["fw_sel"][1]]
    with pytest.raises(MissingRewriteError, match="--allow-missing"):
        apply_rewrites(manifest, partial)
    assert issubclass(MissingRewriteError, ValueError)
    out = {r["id"]: r for r in apply_rewrites(manifest, partial, allow_missing=True)}
    assert out[ids["fw_sel"][1]]["instruction"] == "Sign in"
    assert out[ids["fw_sel"][1]]["instruction_style"] == "raw"
    assert len(out) == len(manifest) - 1


def test_original_mismatch_raises(manifest, rewrites, ids):
    changed = copy.deepcopy(manifest)
    next(r for r in changed if r["id"] == ids["aw_sel"][1])["instruction"] = "More options menu"
    with pytest.raises(ValueError, match="manifest changed"):
        apply_rewrites(changed, rewrites)


def test_rewrite_for_unknown_id_raises(manifest, rewrites):
    extra = make_rec(make_row("androidworld:ghost:0", "os_atlas/androidworld", "Ghost"))
    with pytest.raises(ValueError, match="not in the manifest"):
        apply_rewrites(manifest, rewrites + [extra])


@pytest.mark.parametrize("key", ["aw_raw", "fw_raw", "ui_sel", "ui_raw"])
def test_rewrite_for_unselected_id_raises(manifest, rewrites, ids, key):
    row = next(r for r in manifest if r["id"] == ids[key][0])
    with pytest.raises(ValueError, match="not selected"):
        apply_rewrites(manifest, rewrites + [make_rec(row)])


def test_bad_records_raise(manifest, rewrites):
    with pytest.raises(ValueError, match="duplicate"):
        apply_rewrites(manifest, rewrites + [rewrites[0]])
    bad = copy.deepcopy(rewrites)
    bad[0]["keep"] = "true"
    with pytest.raises(ValueError, match="non-boolean keep"):
        apply_rewrites(manifest, bad)
    with pytest.raises(ValueError, match="has id"):
        apply_rewrites(manifest, {"androidworld:x:0": rewrites[0]})


def test_already_applied_manifest_rejected(manifest, rewrites):
    train = apply_rewrites(manifest, rewrites)
    with pytest.raises(ValueError, match="already have rewrites applied"):
        apply_rewrites(train, [])


def test_contaminated_manifest_rejected(manifest, rewrites):
    bad = manifest + [make_row("screenspot:x:0", "os_atlas/androidworld", "Search")]
    with pytest.raises(meta.ContaminationError):
        apply_rewrites(bad, rewrites)


def test_template_variant_identical_rows_with_original_labels(manifest, rewrites):
    train = apply_rewrites(manifest, rewrites)
    template = template_variant(train)
    assert [r["id"] for r in template] == [r["id"] for r in train]
    for t, m in zip(template, train):
        assert t["instruction"] == m["instruction_original"]
        assert t["instruction_style"] == ("natural" if m["instruction_style"] == "natural" else "raw")
        assert list(t) == list(m)
        assert {k: v for k, v in t.items() if k not in ("instruction", "instruction_style")} == \
            {k: v for k, v in m.items() if k not in ("instruction", "instruction_style")}
    assert template_variant(train) == template
    with pytest.raises(ValueError, match="instruction_original"):
        template_variant(manifest)


def test_extra_raw_and_natural_overrides(manifest, rewrites):
    raw_src, nat_src = "custom/raw_labels", "custom/natural"
    raw_rows = [make_row(rid, raw_src, f"Label {i}") for i, rid in enumerate(pick_ids("customraw", True, 2))]
    raw_rows += [make_row(rid, raw_src, "Unselected") for rid in pick_ids("customraw", False, 1)]
    nat_rows = [make_row(rid, nat_src, "tap the cart") for rid in pick_ids("customnat", True, 1)]
    rows = manifest + raw_rows + nat_rows
    with pytest.raises(ValueError):
        apply_rewrites(rows, rewrites)
    kw = {"extra_raw": (raw_src,), "extra_natural": (nat_src,)}
    with pytest.raises(MissingRewriteError):
        apply_rewrites(rows, rewrites, **kw)
    recs = rewrites + [make_rec(raw_rows[0]), make_rec(raw_rows[1], keep=False, drop_reason="unintelligible")]
    out = {r["id"]: r for r in apply_rewrites(rows, recs, **kw)}
    assert out[raw_rows[0]["id"]]["instruction"] == "open label 0"
    assert out[raw_rows[0]["id"]]["instruction_style"] == "functional"
    assert raw_rows[1]["id"] not in out
    assert out[raw_rows[2]["id"]]["instruction_style"] == "raw"
    assert out[nat_rows[0]["id"]]["instruction_style"] == "natural"
    with pytest.raises(ValueError, match="not selected"):
        apply_rewrites(rows, recs + [make_rec(nat_rows[0])], **kw)


def write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def run_main(tmp_path: Path, manifest, rewrites, data_dir: str, *extra) -> tuple[int, Path]:
    """Write tmp_path/<data_dir>/manifest.jsonl and tmp_path/rewrites.jsonl; apply.py writes next to the manifest."""
    out_dir = tmp_path / data_dir
    out_dir.mkdir(exist_ok=True)
    m = write_jsonl(out_dir / "manifest.jsonl", manifest)
    rw = write_jsonl(tmp_path / "rewrites.jsonl", rewrites)
    return apply.main(["--manifest", str(m), "--rewrites", str(rw), *extra]), out_dir


FILES = ("train_lightning.jsonl", "train_template.jsonl", "apply_stats.json")


def test_main_writes_outputs_deterministically(tmp_path, manifest, rewrites, ids, capsys):
    code, out1 = run_main(tmp_path, manifest, rewrites, "out1")
    assert code == 0
    first = {f: (out1 / f).read_bytes() for f in FILES}
    assert run_main(tmp_path, manifest, rewrites, "out1")[0] == 0
    assert {f: (out1 / f).read_bytes() for f in FILES} == first
    _, out2 = run_main(tmp_path, manifest, rewrites, "out2")  # another copy of the manifest: same train rows
    assert {f: (out2 / f).read_bytes() for f in FILES[:2]} == {f: first[f] for f in FILES[:2]}
    assert not list(out1.glob("*.tmp"))

    lightning = [json.loads(line) for line in first["train_lightning.jsonl"].decode().splitlines()]
    template = [json.loads(line) for line in first["train_template.jsonl"].decode().splitlines()]
    assert lightning == apply_rewrites(manifest, rewrites)
    assert template == template_variant(lightning)
    assert [r["image_path"] for r in lightning] == [r["image_path"] for r in manifest if r["id"] != ids["aw_sel"][0]]

    stats = json.loads(first["apply_stats.json"])
    assert stats["image_root"] == str(out1.resolve())
    assert stats["rows"] == {"manifest": 10, "train": 9, "dropped": 1, "missing": 0}
    assert stats["by_style"] == {"descriptive": 1, "functional": 2, "natural": 2, "positional": 1, "raw": 3}
    assert {k: v for k, v in stats["dropped"].items() if k != "policy"} == {"by_source": {"os_atlas/androidworld": 1}, "by_reason": {"numeric": 1}}
    assert stats["by_source"]["os_atlas/androidworld"]["manifest"] == 5
    assert stats["by_source"]["os_atlas/androidworld"]["train"] == 4
    assert stats["rewrites"]["records"] == 5
    assert stats["rewrites"]["prompt_versions"] == {"v1": 5}
    assert stats["manifest"]["sha256"] == hashlib.sha256((out1 / "manifest.jsonl").read_bytes()).hexdigest()
    for f in FILES[:2]:
        assert stats["outputs"][f] == {"rows": 9, "sha256": hashlib.sha256(first[f]).hexdigest()}
    assert "9 train rows" in capsys.readouterr().out


def test_out_defaults_to_the_manifest_dir_and_elsewhere_is_refused(tmp_path, manifest, rewrites, capsys, monkeypatch):
    data = tmp_path / "data"
    elsewhere = tmp_path / "elsewhere"
    code, _ = run_main(tmp_path, manifest, rewrites, "data", "--out", str(elsewhere))
    err = capsys.readouterr().err
    assert code == 1 and "manifest's directory" in err and "training/sft.py" in err and str(data.resolve()) in err
    assert not elsewhere.exists() and not (data / "train_lightning.jsonl").exists()
    assert run_main(tmp_path, manifest, rewrites, "data", "--out", str(data / "train"))[0] == 1  # a subdir neither

    monkeypatch.chdir(data)  # the same directory spelled differently (relative, trailing "/.", via a symlink) is fine
    assert run_main(tmp_path, manifest, rewrites, "data", "--out", ".")[0] == 0
    assert run_main(tmp_path, manifest, rewrites, "data", "--out", f"{data}/sub/..")[0] == 0
    (tmp_path / "link").symlink_to(data, target_is_directory=True)
    assert run_main(tmp_path, manifest, rewrites, "data", "--out", str(tmp_path / "link"))[0] == 0
    assert sorted(p.name for p in data.iterdir()) == ["apply_stats.json", "manifest.jsonl", "train_lightning.jsonl",
                                                      "train_template.jsonl"]
    assert apply.resolve_out(data / "manifest.jsonl", None) == data.resolve()


def test_main_missing_rewrites_fails_unless_allowed(tmp_path, manifest, rewrites, capsys):
    partial = rewrites[1:]
    code, out = run_main(tmp_path, manifest, partial, "out")
    assert code == 1
    assert "--allow-missing" in capsys.readouterr().err
    assert not (out / "train_lightning.jsonl").exists()
    code, out = run_main(tmp_path, manifest, partial, "out", "--allow-missing")
    assert code == 0
    stats = json.loads((out / "apply_stats.json").read_text())
    assert stats["rows"]["missing"] == 1 and stats["allow_missing"] is True


def test_main_source_overrides(tmp_path, manifest, rewrites, capsys):
    rows = manifest + [make_row(rid, "custom/natural", "tap the cart") for rid in pick_ids("customnat", True, 1)]
    assert run_main(tmp_path, rows, rewrites, "out")[0] == 1
    assert "error:" in capsys.readouterr().err
    code, out = run_main(tmp_path, rows, rewrites, "out", "--natural-sources", "custom/natural, other/src")
    assert code == 0
    stats = json.loads((out / "apply_stats.json").read_text())
    assert stats["natural_sources"] == ["custom/natural", "other/src"]
    assert stats["by_style"]["natural"] == 3


def test_runs_as_a_script_from_any_cwd(tmp_path):
    proc = subprocess.run([sys.executable, str(APPLY_PY), "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "--allow-missing" in proc.stdout


def test_sampler_instruction_style_kept_on_natural_rows_and_not_mistaken_for_applied():
    """The sampler emits instruction_style on every row (GroundCUA icon/functional/text, else None)."""
    base = {"image_path": "images/x.png", "img_w": 100, "img_h": 100, "bbox": [1, 1, 9, 9], "data_type": None,
            "split": None}
    rows = [{**base, "id": "groundcua:app_1:0", "source": "groundcua/groundcua", "platform": "desktop",
             "instruction": "click the gear icon at the top right", "instruction_style": "icon"},
            {**base, "id": "uibert:x:0", "source": "os_atlas/uibert", "platform": "mobile",
             "instruction": "open the menu", "instruction_style": None}]
    raw = [{**base, "id": f"fineweb:p{i}:0", "source": "os_atlas/fineweb", "platform": "web",
            "instruction": f"Link {i}", "instruction_style": None} for i in range(40)]
    unselected = [r for r in raw if not meta.is_selected(r)]
    assert unselected
    train = apply_rewrites(rows + unselected, [])
    assert [r["instruction_style"] for r in train[:2]] == ["icon", "natural"]
    assert {r["instruction_style"] for r in train[2:]} == {"raw"}
    template = template_variant(train)
    assert [r["instruction_style"] for r in template[:2]] == ["icon", "natural"]


def test_splits_filter_keeps_only_listed_splits(tmp_path):
    base = {"image_path": "images/x.png", "img_w": 100, "img_h": 100, "bbox": [1, 1, 9, 9], "data_type": None,
            "platform": "mobile", "source": "os_atlas/uibert", "instruction_style": None}
    rows = [{**base, "id": f"uibert:s{i}:0", "instruction": f"open item {i}", "split": "train" if i % 3 else "dev"}
            for i in range(9)]
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "rewrites.jsonl").write_text("")
    assert apply.main(["--manifest", str(tmp_path / "manifest.jsonl"), "--rewrites", str(tmp_path / "rewrites.jsonl"),
                       "--splits", "train"]) == 0
    for name in ("train_lightning.jsonl", "train_template.jsonl"):
        out = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
        assert {r["split"] for r in out} == {"train"} and len(out) == 6
    stats = json.loads((tmp_path / "apply_stats.json").read_text())
    assert stats["manifest"]["rows"] == 9 and stats["manifest"]["rows_in_splits"] == 6


def test_model_drops_keep_the_raw_label_and_rule_drops_are_removed(manifest, ids):
    by_id = {r["id"]: r for r in manifest}
    sel = [rid for rid in by_id if meta.is_selected(by_id[rid])]
    model_drop, llm_error, rule_drop, *rest = sel
    recs = [make_rec(by_id[model_drop], keep=False, drop_reason="unintelligible", model="NVIDIA-hosted API, x"),
            make_rec(by_id[llm_error], keep=False, drop_reason="llm_error", model="NVIDIA-hosted API, x"),
            make_rec(by_id[rule_drop], keep=False, drop_reason="numeric"),
            *(make_rec(by_id[rid]) for rid in rest)]
    train = {r["id"]: r for r in apply_rewrites(manifest, recs)}
    assert rule_drop not in train
    for rid, reason in ((model_drop, "unintelligible"), (llm_error, "llm_error")):
        r = train[rid]
        assert r["instruction"] == r["instruction_original"] == by_id[rid]["instruction"]
        assert r["instruction_style"] == "raw" and r[apply.RAW_FALLBACK_KEY] == reason
    template = template_variant(list(train.values()))
    assert [r["id"] for r in template] == list(train) and all(r["instruction"] == r["instruction_original"]
                                                              for r in template)
    stats = apply.build_stats(manifest, recs, list(train.values()), [], [])
    assert stats["model_drop_raw_fallback"]["rows"] == 2
    assert stats["model_drop_raw_fallback"]["by_reason"] == {"llm_error": 1, "unintelligible": 1}
