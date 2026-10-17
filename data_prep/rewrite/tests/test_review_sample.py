"""Tests for review_sample.py: card counts, escaping, determinism, missing/corrupt images, crop geometry, CLI."""

import base64
import io
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from data_prep.rewrite import review_sample as rs

REPO = Path(__file__).resolve().parents[3]
EVIL = "<script>alert(\"x\")</script> & 'q'"
EVIL_ESCAPED = "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &#x27;q&#x27;"
EVIL_REWRITE = 'click "<b>bold</b>" now'
MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

WEB = ("fineweb", "page_0", (640, 400), "web", [
    ("Search", (20, 20, 120, 50), "search the site"),
    ("GitHub", (500, 20, 620, 50), "open GitHub"),
    (EVIL, (200, 180, 440, 220), EVIL_REWRITE),
    ("Sign in", (20, 350, 100, 390), "sign in to your account"),
    ("42", (300, 300, 340, 330), ("numeric",)),
])
MOBILE = ("androidworld", "screen_0", (360, 720), "mobile", [
    ("Back", (10, 10, 60, 60), "go back"),
    ("More options", (300, 10, 350, 60), "open more options"),
    ("Settings", (20, 300, 340, 360), "open the settings"),
    ("Home", (150, 650, 210, 700), ("duplicate_neighbor",)),
    ("AutomationID:x", (40, 500, 120, 540), ("unintelligible",)),
    ("Menu", (250, 500, 330, 540), ("llm_error",)),
])
N_KEPT, N_DROPS = 7, 3  # model drops, excluding the llm_error row


def make_run(tmp_path: Path, *, stats=True) -> tuple[Path, Path]:
    """Tiny manifest + PNGs under tmp/slice, hand-written rewrites.jsonl (+ stats.json) under tmp/out."""
    root, out = tmp_path / "slice", tmp_path / "out"
    out.mkdir(parents=True)
    rows, recs = [], []
    for subset, stem, (w, h), platform, targets in (WEB, MOBILE):
        rel = f"images/{subset}/{stem}.png"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        im = Image.new("RGB", (w, h), (238, 238, 232))
        d = ImageDraw.Draw(im)
        for k, (label, bbox, result) in enumerate(targets):
            d.rectangle(bbox, fill=(70, 110, 200) if k % 2 else (120, 120, 120))
            rid = f"{subset}:{stem}:{k}"
            rows.append({"id": rid, "source": f"os_atlas/{subset}", "image_path": rel, "img_w": w, "img_h": h,
                         "instruction": label, "bbox": list(bbox), "platform": platform, "data_type": None,
                         "role": "android.widget.ImageButton" if platform == "mobile" else None, "split": None})
            keep = isinstance(result, str)
            recs.append({"id": rid, "source": f"os_atlas/{subset}", "original": label,
                         "rewritten": result if keep else None, "style": ("functional", "descriptive", "positional")[k % 3],
                         "keep": keep, "drop_reason": None if keep else result[0], "prompt_version": "v1", "model": MODEL,
                         "ts": "2026-10-17T15:00:00+00:00"})
        im.save(root / rel)
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out / "rewrites.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    if stats:
        (out / "stats.json").write_text(json.dumps({
            "run": {"run_id": "run-1", "prompt_version": "v1", "model": MODEL, "model_label": None, "endpoint": "hosted",
                    "dry_run": True},
            "manifest": {"path": str(manifest), "sha256": "deadbeef", "rows": len(rows)},
            "rewrites": {"total": len(recs), "by_source": {
                "os_atlas/fineweb": {"targets": 5, "kept": 4, "dropped": 1, "drop_rate": 0.2},
                "os_atlas/androidworld": {"targets": 6, "kept": 3, "dropped": 3, "drop_rate": 0.5}}},
            "quality": {"first_try_valid_rate": 0.97, "batch_json_ok_rate": 1.0, "drop_rate_raw": 0.2727},
            "gate": {"first_try_valid_ok": True, "drop_rate_ok": False}}), encoding="utf-8")
    return manifest, out


def card_ids(text: str, kind: str) -> list[str]:
    return re.findall(rf'<section class="card {kind}" id="c\d+" data-id="([^"]*)"', text)


def data_urls(text: str, cls: str) -> list[Image.Image]:
    return [Image.open(io.BytesIO(base64.b64decode(b64)))
            for b64 in re.findall(rf'<img class="{cls}" src="data:image/jpeg;base64,([^"]+)"', text)]


def test_card_counts_respect_n_and_drops(tmp_path):
    manifest, out = make_run(tmp_path)
    text = rs.build_review(out, manifest, n=3, drops=2).read_text(encoding="utf-8")
    assert len(card_ids(text, "kept")) == 3 and len(card_ids(text, "drop")) == 2
    text = rs.build_review(out, manifest, n=50, drops=20).read_text(encoding="utf-8")
    kept, drops = card_ids(text, "kept"), card_ids(text, "drop")
    assert len(kept) == N_KEPT and len(set(kept)) == N_KEPT
    assert len(drops) == N_DROPS and "androidworld:screen_0:5" not in drops  # llm_error rows are not drop judgments
    assert text.count('data-kind="kept"') == N_KEPT and text.count('data-kind="drop"') == N_DROPS
    text = rs.build_review(out, manifest, n=0, drops=0).read_text(encoding="utf-8")
    assert card_ids(text, "kept") == [] and card_ids(text, "drop") == []


def test_all_text_is_escaped(tmp_path):
    manifest, out = make_run(tmp_path)
    text = rs.build_review(out, manifest, n=50).read_text(encoding="utf-8")
    assert "<script>alert(" not in text and "<b>bold</b>" not in text
    assert EVIL_ESCAPED in text
    assert "click &quot;&lt;b&gt;bold&lt;/b&gt;&quot; now" in text
    assert text.count("<script>") == 1  # only the sheet's own script
    assert text.count(EVIL_ESCAPED) >= 3  # its own card (label + data attribute) and as a neighbor on other cards


def test_deterministic_for_seed(tmp_path):
    manifest, out = make_run(tmp_path)
    a = rs.build_review(out, manifest, n=4, seed=0, drops=2, html_path=tmp_path / "a.html").read_bytes()
    b = rs.build_review(out, manifest, n=4, seed=0, drops=2, html_path=tmp_path / "b.html").read_bytes()
    assert a == b
    c = rs.build_review(out, manifest, n=4, seed=1, drops=2, html_path=tmp_path / "c.html").read_text(encoding="utf-8")
    assert card_ids(a.decode("utf-8"), "kept") != card_ids(c, "kept")


def test_sample_rows_matches_seeded_sample_over_sorted_ids():
    recs = [{"id": f"x:{i:02d}", "keep": True, "rewritten": f"open {i}"} for i in range(30)][::-1]
    recs += [{"id": "y:0", "keep": False, "drop_reason": "numeric"}, {"id": "y:1", "keep": False, "drop_reason": "llm_error"},
             {"id": "y:2", "keep": True, "rewritten": None}]
    kept, drops = rs.sample_rows(recs, 5, 7, 5)
    assert kept == random.Random(7).sample(sorted(recs[:30], key=lambda r: r["id"]), 5)
    assert [r["id"] for r in drops] == ["y:0"]


def test_missing_and_corrupt_images_render_placeholders(tmp_path):
    manifest, out = make_run(tmp_path)
    (manifest.parent / "images/androidworld/screen_0.png").unlink()
    text = rs.build_review(out, manifest, n=50, drops=0).read_text(encoding="utf-8")
    assert "image missing: images/androidworld/screen_0.png" in text
    assert len(card_ids(text, "kept")) == N_KEPT and len(data_urls(text, "crop")) == 4  # the 4 kept web rows
    (manifest.parent / "images/fineweb/page_0.png").write_bytes(b"not a png")
    text = rs.build_review(out, manifest, n=50).read_text(encoding="utf-8")
    assert "image error:" in text and data_urls(text, "crop") == []


def test_rewrite_row_missing_from_manifest(tmp_path):
    manifest, out = make_run(tmp_path)
    lines = manifest.read_text(encoding="utf-8").splitlines()
    manifest.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")  # drop fineweb:page_0:0 ("Search")
    text = rs.build_review(out, manifest, n=50).read_text(encoding="utf-8")
    assert "row not in manifest" in text and len(card_ids(text, "kept")) == N_KEPT


def test_crop_and_thumbnail_sizes_and_box_drawn(tmp_path):
    manifest, out = make_run(tmp_path)
    text = rs.build_review(out, manifest, n=50).read_text(encoding="utf-8")
    crops, thumbs = data_urls(text, "crop"), data_urls(text, "thumb")
    assert len(crops) == len(thumbs) == N_KEPT + N_DROPS
    assert all(im.width <= 360 and im.height <= 360 for im in crops)
    assert all(im.width <= 220 and im.height <= 260 for im in thumbs)
    for im in crops + thumbs:
        red = sum(1 for r, g, b in im.convert("RGB").getdata() if r > 180 and g < 80 and b < 80)
        assert red > 20


def test_crop_box_geometry():
    assert rs.crop_box((300, 300, 340, 330), 1000, 800) == (160, 155, 480, 475)  # 40x30 box -> 320x320 centered
    assert rs.crop_box((0, 0, 20, 20), 1000, 800) == (0, 0, 320, 320)  # shifted into the image at the corner
    assert rs.crop_box((900, 700, 1000, 800), 1000, 800) == (680, 480, 1000, 800)
    assert rs.crop_box((100, 100, 300, 200), 1000, 800) == (0, 0, 600, 320)  # 3x box: 600 wide, clamped at left
    assert rs.crop_box((10, 10, 50, 50), 200, 150) == (0, 0, 200, 150)  # image smaller than the minimum window
    assert rs.crop_box((0, 0, 1000, 800), 1000, 800) == (0, 0, 1000, 800)


def test_header_and_default_manifest_from_stats(tmp_path):
    _, out = make_run(tmp_path)
    text = rs.build_review(out).read_text(encoding="utf-8")  # manifest path read from stats.json
    assert "prompt <code>v1</code>" in text and MODEL in text
    assert "first-try valid <b>97.0%</b>" in text and "os_atlas/androidworld <b>50.0%</b>" in text
    assert "drop_rate_ok: FAIL" in text and "first_try_valid_ok: ok" in text
    assert 'data-key="t3-review:' in text and 'data-manifest-sha256="deadbeef"' in text


def test_images_root_override(tmp_path):
    manifest, out = make_run(tmp_path)
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    (manifest.parent / "images").rename(moved / "images")
    assert "image missing" in rs.build_review(out, manifest, n=50).read_text(encoding="utf-8")
    text = rs.build_review(out, manifest, images_root=moved, n=50).read_text(encoding="utf-8")
    assert "image missing" not in text and len(data_urls(text, "crop")) == N_KEPT + N_DROPS


def test_cache_warnings_shown(tmp_path):
    cache_mod = pytest.importorskip("data_prep.rewrite.cache")
    manifest, out = make_run(tmp_path)
    recs = rs.read_jsonl(out / "rewrites.jsonl")
    with cache_mod.RewriteCache(out / "cache.sqlite") as cache:
        cache.put_many("v1", [dict(r, attempts=2 if r["original"] == "GitHub" else 1, first_try_valid=True,
                                   warnings=["proper_noun:GitHub"] if r["original"] == "GitHub" else [], errors=[])
                              for r in recs])
    text = rs.build_review(out, manifest, n=50).read_text(encoding="utf-8")
    assert '<span class="chip warn">proper_noun:GitHub</span>' in text
    assert '<span class="chip warn">attempts: 2</span>' in text


def test_no_stats_and_no_manifest_fails_cleanly(tmp_path, capsys):
    manifest, out = make_run(tmp_path, stats=False)
    with pytest.raises(FileNotFoundError):
        rs.build_review(out)
    assert rs.main(["--out", str(out)]) == 1
    assert "--manifest" in capsys.readouterr().err
    text = rs.build_review(out, manifest, n=2).read_text(encoding="utf-8")
    assert "no stats.json" in text and len(card_ids(text, "kept")) == 2


def test_main_writes_default_html(tmp_path, capsys):
    manifest, out = make_run(tmp_path)
    assert rs.main(["--out", str(out), "--manifest", str(manifest), "--n", "2", "--drops", "1", "--seed", "3"]) == 0
    assert (out / "review.html").is_file()
    assert "2 kept cards, 1 drop cards" in capsys.readouterr().out


def test_script_entrypoint_runs_from_repo_root(tmp_path):
    manifest, out = make_run(tmp_path)
    html_path = tmp_path / "sheet" / "r.html"
    res = subprocess.run([sys.executable, "data_prep/rewrite/review_sample.py", "--out", str(out), "--html", str(html_path)],
                         cwd=REPO, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    assert html_path.is_file()


def test_unofficial_banner_and_only_boolean_gate_chips(tmp_path):
    _, out = make_run(tmp_path)
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    stats["run"].update(unofficial_model=True, unofficial_roots=["mlx-community/Lightning-4bit"])
    stats["gate"].update(first_try_valid_min=0.95, faithfulness="manual: review_sample.py --n 50")
    (out / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    text = rs.build_review(out).read_text(encoding="utf-8")
    assert "NOT FOR PUBLICATION: unofficial/community checkpoint (mlx-community/Lightning-4bit)" in text
    assert "first_try_valid_min:" not in text and "faithfulness:" not in text and "drop_rate_ok: FAIL" in text
    clean = rs.build_review(make_run(tmp_path / "b")[1]).read_text(encoding="utf-8")
    assert "NOT FOR PUBLICATION" not in clean
