"""overlay_scan.py: pure-red fraction, flagging, summary counts."""

import json

from PIL import Image, ImageDraw

from data_prep.rewrite import overlay_scan


def _img(path, boxes=0, color=(255, 0, 0)):
    im = Image.new("RGB", (200, 100), (30, 30, 30))
    d = ImageDraw.Draw(im)
    for i in range(boxes):
        d.rectangle([5 + 30 * i, 10, 30 + 30 * i, 40], outline=color, width=2)
        d.rectangle([5 + 30 * i, 0, 30 + 30 * i, 8], fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def test_red_fraction(tmp_path):
    _img(tmp_path / "clean.png")
    _img(tmp_path / "overlay.png", boxes=3)
    _img(tmp_path / "dark_red.png", boxes=3, color=(200, 0, 0))
    assert overlay_scan.red_fraction(tmp_path / "clean.png") == 0
    assert overlay_scan.red_fraction(tmp_path / "overlay.png") > overlay_scan.THRESHOLD
    assert overlay_scan.red_fraction(tmp_path / "dark_red.png") == 0


def test_main_writes_flagged_list(tmp_path):
    _img(tmp_path / "images/linux/a.png", boxes=4)
    _img(tmp_path / "images/linux/b.png")
    rows = [{"id": "linux:a:0", "source": "os_atlas/linux", "image_path": "images/linux/a.png"},
            {"id": "linux:a:1", "source": "os_atlas/linux", "image_path": "images/linux/a.png"},
            {"id": "linux:b:0", "source": "os_atlas/linux", "image_path": "images/linux/b.png"},
            {"id": "fineweb:c:0", "source": "os_atlas/fineweb", "image_path": "images/fineweb/missing.png"}]
    _img(tmp_path / "images/uibert/red_ui.png", boxes=4)
    rows.append({"id": "uibert:r:0", "source": "os_atlas/uibert", "image_path": "images/uibert/red_ui.png"})
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert overlay_scan.main(["--manifest", str(tmp_path / "manifest.jsonl"), "--workers", "1"]) == 0
    out = json.loads((tmp_path / "overlay_flagged.json").read_text())
    assert [f["image_path"] for f in out["flagged"]] == ["images/linux/a.png"]
    assert out["flagged"][0]["rows"] == 2 and out["flagged"][0]["source"] == "os_atlas/linux"
    assert out["summary"]["by_source"]["os_atlas/linux"] == {"images": 2, "flagged_images": 1, "rows": 3,
                                                            "flagged_rows": 2, "other_candidates": 0}
    assert [e["image_path"] for e in out["errors"]] == ["images/fineweb/missing.png"]
    assert [c["image_path"] for c in out["other_candidates"]] == ["images/uibert/red_ui.png"]
    assert out["summary"]["by_source"]["os_atlas/uibert"]["flagged_images"] == 0
