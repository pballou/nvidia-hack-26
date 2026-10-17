"""Tests for review_sample.py: card counts, escaping, determinism, missing/corrupt images, crop and zoom geometry, CLI,
and the sheet's JavaScript (verdicts, fingerprints, tally, keyboard cursor, export/copy) under node with a tiny DOM."""

import base64
import hashlib
import html
import io
import json
import random
import re
import shutil
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
             {"id": "y:2", "keep": True, "rewritten": None},
             {"id": "y:3", "keep": False, "drop_reason": "numeric", "model": rs.RULE_MODEL}]  # rule drop: never sent
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
    crops, thumbs, zooms = data_urls(text, "crop"), data_urls(text, "thumb"), data_urls(text, "zoom")
    assert len(crops) == len(thumbs) == N_KEPT + N_DROPS
    assert len(zooms) == 1  # only the 240 px wide EVIL box is shrunk by its crop; the rest are shown at 1x already
    assert all(im.width <= 360 and im.height <= 360 for im in crops)
    assert all(im.width <= 220 and im.height <= 260 for im in thumbs)
    for im in crops + thumbs + zooms:
        red = sum(1 for r, g, b in im.convert("RGB").getdata() if r > 180 and g < 80 and b < 80)
        assert red > 20
    for cls, im in [("crop", crops[0]), ("thumb", thumbs[0]), ("zoom", zooms[0])]:  # sizes fixed before decoding
        assert f'<img class="{cls}" src="data:image/jpeg;base64,' in text
        assert re.search(rf'<img class="{cls}" src="[^"]+" width="{im.width}" height="{im.height}"', text)


def test_crop_box_geometry():
    assert rs.crop_box((300, 300, 340, 330), 1000, 800) == (160, 155, 480, 475)  # 40x30 box -> 320x320 centered
    assert rs.crop_box((0, 0, 20, 20), 1000, 800) == (0, 0, 320, 320)  # shifted into the image at the corner
    assert rs.crop_box((900, 700, 1000, 800), 1000, 800) == (680, 480, 1000, 800)
    assert rs.crop_box((100, 100, 300, 200), 1000, 800) == (0, 0, 600, 320)  # 3x box: 600 wide, clamped at left
    assert rs.crop_box((10, 10, 50, 50), 200, 150) == (0, 0, 200, 150)  # image smaller than the minimum window
    assert rs.crop_box((0, 0, 1000, 800), 1000, 800) == (0, 0, 1000, 800)


def test_zoom_box_geometry():
    # 10 px tall: 2x cap (2.4x would be needed); 560 output px wide = 280 image px from the box's left margin
    assert rs.zoom_box((100, 200, 700, 210), 1920, 1080) == ((84, 184, 364, 226), 2.0)
    assert rs.zoom_box((100, 100, 700, 120), 1920, 1080) == ((84, 84, 550, 136), 1.2)  # 20 px -> 24 px tall
    assert rs.zoom_box((10, 10, 110, 50), 1920, 1080) == ((0, 0, 126, 66), 1.0)  # tall enough: 1x, margin clamped at 0
    assert rs.zoom_box((0, 0, 1000, 800), 1000, 800) == ((0, 0, 560, 160), 1.0)  # huge box: cut at the top-left
    assert rs.zoom_box((1900, 1070, 1920, 1080), 1920, 1080) == ((1884, 1054, 1920, 1080), 2.0)  # bottom-right corner
    assert rs.zoom_box((10, 10, 50, 10), 200, 100)[1] == 2.0  # zero-height box: capped, no division error


def test_zoom_strip_makes_wide_thin_targets_readable(tmp_path):
    root, out = tmp_path / "slice", tmp_path / "out"
    (root / "images").mkdir(parents=True)
    out.mkdir()
    im = Image.new("RGB", (1920, 1080), (250, 250, 250))
    ImageDraw.Draw(im).rectangle((100, 200, 699, 211), fill=(30, 60, 220))  # 600 x 12 px text line
    ImageDraw.Draw(im).rectangle((900, 500, 939, 539), fill=(30, 60, 220))  # 40 x 40 icon: crop shows it at 1x
    im.save(root / "images/wide.png")
    rows = [{"id": f"windows:wide:{k}", "source": "os_atlas/windows", "image_path": "images/wide.png", "img_w": 1920,
             "img_h": 1080, "instruction": label, "bbox": bbox, "platform": "desktop", "data_type": None, "role": None}
            for k, (label, bbox) in enumerate([("A long thin label", [100, 200, 700, 212]), ("Icon", [900, 500, 940, 540])])]
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out / "rewrites.jsonl").write_text("".join(json.dumps(
        {"id": r["id"], "source": r["source"], "original": r["instruction"], "rewritten": "click " + r["instruction"],
         "style": "functional", "keep": True, "drop_reason": None, "prompt_version": "v3", "model": MODEL}) + "\n"
        for r in rows), encoding="utf-8")
    text = rs.build_review(out, root / "manifest.jsonl", n=50).read_text(encoding="utf-8")
    zooms = data_urls(text, "zoom")
    assert len(zooms) == 1 and "zoom 2.0x" in text
    z = zooms[0].convert("RGB")
    assert z.size == (560, 88)  # (12 + 2 * 16) px tall at 2x; cut to 560 px wide
    blue_rows = [y for y in range(z.height) if sum(1 for x in range(z.width) if z.getpixel((x, y))[2] > 160
                                                    and z.getpixel((x, y))[0] < 90) > 200]
    assert len(blue_rows) >= 22  # the 12 px line is ~24 px tall in the strip (vs 2 px in the 1800 px wide crop)
    assert sum(1 for r, g, b in z.getdata() if r > 180 and g < 80 and b < 80) > 20  # box frame drawn


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


def test_kept_cards_offer_source_error_and_the_tally_excludes_it(tmp_path):
    manifest, out = make_run(tmp_path)
    text = rs.build_review(out, manifest, n=50).read_text(encoding="utf-8")
    sections = re.findall(r'<section class="card (kept|drop)"(.*?)</section>', text, re.S)
    assert len(sections) == N_KEPT + N_DROPS
    for kind, sec in sections:
        values = re.findall(r'<input type="radio" name="v\d+" value="([^"]+)">', sec)
        assert values == (["faithful", "unfaithful", "unsure", "source_error"] if kind == "kept" else ["yes", "no"])
    assert text.count("source error: box \u2260 label <kbd>4</kbd>") == N_KEPT
    assert '<b id="t-r">0</b> rated' in text and '<b id="t-e">0</b> (not rated)' in text
    assert "<kbd>4</kbd> source error" in text and "left out of the faithful rate" in text
    assert '<button id="cp">Copy verdicts</button>' in text and '<button id="dl">' in text


def test_cards_carry_a_fingerprint_of_id_original_rewrite_and_drop_reason(tmp_path):
    manifest, out = make_run(tmp_path)
    recs = {r["id"]: r for r in rs.read_jsonl(out / "rewrites.jsonl")}
    text = rs.build_review(out, manifest, n=50).read_text(encoding="utf-8")
    found = {html.unescape(i): h for i, h in re.findall(
        r'<section class="card [^"]+" id="c\d+" data-id="([^"]*)"[^>]* data-h="([0-9a-f]{16})">', text)}
    assert len(found) == N_KEPT + N_DROPS
    for rid, h in found.items():
        r = recs[rid]
        want = json.dumps([r["id"], r["original"], r["rewritten"], r["drop_reason"]])
        assert h == hashlib.sha256(want.encode("utf-8")).hexdigest()[:16] == rs.card_hash(r)
    github = recs["fineweb:page_0:1"]
    assert rs.card_hash(dict(github, rewritten="open the GitHub link")) != found["fineweb:page_0:1"]
    assert rs.card_hash(dict(github, ts="later", model="other", style="positional")) == found["fineweb:page_0:1"]


def test_neighbor_display_new_and_old_formats():
    m = {"control": "button", "position": "top-left", "size": "small", "neighbors": [
        {"label": "Search", "target_is": "right of"}, {"label": "<Home>", "direction": "below"}, {"label": "Bare"}]}
    out = rs.meta_html(m)
    assert '<span class="nb">target is <i>right of</i> Search</span>' in out
    assert '<span class="nb">&lt;Home&gt; <i>is below</i></span>' in out  # pre-v3 meta: the neighbor's own direction
    assert '<span class="nb">Bare</span>' in out
    assert "neighbors: <i>none</i>" in rs.meta_html(dict(m, neighbors=[]))


def test_sheet_shows_v3_neighbor_relations(tmp_path):
    """Contract: meta.neighbors gives target_is, where the TARGET lies relative to the neighbor. GitHub is right of
    Search, so on the Search card the target is left of GitHub."""
    manifest, out = make_run(tmp_path)
    text = rs.build_review(out, manifest, n=50).read_text(encoding="utf-8")
    card = re.search(r'<section class="card kept" id="c\d+" data-id="fineweb:page_0:0".*?</section>', text, re.S)
    assert card and "target is <i>left of</i> GitHub" in card.group(0)


# The sheet's JavaScript under node with a minimal DOM: one column of 480 px cards every 500 px from y=100, a 100 px
# sticky header and an 800 px window. Actions: key, radio [card, value], note [card, text], scroll y (fires the scroll
# listener), jump y (moves without an event), click element id. clipboard: ok | reject (writeText fails, execCommand
# works) | none (no async clipboard, execCommand works) | fail (neither works).
HARNESS = r"""
const fs = require('fs');
const spec = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const store = Object.assign({}, spec.storage), mode = spec.clipboard || 'ok';
let clip = null, scrollY = 0, made = null;
const doc = {}, win = {};
const on = map => (t, f) => { (map[t] = map[t] || []).push(f); };
function el(extra = {}) {
  const cls = new Set(), ls = {};
  return Object.assign({
    dataset: {}, textContent: '', className: '', value: '', checked: false, style: {},
    classList: {add: (...c) => c.forEach(x => cls.add(x)), remove: (...c) => c.forEach(x => cls.delete(x)),
                toggle: (c, v) => (v ? cls.add(c) : cls.delete(c)), contains: c => cls.has(c)},
    classes: () => [...cls].sort(), addEventListener: on(ls), fire: t => (ls[t] || []).forEach(f => f({})),
    getBoundingClientRect: () => ({top: 0, bottom: 100}), scrollIntoView() {}, setAttribute() {}, select() {},
    remove() {}, click() {}, blur() {}, appendChild() {}, matches: () => false,
  }, extra);
}
const cards = spec.cards.map((c, i) => {
  const radios = c.radios.map(v => el({value: v})), note = el();
  const card = el({dataset: c.dataset, radios, note, querySelectorAll: () => radios, querySelector: () => note});
  const top = () => 100 + 500 * i - scrollY;
  card.getBoundingClientRect = () => ({top: top(), bottom: top() + 480});
  card.scrollIntoView = () => { if (top() < 100 || top() + 480 > 800) scrollY = 500 * i; };
  return card;
});
const header = el(), ids = {};
global.document = {body: el({dataset: spec.body}), activeElement: null, querySelectorAll: () => cards,
  querySelector: s => (s === 'header' ? header : null), getElementById: id => ids[id] || (ids[id] = el()),
  addEventListener: on(doc), createElement: () => (made = el()),
  execCommand: c => { if (c !== 'copy' || mode === 'fail' || mode === 'ok') return false; clip = made.value; return true; }};
global.localStorage = {getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = v; }};
global.history = {scrollRestoration: 'auto'};
global.innerHeight = 800;
global.addEventListener = on(win);
global.requestAnimationFrame = f => f();
global.confirm = () => true;
const writeText = t => (mode === 'reject' ? Promise.reject(new Error('denied')) : (clip = t, Promise.resolve()));
Object.defineProperty(globalThis, 'navigator', {configurable: true,
  value: mode === 'ok' || mode === 'reject' ? {clipboard: {writeText}} : {}});
new Function(spec.js)();
const snap = () => ({cur: cards.findIndex(c => c.classList.contains('cur')), scrollY});
const out = {load: snap(), steps: []};
for (const a of spec.actions) {
  if (a.key) (doc.keydown || []).forEach(f => f({key: a.key, target: el(), preventDefault() {}}));
  if (a.radio) cards[a.radio[0]].radios.find(r => r.value === a.radio[1]).fire('change');
  if (a.note) { cards[a.note[0]].note.value = a.note[1]; cards[a.note[0]].note.fire('input'); }
  if ('jump' in a) scrollY = a.jump;
  if ('scroll' in a) { scrollY = a.scroll; (win.scroll || []).forEach(f => f()); }
  if (a.click) ids[a.click].fire('click');
  out.steps.push(snap());
}
setTimeout(() => {
  Object.assign(out, {clip: clip === null ? null : JSON.parse(clip), store, restoration: history.scrollRestoration,
    text: Object.fromEntries(Object.entries(ids).map(([k, v]) => [k, String(v.textContent)])),
    classes: cards.map(c => c.classes())});
  console.log(JSON.stringify(out));
  process.exit(0);
}, 20);
"""
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


def _dataset(attrs: str) -> dict:
    """data-* attributes -> element.dataset (data-drop-reason -> dropReason), unescaped like a browser."""
    return {re.sub(r"-([a-z])", lambda m: m.group(1).upper(), k): html.unescape(v)
            for k, v in re.findall(r'data-([a-z0-9-]+)="([^"]*)"', attrs)}


def run_js(tmp_path: Path, text: str, storage=None, actions=(), clipboard="ok") -> dict:
    sheet = {"js": re.search(r"<script>(.*)</script>", text, re.S).group(1),
             "body": _dataset(re.search(r"<body ([^>]*)>", text).group(1)), "storage": storage or {},
             "actions": list(actions), "clipboard": clipboard, "cards": [
                 {"dataset": _dataset(tag), "radios": re.findall(r'<input type="radio" name="v\d+" value="([^"]+)"', inner)}
                 for tag, inner in re.findall(r'(<section class="card [^>]*>)(.*?)</section>', text, re.S)]}
    (tmp_path / "spec.json").write_text(json.dumps(sheet), encoding="utf-8")
    (tmp_path / "harness.js").write_text(HARNESS, encoding="utf-8")
    res = subprocess.run([NODE, str(tmp_path / "harness.js"), str(tmp_path / "spec.json")], capture_output=True,
                         text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    return {**json.loads(res.stdout), "sheet": sheet}


@needs_node
def test_js_fingerprints_tally_and_copy_export(tmp_path):
    manifest, out = make_run(tmp_path)
    text = rs.build_review(out, manifest, n=50, drops=20).read_text(encoding="utf-8")
    probe = run_js(tmp_path, text)["sheet"]
    cards, key = [c["dataset"] for c in probe["cards"]], probe["body"]["key"]
    assert [c["kind"] for c in cards] == ["kept"] * N_KEPT + ["drop"] * N_DROPS
    stored = {cards[0]["id"]: {"verdict": "faithful", "note": "kept from before", "h": cards[0]["h"]},
              cards[1]["id"]: {"verdict": "unfaithful", "note": "stale", "h": "0" * 16},  # given to another rewrite
              cards[2]["id"]: {"verdict": "faithful"},  # no fingerprint (pre-v3 sheet): ignored too
              "gone:id": {"verdict": "faithful", "h": "1" * 16}}
    res = run_js(tmp_path, text, storage={key: json.dumps(stored)}, actions=[
        {"key": "4"}, {"key": "3"}, {"radio": [3, "faithful"]}, {"radio": [N_KEPT, "yes"]}, {"note": [5, "hm"]},
        {"click": "cp"}])
    assert res["load"] == {"cur": 1, "scrollY": 500} and res["restoration"] == "manual"  # card 1's verdict is stale
    assert [s["cur"] for s in res["steps"][:2]] == [2, 3]  # each number key judges the card and moves on
    doc = res["clip"]
    assert doc["counts"] == {"faithful": 2, "unfaithful": 0, "unsure": 1, "source_error": 1, "unjudged": 3, "rated": 3,
                             "kept_total": N_KEPT, "faithful_rate": 2 / 3, "pass_line": 0.9, "pass": False,
                             "drops_correct": 1, "drops_wrong": 0, "drops_unjudged": N_DROPS - 1, "drops_total": N_DROPS}
    assert (doc["prompt_version"], doc["seed"], doc["model"]) == ("v1", 0, MODEL)
    assert doc["run_id"] == "run-1" and doc["manifest_sha256"] == "deadbeef" and doc["sheet_key"] == key
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", doc["generated"])
    recs = {r["id"]: r for r in rs.read_jsonl(out / "rewrites.jsonl")}
    assert [c["id"] for c in doc["cards"]] == [c["id"] for c in cards]
    for c in doc["cards"]:
        assert list(c) == ["id", "kind", "source", "style", "original", "rewritten", "drop_reason", "verdict", "note", "h"]
        r = recs[c["id"]]
        assert c["h"] == rs.card_hash(r)
        assert (c["source"], c["style"], c["original"], c["rewritten"], c["drop_reason"]) == (
            r["source"], r["style"], r["original"], r["rewritten"], r["drop_reason"])
    verdicts = [(c["verdict"], c["note"]) for c in doc["cards"]]
    assert verdicts[:6] == [("faithful", "kept from before"), ("source_error", ""), ("unsure", ""), ("faithful", ""),
                            (None, ""), (None, "hm")]
    assert verdicts[N_KEPT][0] == "yes" and all(v is None for v, _ in verdicts[N_KEPT + 1:])
    saved = json.loads(res["store"][key])
    assert saved[cards[1]["id"]] == {"h": cards[1]["h"], "verdict": "source_error"}  # stale entry replaced
    assert saved["gone:id"] == stored["gone:id"]  # entries for cards not on this sheet are left alone
    t = res["text"]
    assert (t["t-f"], t["t-r"], t["t-u"], t["t-s"], t["t-e"], t["t-j"]) == ("2", "3", "0", "1", "1", "4")
    assert t["t-p"] == "66.7%" and t["t-status"] == "FAIL so far (3 left)" and (t["t-dy"], t["t-dj"]) == ("1", "1")
    assert t["cp-ok"] == "copied"
    assert res["classes"][0] == ["v-faithful"] and res["classes"][1] == ["v-source_error"]


@needs_node
def test_js_pass_line_counts_unsure_against_and_ignores_source_errors(tmp_path):
    manifest, out = make_run(tmp_path)
    text = rs.build_review(out, manifest, n=50, drops=0).read_text(encoding="utf-8")
    res = run_js(tmp_path, text, actions=[{"key": k} for k in "1111144"] + [{"click": "cp"}])
    assert res["clip"]["counts"]["faithful_rate"] == 1.0 and res["text"]["t-status"] == "PASS"  # 5/5 rated
    assert res["text"]["t-p"] == "100.0%" and res["clip"]["counts"]["source_error"] == 2
    res = run_js(tmp_path, text, actions=[{"key": k} for k in "1111113"] + [{"click": "cp"}])
    assert res["clip"]["counts"]["faithful_rate"] == 6 / 7 and res["text"]["t-status"] == "FAIL"


@needs_node
def test_js_keys_follow_the_card_in_view(tmp_path):
    manifest, out = make_run(tmp_path)
    text = rs.build_review(out, manifest, n=50, drops=20).read_text(encoding="utf-8")
    res = run_js(tmp_path, text, actions=[
        {"scroll": 2000}, {"key": "1"},  # scrolled to card 4: the cursor follows, 1 judges card 4
        {"jump": 1000}, {"key": "2"},  # moved without a scroll event: the key still retargets, to card 2
        {"jump": 500 * N_KEPT}, {"key": "4"}, {"key": "1"},  # first drop card: 4 does nothing, 1 = yes
        {"key": "k"}, {"jump": 0}, {"key": "j"},  # k from the next drop card; then j from what is in view (card 0)
        {"click": "cp"}])
    assert res["load"] == {"cur": 0, "scrollY": 0}
    assert [s["cur"] for s in res["steps"][:8]] == [4, 5, 5, 3, 3, N_KEPT, N_KEPT + 1, N_KEPT]
    assert res["steps"][9]["cur"] == 1
    assert {i: c["verdict"] for i, c in enumerate(res["clip"]["cards"]) if c["verdict"]} == {
        4: "faithful", 2: "unfaithful", N_KEPT: "yes"}


@needs_node
def test_js_load_focuses_and_scrolls_to_first_unjudged(tmp_path):
    manifest, out = make_run(tmp_path)
    text = rs.build_review(out, manifest, n=50, drops=20).read_text(encoding="utf-8")
    probe = run_js(tmp_path, text)["sheet"]
    cards, key = [c["dataset"] for c in probe["cards"]], probe["body"]["key"]
    done = {c["id"]: {"verdict": "faithful", "h": c["h"]} for c in cards[:4]}
    assert run_js(tmp_path, text, storage={key: json.dumps(done)})["load"] == {"cur": 4, "scrollY": 2000}
    all_done = {c["id"]: {"verdict": "yes" if c["kind"] == "drop" else "unsure", "h": c["h"]} for c in cards}
    res = run_js(tmp_path, text, storage={key: json.dumps(all_done)}, actions=[{"click": "cp"}])
    assert res["load"] == {"cur": 0, "scrollY": 0} and res["text"]["t-status"] == "FAIL"
    assert res["clip"]["counts"]["unjudged"] == 0 and res["clip"]["counts"]["drops_unjudged"] == 0


@needs_node
@pytest.mark.parametrize("clipboard,message", [("reject", "copied"), ("none", "copied"),
                                               ("fail", "copy failed: use Download")])
def test_js_copy_falls_back_to_exec_command(tmp_path, clipboard, message):
    manifest, out = make_run(tmp_path)
    text = rs.build_review(out, manifest, n=2, drops=1).read_text(encoding="utf-8")
    res = run_js(tmp_path, text, actions=[{"key": "1"}, {"click": "cp"}], clipboard=clipboard)
    assert res["text"]["cp-ok"] == message
    if clipboard == "fail":
        assert res["clip"] is None
    else:  # the textarea fallback copies the same document as the download
        assert res["clip"]["counts"]["faithful"] == 1 and len(res["clip"]["cards"]) == 3
