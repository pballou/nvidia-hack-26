"""Write a self-contained HTML contact sheet for auditing Lightning rewrites (T3 quality gate).

Shows n random kept rewrites (context crop with the target box, thumbnail, original label, rewrite, the metadata Lightning
saw) plus a few model drops, with per-card verdicts, a live tally against the 90% pass line, and a JSON download.

Example:
  python data_prep/rewrite/review_sample.py --out /home/ubuntu/data/rewrite_dry --n 50 --drops 20
"""

import argparse
import base64
import hashlib
import html
import io
import json
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_prep.rewrite import meta
from data_prep.rewrite.validate import LLM_ERROR

CROP_MIN_SIDE, CROP_FACTOR = 320, 3
CROP_MAX = (360, 360)
THUMB_MAX = (220, 260)
BOX_COLOR = (230, 0, 0)
JPEG_QUALITY = 82
PASS_LINE = 0.9
KEPT_VERDICTS = ("faithful", "unfaithful", "unsure")
DROP_VERDICTS = ("yes", "no")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def pct(x) -> str:
    return f"{x:.1%}" if isinstance(x, (int, float)) and not isinstance(x, bool) else "n/a"


def resolve_manifest(out_dir: Path, stats: dict, manifest_path=None) -> Path:
    """--manifest if given, else the manifest path recorded in stats.json (absolute, cwd-relative, or OUT-relative)."""
    p = manifest_path or (stats.get("manifest") or {}).get("path") or (stats.get("run") or {}).get("manifest")
    if not p:
        raise FileNotFoundError(f"no manifest path in {out_dir / 'stats.json'}; pass --manifest")
    p = Path(p)
    if not p.is_absolute() and not p.exists() and (out_dir / p).exists():
        p = out_dir / p
    if not p.is_file():
        raise FileNotFoundError(f"manifest not found: {p}")
    return p


def sample_rows(rewrites: list[dict], n: int, seed: int, drops: int) -> tuple[list[dict], list[dict]]:
    """Seeded samples over rows sorted by id: kept+rewritten rows, and model drops (llm_error rows are not judgments)."""
    recs = sorted(rewrites, key=lambda r: r["id"])
    kept = [r for r in recs if r.get("keep") is True and r.get("rewritten")]
    dropped = [r for r in recs if r.get("keep") is False and r.get("drop_reason") != LLM_ERROR]
    return (random.Random(seed).sample(kept, min(max(n, 0), len(kept))),
            random.Random(f"drops:{seed}").sample(dropped, min(max(drops, 0), len(dropped))))


def crop_box(bbox, img_w: int, img_h: int, min_side: int = CROP_MIN_SIDE, factor: int = CROP_FACTOR):
    """Context window per axis: max(factor x box, min_side) centered on the box, shifted then clamped into the image."""
    def axis(a, b, size):
        span = min(size, max(factor * (b - a), min_side))
        lo = min(max((a + b) / 2 - span / 2, 0), size - span)
        return math.floor(lo), min(size, math.ceil(lo + span))

    (l, r), (t, b) = axis(bbox[0], bbox[2], img_w), axis(bbox[1], bbox[3], img_h)
    return l, t, r, b


def fit(im: Image.Image, max_w: int, max_h: int) -> tuple[Image.Image, float]:
    s = min(1.0, max_w / im.width, max_h / im.height)
    if s < 1:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.Resampling.LANCZOS)
    return im, s


def draw_box(im: Image.Image, box, width: int) -> None:
    """Frame drawn just outside the box (so the element stays visible), clamped so edge boxes keep a visible frame."""
    x1, y1, x2, y2 = box
    l, t = min(max(x1 - width, 0), im.width - 1), min(max(y1 - width, 0), im.height - 1)
    r, b = max(min(x2 + width, im.width - 1), l), max(min(y2 + width, im.height - 1), t)
    ImageDraw.Draw(im).rectangle([l, t, r, b], outline=BOX_COLOR, width=width)


def data_url(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=JPEG_QUALITY, subsampling=0)  # 4:4:4 keeps the red box crisp
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_images(row: dict | None, images_root: Path) -> tuple[str | None, str | None, str | None]:
    """(crop data URL, thumbnail data URL, error). Never raises: problems come back as the error string."""
    if row is None:
        return None, None, "row not in manifest"
    path = images_root / row["image_path"]
    if not path.is_file():
        return None, None, f"image missing: {row['image_path']}"
    try:
        with Image.open(path) as src:
            im = src.convert("RGB")
        sx = im.width / row["img_w"] if row.get("img_w") else 1.0
        sy = im.height / row["img_h"] if row.get("img_h") else 1.0
        x1, y1, x2, y2 = (float(v) for v in row["bbox"])
        box = [min(max(x1 * sx, 0), im.width), min(max(y1 * sy, 0), im.height),
               min(max(x2 * sx, 0), im.width), min(max(y2 * sy, 0), im.height)]
        l, t, r, b = crop_box(box, im.width, im.height)
        crop, s = fit(im.crop((l, t, r, b)), *CROP_MAX)
        draw_box(crop, [(box[0] - l) * s, (box[1] - t) * s, (box[2] - l) * s, (box[3] - t) * s], 3)
        thumb, s = fit(im, *THUMB_MAX)
        draw_box(thumb, [v * s for v in box], 2)
        return data_url(crop), data_url(thumb), None
    except Exception as e:  # a corrupt image must not kill the sheet
        return None, None, f"image error: {type(e).__name__}: {e}"


def load_cache_records(out_dir: Path, recs: list[dict]) -> dict[str, dict]:
    """Full cache records (warnings, attempts) for the sampled rows, if OUT/cache.sqlite exists."""
    path = out_dir / "cache.sqlite"
    if not path.is_file() or not recs:
        return {}
    try:
        from data_prep.rewrite.cache import RewriteCache
        found = {}
        with RewriteCache(path) as cache:
            for pv in sorted({str(r.get("prompt_version")) for r in recs}):
                found.update(cache.get_many(pv, [r["id"] for r in recs if str(r.get("prompt_version")) == pv]))
        return found
    except Exception as e:
        print(f"warning: could not read {path}: {type(e).__name__}: {e}", file=sys.stderr)
        return {}


def chip(text, cls="") -> str:
    return f'<span class="chip {cls}">{esc(text)}</span>'


def meta_html(m: dict | None) -> str:
    if m is None:
        return '<div class="meta muted">no metadata (row not in manifest)</div>'
    parts = [f"{k}: <b>{esc(m[k])}</b>" for k in ("control", "kind", "position", "size") if m.get(k)]
    nbs = " · ".join(f'<span class="nb"><i>{esc(x["direction"])}</i> {esc(x["label"])}</span>' for x in m["neighbors"])
    return (f'<div class="meta">Lightning saw: {" · ".join(parts)}</div>'
            f'<div class="meta nbs">neighbors: {nbs or "<i>none</i>"}</div>')


def card_html(idx: int, kind: str, rec: dict, row: dict | None, by_image: dict, images_root: Path,
              cached: dict | None) -> str:
    crop, thumb, err = render_images(row, images_root)
    imgs = (f'<img class="crop" src="{crop}" alt="target crop"><img class="thumb" src="{thumb}" alt="full screenshot">'
            if crop else f'<div class="ph">{esc(err)}</div>')
    m = meta.row_meta(row, by_image, style=rec.get("style") or None) if row is not None else None
    chips = [chip(rec.get("style"), "style"), chip(rec.get("source")), chip((row or {}).get("platform") or "?")]
    flags = [chip(w, "warn") for w in (cached or {}).get("warnings") or []]
    attempts = (cached or {}).get("attempts") or 1
    if attempts > 1:
        flags.append(chip(f"attempts: {attempts}", "warn"))
    if row is not None and row.get("instruction") != rec.get("original"):
        flags.append(chip("manifest label differs from original", "warn"))
    if kind == "kept":
        main = f'<div class="rw">{esc(rec.get("rewritten"))}</div>'
        opts = [(v, i + 1) for i, v in enumerate(KEPT_VERDICTS)]
        question = ""
    else:
        main = f'<div class="rw drop">DROPPED: {esc(rec.get("drop_reason"))}</div>'
        opts = [(v, i + 1) for i, v in enumerate(DROP_VERDICTS)]
        question = '<span class="q">drop correct?</span> '
    radios = "".join(f'<label><input type="radio" name="v{idx}" value="{v}"> {v} <kbd>{k}</kbd></label>'
                     for v, k in opts)
    data = {"id": rec["id"], "kind": kind, "source": rec.get("source"), "style": rec.get("style"),
            "original": rec.get("original"), "rewritten": rec.get("rewritten"), "drop-reason": rec.get("drop_reason")}
    attrs = " ".join(f'data-{k}="{esc(v)}"' for k, v in data.items())
    return (f'<section class="card {kind}" id="c{idx}" {attrs}>'
            f'<div class="top"><span class="num">#{idx + 1}</span> {" ".join(chips)} <code>{esc(rec["id"])}</code></div>'
            f'<div class="imgs">{imgs}</div>{main}'
            f'<div class="orig"><span class="lbl">original label</span> {esc(rec.get("original"))}</div>'
            f'{meta_html(m)}<div class="flags">{" ".join(flags)}</div>'
            f'<div class="verdict">{question}{radios}<input type="text" class="note" placeholder="note"></div></section>')


def header_html(stats: dict, rewrites: list[dict], n_kept_pop: int, n_drop_pop: int, n_kept: int, n_drop: int,
                seed: int, pv: str, model: str) -> str:
    kept = sum(1 for r in rewrites if r.get("keep") is True)
    reasons = {}
    for r in rewrites:
        if r.get("keep") is False:
            reasons[r.get("drop_reason")] = reasons.get(r.get("drop_reason"), 0) + 1
    reason_txt = ", ".join(f"{esc(k)} {v}" for k, v in sorted(reasons.items(), key=lambda kv: (-kv[1], str(kv[0]))))
    run, quality, gate = stats.get("run") or {}, stats.get("quality") or {}, stats.get("gate") or {}
    lines = [f"rewrites <b>{len(rewrites)}</b> · kept <b>{kept}</b> · dropped <b>{len(rewrites) - kept}</b>"
             + (f" ({reason_txt})" if reason_txt else "")]
    if stats:
        gates = " ".join(chip(f"{k}: {'ok' if v else 'FAIL'}", "ok" if v else "bad") for k, v in sorted(gate.items()))
        lines.append(f"first-try valid <b>{pct(quality.get('first_try_valid_rate'))}</b> · batch JSON ok "
                     f"<b>{pct(quality.get('batch_json_ok_rate'))}</b> · drop rate (raw, excl. llm_error) "
                     f"<b>{pct(quality.get('drop_rate_raw'))}</b> · endpoint {esc(run.get('endpoint'))}"
                     f"{' · dry run' if run.get('dry_run') else ''} {gates}")
        by_source = (stats.get("rewrites") or {}).get("by_source") or {}
        if by_source:
            lines.append("drop rate by source: " + " · ".join(
                f"{esc(s)} <b>{pct(v.get('drop_rate'))}</b> (n={esc(v.get('targets'))})" for s, v in sorted(by_source.items())))
    else:
        lines.append('<span class="muted">no stats.json in OUT</span>')
    lines.append(f"sheet: {n_kept} of {n_kept_pop} kept rewrites, {n_drop} of {n_drop_pop} model drops "
                 f"(llm_error not sampled) · seed {seed}")
    return (f'<header><h1>T3 rewrite review · prompt <code>{esc(pv)}</code> · {esc(model)}</h1>'
            + "".join(f'<div class="sum">{x}</div>' for x in lines)
            + f'<div class="tally">faithful <b id="t-f">0</b> / <b id="t-j">0</b> judged (<b id="t-p">n/a</b>) of '
              f'{n_kept} · pass line {PASS_LINE:.0%} <span id="t-status" class="badge">not started</span> · '
              f'unfaithful <b id="t-u">0</b> · unsure <b id="t-s">0</b> · drops correct <b id="t-dy">0</b> / '
              f'<b id="t-dj">0</b> of {n_drop} <button id="dl">Download verdicts JSON</button> '
              f'<button id="clr">Clear verdicts</button></div>'
              '<div class="help">Keys: <kbd>j</kbd>/<kbd>k</kbd> next/prev card · <kbd>1</kbd> faithful · '
              '<kbd>2</kbd> unfaithful · <kbd>3</kbd> unsure (drop cards: <kbd>1</kbd> yes · <kbd>2</kbd> no) · '
              '<kbd>n</kbd> next unjudged. <b>Faithful</b> = the rewrite clearly refers to the boxed element and '
              'invents nothing (no color, shape, text or position that the label/metadata does not support).</div>'
              '</header>')


CSS = """
*{box-sizing:border-box}
body{font:14px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#1b1b1b;background:#f2f2ef;margin:0}
header{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid #bbb;padding:8px 16px}
h1{font-size:17px;margin:0 0 4px} h2{font-size:16px;margin:16px 16px 0}
.sum{font-size:13px;color:#333}
.tally{margin-top:6px;font-size:15px}
.help{font-size:12px;color:#555;margin-top:4px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(600px,1fr));gap:12px;padding:12px 16px}
.card{background:#fff;border:2px solid #d4d4d0;border-radius:6px;padding:10px;scroll-margin-top:200px}
.card.cur{border-color:#1f5fbf;box-shadow:0 0 0 3px rgba(31,95,191,.25)}
.card.v-faithful,.card.v-yes{background:#eaf6ea}
.card.v-unfaithful,.card.v-no{background:#fbeaea}
.card.v-unsure{background:#fff6dc}
.top{font-size:12px;color:#444;margin-bottom:6px} .num{font-weight:700;color:#000}
code{font-size:12px;color:#555;word-break:break-all}
.imgs{display:flex;gap:10px;align-items:flex-start}
.imgs img{border:1px solid #bbb;display:block} .crop{max-width:360px;max-height:360px} .thumb{max-width:220px;max-height:260px}
.ph{width:360px;height:160px;display:flex;align-items:center;justify-content:center;text-align:center;padding:8px;
 background:#eee;color:#666;border:1px dashed #999}
.rw{font-size:22px;font-weight:650;margin:8px 0 2px;color:#000;word-wrap:break-word}
.rw.drop{color:#8a1c1c;font-size:18px}
.orig{font-size:15px;word-wrap:break-word}
.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#666;margin-right:4px}
.meta{font-size:12px;color:#333;margin-top:3px;word-wrap:break-word} .nb i{color:#666}
.muted{color:#777}
.chip{display:inline-block;padding:0 7px;border-radius:9px;background:#e6e6e3;font-size:12px;margin:1px 2px 1px 0}
.chip.style{background:#dde7f7} .chip.warn{background:#ffe2a8} .chip.ok{background:#cfeccf} .chip.bad{background:#f6c4c4}
.flags{margin-top:3px}
.verdict{margin-top:8px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.verdict label{cursor:pointer} .q{font-weight:600}
.note{flex:1;min-width:140px;padding:3px 6px;border:1px solid #bbb;border-radius:4px}
kbd{font-size:11px;border:1px solid #aaa;border-radius:3px;padding:0 3px;background:#f7f7f7}
.badge{padding:1px 8px;border-radius:9px;background:#e6e6e3;font-weight:600}
.badge.pass{background:#bfe6bf} .badge.fail{background:#f4bcbc}
button{font-size:13px;margin-left:6px}
"""

JS = """
(() => {
  const B = document.body.dataset, KEY = B.key, PASS = parseFloat(B.pass);
  const cards = [...document.querySelectorAll('.card')];
  const kept = cards.filter(c => c.dataset.kind === 'kept'), drops = cards.filter(c => c.dataset.kind === 'drop');
  const OPTS = {kept: ['faithful', 'unfaithful', 'unsure'], drop: ['yes', 'no']};
  let state = {};
  try { state = JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch (e) { state = {}; }
  let cur = 0;
  const get = c => state[c.dataset.id] || (state[c.dataset.id] = {});
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} };
  const $ = id => document.getElementById(id);
  function paint(c) {
    const s = state[c.dataset.id] || {};
    c.classList.remove(...OPTS.kept.map(v => 'v-' + v), ...OPTS.drop.map(v => 'v-' + v));
    if (s.verdict) c.classList.add('v-' + s.verdict);
    c.querySelectorAll('input[type=radio]').forEach(r => { r.checked = r.value === s.verdict; });
    const n = c.querySelector('.note');
    if (n !== document.activeElement) n.value = s.note || '';
  }
  function tally() {
    const v = c => (state[c.dataset.id] || {}).verdict;
    const f = kept.filter(c => v(c) === 'faithful').length, j = kept.filter(v).length;
    const t = {faithful: f, judged: j, total: kept.length,
               unfaithful: kept.filter(c => v(c) === 'unfaithful').length, unsure: kept.filter(c => v(c) === 'unsure').length,
               drops_correct: drops.filter(c => v(c) === 'yes').length, drops_judged: drops.filter(v).length,
               drops_total: drops.length, faithful_rate: j ? f / j : null, pass_line: PASS};
    $('t-f').textContent = f; $('t-j').textContent = j; $('t-u').textContent = t.unfaithful; $('t-s').textContent = t.unsure;
    $('t-p').textContent = j ? (100 * f / j).toFixed(1) + '%' : 'n/a';
    $('t-dy').textContent = t.drops_correct; $('t-dj').textContent = t.drops_judged;
    const st = $('t-status');
    st.className = 'badge' + (j ? (f / j >= PASS ? ' pass' : ' fail') : '');
    st.textContent = !j ? 'not started' : (f / j >= PASS ? 'PASS' : 'FAIL') + (j < kept.length ? ' so far (' + (kept.length - j) + ' left)' : '');
    return t;
  }
  function setVerdict(c, val) { get(c).verdict = val; save(); paint(c); tally(); }
  function focus(i, scroll = true) {
    if (!cards.length) return;
    cur = Math.max(0, Math.min(cards.length - 1, i));
    cards.forEach((c, k) => c.classList.toggle('cur', k === cur));
    if (scroll) cards[cur].scrollIntoView({block: 'nearest'});
  }
  cards.forEach((c, i) => {
    c.addEventListener('click', () => focus(i, false));
    c.querySelectorAll('input[type=radio]').forEach(r => r.addEventListener('change', () => setVerdict(c, r.value)));
    const n = c.querySelector('.note');
    n.addEventListener('input', () => { get(c).note = n.value; save(); });
    paint(c);
  });
  document.addEventListener('keydown', e => {
    if (e.target.matches('input[type=text]')) { if (e.key === 'Escape' || e.key === 'Enter') e.target.blur(); return; }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const c = cards[cur];
    if (e.key === 'j') focus(cur + 1);
    else if (e.key === 'k') focus(cur - 1);
    else if (e.key === 'n') { const i = cards.findIndex((x, k) => k > cur && !(state[x.dataset.id] || {}).verdict); if (i >= 0) focus(i); }
    else if (c && ['1', '2', '3'].includes(e.key)) {
      const val = OPTS[c.dataset.kind][+e.key - 1];
      if (!val) return;
      setVerdict(c, val); focus(cur + 1);
    } else return;
    e.preventDefault();
  });
  $('dl').addEventListener('click', () => {
    const out = {sheet: Object.assign({}, B), tally: tally(), cards: cards.map(c => {
      const d = c.dataset, s = state[d.id] || {};
      return {id: d.id, kind: d.kind, source: d.source, style: d.style, original: d.original, rewritten: d.rewritten || null,
              drop_reason: d.dropReason || null, verdict: s.verdict || null, note: s.note || ''};
    })};
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'}));
    a.download = 'review_verdicts_' + B.pv + '_seed' + B.seed + '.json';
    document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  });
  $('clr').addEventListener('click', () => {
    if (!confirm('Clear all verdicts and notes on this sheet?')) return;
    state = {}; save(); cards.forEach(paint); tally();
  });
  tally(); focus(0, false);
})();
"""


def build_review(out_dir, manifest_path=None, images_root=None, n=50, seed=0, drops=20, html_path=None) -> Path:
    """Write the contact sheet for OUT (rewrites.jsonl, optional stats.json and cache.sqlite); return its path."""
    out_dir = Path(out_dir)
    stats_path = out_dir / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    rewrites = read_jsonl(out_dir / "rewrites.jsonl")
    manifest_path = resolve_manifest(out_dir, stats, manifest_path)
    rows = read_jsonl(manifest_path)
    by_id, by_image = {r["id"]: r for r in rows}, meta.index_by_image(rows)
    images_root = Path(images_root) if images_root else manifest_path.parent
    kept, dropped = sample_rows(rewrites, n, seed, drops)
    n_kept_pop = sum(1 for r in rewrites if r.get("keep") is True and r.get("rewritten"))
    n_drop_pop = sum(1 for r in rewrites if r.get("keep") is False and r.get("drop_reason") != LLM_ERROR)
    cached = load_cache_records(out_dir, kept + dropped)
    run = stats.get("run") or {}
    first = rewrites[0] if rewrites else {}
    pv = str(run.get("prompt_version") or first.get("prompt_version") or "?")
    model = str(run.get("model_label") or run.get("model") or first.get("model") or "?")
    cards = [card_html(i, kind, rec, by_id.get(rec["id"]), by_image, images_root, cached.get(rec["id"]))
             for i, (kind, rec) in enumerate([("kept", r) for r in kept] + [("drop", r) for r in dropped])]
    key = hashlib.sha256(json.dumps([pv, seed, [r["id"] for r in kept + dropped]]).encode("utf-8")).hexdigest()[:16]
    body_attrs = {"key": f"t3-review:{key}", "pv": pv, "model": model, "seed": seed, "pass": PASS_LINE,
                  "run-id": run.get("run_id") or "", "manifest-sha256": (stats.get("manifest") or {}).get("sha256") or ""}
    page = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
            f'<title>T3 review {esc(pv)} seed {seed}</title><style>{CSS}</style></head>',
            "<body " + " ".join(f'data-{k}="{esc(v)}"' for k, v in body_attrs.items()) + ">",
            header_html(stats, rewrites, n_kept_pop, n_drop_pop, len(kept), len(dropped), seed, pv, model),
            f'<h2>Kept rewrites ({len(kept)} of {n_kept_pop})</h2><div class="grid">', *cards[:len(kept)], "</div>",
            f'<h2>Model drops ({len(dropped)} of {n_drop_pop}): is the drop correct?</h2><div class="grid">',
            *cards[len(kept):], "</div>", f"<script>{JS}</script></body></html>\n"]
    html_path = Path(html_path) if html_path else out_dir / "review.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text("\n".join(page), encoding="utf-8")
    return html_path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="rewrite.py output dir (rewrites.jsonl, stats.json)")
    ap.add_argument("--manifest", type=Path, help="default: the manifest path recorded in OUT/stats.json")
    ap.add_argument("--images-root", type=Path, help="image_path is relative to this (default: manifest dir)")
    ap.add_argument("--n", type=int, default=50, help="kept rewrites to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drops", type=int, default=20, help="model-dropped rows to sample")
    ap.add_argument("--html", type=Path, help="default: OUT/review.html")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = build_review(args.out, args.manifest, args.images_root, args.n, args.seed, args.drops, args.html)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")
    n_kept, n_drop = text.count('data-kind="kept"'), text.count('data-kind="drop"')
    print(f"wrote {path} ({n_kept} kept cards, {n_drop} drop cards); open in a browser: {path.resolve().as_uri()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
