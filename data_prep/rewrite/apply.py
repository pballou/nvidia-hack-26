"""Apply Lightning rewrites to a manifest and write the two ablation training variants.

train_lightning.jsonl: selected raw-label rows get the Lightning instruction (keep=false dropped); others pass through.
train_template.jsonl: the identical row set and order with the original labels, so the ablation isolates the text.

Example:
  python data_prep/rewrite/apply.py --manifest /home/ubuntu/data/os_atlas_20k/manifest.jsonl \
      --rewrites out/rewrites.jsonl --out /home/ubuntu/data/os_atlas_20k/train
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_prep.rewrite import meta


class MissingRewriteError(ValueError):
    """A row selected for rewriting has no rewrite record."""


def _index_rewrites(rewrites) -> dict[str, dict]:
    if isinstance(rewrites, dict):
        for rid, rec in rewrites.items():
            if "id" in rec and rec["id"] != rid:
                raise ValueError(f"rewrite keyed {rid!r} has id {rec['id']!r}")
        return dict(rewrites)
    by_id = {}
    for rec in rewrites:
        if not isinstance(rec.get("id"), str):
            raise ValueError(f"rewrite record without a string id: {str(rec)[:200]}")
        if rec["id"] in by_id:
            raise ValueError(f"duplicate rewrite id {rec['id']!r}")
        by_id[rec["id"]] = rec
    return by_id


def _sample(ids, n=5) -> str:
    ids = sorted(ids)
    return ", ".join(map(repr, ids[:n])) + (f", ... ({len(ids)} total)" if len(ids) > n else "")


def _apply(manifest, rewrites, extra_raw, extra_natural, allow_missing):
    """Return (train rows, dropped [(row, reason)], missing [row])."""
    meta.assert_clean_manifest(manifest)
    applied = [r["id"] for r in manifest if "instruction_original" in r or "instruction_style" in r]
    if applied:
        raise ValueError(f"rows already have rewrites applied: {_sample(applied)}; pass the original manifest")
    by_id = _index_rewrites(rewrites)
    ids = {r["id"] for r in manifest}
    selected = {r["id"] for r in manifest if meta.is_selected(r, extra_raw=extra_raw, extra_natural=extra_natural)}
    unknown = by_id.keys() - ids
    if unknown:
        raise ValueError(f"rewrites for ids not in the manifest: {_sample(unknown)}")
    unselected = by_id.keys() - selected
    if unselected:
        raise ValueError(f"rewrites for rows not selected for rewriting: {_sample(unselected)}")
    missing_ids = selected - by_id.keys()
    if missing_ids and not allow_missing:
        raise MissingRewriteError(f"{len(missing_ids)} selected rows have no rewrite: {_sample(missing_ids)}; "
                                  "rerun rewrite.py, or pass --allow-missing to keep their raw labels")

    rows, dropped, missing = [], [], []
    for row in manifest:
        out = dict(row)
        rec = by_id.get(row["id"])
        if meta.classify_source(row["source"], extra_raw=extra_raw, extra_natural=extra_natural) == "natural":
            style = "natural"
        elif row["id"] not in selected:
            style = "raw"
        elif rec is None:
            missing.append(row)
            style = "raw"
        else:
            if rec.get("original") != row["instruction"]:
                raise ValueError(f"rewrite for {row['id']!r} was made from {rec.get('original')!r} but the manifest "
                                 f"says {row['instruction']!r}; the manifest changed since rewrite.py ran")
            if rec.get("source", row["source"]) != row["source"]:
                raise ValueError(f"rewrite for {row['id']!r} has source {rec['source']!r}, manifest {row['source']!r}")
            if not isinstance(rec.get("keep"), bool):
                raise ValueError(f"rewrite for {row['id']!r} has non-boolean keep {rec.get('keep')!r}")
            if not rec["keep"]:
                dropped.append((row, rec.get("drop_reason") or "unspecified"))
                continue
            if rec.get("rewritten"):
                if not isinstance(rec.get("style"), str) or not rec["style"]:
                    raise ValueError(f"rewrite for {row['id']!r} has no style")
                out["instruction"] = rec["rewritten"]
                style = rec["style"]
            else:
                style = "raw"
        out["instruction_original"] = row["instruction"]
        out["instruction_style"] = style
        rows.append(out)
    return rows, dropped, missing


def apply_rewrites(manifest: list[dict], rewrites, *, extra_raw=(), extra_natural=(),
                   allow_missing=False) -> list[dict]:
    """Mixed training manifest: rewritten instructions where keep && rewritten, keep=false selected rows dropped."""
    return _apply(manifest, rewrites, extra_raw, extra_natural, allow_missing)[0]


def template_variant(train_rows: list[dict]) -> list[dict]:
    """Same rows and order as train_rows with the original labels restored (natural rows stay natural)."""
    out = []
    for row in train_rows:
        if "instruction_original" not in row:
            raise ValueError(f"row {row.get('id')!r} has no instruction_original; pass apply_rewrites output")
        r = dict(row)
        r["instruction"] = row["instruction_original"]
        if r.get("instruction_style") != "natural":
            r["instruction_style"] = "raw"
        out.append(r)
    return out


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def _counts(values) -> dict:
    return dict(sorted(Counter(values).items()))


def build_stats(manifest, rewrites, rows, dropped, missing) -> dict:
    """Counts for apply_stats.json (no timestamps or output paths, so reruns are byte-identical)."""
    recs = list(rewrites.values()) if isinstance(rewrites, dict) else list(rewrites)
    by_source = {}
    for src in sorted({r["source"] for r in manifest}):
        by_source[src] = {"manifest": sum(r["source"] == src for r in manifest),
                          "train": sum(r["source"] == src for r in rows),
                          "dropped": sum(r["source"] == src for r, _ in dropped),
                          "missing": sum(r["source"] == src for r in missing),
                          "by_style": _counts(r["instruction_style"] for r in rows if r["source"] == src)}
    return {"rewrites": {"records": len(recs), "prompt_versions": _counts(str(r.get("prompt_version")) for r in recs),
                         "models": _counts(str(r.get("model")) for r in recs)},
            "rows": {"manifest": len(manifest), "train": len(rows), "dropped": len(dropped), "missing": len(missing)},
            "by_style": _counts(r["instruction_style"] for r in rows),
            "by_source": by_source,
            "dropped": {"by_source": _counts(r["source"] for r, _ in dropped),
                        "by_reason": _counts(reason for _, reason in dropped)},
            "missing": {"by_source": _counts(r["source"] for r in missing)}}


def _csv(text: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in text.split(",") if s.strip())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--rewrites", type=Path, required=True, help="rewrites.jsonl from rewrite.py")
    ap.add_argument("--out", type=Path, required=True, help="directory for train_lightning/train_template jsonl")
    ap.add_argument("--allow-missing", action="store_true", help="keep raw labels for selected rows with no rewrite")
    ap.add_argument("--raw-sources", type=_csv, default=(), help="extra raw-label sources, comma-separated")
    ap.add_argument("--natural-sources", type=_csv, default=(), help="extra natural-label sources, comma-separated")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = meta.load_manifest(args.manifest, extra_raw=args.raw_sources, extra_natural=args.natural_sources)
        rewrites = load_jsonl(args.rewrites)
        rows, dropped, missing = _apply(manifest, rewrites, args.raw_sources, args.natural_sources, args.allow_missing)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    template = template_variant(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {"train_lightning.jsonl": rows, "train_template.jsonl": template}
    for name, data in outputs.items():
        _write_jsonl(args.out / name, data)
    counts = build_stats(manifest, rewrites, rows, dropped, missing)
    stats = {"manifest": {"path": str(args.manifest), "sha256": _sha256(args.manifest), "rows": len(manifest)},
             "rewrites": {"path": str(args.rewrites), "sha256": _sha256(args.rewrites), **counts.pop("rewrites")},
             "image_root": str(args.manifest.resolve().parent),
             "image_path_note": "image_path is unchanged from the manifest, relative to image_root (the manifest dir)",
             "allow_missing": args.allow_missing,
             "raw_sources": list(args.raw_sources), "natural_sources": list(args.natural_sources),
             "outputs": {n: {"rows": len(d), "sha256": _sha256(args.out / n)} for n, d in outputs.items()},
             **counts}
    (args.out / "apply_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(manifest)} manifest rows -> {len(rows)} train rows ({len(dropped)} dropped, {len(missing)} missing)")
    print(f"styles: {stats['by_style']}")
    if dropped:
        print(f"drop reasons: {stats['dropped']['by_reason']}")
    print(f"wrote {', '.join(str(args.out / n) for n in (*outputs, 'apply_stats.json'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
