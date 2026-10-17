"""Flag screenshots with burned-in annotation overlays (pure-red boxes + "role: label" text, seen in OS-Atlas Linux).

Writes <manifest dir>/overlay_flagged.json; dropping the rows is the sampler owner's call, not this script's.

Example:
  python data_prep/rewrite/overlay_scan.py --manifest data/os_atlas_20k/manifest.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageChops

RED_MIN, OTHER_MAX = 240, 30  # a pixel is "pure red" if R >= 240 and G, B <= 30
THRESHOLD = 0.002  # flag images whose pure-red pixel fraction exceeds this


def red_fraction(path: Path) -> float:
    """Fraction of full-resolution pixels that are pure red (C-speed Pillow channel ops, no downscaling)."""
    with Image.open(path) as im:
        r, g, b = im.convert("RGB").split()
    mask = ImageChops.multiply(r.point(lambda v: 255 if v >= RED_MIN else 0),
                               g.point(lambda v: 255 if v <= OTHER_MAX else 0))
    mask = ImageChops.multiply(mask, b.point(lambda v: 255 if v <= OTHER_MAX else 0))
    return mask.histogram()[255] / (mask.width * mask.height)


def _scan(args):
    path, root = args
    try:
        return path, red_fraction(root / path), None
    except OSError as e:
        return path, None, str(e)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: <manifest dir>/overlay_flagged.json")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sources", default="os_atlas/linux",
                    help="comma list whose hits go in 'flagged'; hits elsewhere go in 'other_candidates' (on the "
                         "20k slice they were all legitimately red UI, checked by eye). Use 'all' to flag every source")
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in open(args.manifest, encoding="utf-8") if line.strip()]
    source, n_rows = {}, Counter()
    for r in rows:
        source.setdefault(r["image_path"], r["source"])
        n_rows[r["image_path"]] += 1
    root = args.manifest.parent
    with ProcessPoolExecutor(args.workers) as pool:
        results = sorted(pool.map(_scan, [(p, root) for p in sorted(source)], chunksize=16))

    scope = None if args.sources == "all" else {s.strip() for s in args.sources.split(",") if s.strip()}
    hits = [{"image_path": p, "source": source[p], "red_fraction": round(f, 5), "rows": n_rows[p]}
            for p, f, _ in results if f is not None and f > args.threshold]
    hits.sort(key=lambda x: (x["source"], -x["red_fraction"], x["image_path"]))
    flagged = [x for x in hits if scope is None or x["source"] in scope]
    others = [x for x in hits if not (scope is None or x["source"] in scope)]
    errors = [{"image_path": p, "error": e} for p, _, e in results if e]
    by_source = {}
    for src in sorted(set(source.values())):
        imgs = [p for p in source if source[p] == src]
        hit = [x for x in flagged if x["source"] == src]
        by_source[src] = {"images": len(imgs), "flagged_images": len(hit), "rows": sum(n_rows[p] for p in imgs),
                          "flagged_rows": sum(x["rows"] for x in hit),
                          "other_candidates": sum(x["source"] == src for x in others)}
    out = {"manifest": str(args.manifest), "rule": f"pure-red pixels (R >= {RED_MIN}, G <= {OTHER_MAX}, "
           f"B <= {OTHER_MAX}) / all pixels > {args.threshold}, full resolution; flagged sources: {args.sources}",
           "command": " ".join(["python", "data_prep/rewrite/overlay_scan.py", *(sys.argv[1:] if argv is None else argv)]),
           "summary": {"images": len(source), "flagged_images": len(flagged), "rows": len(rows),
                       "flagged_rows": sum(x["rows"] for x in flagged), "by_source": by_source},
           "flagged": flagged,
           "other_candidates_note": "over the threshold but outside --sources: on the 20k slice every one was "
                                    "legitimately red UI (buttons, banners, icons), not an overlay; do not drop",
           "other_candidates": others, "errors": errors}
    path = args.out or args.manifest.parent / "overlay_flagged.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out["summary"], indent=1))
    print(f"wrote {path} ({len(flagged)} flagged images, {len(errors)} unreadable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
