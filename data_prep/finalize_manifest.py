"""One pass over the sampler's manifest: drop burned-in-overlay rows, dedup against
ScreenSpot-v2 by perceptual hash, and write the train/dev split.

Writes a backup of the input next to it, then overwrites manifest.jsonl in place
(or writes --out if given). Also writes <out>.stats.json with every dropped id and
every count, so the run is auditable.

Example:
  python data_prep/finalize_manifest.py --manifest /home/ubuntu/data/os_atlas_20k/manifest.jsonl \
      --screenspot-images /home/ubuntu/data/raw/screenspot_v2/screenspotv2_image \
      --overlay-flagged /home/ubuntu/data/os_atlas_20k/overlay_flagged.json
"""

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
from PIL import Image, ImageStat

PHASH_HASH_SIZE = 8  # imagehash's standard 64-bit phash
DEFAULT_DEDUP_THRESHOLD = 8  # Hamming distance on the 64-bit hash; see --dedup-threshold. Calibrate per
# dataset: at this hash size, screenshots (lots of shared UI chrome/whitespace) cluster well below the
# ~32 "random pair" distance, so always inspect min_distance_percentiles before trusting a threshold.
MIN_SSV2_STDDEV = 25  # skip near-blank/low-detail SSv2 images as match targets (see --dedup-threshold docstring
# below): phash collides easily on them, giving false "duplicate" hits unrelated in content. Confirmed by hand
# on this dataset: both matches below this stddev (4.4 and 20.1) were different apps/platforms that just
# happened to both be mostly-blank screens.
DEV_FRACTION = 0.05


def load_rows(manifest: Path) -> list[dict]:
    return [json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()]


def overlay_drop_set(flagged_path: Path | None) -> set[str]:
    if not flagged_path:
        return set()
    data = json.loads(flagged_path.read_text())
    return {f["image_path"] for f in data["flagged"]}


def phash_of(path: Path):
    with Image.open(path) as im:
        return imagehash.phash(im.convert("RGB"), hash_size=PHASH_HASH_SIZE)


def dedup_against_screenspot(rows: list[dict], root: Path, ssv2_dir: Path, threshold: int) -> tuple[set[str], dict]:
    """Perceptual-hash dedup of every unique training image against every (non-degenerate) SSv2 image."""
    ssv2_images = sorted(ssv2_dir.glob("*.png")) + sorted(ssv2_dir.glob("*.jpg"))
    skipped_blank = [p.name for p in ssv2_images if ImageStat.Stat(Image.open(p).convert("L")).stddev[0] < MIN_SSV2_STDDEV]
    ssv2_hashes = {p.name: phash_of(p) for p in ssv2_images if p.name not in skipped_blank}
    train_images = sorted({r["image_path"] for r in rows})

    drop, distances, matches = set(), [], {}
    for i, img in enumerate(train_images):
        h = phash_of(root / img)
        best_name, best_dist = min(((n, h - hh) for n, hh in ssv2_hashes.items()), key=lambda x: x[1])
        distances.append(best_dist)
        if best_dist <= threshold:
            drop.add(img)
            matches[img] = {"screenspot_v2_image": best_name, "hamming_distance": best_dist}
        if (i + 1) % 1000 == 0:
            print(f"  phash: {i + 1}/{len(train_images)} training images checked", flush=True)

    distances.sort()
    report = {
        "threshold": threshold, "hash_size_bits": PHASH_HASH_SIZE ** 2,
        "ssv2_images": len(ssv2_hashes), "ssv2_images_skipped_as_blank": skipped_blank,
        "training_images_checked": len(train_images),
        "flagged_images": len(drop),
        "min_distance_percentiles": {p: distances[int(len(distances) * p / 100)] for p in (1, 2, 5, 10, 25, 50)},
        "matches": matches,
    }
    return drop, report


def assign_split(rows: list[dict], seed: int, dev_fraction: float) -> dict:
    """5% of groups per source go to dev; the rest train. Never both for one group."""
    groups_by_source = defaultdict(set)
    for r in rows:
        groups_by_source[r["source"]].add(r["group"])

    dev_groups = set()
    for source, groups in groups_by_source.items():
        ordered = sorted(groups)
        import random
        random.Random(f"{seed}:{source}").shuffle(ordered)
        n_dev = max(1, round(len(ordered) * dev_fraction))
        dev_groups.update(ordered[:n_dev])

    counts = Counter()
    for r in rows:
        r["split"] = "dev" if r["group"] in dev_groups else "train"
        counts[(r["source"], r["split"])] += 1
    return {"dev_groups": len(dev_groups), "by_source_split": {f"{s}/{sp}": n for (s, sp), n in sorted(counts.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--screenspot-images", type=Path, required=True)
    ap.add_argument("--overlay-flagged", type=Path, default=None,
                    help="output of data_prep/rewrite/overlay_scan.py; omit to skip the overlay-drop step")
    ap.add_argument("--dedup-threshold", type=int, default=DEFAULT_DEDUP_THRESHOLD)
    ap.add_argument("--dev-fraction", type=float, default=DEV_FRACTION)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="default: overwrite --manifest (after a .bak copy)")
    ap.add_argument("--skip-dedup", action="store_true", help="for fast dry runs; never use for a real training manifest")
    args = ap.parse_args()

    root = args.manifest.parent
    rows = load_rows(args.manifest)
    n0 = len(rows)
    stats = {"input_rows": n0, "input_images": len({r["image_path"] for r in rows})}

    overlay_drop = overlay_drop_set(args.overlay_flagged)
    rows = [r for r in rows if r["image_path"] not in overlay_drop]
    stats["overlay_dropped_rows"] = n0 - len(rows)
    stats["overlay_dropped_images"] = len(overlay_drop)
    print(f"overlay: dropped {stats['overlay_dropped_rows']} rows on {len(overlay_drop)} flagged images", flush=True)

    if args.skip_dedup:
        stats["dedup"] = "SKIPPED (--skip-dedup); do not train on this manifest"
    else:
        t0 = time.perf_counter()
        dedup_drop, dedup_report = dedup_against_screenspot(rows, root, args.screenspot_images, args.dedup_threshold)
        rows_before = len(rows)
        rows = [r for r in rows if r["image_path"] not in dedup_drop]
        stats["dedup"] = dedup_report
        stats["dedup"]["dropped_rows"] = rows_before - len(rows)
        stats["dedup"]["elapsed_s"] = round(time.perf_counter() - t0, 1)
        print(f"dedup: dropped {rows_before - len(rows)} rows on {len(dedup_drop)} images "
              f"(threshold {args.dedup_threshold}, {stats['dedup']['elapsed_s']}s)", flush=True)

    stats["split"] = assign_split(rows, args.seed, args.dev_fraction)
    print(f"split: {stats['split']['dev_groups']} dev groups -> {stats['split']['by_source_split']}", flush=True)

    out = args.out or args.manifest
    if out == args.manifest:
        backup = args.manifest.with_suffix(".jsonl.bak")
        shutil.copy(args.manifest, backup)
        print(f"backed up original to {backup}", flush=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    stats["output_rows"] = len(rows)
    stats["output_path"] = str(out)
    (out.with_suffix(".stats.json")).write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out} ({n0 - len(rows)} dropped total)")


if __name__ == "__main__":
    main()
