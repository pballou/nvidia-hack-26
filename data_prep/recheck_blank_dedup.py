"""Re-check the SSv2 images finalize_manifest.py's pHash dedup skipped as "too blank to be a
reliable phash match target" (MIN_SSV2_STDDEV guard), using methods that don't share phash's
blind spot: exact MD5 (raw bytes and decoded pixels, catches literal reuse/re-encoding) plus a
downscaled pixel-diff and a simple global SSIM (catches near-identical content phash collapsed
past, now that we're not relying on phash at all for these).

Example:
  python data_prep/recheck_blank_dedup.py --manifest /home/ubuntu/data/os_atlas_20k/manifest.jsonl \
      --screenspot-images /home/ubuntu/data/raw/screenspot_v2/screenspotv2_image \
      --stats /home/ubuntu/data/os_atlas_20k/manifest.stats.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

DOWNSCALE = 32  # small enough to be cheap at 79 x ~4300 pairs, large enough to catch near-dupes
PIXEL_DIFF_FLAG = 0.03  # mean absolute difference on [0,1]-normalized downscaled grayscale
SSIM_FLAG = 0.90  # global single-window SSIM (simplified: no sliding window, whole-image stats)


def md5_bytes(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def md5_pixels(im: Image.Image) -> str:
    return hashlib.md5(np.asarray(im.convert("RGB")).tobytes()).hexdigest()


def downscaled_gray(path: Path) -> tuple[np.ndarray, Image.Image]:
    im = Image.open(path)
    small = im.convert("L").resize((DOWNSCALE, DOWNSCALE), Image.BILINEAR)
    return np.asarray(small, dtype=np.float64) / 255.0, im


def global_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Single-window SSIM over the whole (small) image; good enough at 32x32 for a coarse check."""
    c1, c2 = (0.01 * 1) ** 2, (0.03 * 1) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--screenspot-images", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True, help="finalize_manifest.py's <out>.stats.json")
    ap.add_argument("--out", type=Path, default=None, help="default: <manifest dir>/blank_dedup_recheck.json")
    args = ap.parse_args()

    stats = json.loads(args.stats.read_text())
    blanks = stats["dedup"]["ssv2_images_skipped_as_blank"]
    train_images = sorted({json.loads(l)["image_path"] for l in args.manifest.open() if l.strip()})
    root = args.manifest.parent

    print(f"re-checking {len(blanks)} blank-skipped SSv2 images against {len(train_images)} training images", flush=True)

    train_md5_bytes, train_md5_pixels = {}, {}
    train_small = np.empty((len(train_images), DOWNSCALE, DOWNSCALE))
    for i, p in enumerate(train_images):
        full = root / p
        train_md5_bytes[md5_bytes(full)] = p
        small, im = downscaled_gray(full)
        train_md5_pixels[md5_pixels(im)] = p
        train_small[i] = small
        if (i + 1) % 1000 == 0:
            print(f"  hashed {i + 1}/{len(train_images)} training images", flush=True)

    flat = train_small.reshape(len(train_images), -1)
    findings = []
    for name in blanks:
        full = args.screenspot_images / name
        b_bytes, b_small, b_im = md5_bytes(full), *downscaled_gray(full)
        b_pixels = md5_pixels(b_im)
        exact_bytes = train_md5_bytes.get(b_bytes)
        exact_pixels = train_md5_pixels.get(b_pixels)

        diffs = np.abs(flat - b_small.reshape(1, -1)).mean(axis=1)
        order = np.argsort(diffs)[:3]
        near = []
        for idx in order:
            d = float(diffs[idx])
            s = global_ssim(train_small[idx], b_small)
            near.append({"training_image": train_images[idx], "pixel_diff": round(d, 4), "ssim": round(s, 4),
                         "flagged": d <= PIXEL_DIFF_FLAG or s >= SSIM_FLAG})
        findings.append({
            "screenspot_v2_image": name,
            "exact_md5_bytes_match": exact_bytes, "exact_md5_pixels_match": exact_pixels,
            "nearest_by_pixel_diff": near,
            "any_flagged": bool(exact_bytes or exact_pixels or any(n["flagged"] for n in near)),
        })
        if findings[-1]["any_flagged"]:
            print(f"  FLAG {name}: {findings[-1]}", flush=True)

    out = args.out or args.manifest.parent / "blank_dedup_recheck.json"
    n_flagged = sum(f["any_flagged"] for f in findings)
    summary = {
        "method": f"MD5 (raw bytes + decoded pixels) exact match, plus {DOWNSCALE}x{DOWNSCALE} grayscale "
                  f"mean-abs-diff (flag <= {PIXEL_DIFF_FLAG}) and single-window SSIM (flag >= {SSIM_FLAG})",
        "ssv2_images_checked": len(blanks), "training_images_checked": len(train_images),
        "flagged": n_flagged, "findings": findings,
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n{n_flagged}/{len(blanks)} blank-skipped SSv2 images flagged by the fallback check")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
