"""Sample a grounding slice from OS-Atlas into canonical JSONL plus extracted images.

Example:
  python data_prep/sample_os_atlas.py --raw /home/ubuntu/data/os_atlas --out /home/ubuntu/data/os_atlas_20k
"""

import argparse
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import ijson
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image

REPO = "OS-Copilot/OS-Atlas-data"

LOCAL_SUBSETS = {
    "linux": ("desktop", "desktop_domain/linux_splited.json", "desktop_domain/linux_images.zip"),
    "macos": ("desktop", "desktop_domain/macos_splited.json", "desktop_domain/macos_images.zip"),
    "androidworld": ("mobile", "mobile_domain/aw_mobile.json", "mobile_domain/mobile_images.zip"),
    "uibert": ("mobile", "mobile_domain/uibert_raw.json", "mobile_domain/UIBert.zip"),
}
WEB_JSON = "web_domain/fineweb_3m.json"
WEB_ZIP = "web_domain/fineweb_3m_s11.zip"

DEFAULT_QUOTAS = {"linux": 3500, "macos": 3500, "androidworld": 3000, "uibert": 3000, "fineweb": 7000}


def clean_elements(elements):
    """Drop invalid boxes and unusable labels; drop labels that point at 2+ distinct boxes."""
    by_text = defaultdict(set)
    meta = {}
    for e in elements:
        text = " ".join(str(e.get("instruction", "")).split())
        l, t, r, b = (float(v) for v in e["bbox"])
        if not (0 <= l < r <= 1 and 0 <= t < b <= 1):
            continue
        if len(text) < 2 or not re.search(r"[A-Za-z]", text):
            continue
        box = (round(l, 4), round(t, 4), round(r, 4), round(b, 4))
        by_text[text].add(box)
        meta[text] = e.get("data_type") or None
    return [(text, next(iter(boxes)), meta[text]) for text, boxes in by_text.items() if len(boxes) == 1]


def merge_by_image(rows):
    merged = defaultdict(list)
    for r in rows:
        merged[r["img_filename"]].extend(r["elements"])
    return merged


def emit(out, source, platform, img_name, img_bytes_fn, picks, sink, stats):
    dest = out / "images" / source / img_name.replace("/", "_")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(img_bytes_fn())
    with Image.open(dest) as im:
        w, h = im.size
    for k, (text, (l, t, r, b), role) in enumerate(picks):
        sink.write(json.dumps({
            "id": f"{source}:{Path(img_name).stem}:{k}",
            "source": f"os_atlas/{source}",
            "image_path": str(dest.relative_to(out)),
            "img_w": w,
            "img_h": h,
            "instruction": text,
            "bbox": [round(l * w, 1), round(t * h, 1), round(r * w, 1), round(b * h, 1)],
            "platform": platform,
            "data_type": None,
            "role": role,
            "split": None,
        }) + "\n")
    stats[source]["screenshots"] += 1
    stats[source]["targets"] += len(picks)


def take(cleaned, cap, remaining, rng):
    rng.shuffle(cleaned)
    return cleaned[: min(cap, remaining)]


def sample_local(name, raw, out, quota, cap, rng, sink, stats):
    platform, ann, zpath = LOCAL_SUBSETS[name]
    for f in (ann, zpath):
        hf_hub_download(REPO, f, repo_type="dataset", local_dir=raw)
    merged = merge_by_image(json.load(open(raw / ann)))
    images = sorted(merged)
    rng.shuffle(images)
    with zipfile.ZipFile(raw / zpath) as z:
        members = {m for m in z.namelist() if not m.endswith("/")}
        for img in images:
            remaining = quota - stats[name]["targets"]
            if remaining <= 0:
                break
            if img not in members:
                stats[name]["missing_image"] += 1
                continue
            picks = take(clean_elements(merged[img]), cap, remaining, rng)
            if picks:
                emit(out, name, platform, img, lambda: z.read(img), picks, sink, stats)


def sample_web(out, quota, cap, rng, sink, stats):
    """Stream the web annotations and range-read only the needed images from one shard zip."""
    fs = HfFileSystem()
    zf = fs.open(f"datasets/{REPO}/{WEB_ZIP}", "rb", block_size=1 << 20)
    with zipfile.ZipFile(zf) as z:
        members = set(z.namelist())
        with fs.open(f"datasets/{REPO}/{WEB_JSON}", "rb", block_size=4 << 20) as jf:
            for row in ijson.items(jf, "item"):
                remaining = quota - stats["fineweb"]["targets"]
                if remaining <= 0:
                    break
                img = row["img_filename"]
                if img not in members:
                    continue
                picks = take(clean_elements(row["elements"]), cap, remaining, rng)
                if picks:
                    emit(out, "fineweb", "web", img, lambda: z.read(img), picks, sink, stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True, help="where OS-Atlas annotation files and zips are downloaded")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap-per-screenshot", type=int, default=8)
    ap.add_argument("--quotas", type=json.loads, default=DEFAULT_QUOTAS, help="JSON {subset: max targets}")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    stats = defaultdict(Counter)
    with open(args.out / "manifest.jsonl", "w") as sink:
        for name, quota in args.quotas.items():
            print(f"sampling {name} (quota {quota})", flush=True)
            if name == "fineweb":
                sample_web(args.out, quota, args.cap_per_screenshot, rng, sink, stats)
            else:
                sample_local(name, args.raw, args.out, quota, args.cap_per_screenshot, rng, sink, stats)
            print(f"  {dict(stats[name])}", flush=True)

    summary = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "per_subset": {k: dict(v) for k, v in stats.items()},
               "total_targets": sum(v["targets"] for v in stats.values()),
               "total_screenshots": sum(v["screenshots"] for v in stats.values())}
    (args.out / "stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
