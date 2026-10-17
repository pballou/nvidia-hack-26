"""Sample a grounding slice (OS-Atlas, optionally GroundCUA) into canonical JSONL plus extracted images.

Default quotas follow the training data mix in docs/plan.md (eval option C: Linux/macOS are
trained on, and SSv2-clean drops the 28 SSv2 rows captured on the same machines). Windows,
GroundCUA and Widget Captioning are opt-in via --quotas. Archives already under --raw are
read locally; anything missing is range-read from Hugging Face.

Example:
  python data_prep/sample_os_atlas.py --raw /home/ubuntu/data/raw/os_atlas --out /home/ubuntu/data/os_atlas_20k
"""

import argparse
import io
import json
import random
import re
import tarfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import fsspec
import ijson
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image

ATLAS = "OS-Copilot/OS-Atlas-data"
GROUNDCUA = "ServiceNow/GroundCUA"

PLATFORM = {
    "linux": "desktop",
    "macos": "desktop",
    "windows": "desktop",
    "groundcua": "desktop",
    "androidworld": "mobile",
    "uibert": "mobile",
    "widget_captioning": "mobile",
    "fineweb": "web",
}
DEFAULT_QUOTAS = {
    "linux": 3500,
    "macos": 3500,
    "androidworld": 3000,
    "uibert": 3000,
    "fineweb": 7000,
}
LOCAL_ZIPS = {
    "linux": ("desktop_domain/linux_splited.json", "desktop_domain/linux_images.zip"),
    "macos": ("desktop_domain/macos_splited.json", "desktop_domain/macos_images.zip"),
    "androidworld": ("mobile_domain/aw_mobile.json", "mobile_domain/mobile_images.zip"),
    "uibert": ("mobile_domain/uibert_raw.json", "mobile_domain/UIBert.zip"),
}
WINDOWS_PARTS = [f"desktop_domain/windows_image_a{s}" for s in "abcd"]
GROUNDCUA_STYLES = {
    "icon": "direct_icon_instructions.json",
    "functional": "functional_instructions.json",
    "text": "direct_text_instructions.json",
}

MAX_WORDS = 12
# GroundCUA icon instructions describe the icon's position, so they run longer.
MAX_WORDS_BY_STYLE = {"groundcua/icon": 20}
MAX_AREA = 0.15
JUNK_RE = re.compile(r"automationid:|https?://|www\.|running window", re.I)
PUA_RE = re.compile(r"[\ue000-\uf8ff]")


def clean_elements(elements, drops):
    """Apply the research-notes quality filters; drop labels that point at 2+ distinct boxes."""
    by_key = defaultdict(set)
    first = {}
    for e in elements:
        text = " ".join(unicodedata.normalize("NFKC", str(e["instruction"])).split())
        l, t, r, b = (float(v) for v in e["bbox"])
        if not (0 <= l < r <= 1 and 0 <= t < b <= 1):
            drops["bad_box"] += 1
            continue
        if (r - l) * (b - t) > MAX_AREA:
            drops["box_over_15pct"] += 1
            continue
        if len(text) < 2 or PUA_RE.search(text) or not re.search(r"[A-Za-z]", text):
            drops["glyph_or_no_letters"] += 1
            continue
        if JUNK_RE.search(text):
            drops["junk_text"] += 1
            continue
        if len(text.split()) > MAX_WORDS_BY_STYLE.get(e.get("style"), MAX_WORDS):
            drops["over_12_words"] += 1
            continue
        key = text.lower().strip(" .,:;!?\"'")
        by_key[key].add((round(l, 4), round(t, 4), round(r, 4), round(b, 4)))
        first.setdefault(key, (text, e.get("role"), e.get("style")))
    kept = []
    for key, boxes in by_key.items():
        if len(boxes) > 1:
            drops["ambiguous_label"] += 1
            continue
        text, role, style = first[key]
        kept.append((text, next(iter(boxes)), role, style))
    return kept


class MultiPartFile(io.RawIOBase):
    """Read a zip that was split into byte-ranged parts (`split`) as one seekable file over HTTP."""

    def __init__(self, fs, paths):
        self.files = [fs.open(p, "rb", block_size=1 << 20) for p in paths]
        self.offsets = [0]
        for p in paths:
            self.offsets.append(self.offsets[-1] + fs.info(p)["size"])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.pos, 2: self.offsets[-1]}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n=-1):
        end = self.offsets[-1]
        n = end - self.pos if n is None or n < 0 else n
        out = bytearray()
        while n > 0 and self.pos < end:
            i = max(k for k in range(len(self.files)) if self.offsets[k] <= self.pos)
            f = self.files[i]
            f.seek(self.pos - self.offsets[i])
            chunk = f.read(min(n, self.offsets[i + 1] - self.pos))
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)


class Sampler:
    def __init__(self, raw, out, cap, keep_prob, rng, sink):
        self.raw, self.out, self.cap, self.keep_prob, self.rng, self.sink = raw, out, cap, keep_prob, rng, sink
        self.stats = defaultdict(Counter)
        self.rows_per_image = Counter()
        self.fs = HfFileSystem()
        self.local_fs = fsspec.filesystem("file")

    def ensure(self, rel):
        """Local path for an OS-Atlas file, downloading it only if it isn't already under --raw."""
        if not (self.raw / rel).exists():
            hf_hub_download(ATLAS, rel, repo_type="dataset", local_dir=self.raw)
        return self.raw / rel

    def open_atlas(self, rel, block_size):
        """Open an OS-Atlas file locally if present, otherwise as a ranged HTTP stream."""
        if (self.raw / rel).exists():
            return open(self.raw / rel, "rb")
        return self.fs.open(f"datasets/{ATLAS}/{rel}", "rb", block_size=block_size)

    def windows_zip(self):
        if all((self.raw / p).exists() for p in WINDOWS_PARTS):
            return MultiPartFile(self.local_fs, [str(self.raw / p) for p in WINDOWS_PARTS])
        return MultiPartFile(self.fs, [f"datasets/{ATLAS}/{p}" for p in WINDOWS_PARTS])

    def remaining(self, source, quota):
        return quota - self.stats[source]["targets"]

    def emit(self, source, img_name, group, fetch, elements, quota, pixel_boxes=False):
        """Write one screenshot and up to `cap` cleaned targets; elements carry normalized boxes unless pixel_boxes."""
        dest = self.out / "images" / source / img_name.replace("/", "_")
        dest.parent.mkdir(parents=True, exist_ok=True)
        fresh = not dest.exists()
        if fresh:
            dest.write_bytes(fetch())
        with Image.open(dest) as im:
            w, h = im.size
        if pixel_boxes:
            elements = [{**e, "bbox": [e["bbox"][0] / w, e["bbox"][1] / h, e["bbox"][2] / w, e["bbox"][3] / h]}
                        for e in elements]
        kept = clean_elements(elements, self.stats[source])
        self.rng.shuffle(kept)
        picks = kept[: min(self.cap, self.remaining(source, quota))]
        if not picks:
            if fresh:
                dest.unlink()
            return
        start = self.rows_per_image[dest]
        self.rows_per_image[dest] += len(picks)
        if fresh:
            self.stats[source]["screenshots"] += 1
        for k, (text, (l, t, r, b), role, style) in enumerate(picks, start):
            self.sink.write(json.dumps({
                "id": f"{source}:{Path(img_name).stem}:{k}",
                "source": f"{'groundcua' if source == 'groundcua' else 'os_atlas'}/{source}",
                "image_path": str(dest.relative_to(self.out)),
                "img_w": w,
                "img_h": h,
                "instruction": text,
                "bbox": [round(l * w, 1), round(t * h, 1), round(r * w, 1), round(b * h, 1)],
                "platform": PLATFORM[source],
                "data_type": role,
                "instruction_style": style,
                "group": group,
                "split": None,
            }) + "\n")
        self.stats[source]["targets"] += len(picks)

    def local_zip(self, source, ann, zpath, quota, group_fn):
        ann_path, zip_path = self.ensure(ann), self.ensure(zpath)
        merged = defaultdict(list)
        for row in json.load(open(ann_path)):
            for e in row["elements"]:
                merged[row["img_filename"]].append({"instruction": e["instruction"], "bbox": e["bbox"],
                                                    "role": e.get("data_type") or None})
        images = sorted(merged)
        self.rng.shuffle(images)
        with zipfile.ZipFile(zip_path) as z:
            members = set(z.namelist())
            for img in images:
                if self.remaining(source, quota) <= 0:
                    break
                if img not in members:
                    self.stats[source]["missing_image"] += 1
                    continue
                self.emit(source, img, group_fn(img), lambda: z.read(img), merged[img], quota)

    def streamed(self, source, ann, zip_file, quota, group_fn):
        """Stream annotations and read only the chosen images from a (local or remote) zip."""
        with zipfile.ZipFile(zip_file) as z:
            members = set(z.namelist())
            with self.open_atlas(ann, 4 << 20) as jf:
                for row in ijson.items(jf, "item"):
                    if self.remaining(source, quota) <= 0:
                        break
                    img = row["img_filename"]
                    if img not in members or self.rng.random() > self.keep_prob:
                        continue
                    elements = [{"instruction": e["instruction"], "bbox": e["bbox"], "role": e.get("data_type") or None}
                                for e in row["elements"]]
                    self.emit(source, img, group_fn(img), lambda: z.read(img), elements, quota)

    def widget_captioning(self, quota):
        ann = "mobile_domain/widget_captioning.json"
        merged = defaultdict(list)
        for row in json.load(open(self.ensure(ann))):
            merged[row["img_filename"]].append({"instruction": row["instruction"], "bbox": row["bbox"], "role": None})
        images = sorted(merged)
        self.rng.shuffle(images)
        zf = self.open_atlas("mobile_domain/rico_imgs.zip", 1 << 20)
        with zipfile.ZipFile(zf) as z:
            members = set(z.namelist())
            for img in images:
                if self.remaining("widget_captioning", quota) <= 0:
                    break
                member = f"combined/{img}"
                if member not in members:
                    self.stats["widget_captioning"]["missing_image"] += 1
                    continue
                self.emit("widget_captioning", img, f"rico:{Path(img).stem}", lambda: z.read(member),
                          merged[img], quota)

    def groundcua(self, quota):
        root = self.raw.parent / "groundcua"
        tar = Path(hf_hub_download(GROUNDCUA, "instruction_tuning.tar.gz", repo_type="dataset", local_dir=root))
        ann_dir = root / "groundcua_data"
        if not ann_dir.exists():
            with tarfile.open(tar) as t:
                t.extractall(root, filter="data")
        # Equal share per instruction style; icon instructions exist on far fewer screenshots than the others.
        styles = list(GROUNDCUA_STYLES.items())
        for i, (style, fname) in enumerate(styles):
            cumulative_quota = quota * (i + 1) // len(styles)
            platform_of = {}
            with open(ann_dir / fname, "rb") as f:
                for r in ijson.items(f, "item"):
                    platform_of[r["id"]] = r["platform"]
            ids = sorted(platform_of)
            self.rng.shuffle(ids)
            chosen = set(ids[:cumulative_quota])
            records = defaultdict(list)
            with open(ann_dir / fname, "rb") as f:
                for r in ijson.items(f, "item"):
                    if r["id"] in chosen:
                        records[r["id"]].append({"instruction": r["instruction"], "bbox": [float(v) for v in r["bbox"]],
                                                 "role": r.get("category") or None, "style": f"groundcua/{style}"})
            for img_id in ids[:cumulative_quota]:
                if self.remaining("groundcua", cumulative_quota) <= 0:
                    break
                plat = platform_of[img_id]
                rel = f"images/{plat}/{img_id}.png"
                fetch = lambda: Path(hf_hub_download(GROUNDCUA, rel, repo_type="dataset", local_dir=root)).read_bytes()
                self.emit("groundcua", f"{img_id}.png", f"groundcua:{plat}", fetch, records[img_id],
                          cumulative_quota, pixel_boxes=True)


def strip_sub(name):
    return re.sub(r"_sub\d+(?=\.\w+$)", "", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True, help="OS-Atlas root (local files are used when present)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap-per-screenshot", type=int, default=8)
    ap.add_argument("--keep-prob", type=float, default=0.25,
                    help="per-screenshot keep probability while streaming Windows/FineWeb, to spread the sample")
    ap.add_argument("--quotas", type=json.loads, default=DEFAULT_QUOTAS, help="JSON {source: max targets}")
    args = ap.parse_args()

    unknown = sorted(set(args.quotas) - set(PLATFORM))
    if unknown:
        raise SystemExit(f"unknown sources {unknown}; choose from {sorted(PLATFORM)}")

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    with open(args.out / "manifest.jsonl", "w") as sink:
        s = Sampler(args.raw, args.out, args.cap_per_screenshot, args.keep_prob, rng, sink)
        for source, quota in args.quotas.items():
            print(f"sampling {source} (quota {quota})", flush=True)
            if source in LOCAL_ZIPS:
                group_fn = (lambda n: f"rico:{Path(n).stem}") if source == "uibert" else \
                    (lambda n, src=source: f"{src}:{strip_sub(n)}")
                s.local_zip(source, *LOCAL_ZIPS[source], quota, group_fn)
            elif source == "widget_captioning":
                s.widget_captioning(quota)
            elif source == "groundcua":
                s.groundcua(quota)
            elif source == "windows":
                s.streamed(source, "desktop_domain/windows_splited.json", s.windows_zip(), quota,
                           lambda n: f"windows:{strip_sub(n)}")
            elif source == "fineweb":
                zf = s.open_atlas("web_domain/fineweb_3m_s11.zip", 1 << 20)
                s.streamed(source, "web_domain/fineweb_3m.json", zf, quota,
                           lambda n: f"fineweb:{n.split('_section_')[0]}")
            print(f"  {dict(s.stats[source])}", flush=True)

    summary = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "per_source": {k: dict(v) for k, v in s.stats.items()},
               "total_targets": sum(v["targets"] for v in s.stats.values()),
               "total_screenshots": sum(v["screenshots"] for v in s.stats.values()),
               "not_yet_done": ["pHash dedup vs ScreenSpot-v2 (all sources, incl. FineWeb overlap check)",
                                "dev split by `group`", "text/icon proxy (data_type holds the control type)"]}
    (args.out / "stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
