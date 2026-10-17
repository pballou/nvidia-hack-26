"""Synthetic canonical manifests (sample_os_atlas.py schema) with small PNGs, so tests never touch real data."""

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

DEFAULT_SOURCES = ("os_atlas/androidworld", "os_atlas/fineweb", "os_atlas/uibert")
PLATFORMS = {"androidworld": "mobile", "uibert": "mobile", "widget_captioning": "mobile", "fineweb": "web",
             "windows": "desktop"}
SIZES = {"mobile": (360, 720), "web": (640, 400), "desktop": (640, 400)}
RAW_LABELS = {
    "mobile": ["More options", "Navigate up", "Search", "Settings", "Send message", "Camera", "Bluetooth", "Gmail",
               "Battery saver", "YouTube", "Wi-Fi", "Notifications", "Add contact", "Share"],
    "web": ["Sign in", "Download PDF", "Contact us", "Privacy Policy", "Next page", "Subscribe", "Add to cart",
            "Read more", "iPhone 15 Pro", "USB-C cable", "Menu", "Search", "Pricing", "Docs"],
    "desktop": ["File", "Edit", "View", "Save As", "Recycle Bin", "Microsoft Edge", "Close", "Minimize", "Start",
                "Task View", "Control Panel", "Notepad", "Paste", "Copy"],
}
NATURAL_LABELS = ["click on the search icon", "open the settings menu", "go back to the previous page",
                  "select the camera button", "share this photo", "turn on bluetooth", "open the navigation drawer",
                  "add a new contact", "send the message", "view more options", "show notifications",
                  "play the video", "check the battery saver", "open gmail"]
ROLES = {"mobile": ["android.widget.ImageButton", "android.widget.TextView", "android.widget.Button", None],
         "web": [None, None, "button", "link"], "desktop": ["ControlType.Button", "ControlType.MenuItem", None]}
COLORS = [(245, 245, 245), (30, 30, 36), (230, 240, 255), (255, 250, 235), (220, 235, 220)]


def _subset(source: str) -> str:
    return source.removeprefix("os_atlas/").replace("/", "_")


def _box(rng, w, h) -> list[float]:
    bw, bh = rng.randint(max(8, w // 40), max(16, w // 4)), rng.randint(max(8, h // 60), max(16, h // 10))
    x1, y1 = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
    return [round(x1, 1), round(y1, 1), round(x1 + bw, 1), round(y1 + bh, 1)]


def make_rows(*, n_images=8, per_image=6, seed=0, sources=DEFAULT_SOURCES) -> list[dict]:
    """Rows for n_images screenshots (round-robin over sources), per_image targets each. Raw-label images rotate
    in a numeric label ("42"), junk ("AutomationID:x"), a proper noun ("GitHub") and a duplicated label per shot."""
    rng = random.Random(seed)
    rows = []
    for i in range(n_images):
        source = sources[i % len(sources)]
        subset = _subset(source)
        platform = PLATFORMS.get(subset, "desktop" if subset.startswith("groundcua") else "web")
        w, h = SIZES[platform]
        natural = subset in ("uibert", "widget_captioning") or subset.startswith("groundcua")
        pool = NATURAL_LABELS if natural else RAW_LABELS[platform]
        labels = rng.sample(pool, min(per_image, len(pool)))
        labels += [f"{rng.choice(pool)} {j}" for j in range(per_image - len(labels))]
        if not natural and per_image >= 3:
            labels[-1] = ("42", "AutomationID:x", "GitHub", labels[-1])[i % 4]
            if i % 3 == 0:
                labels[1] = labels[0]
        stem = f"{subset}_{seed}_{i:04d}"
        for k, label in enumerate(labels):
            kind = rng.choice([None, None, "text", "icon"]) if platform == "web" else None
            rows.append({"id": f"{subset}:{stem}:{k}", "source": source, "image_path": f"images/{subset}/{stem}.png",
                         "img_w": w, "img_h": h, "instruction": label, "bbox": _box(rng, w, h), "platform": platform,
                         "data_type": kind, "role": rng.choice(ROLES[platform]), "split": None})
    return rows


def write_pngs(root: Path, rows) -> None:
    """One solid-color PNG per image_path with every target box outlined."""
    by_image = {}
    for r in rows:
        by_image.setdefault(r["image_path"], []).append(r)
    for n, (rel, targets) in enumerate(by_image.items()):
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (targets[0]["img_w"], targets[0]["img_h"]), COLORS[n % len(COLORS)])
        draw = ImageDraw.Draw(img)
        for t in targets:
            draw.rectangle(t["bbox"], outline=(90, 120, 200), width=2)
        img.save(dest)


def make_manifest(dir: Path, *, n_images=8, per_image=6, seed=0, sources=DEFAULT_SOURCES, write_images=True) -> Path:
    """Write dir/manifest.jsonl (+ dir/images/<subset>/*.png) and return the manifest path."""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    rows = make_rows(n_images=n_images, per_image=per_image, seed=seed, sources=sources)
    path = dir / "manifest.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if write_images:
        write_pngs(dir, rows)
    return path

