"""Build Gym rows for resources_servers/gui_grounding from canonical grounding samples.

    # OS-Atlas dev split (checkpoint selection)
    python -m data_prep.gym_rows manifest --manifest /home/ubuntu/data/<slice>/manifest.jsonl \
        --split dev --out /home/ubuntu/data/gym/os_atlas_dev.jsonl
    # ScreenSpot-v2-clean (final held-out test): 1,272 rows on load, 1,244 after the exclusion
    python -m data_prep.gym_rows ssv2 --raw /home/ubuntu/data/raw/screenspot_v2 \
        --out /home/ubuntu/data/gym/ssv2_clean.jsonl

- Each row's prompt is common/grounding.py's grounding prompt, and its screenshot is a
  file:// URL to the same pre-resized PNG that training uses (data_prep/images.py).
- The canonical fields verify() needs ride along at the top level. bbox is
  [x1, y1, x2, y2] in pixels of the original img_w x img_h screenshot; the predicted
  0-1000 point is resize-invariant, so verify() converts it with those dims.
- ScreenSpot-v2 is never filtered, except for the 28 desktop rows in
  benchmarks/screenspot_v2/excluded_rows.json (eval option C, AGENTS.md).
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

from common import grounding
from data_prep.images import DEFAULT_CACHE, preresize

SSV2_FILES = {"mobile": "screenspot_mobile_v2.json", "desktop": "screenspot_desktop_v2.json",
              "web": "screenspot_web_v2.json"}
SSV2_EXCLUDED = Path(__file__).resolve().parents[1] / "benchmarks/screenspot_v2/excluded_rows.json"
ROW_FIELDS = ("id", "source", "instruction", "platform", "data_type", "bbox", "img_w", "img_h")


def ssv2_samples(raw: Path) -> list[dict]:
    """ScreenSpot-v2 (revision 5efbb1f1b5) as canonical samples, minus the 28 excluded rows."""
    excluded = {(r["img_filename"], r["instruction"]) for r in json.loads(SSV2_EXCLUDED.read_text())["rows"]}
    loaded, kept = 0, []
    for platform, fname in SSV2_FILES.items():
        for k, r in enumerate(json.loads((raw / fname).read_text())):
            loaded += 1
            if (r["img_filename"], r["instruction"]) in excluded:
                continue
            image_path = f"screenspotv2_image/{r['img_filename']}"
            with Image.open(raw / image_path) as im:
                w, h = im.size
            x, y, bw, bh = r["bbox"]  # SSv2 boxes are [x, y, w, h] in pixels
            kept.append({
                "id": f"ssv2:{platform}:{k}", "source": f"screenspot_v2/{platform}", "image_path": image_path,
                "img_w": w, "img_h": h, "instruction": r["instruction"], "bbox": [x, y, x + bw, y + bh],
                "platform": platform, "data_type": r["data_type"], "split": "test",
            })
    assert loaded == 1272 and len(kept) == 1244, f"expected 1,272 -> 1,244 SSv2 rows, got {loaded} -> {len(kept)}"
    return kept


def gym_row(sample: dict, image_url: str, family: str, max_output_tokens: int) -> dict:
    return {
        "responses_create_params": {
            "input": [{"role": "user", "type": "message", "content": [
                {"type": "input_image", "image_url": image_url, "detail": "auto"},
                {"type": "input_text", "text": grounding.grounding_prompt(sample["instruction"], family)},
            ]}],
            "temperature": 0.0,
            "max_output_tokens": max_output_tokens,
        },
        **{k: sample.get(k) for k in ROW_FIELDS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["manifest", "ssv2"])
    ap.add_argument("--manifest", type=Path, help="canonical manifest JSONL (kind=manifest)")
    ap.add_argument("--split", help="keep only rows with this split (kind=manifest)")
    ap.add_argument("--raw", type=Path, default=Path("/home/ubuntu/data/raw/screenspot_v2"), help="kind=ssv2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, help="random sample of N rows (seed 0)")
    ap.add_argument("--family", choices=grounding.FAMILIES, default="qwen3_5")
    ap.add_argument("--max-pixels", type=int, default=grounding.MAX_PIXELS)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--max-output-tokens", type=int, default=128)
    args = ap.parse_args()

    if args.kind == "ssv2":
        root, samples = args.raw, ssv2_samples(args.raw)
    else:
        root = args.manifest.parent
        samples = [json.loads(line) for line in args.manifest.open()]
        samples = [s for s in samples if args.split is None or s.get("split") == args.split]
    if args.limit:
        samples = random.Random(0).sample(samples, min(args.limit, len(samples)))
    assert samples, "no rows selected"

    cached = preresize({s["image_path"] for s in samples}, root, args.cache, args.max_pixels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for s in samples:
            f.write(json.dumps(gym_row(s, "file://" + cached[s["image_path"]], args.family, args.max_output_tokens)) + "\n")
    print(f"wrote {len(samples)} rows to {args.out}: "
          f"{dict(Counter((s['platform'], s.get('data_type')) for s in samples))}")


if __name__ == "__main__":
    main()
