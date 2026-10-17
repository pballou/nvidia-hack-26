"""Zero-shot click-grounding smoke test of a base Qwen VL model (Qwen3-VL, Qwen3.5) on a random manifest sample.

Example:
  python scripts/base_grounding_smoke.py --data /home/ubuntu/data/os_atlas_20k --n 10 --seed 0
  python scripts/base_grounding_smoke.py --data <dir> --n 500 --model Qwen/Qwen3.5-2B --dtype bfloat16
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPT = (
    'This is a screenshot. Where should I click to accomplish this instruction: "{instruction}"?\n'
    'Output the click point in JSON format as {{"point_2d": [x, y]}}.'
)
POINT_RE = re.compile(r"point_2d\"?\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")
NUMS_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")


def parse_point(text):
    """Qwen VL models emit points on a 0-1000 relative grid; return (x, y) normalized to [0, 1]."""
    m = POINT_RE.search(text) or NUMS_RE.search(text)
    if not m:
        return None
    return float(m.group(1)) / 1000, float(m.group(2)) / 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.data / "manifest.jsonl")]
    sample = random.Random(args.seed).sample(rows, args.n)

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, revision=args.revision, dtype=getattr(torch, args.dtype)
    ).to("cuda").eval()

    out_path = args.out or args.data / f"base_smoke_seed{args.seed}_n{args.n}.jsonl"
    hits = 0
    with open(out_path, "w") as sink:
        for i, r in enumerate(sample):
            image = Image.open(args.data / r["image_path"]).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT.format(instruction=r["instruction"])},
            ]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            ).to(model.device)
            t = time.time()
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            text = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

            w, h = r["img_w"], r["img_h"]
            x1, y1, x2, y2 = r["bbox"]
            box = [x1 / w, y1 / h, x2 / w, y2 / h]
            pt = parse_point(text)
            hit = pt is not None and box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]
            hits += hit
            rec = {
                "id": r["id"], "platform": r["platform"], "instruction": r["instruction"],
                "bbox_norm": [round(v, 4) for v in box],
                "pred_norm": [round(v, 4) for v in pt] if pt else None,
                "pred_px": [round(pt[0] * w), round(pt[1] * h)] if pt else None,
                "hit": hit, "raw_output": text,
                "input_tokens": int(inputs["input_ids"].shape[1]), "latency_s": round(time.time() - t, 2),
            }
            sink.write(json.dumps(rec) + "\n")
            print(f"[{i + 1}/{args.n}] {'HIT ' if hit else 'MISS'} {r['platform']:7s} {r['instruction'][:50]!r} "
                  f"pred={rec['pred_norm']} box={rec['bbox_norm']} raw={text!r}", flush=True)

    print(f"\naccuracy: {hits}/{args.n} = {hits / args.n:.0%}  (results: {out_path})")


if __name__ == "__main__":
    main()
