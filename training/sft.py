"""LoRA SFT for the grounder with TRL (docs/plan.md M2, docs/gpu-box.md §5).

    python -m training.sft --model qwen3.5 --train /home/ubuntu/data/<slice>/manifest.jsonl \
        --out /home/ubuntu/ckpt/<run> [--max-steps 20] [--max-pixels 1003520]

- Rows are canonical grounding samples (docs/plan.md "Shared contracts"). `image_path`
  is relative to the manifest's directory. Rows whose split is "dev" or "test" are
  skipped, and ScreenSpot rows are refused outright.
- One target per example (no packing). The prompt comes from common/grounding.py; the
  target is the bbox center as {"point_2d": [x, y]} on the 0-1000 grid, and
  parse_point must read it back unchanged.
- Screenshots are pre-resized with preprocess_screenshot into <cache>/<max_pixels>/,
  and the processor is capped at the same MAX_PIXELS, so nothing is resized twice.
- Before training, the rendered prompt is checked against the serving-side generation
  prompt (the same chat-template call vLLM makes), so train and serve can't drift.
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import torch
from datasets import Dataset, Image, List
from PIL import Image as PILImage
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback
from trl import SFTConfig, SFTTrainer
from trl.data_utils import apply_chat_template, prepare_multimodal_messages

from common import grounding
from training.lora import MODELS, lora_config

FAMILY = {"qwen3.5": "qwen3_5", "qwen3-vl": "qwen3_vl"}
# Qwen3.5 is served with thinking off (docs/gpu-box.md §3), so train the same way.
TEMPLATE_KWARGS = {"qwen3.5": {"enable_thinking": False}, "qwen3-vl": {}}


def load_rows(manifest: Path, limit: int | None, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in manifest.open()]
    if bad := [r["id"] for r in rows if "screenspot" in r["source"].lower()]:
        sys.exit(f"refusing to train on ScreenSpot rows: {bad[:5]}")
    rows = [r for r in rows if r.get("split") not in ("dev", "test")]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def target_text(row: dict) -> str:
    x1, y1, x2, y2 = row["bbox"]
    x, y = grounding.to_grid(((x1 + x2) / 2, (y1 + y2) / 2), row["img_w"], row["img_h"])
    pt = [min(1000, max(0, round(x))), min(1000, max(0, round(y)))]
    text = json.dumps({"point_2d": pt})
    assert grounding.parse_point(text) == tuple(pt), text
    return text


def _resize_one(job: tuple[Path, Path, int]) -> None:
    src, dst, max_pixels = job
    if dst.exists():
        return
    grounding.MAX_PIXELS = max_pixels  # set per worker; preprocess reads it at call time
    dst.parent.mkdir(parents=True, exist_ok=True)
    with PILImage.open(src) as im:
        grounding.preprocess_screenshot(im).save(dst)


def preresize(rows: list[dict], root: Path, cache: Path, max_pixels: int) -> dict[str, str]:
    """Resize each unique screenshot once; returns image_path -> cached PNG path."""
    out = {p: cache / str(max_pixels) / root.name / Path(p).with_suffix(".png") for p in {r["image_path"] for r in rows}}
    with Pool(16) as pool:
        pool.map(_resize_one, [(root / p, dst, max_pixels) for p, dst in out.items()], chunksize=8)
    return {p: str(dst) for p, dst in out.items()}


def build_dataset(rows: list[dict], cached: dict[str, str], model: str) -> Dataset:
    examples = []
    for r in rows:
        ex = {
            "images": [cached[r["image_path"]]],
            "prompt": [{"role": "user", "content": grounding.grounding_prompt(r["instruction"], FAMILY[model])}],
            "completion": [{"role": "assistant", "content": target_text(r)}],
        }
        if TEMPLATE_KWARGS[model]:  # an empty struct column breaks Arrow, so omit it
            ex["chat_template_kwargs"] = TEMPLATE_KWARGS[model]
        examples.append(ex)
    return Dataset.from_list(examples).cast_column("images", List(Image()))


def check_format(processor, example: dict, model: str) -> None:
    """Training prompt == serving generation prompt; the image isn't resized again."""
    kwargs = TEMPLATE_KWARGS[model]
    prompt = prepare_multimodal_messages(example["prompt"], images=example["images"])
    rendered = apply_chat_template(
        {"prompt": prompt, "completion": prepare_multimodal_messages(example["completion"]), "chat_template_kwargs": kwargs},
        processor,
    )
    serve_prompt = processor.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False, **kwargs)
    print(f"prompt tail: {rendered['prompt'][-80:]!r}\ncompletion: {rendered['completion']!r}", flush=True)
    assert rendered["prompt"] == serve_prompt, f"train/serve prompt mismatch:\n{serve_prompt[-200:]!r}"
    img = example["images"][0]
    _, gh, gw = processor(images=[img], text=[serve_prompt], return_tensors="pt")["image_grid_thw"][0].tolist()
    patch = processor.image_processor.patch_size
    assert (gw * patch, gh * patch) == img.size, f"processor resized {img.size} to {(gw * patch, gh * patch)}"


class StepTimer(TrainerCallback):
    def __init__(self):
        self.times, self._t = [], None

    def on_step_begin(self, args, state, control, **kwargs):
        self._t = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        self.times.append(time.perf_counter() - self._t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--train", type=Path, required=True, help="canonical manifest JSONL")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-pixels", type=int, default=grounding.MAX_PIXELS)
    ap.add_argument("--cache", type=Path, default=Path("/home/ubuntu/data/cache/preprocessed"))
    ap.add_argument("--limit", type=int, help="use only the first N rows (after shuffling)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-linear-attn", action="store_true", help="Qwen3.5 fallback LoRA targets")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    rows = load_rows(args.train, args.limit, args.seed)
    t0 = time.perf_counter()
    cached = preresize(rows, args.train.parent, args.cache, args.max_pixels)
    print(f"{len(rows)} rows, {len(cached)} screenshots pre-resized in {time.perf_counter() - t0:.0f}s", flush=True)
    ds = build_dataset(rows, cached, args.model)

    model_id = MODELS[args.model]
    processor = AutoProcessor.from_pretrained(model_id)
    processor.image_processor.size.longest_edge = args.max_pixels
    processor.image_processor.size.shortest_edge = grounding.MIN_PIXELS
    check_format(processor, ds[0], args.model)

    per_step = args.batch_size * args.grad_accum
    total = args.max_steps if args.max_steps > 0 else math.ceil(args.epochs * len(ds) / per_step)
    config = SFTConfig(
        output_dir=str(args.out),
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(0.03 * total)),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=None,  # never truncate image tokens
        remove_unused_columns=False,  # keep chat_template_kwargs for the collator
        logging_steps=1,
        save_steps=args.save_steps,
        report_to="none",
        dataloader_num_workers=4,
        seed=args.seed,
    )
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16)
    timer = StepTimer()
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=ds,
        processing_class=processor,
        peft_config=lora_config(args.model, linear_attn=not args.no_linear_attn),
        callbacks=[timer],
    )
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    trainer.save_model(str(args.out / "adapter"))

    steady = timer.times[3:] or timer.times
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    summary = {
        "model": model_id,
        "max_pixels": args.max_pixels,
        "rows": len(ds),
        "steps": len(timer.times),
        "examples_per_step": per_step,
        "median_s_per_step": round(statistics.median(steady), 2),
        "examples_per_s": round(per_step / statistics.median(steady), 2),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "train_runtime_s": round(result.metrics["train_runtime"], 1),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
