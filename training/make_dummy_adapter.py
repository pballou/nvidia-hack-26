"""Gate D1 check: save a LoRA adapter built from our real config, for vLLM to load.

    python -m training.make_dummy_adapter qwen3.5 /home/ubuntu/ckpt/dummy-qwen3.5

The adapter goes to <out>/adapter. Its weights are random (init_lora_weights=False),
so `grounder` must answer differently from `base`. That proves vLLM applies the
adapter instead of silently ignoring it.
"""

import argparse

import torch
from peft import get_peft_model
from transformers import AutoModelForImageTextToText

from training.lora import MODELS, lora_config

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out")
parser.add_argument("--no-linear-attn", action="store_true", help="Qwen3.5 fallback targets")
args = parser.parse_args()

model = AutoModelForImageTextToText.from_pretrained(MODELS[args.model], dtype=torch.bfloat16)
config = lora_config(args.model, linear_attn=not args.no_linear_attn)
config.init_lora_weights = False
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()

targeted = [n for n, m in peft_model.named_modules() if hasattr(m, "lora_A")]
assert targeted and not any(".visual." in n for n in targeted), "LoRA must skip the vision tower"
print(f"{len(targeted)} LoRA modules, e.g. {targeted[0]}")
peft_model.save_pretrained(f"{args.out}/adapter")
print(f"saved {args.out}/adapter")
