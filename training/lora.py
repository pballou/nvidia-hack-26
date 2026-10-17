"""The one LoRA config we train and serve with (docs/plan.md "Fine-tuning method").

LoRA on the language model only; the vision tower stays frozen. The training scripts
and make_dummy_adapter.py both build their config here, so the gate D1 vLLM check
tests exactly what we train.
"""

from peft import LoraConfig

MODELS = {"qwen3.5": "Qwen/Qwen3.5-2B", "qwen3-vl": "Qwen/Qwen3-VL-2B-Instruct"}

R, ALPHA, DROPOUT = 16, 32, 0.05

_ATTN = r"self_attn\.(q|k|v|o)_proj"
_MLP = r"mlp\.(gate|up|down)_proj"
# Qwen3.5's Gated DeltaNet layers (18 of its 24). Drop these if vLLM rejects them
# (docs/gpu-box.md §3).
_LINEAR_ATTN = r"linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)"


def lora_config(model: str, linear_attn: bool = True) -> LoraConfig:
    parts = [_ATTN, _MLP]
    if model == "qwen3.5" and linear_attn:
        parts.append(_LINEAR_ATTN)
    return LoraConfig(
        r=R,
        lora_alpha=ALPHA,
        lora_dropout=DROPOUT,
        bias="none",
        # PEFT full-matches a string against module names. Anchoring on language_model
        # means nothing in model.visual can match.
        target_modules=rf"model\.language_model\.layers\.\d+\.({'|'.join(parts)})",
    )
