"""Re-key a PEFT LoRA adapter for vllm-metal's LoRA runtime (base + LoRA on the Mac, §3b).

  ~/.venv-merge/bin/python scripts/gate_g_lora_keys.py --adapter ~/models/gui-assist/<name> \
      --out ~/models/gui-assist-mlx-keys/<name>
  LORA=~/models/gui-assist-mlx-keys/<name> scripts/run_local_grounder.sh

vllm-metal 0.31 wraps the language model's Linear layers under mlx-vlm's names
(`model.layers.N.<module>`), while our adapters, trained with transformers, name them
`model.language_model.layers.N.<module>`. Its lookup only accepts an exact name or a unique
trailing-suffix match, so the original adapter fails with "matched 0 wrapped modules". On
2026-10-18 the 186 modules vllm-metal wrapped for Qwen3.5-2B were exactly our 186 adapter
modules after this prefix swap.

The output is the same adapter with renamed keys: identical tensors (checked value for value),
adapter_config.json and every other file copied unchanged, and RENAMED_FROM.json recording the
source, its sha256 and the rename. The H200 keeps serving the original adapter.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

OLD, NEW = "base_model.model.model.language_model.layers.", "base_model.model.model.layers."
WEIGHTS = "adapter_model.safetensors"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    src, dst = args.adapter.expanduser(), args.out.expanduser()
    if dst.exists():
        raise SystemExit(f"{dst} exists")

    f = safe_open(src / WEIGHTS, "pt")
    keys = list(f.keys())
    bad = [k for k in keys if not k.startswith(OLD)]
    assert not bad, f"keys without the expected prefix {OLD!r}: {bad[:3]}"
    renamed = {k: NEW + k[len(OLD):] for k in keys}
    assert len(set(renamed.values())) == len(keys), "rename collision"

    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    save_file({renamed[k]: f.get_tensor(k) for k in keys}, tmp / WEIGHTS, metadata={"format": "pt"})
    for p in src.iterdir():
        if p.is_file() and p.name != WEIGHTS:
            shutil.copyfile(p, tmp / p.name)
    with safe_open(tmp / WEIGHTS, "pt") as g:
        assert set(g.keys()) == set(renamed.values())
        for k in keys:
            assert torch.equal(g.get_tensor(renamed[k]), f.get_tensor(k)), k
    info = {"source": str(src), "source_sha256": sha256(src / WEIGHTS), "rename": {"from": OLD, "to": NEW},
            "why": "vllm-metal 0.31 names the wrapped LoRA modules model.layers.N.* (mlx-vlm language model)",
            "tensors": len(keys), "modules": len(keys) // 2, "sha256": sha256(tmp / WEIGHTS)}
    (tmp / "RENAMED_FROM.json").write_text(json.dumps(info, indent=2) + "\n")
    tmp.rename(dst)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
