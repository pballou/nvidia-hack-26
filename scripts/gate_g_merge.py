"""Gate G step 1: merge base Qwen/Qwen3.5-2B + a PEFT LoRA adapter into one bf16 checkpoint.

vllm-metal's stable release can't load LoRA adapters for Qwen VLMs (docs/local-mac.md
"What's verified"), so the fully local grounder serves a merged checkpoint (§3b). Run it on
CPU in its own venv with the pinned training stack (docs/gpu-box.md: transformers 5.17.0,
peft 0.21.2, torch 2.13.0):

  uv venv ~/.venv-merge --python 3.12 --seed
  uv pip install --python ~/.venv-merge/bin/python "torch==2.13.0" "transformers==5.17.0" \
      "peft==0.21.2" "accelerate==1.15.0" safetensors "pillow==12.3.0"
  scp -r permanent-tan-marmot:/home/ubuntu/ckpt/sft-template/adapter ~/models/adapters/sft-template
  ~/.venv-merge/bin/python scripts/gate_g_merge.py --adapter ~/models/adapters/sft-template \
      --out ~/models/grounder-merged-sft-template

The output directory is the base snapshot with one change: each LoRA-targeted weight becomes
bf16(W + alpha/r * B @ A), merged in float32 by peft's own LoRA layer (one rounding to bf16 at
the end). Every other file (config, tokenizer, chat template, processor configs, safetensors
index) and every other tensor (vision tower, norms, embeddings, mtp.*) is copied bit-exact from
the base, with the same names, dtypes, shapes and order. That's what the H200 serves (base
weights and tokenizer, LoRA applied per request), so the only intended difference on the Mac is
merge rounding. mtp.* stays because the base has it; nothing loads it without speculative
decoding, which we never enable.

Memory: Lightning shares this Mac and the system sits a few GB above the memory-pressure
warning, so nothing here holds the whole model. The transformers model is built on the meta
device (structure only), the merge runs one module at a time, and the output file is streamed
tensor by tensor. Peak RSS stays well under 1 GB. The script aborts if pressure leaves
"normal".

Checks (any failure aborts before the output directory appears):
- transformers 5.17.0's Qwen3.5 model has exactly the base checkpoint's tensor names (minus
  mtp.* and the tied lm_head);
- the adapter's target_modules regex, applied by peft to that model, wraps exactly the modules
  the adapter file has weights for;
- peft's merged weight matches a manual W + scale * B @ A in float32 for every module;
- re-reading the written file: same names, dtypes and shapes as the base, and exactly the
  targeted tensors differ.
"""

import argparse
import copy
import hashlib
import json
import shutil
import struct
import subprocess
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

BASE_REPO = "Qwen/Qwen3.5-2B"
# Latest commit on the Hub (2026-03-02); the H200 downloaded it on 2026-10-17, so both match.
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
WEIGHTS = "model.safetensors-00001-of-00001.safetensors"
SKIP_FILES = {".gitattributes", WEIGHTS}
ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_pressure(step: str) -> None:
    """Abort if macOS memory pressure leaves 'normal' (1): Lightning shares this Mac."""
    level = int(subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                               capture_output=True, text=True, check=True).stdout)
    if level != 1:
        raise SystemExit(f"memory pressure level {level} at {step}; aborting the merge")


def write_safetensors(path: Path, names: list[str], meta: dict, get, metadata: dict) -> None:
    """Stream tensors into a safetensors file (same layout rules as the library: 8-byte
    little-endian header length, space-padded JSON header, contiguous data_offsets)."""
    header, off = {"__metadata__": metadata}, 0
    for k in names:
        dtype, shape = meta[k]
        n = ITEMSIZE[dtype] * (torch.Size(shape).numel())
        header[k] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for k in names:
            t = get(k).contiguous()
            dtype, shape = meta[k]
            assert list(t.shape) == shape and t.element_size() == ITEMSIZE[dtype], k
            f.write(t.reshape(-1).view(torch.uint8).numpy().tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True, help="PEFT adapter dir (adapter_config.json + adapter_model.safetensors)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (keep low: Lightning shares this Mac)")
    ap.add_argument("--force", action="store_true", help="replace an existing --out")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()

    import peft
    import transformers
    from peft import LoraConfig, inject_adapter_in_model
    from peft.tuners.lora import LoraLayer
    from transformers import AutoConfig, AutoModelForImageTextToText

    out = args.out.expanduser()
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists (pass --force to replace it)")
    adapter = args.adapter.expanduser()
    acfg = json.loads((adapter / "adapter_config.json").read_text())
    assert acfg["base_model_name_or_path"] == BASE_REPO, acfg["base_model_name_or_path"]
    assert acfg["peft_type"] == "LORA" and not acfg["use_dora"] and not acfg["use_rslora"], acfg
    assert not acfg.get("modules_to_save") and acfg["bias"] == "none" and not acfg["fan_in_fan_out"], acfg
    scale = acfg["lora_alpha"] / acfg["r"]
    lcfg = LoraConfig.from_pretrained(str(adapter))
    base_dir = Path(snapshot_download(BASE_REPO, revision=BASE_REVISION, local_files_only=True))
    check_pressure("start")

    # Base tensors are read on demand from the mmap'd file.
    base_f = safe_open(base_dir / WEIGHTS, "pt")
    names = list(base_f.keys())
    meta = {k: (base_f.get_slice(k).get_dtype(), list(base_f.get_slice(k).get_shape())) for k in names}
    lora_f = safe_open(adapter / "adapter_model.safetensors", "pt")
    # base_model.model.<module>.lora_A.weight -> <module>
    targets = sorted({k.removeprefix("base_model.model.").rsplit(".lora_", 1)[0] for k in lora_f.keys()})
    assert len(list(lora_f.keys())) == 2 * len(targets), "every module needs exactly one lora_A and one lora_B"
    print(f"base {BASE_REPO}@{BASE_REVISION[:12]}: {len(names)} tensors; adapter: {len(targets)} modules, scale {scale}")

    # Structure check on the meta device: no weights are allocated.
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(base_dir))
    model_keys = set(model.state_dict()) - {"lm_head.weight"}  # tied to embed_tokens
    base_keys = {k for k in names if not k.startswith("mtp.")}
    assert model.config.tie_word_embeddings or model.config.text_config.tie_word_embeddings
    assert model_keys == base_keys, f"name mismatch: model-only {sorted(model_keys - base_keys)[:5]}, base-only {sorted(base_keys - model_keys)[:5]}"
    inject_adapter_in_model(lcfg, model, low_cpu_mem_usage=True)
    wrapped = sorted(n for n, m in model.named_modules() if isinstance(m, LoraLayer))
    assert wrapped == targets, f"peft wraps {len(wrapped)} modules, adapter has weights for {len(targets)}"
    del model

    # Merge one module at a time with peft's own LoRA layer, in float32.
    one = copy.deepcopy(lcfg)
    one.target_modules, one.lora_dropout = {"proj"}, 0.0
    merged, max_peft_vs_manual = {}, 0.0
    for i, t in enumerate(targets):
        w = base_f.get_tensor(f"{t}.weight")
        assert w.dtype == torch.bfloat16 and w.dim() == 2, (t, w.dtype)
        a = lora_f.get_tensor(f"base_model.model.{t}.lora_A.weight")
        b = lora_f.get_tensor(f"base_model.model.{t}.lora_B.weight")
        holder = torch.nn.Module()
        holder.proj = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=torch.float32)
        holder.proj.weight.data.copy_(w.float())
        inject_adapter_in_model(one, holder)
        layer = holder.proj
        assert isinstance(layer, LoraLayer) and layer.scaling["default"] == scale, t
        layer.lora_A["default"].weight.data.copy_(a)
        layer.lora_B["default"].weight.data.copy_(b)
        layer.merge(safe_merge=True)  # raises on NaN/inf
        w_merged = layer.get_base_layer().weight.data
        manual = w.float() + scale * (b @ a)
        d = (w_merged - manual).abs().max().item()
        assert d <= 1e-5 * max(1.0, manual.abs().max().item()), f"{t}: peft merge differs from manual by {d}"
        max_peft_vs_manual = max(max_peft_vs_manual, d)
        merged[f"{t}.weight"] = w_merged.to(torch.bfloat16)
        if i % 40 == 0:
            check_pressure(f"merge {i}/{len(targets)}")

    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    write_safetensors(tmp / WEIGHTS, names, meta, lambda k: merged[k] if k in merged else base_f.get_tensor(k),
                      {"format": "pt"})
    del merged
    for p in base_dir.iterdir():
        if p.name not in SKIP_FILES:
            shutil.copyfile(p.resolve(), tmp / p.name)  # snapshot entries are symlinks into blobs/
    check_pressure("written")

    # Re-read what was written: same layout as the base, and exactly the targeted tensors differ.
    changed, delta_max = [], 0.0
    with safe_open(tmp / WEIGHTS, "pt") as f:
        assert set(f.keys()) == set(names), "written tensor names differ from the base"
        for k in names:
            assert (f.get_slice(k).get_dtype(), list(f.get_slice(k).get_shape())) == meta[k], k
            v, b = f.get_tensor(k), base_f.get_tensor(k)
            if not torch.equal(v, b):
                changed.append(k)
                delta_max = max(delta_max, (v.float() - b.float()).abs().max().item())
    assert sorted(changed) == sorted(f"{t}.weight" for t in targets), f"{len(changed)} tensors changed, expected {len(targets)}"

    info = {
        "base": f"{BASE_REPO}@{BASE_REVISION}",
        "base_weights_sha256": sha256(base_dir / WEIGHTS),
        "adapter": str(adapter),
        "adapter_model_sha256": sha256(adapter / "adapter_model.safetensors"),
        "adapter_config": {k: acfg[k] for k in ("r", "lora_alpha", "target_modules", "peft_version")},
        "merged_modules": len(targets),
        "scale": scale,
        "merge_dtype": "float32 -> bfloat16",
        "max_abs_peft_vs_manual_fp32": max_peft_vs_manual,
        "max_abs_change_vs_base_bf16": delta_max,
        "merged_weights_sha256": sha256(tmp / WEIGHTS),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "seconds": round(time.time() - t0, 1),
    }
    (tmp / "gate_g_merge.json").write_text(json.dumps(info, indent=2) + "\n")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    print(json.dumps(info, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
