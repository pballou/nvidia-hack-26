#!/usr/bin/env bash
# Full train-split LoRA SFT for the template-vs-Lightning ablation (docs/tasks/B-night-loop.md).
# Both variants use the same config and seed; only the manifest differs.
#
#   training/sft_full.sh template  /home/ubuntu/data/os_atlas_20k/manifest.jsonl
#   training/sft_full.sh lightning /home/ubuntu/data/os_atlas_20k/train_lightning.jsonl
#
# Output: /home/ubuntu/ckpt/sft-<name>/{adapter,summary.json,train.log,run.json}
set -euo pipefail
name=${1:?name} manifest=${2:?manifest.jsonl}
source /home/ubuntu/serving.env
source /home/ubuntu/venvs/train/bin/activate
export HF_HUB_OFFLINE=1
cd "$(dirname "$0")/.."
out=/home/ubuntu/ckpt/sft-$name
mkdir -p "$out"
python3 - "$manifest" "$out/run.json" <<'EOF'
import hashlib, json, sys, time
from pathlib import Path
m = Path(sys.argv[1])
rev = Path("REVISION").read_text().strip() if Path("REVISION").exists() else None
Path(sys.argv[2]).write_text(json.dumps({
    "manifest": str(m), "manifest_sha256": hashlib.sha256(m.read_bytes()).hexdigest(),
    "code_revision": rev, "start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}, indent=2))
EOF
cat "$out/run.json"
python -m training.sft --model qwen3.5 --train "$manifest" --out "$out" --seed 0 --save-steps 200 \
  2>&1 | tee "$out/train.log"
echo "SFT_DONE $name"
