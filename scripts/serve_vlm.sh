#!/usr/bin/env bash
# Serve a grounder candidate with vLLM on the Brev box (docs/gpu-box.md §3).
#
#   scripts/serve_vlm.sh qwen3.5|qwen3-vl [port] [extra vLLM args...]
#
# - Served names are `base` plus the HF repo id. Port 8001 (container `vllm-vlm`) is the
#   team contract; other ports get `vllm-vlm-<port>` (e.g. a second candidate for M1).
# - After SFT, add the adapter, e.g.:
#     scripts/serve_vlm.sh qwen3.5 8001 --enable-lora --max-lora-rank 64 \
#       --lora-modules grounder=/home/ubuntu/ckpt/<run>/adapter
# - GPU_UTIL overrides --gpu-memory-utilization (default 0.30; 0.45 for --num-repeats 8).
# - Weights come from scripts/download.py. HF_HUB_OFFLINE=1 keeps this root container
#   from writing into the shared HF cache that the training venv also uses.
set -euo pipefail
source /home/ubuntu/serving.env

case "${1:?usage: serve_vlm.sh qwen3.5|qwen3-vl [port] [vLLM args...]}" in
  qwen3.5)
    model=Qwen/Qwen3.5-2B
    # No --reasoning-parser for Qwen3.5: answers would land in `reasoning` (vLLM #53284).
    model_args=(--default-chat-template-kwargs '{"enable_thinking": false}')
    ;;
  qwen3-vl)
    model=Qwen/Qwen3-VL-2B-Instruct
    model_args=()
    ;;
  *) echo "unknown model: $1" >&2; exit 2 ;;
esac
port=${2:-8001}
shift $(( $# < 2 ? $# : 2 ))
name=vllm-vlm
[[ $port == 8001 ]] || name=vllm-vlm-$port

docker run -d --name "$name" --gpus all --ipc=host -p "127.0.0.1:$port:$port" \
  -e HF_HUB_OFFLINE=1 \
  -v /home/ubuntu/hf:/root/.cache/huggingface -v /home/ubuntu/vllm-cache:/root/.cache/vllm \
  -v /home/ubuntu/data:/home/ubuntu/data:ro -v /home/ubuntu/ckpt:/home/ubuntu/ckpt:ro \
  "$VLLM_IMAGE" \
  --model "$model" --served-model-name base "$model" \
  --port "$port" --gpu-memory-utilization "${GPU_UTIL:-0.30}" \
  --max-model-len 16384 --max-num-seqs 128 --max-num-batched-tokens 16384 \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --mm-processor-kwargs "{\"size\": {\"longest_edge\": ${GROUNDER_MAX_PIXELS:?}, \"shortest_edge\": ${GROUNDER_MIN_PIXELS:?}}}" \
  --allowed-local-media-path /home/ubuntu/data \
  "${model_args[@]}" "$@"

until curl -sf "localhost:$port/health" >/dev/null; do
  docker ps -q -f "name=^$name\$" | grep -q . || { docker logs --tail 80 "$name"; exit 1; }
  sleep 5
done
echo "$name ($model) ready on :$port"
