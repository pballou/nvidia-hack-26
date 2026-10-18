#!/usr/bin/env bash
# Serve the merged grounder on this Mac via vllm-metal (docs/local-mac.md §3b, Gate G).
#
#   MODEL=~/models/grounder-merged-<name> scripts/run_local_grounder.sh   # merged, port 8011
#   LORA=~/models/gui-assist/<name> scripts/run_local_grounder.sh         # base + LoRA, port 8011
# (8001 is the H200 tunnel.)
#
# Runs in the foreground (use tmux) and exec's vllm, so the PID in $PIDFILE is the server's.
# scripts/gate_g_watch.py uses that file to stop this server (and only this server) if memory
# pressure leaves normal or Lightning on :8000 slows down. Never touches :8000.
#
# Two modes, one model instance either way (one 2B under the parameter cap):
#   merged (default): MODEL = a merged checkpoint (scripts/gate_g_merge.py), served as `grounder`.
#   LoRA:   LORA=<PEFT adapter dir> serves the base snapshot as `base` plus the adapter as
#           `grounder` (adapter switched per request), exactly like the H200 (gpu-box.md §3).
# Env: MODEL, LORA, PORT, GPU_UTIL (<= 0.12: Lightning holds 0.68 of Metal memory), VENV (the
# vllm-metal venv), LOGDIR, PIDFILE.
# No extra args: vLLM's argparse is last-wins, so they could override the pinned flags or turn
# on speculative decoding (--spec-*, --config), which AGENTS.md forbids.
#
# Requires a live watchdog (scripts/gate_g_watch.py writes $PIDFILE.watch once armed); set
# NO_WATCHDOG=1 only if you are watching memory pressure and Lightning yourself.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
LORA=${LORA:-}
BASE_SNAPSHOT=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
if [ -n "$LORA" ]; then
  MODEL=${MODEL:-$BASE_SNAPSHOT}
  SERVED=base
else
  MODEL=${MODEL:-$HOME/models/grounder-merged-sft-template}
  SERVED=grounder
fi
# SERVED_NAME=base with MODEL=<base snapshot> and no LORA serves plain base (for A/B checks).
SERVED=${SERVED_NAME:-$SERVED}
case "$SERVED" in base|grounder) ;; *) echo "run_local_grounder: SERVED_NAME must be base or grounder" >&2; exit 1 ;; esac
PORT=${PORT:-8011}
GPU_UTIL=${GPU_UTIL:-0.08}
# vllm-metal's startup profile runs a dummy forward of max_num_batched_tokens tokens and keeps
# its logits (8192 x 248k vocab, ~4 GB). 2560 still fits a 2 MP image (2,025 tokens) plus the
# prompt in one prefill chunk, as on the H200.
MAX_BATCHED=${MAX_BATCHED:-2560}
MIN_FREE_PCT=${MIN_FREE_PCT:-26}
VENV=${VENV:-$HOME/.venv-vllm-metal}
LOGDIR=${LOGDIR:-$HOME/models/logs}
PIDFILE=${PIDFILE:-$LOGDIR/local-grounder-$PORT.pid}

die() { echo "run_local_grounder: $*" >&2; exit 1; }
[ "$PORT" != 8000 ] || die "port 8000 is Lightning's"
awk -v u="$GPU_UTIL" 'BEGIN { exit !(u > 0 && u <= 0.12) }' || die "GPU_UTIL must be in (0, 0.12], got $GPU_UTIL"
[ $# -eq 0 ] || die "extra args aren't passed through (pinned flags only; edit this script instead)"

# The image cap MUST equal common/grounding.py (frozen at gate D1) and the H200's serving.env.
GROUNDER_MAX_PIXELS=$(sed -nE 's/^MAX_PIXELS: int = ([0-9_]+).*/\1/p' "$REPO/common/grounding.py" | tr -d _)
GROUNDER_MIN_PIXELS=$(sed -nE 's/^MIN_PIXELS: int = ([0-9_]+).*/\1/p' "$REPO/common/grounding.py" | tr -d _)
[ "$GROUNDER_MAX_PIXELS" = 2073600 ] && [ "$GROUNDER_MIN_PIXELS" = 65536 ] \
  || die "pixel cap in common/grounding.py is $GROUNDER_MAX_PIXELS/$GROUNDER_MIN_PIXELS, expected the frozen 2073600/65536"

[ -f "$MODEL/config.json" ] || die "no checkpoint at $MODEL (run scripts/gate_g_merge.py)"
LORA_ARGS=()
if [ -n "$LORA" ]; then
  [ -f "$LORA/adapter_config.json" ] && [ -f "$LORA/adapter_model.safetensors" ] \
    || die "LORA=$LORA has no adapter_config.json + adapter_model.safetensors"
  # Rank cap as on the H200 (it must be >= the trained rank, 16 tonight).
  LORA_ARGS=(--enable-lora --max-lora-rank 64 --lora-modules "grounder=$LORA")
fi
! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || die "port $PORT is already in use"
level=$(sysctl -n kern.memorystatus_vm_pressure_level)
[ "$level" = 1 ] || die "memory pressure level is $level (1 = normal); not starting"
# Headroom check with Lightning's memory wired: MLX unwires Lightning's ~80 GB a few seconds
# after its last request, which makes "free" read ~80% while the real headroom is ~23%. A tiny
# request re-wires it first. The grounder takes ~GPU_UTIL x 115 GB + ~1 GB of process memory,
# and macOS turned yellow near 15-16% free on 2026-10-17.
if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  curl -s -m 60 -o /dev/null http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"lightning","temperature":0,"max_tokens":4,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"Reply with the single word OK."}]}' \
    || die "Lightning on :8000 didn't answer the warm-up request"
fi
free_pct=$(memory_pressure -Q | sed -nE 's/.*free percentage: ([0-9]+)%.*/\1/p')
[ "${free_pct:-0}" -ge "$MIN_FREE_PCT" ] || die "only ${free_pct}% free with Lightning wired (need $MIN_FREE_PCT%); free memory first"

if [ "${NO_WATCHDOG:-0}" != 1 ]; then
  wpid=$(cat "$PIDFILE.watch" 2>/dev/null || true)
  [ -n "$wpid" ] && ps -ww -o command= -p "$wpid" 2>/dev/null | grep -q gate_g_watch.py \
    || die "no armed watchdog ($PIDFILE.watch); start scripts/gate_g_watch.py first"
fi

mkdir -p "$LOGDIR"
echo $$ > "$PIDFILE.tmp" && mv "$PIDFILE.tmp" "$PIDFILE"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1
# Lightning's engine is also titled VLLM::EngineCore; this one becomes VLLM_GROUNDER::EngineCore.
# Still never stop anything by name: stop the PID in $PIDFILE and its children.
export VLLM_PROCESS_NAME_PREFIX=VLLM_GROUNDER
echo "run_local_grounder: pid $$, $MODEL as $SERVED${LORA:+ + LoRA grounder=$LORA} on 127.0.0.1:$PORT, gpu util $GPU_UTIL, cap $GROUNDER_MAX_PIXELS/$GROUNDER_MIN_PIXELS"
# No --reasoning-parser for Qwen3.5 (it would move the answer into `reasoning`; gpu-box.md §3).
# vllm-metal allocates GPU_UTIL x Metal's working set (115.45 GB) up front for this hybrid model,
# and the multimodal processor cache (default 4 GiB) sits in both the API and engine processes:
# turn it off, since every screenshot is new anyway.
# ${arr[@]+...}: an empty array is an error under set -u in macOS's bash 3.2.
exec vllm serve "$MODEL" --served-model-name "$SERVED" ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} \
  --host 127.0.0.1 --port "$PORT" --max-model-len 16384 --max-num-seqs 1 --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-batched-tokens "$MAX_BATCHED" --mm-processor-cache-gb 0 --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --mm-processor-kwargs "{\"size\": {\"longest_edge\": ${GROUNDER_MAX_PIXELS}, \"shortest_edge\": ${GROUNDER_MIN_PIXELS}}}" \
  --default-chat-template-kwargs '{"enable_thinking": false}' --no-enable-prefix-caching
