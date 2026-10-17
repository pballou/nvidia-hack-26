#!/usr/bin/env bash
# Overnight Gym improvement loop (docs/tasks/B-night-loop.md). Run it inside tmux:
#
#   tmux new-session -d -s night -n loop
#   tmux send-keys -t night:loop "training/night_loop.sh --init-adapter /home/ubuntu/ckpt/sft-<winner>/adapter \
#       --train-file /home/ubuntu/data/os_atlas_20k/<winner's train file>.jsonl" Enter
#
# Everything else (knobs, STOP, round_<r>/, FINAL.json) lives in /home/ubuntu/night; see
# training/night_loop.py. Extra args are passed through (e.g. --night-dir for a rehearsal).
set -uo pipefail
source /home/ubuntu/serving.env
source /home/ubuntu/venvs/train/bin/activate
export HF_HUB_OFFLINE=1
cd "$(dirname "$0")/.."
night=/home/ubuntu/night
for ((i = 1; i <= $#; i++)); do [[ ${!i} == --night-dir ]] && { j=$((i + 1)); night=${!j}; }; done
mkdir -p "$night"
python -m training.night_loop "$@" 2>&1 | tee -a "$night/night_loop.log"
echo "NIGHT_LOOP_EXIT ${PIPESTATUS[0]}" | tee -a "$night/night_loop.log"
