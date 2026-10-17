#!/usr/bin/env bash
# Overnight Gym improvement loop (docs/tasks/B-night-loop.md). Run it inside tmux:
#
#   tmux new-session -d -s night -n loop
#   tmux send-keys -t night:loop "training/night_loop.sh --init-adapter /home/ubuntu/ckpt/sft-<winner>/adapter \
#       --train-file /home/ubuntu/data/os_atlas_20k/<winner's train file>.jsonl" Enter
#
# Everything else (knobs, STOP, round_<r>/, FINAL.json) lives in /home/ubuntu/night; see
# training/night_loop.py. Extra args are passed through (e.g. --night-dir for a rehearsal).
# The loop appends its own log to <night dir>/night_loop.log (no tee: a closed pane must not
# break the cleanup path); follow it with `tail -f`.
set -uo pipefail
source /home/ubuntu/serving.env
source /home/ubuntu/venvs/train/bin/activate
export HF_HUB_OFFLINE=1
cd "$(dirname "$0")/.."
python -m training.night_loop "$@"
echo "NIGHT_LOOP_EXIT $?"
