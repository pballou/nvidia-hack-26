#!/usr/bin/env bash
# Start the local speech-to-text sidecar for the console's voice input: NVIDIA Parakeet
# TDT 0.6B v3 (official GGUF) on NeMo-Speech.cpp 0.2.0, at 127.0.0.1:8096.
# Install it first (README.md, Quick start, "Voice input"). Don't start a second copy if
# one is already serving (check: curl -s 127.0.0.1:8096/health).
# Usage: scripts/run_asr.sh     ASR_PORT=8096
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"   # where install.sh --no-modify-path puts the binary
if ! command -v nemo-speech >/dev/null 2>&1; then
  echo "error: nemo-speech not found. Install it (README.md, Quick start, Voice input)." >&2
  exit 1
fi

exec nemo-speech serve --asr-model parakeet-tdt --host 127.0.0.1 --port "${ASR_PORT:-8096}" \
  --no-ui --asr.batching.enabled=false
