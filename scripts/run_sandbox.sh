#!/usr/bin/env bash
# Start the sandbox target app ("Acme Utilities", docs/tasks/T1-app.md) on 127.0.0.1:8090.
# Usage: scripts/run_sandbox.sh [extra uvicorn args, e.g. --reload]
#   SANDBOX_PORT=8090  PYTHON=.venv-gym/bin/python
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv-gym/bin/python}"
if [ ! -x "$PY" ]; then
  echo "error: no Python at $PY. Create the venv first (README.md, Quick start)." >&2
  exit 1
fi

exec "$PY" -m uvicorn sandbox.app:app --host 127.0.0.1 --port "${SANDBOX_PORT:-8090}" "$@"
