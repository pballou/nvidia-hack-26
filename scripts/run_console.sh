#!/usr/bin/env bash
# Start the GUI assist console (docs/tasks/T1-app.md, part B) on 127.0.0.1:8095.
# Usage: scripts/run_console.sh [--agent mock|real] [other `python -m console` flags]
#   (--headed exists for debugging only: unmeasured: headed fonts render different pixels (pay-50 fell from 8/8 to 1/4); record headless)
#   Defaults to --agent mock. The sandbox must be running (scripts/run_sandbox.sh).
#   PYTHON=.venv-gym/bin/python
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv-gym/bin/python}"
if [ ! -x "$PY" ]; then
  echo "error: no Python at $PY. Create the venv first (README.md, Quick start)." >&2
  exit 1
fi

has_agent=0
for a in "$@"; do
  case "$a" in --agent|--agent=*) has_agent=1 ;; esac
done
if [ "$has_agent" = 0 ]; then
  set -- --agent mock "$@"
fi

exec "$PY" -m console "$@"
