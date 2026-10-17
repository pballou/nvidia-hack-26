#!/usr/bin/env bash
# One command for the demo: start the sandbox (:8090) and the GUI assist console (:8095),
# check that the VLM (:8001) and Lightning (hosted or :8000) respond, print the URL.
# Ctrl-C stops whatever this script started.
#
# Usage: scripts/demo_up.sh [--agent mock|real] [--lightning hosted|local] [--headed] [--skip-models]
#   --agent mock      scripted stand-in agent, no models needed (default)
#   --agent real      T2's agent (agent/core.py): Lightning plans, the VLM grounds
#   --lightning       which Lightning the console defaults to and this script checks (default local:
#                     the official BF16 checkpoint served as `lightning` on :8000; hosted is slow)
#   --headed          show the Playwright browser window (for the video)
#   --skip-models     don't check :8001 / Lightning
# Logs: runs/demo_logs/{sandbox,console}.log
set -euo pipefail
cd "$(dirname "$0")/.."

AGENT=mock LIGHTNING=local HEADED=0 SKIP_MODELS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --agent) AGENT="${2:?--agent needs mock|real}"; shift 2 ;;
    --agent=*) AGENT="${1#*=}"; shift ;;
    --lightning) LIGHTNING="${2:?--lightning needs hosted|local}"; shift 2 ;;
    --lightning=*) LIGHTNING="${1#*=}"; shift ;;
    --headed) HEADED=1; shift ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
case "$AGENT" in mock|real) ;; *) echo "--agent must be mock or real" >&2; exit 2 ;; esac
case "$LIGHTNING" in hosted|local) ;; *) echo "--lightning must be hosted or local" >&2; exit 2 ;; esac

PY="${PYTHON:-.venv-gym/bin/python}"
if [ ! -x "$PY" ]; then
  echo "error: no Python at $PY. Create the venv first (README.md, Quick start)." >&2
  exit 1
fi

SANDBOX_URL=http://127.0.0.1:8090
CONSOLE_URL=http://127.0.0.1:8095
LOGS=runs/demo_logs
mkdir -p "$LOGS"
PIDS=""
cleanup() {
  for pid in $PIDS; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT TERM

up() { curl -sf -m 2 -o /dev/null "$1"; }

wait_up() {  # url name pid
  for _ in $(seq 1 60); do
    up "$1" && return 0
    kill -0 "$3" 2>/dev/null || { echo "error: $2 exited; see $LOGS/$2.log" >&2; tail -20 "$LOGS/$2.log" >&2; exit 1; }
    sleep 0.5
  done
  echo "error: $2 did not come up in 30 s; see $LOGS/$2.log" >&2
  exit 1
}

# 1. Sandbox
if up "$SANDBOX_URL/api/health"; then
  echo "sandbox  already running at $SANDBOX_URL"
else
  scripts/run_sandbox.sh >"$LOGS/sandbox.log" 2>&1 &
  PIDS="$PIDS $!"
  wait_up "$SANDBOX_URL/api/health" sandbox "$!"
  echo "sandbox  started at $SANDBOX_URL"
fi

# 2. Models (warnings only: the mock agent needs none of them)
model_ids() { "$PY" -c 'import json,sys; print(" ".join(m["id"] for m in json.load(sys.stdin).get("data", [])))'; }
if [ "$SKIP_MODELS" = 0 ]; then
  if ids=$(curl -sf -m 5 http://127.0.0.1:8001/v1/models | model_ids); then
    echo "vlm      :8001 up, served: ${ids:-none}"
    case " $ids " in *" base "*) ;; *) echo "warning: served name 'base' missing on :8001 (contracts.md §3)" >&2 ;; esac
    case " $ids " in *" grounder "*) ;; *) echo "note:    no 'grounder' adapter yet; use Grounder = base in the console" ;; esac
  else
    echo "warning: VLM :8001 not reachable. Open the tunnel (docs/local-mac.md §3a) or start it locally (§3b)." >&2
  fi
  if [ "$LIGHTNING" = local ]; then
    if ids=$(curl -sf -m 5 http://127.0.0.1:8000/v1/models | model_ids); then
      echo "lightning local :8000 up, served: ${ids:-none}"
      case " $ids " in *" lightning "*) ;; *) echo "warning: served name 'lightning' missing on :8000" >&2 ;; esac
    else
      echo "warning: local Lightning :8000 not reachable (docs/local-mac.md §2)." >&2
    fi
  else
    if [ -z "${NVIDIA_API_KEY:-}" ] && [ -f .env ]; then
      set -a; . ./.env; set +a
    fi
    HOSTED=https://integrate.api.nvidia.com/v1
    if [ -z "${NVIDIA_API_KEY:-}" ]; then
      echo "warning: NVIDIA_API_KEY is not set (.env) for hosted Lightning." >&2
    elif ! curl -sf -m 20 "$HOSTED/models" | model_ids | grep -q 'nemotron-3.5-lightning'; then
      echo "warning: hosted Lightning not listed at $HOSTED/models (network?)" >&2
    elif [ "$AGENT" = real ]; then
      # A bad key is rejected right away; a good one queues, so a timeout counts as OK.
      # The key goes to curl on stdin, so it never shows up in the process list.
      code=$(printf 'Authorization: Bearer %s\nContent-Type: application/json\n' "$NVIDIA_API_KEY" |
        curl -s -o /dev/null -w '%{http_code}' -m 8 -H @- "$HOSTED/chat/completions" \
          -d '{"model":"nvidia/nemotron-3.5-lightning-30b-a3b","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}' || true)
      case "$code" in
        401|403) echo "warning: hosted Lightning rejected NVIDIA_API_KEY (HTTP $code)" >&2 ;;
        *) echo "lightning hosted reachable, key accepted; expect 25-80 s per step" ;;
      esac
    else
      echo "lightning hosted reachable (key not checked in mock mode)"
    fi
  fi
fi

# 3. Console
if up "$CONSOLE_URL/api/config"; then
  echo "console  already running at $CONSOLE_URL (restart it to change --agent/--headed)"
else
  set -- --agent "$AGENT" --lightning "$LIGHTNING" --sandbox-url "$SANDBOX_URL"
  [ "$HEADED" = 1 ] && set -- "$@" --headed
  scripts/run_console.sh "$@" >"$LOGS/console.log" 2>&1 &
  PIDS="$PIDS $!"
  wait_up "$CONSOLE_URL/api/config" console "$!"
  echo "console  started (agent: $AGENT, lightning: $LIGHTNING$([ "$HEADED" = 1 ] && echo ', headed'))"
fi

echo
echo "  Open $CONSOLE_URL   (sandbox: $SANDBOX_URL)"
echo
if [ -n "$PIDS" ]; then
  echo "Ctrl-C stops the servers started here. Logs: $LOGS/"
  wait
fi
