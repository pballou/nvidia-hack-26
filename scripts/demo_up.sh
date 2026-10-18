#!/usr/bin/env bash
# One command for the demo: start the sandbox (:8090) and the GUI assist console (:8095),
# check that the VLM (configs/agent.yaml vlm.base_url, normally :8001, or --vlm-url) and Lightning
# (hosted or :8000) respond and serve the names the console will ask for, print the URL.
# Ctrl-C stops whatever this script started.
#
# Usage: scripts/demo_up.sh [--agent mock|real] [--lightning hosted|local] [--grounder base|grounder]
#                           [--thinking off|retry|on] [--clarify on|off]
#                           [--vlm-url URL] [--summary-model base|grounder] [--vlm-label TEXT]
#                           [--asr auto|parakeet|stub] [--headed] [--skip-models]
#   --agent mock      scripted stand-in agent, no models needed (default)
#   --agent real      T2's agent (agent/core.py): Lightning plans, the VLM grounds
#   --lightning       which Lightning the console defaults to and this script checks (default local:
#                     the official BF16 checkpoint served as `lightning` on :8000; hosted is slow)
#   --grounder        the console's initial Grounder choice (default grounder = the LoRA; base = adapter off)
#   --thinking        the console's initial Thinking choice (default off, the recording setting; retry, on)
#   --clarify         the console's initial Request check (default on: one short Lightning call before the
#                     browser opens; an unclear request gets a question, the answer starts a follow-up run)
#   --vlm-url URL     the VLM's OpenAI base URL for the real agent and this script's check, e.g.
#                     http://127.0.0.1:8011/v1 for the fully local Mac grounder (default: configs/agent.yaml
#                     vlm.base_url, or $AGENT_VLM_BASE_URL)
#   --summary-model   the served name that writes the screen summaries (default: configs/agent.yaml
#                     summary_model, or $AGENT_SUMMARY_MODEL). grounder when the VLM serves only the grounder
#   --vlm-label TEXT  how the console names the VLM, e.g. "Mac (vllm-metal)" (default: by port, 8001 = H200
#                     via tunnel, 8011 = Mac (vllm-metal); or $CONSOLE_VLM_LABEL)
#   --asr             voice input engine (default auto = parakeet: the local sidecar at $CONSOLE_ASR_URL,
#                     default 127.0.0.1:8096, scripts/run_asr.sh; voice stays off until it answers, and the
#                     page turns it on by itself). stub = fixed text for UI testing, never auto-run, only
#                     when asked for. $CONSOLE_ASR overrides the default.
#   --headed          show the Playwright browser window. Debugging only,
#                     unmeasured: headed fonts render different pixels (pay-50 fell from 8/8 to 1/4); record headless
#   --skip-models     don't check the VLM / Lightning
# Env: SANDBOX_PORT (8090), CONSOLE_PORT (8095), LIGHTNING_LOCAL_URL (http://127.0.0.1:8000/v1, what
#      --lightning local checks), PYTHON (.venv-gym/bin/python).
# Fully local (Mac): scripts/demo_up.sh --agent real --vlm-url http://127.0.0.1:8011/v1 --summary-model base \
#                      --vlm-label "Mac (vllm-metal)"
# Logs: runs/demo_logs/{sandbox,console}.log
set -euo pipefail
cd "$(dirname "$0")/.."

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

AGENT=mock LIGHTNING=local ASR="${CONSOLE_ASR:-auto}" HEADED=0 SKIP_MODELS=0
GROUNDER=grounder VLM_URL_ARG="" SUMMARY_MODEL_ARG="" VLM_LABEL="${CONSOLE_VLM_LABEL:-}"
THINKING=off CLARIFY=on
while [ $# -gt 0 ]; do
  case "$1" in
    --thinking) THINKING="${2:?--thinking needs off|retry|on}"; shift 2 ;;
    --thinking=*) THINKING="${1#*=}"; shift ;;
    --clarify) CLARIFY="${2:?--clarify needs on|off}"; shift 2 ;;
    --clarify=*) CLARIFY="${1#*=}"; shift ;;
    --agent) AGENT="${2:?--agent needs mock|real}"; shift 2 ;;
    --agent=*) AGENT="${1#*=}"; shift ;;
    --lightning) LIGHTNING="${2:?--lightning needs hosted|local}"; shift 2 ;;
    --lightning=*) LIGHTNING="${1#*=}"; shift ;;
    --grounder) GROUNDER="${2:?--grounder needs base|grounder}"; shift 2 ;;
    --grounder=*) GROUNDER="${1#*=}"; shift ;;
    --vlm-url) VLM_URL_ARG="${2:?--vlm-url needs a URL, e.g. http://127.0.0.1:8011/v1}"; shift 2 ;;
    --vlm-url=*) VLM_URL_ARG="${1#*=}"; shift ;;
    --summary-model) SUMMARY_MODEL_ARG="${2:?--summary-model needs base|grounder}"; shift 2 ;;
    --summary-model=*) SUMMARY_MODEL_ARG="${1#*=}"; shift ;;
    --vlm-label) VLM_LABEL="${2:?--vlm-label needs a text}"; shift 2 ;;
    --vlm-label=*) VLM_LABEL="${1#*=}"; shift ;;
    --asr) ASR="${2:?--asr needs auto|parakeet|stub}"; shift 2 ;;
    --asr=*) ASR="${1#*=}"; shift ;;
    --headed) HEADED=1; shift ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
case "$AGENT" in mock|real) ;; *) echo "--agent must be mock or real" >&2; exit 2 ;; esac
case "$LIGHTNING" in hosted|local) ;; *) echo "--lightning must be hosted or local" >&2; exit 2 ;; esac
case "$GROUNDER" in base|grounder) ;; *) echo "--grounder must be base or grounder" >&2; exit 2 ;; esac
case "$THINKING" in off|retry|on) ;; *) echo "--thinking must be off, retry or on" >&2; exit 2 ;; esac
case "$CLARIFY" in on|off) ;; *) echo "--clarify must be on or off" >&2; exit 2 ;; esac
case "$SUMMARY_MODEL_ARG" in ""|base|grounder) ;; *) echo "--summary-model must be base or grounder" >&2; exit 2 ;; esac
case "$VLM_URL_ARG" in ""|http://*|https://*) ;; *) echo "--vlm-url must be an http(s) URL, e.g. http://127.0.0.1:8011/v1" >&2; exit 2 ;; esac
case "$ASR" in auto|parakeet|stub) ;; *) echo "--asr must be auto, parakeet or stub" >&2; exit 2 ;; esac
VLM_URL_ARG="${VLM_URL_ARG%/}"

PY="${PYTHON:-.venv-gym/bin/python}"
if [ ! -x "$PY" ]; then
  echo "error: no Python at $PY. Create the venv first (README.md, Quick start)." >&2
  exit 1
fi

SANDBOX_PORT="${SANDBOX_PORT:-8090}"; export SANDBOX_PORT   # run_sandbox.sh reads the same variable
SANDBOX_URL="http://127.0.0.1:$SANDBOX_PORT"
CONSOLE_PORT="${CONSOLE_PORT:-8095}"
CONSOLE_URL="http://127.0.0.1:$CONSOLE_PORT"
LIGHTNING_LOCAL_URL="${LIGHTNING_LOCAL_URL:-http://127.0.0.1:8000/v1}"   # what --lightning local checks
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
  # --vlm-url / --summary-model win; otherwise the agent's own config decides (env AGENT_VLM_BASE_URL and
  # AGENT_SUMMARY_MODEL override configs/agent.yaml there).
  CFG=$("$PY" -c 'from agent.config import load_config; c = load_config(); print(c.vlm.base_url.rstrip("/"), c.summary_model)' \
    2>/dev/null || echo "http://127.0.0.1:8001/v1 base")
  VLM_URL="${VLM_URL_ARG:-${CFG% *}}"
  SUMMARY_MODEL="${SUMMARY_MODEL_ARG:-${CFG##* }}"
  if ids=$(curl -sf -m 5 "$VLM_URL/models" | model_ids); then
    echo "vlm      $VLM_URL up, served: ${ids:-none} (grounder: $GROUNDER, summary: $SUMMARY_MODEL)"
    case " $ids " in *" $GROUNDER "*) ;; *)
      echo "warning: grounder '$GROUNDER' is not served at $VLM_URL (serves: ${ids:-nothing}); the console rejects runs" \
        "that ask for it. Pick a served one with --grounder or in the console's Options." >&2 ;;
    esac
    case " $ids " in *" $SUMMARY_MODEL "*) ;; *)
      echo "warning: summary model '$SUMMARY_MODEL' is not served at $VLM_URL (serves: ${ids:-nothing});" \
        "screen summaries will fail. Use --summary-model with a served name." >&2 ;;
    esac
  else
    echo "warning: VLM $VLM_URL not reachable. Open the tunnel (docs/local-mac.md §3a) or start it locally (§3b)." >&2
  fi
  if [ "$LIGHTNING" = local ]; then
    if ids=$(curl -sf -m 5 "$LIGHTNING_LOCAL_URL/models" | model_ids); then
      echo "lightning local $LIGHTNING_LOCAL_URL up, served: ${ids:-none}"
      case " $ids " in *" lightning "*) ;; *) echo "warning: served name 'lightning' missing at $LIGHTNING_LOCAL_URL" >&2 ;; esac
    else
      echo "warning: local Lightning $LIGHTNING_LOCAL_URL not reachable (docs/local-mac.md §2)." >&2
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

# 3. Voice input (local Parakeet sidecar, scripts/run_asr.sh)
ASR_URL="${CONSOLE_ASR_URL:-http://127.0.0.1:8096}"
if [ "$ASR" = stub ]; then
  echo "note:    voice input uses the stub (fixed text, not speech recognition; it never runs by itself)"
else
  ASR=parakeet   # auto never falls back to the stub
  if up "$ASR_URL/health"; then
    echo "asr      Parakeet sidecar up at $ASR_URL"
  else
    echo "warning: Parakeet sidecar not answering at $ASR_URL yet (scripts/run_asr.sh). Voice input is off until" \
      "it does; the page turns it on by itself." >&2
  fi
fi

# 4. Console
if up "$CONSOLE_URL/api/config"; then
  echo "console  already running at $CONSOLE_URL (restart it to change --agent/--grounder/--thinking/--clarify/--vlm-url/--summary-model/--asr/--headed)"
else
  set -- --agent "$AGENT" --lightning "$LIGHTNING" --grounder "$GROUNDER" --thinking "$THINKING" --clarify "$CLARIFY" \
    --asr "$ASR" --sandbox-url "$SANDBOX_URL" --port "$CONSOLE_PORT"
  if [ -n "$VLM_URL_ARG" ]; then set -- "$@" --vlm-url "$VLM_URL_ARG"; fi
  if [ -n "$SUMMARY_MODEL_ARG" ]; then set -- "$@" --summary-model "$SUMMARY_MODEL_ARG"; fi
  if [ -n "$VLM_LABEL" ]; then set -- "$@" --vlm-label "$VLM_LABEL"; fi
  if [ "$HEADED" = 1 ]; then set -- "$@" --headed; fi
  scripts/run_console.sh "$@" >"$LOGS/console.log" 2>&1 &
  PIDS="$PIDS $!"
  wait_up "$CONSOLE_URL/api/config" console "$!"
  desc="agent: $AGENT, lightning: $LIGHTNING, grounder: $GROUNDER, thinking: $THINKING, request check: $CLARIFY"
  if [ -n "$VLM_URL_ARG" ]; then desc="$desc, vlm: $VLM_URL_ARG"; fi
  if [ -n "$SUMMARY_MODEL_ARG" ]; then desc="$desc, summary: $SUMMARY_MODEL_ARG"; fi
  if [ -n "$VLM_LABEL" ]; then desc="$desc, vlm label: $VLM_LABEL"; fi
  desc="$desc, asr: $ASR"
  if [ "$HEADED" = 1 ]; then desc="$desc, headed"; fi
  echo "console  started ($desc)"
fi

echo
echo "  Open $CONSOLE_URL   (sandbox: $SANDBOX_URL)"
echo
if [ -n "$PIDS" ]; then
  echo "Ctrl-C stops the servers started here. Logs: $LOGS/"
  wait
fi
