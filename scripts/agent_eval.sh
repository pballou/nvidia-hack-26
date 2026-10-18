#!/usr/bin/env bash
# Web-agent eval on the NeMo Gym `web_agent` env (T2): the same rows and Lightning, once per grounder
# served name, then `gym eval compare` base vs grounder.
#
#   scripts/agent_eval.sh                               # base then grounder, local Lightning on :8000
#   GROUNDERS=base REPEATS=1 scripts/agent_eval.sh      # one grounder
#   GROUNDERS=grounder OUT_DIR=results/agent_final \
#     BASELINE=results/agent_v4_sft_template/agent_base.jsonl scripts/agent_eval.sh
#                                                       # rerun only the grounder arm, compare to an earlier base
#   THINKING=on scripts/agent_eval.sh                   # thinking-on variant (results agent_<g>_thinking.*)
#   LIGHTNING=hosted scripts/agent_eval.sh              # build.nvidia.com (NVIDIA_API_KEY from .env; 25-80 s/call)
#
# Knobs (env): GROUNDERS ("base grounder"), LIGHTNING (local | hosted), THINKING (off | on; on also raises
#   max_output_tokens to configs/agent.yaml sampling.thinking_max_tokens), TASKS (rows jsonl, default
#   data/example.jsonl), REPEATS (3), CONCURRENCY (1), SUMMARY_MODEL (base), OUT_DIR (results),
#   HEAD_PORT (11000), READY_TIMEOUT_S (2400), OVERWRITE=1 (replace results in OUT_DIR),
#   REINSTALL=1 (rebuild server venvs), FORCE=1 (allow placeholder rows or a local Lightning other than the
#   official checkpoint: plumbing runs, never reported), EXTRA_START_ARGS (more `gym env start` overrides;
#   not for thinking: use THINKING), GYM (gym CLI), ENV_FILE (.env, hosted only), BASELINE (an earlier
#   agent_base*.jsonl to compare against instead of OUT_DIR's; refused unless its run_info shows the same
#   rows sha256, Lightning checkpoint and thinking mode).
#   The sandbox URL, VLM URL and model family come from configs/agent.yaml (AGENT_* env overrides it).
# Needs: the sandbox, the VLM serving `base` and `grounder` from ONE vLLM process (AGENTS.md), and
#   Lightning. Reported runs: rows with real VLM start screens (make_tasks.py without
#   --placeholder-observation) and the official NVIDIA BF16 Lightning (OFFICIAL_LIGHTNING_ROOT below);
#   run_info records reportable=yes|no. Base-vs-tuned runs are Sunday 7am+ (docs/plan.md M5). Log every
#   run in docs/runs.md. Never `set -x` here (the hosted key is in the env).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"  # Gym resolves configs, server venvs, env.yaml and ./results from the cwd
# Gym searches $NEMO_GYM_EXTRA_ROOTS before the cwd (configs, --input, server dirs, imports), and SETUP.md
# exports it per checkout: pin it here so a shell from another checkout can't swap in its rows or agent/.
export NEMO_GYM_EXTRA_ROOTS="$REPO"

GROUNDERS="${GROUNDERS:-base grounder}"
LIGHTNING="${LIGHTNING:-local}"
THINKING="${THINKING:-off}"
TASKS="${TASKS:-resources_servers/web_agent/data/example.jsonl}"
REPEATS="${REPEATS:-3}"
# Sequential by default: local Lightning allows 2 concurrent requests and is shared with T1's console;
# hosted rate limits are unknown (vllm_model retries a 429 only 3 times).
CONCURRENCY="${CONCURRENCY:-1}"
SUMMARY_MODEL="${SUMMARY_MODEL:-base}"
OUT_DIR="${OUT_DIR:-results}"
HEAD_PORT="${HEAD_PORT:-11000}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-2400}"  # a cold first start builds 4 server venvs (~10+ min)
FORCE="${FORCE:-0}"
GYM="${GYM:-$REPO/.venv-gym/bin/gym}"
PY="$(dirname "$GYM")/python"
LOCAL_LIGHTNING_URL=http://127.0.0.1:8000/v1        # = web_agent_local.yaml
OFFICIAL_LIGHTNING_ROOT=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16  # the only reportable local planner
HOSTED_LIGHTNING_URL=https://integrate.api.nvidia.com/v1  # = web_agent.yaml
AGENT_NAME=web_agent_simple_agent
SERVER_KEY=web_agent.resources_servers.web_agent

die() { echo "agent_eval: $*" >&2; exit 1; }
warn() { echo "agent_eval: WARNING: $*" >&2; }

[[ -x "$GYM" ]] || die "gym CLI not found at $GYM (set GYM=)"
[[ -x "$PY" ]] || die "no python next to $GYM"
case "$LIGHTNING" in
  local) CONFIG=resources_servers/web_agent/configs/web_agent_local.yaml ;;
  hosted) CONFIG=resources_servers/web_agent/configs/web_agent.yaml ;;
  *) die "LIGHTNING must be local or hosted, got '$LIGHTNING'" ;;
esac
case "$THINKING" in off) SUFFIX="" ;; on) SUFFIX=_thinking ;; *) die "THINKING must be off or on, got '$THINKING'" ;; esac
for g in $GROUNDERS $SUMMARY_MODEL; do
  case "$g" in base|grounder) ;; *) die "served names are base | grounder, got '$g'" ;; esac
done
[[ -f "$TASKS" ]] || die "no rows file $TASKS (make it with resources_servers/web_agent/make_tasks.py)"
TASKS="$(cd "$(dirname "$TASKS")" && pwd)/$(basename "$TASKS")"  # absolute: Gym returns absolute paths as is
read -r -a EXTRA <<<"${EXTRA_START_ARGS:-}"
[[ "${EXTRA_START_ARGS:-}" != *enable_thinking* ]] ||
  die "use THINKING=on, not enable_thinking in EXTRA_START_ARGS (it also raises max_output_tokens and is logged)"
REPORTABLE=yes

if [[ "$LIGHTNING" == hosted ]]; then
  # Gym never reads .env; web_agent.yaml reads ${oc.env:NVIDIA_API_KEY}, so export it. Never echo it.
  ENV_FILE="${ENV_FILE:-$REPO/.env}"
  if [[ -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; fi
  [[ -n "${NVIDIA_API_KEY:-}" ]] || die "NVIDIA_API_KEY is not set (add it to .env)"
fi

# --- preflight -----------------------------------------------------------------------------------
# Rows: Gym's row model + the env's invariants (stale prompts/tools, tasks drift), then their start screens.
"$PY" resources_servers/web_agent/make_tasks.py --check "$TASKS" || die "$TASKS failed make_tasks.py --check"
SOURCES="$("$PY" -c 'import json, sys
print(" ".join(sorted({str(json.loads(l).get("initial_observation_source")) for l in open(sys.argv[1]) if l.strip()})))' "$TASKS")"
if [[ " $SOURCES " == *" placeholder "* ]]; then
  [[ "$FORCE" == 1 ]] || die "$TASKS has placeholder start screens (make_tasks.py --placeholder-observation): \
plumbing only. Regenerate the rows with the VLM up, or FORCE=1 for a plumbing run."
  warn "FORCE=1 with placeholder start screens: plumbing run, do not report these numbers"
  REPORTABLE=no
fi
ROWS_MAX_OUT="$("$PY" -c 'import json, sys
print(",".join(sorted({str(json.loads(l)["responses_create_params"].get("max_output_tokens")) for l in open(sys.argv[1]) if l.strip()})))' "$TASKS")"
[[ " $SOURCES " == *" placeholder "* || " $SOURCES " == *" vlm:$SUMMARY_MODEL "* ]] ||
  warn "rows' start screens come from '$SOURCES' but SUMMARY_MODEL=$SUMMARY_MODEL describes every later screen"

# The settings the server will use: configs/agent.yaml + AGENT_* env (agent/config.py), passed explicitly.
CFG="$("$PY" -c 'from agent.config import load_config
c = load_config()
print(c.sandbox_url, c.vlm.base_url, c.model_family, c.sampling.thinking_max_tokens)')" || die "could not load configs/agent.yaml"
read -r SANDBOX_URL VLM_URL MODEL_FAMILY THINKING_MAX_TOKENS <<<"$CFG"
# Reasoning counts against max_output_tokens, so thinking-on calls get thinking_max_tokens (agent/config.py).
if [[ "$THINKING" == on ]]; then MAX_OUT="$THINKING_MAX_TOKENS"; else MAX_OUT="$ROWS_MAX_OUT"; fi

# Never GET /api/state for a made-up sid: the sandbox creates state for it. The test-only fake sandbox
# (agent/testing/fake_sandbox.py) has no /api/health, so fall back to its dashboard page.
curl -sf -m 5 -o /dev/null "$SANDBOX_URL/api/health" ||
  curl -sf -m 5 -o /dev/null "$SANDBOX_URL/" ||
  die "sandbox at $SANDBOX_URL is down"

VLM_MODELS="$(curl -sf -m 10 "$VLM_URL/models" | "$PY" -c 'import json, sys
print(" ".join(m["id"] + "=" + str(m.get("root")) for m in json.load(sys.stdin)["data"]))' 2>/dev/null)" ||
  die "VLM at $VLM_URL is down (tunnel? ask Kyle)"
for need in $GROUNDERS $SUMMARY_MODEL; do
  [[ " $VLM_MODELS " == *" $need="* ]] || die "VLM at $VLM_URL doesn't serve '$need' (serves: $VLM_MODELS)"
done

if [[ "$LIGHTNING" == local ]]; then
  LIGHTNING_ROOT="$(curl -sf -m 10 "$LOCAL_LIGHTNING_URL/models" | "$PY" -c 'import json, sys
print(next((m.get("root") or m["id"] for m in json.load(sys.stdin)["data"] if m["id"] == "lightning"), ""))' 2>/dev/null)" ||
    die "Lightning at $LOCAL_LIGHTNING_URL is down"
  [[ -n "$LIGHTNING_ROOT" ]] || die "$LOCAL_LIGHTNING_URL serves no model named 'lightning'"
  # Only the official NVIDIA BF16 checkpoint is reported: its HF repo id, or a local download of that repo
  # (an absolute path ending in the repo name). A community conversion is plumbing only, whatever its name.
  if [[ "$LIGHTNING_ROOT" != "$OFFICIAL_LIGHTNING_ROOT" &&
        ! ( "$LIGHTNING_ROOT" == /* && "${LIGHTNING_ROOT%/}" == */"${OFFICIAL_LIGHTNING_ROOT#*/}" ) ]]; then
    [[ "$FORCE" == 1 ]] || die "local Lightning is '$LIGHTNING_ROOT', not the official $OFFICIAL_LIGHTNING_ROOT. \
FORCE=1 for a plumbing run (never reported)."
    warn "FORCE=1 with Lightning '$LIGHTNING_ROOT': plumbing run, do not report these numbers"
    REPORTABLE=no
  fi
else
  curl -s -m 15 -o /dev/null "$HOSTED_LIGHTNING_URL/models" || die "hosted Lightning at $HOSTED_LIGHTNING_URL is unreachable"
  LIGHTNING_ROOT="nvidia/nemotron-3.5-lightning-30b-a3b (hosted)"
  warn "hosted Lightning: development numbers; reported runs use the self-hosted BF16 checkpoint"
  REPORTABLE=no
fi

for g in $GROUNDERS; do  # gym eval run truncates its output; don't lose a finished run by accident
  [[ ! -e "$OUT_DIR/agent_$g$SUFFIX.jsonl" || "${OVERWRITE:-0}" == 1 ]] ||
    die "$OUT_DIR/agent_$g$SUFFIX.jsonl exists; set OUT_DIR=<new dir> or OVERWRITE=1"
done
if curl -s -m 2 -o /dev/null "http://127.0.0.1:$HEAD_PORT/"; then
  die "something already listens on :$HEAD_PORT (a Gym head server?); stop it or set HEAD_PORT"
fi

# --- servers -------------------------------------------------------------------------------------
GYM_PID=""
stop_env() {
  [[ -n "$GYM_PID" ]] || return 0
  # Interrupt the whole process group, like Ctrl-C: Gym's RunHelper stops only on KeyboardInterrupt, and
  # it SIGINTs each server's `bash -c` wrapper (not the python under it) and kills the wrapper after 1 s,
  # which would orphan the servers. `set -m` gave the job its own group with SIGINT not ignored.
  # (app.py may log "could not close the browser pool": Playwright's driver got the SIGINT too.)
  kill -INT -- "-$GYM_PID" 2>/dev/null || true
  local i
  for ((i = 0; i < 90; i++)); do kill -0 -- "-$GYM_PID" 2>/dev/null || break; sleep 1; done
  kill -TERM -- "-$GYM_PID" 2>/dev/null && sleep 3
  kill -KILL -- "-$GYM_PID" 2>/dev/null || true
  wait "$GYM_PID" 2>/dev/null || true
  GYM_PID=""
  for ((i = 0; i < 30; i++)); do curl -s -m 1 -o /dev/null "http://127.0.0.1:$HEAD_PORT/" || break; sleep 1; done
}
trap stop_env EXIT
trap 'exit 130' INT TERM

wait_ready() {  # $1 = server log
  local deadline=$((SECONDS + READY_TIMEOUT_S))
  until curl -sf -m 5 -o /dev/null "http://127.0.0.1:$HEAD_PORT/readyz"; do
    if ! kill -0 "$GYM_PID" 2>/dev/null; then tail -n 40 "$1" >&2; die "gym env start exited (log: $1)"; fi
    if ((SECONDS > deadline)); then tail -n 40 "$1" >&2; die "servers not ready after ${READY_TIMEOUT_S}s (log: $1)"; fi
    sleep 5
  done
}

sha256() { "$PY" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"; }

mkdir -p "$OUT_DIR"
TASKS_SHA="$(sha256 "$TASKS")"
if [[ -n "${BASELINE:-}" ]]; then  # an earlier base arm is only comparable on the same rows: check before running
  BINFO="${BASELINE%.jsonl}_run_info.txt"
  [[ -f "$BASELINE" && -f "$BINFO" ]] || die "BASELINE=$BASELINE or its run_info ($BINFO) is missing"
  grep -q "sha256=$TASKS_SHA " "$BINFO" || die "BASELINE used different rows than $TASKS (sha256 mismatch, see $BINFO)"
  [[ " $GROUNDERS " != *" base "* ]] || warn "BASELINE is set, so this run's own base arm is not used in the compare"
fi
FAILED=()
for GROUNDER in $GROUNDERS; do
  STEM="$OUT_DIR/agent_$GROUNDER$SUFFIX"
  OUT="$STEM.jsonl"
  LOG="${STEM}_servers.log"
  INFO="${STEM}_run_info.txt"
  START_ARGS=(--config "$CONFIG"
    "++$SERVER_KEY.grounder_model=$GROUNDER" "++$SERVER_KEY.summary_model=$SUMMARY_MODEL"
    "++$SERVER_KEY.sandbox_url=$SANDBOX_URL" "++$SERVER_KEY.vlm_base_url=$VLM_URL"
    "++$SERVER_KEY.model_family=$MODEL_FAMILY"
    "++head_server.port=$HEAD_PORT" ++server_spinup_timeout_seconds=1800)
  [[ "${REINSTALL:-0}" == 1 ]] || START_ARGS+=(++skip_venv_if_present=true)
  RUN_ARGS=(--no-serve --agent "$AGENT_NAME" --input "$TASKS" --output "$OUT"
    --num-repeats "$REPEATS" --concurrency "$CONCURRENCY"
    "++head_server.port=$HEAD_PORT" +route_failures_to_sidecar=true)  # a failed /run -> ${STEM}_failures.jsonl
  if [[ "$THINKING" == on ]]; then
    START_ARGS+=(++lightning_policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking=true)
    RUN_ARGS+=(--max-output-tokens "$THINKING_MAX_TOKENS")  # -> +responses_create_params.max_output_tokens
  fi
  START_ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})

  {
    echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "grounder_model=$GROUNDER summary_model=$SUMMARY_MODEL model_family=$MODEL_FAMILY"
    echo "lightning=$LIGHTNING ($LIGHTNING_ROOT) thinking=$THINKING max_output_tokens=$MAX_OUT reportable=$REPORTABLE"
    echo "vlm=$VLM_URL served: $VLM_MODELS"
    echo "sandbox=$SANDBOX_URL"
    echo "rows=$TASKS sha256=$TASKS_SHA start_screens=$SOURCES repeats=$REPEATS concurrency=$CONCURRENCY"
    echo "gym env start ${START_ARGS[*]}"
    echo "gym eval run ${RUN_ARGS[*]}"
  } | tee "$INFO"

  set -m
  "$GYM" env start "${START_ARGS[@]}" </dev/null >"$LOG" 2>&1 &
  GYM_PID=$!
  set +m
  echo "== $GROUNDER: gym env start pid $GYM_PID, log $LOG; waiting for :$HEAD_PORT/readyz"
  wait_ready "$LOG"

  if "$GYM" eval run "${RUN_ARGS[@]}"; then
    "$GYM" eval profile --inputs "${STEM}_materialized_inputs.jsonl" --rollouts "$OUT" ||
      warn "gym eval profile failed for $OUT"
  else
    FAILED+=("$GROUNDER")
  fi
  stop_env
  echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$INFO"
  [[ -s "${STEM}_failures.jsonl" ]] && warn "failed rollouts (not scored) in ${STEM}_failures.jsonl"
  echo "== $GROUNDER: $OUT, ${STEM}_aggregate_metrics.json"
done

# --- base vs grounder ----------------------------------------------------------------------------
BASE="${BASELINE:-$OUT_DIR/agent_base$SUFFIX.jsonl}"
CAND="$OUT_DIR/agent_grounder$SUFFIX.jsonl"
if [[ -n "${BASELINE:-}" && -f "${CAND%.jsonl}_run_info.txt" ]]; then  # same planner build and thinking mode
  CINFO="${CAND%.jsonl}_run_info.txt"
  for pat in 'lightning=[^)]*)' 'thinking=[^ ]*' 'max_output_tokens=[^ ]*'; do
    [[ "$(grep -o "$pat" "$BINFO" | head -1)" == "$(grep -o "$pat" "$CINFO" | head -1)" ]] ||
      die "BASELINE differs from this run in '${pat%%=*}' (compare $BINFO and $CINFO); not comparable"
  done
  echo "comparing against BASELINE=$BASELINE (same rows sha256 $TASKS_SHA, same Lightning and thinking)"
fi
if [[ -f "${BASE%.jsonl}_aggregate_metrics.json" && -f "${CAND%.jsonl}_aggregate_metrics.json" ]]; then
  "$GYM" eval compare --baseline "$BASE" --candidates "$CAND" --agent "$AGENT_NAME" \
    --output-dir "$OUT_DIR/agent_compare$SUFFIX" || warn "gym eval compare failed"
  echo "Compare report: $OUT_DIR/agent_compare$SUFFIX/ (check both runs used the same rows: *_run_info.txt)"
  echo "Next: run BLADE failure analysis (.agents/skills/nemo-gym-blade-analysis) on $BASE vs $CAND;"
  echo "      let its taxonomy drive the next data/prompt/verifier change."
fi
echo "Log this run in docs/runs.md: command, $OUT_DIR/agent_*_run_info.txt, served names, Lightning checkpoint,"
echo "start/end and the key metrics from $OUT_DIR/agent_*_aggregate_metrics.json."
((${#FAILED[@]} == 0)) || die "gym eval run failed for: ${FAILED[*]}"
