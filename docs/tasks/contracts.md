# Shared contracts for stream C tasks (T1, T2, T3)

Every task codes against these interfaces, so they can be built in parallel. If you
need to change a contract, change this file in the same commit and say so in team
chat.

## Directory ownership (avoids merge conflicts)

| Path | Owner |
| --- | --- |
| `sandbox/` (target web app, tasks, checks) | T1 |
| `console/` (GUI assist user-facing app) | T1 |
| `agent/` (planner, browser, grounder client, traces) | T2 |
| `resources_servers/web_agent/` (Gym env) | T2 |
| `configs/agent.yaml` | T2 |
| `common/grounding.py` | **T2 creates the first version now** (stream A, Charlie, reviews and then owns it) |
| `data_prep/rewrite/` (Lightning data engine) | T3 |
| `data_prep/*.py` (other), `resources_servers/gui_grounding/`, `training/` | Charlie / Kyle. **Don't edit.** |

**Git:** small commits, `git pull --rebase` before every push, stay inside your own
paths, never force-push, never commit secrets.

## 1. Sandbox HTTP contract (T1 provides; T2 and T3 consume)

- Base URL: `http://127.0.0.1:8090`. Port 8000 is Lightning, 8001 the VLM, and 11000
  Gym's head server.
- **Session:** the first request carries `?sid=<sid>`. The server sets a `sid`
  cookie, and later navigation uses the cookie. Each `sid` has isolated in-memory
  state.
- `POST /api/reset` with body `{"sid": str, "task_id": str | null}` → `200
  {"ok": true}`. It resets that session to the task's initial state (or the default
  state).
- `GET /api/state?sid=<sid>` → `200` with the session's full state as JSON. It's
  read only by verifiers, **never by the agent**.
- **Tasks:** `sandbox/tasks.json` is a list of
  `{id, request, start_path, initial_state_overrides, check, steps_min, tags}`.
  - `check` = `{"all": [ {"path": "settings.accessibility.high_contrast", "op": "eq", "value": true}, ... ]}`.
  - `op` ∈ `eq`, `neq`, `contains`, `len_gte`, `exists`.
- **Checks:** `sandbox/checks.py: evaluate(state: dict, check: dict) -> tuple[bool, str]`
  is pure, with no I/O, and returns (passed, explanation). It's used by
  `sandbox/tests`, the Gym `verify()` and the console.

## 2. Model interface: `common/grounding.py` (T2 creates; everyone imports)

```python
MAX_PIXELS: int = 2_073_600   # FROZEN at 2 MP by gate D1 (must equal serving.env GROUNDER_MAX_PIXELS)
MIN_PIXELS: int = 65_536
def preprocess_screenshot(img: PIL.Image.Image) -> PIL.Image.Image   # RGB; aspect-preserving resize into [MIN,MAX] pixels; W,H multiples of 32; no pad/crop/stretch
def to_data_url(img: PIL.Image.Image) -> str                         # PNG base64 data URL
def build_grounding_messages(image_url: str, instruction: str, family: str = "qwen3_vl") -> list[dict]   # OpenAI chat format
def parse_point(text: str) -> tuple[float, float] | None             # returns the point on the 0–1000 grid
def to_pixels(pt1000: tuple[float, float], w: int, h: int) -> tuple[float, float]
def build_summary_messages(image_url: str) -> list[dict]             # structured screen description prompt
```

- Start from Charlie's prompt in `scripts/base_grounding_smoke.py`
  (`{"point_2d": [x, y]}`, 0–1000). Keep `family` switchable for Qwen3.5.
- Pure functions only, with no network calls. Unit-test them.

## 3. Endpoints and served names

- **VLM:** `http://127.0.0.1:8001/v1` (tunnel to the H200, or Mac-local).
  - `base` = adapter off: summaries and baselines.
  - `grounder` = LoRA adapter on: clicks. Since Sat 18:42 it is `sft-template`. The
    overnight loop starts from `sft-lightning` (the round-0 pick on dev); from Sunday
    ~7:00 `grounder` is the loop's final adapter (`FINAL.json`).
- **Lightning:** local `http://127.0.0.1:8000/v1`, model `lightning` (the default:
  official BF16 on Patrick's Mac via vllm-metal, for the demo and agent evals), or
  hosted (`https://integrate.api.nvidia.com/v1`, model
  `nvidia/nemotron-3.5-lightning-30b-a3b`, key `NVIDIA_API_KEY`; batch jobs). Use the
  OpenAI-compatible chat completions API with tools.

## 4. Agent config: `configs/agent.yaml` (T2; env vars override)

```yaml
sandbox_url: http://127.0.0.1:8090
lightning: {base_url: http://127.0.0.1:8000/v1, model: lightning, api_key_env: null}   # AGENT_LIGHTNING=hosted = build.nvidia.com
vlm: {base_url: http://127.0.0.1:8001/v1, api_key: EMPTY}   # :8001 = Qwen3.5-2B (gate D1)
grounder_model: base        # base | grounder
summary_model: base         # base | grounder (grounder when Mac-local fully local)
model_family: qwen3_5       # gate D1: Qwen/Qwen3.5-2B (qwen3_vl was the other candidate)
max_steps: 12
thinking: retry             # off | on | retry (on only after a no-change/failed step)
viewport: {width: 1280, height: 800}
runs_dir: runs
```

## 5. Agent run interface (T2 provides; T1's console consumes)

```python
async def run_task(request: str, *, start_url: str, cfg: AgentConfig,
                   on_event: Callable[[StepEvent], Awaitable[None]] | None = None) -> RunResult
```

- `StepEvent` is JSON-serializable: `{run_id, step, kind, ts, payload, screenshot_path?, overlay_path?, timings_ms?}`.
  `kind` ∈ `start`, `observation`, `plan`, `tool_call`, `grounding`, `action`,
  `error`, `done`.
- `RunResult` = `{run_id, success_claimed: bool, summary: str, steps: int, trace_dir: str}`.
- Traces go to `runs/<run_id>/`: `events.jsonl`, `step_XX.png`, `step_XX_overlay.png`.
  This is gitignored (`/runs/`).

## 6. Rules that apply to every task (from AGENTS.md)

- **The agent never reads the DOM or the app state.** It acts only through
  screenshots → grounder → pixel clicks. DOM/state access is allowed only in
  `sandbox/checks.py` callers (verifiers, tests).
- **Playwright:** viewport 1280×800, `device_scale_factor=1`, and always
  `preprocess_screenshot` before sending to the VLM. Convert 0–1000 points back to
  viewport CSS px.
- **Models are fixed: Lightning (30B), one VLM (Qwen3.5-2B + LoRA), and NVIDIA
  Parakeet TDT 0.6B for local speech-to-text** (approved 2026-10-17). Card counts:
  30 + 2 + 0.6 = **32.6B ≤ 33B**. The rule is "official parameter count published
  in the model developer's model card". Raw HF tensor totals differ (Lightning
  ~31.6B), so report card counts only. Add no other models: no judges, drafters or cloud speech APIs.
- **Pins:** Playwright `1.62.0` (matches Gym's browser build), Python 3.13 (shared
  with `.venv-gym`), `nemo-gym==0.7.0`.
