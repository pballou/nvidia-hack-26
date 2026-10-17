# T2: Orchestrator: Lightning agent core + NeMo Gym `web_agent` env

**Owner:** Patrick (stream C). **Time box:** ~4 h.
- **First deliverable within 30 min:** `common/grounding.py`.
- The loop works on the sandbox with hosted Lightning by ~5pm Saturday.
- The Gym env runs evals by ~8pm.

**Read first:** `AGENTS.md`, `docs/plan.md` ("Decisions so far", served-name
contract), `docs/research-notes.md` §3 (NeMo Gym facts and gotchas) and §6
(Lightning), `docs/tasks/contracts.md` (binding interfaces).

## Why this exists

Nemotron 3.5 Lightning (text-only, 30B-A3B) is the planner and tool-caller. It never
sees pixels. After each action it gets a **text observation**: a structured screen
summary written by the VLM (`base`, adapter off), the URL/title, and whether the
screen changed. It acts through `click(target_description)`. The **grounder** VLM
turns each description into a point on the screenshot, and Playwright clicks there.

The same tools and observations power two things:
- **the interactive agent** used by the console (T1);
- **a NeMo Gym environment.** Its `verify()` scores tasks from sandbox state. It gives
  us the base-vs-tuned agent numbers for the Gym bounty and the README.

## Step 0, first 30 min: `common/grounding.py` + tests

- Implement contracts.md §2 exactly. Pure functions only.
- Start from Charlie's prompt and parser in `scripts/base_grounding_smoke.py`
  (`{"point_2d": [x, y]}` on 0–1000).
- **`preprocess_screenshot`:** the smart-resize math, with dims multiples of 32 inside
  `[MIN_PIXELS, MAX_PIXELS]`, preserving aspect ratio, RGB, alpha composited on white.
- **`build_summary_messages`:** a structured description prompt. Ask for:
  - page title and headings;
  - visible interactive elements (buttons/icons/links/inputs/toggles), each with a
    short visual description and a coarse location (top-left … bottom-right);
  - selected/checked states;
  - open dialogs and toasts.
  Cap it at ~40 lines, in plain text, so it's cheap for Lightning.
- **Tests:** resize edge cases (tall phone, wide desktop, tiny, huge), round-trip
  0–1000 ↔ pixels, and parser robustness (extra text, decimals, malformed input).
- **Commit and push immediately**, then tell Charlie and Kyle in chat that it exists.
  Charlie reviews it and owns it after that.

## Part A: agent core (`agent/`)

| Module | Responsibility |
| --- | --- |
| `agent/config.py` | Loads `configs/agent.yaml` (contracts.md §4) with env-var overrides; builds `AgentConfig`. Provide the yaml too. |
| `agent/browser.py` | Async Playwright 1.62.0 wrapper. One browser per process; **one context per session** (viewport 1280×800, `device_scale_factor=1`, reduced motion, fixed locale/timezone). Methods: `screenshot() -> PIL`, `click(x,y)`, `type(text)`, `press(key)`, `scroll(dir, amount)`, `settle()` (network idle + short fixed delay), `screen_changed(before, after)` (downsampled grayscale diff with a threshold). **No DOM queries anywhere.** |
| `agent/vlm.py` | OpenAI-compatible client for `:8001`. `locate(img, target) -> (x_css, y_css, raw)` uses `grounder_model` plus `common/grounding.py`; `summarize(img) -> str` uses `summary_model`. Temperature 0, timeouts, one retry. Returns errors instead of raising. |
| `agent/prompts.py` | **Single source** for the Lightning system prompt and the tool schema, used by both the interactive planner and the Gym task rows. |
| `agent/observation.py` | Builds the per-step text observation: `{step, url, title, screen_changed, last_action_result, screen_summary}`. Keep it compact. |
| `agent/planner.py` | The interactive Lightning loop (chat completions plus tools). |
| `agent/trace.py` | `StepEvent`/`RunResult` (contracts.md §5); writes `runs/<run_id>/events.jsonl`, `step_XX.png`, and `step_XX_overlay.png` (crosshair at the click, target text, step number). |
| `agent/core.py` | `run_task(request, *, start_url, cfg, on_event)` per contracts.md §5. |

**Tools** (defined once in `agent/prompts.py`):
- `click(target_description)`
- `type_text(text, target_description?)`: click the field first if given
- `press_key(key)`
- `scroll(direction: up|down, amount: small|page)`
- `done(summary, success: bool)`

Tool descriptions should teach the target style the grounder was trained on: short,
specific, visual and positional, e.g. "the gear icon in the top-right toolbar" or "the
Download button in the September row". Coordinate the wording with T3's rewrite
styles.

**Planner behavior:**
- **One action per step.** After each action, wait for `settle()`, re-screenshot, and
  build a new observation.
- **Recovery:**
  - Malformed or unknown tool calls get a corrective tool-result message.
  - Tool errors are returned as observations (`ok:false, error`).
  - If a click changed nothing, say so explicitly in the observation.
  - With `thinking: retry`, enable thinking (`chat_template_kwargs.enable_thinking=true`)
    for the next call only. Keep the default off for latency.
- **Termination:** `done` ends the interactive loop. `max_steps` (12) is a hard stop.
- **Context budget:** keep the system prompt, the request, the last ~6 step
  exchanges in full, and one-line summaries of older steps. vllm-metal has no prefix
  cache for Lightning, so prompts must stay short.
- **Timings per stage** (summary, plan, ground, act) go in every event, for the
  latency story.
- **Lightning endpoint:** hosted or local per config. Both are OpenAI-compatible. Use
  temperature 1.0 and top_p 0.95 (the model card's values). Never add speculative
  decoding.

## Part B: Gym env (`resources_servers/web_agent/`)

- **Start from Gym's `resources_servers/interactive_browser`** (in the installed
  `nemo-gym==0.7.0` package; it was first published 2026-09-22, so it's allowed).
  Copy only what you need, and record the provenance (path, Gym tag v0.7.0, commit
  `1f8afc7`) in a header comment and the README.
- **Use the same tools as Part A.** Each FastAPI route name equals the tool name, and
  each route reuses `agent/browser.py`, `agent/vlm.py` and `agent/observation.py`.
  Return **200 + `{ok, observation, error?, click?}`**, never non-2xx.
- **`seed_session`:**
  - Create a context for this `session_id`.
  - `POST /api/reset {sid, task_id}` on the sandbox, then go to
    `start_path?sid=<sid>`.
  - Return the initial observation to the policy as the first user message: put
    it in the task row's input, or have the first tool result carry it. Pick one
    approach and document it.
- **`verify`:**
  - `GET /api/state?sid=`, then `sandbox.checks.evaluate(state, row.check)`, giving
    reward 1.0/0.0.
  - Return `task_id, tags, steps, n_clicks, claimed_success` on the VerifyResponse,
    so they reach the metrics.
  - **Wrap everything in try/except.** Any failure returns `reward=0,
    mask_sample=True, failure_reason`. One exception would otherwise abort the whole
    `gym eval run`.
  - Release the context inside `verify` (the legacy `/run` path doesn't call
    `close_session`).
- **Declare every row field** you need (`task_id, start_path, check, tags`) on the
  Seed/Verify request subclasses, or Gym silently drops it.
- **Config** `configs/web_agent.yaml`: `web_agent` resources server (`num_workers: 1`),
  `simple_agent` with `max_steps: 12`, the `environment_servers.legacy_agent` block,
  and `policy_model` as `vllm_model`.
  - Hosted Lightning by default: `uses_reasoning_parser: true`,
    `chat_template_kwargs: {enable_thinking: false}`,
    `return_token_id_information: false`.
  - Also provide a local `:8000` variant.
  - Note: Gym's `simple_agent` stops when the model replies without a tool call, not
    on `done`. Our `done` tool should return a final observation telling the model to
    stop.
- **Task rows:** `resources_servers/web_agent/make_tasks.py` turns
  `sandbox/tasks.json` into Gym JSONL. Each row gets `responses_create_params`
  (system prompt plus the request, tools from `agent/prompts.py`,
  `parallel_tool_calls: false`) plus `task_id, start_path, check, tags`. Commit
  `data/example.jsonl` (5 tasks). The full file is generated, not committed.
- **Imports inside the Gym server venv:** Gym builds a per-server venv from
  `requirements.txt` (pin `nemo-gym[dev]==0.7.0`, `playwright==1.62.0`, `openai`,
  `pillow`). Our `agent`, `common` and `sandbox` packages must be importable. There is
  deliberately no root `pyproject.toml` (it would switch Gym's venv builder into
  editable mode for every server), so `app.py` inserts the repo root into `sys.path`,
  clearly commented (AGENTS.md).
- **Eval script** `scripts/agent_eval.sh`:
  1. `gym env start --config resources_servers/web_agent/configs/web_agent_local.yaml`
     (local Lightning, the default; `LIGHTNING=hosted` uses `web_agent.yaml`).
  2. `gym eval run --no-serve --agent web_agent_simple_agent --input <tasks.jsonl> --output results/agent_<grounder_model>.jsonl --num-repeats 3`.
  3. Then `gym eval compare` between base and grounder, plus a reminder to run BLADE.
  - Base runs can happen whenever Kyle's `:8001` is up. **Base-vs-tuned is done**
    (Sat 19:30; 12 tasks × 3, `grounder` = `sft-template`): verified 18/36 → 22/36
    ([docs/status.md](../status.md)). Sunday reruns only if the overnight GRPO adapter
    wins.

## Testing without a GPU

- **Test-only oracle grounder:** `agent/testing/oracle_grounder.py` locates targets
  through DOM bounding boxes.
  - It exists **only** to test loop mechanics in CI and on a laptop.
  - Guard it: it raises if used outside pytest or a `--use-oracle-grounder` debug
    flag, and it's never used in Gym evals or the demo.
  - With hosted Lightning plus the oracle, the loop must solve ≥ 3 easy sandbox tasks.
- **Fake Lightning** (a scripted OpenAI-compatible stub) for unit tests of recovery:
  malformed tool calls, unknown tool, no-change click, `max_steps`.
- **`gym env test --resources-server web_agent`** must pass. Include verifier cases:
  correct state → 1.0; wrong state → 0.0; sandbox down → `mask_sample`.

## Metrics for the Lightning bounty

`scripts/agent_metrics.py` summarizes `runs/` and Gym rollouts:
- task success;
- steps per task;
- tool-call validity rate;
- recovery count;
- per-stage latency p50/p90;
- thinking off vs retry.

These numbers go into `docs/bounty-lightning.md` as they land.

## Acceptance criteria

- [ ] `common/grounding.py` is pushed with tests within ~30 min, and Charlie and Kyle
      are notified.
- [ ] `run_task` works end to end on ≥ 3 easy sandbox tasks with hosted Lightning and
      the oracle grounder (no GPU). Traces and overlays are written, and events
      stream to T1's console.
- [ ] It works with the real `base` VLM on ≥ 1 task once Kyle's `:8001` is up (the
      tunnel, or Mac-local).
- [ ] The Gym env passes `gym env test`, and `gym eval run` on `data/example.jsonl`
      produces rewards from sandbox state. Failures are masked; they never abort the
      run.
- [ ] Switching `grounder_model` / `summary_model` between `base` and `grounder` is
      config-only.
- [ ] No DOM or state access in the agent path. Only verifiers, tests and the guarded
      oracle touch it.

## Out of scope

Training, Charlie's `gui_grounding` env, Kyle's serving setup, and the console UI (T1).
