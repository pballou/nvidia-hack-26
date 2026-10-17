# GUI assist console

The localhost app a user or judge sees: type a request, watch the agent work step by
step in the sandbox, see where it clicked and why, and get a verdict checked against
sandbox state. It's a FastAPI backend plus one static page (`console/static/`), with no
build step. Spec: [docs/tasks/T1-app.md](../docs/tasks/T1-app.md) Part B.

## Run

Run from the repo root, with the sandbox up (`uvicorn sandbox.app:app --port 8090`):

```bash
.venv-gym/bin/python -m console --agent mock              # scripted mock agent, no models needed
.venv-gym/bin/python -m console --agent real              # T2's agent.core (Lightning + grounder VLM)
.venv-gym/bin/python -m console --agent real --headed     # also show the Playwright window (video)
# options: --host 127.0.0.1 --port 8095 --sandbox-url http://127.0.0.1:8090 --runs-dir runs --step-delay 0.8
```

Then open http://127.0.0.1:8095.
- `--agent real` exits at startup if `agent.core` can't be imported.
- `--runs-dir` and `--step-delay` only affect the mock. The real agent writes traces
  to its own `runs_dir` (`configs/agent.yaml`). The console serves images from wherever
  the events point.
- Tests: `.venv-gym/bin/python -m pytest console/tests -q`. They start a real sandbox
  and console on free ports; the mock tests launch headless Chromium.

## HTTP API (binding for `console/static`)

| Route | Returns |
| --- | --- |
| `GET /`, `GET /static/*` | `console/static/index.html` and assets |
| `GET /api/config` | `{agent: "mock"\|"real", sandbox_url, headed, defaults: {grounder: "base", thinking: "retry", lightning: "hosted"}}` |
| `GET /api/tasks` | `[{id, request, start_path, tags, steps_min}]` from `sandbox/tasks.json` (no checks) |
| `POST /api/run` | Body `{request (1..500 chars), task_id?: str\|null, options?: {grounder: base\|grounder, thinking: off\|retry\|on, lightning: hosted\|local}}` → `{run_id: "c-<hex>", sid, start_url}`. Errors: 400 `{error}` (validation), 409 `{error, active_run_id}` (one run at a time), 503 `{error}` (sandbox unreachable, or the agent can't be imported or configured). |
| `GET /api/runs/{run_id}` | `{run_id, status, request, task_id, options, agent_run_id, result, verdict, n_events}`. `n_events` counts stream messages (it equals the last `id`). |
| `GET /api/runs/{run_id}/events` | SSE stream (below) |
| `POST /api/runs/{run_id}/stop` | `{ok: true}`. Idempotent; waits up to ~10 s for the agent to stop. |
| `GET /runs/{agent_run_id}/{file}` | Trace PNG; `file` must match `step_XX.png` or `step_XX_overlay.png`. |

Unknown runs return 404 `{error}`.

**SSE stream.** Every message has `id: <seq>` (1, 2, …). The stream replays all
messages so far, then goes live. Reconnects honor `Last-Event-ID` (or
`?last_event_id=`). A `: ping` comment is sent every 15 s. Once a run is over and
nothing is left to send, the endpoint answers 204, which tells `EventSource` to stop
reconnecting (still call `close()` on `end`).

Messages arrive in this order:
`status(starting)`, `status(running)`, `step`×N, `result`?, `verdict`?,
`status(done|stopped|error)`, `end`.

| Event | Data |
| --- | --- |
| `status` | `{status: starting\|running\|done\|stopped\|error, message?}` |
| `step` | A StepEvent (contracts.md §5: `run_id, step, kind, ts, payload, screenshot_path?, overlay_path?, timings_ms?`), plus `screenshot_url` / `overlay_url` when it has images. Kinds and payloads are in `agent/trace.py`. |
| `result` | RunResult `{run_id, success_claimed, summary, steps, trace_dir}` (only when the agent returns) |
| `verdict` | `{task_id, passed, explanation, label: "verified by sandbox state, not seen by the agent"}` (only when `task_id` was given, after any ending) |
| `end` | `{}`, then the stream closes |

**How a run works.** `sid = "console-" + run_id`. The console first calls
`POST {sandbox}/api/reset {sid, task_id}`. It then starts the agent at
`{sandbox}{start_path}?sid=…`. After the run ends (done, stopped or error), it reads
`GET /api/state?sid=` and calls `sandbox.checks.evaluate`. Here the console acts as the
verifier: the agent never sees state. Agents run on one dedicated worker thread with
one persistent asyncio loop (`runner.py`), never on uvicorn's loop. Stopping a run
cancels its task on that loop.

## Mock agent (`mock_agent.py`)

It has the same `run_task(request, *, start_url, cfg, on_event)`, event kinds and trace
layout (`runs/<run_id>/events.jsonl`, `step_XX.png`, `step_XX_overlay.png`) as T2's
agent, and every payload carries `"mock": true`.
- **Real browser:** it drives a real headless Chromium (1280×800, dsf 1, reduced
  motion) **from pixels only**: `goto`, screenshots, mouse clicks and wheel, keyboard,
  `page.url` and `page.title`.
- **Plans:** these come from `console/mock_scripts.json` and stand in for Lightning and
  the grounder. A request matches a script by its exact task text (case and whitespace
  don't matter). Any other request observes the screen, plans "Mock agent: no scripted
  plan for this request" and ends with `done(success=false)`.
- **Bad scripts:** a missing, malformed or stale script never crashes the agent. It
  emits an `error` event and then `done(reason="error")`. A script is stale when the page
  path after a step doesn't match the recorded one.
- **Stopping:** stopping a run emits `done(reason="stopped")`, then closes the browser.
- **Timings:** `screen_changed` is a downsampled image diff, and `act` timings are
  measured. `summary`, `plan` and `ground` timings are representative simulated numbers.
  `--step-delay` paces the steps.

**Calibration.** `console/tools/calibrate_mock.py` writes `mock_scripts.json`:

```bash
.venv-gym/bin/python -m console.tools.calibrate_mock --sandbox-url http://127.0.0.1:8090 [--tasks id,id] [--headed] [--no-replay]
```

It has a hand-written step list for each of the 12 tasks: the tool, a target description
in the grounder's style, plain-words reasoning, and a screen summary. For each task it:
1. resets a fresh sid;
2. measures each target's center with DOM locators at that moment;
3. adds scroll steps for targets below the fold, and a dismiss step if a toast covers one;
4. performs each step the way the mock will (pixel clicks);
5. checks `evaluate()`;
6. replays every script through the mock agent and prints a table.

The JSON holds only pixel coordinates and text. Re-run it after any sandbox layout change.

## The no-DOM rule

Agent-facing code never reads the DOM or the app state: neither `mock_agent.py` nor
T2's agent. `calibrate_mock.py` is the **only** place in `console/` that uses
selectors (`data-testid`, `elementFromPoint`, `window.AcmeTestHooks`). It's a dev tool,
like the sandbox's scripted oracle, and its output contains no selectors. The console
backend reads `/api/state` only to show the verdict, after the run.
