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
# options: --lightning local|hosted (initial toggle, default local) --host 127.0.0.1 --port 8095
#          --sandbox-url http://127.0.0.1:8090 --runs-dir runs --step-delay 0.8
```

Then open http://127.0.0.1:8095.
- `--agent real` exits at startup if `agent.core` can't be imported.
- `--runs-dir` and `--step-delay` only affect the mock. The real agent writes traces
  to its own `runs_dir` (`configs/agent.yaml`). The console serves images from wherever
  the events point.
- **Tests:** `.venv-gym/bin/python -m pytest console/tests -q` (about 70 s). They
  start a real sandbox and console on free ports; the mock tests launch headless
  Chromium. Two files carry the `e2e` marker. Skip both with `-m "not e2e"` (about
  15 s).
  - `test_mock_e2e.py` (about 45 s) runs all 12 sandbox tasks through
    `POST /api/run` and the SSE stream with the real mock agent, and checks that each
    verdict passes. The mock only clicks pixels, so this proves every task can be
    solved from pixels alone.
  - `test_console_ui.py` (about 10 s) drives the page by keyboard only: pick a task,
    Enter to run, see the ✓ banner, Escape to stop, free text. It also checks for no
    console errors and no off-host requests.

## HTTP API (binding for `console/static`)

| Route | Returns |
| --- | --- |
| `GET /`, `GET /static/*` | `console/static/index.html` and assets |
| `GET /api/config` | `{agent: "mock"\|"real", sandbox_url, headed, defaults: {grounder: "base", thinking: "retry", lightning: "local"}, asr: {available, model, label, error?}}` (`--lightning` sets `defaults.lightning`) |
| `GET /api/tasks` | `[{id, request, start_path, tags, steps_min}]` from `sandbox/tasks.json` (no checks) |
| `POST /api/run` | Body `{request (1..500 chars), task_id?: str\|null, options?: {grounder: base\|grounder, thinking: off\|retry\|on, lightning: hosted\|local}, voice?: {text, model, latency_ms, duration_s}}` (`voice` is logged in `console.json`, never sent to the agent) → `{run_id: "c-<hex>", sid, start_url}`. Errors: 400 `{error}` (validation), 409 `{error, active_run_id}` (one run at a time), 503 `{error}` (sandbox unreachable, or the agent can't be imported or configured). |
| `GET /api/runs/{run_id}` | `{run_id, status, request, task_id, options, agent_run_id, result, verdict, n_events}`. `n_events` counts stream messages (it equals the last `id`). |
| `GET /api/runs/{run_id}/events` | SSE stream (below) |
| `POST /api/runs/{run_id}/stop` | `{ok: true}`. Idempotent; waits up to ~10 s for the agent to stop. |
| `POST /api/transcribe` | Voice input. Body: a 16 kHz mono PCM16 WAV, either raw `audio/wav` or multipart field `audio`, 0.2–60 s → `{text, duration_s, latency_ms, model}`. Errors: 400 bad audio, 413 too large, 503 engine unavailable. |
| `GET /runs/{agent_run_id}/{file}` | Trace PNG; `file` must match `step_XX.png` or `step_XX_overlay.png`. |
| `GET /logs`, `GET /api/logs/runs[/{run_id}[/download]]` | The Logs page and its API (`console/logs.py`): every run and event from `runs/<id>/events.jsonl` plus `console.json` |

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
- **Events:** every click, and every `type_text` with a target, gets a `grounding` event
  (`target, x, y, point_1000, raw, model`) and a `step_XX_overlay.png`. Every
  `tool_call` carries `name, arguments, valid, description` (the plain-words line).
- **Timings:** `screen_changed` is a downsampled image diff. Only measured `act`
  timings are reported. The mock runs no Lightning or VLM, so it reports no `plan`,
  `summary` or `ground` latency rather than making numbers up. `--step-delay` paces
  the steps.

**Calibration.** `console/tools/calibrate_mock.py` writes `mock_scripts.json`:

```bash
.venv-gym/bin/python -m console.tools.calibrate_mock --sandbox-url http://127.0.0.1:8090 [--tasks id,id] [--headed] [--no-replay]
```

It has a hand-written plan for each of the 12 tasks: the path a pixel agent would take,
written from screenshots of the real pages. Each step has:
- the tool call, with the target description in the grounder's style (short, visual,
  positional, e.g. "the download icon at the right end of the September 2026 row");
- plain-words reasoning;
- the screen description the VLM would write afterwards, in PAGE / ELEMENTS / STATE /
  OVERLAYS sections (an earlier `SUMMARY_PROMPT`; the current one splits ELEMENTS into
  NAV and MAIN).

Targets below the fold get explicit mouse-wheel scroll steps (`star-landlord-message`,
`stop-promotions`). For each task the tool:
1. resets a fresh sid;
2. measures each target's center with DOM locators at that moment;
3. checks the point is really clickable (`elementFromPoint`). If it isn't, it adds a
   scroll or toast-dismiss step and reports it in the `auto` column; make it explicit in
   the plan instead;
4. performs each step the way the mock will (pixel clicks, wheel, keys);
5. validates the script against the agent's tool schema (`agent/prompts.py`), checks the
   viewport bounds, and checks that only replay keys appear (no selectors);
6. checks `evaluate()`;
7. replays every script through the mock agent from a fresh reset. A replay must have no
   errors, end in `done(success=true)`, change the screen on every action, and give
   every click a grounding event and an overlay; then `evaluate()` must pass;
8. prints a table (steps, auto-added steps, walk, replay, seconds, explanation).

It exits 1 if any task fails, and 2 if the sandbox is unreachable. The JSON holds only
pixel coordinates and text. Re-run it after any sandbox layout change, then re-run
`test_mock_e2e.py`.

## Voice input

It's local only: no Web Speech API, no MediaRecorder, no cloud.

- **Recording:** click **Speak**, or hold Right Option (`AltRight`, configurable in
  Options). Space-hold works when focus isn't on a field or control.
- **Capture:** `voice.js` records raw PCM through an AudioWorklet (`pcm-worklet.js`,
  with a ScriptProcessor fallback). `wav.js` resamples to 16 kHz if the browser
  refuses a 16 kHz context, and encodes PCM16 WAV.
- **Transcription:** the WAV is POSTed to `/api/transcribe`.
- **After transcription:** the transcript fills the request box and selects the
  matching task, if any. The log shows `Heard (engine, N ms): …`. The run starts
  after a 1.5 s countdown, which Esc cancels. Saying "stop" or "cancel" during a run
  stops it.
- **Engines** (`asr.py`):
  - `stub` is the default (fixed text; `CONSOLE_ASR_STUB_TEXT`).
  - `parakeet` is NVIDIA Parakeet TDT 0.6B v3 through the NeMo-Speech.cpp sidecar at
    `CONSOLE_ASR_URL` (default `http://127.0.0.1:8096`, `scripts/run_asr.sh`).
  - Select one with `--asr parakeet` or `CONSOLE_ASR=parakeet`. `demo_up.sh` picks
    Parakeet automatically when the sidecar answers.
- **Page API:** `console.js` exposes `window.AcmeConsole` (`ready`, `config()`,
  `isRunning()`, `setRequest(text)`, `run()`, `stop()`, `matchTask()`) and fires
  `acme:run-start` / `acme:run-end`, which `voice.js` uses.
- **Tests:** `test_asr.py` (validation, engines, fake sidecar), `test_voice.py` (the
  WAV encoder, the endpoint, the fake-mic page flow, and a static scan for forbidden
  APIs) and `test_voice_ui.py` (push-to-talk and the countdown, with Chromium's fake
  media flags).

## The no-DOM rule

Agent-facing code never reads the DOM or the app state: neither `mock_agent.py` nor
T2's agent. `calibrate_mock.py` is the **only** place in `console/` that uses
selectors (`data-testid`, `elementFromPoint`, `window.AcmeTestHooks`). It's a dev tool,
like the sandbox's scripted oracle, and its output contains no selectors. The console
backend reads `/api/state` only to show the verdict, after the run.
