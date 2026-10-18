# GUI Assist console

The localhost app a user or judge sees: type a request, watch the agent work step by
step in the sandbox, see where it clicked and why, and get a verdict checked against
sandbox state. It's a FastAPI backend plus two static pages (`console/static/`): the
console and the [Logs page](#logs-page). There's no build step. Spec: [docs/tasks/T1-app.md](../docs/tasks/T1-app.md) Part B.

## Run

Run from the repo root, with the sandbox up (`uvicorn sandbox.app:app --port 8090`):

```bash
.venv-gym/bin/python -m console --agent mock              # scripted mock agent, no models needed
.venv-gym/bin/python -m console --agent real              # T2's agent.core (Lightning + grounder VLM)
.venv-gym/bin/python -m console --agent real --headed     # debugging only: headed fonts render different pixels
                                                          # (unmeasured; pay-50 fell from 8/8 to 1/4). Record headless.
# fully local on the Mac: the merged grounder on :8011 writes the screen summaries too
.venv-gym/bin/python -m console --agent real --vlm-url http://127.0.0.1:8011/v1 --summary-model grounder \
    --vlm-label "Mac (vllm-metal)"
# options: --lightning local|hosted (initial toggle, default local) --grounder base|grounder (initial choice,
#          default grounder) --thinking off|retry|on (initial choice, default off) --clarify on|off (initial
#          request check, default on) --asr parakeet|stub --host 127.0.0.1 --port 8095
#          --sandbox-url http://127.0.0.1:8090 --runs-dir runs --step-delay 0.8
# real agent only: --vlm-url URL --summary-model base|grounder --vlm-label TEXT (or CONSOLE_VLM_LABEL)
```

Then open <http://127.0.0.1:8095> (the Logs page is <http://127.0.0.1:8095/logs>).

- `--agent real` exits at startup if `agent.core` can't be imported. It prints where the
  VLM and Lightning are, and warns if the VLM doesn't answer or doesn't serve the default
  grounder or the summary model (see [Endpoints](#endpoints-where-the-models-are)).
- `--vlm-url` and `--summary-model` become `agent.config.load_config` overrides
  (`vlm={"base_url": URL}`, `summary_model=...`). Without them the agent's config decides,
  so `AGENT_VLM_BASE_URL` and `AGENT_SUMMARY_MODEL` keep working; a flag beats the env var.
  The mock ignores them (it runs no models).
- `--thinking` and `--clarify` set the console's defaults (`/api/config` `defaults.thinking` /
  `defaults.clarify`); each run can override them in `options`. `scripts/demo_up.sh` takes the
  same two flags (defaults off / on) and passes them on. See
  [Request check and follow-ups](#request-check-and-follow-ups).
- `--runs-dir` and `--step-delay` only affect the mock. The real agent writes traces
  to its own `runs_dir` (`configs/agent.yaml`). The console serves images from wherever
  the events point.
- **Tests:** `.venv-gym/bin/python -m pytest console/tests -q` (a few minutes). They
  start a real sandbox and console on free ports; the mock tests launch headless
  Chromium. The slow browser tests carry the `e2e` marker. Skip them with `-m "not e2e"`
  (about 40 s). No test calls a model: `test_endpoints.py` runs the real agent's config
  path with a fake `run_task` and a fake `GET /v1/models` server.
  - `test_mock_e2e.py` (about 45 s) runs all 12 sandbox tasks through
    `POST /api/run` and the SSE stream with the real mock agent, and checks that each
    verdict passes. The mock only clicks pixels, so this proves every task can be
    solved from pixels alone.
  - `test_console_ui.py` (about 10 s) drives the page by keyboard only: pick a task,
    Enter to run, see the ✓ banner, Escape to stop, free text. It also checks for no
    console errors and no off-host requests.
  - `test_task_match.py` (about 6 s) checks [task matching](#matching-a-request-to-a-sandbox-task)
    in the real page: the full match table (run it with `-s` to print every score), the
    picker, Free text undoing a match, the banner and Logs lines, and the `task_match` API.
  - `test_clarify_api.py` (about 12 s) checks the [request check](#request-check-and-follow-ups)
    backend: defaults and flags (also `demo_up.sh`), the per-run option reaching the agent's
    config, `clarifies` validation and the built request, `parent_run_id` reaching `run_task`,
    the `console.json` links, the restated late match's rules and verdict, and the mock's
    clarify flows (one e2e run: ask, follow-up, restated match, pay-50 replay, ✓).

## HTTP API (binding for `console/static`)

| Route | Returns |
| --- | --- |
| `GET /`, `GET /static/*` | `console/static/index.html` and assets |
| `GET /api/config` | `{agent: "mock"\|"real", sandbox_url, headed, defaults: {grounder: "grounder", thinking: "off", lightning: "local", clarify: "on"}, asr: {available, model, label, fixed_text?, error?}, endpoints: {vlm, lightning}}` (`--grounder`, `--lightning`, `--thinking` and `--clarify` set those defaults; `endpoints` below) |
| `GET /api/tasks` | `[{id, request, start_path, tags, steps_min}]` from `sandbox/tasks.json` (no checks) |
| `POST /api/run` | Body `{request (1..500 chars), task_id?: str\|null, options?: {grounder: base\|grounder, thinking: off\|retry\|on, lightning: hosted\|local, clarify: on\|off}, voice?: {text, model, latency_ms, duration_s}, task_match?: {method: picked\|exact\|fuzzy, score: 0..1\|null, matched_from: str (≤500)}, clarifies?: {parent_run_id, answer (1..300 chars)}}` (`voice` and `task_match` are logged in `console.json`, never sent to the agent; `task_match` needs a `task_id`, has exactly those three fields, and is recorded as `exact` when left out with a request that is the task's text; `clarifies` makes it a [follow-up](#request-check-and-follow-ups), and then `request` is optional) → `{run_id: "c-<hex>", sid, start_url, request}` (`request`: the text the agent gets; for a follow-up, the one the backend built). Errors: 400 `{error}` (validation; a follow-up whose parent is unknown, hasn't ended or didn't end with `needs_input`, or whose `request` isn't the backend's text; or `grounder model '<x>' is not served at <url> (serves: ...)` when the VLM's served names are known and lack it), 409 `{error, active_run_id}` (one run at a time), 503 `{error}` (sandbox unreachable, or the agent can't be imported or configured). |
| `GET /api/runs/{run_id}` | `{run_id, status, request, task_id, task_match, options, agent_run_id, result, verdict, endpoints, parent_run_id, clarifies, n_events}`. `n_events` counts stream messages (it equals the last `id`). `endpoints` is the snapshot taken at run start. `task_id` / `task_match` can change during the run (a restated match). |
| `GET /api/runs/{run_id}/events` | SSE stream (below) |
| `POST /api/runs/{run_id}/stop` | `{ok: true}`. Idempotent; waits up to ~10 s for the agent to stop. |
| `POST /api/runs/{run_id}/task_match` | Body `{task_id, method: "restated", score: 0..1\|null, matched_from}`: check this free-text run as `task_id`, because the agent's restatement matches it ([restated match](#a-task-from-the-agents-restatement)) → `{ok: true, run_id, task_id, task_match}`. 404 unknown run; 400 `{error}` bad body, or `matched_from` isn't the run's restatement; 409 `{error}` the run already has a task, it has ended (the verdict is decided), there's no restatement (clarify off, or it hasn't proceeded), or the task starts on another page or from its own state. |
| `POST /api/transcribe` | Voice input. Body: a 16 kHz mono PCM16 WAV, either raw `audio/wav` or multipart field `audio`, 0.2–60 s → `{text, duration_s, latency_ms, model}`. Errors: 400 bad audio, 413 too large, 503 engine unavailable. |
| `GET /runs/{agent_run_id}/{file}` | Trace PNG; `file` must match `step_XX.png` or `step_XX_overlay.png`. |
| `GET /logs`, `GET /api/logs/runs[/{run_id}[/download]]` | The Logs page and its API (`console/logs.py`): every run and event from `runs/<id>/events.jsonl` plus `console.json` ([Logs page](#logs-page)) |

Unknown runs return 404 `{error}`.

### Endpoints (where the models are)

`/api/config` `endpoints` (and `console.json` `endpoints`, and `GET /api/runs/{id}`):

```text
{"vlm": {"base_url", "label", "summary_model": "base"|"grounder", "served": [str] | null, "reachable": bool} | null,
 "lightning": {"base_url", "model", "label"} | null}
```

- **Mock:** both are `null` (it runs no models).
- **Real:** the values come from the agent's cfg after the console's overrides
  (`make_cfg(options)`, `console/agent_bridge.py`). `/api/config` uses the default options;
  a run uses its own (so the Lightning toggle shows up in its snapshot).
- **`served`:** the ids from `GET <vlm base_url>/models` (2 s timeout), re-checked at most
  every 10 s. When that fails: `served: null, reachable: false`, and runs aren't blocked
  (the console doesn't guess).
- **Labels:** VLM: `--vlm-label` / `CONSOLE_VLM_LABEL` if given, else by port: 8001
  `H200 via tunnel`, 8011 `Mac (vllm-metal)`, else `host:port`. Lightning: `local BF16 (:8000)`
  for 127.0.0.1:8000, `hosted API` for integrate.api.nvidia.com, else `host:port`.
- **Grounder check:** `POST /api/run` answers 400 when `options.grounder` isn't in a known
  `served` list, e.g. Grounder `base` against the Mac's grounder-only server.
- **`console.json` `endpoints`:** the snapshot for that run's options, taken at run start.

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
| `result` | RunResult `{run_id, success_claimed, summary, steps, trace_dir, reason?, question?}` (only when the agent returns; `reason` is the done event's, `question` is set for `needs_input`) |
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
- **Request check** (`cfg.clarify` on, the console's default; off: no `clarify` event, as
  before): a step-0 `clarify` event right after `start`, before the browser opens, in
  `agent/trace.py`'s shape with `model: "mock (no Lightning)"`, `prompt_version: "clarify-v4
  (mock)"`, `prompt_sha: "mock"` and no latency (no model ran).
  - `Allors towards my bill.` and `Pay something toward my bill` (case and a final period
    aside) ask "What would you like to do with your bill?" / "How much would you like to pay
    toward your bill?", then `done(needs_input)` with `question`, `request` and `steps: 0`. No
    browser opens.
  - A follow-up of one of them answered `fifty dollars` / `$50` is restated as `Pay $50 toward my
    bill.`; anything else proceeds with the request trimmed. The restatement, not the user's
    words, picks the script, so that follow-up replays pay-50 (it starts on `/`, like free text).
  - A request already answered twice skips the check (`skipped`), like `agent/clarify.py`.
  - A follow-up's `start` event carries `parent_run_id`.
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

- **Warm microphone:** the mic stays open after the first take, so the start of a
  sentence isn't clipped.
  - The first press says "Starting the microphone…" and records from the first audio
    frame.
  - Later presses start with about 400 ms of pre-roll from just before the key went
    down, and call no `getUserMedia`.
  - "Listening…" shows only once frames arrive.
  - While warm, the page shows "Microphone ready (releases after 5 min idle)" with a
    Turn off button.
  - It also releases when the page is hidden. On macOS a fully covered window counts
    as hidden. Record headless (no agent window); with --headed, keep that window from covering
    the console completely.

- **Recording:** click **Speak**, or hold Right Option (`AltRight`, configurable in
  Options). Space-hold works when focus isn't on a field or control.
- **Capture:** `voice.js` records raw PCM through an AudioWorklet (`pcm-worklet.js`,
  with a ScriptProcessor fallback). `wav.js` resamples to 16 kHz if the browser
  refuses a 16 kHz context, and encodes PCM16 WAV.
- **Transcription:** the WAV is POSTed to `/api/transcribe`.
- **After transcription:** the transcript fills the request box and the picker shows the
  task it [matches](#matching-a-request-to-a-sandbox-task), if any (an exact match shows the
  task's own text; a fuzzy one keeps what was said). Choosing a task or Free text cancels the
  auto-run. The log shows `Heard (engine, N ms): …`.
- **Auto-run** (Options, "After voice input", saved in localStorage): run in 1.5 s (the
  default), run in 5 s, or don't auto-run. Esc or Cancel stops the countdown. Nothing
  runs by itself when the transcript is over 500 characters, when you edited the request
  while it was transcribing (your text is kept), or with the fixed-text stub.
- **Stop phrases:** during a run, a transcript that is only a stop phrase ("stop", "stop
  it", "cancel that", "please stop", "ok stop now", ...) stops it. The list is closed and
  anchored, so "Stop sending me promotions." is a request, not a stop. With nothing
  running, a stop phrase starts nothing.
- **Engines** (`asr.py`):
  - `parakeet` is the default: NVIDIA Parakeet TDT 0.6B v3 through the NeMo-Speech.cpp
    sidecar at `CONSOLE_ASR_URL` (default `http://127.0.0.1:8096`, `scripts/run_asr.sh`).
    It counts toward the 33B cap: 30 + 2 + 0.6 = 32.6B.
  - `stub` is opt-in only (`--asr stub` or `CONSOLE_ASR=stub`): fixed text
    (`CONSOLE_ASR_STUB_TEXT`), labelled `stub (fixed text)`, and `/api/config` says
    `asr.fixed_text: true`, so the page never runs its text by itself. For UI testing.
- **Re-check:** until the sidecar's `/health` answers, `/api/config` reports voice input
  as unavailable and the mic shows as off. The page asks again every 4 s and turns voice
  on by itself when it comes up (a missing microphone re-checks too). `demo_up.sh`
  defaults to Parakeet and only warns when the sidecar is down.
- **Page API:** `console.js` exposes `window.AcmeConsole` (`ready`, `config()`,
  `isRunning()`, `setRequest(text)`, `run()`, `stop()`, `matchTask(text)` → task or null,
  `matchTaskDetailed(text)`, `explainMatch(text)`) and fires `acme:run-start` /
  `acme:run-end`, which `voice.js` uses.
- **Tests:** `test_asr.py` (validation, engines, fake sidecar), `test_voice.py` (the
  WAV encoder, the endpoint, the fake-mic page flow, and a static scan for forbidden
  APIs) and `test_voice_ui.py` (push-to-talk and the countdown, with Chromium's fake
  media flags).

## Logs page

<http://127.0.0.1:8095/logs> (`static/logs.html`, `logs.js`, `logs.css`; backend
`logs.py`) shows every run, event by event: what each model saw, decided and did. It is
read-only and never reads sandbox state; the verdict is the one the console recorded.

| Route | Returns |
| --- | --- |
| `GET /logs` | the page |
| `GET /api/logs/runs?limit=200` | `{runs: [RunSummary, ...], total}`, newest first (limit 1..1000) |
| `GET /api/logs/runs/{run_id}` | `{summary, console, events}` (pretty-printed) |
| `GET /api/logs/runs/{run_id}/download` | the same bytes as an attachment, `run-<run_id>.json` |

- **Sources:** `<root>/<run_id>/events.jsonl` (the agent's events) and `console.json`, for
  every root: `--runs-dir`, the agent's `cfg.run_dir_root`, and every root a run used.
- **`console.json`** (written by `runner.py` into the trace dir as soon as it is known,
  status `running`, then atomically at the end, before `end`, and when a restated match is
  accepted): `{console_run_id, agent, request, task_id, task_match, options, sid, start_url, voice,
  endpoints, parent_run_id, clarifies, status, result, verdict, started_ts, ended_ts}`.
  `options.clarify` is the run's request check (`on` / `off`).
- **RunSummary:** run and console ids, agent (`mock` when any payload says so), times,
  request, task, `task_match` and `task_line` ("Checked as task: pay-50 (matched from your
  request)" or "(picked from the list)", the banner's line, shown in the summary card's Task
  row; no parenthesis for a `console.json` from before `task_match`), models (Lightning, grounder, summary, ASR), status, steps, claimed and
  verified success, total time, p50 per stage (voice, summary, plan, ground, act), counts
  (events, errors, invalid tool calls, unreadable lines), `n_stuck_warnings`,
  `n_stuck_stops`, `endpoints`, voice, prompt version and SHAs, and for the request check
  `clarify` (`{action, question, normalized_request, model, prompt_version, prompt_sha, usage,
  latency_ms, error, skipped}` or `null`), `question` (a `needs_input` run's), `parent_run_id`
  and `child_run_ids`. A field the trace doesn't have is `null` and shows as "—".
- **Status:** the done event's reason (`done`, `max_steps`, `stuck`, `needs_input`, `stopped`,
  `error`), else `console.json` (`running` only while this console is running it), else `unknown`.
- **Request check** (`agent/clarify.py`): a `clarify` row at step 0 ("Request check: unclear,
  asked the user “…”" or "clear, went ahead · restated as “…”"; failures and skips say so),
  with its model, latency, prompt version and SHA, tokens and tool call in the details. A run
  that asked has the Needs input chip (also in the status filter) and a done row with the
  question; the summary card has a Request check row.
- **Follow-ups:** `parent_run_id` comes from `console.json`, else the start event;
  `child_run_ids` are the runs whose parent is this run (oldest first, from every run found).
  The summary card shows "Follow-up of `<id>` · your answer: “…”" and "Answered by `<id>`"
  links that open the run in place; the table marks rows "follow-up" / "answered".
- **Stuck** (`agent/progress.py`): the agent stopped a run that made no progress. The
  done summary is its question to the user ("I'm stuck: ..."), and `success_claimed` is
  false. The page shows:
  - a Stuck status chip, and ⚠N in the table for N stuck warnings in a run that went on;
  - action rows that start with `Stuck warning: <reason>` or `Stopped as stuck: <reason>`;
  - the done row with the agent's question;
  - a "Progress check" line in the summary card.

  The counts come from the done event's stats, else from the action verdicts, else `null`
  (traces from before the progress check).
- **Endpoints** (the "Where it ran" column and summary line): the VLM label, host:port and
  summary model, and the Lightning label. They come from `console.json`'s snapshot at run
  start (`source: "console"`, with the served names), else from the start event's config
  (`source: "trace"`: the label follows the port and nobody checked what was served).
- **Deep links:** `/logs?run=<id>&kind=plan,grounding&model=base&q=text&status=stuck`.
  `run` opens a run, `kind` / `model` / `q` filter its events, and `status` filters the runs
  table. Changing a filter updates the URL, and Back and Forward work.
- **Keys:**
  - Runs table: one tab stop. ↑↓ move, PageUp/PageDown move 10, Home/End, Enter or Space
    opens. ←→ scroll a narrow table sideways.
  - Event log: one tab stop. ↑↓, Home/End, Enter shows or hides details, → opens, ← closes.
  - Search box: Esc clears it.
  - Screenshot dialog: Enter on a thumbnail opens it, Esc closes it and focus goes back.
- **Copy JSON / Download JSON:** byte-identical (one serializer, `detail_json`). Without a
  clipboard the JSON is shown, selected, in a text box.
- **Other:** a 3 s auto-refresh runs only while the selected run is still going, and its
  end is announced once. High contrast is shared with the console page. Everything is
  rendered as text, never as HTML.
- **Tests:** `test_logs.py` (summaries, stuck counts, endpoints, the HTTP API),
  `test_logs_ui.py` and `test_logs_e2e.py` (Chromium: keyboard, filters, copy and
  download, the real traces, the stuck trace, contrast in both themes).
  - `fixtures/real_traces/` holds two real runs.
  - `fixtures/stuck_trace/` is a synthetic run in `agent/core.py`'s exact shapes. Its
    verdicts, reason and question come from `ProgressTracker`.
  - `test_logs.clarify_traces()` is a synthetic needs_input run and its follow-up, with clarify
    payloads from `agent.clarify.Clarification` itself (`test_logs_e2e.py` renders them).

## Matching a request to a sandbox task

A run is checked against the portal's state only when it runs as one of the 12 sandbox
tasks (`task_id`: its start page, initial state and check). Picking a task from the list does
that; so does a request that matches one. `console.js` `createTaskMatcher` does the matching,
in the page, for typed text and voice transcripts alike.

- **Exact:** the request is the task's text once case, accents, punctuation and how an amount
  is written are ignored (`requestKey`; `console/app.py` `request_key` is the same function).
  The box then shows the task's own text (the same words).
- **Fuzzy:** both texts become content tokens: filler dropped ("please", "can you", "I'd like
  to", "um", "okay", ...), synonyms folded ("towards" → "toward", "texts" → "sms", "font" →
  "text", "enable" → "turn on", "stop" → "off"), numbers normalized ("$50", "50 dollars" and
  "fifty dollars" are all `50`), light stemming. A task is a candidate only if:
  - the request has all of its anchor words (`TASK_ANCHORS`: e.g. pay-50 needs "pay" and
    `50`, high-contrast needs "high", "contrast" and "on");
  - nothing contradicts it: another number, month, district or channel, the opposite on/off,
    a negated anchor ("don't ..."), a schedule word ("every month"), another task's key word,
    or a second request ("... and reduce motion").

  Score: weighted overlap (anchors 2.5, a task word the request skips counts half). The best
  candidate wins if it scores ≥ 0.5 and beats the next candidate by ≥ 0.15. The box keeps the
  user's words: the agent sees what they said, and only `task_id` changes.
- **Anything else is free text:** no task and no verdict. The matcher is conservative on purpose,
  because a wrong task gives a wrong verdict. "Pay my bill", "Download my August statement",
  "Turn off high contrast", "Report an outage in Hillcrest" and "Turn on SMS for billing" all
  stay free text.
- **The picker shows the match** before the run, so it can be undone with Free text (that task
  isn't offered again until the box is emptied). Voice input updates the picker at once (and
  changing it cancels the auto-run). Typed text updates it when the box loses focus or on Run;
  while typing, a note says what it matches.
- **Recorded:** `POST /api/run` sends `task_match` `{method: picked|fuzzy, score, matched_from}`.
  For an exact match the page leaves it out and the backend records `{method: "exact", score: 1,
  matched_from: request}` itself. Both the banner (above the verdict) and the Logs page say
  "Checked as task: pay-50 (matched from your request)" or "(picked from the list)".
- **Debugging:** `AcmeConsole.explainMatch(text)` returns the tokens and, for every task, its
  score and what ruled it out. The table in `test_task_match.py` (run with `-s`) prints every case.
- **The mock agent** only has scripts for the tasks' exact texts. A fuzzy-matched run under the
  mock gets "no scripted plan" and a ✗ verdict; the real agent reads the user's words.
- **During a run:** a free-text run can still become a task when the agent's restatement matches
  one (`method: "restated"`, [below](#a-task-from-the-agents-restatement)).

## Request check and follow-ups

The request check is `agent/clarify.py`: before the browser opens, one short Lightning call
decides whether a request is clear enough to act on. It is on by default in the console
(`--clarify on`, Options) and off everywhere else (`configs/agent.yaml`, the eval harness).

- **Per run:** `options.clarify` (`on` / `off`) becomes the agent's `load_config` override
  `{"clarify": value}` (the mock reads the same option); `console.json` records it in `options`.
  With it off there is no `clarify` event at all, and everything else works as before.
- **The event:** `clarify` at step 0, right after `start`, before any observation:
  `{action: proceed|ask, request, question? (ask), normalized_request? (proceed), model,
  prompt_version, prompt_sha, usage?, latency_ms?, error?, skipped?, tool_call?}`, with
  `timings_ms.clarify`. `normalized_request` is shown, never planned with: the agent gets the
  user's own words. `error`: the call failed, so the run went ahead anyway.
- **An ask** ends the run at once: done reason `needs_input`, `steps: 0`, `success_claimed:
  false`, `summary` = `question`, plus `question` and `request`. The console's status is `done`;
  a verdict is computed only if the run had a task.

### Follow-up runs

The user's answer starts a new run: `POST /api/run` with `clarifies: {parent_run_id, answer}`.

- `parent_run_id` is the **agent** run id of the run that asked (its trace dir name). The
  backend looks for it in its own runs, then in the trace folders, and requires its done reason
  to be `needs_input`.
- The request is `agent.clarify.follow_up_request(<parent done.payload.request>, answer)`, i.e.
  `previous + " (clarified: " + answer + ")"`, with the answer's whitespace collapsed (1–300
  characters). The page may send its preview as `request` (to run its matcher on it); it must be
  the same text, whitespace aside, or the run is refused with 400 and the backend's text.
- The response's `request` is the built text. The agent gets `run_task(...,
  parent_run_id=<parent>)` (only for follow-ups), and its start event records it.
- `console.json` gets `parent_run_id` and `clarifies: {parent_run_id, answer,
  previous_request}`; the Logs page links the two runs both ways.
- A follow-up may carry a `task_id` and `task_match` like any run (the page matched the preview).

### A task from the agent's restatement

A free-text run has no task, so no verdict. When its `clarify` event proceeds with a
`normalized_request` that matches a task (`matchTaskDetailed` in the page), and the task's
distinguishing words are already in the user's own words, the page reports it.

The distinguishing words are the task's number, month, district and channel tokens, plus
required words no other task has (`matchSupport`). The user's own words are the request plus
any clarification answer. This guard stops a self-fulfilling ✓: "Allors towards my bill.
(clarified: fifty dollars)" supports pay-50, but a bare "Allors towards my bill." restated as
"Pay $50 toward my bill." doesn't, so no task is picked and the line says why. When the
guard passes, the page sends:
`POST /api/runs/{console_run_id}/task_match {task_id, method: "restated", score, matched_from:
<normalized_request>}`. The backend accepts it only if:

- the run has no task yet and hasn't ended (the verdict isn't decided);
- `matched_from` is the run's restatement (whitespace aside);
- the task starts where the run started: the same page (free text starts on `/`) and the
  default state (no `initial_state_overrides`), since the run's sandbox session was reset without
  a task.

The verdict at the end then checks that task. The banner and the Logs page say so, naming the
text that chose it: `Checked as task: pay-50 (matched from the agent's restatement: "Pay $50
toward my bill.")`. The agent never sees any of it.

## The no-DOM rule

Agent-facing code never reads the DOM or the app state: neither `mock_agent.py` nor
T2's agent. `calibrate_mock.py` is the **only** place in `console/` that uses
selectors (`data-testid`, `elementFromPoint`, `window.AcmeTestHooks`). It's a dev tool,
like the sandbox's scripted oracle, and its output contains no selectors. The console
backend reads `/api/state` only to show the verdict, after the run.
