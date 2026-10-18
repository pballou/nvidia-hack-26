# T1: Sandbox target app + GUI Assist console

**Owner:** Patrick (stream C). **Time box:** ~3 h. Sandbox tasks usable by ~4pm
Saturday; console demo-ready by ~10pm Saturday.

**Read first:** `AGENTS.md`, `docs/plan.md` ("Decisions so far"),
`docs/tasks/contracts.md`. The contracts there are binding: if you change one, update
that file in the same commit.

## Why this exists

GUI Assist is an accessibility agent: a user says what they want, and the agent does
it in an app **from pixels alone**. We need two things:

1. **A sandbox target app** that the agent operates in our NeMo Gym `web_agent` env
   and in the demo. It must give verifiable per-session state, and it must include UI
   that defeats DOM-based agents.
2. **The GUI Assist console:** the localhost app a judge or user sees. Type a request,
   watch the agent work step by step, see where it clicked and why.

## Part A: sandbox target app (`sandbox/`), done first

**Concept:** "Acme Utilities" customer portal, a realistic account app where people
with low vision or motor impairments often struggle.

**Stack:**
- Python 3.13, FastAPI, Jinja2 templates, vanilla JS and CSS.
- No build step, no CDNs, no external fonts or images: everything is served locally.
- Deterministic rendering: no animations or transitions, a fixed "today" date, fixed
  seeded data, and `prefers-reduced-motion` respected.
- Run with `uvicorn sandbox.app:app --port 8090`.

**Sessions and state:** follow contracts.md §1 exactly.
- `?sid=` sets the cookie, and state is in-memory per sid.
- `POST /api/reset` applies `initial_state_overrides` from the task.
- `GET /api/state` returns everything that verifiers check.
- Two concurrent sids must be fully isolated.

**Pages and widgets** (all usable at 1280×800):
- **Global chrome:**
  - A top bar with **icon-only** buttons: bell (notifications dropdown), gear
    (settings), question mark (help), avatar (profile menu with Sign out).
  - A collapsible hamburger side nav.
  - **Some icon buttons deliberately have no text and no aria-label.** That's the
    accessibility gap the vision agent closes.
- **Dashboard:** balance card, quick actions, a recent-activity list.
- **Billing:**
  - A statements table with an icon-only download button per row. Downloads just
    record `billing.downloaded` in state; no real file.
  - A 2-step "Pay bill" flow (amount, then a confirm dialog).
  - Autopay toggle.
- **Usage:** a monthly bar chart drawn on **`<canvas>`**. Clicking a bar shows that
  month's details panel and sets `usage.selected_month`. There's no DOM element per
  bar.
- **Outage map:** a **`<canvas>`** map with ~6 labeled districts. Clicking a district
  opens a "Report outage" modal, and submitting appends to `outages.reports`.
- **Messages:**
  - An inbox list with star toggles and an icon-only "mark all read".
  - Opening a message marks it read.
  - A compose flow (to a fixed contact list, subject, body) that appends to
    `messages.sent`.
- **Settings:**
  - Tabs: Profile, Notifications, Accessibility.
  - Notifications: per-category channel toggles (email/SMS/push).
  - Accessibility: high contrast (it actually changes the CSS theme), text size as
    an S/M/L segmented control (it actually rescales the UI), reduce motion.
- **Help:** FAQ accordions and a contact form.
- **Every state change is visible on screen** (toasts, toggles, highlighted rows), so
  the agent's "screen changed?" signal works.
- At least 2 tasks need a scroll.

**Tasks** (`sandbox/tasks.json`, schema in contracts.md §1): **12 tasks**, phrased the
way a real user would say them.
- **Mix:** 4 easy (1 step), 5 medium (2–3 steps), 3 hard (4+ steps). At least 3 hit
  icon-only controls and at least 2 need the canvas.
- **Examples:**
  - "Make the text bigger, I can't read this" (text size L).
  - "Turn on high contrast."
  - "Download my September statement."
  - "Report an outage in the Riverside district."
  - "Which month did I use the most power? Open it." (canvas: select the max month)
  - "Turn off SMS for billing alerts."
  - "Star the message from the landlord."
  - "Pay $50 toward my bill."
  - "Mark all my messages as read."
  - "Send a message to support saying my meter is broken."
- **Checks** must be strict enough that doing the wrong thing fails. For example,
  downloading August instead of September fails.

**`sandbox/checks.py`:** the `evaluate(state, check)` from contracts.md. Pure
functions, unit-tested.

**Validation (required):**
- **Scripted oracle:** `sandbox/tests/test_oracle.py` uses Playwright with **DOM
  selectors, which only the oracle is allowed**. It solves every task from a fresh
  reset, then asserts `evaluate()` passes. It also asserts that a deliberately wrong
  action fails the check. This proves all 12 tasks are solvable and verifiable.
- **Unit tests:** state transitions, reset, session isolation, `checks.py` operators.
- **Determinism:** take two screenshots of each page in fresh sessions at 1280×800,
  `device_scale_factor=1`; they must be pixel-identical. Commit a small gallery at
  `sandbox/screenshots/*.png` for the README and the video.
- **Network:** a Playwright route check that no request leaves `127.0.0.1`.

## Part B: GUI Assist console (`console/`), the user-facing app

**Stack:** FastAPI backend plus one static HTML/JS page. No build step. Port **8095**.

**Layout:**
- **Left panel:**
  - A big request box (text input, Enter to run).
  - A task picker that pre-fills one of the 12 sandbox requests, plus free text.
  - Run and Stop buttons.
  - Toggles: grounder `base` / `grounder`, thinking `off` / `retry` / `on`, Lightning
    hosted / local.
  - A live step log: for each step, Lightning's tool call in plain words ("Clicking:
    the gear icon in the top bar"), short reasoning when present, and latency.
- **Right panel:**
  - The latest screenshot with the **click overlay**: crosshair and target text.
  - Step thumbnails you can click to replay.
- **Final banner:**
  - The agent's own done-summary.
  - For sandbox tasks, a verified ✓/✗ from `evaluate()` on `/api/state`, labeled
    "verified by sandbox state, not seen by the agent".

**Backend:**
- `POST /api/run {request, task_id?, options}` starts `agent.core.run_task(...)`
  (contracts.md §5) and returns `run_id`.
- `GET /api/runs/{run_id}/events` streams `StepEvent`s as **SSE**.
- `POST /api/runs/{run_id}/stop`.
- Images are served from `runs/<run_id>/`.

**Build against a mock first** so this isn't blocked on T2:
- `console/mock_agent.py` emits a scripted sequence of StepEvents with real sandbox
  screenshots.
- Switch with `--agent mock|real`. The real one imports T2's `agent.core`.

**Demo polish:**
- Big readable fonts. High-contrast option for the console itself.
- Its keyboard flow works without a mouse (accessibility credibility).
- `--headed` option to show the Playwright browser window side by side for the video.

**Voice input (approved 2026-10-17):** local NVIDIA Parakeet TDT 0.6B behind
`POST /api/transcribe`, with a mic button and push-to-talk (Right Option, or Space
outside text inputs). 16 kHz WAV is captured in the browser. Never use the cloud
Web Speech API. Cap: 30 + 2 + 0.6 = 32.6B.

## Run scripts and README

- `scripts/run_sandbox.sh`, `scripts/run_console.sh`, plus a top-level
  `scripts/demo_up.sh`. It starts the sandbox and console, checks that the VLM
  (`:8001`) and Lightning (hosted or `:8000`) respond, and prints the URL.
- **Fill README "Quick start"** with copy-paste commands: setup, start the sandbox,
  open the tunnel or start the local models (link `docs/local-mac.md`), start the
  console, run a task.

## Acceptance criteria

- [ ] The oracle test passes all 12 tasks from fresh resets, and the wrong-action
      tests fail as expected.
- [ ] Unit tests pass, session isolation is proven, determinism and no-egress checks
      pass.
- [ ] `tasks.json` and `checks.py` match contracts.md exactly. T2's Gym env can load
      them unchanged.
- [ ] The console runs end to end with `--agent mock`. With `--agent real`, it runs
      one easy sandbox task live once T2 lands.
- [ ] The README Quick start works on a fresh clone (Patrick's Mac).
- [ ] No DOM or state access anywhere in agent-facing code paths. Only
      `checks.py`/verifiers/tests touch state.

## Out of scope

Training, Charlie's data code, and Kyle's serving. No auth, no database, no
deployment.
