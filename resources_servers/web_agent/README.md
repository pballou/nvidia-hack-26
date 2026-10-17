# `web_agent`: NeMo Gym env for the vision-only accessibility agent

Nemotron 3.5 Lightning (the policy) gets a user request and a text description of the screen.
It acts only through five tools. A `click` target description goes to the grounding VLM (served
names `base` / `grounder`), which returns a point, and Playwright clicks those pixels. The reward
is the task's check on the sandbox's state (`sandbox/checks.py`): 1.0 if it passes, else 0.0.
This env gives the base-vs-tuned grounder comparison inside a real agent loop (docs/plan.md M4).

The tools and observations are the interactive agent's own (`agent/`). Every tool route calls
`agent.session.AgentSession.execute`, so the console agent and this env behave the same way.

| File | What it is |
| --- | --- |
| `app.py` | The resources server (seed, tools, verify, metrics) |
| `make_tasks.py` | `sandbox/tasks.json` → Gym rows (`data/example.jsonl` = 4 easy + 1 medium task) |
| `configs/web_agent.yaml` | Hosted Lightning (build.nvidia.com) |
| `configs/web_agent_local.yaml` | Self-hosted Lightning on `:8000`, served name `lightning` |
| `tests/test_app.py` | Server tests (fakes, plus one real-Chromium episode) |

## How to run

Run everything from the repo root.

- **Eval (base vs grounder, then `gym eval compare`):** `scripts/agent_eval.sh`. The script lists
  its knobs at the top. It needs the sandbox (`:8090`), the VLM (`vlm.base_url` in
  `configs/agent.yaml`) and Lightning.
  - Reported runs need rows with real VLM start screens and the official BF16 Lightning: the
    served root must be exactly `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (or an
    absolute local path ending in that repo name). Each `*_run_info.txt` records `reportable=yes|no`.
  - Runs are sequential (`--concurrency 1`). Local Lightning allows 2 concurrent requests, and
    T1's console shares it.
  - Thinking is off by default. `THINKING=on` turns it on, raises `max_output_tokens` to
    `sampling.thinking_max_tokens` (4096; reasoning counts against it) and writes
    `agent_<grounder>_thinking.*`, so it never pairs with a thinking-off run in `gym eval compare`.
  - The script pins `NEMO_GYM_EXTRA_ROOTS` to its own checkout. Gym searches that variable before
    the cwd for configs, `--input`, server dirs and imports, and SETUP.md exports it per checkout.
    If you run `gym` by hand, point it at this checkout too.
  - The base-vs-tuned run (12 tasks × 3, `grounder` = `sft-template`) finished Sat 19:30:
    verified 18/36 → 22/36 ([docs/status.md](../../docs/status.md)). Sunday reruns only if
    the overnight GRPO adapter wins.
- **Server tests:**
  - `.venv-gym/bin/python -m pytest resources_servers/web_agent/tests -q` (fast; no GPU, no sandbox)
  - `.venv-gym/bin/gym env test --resources-server web_agent` (the same tests in the server venv
    Gym builds)
- **Servers by hand:** `.venv-gym/bin/gym env start --config resources_servers/web_agent/configs/web_agent_local.yaml`,
  then `gym eval run --no-serve --agent web_agent_simple_agent --input <rows.jsonl> --output results/<name>.jsonl`.

**Switching served names is config-only.** The server reads `configs/agent.yaml` through
`agent/config.py` (yaml < `AGENT_*` env vars). It then applies the non-null fields of its Gym
config block on top:
- `grounder_model`, `summary_model`, `model_family`
- `vlm_base_url`, `sandbox_url`, `headless`

`configs/web_agent.yaml` sets those fields from `${oc.env:AGENT_GROUNDER_MODEL,null}`-style
variables. When a variable is unset the field is null, so `configs/agent.yaml` decides, as it does
for the interactive agent. `scripts/agent_eval.sh` passes
`++web_agent.resources_servers.web_agent.grounder_model=…` explicitly. The values are fixed at
`gym env start`, so restart the servers to switch. Every rollout records `grounder_model` and
`summary_model`.

Gym silently drops config keys a server doesn't declare, and a typo such as `grounder_modle`
would quietly keep `base`. So `WebAgentConfig` collects unknown keys, and the server refuses to
start if there are any.

## Initial observation (the approach we picked)

Gym's legacy `/run` path (`simple_agent`) uses only the cookies from `seed_session`'s response.
Its body never reaches the model. So the start screen has to be in the row:

1. **`make_tasks.py` precomputes it** with the interactive agent's code path: fresh sid →
   `POST /api/reset` → `BrowserSession.goto(start_url)` → `AgentSession.reset()`, which settles,
   takes a screenshot and gets one VLM summary.
2. **The row stores it in three places:**
   - the row's first user message, `prompts.initial_user_message(request, obs.render())`;
   - `initial_observation` (the `Observation` dict);
   - `initial_screen_sha`: `make_tasks.screen_sha(img)`, the sha256 of the start screenshot as
     160×100 grayscale (PIL `convert("L")`, `BOX` resize, raw bytes). `app.py` imports this same
     function.
3. **`seed_session`** does the following:
   - resets the sandbox for a fresh sid, opens a context and loads the start page;
   - settles once more and takes the start screenshot again (`goto()` then `settle()`, like
     `reset()`, so it hashes the same frame as the row);
   - primes the `AgentSession` without a VLM call: `last_screenshot` is the live screenshot,
     and `last_observation` comes from the row;
   - records `initial_obs_stale`, which is true when the live fingerprint differs from the row's.
4. **Staleness is a flag, not a mask.** The first click still grounds on the live screenshot.
   Fonts differ between machines, so generate the rows on the same stack that runs the eval.
5. **Placeholder rows are plumbing only.** `make_tasks.py --placeholder-observation` needs no
   VLM. Its rows have the summary "(screen description not generated: VLM unavailable at
   generation time)", `initial_screen_sha: null` (so no stale check) and
   `initial_observation_source: "placeholder"`. `scripts/agent_eval.sh` refuses to run them unless
   `FORCE=1`, and they are never reported.

The start-screen text is model output, so `make_tasks.py` writes a `<rows>.meta.json` with the
exact command, config and prompt hashes (AGENTS.md open-data rule).

## Tools, `done` and `max_steps: 12`

- **Routes.** `POST /click`, `/type_text`, `/press_key`, `/scroll` and `/done` take a loose JSON
  body (never a typed model, so the model never sees a raw 422).
  - Each route calls `AgentSession.execute(name, body)` and **always** answers
    200 + `{ok, observation, error?, click?}`.
  - The body is `json.dumps(payload, ensure_ascii=False)`, the same bytes the interactive
    planner puts in its tool message. FastAPI's `JSONResponse` would use compact separators.
  - A catch-all `POST /{tool_name}` is registered last. It turns an invented tool name into the
    same 200 `ok:false` answer, listing the real tool signatures.
  - Each call is bounded by `tool_timeout_s` (240 s), because Gym's HTTP client has no timeout.
    A timeout returns `ok:false` and masks the rollout.
- **`tool_choice` stays unset (`auto`) in the rows.** `simple_agent` ends an episode only on a
  reply without a tool call, so `required` would run every task to `max_steps`.
  `make_tasks.py --check` rejects rows that set it. (The interactive planner uses `required`.)
- **Calls with bad JSON never reach the server.** `simple_agent` answers them itself with
  `{"error": "Invalid tool call arguments: ..."}`. So `steps` and `n_invalid_calls` here miss
  them; `scripts/agent_metrics.py` computes validity from `response.output` instead.
- **Schema checks.** `agent.session.validate_args` checks wrong or missing argument names, wrong
  types and bad enum values. On a mismatch nothing runs, and the corrective error says exactly
  what to send. This matters because vllm-metal doesn't enforce the schema.
- **`done` ("first finish wins").** `done` records `{summary, success}` and returns `DONE_NOTE`,
  which tells the model to reply without a tool. The server then snapshots `GET /api/state`. Any
  later tool call is refused with `ALREADY_DONE` and changes nothing. `verify` scores the
  snapshot if `done` was called, otherwise the live state.
- **Why `max_steps: 12`.** `simple_agent` doesn't stop on `done`, only when the model replies
  without a tool call. Its `max_steps` counts every model call, but it runs the tool calls of
  call N before it checks `step >= max_steps`. So 12 = the interactive agent's 12 Lightning calls
  (`configs/agent.yaml` `max_steps`), a `done` on call 12 included. `verify` scores the `done`
  snapshot, so the final text reply after `done` is never needed for the reward. Both paths
  give an episode the same action budget.
- **Hallucinated reserved names.** A model that "calls" `seed_session`, `close_session` or
  `verify` can't hurt its episode:
  - a legacy seed is ignored if it has no `responses_create_params`, or if its session already
    has an episode (real `/run` seeds arrive without a cookie, so they always get a fresh session);
  - an empty close body closes nothing (unlike interactive_browser);
  - a verify body counts only if it has simple_agent's shape (`body.model_dump() | {response}`):
    `response` and `responses_create_params` objects and the episode's `task_id`, which the model
    never sees. Anything else scores nothing, shows no state and releases nothing.

## `verify` and masking

`verify` never raises and never returns non-2xx. It parses the body itself, and every
`WebAgentVerifyRequest` field is optional.

- **Scoring.** It reads `GET /api/state?sid=` (httpx, timeout, `trust_env=False`), checks the
  session, then runs `sandbox.checks.evaluate(state, row.check)`.
- **Release.** It releases the browser context and the VLM client in `finally`, because the
  legacy path never calls `/close_session`.
- **Masking.** Any failure returns `reward=0, mask_sample=True`, a `failure_reason` and a
  `failure_kind`. Masked rollouts drop out of every quality metric and show up only in
  `coverage/*`.

| `failure_kind` | When |
| --- | --- |
| `provider_unavailable` | Sandbox or browser down at seed or verify; or any VLM infrastructure error, browser crash or hang, or tool timeout during the episode |
| `session_lost` | No session at verify, or the sandbox state belongs to another session (see below) |
| `web_agent:bad_row` | The row lacks `task_id`, `start_path` (starting with `/`) or a `check` object |
| `web_agent:reset_rejected` | The sandbox answered `/api/reset` with a 4xx (e.g. a task_id unknown to `tasks.json`) |
| `web_agent:bad_request` | The `/verify` body isn't simple_agent's (no `response` or `responses_create_params` object, or another `task_id`) |
| `verifier_error` | Anything else |

**VLM outages are masked, not scored.** `agent/vlm.py` never raises. Transport, timeout and HTTP
errors come back as error strings starting with "VLM" (`VLM unreachable`, `VLM timed out`,
`VLM error <code>`, `VLM call failed`). `AgentSession` turns them into `ok:false` observations.
- Each episode wraps its VLM client and counts these errors (`n_vlm_infra_errors`), plus tool
  timeouts.
- **A dead or hung browser counts too.** That means a closed target, a renderer crash or a
  Playwright screenshot timeout. The server matches Playwright's error text
  (`TargetClosedError: ...`, or `<Exc>: <Class.method>: ...` containing `Target crashed` or
  `Timeout <n>ms exceeded`). It looks in the tool error and in AgentSession's
  `(screen unavailable: ...)` summary, which is built from an exception, never from VLM text. The
  anchoring matters because a grounding miss quotes the model's target text.
- `verify` masks every affected rollout, whatever its reward. Masking only the failures would
  inflate success during an outage, and an unmasked outage would bias base vs grounder.
- Model-quality failures are the policy's to own and stay scored: `could not parse a point`, or
  an empty screen description.

**Session check.** The real sandbox writes `meta.sid` and `meta.task_id` at `/api/reset`. It
recreates a default state for an unknown sid (after a restart or an LRU eviction), which would
score a false 0. So `verify` masks when either value differs. The test-only
`agent/testing/fake_sandbox.py` doesn't expose `meta`, so there the check is skipped and
`check_explanation` says so.

**Sessions.** The sandbox sid is `gym-` + Gym's per-rollout session id, never the task id
(`--num-repeats` runs the same task concurrently). The server supports both the legacy seed
body (the row) and the typed `ResourcesSeedSessionRequest` (`task_data`), and passes Gym's
session-conformance check. It runs one Chromium per process (started by the first seed, closed in
the FastAPI lifespan) and one context per session, with `num_workers: 1`. A `/run` that fails
before `verify` (e.g. a model-call error) would leak its context, so every seed closes episodes
older than `episode_ttl_s` (1 h).

## Metrics (VerifyResponse fields)

Gym auto-profiles every int and bool (`mean/steps`, `mean/claimed_success`, …).

| Field | Meaning |
| --- | --- |
| `task_id`, `tags` | Task identity; `compute_metrics` adds `tag/<tag>/mean_reward` and `tag/<tag>/rollouts` |
| `steps` | Tool calls that reached the server (valid or not) |
| `n_clicks` | Pixel clicks (`click`, and `type_text` with a target) |
| `n_invalid_calls` | Calls rejected by the schema (unknown tool, wrong/missing args, bad enum). Validity = 1 − n_invalid_calls / steps |
| `n_failed_actions` | Valid calls that returned `ok:false` (grounding or browser failure) |
| `n_calls_after_done` | Valid calls refused with `ALREADY_DONE` |
| `n_vlm_infra_errors` | VLM transport/timeout/HTTP errors, browser crashes or hangs, and tool timeouts (> 0 masks the rollout) |
| `summary_ms`, `ground_ms`, `act_ms`, `settle_ms`, `screenshot_ms` | Per-episode stage totals (Gym reports mean/median) |
| `latency_ms` | `{stage: [ms per call]}` for p50/p90 in `scripts/agent_metrics.py` (Gym skips dicts) |
| `done_called`, `claimed_success` | Whether `done` ran, and its `success` argument |
| `initial_obs_stale`, `initial_observation_source` | Start-screen staleness flag; `vlm:<summary_model>` or `placeholder` |
| `state_source`, `check_explanation` | `done_snapshot` or `live`; the checker's explanation |
| `grounder_model`, `summary_model`, `prompt_version` | Provenance of the rollout |

## The oracle is never used here

`agent/testing/oracle_grounder.py` reads the DOM, and it exists only to test loop mechanics.
- **The server never imports it.** Its grounder and summarizer come only from
  `app.make_vlm(cfg, browser_session)`, which returns `agent.vlm.VLMClient`.
- **`use_oracle_grounder: true` in the agent config** makes the server refuse to start.
- **Only pytest swaps it in.** `tests/test_app.py` monkeypatches `make_vlm` with the oracle, for
  the one real-Chromium episode against the fake sandbox. The oracle itself also refuses to run
  outside pytest.
- **Verifier-only state access.** Nothing in the tool path reads the DOM or the sandbox state.
  Only `verify` and the `done` snapshot (both on the verifier side) call `/api/state`.

## Imports and the server venv

`gym env start` runs `python app.py` with cwd and `PYTHONPATH` set to this dir only. So `app.py`
puts the repo root on `sys.path` first, to make `agent/`, `common/` and `sandbox/` importable.
It imports `nemo_gym` before doing so: that import moves `$NEMO_GYM_EXTRA_ROOTS` to the front of
`sys.path`, and the variable may name another checkout. Inserting afterwards means the server
always imports its own checkout. `make_tasks.py` does the same.

There is deliberately no repo-root `pyproject.toml`, and no `resources_servers/__init__.py`:
- A root `pyproject.toml` would switch Gym's venv builder into editable mode for every server in
  this repo.
- A `resources_servers/__init__.py` would shadow Gym's namespace package.

`requirements.txt` pins the server venv to the `.venv-gym` versions. `app.py` installs
Playwright's Chromium on first start if it is missing (`ensure_chromium`). A bare Linux box also
needs `python -m playwright install-deps chromium`.

## Provenance

Adapted from NeMo Gym `resources_servers/interactive_browser` (github.com/NVIDIA-NeMo/Gym, PyPI
`nemo-gym==0.7.0`, Gym tag `v0.7.0` = commit `1f8afc71d45fac14872d86fe300374afa2001e7a`, committed
2026-10-08). Apache-2.0, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. NeMo Gym has been
open source for far longer than 2 weeks, and the 2-week rule applies per project.

Last change to each source path at that commit, the equivalent of `git log -1 1f8afc7 -- <path>`
(read from the GitHub API, `repos/NVIDIA-NeMo/Gym/commits?sha=1f8afc7&path=<path>&per_page=1`):

| Path | Last changed | Commit |
| --- | --- | --- |
| `resources_servers/interactive_browser/app.py` | 2026-10-02 | `bdc018ac9aaea2b96444af915673ba27d32763e7` |
| `resources_servers/interactive_browser/browser/_chromium.py` | 2026-09-22 (the server's first commit) | `6bbc05ed9d1664dc695faaf2b0e71fc9863e08ca` |
| `resources_servers/interactive_browser/configs/interactive_browser.yaml` | 2026-09-29 | `016bb6b811aa3cf27217d3385e981771115f7b88` |

| Copied or adapted (from `app.py` unless noted) | Not used |
| --- | --- |
| The `num_workers=1` guard and `ray_enabled=False` | The `browser/` backends (they read the DOM) |
| The typed-vs-legacy seed union (`union_mode="left_to_right"`) | Element-id tools |
| The typed-session registries and the typed close (minus its empty-body close) | URL/DOM scoring |
| Release on re-seed and in verify's `finally` | Providers, `site/` |
| `ensure_chromium` (from `browser/_chromium.py`) | — |
| Config layout, `data/.gitignore`, test patterns | — |

Everything else here was written during the hackathon.
