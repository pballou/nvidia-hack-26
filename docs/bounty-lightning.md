# Best Use of NVIDIA Nemotron Lightning: GUI Assist

> **Status:** draft of Sat 2026-10-17. Numbers marked **TBD** come from Sunday's runs. The Saturday
> reads are n=1 per task and are labeled that way.

## 1. Summary

GUI Assist is a vision-only accessibility agent. A user says what they want ("Make the text bigger,
I can't read this."), and the agent does it in the app on screen, from pixels alone. Nemotron 3.5
Lightning is the agent's brain, and it never sees the screen. After each action it reads a short
text observation: the URL, whether the screen changed, what its last action did, and a screen
description written by a 2B VLM. It then calls one of five tools. When it calls
`click("the L option in the Text size row")`, a fine-tuned 2B grounder turns that description into
a pixel, and Playwright clicks there. Offline, Lightning also rewrites raw accessibility labels into
click instructions for the grounder's SFT data. The final grounder trained on them: its SFT stage
used the Lightning-rewritten set, picked on dev over raw labels (§2b). The system fits the 33B cap at 32.6B by model-card counts: Lightning 30B, one Qwen3.5-2B VLM
whose LoRA adapter is switched per request, so it counts once (`base` describes screens, `grounder`
clicks), and Parakeet TDT 0.6B for local voice input. Raw HF tensor totals are higher; the rule
uses card counts. There is no speculative decoding and no judge model.

## 2. What Lightning does

### 2a. Planner and tool-caller

```text
 user request
      │
      ▼
 Lightning (text only) ──► one tool call, e.g. click("the gear icon in the top-right toolbar")
      ▲                       │  schema check (an invalid call runs nothing, gets a corrective result)
      │                       ▼
      │                  grounder: Qwen3.5-2B + LoRA (served `grounder`): screenshot + description → point
      │                       ▼
      │                  Playwright clicks that pixel → settle → new screenshot
      │                       ▼
 text observation ◄────── screen description: same 2B, adapter off (served `base`)
 (URL, title, changed?, last action result, PAGE / NAV / MAIN / STATE / OVERLAYS)
```

| Tool | Why this shape |
| --- | --- |
| `click(target_description)` | A short visual phrase, the way you'd point something out to a sighted helper: visible text or icon, element type, position. This is the style T3's rewrites teach. The final grounder trained on them (§2b); the Saturday pilot adapter and `sft-template` used raw labels. The prompt forbids coordinates, selectors and ids (the validator doesn't enforce this). |
| `type_text(text, target_description?)` | With a target, the field is grounded, clicked and its text replaced. Enter is never pressed, so submitting is a separate step. |
| `press_key(key)` | Enter, Tab, Escape, arrows; `ctrl+a` is normalized to `ControlOrMeta+a`. |
| `scroll(direction: up\|down, amount: small\|page)` | Enums instead of pixels: Lightning can't see how far to scroll. |
| `done(summary, success: bool)` | An explicit finish with a self-report, so claimed success can be checked against verified success. |

One action per step: the interactive planner executes only the first call, and extra calls get a
"not executed" result. Gym rows request `parallel_tool_calls: false`, but `simple_agent` runs every
call it receives. The tool schema and the system prompt are defined once, in `agent/prompts.py`.
Both the interactive planner and the Gym rows use them, and `make_tasks.py --check` rejects rows
whose prompt or tools have drifted.

### 2b. Data engine: instruction rewriting (T3)

The OS-Atlas Linux, macOS, AndroidWorld and FineWeb "instructions" are raw labels ("More options"),
but users phrase requests ("open more options"). Lightning rewrites 60% of those rows (selected by a
hash of the id) into short click instructions. The styles are functional, descriptive and
positional, about 50/30/20.

- **Input:** metadata only (label, control type, platform, coarse position and size, nearby
  labels). Lightning never sees the image.
- **Drops:** when no faithful instruction is possible, it returns `keep=false` with a reason.
- **Settings:** thinking off, temperature 0.7, prompt `v4`. Only the official checkpoint or NVIDIA's
  hosted endpoint may produce published rows.
- **To be published with (TBD):** `rewrites.jsonl`, `stats.json`, and `METHOD.md` (the verbatim
  prompt).
- **Where:** code in `data_prep/rewrite/`, spec in `docs/tasks/T3-lightning-data-engine.md`.

**Rewrite counts** (T3's `train_files.READY.json`, prompt `v4`):
- 9,384 rows were sent for rewriting: 8,028 through the hosted endpoint and 1,003 through the
  local official checkpoint. The numeric rule handled the other 353.
- **8,620 rewrites were kept.** In 411 rows Lightning declined (`keep=false`) and the raw label
  stayed. The numeric rule removed 353 rows, so the train files have 18,299 rows.
- **Human audit:** 42 of 43 checkable rewrites were faithful (85.7% if "unsure" counts as a
  failure).

**Template-vs-Lightning ablation** (trained Sat; same config and seed, measured through the Gym
`gui_grounding` env; [runs.md](runs.md) Sat 16:35 and 18:44):

| SFT variant | Train rows | Dev (selection) | SSv2-clean (table only) |
| --- | --- | --- | --- |
| `sft-template` (raw labels) | 18,652 | 85.66 / 85.86 (two launches) | **89.07** |
| `sft-lightning` (Lightning instructions) | 18,299 | **86.27** | 88.10 |

- **The pick followed the dev rule, so `sft-lightning` became round 0.** On dev it was +0.41
  over template in the same harness. That is about 2× the ~0.2-point run-to-run noise but well
  inside the ~1.1-point sampling SE.
- **On SSv2-clean, template scored higher** (by 12 rows of 1,244, about 1 point, mostly on icons:
  81.7 vs 84.8). That eval is for
  the table only and wasn't used to choose.
- **The final grounder is `sft-lightning` plus one kept GRPO round** from the NeMo Gym loop: dev
  87.00 and SSv2-clean 90.35 ([bounty-nemo-gym.md](bounty-nemo-gym.md)).
- **The variants differ in more than the text.** Template also has the 353 rows the numeric rule
  removed. So the honest reading is that the rewritten instructions roughly tie with raw labels on
  grounding accuracy, rather than clearly winning.

**TBD (T3 owner):** throughput and tokens for the rewrite run.

### 2c. Pre-flight request check

Before the browser opens, one more Lightning call (thinking off, ~1.6 s) checks the spoken request.
It either proceeds or asks one short question, e.g. for a clipped "Allors towards my bill." or a
payment with no clear amount (§4.5). False asks / clear requests and asks / ask cases, live: on the
acceptance set 0/62 and 22/22; on held-out B 1/45 and 17/19; on held-out C, run once on the frozen
prompt, 4/45 and 14/17. It's off for evals and the CLI. The console turns it on, with a visible
toggle.

## 3. Why Lightning

- **Text-only, with tool calls we can check against a schema.** Perception lives in a 2B VLM and
  reasoning in a text model. Lightning plans from descriptions. Measured so far (n=1 reads): 0
  invalid of 27 in-loop calls with `tool_choice=required`, 1 of 43 with `auto` (§4.1).
- **Interactive latency.** It's 30B total but about 3B active (MoE, hybrid Mamba-2). Self-hosted on
  an idle server (short probe), a planner call took a median of about 1.2 s with thinking off and
  2-3 s with it on. In the first real-sandbox read (full prompts, shared server, thinking-retry
  steps included), plan p50/p90 was 4.8/16.8 s; over all 12 tasks it was 4.1/5.5 s with thinking
  off and 4.3/14.5 s with retry (§6).
- **Self-hostable official weights.** We serve NVIDIA's BF16 checkpoint
  (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`) on a Mac with vllm-metal (`:8000`, served
  name `lightning`). Hosted build.nvidia.com measured a p50 of about 26 s per call, max 94 s
  (thinking off, 17 calls). That is too slow for a user waiting on a loop of up to 12 steps.
- **Per-request thinking switch** (`chat_template_kwargs.enable_thinking`), so we spend reasoning
  only where it helps (§4.3).
- **Fits the budget:** 30B + 2B + 0.6B (Parakeet, voice input) = 32.6B ≤ 33B by model-card counts
  (`base` and `grounder` are one model with a LoRA adapter). The H200 stays free for the grounder.

## 4. How we optimize it

### 4.1 Tool schema and corrective validation

- **What:** vllm-metal doesn't enforce tool schemas, so every call is checked. Malformed JSON is
  caught when the reply is parsed (`agent/planner.py`, then `invalid_json_error`), and the tool
  name, argument names, enums and types by `agent/session.py:validate_args`.
  - An invalid call executes nothing, and its tool result quotes the signature, e.g.
    `Expected click(target_description: string)`.
  - Invalid calls are counted. Argument names are never renamed. The only coercions: `"true"` /
    `"false"` strings become booleans, and an empty optional string counts as not given.
  - Tool calls written as text (qwen3_coder XML or JSON) are recovered, validated like any call,
    and flagged `from_content`.
- **Why `required` here and `auto` in Gym:**
  - The interactive planner uses `tool_choice=required`.
  - Gym's `simple_agent` ends an episode only on a reply without a tool call, so Gym rows keep
    `auto`, and `make_tasks.py --check` enforces it.
  - `simple_agent` answers bad-JSON calls itself, so `agent_metrics.py` re-validates every call from
    `response.output`.
- **Evidence:**

| Lightning build | `tool_choice` | Result |
| --- | --- | --- |
| Community 4-bit (plumbing only, not reportable) | — | 1-2 of 5 calls invented argument names |
| Official BF16 | `required` | 5/5 correct argument names |
| Official BF16 | `auto` | 4/5 correct argument names |
| Official BF16, first real-sandbox read (§5, n=1) | `required` | 0 invalid of 27 calls |
| Official BF16, Gym acceptance read (§5, 5 real rows, n=1) | `auto` | 1 invalid of 43 calls |

### 4.2 Observation design

```text
Step N | URL: /settings?tab=accessibility | Title: …
Last action: clicked "…" at (x, y)
The screen did NOT change after your last action. The click probably missed …   (or: The screen changed.)
Screen:  PAGE / NAV / MAIN / STATE / OVERLAYS  (≤ 40 lines, ≤ 240 chars each, repeated lines dropped)
```

- **NAV compression.** Bars that appear on every page get at most two lines. Before this, they used
  up the 40-line budget and the page's own controls were cut off. Each list or table row gets one
  line, so per-row icon buttons survive.
- **No-change note.** A downsampled pixel diff of the screenshots before and after decides whether
  the screen changed, and a miss is stated outright.
- **STATE-based instructions.** STATE holds the current values. The prompt says "bigger" means an
  option past the current one, and asks that the latest observation show the result before
  `done(success=true)` (a prompt rule; the code doesn't enforce it).
- **URL.** `?tab=` stays, because it tells Lightning which tab is open. The sandbox `sid` is
  dropped.
- **Description decoding.** Greedy decoding alone made Qwen3-VL-2B loop on sparse screens. A
  `frequency_penalty` of 0.3 stopped the loops we saw (0.6 broke Qwen3.5), and repeated lines are
  still dropped as a backstop (`common/grounding.py`, `SUMMARY_PROMPT` and `SUMMARY_SAMPLING`).
- **Prompt examples come from other apps** (since `agent-v4`, Saturday evening): an Orders
  table, a Wi-Fi toggle, a Repeat control, a music app's menus. The fact-check of this write-up
  found that `agent-v3` used one eval task's own target as an example ("the star next to the
  message from the landlord"), mirrored sandbox rows (a Paid statement with a download icon), and
  that the 2B describer copied the NAV example's items and "red 2 badge" into sandbox
  descriptions. The Saturday reads below ran on `agent-v3`; Sunday's run on `agent-v4`.

### 4.3 Thinking control

- **Defaults:** the interactive agent uses `thinking: retry` (`configs/agent.yaml`); Gym runs
  default to off (`agent_eval.sh`, `THINKING=off`).
- **Cost:** on the idle-server probe, about 1.2 s per call with thinking off against 2-3 s with it
  on. In the loop (12 tasks, full prompts, shared server), plan p50/p90 was 4.1/5.5 s with
  thinking off and 4.3/14.5 s with retry: the median barely moves, the tail grows (§6).
- **`thinking: retry`** runs each call with thinking off, and turns it on only for the call after
  a problem step: a failed or invalid call, no screen change, or no tool call.
  - It never turns on twice in a row after a no-tool-call reply. Otherwise a thinking reply cut off
    at the token limit would keep thinking on until `max_steps`.
- **Budget:** 1024 max tokens with thinking off. Thinking calls get `thinking_max_tokens: 4096`,
  because reasoning counts against `max_tokens`. A truncated reply gets a "call one tool now"
  correction.
- **Gym limitation:** Gym can't express per-step retry. Gym runs are off (or `THINKING=on`), and
  retry is measured with the interactive harness.
- **Saturday read, off vs retry** (all 12 tasks × 1, base grounder, `agent-v3`; n=1, so success
  differences are noise):

| | thinking off | thinking retry |
| --- | --- | --- |
| Verified success | 5/12 | 6/12 |
| Claimed = verified | 6/12 | **9/12** |
| No-change actions | 45 | **27** |
| Hard tasks verified | 0/3 | 2/3 |
| Mean steps | 9.1 | 8.0 |
| Thinking calls | 0/109 | 29/96 |
| Plan p50 / p90 | 4.1 / 5.5 s | 4.3 / 14.5 s |

  Reasoning only after a problem step mostly improves Lightning's self-assessment (its done claims
  match the sandbox more often) and cuts wasted clicks, for a heavier latency tail.

- **Saturday night, off vs retry vs on** (all 12 tasks × 3 repeats, tuned grounder `sft-template`,
  `agent-v4`, interactive agent; `docs/runs.md` Sat 20:34). The on arm has 32 of 36 runs because
  the grounder was swapped at 22:01, so this compares the same 32 jobs:

| Same 32 jobs | off | retry | on |
| --- | --- | --- | --- |
| Verified success | 19/32 | 21/32 | **26/32** |
| False success claims | 9 | 9 | **4** |
| No-change actions | 72 | 54 | **38** |
| Mean steps | 10.4 | 9.0 | **5.9** |
| Plan p50 / p90 per call | 6.3 / 7.7 s | 5.9 / 16.7 s | 10.9 / 24.7 s |
| Wall-clock per task p50 / p90 | 65 / 220 s | 68 / 279 s | **53** / 293 s |
| Thinking tokens per run | 0 | 616 | 1,502 |

  Thinking on every call wins on the hardest tasks (same jobs): the canvas chart 0/3 → 2/3 and
  the scroll-only message 0/2 → 2/2. On vs off, paired: on won 8 jobs and off won 1 (exact McNemar
  p = 0.039); on vs retry p = 0.18. Each call is slower, but runs need half the steps, so the
  median task finishes sooner. **Thinking on is a measured option, not the default:** +5 verified
  of 32 vs retry at 10.9 s per planning call (retry 5.9 s). **The demo recording uses thinking
  off,** the Gym-measured configuration in which the final grounder passed all 5 scripted tasks
  3/3. It also avoids 25-45 s thinking pauses on camera: a fully local plan step takes ~4 s off vs
  ~13 s with thinking, and the 26/32 was measured with `sft-template`. `retry` stays the
  interactive default in `configs/agent.yaml`. Gym evals stay off, so the reported base-vs-tuned
  comparison is unchanged.

### 4.4 Context budget

vllm-metal has no prefix cache for Lightning, so every step re-prefills the whole prompt.

- **Interactive planner (`agent/planner.py`):** a call carries the system prompt, the request with
  the first screen, and the last 6 steps verbatim (`history_full_steps: 6`). Older steps collapse
  into one-liners, e.g. `click("the gear icon …") -> ok, screen changed`, and the stale first
  screen is dropped. Reasoning is never replayed into the history.
- **Gym:** `simple_agent` resends the full conversation each turn (no collapsing), and `vllm_model`
  attaches earlier reasoning to replayed assistant messages, so a 12-step Gym episode carries every
  observation.

### 4.5 Robustness

- **Recovery paths.**
  - A missing tool call gets a corrective user message, and an invalid call a corrective tool
    result.
  - A grounding or browser failure returns `ok:false` with the error.
  - If an action ran but the screenshot after it failed, the action isn't reported as failed, so
    Lightning doesn't repeat it and flip a toggle back.
- **Timeouts.** Lightning 180 s per call, VLM 45 s (one retry), navigation 15 s, Gym tool call
  240 s.
- **Stuck stop instead of a tight step cap** (interactive agent; `agent/progress.py`). After 4
  actions in a row with no progress, or the same action 3 times with no change (same target, or
  a click within 12 px), Lightning is told it is stuck and to change approach. If 3 more actions
  make no progress, the run ends as `stuck` with a summary of what it tried and a question for
  the user. The cap is 25 steps. Replayed on Saturday's 47 real runs, an immediate stop would have
  cut off 2 of the 29 successes (both recovered after 5-6 no-change clicks); warn-then-stop cut
  off none. Gym runs keep `max_steps` 12 and no stuck stop, so reported numbers stay comparable.
- **Rate limits.** A 429 waits (`Retry-After`, else 15-60 s backoff, up to 4 times) without
  spending a step. A 5xx gets one retry, and two Lightning errors in a row end the run.
- **Infra errors are masked, not scored.**
  - In Gym, any VLM transport error, browser crash or tool timeout masks the rollout
    (`mask_sample=True`, `failure_kind`), whatever its reward.
  - `agent_metrics.py` likewise excludes stopped, incomplete and errored runs.
  - Model-quality failures stay scored.
- **Pre-flight "understand the request" check** (`agent/clarify.py`, interactive only, off by
  default). A clipped voice take ("Allors towards my bill.") once made the agent pay the full
  balance. The pre-flight is one Lightning call before the browser opens, with thinking off, T=0
  and two tools: `proceed(normalized_request)` or `ask_user(question)`. It asks only for a garbled
  request, a payment or account change missing a detail, or an unclear target. That ends the run
  as `needs_input`, and the answer starts a follow-up run. One call takes ~1.6 s on local
  Lightning.
  - **Acceptance set** (12 task phrasings, 50 spoken and near-boundary variants, 22 ask cases):
    0 false asks, 22/22 asks.
  - **Blind held-out sets:**
    - B (used to tune v3 → v4): 1 false ask in 45 and 17/19 asks.
    - C (run once on the frozen prompt): 4 false asks in 45 and 14/17 asks.
  - Because of the held-out false asks it is off in `configs/agent.yaml` (evals, CLI). The
    console turns it on, with a visible toggle: the demo uses acceptance-set phrasings, and a
    false ask costs one extra answer. `docs/results.md` has the counts.

## 5. Measured in NeMo Gym

**The `web_agent` env** (`resources_servers/web_agent/`):

- **Same agent code.** Every tool route calls the interactive agent's `AgentSession.execute`, with
  the same tools, prompt and observations.
- **Reward:** the task's check on sandbox state. Only the verifier reads state.
- **Start screen in the row.** Gym's legacy `/run` drops the `seed_session` response, so the start
  observation is baked into each row, with a staleness flag.
- **Step budget.** `max_steps: 12` counts model calls, the same as the interactive limit.
- **VerifyResponse fields:** `reward`, `claimed_success`, `steps`, `n_invalid_calls`,
  `n_failed_actions`, `n_vlm_infra_errors`, per-stage `latency_ms`, `grounder_model`,
  `prompt_version`.

**Base vs tuned** (`scripts/agent_eval.sh`):

1. The same rows and the same Lightning run once per served name (`base`, then `grounder`, from one
   vLLM process), with `--num-repeats 3`.
2. Preflight refuses placeholder rows and a local Lightning other than the official BF16
   (`FORCE=1` allows either for a plumbing run), marks hosted and `FORCE` runs non-reportable, and
   records `reportable=yes|no`.
3. Then `gym eval compare`; BLADE on the pair is run by hand afterward.

**Results.** Planner: official BF16 Lightning, local (one run unrecorded, see ²). Screen descriptions:
`base`.

| Read | Grounder | Thinking | Tasks × repeats | Verified | Claimed | Invalid calls | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sat, first real-sandbox read¹ | `base` (Qwen3.5-2B) | retry | 4 easy × 1 | 3/4 | 2/4 | 0/27 | n=1 |
| Sat, pilot adapter read² | Kyle's 2k-row pilot LoRA | retry | 4 easy × 1 | 4/4 | 3/4 | TBD | **n=1, NOT FINAL** |
| Sat, Gym acceptance³ | `base` | off | 5 rows (4 easy + 1 medium) × 1 | 3/5 | 2/5 (1 false claim) | 1/43 | n=1, not final |
| Sat, interactive, all tasks⁴ | `base` | off | 12 × 1 | 5/12 | 7/12 | 0/109 | n=1 |
| Sat, interactive, all tasks⁴ | `base` | retry | 12 × 1 | 6/12 | 7/12 | 1/96 | n=1 |
| Sat night, Gym⁵ | `base` | off | 12 × 3 | 18/36 | 24/36 (26 match, 8 false) | 10/259 | reportable (`agent-v4`) |
| Sat night, Gym⁵ | `grounder` = sft-template | off | 12 × 3 | **22/36** | 25/36 (31 match, 4 false) | 6/234 | reportable (`agent-v4`) |
| Sat night, Gym⁶ | `grounder` = FINAL (sft-lightning + GRPO) | off | 12 × 3 | 17/36 | 27/36 (24 match, 11 false) | 6/256 | reportable (`agent-v4`) |
| Sat night, interactive (§4.3) | `grounder` = sft-template | off / retry / on | 12 × 3 (on: 32) | 20/36 · 22/36 · **26/32** | 28/36 · 32/36 · 30/32 | 6/374 · 7/328 · 0/189 | reportable (`agent-v4`) |

All Saturday reads used prompt `agent-v3` (see §4.2) and the task checks from before sandbox
`68c7671`, which made the checks stricter (no unrequested payments, messages, reports or autopay;
every notification row and star pinned). Sunday's runs use `agent-v4` and the stricter checks.

¹ Sandbox `02a8e1b`, VLM on `:8001`, no oracle (`runs/summary_20261017-160547.json`):
`AGENT_SANDBOX_URL=http://127.0.0.1:8091 $PY scripts/run_agent_tasks.py --tags easy --lightning local
--grounder base --thinking retry` (§8).

² Same, with the pilot adapter as `grounder` on `:8003` (`AGENT_VLM_BASE_URL=http://127.0.0.1:8003/v1`,
`--grounder grounder`; `runs/summary_20261017-163513.json`). The Lightning checkpoint wasn't
recorded for this read's `text-size-large` run (the `/models` probe hit a connection error); the
other three runs report the official BF16. `runs/` isn't committed, and neither read is in
`docs/runs.md` yet (TBD).

³ Logged in `docs/runs.md` (Sat 16:43; artifacts in `results/acceptance_base/` on Patrick's Mac,
not committed), `reportable=yes`. The false claim was `high-contrast` (reward 0, success claimed).

⁵ `docs/runs.md` Sat 18:45: official BF16 Lightning, sandbox `68c7671`, the same rows and planner
in both conditions, `base` screen descriptions in both, 0 masked. With the tuned grounder,
no-change actions fell 83 → 45 and failed actions 10 → 0 (CI excludes 0), and Lightning's done
claims matched the sandbox 31/36 vs 26/36. `gym eval compare`: reward +0.111, 95% CI
[−0.06, +0.28] (n=36: suggestive, not significant); 3 fail→pass, 1 pass→fail. With a strict
`mark-all-read` check, tuned is 20/36 (+2), as the current check doesn't guard the bell
notifications ([BLADE](blade/agent_base_vs_sft.md)); the check stays as-is until after Sunday's
grounder rerun.

⁶ `docs/runs.md` Sat 22:04: the FINAL grounder arm only, against the same base arm (identical
rows, planner and thinking). Its 17/36 is within noise of base (reward −0.028, 95% CI
[−0.20, +0.14]); failed actions 10 → 0 and text-size-large 3/3 still hold. The drop from SFT's
22/36 is not a click problem: about 6 of the 9 rollouts that changed trace to the planner or the
screen summaries, the 2 grounder-caused ones cancel, and FINAL hits 96.5% of well-posed targets
([blade/agent_final_vs_sft.md](blade/agent_final_vs_sft.md)). For
the planner this is the point: the agent's end-to-end success now depends mostly on Lightning's
choices and on what the description shows, which is where thinking on helps (§4.3).

⁴ `$PY scripts/run_agent_tasks.py --tasks sandbox/tasks.json --sandbox-url http://127.0.0.1:8091
--grounder base --thinking off|retry` (`runs/summary_20261017-165130.json` and
`-170457.json`). Failures cluster on scroll-dependent targets (`star-landlord-message`,
`stop-promotions`), the canvas chart (`max-usage-month`), a dense toggle grid with a false done
claim (`sms-billing-off`), and, with thinking off, paying the full balance instead of $50
(`pay-50`).

**What the Saturday reads showed:**

- **`text-size-large` failed on the base grounder: a grounding miss.** Lightning asked for the L
  option six times (steps 2, 5, 7 and 9-11), and each click landed at x 1108-1142 (L is at about
  x 1192); text size stayed M. In between it tried XL, M, the row and the label, and its step-1
  target was a copied STATE line (§7). The intent was right and the clicks were wrong. The pilot
  adapter passed this task's state check (n=1), but that run hit the 12-step limit without calling
  `done`.
- **`mark-all-read` passed verification, but Lightning didn't claim it.** The description never
  showed the read state.

Plumbing on the fake sandbox (not reportable): the Gym eval finished 6/6 rollouts without aborting,
and the oracle harness verified 5/5 fake tasks across 5 runs.

## 6. Latency

From the first real-sandbox read (base grounder, thinking retry):

| Stage | p50 | p90 | Note |
| --- | --- | --- | --- |
| plan (one Lightning call) | 4.8 s | 16.8 s | Includes thinking-retry steps; the server was shared |
| summary (2B, `base`) | 1.15 s | 1.80 s | Screen description after each action |
| ground (2B) | 0.73 s | 0.84 s | Description → point |
| settle | 0.51 s | — | Network idle + 350 ms + network idle |
| screenshot | 57 ms | — | |
| act | 11 ms | — | |

All 12 tasks (base grounder, n=1), plan p50/p90: 4.1/5.5 s with thinking off, 4.3/14.5 s with
retry; wall time per task p50 67 s (off) and 80 s (retry). Idle-server planner: about 1.2 s per
step with thinking off, 2-3 s with it on. Hosted: p50 about 26 s, max 94 s. **Sunday table: TBD**
(`agent_metrics.py`).

## 7. Limitations and next steps

- **Claimed vs verified, in both directions (n=1).** Lightning can only confirm what the
  description shows. The Saturday base read under-claimed (2/4 claimed, 3/4 verified:
  `mark-all-read`), and the Gym acceptance read had one false success claim (`high-contrast`).
  Next: have the description report read/selected state, or run a targeted check before `done`
  with the same 2B model, which adds no parameters.
- **A 2B describer caps the planner.** In several traces (e.g. `runs/20261017-160547-aad442`,
  `-160900-a66c04`, `-163513-3e51dd`), the base VLM copied the `agent-v3` NAV example's "bell icon with a red
  2 badge" (the sandbox bell shows 3). In `20261017-160547-aad442`, Lightning used a STATE line
  verbatim as a click target (`Text size "M" is selected`). Next: measure `SUMMARY_MODEL=grounder`.
- **Small n.** The Saturday reads are n=1 on 4 easy tasks (5 rows in the Gym read). The final set is 12 tasks × 3 repeats
  at temperature 1.0. We report counts, not just percentages.
- **One app, and our prompts were iterated on it.** The Saturday reads ran on `agent-v3`, whose
  planner prompt used an eval task's own target ("the star next to the message from the
  landlord") as an example, so those absolute numbers are optimistic. `agent-v4` replaced every
  example with ones from other apps, and the rows were regenerated. Prompt wording was still tuned
  by watching traces on these 12 tasks, so treat absolute success as in-domain. The grounder
  never trains on sandbox screenshots, and base and tuned get identical rows, prompts and tools,
  so the comparison stays fair.
- **No prefix cache on vllm-metal**, so every step re-prefills. Next: enable prefix caching once
  vllm-metal supports it for Lightning.
- **Slow steps on the fully local Mac**, e.g. a 31.9 s done step in a smoke run. A no-change
  click (an already-downloaded badge) made `retry` think on the next call, which then decoded
  ~1,100 tokens at the normal ~38 tok/s. The statements table also yields 430-600-token
  descriptions at ~76 tok/s.
  Post-Sunday options, each needing a re-measure:
  - reuse the last description when the screen didn't change;
  - cap `thinking_max_tokens` at ~768;
  - don't let a no-change step trigger retry thinking (it is retry's main recovery signal);
  - lower the summary `max_tokens` to ~480 (that would also cut real ~470-token descriptions).

## 8. Reproduce

Run from the repo root.

```bash
# Sandbox :8090; VLM :8001 = base + grounder from ONE vLLM process (docs/gpu-box.md §3); Lightning :8000
scripts/run_sandbox.sh
source ~/.venv-vllm-metal/bin/activate   # on the Mac (docs/local-mac.md §2)
HF_HUB_OFFLINE=1 vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --served-model-name lightning --host 127.0.0.1 --port 8000 \
  --max-model-len 65536 --max-num-seqs 2 --gpu-memory-utilization 0.68 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3   # no --speculative-config

PY=.venv-gym/bin/python
# Interactive harness (run_task per task, then the sandbox check on the verifier side)
$PY scripts/run_agent_tasks.py --tasks sandbox/tasks.json --tags easy --lightning local --grounder base --thinking retry
$PY scripts/run_agent_tasks.py --tasks sandbox/tasks.json --lightning local --grounder grounder --thinking off
$PY scripts/run_agent_tasks.py --tasks sandbox/tasks.json --lightning local --grounder grounder --thinking retry

# Gym rows for all 12 tasks (generated, not committed), then base vs grounder x 3 repeats + compare
$PY resources_servers/web_agent/make_tasks.py --tasks sandbox/tasks.json --summary-model base \
  --out resources_servers/web_agent/data/tasks_all.jsonl
TASKS=resources_servers/web_agent/data/tasks_all.jsonl REPEATS=3 scripts/agent_eval.sh

# Tables: success, claimed vs verified, validity, recoveries, p50/p90 per stage
$PY scripts/agent_metrics.py --summary runs/summary_<timestamp>.json --group-by thinking,grounder_model
$PY scripts/agent_metrics.py --no-runs --rollouts results/agent_base.jsonl results/agent_grounder.jsonl
```

**Knobs:**

- **Agent:** env vars override `configs/agent.yaml`: `AGENT_LIGHTNING` (`local|hosted`),
  `AGENT_GROUNDER_MODEL`, `AGENT_SUMMARY_MODEL`, `AGENT_THINKING` (`off|on|retry`),
  `AGENT_VLM_BASE_URL`, `AGENT_SANDBOX_URL`, `AGENT_MAX_STEPS`.
- **`agent_eval.sh`:** `GROUNDERS`, `THINKING`, `TASKS`, `REPEATS`, `SUMMARY_MODEL`, `OUT_DIR`.
  `FORCE=1` allows plumbing runs, which are never reported.
- **Saturday reads:** sandbox `:8091`. The pilot read also set
  `AGENT_VLM_BASE_URL=http://127.0.0.1:8003/v1`.
- **Logging:** log every run in `docs/runs.md`.
