# AGENTS.md

Guidance for coding agents and teammates working in this repo.

## Project

We fine-tune a ≤3B VLM for GUI click grounding (screenshot + instruction → click
point) and improve it through NVIDIA NeMo Gym. In the demo, Nemotron 3.5 Lightning
plans browser tasks and calls the grounder as a tool. The use case is accessibility:
a user describes what they want, and the agent does it from pixels alone.

- Plan, decisions, milestones, workstreams: [docs/plan.md](docs/plan.md).
- Verified facts and sources: [docs/research-notes.md](docs/research-notes.md).
  Read it before re-researching anything.
- Submission checklist, publishing, video plan: [docs/submission.md](docs/submission.md).
- Tool setup: [SETUP.md](SETUP.md). GPU box serving and training (pinned stack):
  [docs/gpu-box.md](docs/gpu-box.md).

## Hard constraints

- **Deadline:** everything, including the demo video, is due **Sunday 2026-10-18,
  11:00am EDT**.
  - SFT is done by Saturday evening.
  - GRPO is gated and hard-stops at ~1am Sunday.
  - Sunday morning is only evals, the demo and publishing.
- **Parameter budget:** ≤ 33B total across *all* generative models in the system,
  including any used for verification, critique, routing, or speculative decoding
  (embedding and reranking models are excluded).
  - Counts come from the official number on each model card, and MoE models count
    total parameters, not active.
  - Lightning's card says 30B, so the grounding VLM must be ≤ 3B. No separate judge
    LLM.
  - **No speculative decoding of any kind:** not DSpark, not DFlash, not Lightning's
    built-in MTP (no `--speculative-config`). Don't copy serve commands from
    Lightning's model card, because its recipes enable DSpark.
  - **Never run the base and the fine-tuned VLM as separate instances.** Separate
    instances count as two models (30 + 2 + 2 > 33).
  - **One loaded model with a LoRA adapter switched per request counts as one
    model (32B total).** That's our design: served name `base` = adapter off,
    `grounder` = adapter on, from one vLLM process.
  - Keep the screen-summary model name configurable (`base` or `grounder`). The
    fallback is the tuned model for both roles.
- **Code provenance:** all code must be written during the hackathon.
  - Rule text: "Open source code is allowed only if it's been open-sourced for 2+
    weeks." **Confirmed: it applies per project, not per release.** So current
    releases of NeMo Gym, vLLM and TRL are fine, and so is the vLLM fix file.
  - When copying code, record the upstream commit SHA and that path's last-changed
    date (`git log -1 <sha> -- <path>`).
- **Open data:** cite every data source. Any model-generated training data must be
  published together with the exact command and config that made it. That includes
  Lightning-rewritten instructions, Lightning keep/drop filter decisions, and
  reward=1 rollouts used as SFT data.
- **Evaluation (decided 2026-10-17, "option C"):**
  - An OS-Atlas `dev` split (by screenshot) is for choosing checkpoints.
  - **ScreenSpot-v2-clean** is the final held-out test: SSv2 (revision `5efbb1f1b5`)
    minus the 28 desktop rows listed in `benchmarks/screenspot_v2/excluded_rows.json`, so **1,244 rows**.
  - Never train on ScreenSpot-v2 or v1. Deduplicate training images against SSv2
    with a perceptual hash.
- **Why 28 rows are excluded:** we *do* train on OS-Atlas Linux/macOS desktop data,
  and those 28 SSv2 screenshots were captured on the same machines. That
  exclusion is the only filtering of SSv2.
  - Assert 1,272 rows on load and 1,244 after the exclusion.
  - Report SSv2-clean as the headline, with per-platform and text/icon splits.
    State the exclusion openly.
- **Never use** Wave-UI `source=="screenspot"` rows or GroundUI-18K (they contain
  ScreenSpot). Details: docs/research-notes.md §5.
- **Choose checkpoints on the `dev` split** (held out from the training sources by
  screenshot; grouping rules in docs/plan.md), never on ScreenSpot-v2-clean.
- **Public repo:** commit only public datasets and the sandbox app. No proprietary
  screenshots, customer data, credentials, or internal-system details. Secrets live
  in `.env` and NeMo Gym's `env.yaml`, both gitignored. Never paste tokens into a
  coding agent.

## Architecture decisions

- **NeMo Gym** (PyPI `nemo-gym==0.7.0`) provides our environments, evals and rewards:
  - `resources_servers/gui_grounding` (reward = point in bbox; stream A);
  - `resources_servers/web_agent` (Lightning plus a Playwright sandbox; stream C).
  Keep them in this repo in Gym's layout. Don't fork Gym.
- **Training uses TRL** (LoRA SFT, then gated GRPO), with a reward function that
  POSTs to the Gym resources server's `/verify`.
  - NeMo RL + Gym can't train small Qwen VLMs: it asserts a Nemotron-only processor.
  - VeRL + Gym is text-only.
  - Don't switch frameworks mid-weekend.
- **Base-model shortlist:** `Qwen/Qwen3.5-2B` and `Qwen/Qwen3-VL-2B-Instruct`. Choose
  at gate D1 in docs/plan.md.
- **Served names (vLLM :8001):** `base` is the base weights (baselines, and the
  screen summaries Lightning reads). `grounder` is the current LoRA adapter on top of
  `base`.
- **Lightning** is text-only: give it text observations, never images.
  - Development: use the hosted build.nvidia.com endpoint.
  - Final agent eval and demo: self-hosted. Hybrid first: Lightning on Patrick's
    Mac via vllm-metal (official BF16), grounder on the H200. See docs/local-mac.md
    and docs/plan.md "Hosting".

## Data and model-interface conventions

- **Bounding boxes:**
  - OS-Atlas `bbox` is `[left, top, right, bottom]` normalized to [0, 1].
  - ScreenSpot-v2 `bbox` is `[x, y, w, h]` in pixels.
  - Convert both to the canonical pixel xyxy schema in docs/plan.md.
- **Training-data filters** (training only, never SSv2): drop invalid or zero-area
  boxes, and rows whose instruction text appears twice on the same screenshot.
- **Image preprocessing has one owner:** `common/grounding.py:preprocess_screenshot`.
  - It converts to RGB and resizes, preserving aspect ratio, to the shared
    `MAX_PIXELS`, with dims that are multiples of 32.
  - No padding, cropping, stretching, or manual mean/std.
  - Data prep, Gym rows, TRL, vLLM (`--mm-processor-kwargs`) and the MVP all use the
    same cap. Details: docs/plan.md "Decisions so far".
- **Coordinates:** Qwen3-VL emits points on a relative 0–1000 grid. Qwen3.5 is
  assumed to do the same; stream A verifies this at the M1 baseline.
  - Prompts, output parsing and coordinate conversion live only in
    `common/grounding.py`; everything imports from there.
  - Spot-check visually after any conversion change.
- **Large artifacts stay out of git:** datasets, images, checkpoints, and full
  rollouts go to Hugging Face or the Brev box. Only small sampled rollouts that use
  `file://` paths (no base64) are committed, e.g. for BLADE packages.

## NeMo Gym rules of thumb

- **Exceptions abort the whole run.** An exception in a tool or `verify()` aborts the
  entire `gym eval run`. Catch it, and return `reward=0, mask_sample=True` with a
  `failure_reason`.
- **Declare row fields.** Every row field `verify()` needs (bbox, img_w/h, platform,
  data_type) must be declared on the VerifyRequest subclass, or it's silently
  dropped. Put slice fields on the VerifyResponse so they reach the metrics.
- **Avoid base64 screenshots.** Prefer `file://` image URLs under `/home/ubuntu/data`
  when Gym and vLLM share the box.
- **`simple_agent` doesn't stop on a `done` tool.** It stops when the model replies
  without a tool call, or at `max_steps`.
- **Run `gym` from the repo root** (config, server venvs, `./cache` and `./results`
  are all relative to the current directory).
- **Failure analysis:** after every comparable pair of eval runs (e.g. base vs.
  tuned), run the **BLADE** skill (`.agents/skills/nemo-gym-blade-analysis`, also
  under `.claude/skills/`) on the artifacts.
  - Let its failure taxonomy drive the next data, prompt or verifier change.
  - Use `nemo-gym-reward-profiling` to choose GRPO prompts.
- **`add-benchmark` skill:** skip every GitLab/MLflow dataset-registry step
  (`gym dataset upload --storage gitlab`, `gitlab_identifier`, `mlflow_tracking_*`,
  `+data_source=gitlab`). We have no registry.
  - Keep train/dev JSONL in `/home/ubuntu/data` and on HF.
  - Commit only `data/example*.jsonl`.
  - Never ask anyone for a GitLab/MLflow token.

## Compute

- **Lightning checkpoint: official NVIDIA BF16 only,** `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`. The bounty
  requires an official NVIDIA checkpoint, and NVIDIA publishes only BF16 and NVFP4;
  the team chose BF16 for precision.
  - Never use community quants (mlx-community, GGUF, Ollama) for anything we report
    or demo. The hosted build.nvidia.com endpoint is NVIDIA's own and is fine for
    development.
- **Pinned GPU stack:** vLLM image v0.31.0 (by digest), torch 2.13.0+cu130,
  transformers 5.17.0, TRL 1.16.0. Never `pip install -U` these.
  - Self-hosting Lightning on the H200 requires the vLLM #61603 patch
    (docs/gpu-box.md §1). It's allowed under the per-project 2-week rule.
- **One shared Brev H200** (`permanent-tan-marmot`) at ~$6.90/hr; about 23 GPU-hours
  for the whole event.
  - Stream B (Kyle) owns serving and the GPU schedule.
  - Announce GPU jobs in team chat, and run them in tmux.
  - Stop the instance when nobody is using it.
- **Shared box hygiene:** use `brev login` (browser flow), not `--token`. On the box:
  - never store GitHub credentials;
  - for HF uploads, use only a fine-grained token scoped to write only `patrickballou/gui-assist-grounder` and `patrickballou/gui-assist-data`, exported in the upload shell;
  - never run `hf auth login` there. It writes the token into the shared `HF_HOME`,
    which is mounted into the vLLM containers.
- **Log every GPU run** in `docs/runs.md`: command, config, adapter/served name,
  start/end, result.

## Working norms

- Pin versions for anything that runs on the GPU box, and record the exact command
  used for each training or eval run.
- Every reported number must be reproducible from a committed script and config.
