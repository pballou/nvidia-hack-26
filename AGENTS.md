# AGENTS.md

Guidance for coding agents and teammates working in this repo.

## Project

We fine-tune a ≤3B VLM for GUI click grounding (screenshot + instruction → click
point) and improve it through NVIDIA NeMo Gym. In the demo, Nemotron 3.5 Lightning
plans browser tasks and calls the grounder as a tool.

- Plan, milestones, workstreams: [docs/plan.md](docs/plan.md).
- Verified facts and sources: [docs/research-notes.md](docs/research-notes.md).
  Read it before re-researching anything.
- Tool setup: [SETUP.md](SETUP.md). GPU box serving and training (pinned stack):
  [docs/gpu-box.md](docs/gpu-box.md).

## Hard constraints

- **Deadline:** everything, including the demo videos, is due **Sunday 2026-10-18,
  11:00am EDT**. Fine-tuning should be finished by the end of Saturday.
- **Parameter budget:** ≤ 33B total across *all* generative models in the system,
  including any used for verification, critique, routing, or speculative decoding
  (embedding and reranking models are excluded).
  - Counts come from the official number on each model card, and MoE models count
    total parameters, not active.
  - Lightning's card says 30B, so the grounding VLM must be ≤ 3B. Don't add external
    draft models (DSpark/DFlash) or a separate judge LLM.
- **Code provenance:** all code must be written during the hackathon. Open-source
  code may be reused only if its project has been public for 2+ weeks. When copying
  a template (e.g. from NeMo Gym), note its source path and first-commit date.
- **Open data:** cite every data source. Any synthetic data we generate (e.g.
  Lightning-rewritten instructions) must be published together with the method
  used to make it.
- **Never train on ScreenSpot-v2 (or v1).** It is the held-out benchmark; pin
  revision `5efbb1f1b5`. Deduplicate all training images against it (perceptual
  hash) before training.
- **Don't use OS-Atlas Linux/macOS desktop data** (`linux_splited.json`,
  `macos_splited.json`). It was captured on the same machines as SSv2's new desktop
  screenshots, and the test targets are labeled in it. Dedup can't remove that.
  Also never use Wave-UI `source=="screenspot"` rows or GroundUI-18K. Details:
  docs/research-notes.md §5.
- **Report SSv2 three ways:** all 1,272 rows, the 1,128 rows on v1 images, and the
  144 new rows.
- **Choose checkpoints on the `dev` split** (carved from OS-Atlas by screenshot), not
  on ScreenSpot-v2.
- **Public repo:** commit only public datasets and the sandbox app. No proprietary
  screenshots, customer data, credentials, or internal-system details. Secrets live
  in `.env` and NeMo Gym's `env.yaml`, both gitignored.

## Architecture decisions

- **NeMo Gym** (PyPI `nemo-gym==0.7.0`) provides our environments, evals and rewards:
  - `resources_servers/gui_grounding` (reward = point in bbox);
  - `resources_servers/web_agent` (Lightning plus a Playwright sandbox).
  Keep them in this repo in Gym's layout. Don't fork Gym.
- **Training uses TRL** (SFT, then GRPO), with a reward function that POSTs to the Gym
  resources server's `/verify`.
  - NeMo RL + Gym can't train small Qwen VLMs: it asserts a Nemotron-only processor.
  - VeRL + Gym is text-only.
  - Don't switch frameworks mid-weekend.
- **Base-model shortlist:** `Qwen/Qwen3.5-2B` and `Qwen/Qwen3-VL-2B-Instruct`. Choose
  after the M1 baseline sweep in docs/plan.md.
- **Lightning:** use the hosted build.nvidia.com endpoint during development. For the
  demo, self-host the NVFP4 checkpoint with vLLM. Lightning is text-only: give it text
  observations, never images.

## Data and model-interface conventions

- **OS-Atlas** `bbox` is `[left, top, right, bottom]` normalized to [0, 1].
  **ScreenSpot-v2** `bbox` is `[x, y, w, h]` in pixels. Convert both to the canonical
  pixel xyxy schema in docs/plan.md. Drop invalid or zero-area boxes, and rows whose
  instruction text appears twice on the same screenshot.
- **Qwen3-VL / Qwen3.5 emit points on a relative 0–1000 grid.** Prompts, output
  parsing and coordinate conversion live only in `common/grounding.py`; everything
  imports from there. Spot-check visually after any conversion change.
- **Large artifacts stay out of git:** datasets, images, checkpoints, and full
  rollouts go to Hugging Face or the Brev box.

## NeMo Gym rules of thumb

- **Exceptions abort the whole run.** An exception in a tool or `verify()` aborts the
  entire `gym eval run`. Catch it, and return `reward=0, mask_sample=True` with a
  `failure_reason`.
- **Declare row fields.** Every row field `verify()` needs (bbox, img_w/h, platform,
  data_type) must be declared on the VerifyRequest subclass, or it's silently
  dropped. Put slice fields on the VerifyResponse so they reach the metrics.
- **Avoid base64 screenshots.** Prefer `file://` image URLs when Gym and vLLM share
  the box.
- **`simple_agent` doesn't stop on a `done` tool.** It stops when the model replies
  without a tool call, or at `max_steps`.
- **Failure analysis:** after every comparable pair of eval runs, run the
  **BLADE** skill (`.agents/skills/nemo-gym-blade-analysis`, also under
  `.claude/skills/`) on the artifacts. Let its failure taxonomy drive the next data,
  prompt or verifier change. Use `nemo-gym-reward-profiling` to choose GRPO prompts.

## Compute

- **Pinned GPU stack:** vLLM image v0.31.0 (by digest), torch 2.13.0+cu130,
  transformers 5.17.0, TRL 1.16.0. Never `pip install -U` these.
  - Serving Lightning on the H200 requires the vLLM #61603 patch (docs/gpu-box.md §1).
  - Use `brev login` (browser flow), not `--token`.

- **One shared Brev H200** (`permanent-tan-marmot`) at ~$6.90/hr; about 23 GPU-hours
  for the whole event.
  - Announce GPU jobs in team chat, and run them in tmux.
  - Stop the instance when nobody is using it.
- **Log every GPU run** in `docs/runs.md`: command, config, start/end, result.

## Working norms

- Pin versions for anything that runs on the GPU box, and record the exact command
  used for each training or eval run.
- Every reported number must be reproducible from a committed script and config.
