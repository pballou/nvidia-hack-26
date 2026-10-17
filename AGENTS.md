# AGENTS.md

Guidance for coding agents and teammates working in this repo.

## Project

A GUI visual-grounding model (screenshot + instruction → click point), fine-tuned
from a small open VLM, and a browser agent in which Nemotron 3.5 Lightning plans
tasks and calls the grounding model as a tool. Eval and training run through
NVIDIA NeMo Gym environments. See README.md for the overview.

## Hard constraints

- **Parameter budget:** the hackathon caps total parameters across the models in
  the submission at 33B. Lightning uses about 30B of that, so the grounding VLM must
  be ≤ ~3B total parameters, vision encoder included. Check exact counts from the
  safetensors metadata before adopting any model.
- **Never train on ScreenSpot-v2.** It is the held-out benchmark. Deduplicate
  training images against it (perceptual hash) before training.
- **Choose checkpoints on a dev split carved from the training data**, not on
  ScreenSpot-v2.
- **Public repo:** commit only public datasets and the sandbox app. No proprietary
  screenshots, customer data, credentials, or internal-system details. Secrets live
  in `.env` / NeMo Gym `env.yaml`, both gitignored.
- **Compute is metered** (Brev credits). Stop GPU instances when idle, and prefer
  hosted endpoints for development that doesn't need the GPU.

## Data conventions

- OS-Atlas annotations: `{img_filename, instruction, bbox}` (desktop and AndroidWorld
  also have `data_type`), with `bbox` = `[left, top, right, bottom]` normalized to
  [0, 1]. Some boxes fall outside [0, 1] or have zero area; drop them rather than
  clipping silently.
- Use one coordinate convention end to end (data, prompt, model output, reward,
  eval). Convert at the edges, and visually spot-check a sample after every
  conversion.
- Keep large artifacts (datasets, images, checkpoints, rollouts) out of git. Publish
  them to Hugging Face instead.

## Working norms

- Pin versions for anything that runs on the GPU box, and record the exact command
  used for each training or eval run.
- Every reported number must be reproducible from a committed script and config.
