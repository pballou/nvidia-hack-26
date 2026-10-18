# Best Use of Red Hat vLLM: GUI Assist

> **Status:** draft, Sun 2026-10-18. Collected from [runs.md](runs.md), [results.md](results.md),
> [local-mac.md](local-mac.md), [gpu-box.md](gpu-box.md) and `training/grpo.py`; nothing was
> re-run for this write-up. Owner: Kyle (serving).

## 1. Summary

GUI Assist is a vision-only accessibility agent. Nemotron 3.5 Lightning plans, and a fine-tuned
Qwen3.5-2B grounder turns "the gear icon in the top-right" into a click. vLLM serves every
generative model in the system, at every stage (the only other model is Parakeet TDT 0.6B for
speech input, in NeMo-Speech.cpp):

- **Serving:** one vLLM process on the H200 serves the 2B VLM under two names, `base` (screen
  descriptions) and `grounder` (our LoRA adapter), switched per request.
- **Measurement and training:** every NeMo Gym eval and profile hits that same server script, and
  GRPO rollouts come from vLLM colocated with TRL on the same GPU.
- **On a MacBook:** vllm-metal, the vLLM project's Apple-GPU plugin, serves official BF16
  Lightning and one Qwen3.5-2B instance with the same `base` / `grounder` LoRA switching as the
  H200. The demo runs fully local.

Measured (§4):
- **Efficiency:** one process for base and tuned keeps the system at **32.6B** by model-card
  counts, under the 33B cap. Two instances would be 34.6B. Serving stayed up while SFT and GRPO
  trained on the same H200.
- **Throughput:** Gym's GRPO profile drew **32,000 sampled replies** (4,000 screenshots × 8) in
  about **7 minutes**, at 0.40 of the GPU.
- **Latency:** grounding p50 **0.64–0.75 s** from the Mac through an SSH tunnel. Lightning on
  vllm-metal takes about **1.2 s** per idle step, against a p50 of about 26 s on the hosted free
  tier.
- **Reliability:** the same adapter as a runtime LoRA on two vLLM backends (CUDA on the H200,
  Metal on the Mac) gave **87.0% vs 87.0%** on 200 dev rows, with identical prompt tokens on every
  row and 99% of hit/miss decisions agreeing. GRPO rollout log-probs match the trainer to
  0.01–0.03.

## 2. How vLLM fits

| Role | Where | What vLLM serves | Who calls it |
| --- | --- | --- | --- |
| Clicks and screen descriptions | H200 `:8001` | Qwen3.5-2B as `base` and `grounder` (LoRA on, per request) | The agent's tools, the console, Gym `web_agent` |
| Measurement | H200 `:8001` / `:8004` / `:8005` | The same server script, with the adapter under test as `grounder` | Every `gym eval run` and `gym eval profile`: baselines, dev picks, ScreenSpot-v2-clean, GRPO profiles |
| Training rollouts | H200, inside the trainer | TRL `vllm_mode="colocate"` | `GRPOTrainer`: 16,176 rollouts over two rounds, each scored by the Gym `/verify` |
| Planner and data engine | Mac `:8000`, vllm-metal | Official BF16 Lightning as `lightning` | The agent, Gym `web_agent`, 1,003 of the SFT instruction rewrites |
| Fully local grounder | Mac `:8011`, vllm-metal | Qwen3.5-2B as `base` and `grounder` (the final LoRA), one instance | The demo recording |

**Why it's central, not bolted on:**
- **The served names are the contract.** All code (agent, Gym envs, eval scripts) is written
  against `base` and `grounder` on an OpenAI-compatible endpoint ([gpu-box.md](gpu-box.md) §3). A
  new adapter is a serve flag, not a code change. The night loop's `finish()` put the final
  adapter behind `grounder` with no other changes.
- **The parameter cap depends on it.** Per-request LoRA lets base and tuned count once, on the
  H200 and on the Mac. Without it the base-vs-tuned design breaks the 33B rule.
- **What we measure is what we serve.** Gym evals, the GRPO reward rollouts and the demo use the
  same weights, processor, image cap and prompt (`common/grounding.py`).

## 3. Configuration and what each choice buys

### 3a. The grounder server (H200)

```bash
scripts/serve_vlm.sh qwen3.5 8001 --enable-lora --max-lora-rank 64 --lora-modules grounder=<adapter>
# expands to: vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd… --model Qwen/Qwen3.5-2B --served-model-name base Qwen/Qwen3.5-2B
#   --gpu-memory-utilization 0.30 --max-model-len 16384 --max-num-seqs 128 --max-num-batched-tokens 16384
#   --limit-mm-per-prompt '{"image": 1, "video": 0}'
#   --mm-processor-kwargs '{"size": {"longest_edge": 2073600, "shortest_edge": 65536}}'
#   --allowed-local-media-path /home/ubuntu/data --default-chat-template-kwargs '{"enable_thinking": false}'
```

| Setting | Why |
| --- | --- |
| `--enable-lora --max-lora-rank 64 --lora-modules grounder=…` | Base and tuned from one process. Our adapter (r16, α32) also targets Qwen3.5's linear-attention projections. Before training, vLLM loaded a dummy adapter with the exact PEFT config and ignored 0 language-model modules (gate D1). |
| `--mm-processor-kwargs` 2 MP cap | The same cap as training and the Gym rows (`MAX_PIXELS = 2_073_600`), about 2,025 image tokens per screenshot. Without it, vLLM would resize screenshots differently from what the model was trained on. |
| `--limit-mm-per-prompt '{"image": 1, "video": 0}'` | One screenshot per request, and no video budget in memory profiling. |
| `--allowed-local-media-path /home/ubuntu/data` | Gym rows use `file://` screenshot paths, so multi-MB images never cross Gym's HTTP servers as base64. |
| `enable_thinking: false`, no `--reasoning-parser` | Qwen3.5 answers directly. With a reasoning parser, the answer comes back as `reasoning` with `content: null` (vLLM #53284). |
| `--max-num-batched-tokens 16384`, `--max-num-seqs 128` | About 8 screenshot prefills per scheduler step, for Gym concurrency 64 (evals) and 192 (profiles). |
| `--gpu-memory-utilization 0.30` (0.40 for the loop's eval server) | About 42 GiB, leaving room for SFT (peak 8.6 GiB) and GRPO (peak 45 GiB) on the same H200. |
| Image pinned by digest, v0.31.0 | Same vLLM core as vllm-metal 0.31.0 on the Mac, and inside TRL 1.16.0's supported range. |
| No `--speculative-config` | Draft and MTP models count toward the 33B cap. |

### 3b. GRPO rollouts (TRL colocate)

`training/grpo.py` runs `GRPOTrainer` with `vllm_mode="colocate"`, `vllm_gpu_memory_utilization`
0.25, `vllm_max_model_length` 3072 and `vllm_importance_sampling_correction` on. Server mode
isn't an option on one GPU, because TRL's client only does NCCL weight sync. TRL has no CI for
VLM GRPO with colocate, so we read the installed sources and fixed four things:

1. **Double image expansion.** TRL passes prompt ids with the image already expanded, and vLLM
   expands it again. We collapse each image-pad run to one token so vLLM's expansion reproduces
   TRL's ids exactly, and assert it. The 3,072-token limit fails loudly if it ever doubles.
2. **Left padding:** the processor must pad on the left, so Qwen3.5's image positions line up.
3. **The same 2 MP cap** as the server. TRL passes no `mm_processor_kwargs`, so screenshots are
   pre-resized.
4. **Weight drift.** Each sync merges the LoRA, pushes, then unmerges in bf16, and that drifted
   the base weights a little every step. We restore an exact snapshot after each unmerge.

Result: vLLM-vs-trainer log-prob difference about 0.01–0.03 and importance ratio ≈ 1, so the
rollouts the reward scores are the ones the trainer learns from.

### 3c. Lightning on the Mac (vllm-metal)

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --served-model-name lightning --port 8000 \
  --max-model-len 65536 --max-num-seqs 2 --gpu-memory-utilization 0.68 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3
```

- **Official BF16 only** (the Lightning bounty requires an official NVIDIA checkpoint).
  `rewrite.py` and `agent_eval.sh` check `/v1/models` and refuse anything else for reported runs.
- **Memory:** 0.60 fails, because the Metal limit is 115.45 GB and weights plus overhead are about
  69.9 GB. 0.68 (about 78.5 GB) works and leaves room for the local grounder.
- **Structured outputs:** `qwen3_coder` and `nemotron_v3` return tool calls and reasoning as
  separate fields. Thinking is switched per request with `chat_template_kwargs`.
- **No speculative decoding.** The model card's serve recipes enable DSpark, which would count
  toward the cap.
- **Schema checks in the agent:** vllm-metal doesn't enforce the tool schema, so the agent
  validates every call and sends back a corrective tool result.

### 3d. The fully local grounder (vllm-metal)

- **Base + LoRA in one instance, as on the H200** (`LORA=` mode, `--enable-lora
  --max-lora-rank 64`). `base` writes the screen summaries and `grounder` clicks.
- **The adapter needs re-keyed module names.** vllm-metal loads Qwen3.5 through mlx-vlm, whose
  module names differ from PEFT's (`model.layers.N.…` vs `model.language_model.layers.N.…`). The
  original adapter fails with "matched 0 wrapped modules". `scripts/gate_g_lora_keys.py` writes a
  copy with the prefix swapped and the tensors unchanged, and then vllm-metal wraps all 186
  modules. The published adapter keeps the original PEFT keys.
- **Merged checkpoint as the fallback.** `scripts/gate_g_merge.py` merges on CPU (peft's merge
  equals the manual `W + 2·B·A`, max difference 0.0), but it serves only `grounder`. That would
  mean tuned-model summaries, and those lost pay-50's confirm dialog (see §4d).
- **`--max-num-batched-tokens 2560`** cut the startup memory profile's overhead from 6.59 GB to
  about 2.25 GB. That's what lets the grounder (a 9.24 GB budget at `--gpu-memory-utilization
  0.08`) fit next to Lightning.
- **`--max-num-seqs 1`**, multimodal processor cache off, no prefix caching. A watchdog stops the
  grounder on memory pressure or a Lightning slowdown, and requests wait while Lightning is busy.

## 4. Measured results

### 4a. Efficiency

- **One process, two roles:** base and tuned from one process keep the system at 32.6B by
  model-card counts (Lightning 30B + Qwen3.5-2B + Parakeet 0.6B). Two instances would be 34.6B.
- **Serving, eval and training on one H200, at the same time:**
  - **The night loop:** `:8001` (0.30) kept serving the Mac's agent experiments the whole time.
    Alongside it, the loop's eval server (0.40) ran the Gym profiles and dev evals, and GRPO
    trained with its own colocated vLLM (0.25).
  - **GRPO speed:** 19.6–20.6 s per step (48 rollouts, then a training step). Sharing the GPU
    with a full SFT run, it was 31.9 s per step, still under our 35 s gate.
- **Shorter answers:** the fine-tuned grounder answers in about half the tokens of base
  (37.6 → 16.7 output tokens), so each grounding call decodes less.

### 4b. Throughput (Gym through vLLM)

| Job | Requests | Time | Rate | Setup |
| --- | --- | --- | --- | --- |
| Night-loop round-1 profile | 4,000 screenshots × 8 samples at T=1.0 = 32,000 | about 7 min (19:41 → 19:48) | about 75 replies/s | `:8004`, 0.40 of the GPU, Gym concurrency 192 |
| Loop rehearsal 1 profile | 400 × 8 = 3,200 | 49 s | about 65 replies/s | Same loop code and settings |
| GRPO round 1 rollouts | 170 steps × 8 prompts × 6 = 8,160 | 170 steps at 20.6 s/step (rollouts and training) | — | Colocated vLLM, 0.25 |
| Lightning rewrites, local (Mac) | 120 batched requests, 1,304 rows | 12.5 min | about 104 rows/min | vllm-metal `:8000`, client concurrency 3 |
| Lightning rewrites, hosted (rate-limited) | 405 batched requests, 3,899 rows | 56 min | about 69 rows/min | build.nvidia.com, capped at 36 requests/min |

Each grounding request carries a screenshot of about 2,000 image tokens. When the hosted API
rate-limited at about 18:00 on Saturday, the Mac's vllm-metal server joined the data engine,
working the queue from the other end (`--reverse`). 1,003 of the final rewrites came from it
([METHOD.md](../data_prep/rewrite/runs/os_atlas_final_v4/METHOD.md)). The hosted run's 36
requests/min is our client cap for the free tier's rate limit.

### 4c. Latency

| Path | p50 | p90 | Conditions |
| --- | --- | --- | --- |
| Grounding: H200 `grounder`, Mac client through the tunnel | 0.64–0.75 s | 0.80–1.62 s | Gate G, 200 sequential dev rows, two runs (the 0.64 s run shared `:8001` with an agent eval) |
| Grounding inside the agent (`base`) | 0.73 s | 0.84 s | First sandbox read |
| Screen description (`base`, H200) | 1.15 s | 1.80 s | Same read |
| Grounding: runtime LoRA on vllm-metal | 1.86 s | — | Gate G, 200 sequential dev rows |
| Grounding: merged on vllm-metal | 1.77 s | 1.82 s | Lightning idle |
| Screen description (`base`, vllm-metal) | 2.9–4.5 s | — | Spot-check on 3 demo screenshots |
| Lightning step, vllm-metal | about 1.2 s | — | Idle server, short probe, thinking off |
| Lightning plan call inside the agent, vllm-metal | 4.1 s | 5.5 s | All 12 tasks, thinking off, full observations |
| Lightning, hosted free tier | about 26 s | max 94 s | 17 calls, thinking off; development only |

The gap between the idle probe and the in-agent plan call is prefill: prefix caching is off for
Nemotron-H on vllm-metal, so every planner step re-prefills its observation. We keep
observations compact for that reason ([bounty-lightning.md](bounty-lightning.md) §4.2).

### 4d. Reliability and parity

- **Same adapter, two vLLM backends** (Gate G, [local-mac.md](local-mac.md)): the H200 serves
  the final adapter as a runtime LoRA on CUDA, and the Mac serves it as a runtime LoRA on Metal.
  On the same 200 dev rows:
  - `prompt_tokens` was identical on all 200;
  - the hit rate was 87.0% on both;
  - 99.0% of hit/miss decisions agreed (one discordant row each way);
  - 158 of 200 replies were byte-identical.
  The merged checkpoint on Metal gave the same 87.0% and 99.0% (159 byte-identical, median
  point distance 0 on the 0–1000 grid), so runtime LoRA and the merge give the same answers.
- **No LoRA leakage into `base`:** `base` screen summaries at T=0 were byte-identical in four
  conditions: a fresh base + LoRA server, alternating `grounder` → `base` requests (the agent's
  pattern), after a 200-request `grounder` burst, and a plain server without `--enable-lora`.
- **Why `base` summaries matter:** the tuned adapter writes a summary as one JSON line, and the
  agent's 240-character line clip then cut it. Fully local pay-50 failed that way, because
  Lightning never saw the "Confirm your payment" dialog. Serving `base` from the same instance
  fixed it.
- **No serving failures in the reportable agent evals:** 0 VLM infra errors and 0 masked rollouts
  across the three Gym `web_agent` arms (108 episodes).
- **The training loop ran clean:** 16,176 colocated rollouts scored with 0 reward-call failures,
  peak 45 GiB, and no crashes.

## 5. Limitations and next steps

- **No controlled serving benchmark.** We didn't run `vllm bench serve`. The numbers come from
  real jobs, and several latencies were measured on a shared GPU or through a tunnel. Each row in
  §4c says which.
- **T=0 evals aren't bit-exact.** The same adapter scored 85.66 and 85.86 on dev in two launches
  (vLLM batching), about 0.2 points of run-to-run noise. A re-run of the pilot adapter reproduced
  81.32% exactly.
- **Adapter swaps restart the server.** The final swap took `:8001` down for about 90 s, and the
  final agent arm then ran on a fresh process. BLADE found summarizer differences on identical
  inputs there, which confounds that arm. **Next:** runtime LoRA loading with several adapters
  resident (`--max-loras`), so base, SFT and final arms can interleave on one process.
- **vllm-metal gaps:** no prefix caching for Nemotron-H; LoRA for Qwen VLMs needs the adapter
  re-keyed to mlx-vlm's module names; and the fully local setup fits only with other memory-heavy
  apps closed.
- **Lightning on the H200 was prepared but never run.** That config (fp8 KV cache, prefix caching,
  the backported vLLM #61603 fix for Hopper) is in [gpu-box.md](gpu-box.md) §2. Lightning stayed
  on the Mac.

## 6. Reproduce

The H200 (pinned stack in [gpu-box.md](gpu-box.md); every run logged in [runs.md](runs.md)):

```bash
source /home/ubuntu/serving.env
scripts/serve_vlm.sh qwen3.5 8001 --enable-lora --max-lora-rank 64 \
  --lora-modules grounder=/home/ubuntu/ckpt/night/20261017-193859/round_1/grpo/adapter
curl -s localhost:8001/v1/models    # expect base, Qwen/Qwen3.5-2B, grounder
# A Gym eval through vLLM (dev + ScreenSpot-v2-clean), on its own ports (:8005 / :11011 / :12011)
python -m training.gymctl eval --adapter <adapter> --name <name> --ssv2 --gpu-util 0.30
```

The Mac ([local-mac.md](local-mac.md) §0–§3b): install vllm-metal 0.31.0 from the release wheels,
serve Lightning with the command in §3c, then base + the LoRA in one instance:

```bash
~/.venv-merge/bin/python scripts/gate_g_lora_keys.py --adapter <adapter dir> --out ~/models/gui-assist-mlx-keys/<name>
python3 -I scripts/gate_g_watch.py --pidfile ~/models/logs/local-grounder-8011.pid --port 8011 \
  --log ~/models/gate_g/watch.jsonl &
LORA=~/models/gui-assist-mlx-keys/<name> GPU_UTIL=0.08 MAX_BATCHED=2560 MIN_FREE_PCT=26 scripts/run_local_grounder.sh
```

Gate G's parity check is `scripts/gate_g.py` (200 dev rows to both servers, then the verdict).
