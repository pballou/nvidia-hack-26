# Local Mac demo (Patrick's M4 Max, 128 GB)

How much of the demo can run on the laptop. Verified 2026-10-17 against vllm-metal,
mlx-lm and llama.cpp sources, releases and issues. Since then, Lightning (§0–§2) and the
tunnel (§3a) have run the agent evals and are the demo setup; §3b / Gate G is in progress
([status.md](status.md)). Owner: Patrick (stream C). **Time-box all Mac-local work to about 1.5 h.**

## Decision: hybrid first, fully local if it passes, H200 as the fallback

| Mode | Lightning (planner) | Grounder | When |
| --- | --- | --- | --- |
| **Hybrid (primary)** | Mac, `vllm serve` via **vllm-metal**, **official NVIDIA BF16** | H200 vLLM, base + LoRA (gpu-box.md §3), via a tunnel on 8001 | If Gate L passes |
| Fully local | Mac, vllm-metal | Mac, vllm-metal, **merged** tuned checkpoint for both roles | Only if Gate G also passes |
| All-H200 (fallback) | H200, official BF16 + #61603 patch (gpu-box.md §2) | H200 (gpu-box.md §3) | If Gate L fails |

**Why hybrid:**
- **Lightning is the easy half to make local:** stock weights, text-only, and the same
  chat template and parsers.
- **The grounder stays on the exact eval stack:** same weights, processor and
  base/grounder switching. So the coordinates in the video match our measured
  numbers.
- **It frees the H200 on Sunday.** The demo doesn't depend on the Lightning Hopper
  patch.
- **It keeps "vLLM serves both models" true.** vllm-metal is the official
  vllm-project plugin for Apple GPUs.

**Parameter cap:**
- Hybrid: one Lightning (30B) plus one grounder instance (2B) plus Parakeet TDT 0.6B
  (voice input), so 32.6B by model-card counts.
- Fully local: Lightning plus **one** merged grounder used for both clicks and screen
  summaries (plus Parakeet, same 32.6B). Never load a second VLM copy.
- Show base-vs-tuned from the H200 or the Gym numbers, not from a second local
  instance.
- No speculative decoding or MTP flags anywhere.

## What's verified

| Item | Status |
| --- | --- |
| vllm-metal v0.31.0 (released 2026-10-13, tracks vLLM core 0.31.0, same as the H200) | Confirmed |
| Nemotron-H / Lightning on vllm-metal | **Experimental.** Prefix caching is off for it, so every planner step re-prefills; keep prompts short and thinking off. |
| Lightning tool calling (`qwen3_coder` + `nemotron_v3`) on vllm-metal | **Works with official BF16** (quick check, 2026-10-17): median **1.2 s/step** with thinking off; reasoning comes back separately with thinking on. With `tool_choice: "required"`, argument names were 5/5 correct; with `auto`, 4/5. |
| Qwen3.5 / Qwen3-VL image input on vllm-metal | Experimental. It uses upstream vLLM's HF processor, so preprocessing matches the H200. |
| LoRA on vllm-metal for Qwen VLMs | Not usable in the stable release (adapter key mismatch, fixed only in dev builds), so serve the **merged** checkpoint locally |
| Fallback planner: llama-server (already installed, b11429) with `ggml-org` Lightning GGUF Q8_0 | Tool parser confirmed; local, but not vLLM |
| llama.cpp or mlx-vlm for the grounder | **Don't.** Their preprocessing differs (llama.cpp pads), there's conversion work, and no time to re-validate. |

## 0. Install vllm-metal (pinned; don't use the stock `install.sh`)

The stock script calls a rate-limited GitHub API and isn't pinned. Install the two
release wheels directly instead:

```bash
uv venv ~/.venv-vllm-metal --python 3.12 --seed
source ~/.venv-vllm-metal/bin/activate
uv pip install "https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl"
uv pip install "https://github.com/vllm-project/vllm-metal/releases/download/v0.31.0/vllm_metal-0.31.0-cp312-cp312-macosx_15_0_arm64.whl"
uv pip list 2>/dev/null | grep -E '^(vllm|vllm-metal|mlx|mlx-lm|mlx-vlm|transformers) '
# expect vllm 0.31.0+cpu, vllm-metal 0.31.0, mlx 0.32.1, mlx-vlm 0.6.x, transformers 5.1x (<5.18)
```

## 1. Weights (download once, then serve offline)

**Official NVIDIA BF16 only** (bounty requirement; no mlx-community/GGUF quants for anything we report or demo).
mlx-lm's `nemotron_h` loader reads NVIDIA's HF checkpoint directly: it drops the `mtp.*` weights and converts the conv1d layout.

```bash
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16     # 65.8 GB
# Gate G only: the merged grounder that stream B exports for publishing
rsync -avP permanent-tan-marmot:/home/ubuntu/ckpt/<run>/merged/ ~/models/grounder-merged/
```

## 2. Lightning on :8000

```bash
source ~/.venv-vllm-metal/bin/activate
HF_HUB_OFFLINE=1 vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --served-model-name lightning --host 127.0.0.1 --port 8000 \
  --max-model-len 65536 --max-num-seqs 2 --gpu-memory-utilization 0.68 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3
# Verified 2026-10-17 on the M4 Max:
#   - 0.60 fails ("not enough Metal memory for KV cache"): the Metal limit is 115.45 GB, and weights + overhead are ~69.9 GB.
#   - 0.68 works (~78.5 GB). A local grounder at 0.12 still fits.
#   - Healthy in ~15 s from cache. /v1/models root = nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.
# Never add --speculative-config.
```

## 3a. Hybrid: grounder from the H200

```bash
brev login     # browser flow, right before using it
ssh -N -o ControlPath=none -o ExitOnForwardFailure=yes -L 8001:localhost:8001 permanent-tan-marmot
# On the box: gpu-box.md §3 with the adapter (served names base + grounder). No Lightning container needed.
```

## 3b. Fully local grounder on :8001 (only after Gate G passes)

```bash
source ~/.venv-vllm-metal/bin/activate
export GROUNDER_MAX_PIXELS=2073600 GROUNDER_MIN_PIXELS=65536   # = common/grounding.py (2 MP, frozen at gate D1)
HF_HUB_OFFLINE=1 vllm serve ~/models/grounder-merged --served-model-name grounder \
  --host 127.0.0.1 --port 8001 --max-model-len 16384 --max-num-seqs 1 --gpu-memory-utilization 0.12 \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --mm-processor-kwargs "{\"size\": {\"longest_edge\": ${GROUNDER_MAX_PIXELS}, \"shortest_edge\": ${GROUNDER_MIN_PIXELS}}}" \
  --default-chat-template-kwargs '{"enable_thinking": false}' --no-enable-prefix-caching
# No --reasoning-parser. For Qwen3-VL-2B: drop --default-chat-template-kwargs and send temperature 0.
# In this mode the app's summary_model is "grounder" (one instance for both roles).
```

## Gates

**Gate L: local Lightning.** Today by ~5pm, ≤ 60 min, mostly download time.
1. Smoke test:
   ```bash
   curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"lightning","temperature":1.0,"top_p":0.95,"max_tokens":512,"chat_template_kwargs":{"enable_thinking":false},"tools":[{"type":"function","function":{"name":"click","description":"Click a UI element","parameters":{"type":"object","properties":{"target":{"type":"string"}},"required":["target"]}}}],"messages":[{"role":"user","content":"Click the Sign in button."}]}' | python3 -c 'import json,sys;r=json.load(sys.stdin);c=r["choices"][0];print(c["finish_reason"],c["message"].get("tool_calls"),r["usage"])'
   ```
   Expect `finish_reason` `tool_calls` and a `click(target=…)`.
2. Run 20 real agent turns through the app code with thinking off, plus 5 with
   thinking on.
3. **Pass if:**
   - at least 19/20 turns produce valid tool calls with parseable arguments;
   - reasoning comes back separate from the content;
   - no engine crashes;
   - median step ≤ ~8 s at a realistic prompt length.
4. **Validate arguments in the agent anyway.** On the community 4-bit build, 1–2 of 5
   calls used wrong argument names even with `tool_choice: "required"` (vllm-metal
   doesn't enforce the schema). The agent checks every call against the schema
   (`agent/session.py:validate_args`) and sends back a corrective tool result.
5. **If BF16 on Mac fails,** use the H200 official BF16 (gpu-box.md §2) through the
   tunnel. llama-server with a community GGUF is **emergency only and doesn't
   count for the bounty**:
   ```bash
   hf download ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf --local-dir ~/models/lightning-gguf
   llama-server -m ~/models/lightning-gguf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf --alias lightning \
     --host 127.0.0.1 --port 8000 -ngl 99 -fa on -c 65536 -np 1 --jinja --temp 1.0 --top-p 0.95 --no-webui
   ```
   Don't use the model card's `:Q4_K_M` tag (that file doesn't exist), and add no
   `--spec-type`/`-md`/mtp files.

**Gate G: local grounder.** Sat ~8pm, ≤ 45 min.
1. Serve the merged checkpoint (§3b) locally, and the H200 grounder on tunnel port
   18001 (`ssh -N -L 18001:localhost:8001 permanent-tan-marmot`).
2. Send the same 200 dev rows to both. Use images already run through
   `preprocess_screenshot`, sent as PNG, with the exact prompt from
   `common/grounding.py`.
3. **Pass if:**
   - `usage.prompt_tokens` is identical on every row;
   - the hit rate is within 2 points;
   - at least 90% of the hit/miss decisions match.
4. **If GRPO replaces SFT overnight,** re-merge and re-run Gate G at 7am Sunday
   (~20 min).
5. **If it fails,** stay hybrid.

## Recording day

- **Record the hybrid take first;** it's the safety take. Then the fully local take,
  if Gate G passed.
- **Pre-download everything and set `HF_HUB_OFFLINE=1`.** Keep the Mac on power, with
  Low Power Mode off.
- **Use a fixed Playwright viewport** (1280×800, `device_scale_factor=1`), and always
  run `preprocess_screenshot` client-side.
- **Pause Gym evals** on the H200 while recording, so the grounder isn't contended.
- **Only say "runs locally on a MacBook" about what actually ran there.**

## Gate G results

Scripts: [scripts/gate_g_merge.py](../scripts/gate_g_merge.py) (merge),
[scripts/run_local_grounder.sh](../scripts/run_local_grounder.sh) (serve, §3b),
[scripts/gate_g_watch.py](../scripts/gate_g_watch.py) (watchdog),
[scripts/gate_g.py](../scripts/gate_g.py) (rows, requests, verdict). Artifacts are in
`~/models/gate_g/` on Patrick's Mac.

### Attempt 1: Sat 2026-10-17, 19:15–19:40 EDT (T4). Not run: stopped on memory pressure

**Merge (done, reusable):**
- **Inputs:**
  - Base `Qwen/Qwen3.5-2B@15852e8c`. That's the Hub's latest commit (2026-03-02), so it's
    the same snapshot the H200 downloaded.
  - Adapter `/home/ubuntu/ckpt/sft-template/adapter`, copied with scp only.
    `adapter_model.safetensors` sha256 `63b4dcd1…`. r16/α32 on 186 modules: q/k/v/o,
    MLP, and GDN `in_proj_qkv/z/a/b` + `out_proj`.
- **Command:** `~/.venv-merge/bin/python scripts/gate_g_merge.py --adapter
  ~/models/adapters/sft-template --out ~/models/grounder-merged-sft-template`. It ran on
  CPU in 14 s with torch 2.13.0, transformers 5.17.0 and peft 0.21.2.
- **Output:** weights sha256 `b8460afd…`.
  - Only the 186 targeted tensors differ from base. Everything else is bit-identical:
    vision tower, `mtp.*`, and every config, tokenizer and template file.
  - peft's float32 merge equals the manual `W + 2·B·A` exactly (max diff 0.0).
- **Memory:** the script never holds the whole model. A first version that loaded the
  model in fp32 pushed pressure to yellow for ~8 s (19:22:21–27) before its own guard
  aborted it.

**Serving attempt** (`GPU_UTIL=0.05`, port 8011, mm processor cache off, watchdog armed first),
started after the orchestrator's GO (T2's eval had finished at ~19:30):

| Time | What happened |
| --- | --- |
| 19:32:09 | Watchdog baseline: Lightning probe median 0.157 s, ITL 42 ms, prefill 1,026 tok/s, 23% free, pressure normal |
| 19:33:18 | vllm-metal resolved `Qwen3_5ForConditionalGeneration`, loaded the merge through MLX-VLM (0.76 s), and set up hybrid GDN paged attention |
| 19:33:24 | The startup profile ran a dummy 8,192-token text forward (no vision pass: vllm-metal only profiles encoders for adapters with `profile_features`, i.e. Gemma 4). Then KV sizing failed with `model_memory=4.43GB, overhead=6.59GB, kv_budget=-5.24GB`. Pressure turned yellow during the profile; the watchdog sent SIGTERM |
| 19:33:44 | Two EngineCore children ignored SIGTERM and were SIGKILLed. Lightning (PGID 38782) was not touched |
| 19:37:18 | The first Lightning request afterwards took 4.0 s (warm: 0.16 s). The kernel had compressed Lightning's idle memory (compressor 21 → 60 GB, 4.2 GB to swap). After that request: normal pressure, compressor 20 GB, 23% free. Lightning's counters show no team requests in that window |

**Findings:**
- **vllm-metal 0.31.0 loads Qwen3.5-2B and runs its text stack.** The hybrid GDN +
  attention layers executed on Metal during the profile.
  - The vision tower loaded (MLX-VLM) but hasn't run yet. Its activations for a 2 MP image
    (8,100 patches) also aren't in the util budget, so the first image request adds memory
    on top of it.
  - Preprocessing goes through upstream vLLM's HF `Qwen3VLProcessor` (read from the
    source), so `--mm-processor-kwargs` applies exactly as on the H200.
  - No request was served, so image support, prompt-token identity and accuracy are
    still unmeasured.
- **Memory is the blocker, not model support.**
  - `--gpu-memory-utilization` multiplies Metal's working set (115.45 GB). For this hybrid
    model, the whole budget is allocated at startup.
  - The budget must hold the weights (4.43 GB), the profile overhead, and the KV cache.
    The overhead was 6.59 GB at the default `max_num_batched_tokens=8192`; most of it is
    the dummy forward's logits (8,192 × 248k vocab).
  - So 0.12 (13.9 GB) would start; 0.05 can't.
- **The real headroom is ~8–10 GB, and the grounder needs ~9–11 GB plus the first
  image's vision pass.**
  - With Lightning's memory wired, the Mac has ~23% free.
  - macOS turned yellow near 15–16% free, and once at 43% "free" during the fast
    allocation.
- **MLX unwires Lightning's ~80 GB a few seconds after its last request.**
  `memory_pressure` then reads ~80% free. A new allocation makes the kernel compress
  Lightning, and Lightning's next request pays for it. Measure headroom only right after
  a Lightning request; the launcher now does this.
- **§2's "a local grounder at 0.12 still fits" no longer holds** with Chrome, VS Code and
  Teams open. It's true of Metal's budget arithmetic, not of this Mac's physical memory.

**Retry (only on the orchestrator's GO, after Teams and most Chrome tabs are closed):**

```bash
python3 -I scripts/gate_g_watch.py --pidfile ~/models/logs/local-grounder-8011.pid --port 8011 \
  --log ~/models/gate_g/watch.jsonl &                     # arm first: it baselines Lightning, then waits
GPU_UTIL=0.08 MAX_BATCHED=2560 MIN_FREE_PCT=26 scripts/run_local_grounder.sh   # warms Lightning, refuses below 26% free
~/.venv-merge/bin/python scripts/gate_g.py run --manifest <os_atlas_final>/manifest.jsonl \
  --out ~/models/gate_g/sft-template --endpoint local=http://127.0.0.1:8011/v1
~/.venv-merge/bin/python scripts/gate_g.py compare ~/models/gate_g/sft-template --a local --b h200
```

- **`--max-num-batched-tokens 2560`** cuts the profile's logits from ~4 GB to ~1.3 GB. A
  2 MP image (2,025 tokens) plus the prompt still fits in one prefill chunk, as on the
  H200. Serving still projects full-vocabulary logits for every prefill row, so each image
  request adds ~1 GB of transient memory.
- **Expect some numeric drift that isn't vllm-metal being wrong:** merge rounding
  (bf16(W + 2·B·A) vs. runtime LoRA), mlx-vlm rounding RMSNorm's `1 + w` to bf16, bf16
  GDN gates, and the H200's own batch nondeterminism. If agreement misses 90%, comparing
  base against base (the base snapshot here vs. :8001 `base`) separates backend
  numerics from the merge.
- **33B cap:** the Mac copy and the H200 `grounder` coexist only for this offline check.
  The fully local demo uses the Mac grounder alone, with no H200 grounder in the loop.
- **Rows:** 200 of the 969 `dev` rows in the finalized manifest (sha256
  `4ceb7ea3d7ac8e85…`; local copy in T3's worktree `data/os_atlas_final/`), sampled with
  `random.Random(0)`. Each is sent as a `preprocess_screenshot` PNG with the qwen3_5
  prompt, temperature 0.
- **The H200 half is already collected** (19:46–19:50, sequential; `/v1/models` showed
  `grounder` = `/home/ubuntu/ckpt/sft-template/adapter`, recorded in `selection.json`):
  - 200/200 answered and parsed. Hit rate **85.0%** (desktop 85.3 on 68 rows, mobile 86.0
    on 50, web 84.1 on 82), in line with sft-template's 85.66% on the full dev split.
  - p50 0.75 s through the tunnel.
  - `prompt_tokens` = image tokens + 52–59 text tokens.
- **If the retry can't happen,** stay hybrid. The video then says "Lightning and voice run
  on this Mac; the 2B grounder runs on one H200".
- **After GRPO, merge the FINAL adapter in the morning** (FINAL.json in the 3am copy at
  `~/models/gui-assist/`) and re-run Gate G against a :8001 that serves that same adapter.
