# Local Mac demo (Patrick's M4 Max, 128 GB)

How much of the demo can run on the laptop. Verified 2026-10-17 against vllm-metal,
mlx-lm and llama.cpp sources, releases and issues. Since then, Lightning (§0–§2) and the
tunnel (§3a) have run the agent evals and are the demo setup. Gate G passed on 2026-10-17,
so the fully local grounder (§3b) works too ([Gate G results](#gate-g-results),
[status.md](status.md)). Owner: Patrick (stream C). **Time-box all Mac-local work to about 1.5 h.**

## Decision: hybrid first, fully local if it passes, H200 as the fallback

| Mode | Lightning (planner) | Grounder | When |
| --- | --- | --- | --- |
| **Hybrid (primary)** | Mac, `vllm serve` via **vllm-metal**, **official NVIDIA BF16** | H200 vLLM, base + LoRA (gpu-box.md §3), via a tunnel on 8001 | If Gate L passes |
| Fully local | Mac, vllm-metal | Mac, vllm-metal, **merged** tuned checkpoint for both roles, on :8011 | Only if Gate G also passes |
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
# Fully local grounder only: the base model, plus the LoRA adapter (scp the adapter dir only)
hf download Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc   # 4.6 GB
scp -r permanent-tan-marmot:/home/ubuntu/ckpt/<run>/adapter ~/models/adapters/<name>
```

The merged checkpoint is built on this Mac with `scripts/gate_g_merge.py` (§3b), not
copied from the box.

## 2. Lightning on :8000

```bash
source ~/.venv-vllm-metal/bin/activate
HF_HUB_OFFLINE=1 vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --served-model-name lightning --host 127.0.0.1 --port 8000 \
  --max-model-len 65536 --max-num-seqs 2 --gpu-memory-utilization 0.68 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3
# Verified 2026-10-17 on the M4 Max:
#   - 0.60 fails ("not enough Metal memory for KV cache"): the Metal limit is 115.45 GB, and weights + overhead are ~69.9 GB.
#   - 0.68 works (~78.5 GB). A local grounder fits next to it at 0.08 with --max-num-batched-tokens 2560,
#     and only with Teams/Chrome closed (§3b, Gate G results).
#   - Healthy in ~15 s from cache. /v1/models root = nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.
# Never add --speculative-config.
```

## 3a. Hybrid: grounder from the H200

```bash
brev login     # browser flow, right before using it
ssh -N -o ControlPath=none -o ExitOnForwardFailure=yes -L 8001:localhost:8001 permanent-tan-marmot
# On the box: gpu-box.md §3 with the adapter (served names base + grounder). No Lightning container needed.
```

## 3b. Fully local grounder on :8011 (only after Gate G passes)

:8001 is the H200 tunnel (§3a). The local grounder uses :8011, so both can be up at once,
which Gate G needs.

```bash
# Merge once (CPU, streaming): base Qwen3.5-2B + the adapter -> one bf16 checkpoint
~/.venv-merge/bin/python scripts/gate_g_merge.py --adapter <adapter dir> --out ~/models/grounder-merged-<name>
# Watchdog first (it stops the grounder on memory pressure or a Lightning slowdown), then the server
python3 -I scripts/gate_g_watch.py --pidfile ~/models/logs/local-grounder-8011.pid --port 8011 \
  --log ~/models/gate_g/watch.jsonl &
MODEL=~/models/grounder-merged-<name> GPU_UTIL=0.08 MAX_BATCHED=2560 MIN_FREE_PCT=26 scripts/run_local_grounder.sh
```

- **[scripts/run_local_grounder.sh](../scripts/run_local_grounder.sh) pins the flags:**
  - served name `grounder`;
  - the frozen 2 MP `--mm-processor-kwargs` (read from `common/grounding.py`);
  - `--max-num-seqs 1`, `--max-num-batched-tokens 2560`;
  - multimodal processor cache off, `enable_thinking` off, no prefix caching;
  - no reasoning parser and no speculative decoding.
- **It refuses to start** without a live watchdog, or with less than 26% free (measured
  after warming Lightning).
- **Don't use util 0.12 with the default 8,192 batched tokens.** Next to Lightning, that
  setup turned memory pressure yellow ([Gate G results](#gate-g-results), attempt 1).
- **One instance does both roles:** in this mode the app's `summary_model` is `grounder`.
  For the console and recording, see
  [Fully local mode for the video](#fully-local-mode-for-the-video).

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
1. Serve the merged checkpoint locally on :8011 (§3b), and keep the normal :8001 tunnel to
   the H200 grounder (§3a). Attempt 2 did exactly this. A second tunnel on 18001
   (`ssh -N -L 18001:localhost:8001 permanent-tan-marmot`) is optional.
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

**Verdict: PASS** (attempt 2, Sat 2026-10-17 20:22 EDT, T4). The merged sft-template
grounder on vllm-metal matches the H200 `grounder` on 200 dev rows:
- `prompt_tokens` identical on all 200;
- hit rate 86.0% vs 85.0%;
- 99.0% of hit/miss decisions agree;
- local p50 1.77 s.

It only fits in memory with Teams and most Chrome tabs closed (attempt 1).

Scripts: [scripts/gate_g_merge.py](../scripts/gate_g_merge.py) (merge),
[scripts/run_local_grounder.sh](../scripts/run_local_grounder.sh) (serve, §3b),
[scripts/gate_g_watch.py](../scripts/gate_g_watch.py) (watchdog),
[scripts/gate_g.py](../scripts/gate_g.py) (rows, requests, verdict). Artifacts are in
`~/models/gate_g/` on Patrick's Mac.

### FINAL adapter (round-1 GRPO): H200 half captured, local half pending

The night loop finished at 22:02 with FINAL = round 1 GRPO,
`/home/ubuntu/ckpt/night/20261017-193859/round_1/grpo/adapter` (dev 87.00).

- **H200 half** (Sat 22:04–22:07, sequential, same 200 rows; `selection.json` records
  :8001 `grounder` = that adapter): 200/200 answered and parsed.
  - Hit rate **87.0%**: desktop 91.2 (68 rows), mobile 84.0 (50), web 85.4 (82).
  - p50 0.64 s. :8001 was shared with T2's agent eval, so latency isn't comparable.
  - Results are in `~/models/gate_g/night-20261017-193859-round_1-grpo/`.
- **Merge** (CPU, 25.5 s, pressure normal):
  - Adapter `~/models/gui-assist/night-20261017-193859-round_1-grpo/`, copied from the
    box; `adapter_model.safetensors` sha256 `54ce9389…`.
  - Output `~/models/grounder-merged-night-20261017-193859-round_1-grpo`, weights sha256
    `cd120bc0…`.
  - 186 modules; peft's merge equals the manual `W + 2·B·A` (max diff 0.0).
- **Local half:** waits until Patrick has closed Teams and Chrome. Then run
  `MODEL=~/models/grounder-merged-night-20261017-193859-round_1-grpo` with
  [the procedure below](#how-to-run-gate-g-attempt-2s-exact-procedure), using
  `--endpoint local=…` only and `--out ~/models/gate_g/night-20261017-193859-round_1-grpo`.
  `compare` then pairs it with the H200 half already captured.

### Attempt 2: Sat 2026-10-17, 20:15–20:22 EDT (T4). PASS

**Setup:**
- The merge from attempt 1 (below), served with `GPU_UTIL=0.08 MAX_BATCHED=2560
  MIN_FREE_PCT=26 scripts/run_local_grounder.sh` on :8011, watchdog armed first.
- Patrick had quit Teams and closed Chrome.
- The local half ran against the H200 half collected at 19:46–19:50. At that time
  `/v1/models` showed :8001 `grounder` = `/home/ubuntu/ckpt/sft-template/adapter`, recorded
  in `selection.json`.

| Check | Local (Mac, vllm-metal, merged) | H200 (`grounder`, runtime LoRA) | Rule |
| --- | --- | --- | --- |
| `prompt_tokens` identical | 200/200 rows (image tokens + 52–56 text tokens) | same | every row ✓ |
| Hit rate | **86.0%** | **85.0%** | gap ≤ 2 points: +1.0 ✓ |
| Hit/miss agreement | **99.0%** (198/200; 2 local-only hits, McNemar p = 0.5) | | ≥ 90% ✓ |
| Latency p50 / p90 | **1.77 s** / 1.82 s (sequential, `--max-num-seqs 1`) | 0.75 s / 1.62 s (through the tunnel) | reported |

- **Per platform** (local / H200): desktop 88.2 / 85.3 (68 rows), mobile 86.0 / 86.0 (50),
  web 84.1 / 84.1 (82).
- **Agreement detail:**
  - 156/200 replies are byte-identical.
  - The median point distance is 0 and the p90 is 1 on the 0–1000 grid.
  - 5 rows differ by more than 50: the 2 local-only hits and 3 rows both servers miss.
- **Memory:**
  - Preflight: 35% free with Lightning wired. The compressor was at 4.7 GB, down from
    21 GB with Teams and Chrome open.
  - With the grounder up: 28% free at the lowest. Pressure stayed normal throughout, and
    the KV commit probe saw no compression or swap.
  - The grounder's budget at 0.08 is 9.24 GB: weights 4.43 GB, profile overhead
    ~2.25 GB (6.59 GB at the default 8,192 batched tokens), and KV/state 2.56 GB.
- **Contention:** Lightning's tiny warm probe went from 0.155 s to ~0.41–0.45 s when it
  overlapped a grounder request. gate_g.py holds local requests while Lightning is busy;
  that happened 3 times, for at most 0.51 s.
- **Artifacts:** `~/models/gate_g/sft-template/` (`selection.json`, `local.jsonl`,
  `h200.jsonl`, `summary_local_vs_h200.json`) and `~/models/gate_g/attempt2/` (server,
  watchdog and run logs).

### Fully local end-to-end check: 20:25–20:32 EDT (T4)

**Setup:** three demo tasks through the real agent, n=1 each, no config edits:
- the local grounder did both clicks and screen summaries (one instance);
- local Lightning, thinking `retry`, sandbox :8091;
- `scripts/run_agent_tasks.py --ids pay-50,download-september-statement,text-size-large
  --pin-grounder-adapter --pause-on-memory-pressure`, with `AGENT_VLM_BASE_URL`,
  `AGENT_GROUNDER_MODEL`, `AGENT_SUMMARY_MODEL`, `AGENT_LIGHTNING` and `AGENT_THINKING`
  as overrides.

**Results** (summary: `~/models/gate_g/e2e-runs/summary_20261017-202533.json`):

| Task | Verified | Claimed | Steps |
| --- | --- | --- | --- |
| pay-50 | ✓ (6/6 conditions) | ✓ | 6 |
| download-september-statement | ✗: `billing.downloaded` got `['2026-03', '2025-11', '2026-09']`, expected `['2026-09']` | ✓ | 15 |
| text-size-large | ✓ (7/7) | ✓ | 4 |

- **download-september-statement** reached September, but only after downloading two
  wrong rows. Clicks on "the download icon in the September 2026 row" kept reporting
  NO change, so the agent scrolled and retried with thinking on. It's a row-disambiguation
  miss, not an infrastructure failure.
- **Totals:** 25 tool calls, 0 invalid, 4 recoveries, 9 thinking calls.
- **Per-stage p50 (p90):**
  - plan (Lightning): 4.59 s (13.7 s). Thinking off 4.29 s (n=16), thinking on 13.0 s
    (n=9).
  - ground (local grounder): 1.00 s (1.01 s).
  - summary (local grounder, ≤600 tokens): 6.17 s (7.93 s). This is the slowest stage in
    fully local mode.
- **Memory:** pressure stayed normal the whole time, with 28% free at the lowest.
  Lightning's warm probe stayed at 0.15–0.25 s.

### How to run Gate G (attempt 2's exact procedure)

```bash
python3 -I scripts/gate_g_watch.py --pidfile ~/models/logs/local-grounder-8011.pid --port 8011 \
  --log ~/models/gate_g/watch.jsonl &                     # arm first: it baselines Lightning, then waits
GPU_UTIL=0.08 MAX_BATCHED=2560 MIN_FREE_PCT=26 scripts/run_local_grounder.sh   # refuses without the watchdog or below 26% free
~/.venv-merge/bin/python scripts/gate_g.py run --manifest <os_atlas_final>/manifest.jsonl \
  --out ~/models/gate_g/<name> --endpoint local=http://127.0.0.1:8011/v1 --endpoint h200=http://127.0.0.1:8001/v1
~/.venv-merge/bin/python scripts/gate_g.py compare ~/models/gate_g/<name> --a local --b h200
kill -TERM "$(cat ~/models/logs/local-grounder-8011.pid)"   # when done; the watchdog then exits
```

- **The launcher's guards:**
  - It warms Lightning before measuring free memory. MLX unwires Lightning a few seconds
    after its last request, and `memory_pressure` then reads ~80% free.
  - It refuses to start without a live watchdog.
  - It passes no extra args to `vllm serve`, so the pinned flags can't be overridden and
    speculative decoding can't be switched on.
  - It turns off the multimodal processor cache.
- **`--max-num-batched-tokens 2560`** cuts the startup profile's logits from ~4 GB to
  ~1.3 GB. A 2 MP image (2,025 tokens) plus the prompt still fits in one prefill chunk, as
  on the H200. Serving still projects full-vocabulary logits per prefill row, so each image
  request adds ~1 GB of transient memory.
- **Rows:** 200 of the 969 `dev` rows in the finalized manifest (sha256
  `4ceb7ea3d7ac8e85…`; local copy in T3's worktree `data/os_atlas_final/`), sampled with
  `random.Random(0)`. Each is sent as a `preprocess_screenshot` PNG with the qwen3_5
  prompt, temperature 0.
  - `compare` refuses a run whose two legs hit the same URL or the same weights.
  - Errored rows are retried on resume.
- **Expect some numeric drift that isn't vllm-metal being wrong:** merge rounding
  (bf16(W + 2·B·A) vs. runtime LoRA), mlx-vlm rounding RMSNorm's `1 + w` to bf16, bf16
  GDN gates, and the H200's own batch nondeterminism. If agreement ever misses 90%,
  comparing base against base (the base snapshot here vs. :8001 `base`) separates backend
  numerics from the merge.
- **33B cap:** the Mac copy and the H200 `grounder` coexist only for this offline check.
  The fully local demo uses the Mac grounder alone.
- **Morning step, if GRPO's FINAL adapter differs from sft-template:** re-merge and re-run
  Gate G (~20 min). Use the FINAL.json adapter from the copy at `~/models/gui-assist/`,
  run `scripts/gate_g_merge.py --adapter <it> --out ~/models/grounder-merged-<name>`, then
  `MODEL=~/models/grounder-merged-<name> scripts/run_local_grounder.sh`. Re-collect the
  H200 half while :8001 `grounder` serves that same adapter (`selection.json` records it).

### Attempt 1: Sat 2026-10-17, 19:15–19:40 EDT (T4). Not run: stopped on memory pressure

**Merge (done, reused by attempt 2):**
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
| 19:32:09 | Watchdog baseline: Lightning probe median 0.157 s, ITL 42 ms, 23% free, pressure normal |
| 19:33:18 | vllm-metal resolved `Qwen3_5ForConditionalGeneration`, loaded the merge through MLX-VLM (0.76 s), and set up hybrid GDN paged attention |
| 19:33:24 | The startup profile ran a dummy 8,192-token text forward (no vision pass: vllm-metal only profiles encoders for adapters with `profile_features`, i.e. Gemma 4). Then KV sizing failed with `model_memory=4.43GB, overhead=6.59GB, kv_budget=-5.24GB`. Pressure turned yellow during the profile; the watchdog sent SIGTERM |
| 19:33:44 | The API server had exited; two EngineCore children ignored SIGTERM and were SIGKILLed. Lightning (PGID 38782) was not touched |
| 19:37:18 | The first Lightning request afterwards took 4.0 s (warm: 0.16 s). The kernel had compressed Lightning's idle memory (compressor 21 → 60 GB, 4.2 GB to swap). After that request: normal pressure, compressor 20 GB, 23% free. Lightning's counters show no team requests in that window |

**What attempt 1 taught:**
- **`--gpu-memory-utilization` multiplies Metal's working set (115.45 GB),** and for this
  hybrid model the whole budget is allocated at startup.
  - The budget must hold the weights, the startup profile's overhead (mostly the dummy
    forward's 8,192 × 248k-vocab logits at the default batch), and the KV cache.
  - The vision tower's activations aren't in the budget at all.
- **With Teams and Chrome open, the real headroom was ~8–10 GB:** ~23% free with Lightning
  wired, and yellow near 15–16% (once at 43% "free", during a fast allocation).
- **MLX unwires Lightning's ~80 GB a few seconds after its last request.** A big allocation
  then makes the kernel compress Lightning, and Lightning's next request pays for it.
- **§2's "a local grounder at 0.12 still fits"** is true of Metal's budget arithmetic,
  not of this Mac's physical memory with the browser and Teams open.

## Fully local mode for the video

Lightning, voice and the 2B grounder all on this Mac: 30 + 2 + 0.6 = 32.6B. One grounder
instance does both clicks and screen summaries. Prerequisites:
- Gate G has passed for the adapter you're showing.
- Teams is quit and Chrome is closed (Playwright's own browser is fine).
- The Mac is on power, with Low Power Mode off.
- Lightning is up on :8000 (§2).

```bash
# 1. Two tmux panes: the watchdog first, then the grounder. The relaxed probe margin keeps
#    the agent's own planner calls from tripping it; memory triggers are unchanged.
python3 -I scripts/gate_g_watch.py --pidfile ~/models/logs/local-grounder-8011.pid --port 8011 \
  --log ~/models/gate_g/watch-video.jsonl --probe-every 30 --probe-margin-s 1.5
GPU_UTIL=0.08 MAX_BATCHED=2560 MIN_FREE_PCT=26 scripts/run_local_grounder.sh   # healthy in ~45 s
# 2. The console with the real agent, pointed at the Mac grounder for clicks and summaries
#    (--grounder defaults to grounder):
scripts/demo_up.sh --agent real --headed --vlm-url http://127.0.0.1:8011/v1 \
  --summary-model grounder --vlm-label "Mac (vllm-metal)"
#    Same thing as env vars (a flag beats its env var):
#    AGENT_VLM_BASE_URL=http://127.0.0.1:8011/v1 AGENT_GROUNDER_MODEL=grounder AGENT_SUMMARY_MODEL=grounder \
#      scripts/demo_up.sh --agent real --headed
# 3. Record with Cmd+Shift+5 (macOS screen recording), not Zoom.
```

- **What to check:** `demo_up.sh` checks that :8011 serves `grounder` for both roles.
  The console disables the `base` option, and its header reads "VLM: Mac (vllm-metal) ·
  127.0.0.1:8011 · serves grounder", so the video shows where the grounder runs.
- **The H200 isn't needed.** Close the 8001 tunnel, so no H200 grounder is in the loop.
- **Expect per step:** ~4.3 s to plan (thinking off; ~13 s when a thinking retry kicks
  in), ~1 s to ground a click, ~6 s for each screen summary.
- **If the watchdog trips** (it writes `~/models/gate_g/STOPPED`), switch to hybrid.

**Switch back to hybrid** (grounder on the H200):

```bash
kill -TERM "$(cat ~/models/logs/local-grounder-8011.pid)"   # or Ctrl-C in its pane; the watchdog exits by itself
ssh -N -o ControlPath=none -o ExitOnForwardFailure=yes -L 8001:localhost:8001 permanent-tan-marmot   # §3a
scripts/demo_up.sh --agent real --headed      # no --vlm-url / AGENT_* overrides: configs/agent.yaml points at :8001
```
