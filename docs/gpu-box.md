# GPU box: serving and training (`permanent-tan-marmot`)

Copy-paste commands for the shared Brev H200. Every pin was verified on 2026-10-17
against PyPI, Docker Hub, the GitHub sources and the model cards. Nothing has been
run on the box yet, so treat the first run of each section as a smoke test, and log
GPU jobs in [runs.md](runs.md). Owner: workstream B (Kyle).

**Box:** 1× H200 (143 GB), driver 580.173.02 (CUDA 13.0), Ubuntu 24.04, 16 vCPU,
196 GB RAM, 2.4 TB disk, Docker with the NVIDIA runtime, no `nvcc`. `~/.cache` is a
symlink to `/ephemeral/cache`; keep weights under `/home/ubuntu`.

## Pinned stack (don't upgrade mid-weekend)

| Piece | Version | Why this one |
| --- | --- | --- |
| vLLM serving image | `vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3` | Latest stable; CUDA 13.0.2 base runs on driver 580. Avoid `latest`, `nightly` and `cu134-*`. |
| Training: torch / torchvision | `2.13.0+cu130` / `0.28.0` | Pinned by vllm 0.31.0 |
| transformers | `5.17.0` | vllm 0.31.0 requires `<5.18`. **Never `-U`.** |
| trl | `1.16.0` | Supports vllm ≤ 0.31.0 |
| peft / accelerate / datasets | `0.21.2` / `1.15.0` / `5.1.0` | — |
| Qwen3.5 kernels | `flash-linear-attention[cuda]==0.5.2`, `causal-conv1d==1.7.0` (prebuilt), `kernels==0.16.2` | Without them, Qwen3.5 falls back to slow torch code |
| NeMo Gym | `nemo-gym==0.7.0`, Python 3.13.14, in its **own** venv | It pins `openai==2.44.0` and needs Python 3.13 |

## 1. Prepare (once)

```bash
cat > /home/ubuntu/serving.env <<'EOF'
export VLLM_IMAGE=vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
export VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm
export HF_HOME=/home/ubuntu/hf
EOF
source /home/ubuntu/serving.env          # re-run in EVERY new shell / tmux pane
id -nG | grep -qw docker || echo "not in docker group: prefix docker commands with sudo"
mkdir -p /home/ubuntu/hf /home/ubuntu/vllm-cache /home/ubuntu/data /home/ubuntu/patches /home/ubuntu/ckpt
docker pull "$VLLM_IMAGE"                # ~9 GB

# Pre-download weights as ubuntu (not root), so the training venv can share the cache
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf -e HF_TOKEN \
  -v /home/ubuntu/hf:/hf --entrypoint python3 "$VLLM_IMAGE" -c \
  "from huggingface_hub import snapshot_download as d; [print(d(r)) for r in ['nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4','Qwen/Qwen3.5-2B','Qwen/Qwen3-VL-2B-Instruct']]"
# sizes: Lightning NVFP4 21.6 GB, Qwen3.5-2B 4.6 GB, Qwen3-VL-2B 4.3 GB
```

**Lightning on Hopper needs a patch.** vLLM [#61603](https://github.com/vllm-project/vllm/issues/61603)
makes Lightning return NaN or empty output on sm90 GPUs like the H200. The fix (PR
#61718, merged 2026-10-17) isn't in any release yet. Backport the fixed file:

```bash
curl -fsSL -o /home/ubuntu/patches/ssd_chunk_scan.py \
  https://raw.githubusercontent.com/vllm-project/vllm/783380af397ceeae91a40f4f759439eb8c1ab690/vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py
echo "b3851069c1e37c25dd9afaecc348f8f8868b3c50bb0d1fe5e0dde3dfa23b3a01  /home/ubuntu/patches/ssd_chunk_scan.py" | sha256sum -c -
```

## 2. Serve Lightning NVFP4 on :8000 (demo and final agent eval only)

During development, use the hosted endpoint instead (SETUP.md §4). Start Lightning
first, then the grounder.

```bash
source /home/ubuntu/serving.env
docker run -d --name vllm-lightning --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
  -v /home/ubuntu/hf:/root/.cache/huggingface -v /home/ubuntu/vllm-cache:/root/.cache/vllm \
  -v /home/ubuntu/patches/ssd_chunk_scan.py:${VLLM_PKG:?source /home/ubuntu/serving.env first}/model_executor/layers/mamba/ops/ssd_chunk_scan.py:ro \
  -e VLLM_HUMMING_MOE_GEMM_TYPE=indexed \
  "$VLLM_IMAGE" \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 lightning \
  --port 8000 --gpu-memory-utilization 0.45 \
  --max-model-len 65536 --max-num-seqs 64 --max-num-batched-tokens 16384 \
  --kv-cache-dtype fp8 --enable-prefix-caching --mamba-cache-mode align --async-scheduling \
  --moe-backend humming --linear-backend humming \
  --mamba-backend flashinfer --mamba-ssu-algorithm horizontal \
  --mamba-ssm-cache-dtype float16 --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3
docker exec vllm-lightning grep -c _drop_racy_wgmma_configs $VLLM_PKG/model_executor/layers/mamba/ops/ssd_chunk_scan.py   # expect >= 2 (patch active)
until curl -sf localhost:8000/health >/dev/null; do docker ps -q -f name=vllm-lightning | grep -q . || { docker logs --tail 80 vllm-lightning; break; }; sleep 5; done
```

- **No speculative decoding.** We don't use the DSpark/DFlash drafters, because they
  count toward the 33B cap.
- **If startup fails in the humming or SSU code:** drop
  `-e VLLM_HUMMING_MOE_GEMM_TYPE=indexed --moe-backend humming --linear-backend humming --mamba-ssu-algorithm horizontal`.
- **Thinking** is on by default. Turn it off per request with
  `"chat_template_kwargs":{"enable_thinking":false}`. Sampling: temperature 1.0,
  top_p 0.95.
- **If you get NaN or empty output**, `docker restart` won't fix it. Recreate the
  container instead: `docker rm -f vllm-lightning && sudo rm -rf /home/ubuntu/vllm-cache/*`,
  check `serving.env` and the patch, then re-run.

NaN check: 5 sequential prompts. Each should return HTTP 200 with non-empty text.

```bash
for i in 1 2 3 4 5; do python3 -c 'import json;print(json.dumps({"model":"lightning","max_tokens":48,"temperature":0,"logprobs":True,"chat_template_kwargs":{"enable_thinking":False},"messages":[{"role":"user","content":"Summarize in one line: "+"The quick brown fox jumps over the lazy dog. "*40}]}))' | curl -s -o /tmp/nan.json -w '%{http_code} ' localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d @-; python3 -c 'import json;d=json.load(open("/tmp/nan.json"));c=d.get("choices");print(repr((c[0]["message"].get("content") or "")[:60])) if c else print(d)'; done
```

## 3. Serve the grounder on :8001

**Qwen3.5-2B:**

```bash
source /home/ubuntu/serving.env
docker run -d --name vllm-vlm --gpus all --ipc=host -p 127.0.0.1:8001:8001 \
  -v /home/ubuntu/hf:/root/.cache/huggingface -v /home/ubuntu/vllm-cache:/root/.cache/vllm \
  -v /home/ubuntu/data:/home/ubuntu/data:ro \
  "$VLLM_IMAGE" \
  --model Qwen/Qwen3.5-2B --served-model-name grounder Qwen/Qwen3.5-2B \
  --port 8001 --gpu-memory-utilization 0.30 \
  --max-model-len 16384 --max-num-seqs 128 --max-num-batched-tokens 16384 \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --mm-processor-kwargs '{"size": {"longest_edge": 4194304, "shortest_edge": 65536}}' \
  --allowed-local-media-path /home/ubuntu/data \
  --default-chat-template-kwargs '{"enable_thinking": false}'
until curl -sf localhost:8001/health >/dev/null; do docker ps -q -f name=vllm-vlm | grep -q . || { docker logs --tail 80 vllm-vlm; break; }; sleep 5; done
```

- **Don't pass a `--reasoning-parser` for Qwen3.5.** It would return the answer as
  `reasoning` with `content: null` (vLLM #53284).
- **Pixel cap:** `size.longest_edge` / `shortest_edge` are max and min total pixels.
  Always pass both. One token covers 32×32 px, so this cap is ≤ 4096 image tokens.
  **Use the same cap in TRL** (decision #8: choose ~1 MP vs ~2 MP vs this 4 MP
  during baselines).
- **If requests hang under load** (#37729): add `--gdn-prefill-backend triton`; if it
  still hangs, add `--no-enable-prefix-caching`.
- **Tool calling** (if needed): `--enable-auto-tool-choice --tool-call-parser qwen3_coder`.
- **`file://` images** must sit under `/home/ubuntu/data`, which is mounted at the
  same path inside the container.

**Qwen3-VL-2B instead:**
- Use the same command with
  `--model Qwen/Qwen3-VL-2B-Instruct --served-model-name grounder Qwen/Qwen3-VL-2B-Instruct`,
  and remove `--default-chat-template-kwargs`.
- Tools: `--tool-call-parser hermes`.
- Its default sampling is temperature 0.7, so send `temperature: 0` for evals.
- It outputs relative 0–1000 coordinates.

**Smoke test:**

```bash
curl -s localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"grounder","temperature":0,"max_tokens":64,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"}},{"type":"text","text":"Describe this image in one sentence."}]}]}' | python3 -c 'import json,sys;r=json.load(sys.stdin);print(r["choices"][0]["message"]["content"]);print(r["usage"])'
```

## 4. GPU memory plans (fractions of ~140 GiB; start one container at a time)

| Situation | Plan |
| --- | --- |
| Eval / agent dev | Lightning 0.45 + grounder 0.30. Raise the grounder to 0.45 for `--num-repeats 8` evals. |
| SFT | Stop both servers (`docker stop vllm-vlm vllm-lightning`). |
| GRPO (TRL colocate) | Stop the grounder server. Use hosted Lightning or stop it too. Set `vllm_gpu_memory_utilization` explicitly (the TRL default is 0.3). |
| Demo | Lightning 0.45 + grounder 0.30. All training stopped. |

## 5. Training venv (`/home/ubuntu/venvs/train`)

```bash
command -v gcc >/dev/null && command -v g++ >/dev/null || (sudo apt-get update && sudo apt-get install -y build-essential)   # Triton JIT needs a compiler
curl -LsSf https://astral.sh/uv/0.13.1/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
grep -q UV_LINK_MODE ~/.bashrc || echo 'export UV_LINK_MODE=copy' >> ~/.bashrc; export UV_LINK_MODE=copy   # ~/.cache is on /ephemeral
uv venv /home/ubuntu/venvs/train --python 3.12 --managed-python    # MUST be 3.12 (uv defaults to 3.15; vLLM needs <3.15)
source /home/ubuntu/venvs/train/bin/activate

# Step A: PyPI + PyTorch cu130 index
uv pip install --torch-backend cu130 --only-binary :all: \
  "torch==2.13.0" "torchvision==0.28.0" "vllm==0.31.0" "transformers==5.17.0" \
  "trl[vllm,peft]==1.16.0" "peft==0.21.2" "accelerate==1.15.0" "datasets==5.1.0" \
  "kernels==0.16.2" "flash-linear-attention[cuda]==0.5.2" \
  "qwen-vl-utils==0.0.14" "pillow==12.3.0" "imagehash==4.3.2" "wandb==0.30.0"
# Step B: prebuilt CUDA extensions (no nvcc on the box). Keep as a separate command.
uv pip install --no-deps --index-url https://wheels.astral.sh/simple/cu130/ \
  "causal-conv1d==1.7.0+cu.13.0.torch.2.13" "flash-attn==2.8.3.post1+cu.13.0.torch.2.13"
```

**Sanity checks:**

```bash
uv pip check && python -c "import torch,transformers,trl,vllm,fla,causal_conv1d; from transformers import AutoProcessor; AutoProcessor.from_pretrained('Qwen/Qwen3.5-2B'); assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__, trl.__version__, vllm.__version__)"
# expect: 2.13.0+cu130 5.17.0 1.16.0 0.31.0
# Qwen3.5 fast-path check (the slow-path warning only fires during a forward pass):
python - <<'EOF' 2>&1 | tee /tmp/qwen35_fastpath.log
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
m = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-2B", dtype=torch.bfloat16, device_map="cuda")
p = AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B")
ids = p.tokenizer("hello world " * 64, return_tensors="pt").to("cuda")
with torch.no_grad(): print("logits", m(**ids).logits.shape)
EOF
grep -q "falling back to its reference PyTorch implementation" /tmp/qwen35_fastpath.log && echo "SLOW PATH - fix fla/causal_conv1d" || echo "fast path OK"
```

**TRL notes (v1.16.0, one GPU):**
- **Use `GRPOTrainer` with `vllm_mode="colocate"`.**
  - Server mode doesn't work on one GPU: TRL's client only does NCCL weight sync.
  - `trl.experimental` `AsyncGRPOTrainer` refuses Qwen3.5.
- **Set these explicitly:** `vllm_max_model_length` (≥ image tokens + prompt +
  completion), `vllm_gpu_memory_utilization`, and
  `chat_template_kwargs={"enable_thinking": False}` for Qwen3.5.
- **Pre-resize screenshots** to the same pixel cap as serving. TRL colocate passes no
  `mm_processor_kwargs`.
- **Load the model** with `AutoModelForImageTextToText` (not `AutoModelForCausalLM`).
- **VLM SFT:**
  - Use a prompt-completion dataset with `SFTConfig(max_length=None)`; completion-only
    loss is the default.
  - `packing` and `padding_free` error out on vision data.
  - `assistant_only_loss` isn't available on this stack.
- **TRL has no CI for VLM GRPO with vLLM colocate.** Run a 20-step smoke test before
  the real run.

## 6. NeMo Gym on the box (`/home/ubuntu/<you>/nvidia-hack-26`)

Same as SETUP.md §6, with two differences:
- **Paths:** set `export HF_HOME=/home/ubuntu/hf` first, and point `env.yaml` at
  `http://127.0.0.1:8001/v1` (served name `grounder`, `policy_api_key: EMPTY`).
- **Run Gym on the box** for anything bigger than a smoke test with real screenshots.
  That keeps multi-MB images off the tunnel.

## 7. From a laptop

```bash
brev login                                                 # browser flow; re-run if you see "malformed refresh token"
ssh -N -o ControlPath=none -o ExitOnForwardFailure=yes -L 8000:localhost:8000 -L 8001:localhost:8001 permanent-tan-marmot
# or one port at a time: brev port-forward permanent-tan-marmot -p 8001:8001
curl -s localhost:8001/v1/models
```

## 8. Teardown

```bash
docker rm -f vllm-vlm vllm-lightning
brev stop permanent-tan-marmot        # from a laptop, when nobody needs the GPU
```
