# Setup

Gets a teammate from a fresh Mac to working on the shared Brev H200. Do the steps in
order; each one ends with a check. Commands assume macOS with zsh, the default shell.
**You don't need Homebrew.**

## 1. Get access (ask the team captain)

1. **GitHub:** ask to be added as a collaborator on `pballou/nvidia-hack-26`.
2. **Brev org:** ask for an invite link. The captain gets one from the Brev console
   Team tab → "Generate Invite Link", or by running `brev invite`. Open the link while
   signed in to <https://brev.nvidia.com>.
3. **SSH to the instance:** the instance owner adds you on the instance page →
   **Share SSH Access**. The instance must be running.

## 2. GitHub over SSH

```bash
git clone git@github.com:pballou/nvidia-hack-26.git
```

If you get `Permission denied (publickey)`, check what your key offers:

```bash
ssh -vT git@github.com 2>&1 | grep -iE 'skipping|offering|authenticated'
```

`Skipping ssh-ed25519 key ... not in PubkeyAcceptedAlgorithms` means your Mac has a
managed SSH policy that only allows ECDSA keys. Create an ECDSA key, add the `.pub`
file to GitHub (Settings → SSH keys → Authentication key), and retry:

```bash
ssh-keygen -t ecdsa -b 256 -f ~/.ssh/id_ecdsa_github -C "github"
cat >> ~/.ssh/config <<'EOF'

Host github.com
  IdentityFile ~/.ssh/id_ecdsa_github
  IdentitiesOnly yes
EOF
cat ~/.ssh/id_ecdsa_github.pub      # paste into GitHub
ssh -T git@github.com               # expect: "Hi <you>! You've successfully authenticated"
```

You can also clone over HTTPS: `git clone https://github.com/pballou/nvidia-hack-26.git`.
Pushing over HTTPS needs a personal access token.

## 3. Brev CLI (no Homebrew needed)

Use the official installer. Without brew, it downloads the release binary into
`~/.local/bin`. No sudo is needed.

```bash
curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
brev --version                      # expect v0.6.337 or newer
```

<details><summary>Manual install, if the script is blocked</summary>

```bash
V=0.6.337; ARCH=$(uname -m | sed 's/x86_64/amd64/')   # arm64 on Apple Silicon
cd "$(mktemp -d)"
curl -fLO "https://github.com/brevdev/brev-cli/releases/download/v$V/brev-cli_${V}_darwin_${ARCH}.tar.gz"
curl -fLO "https://github.com/brevdev/brev-cli/releases/download/v$V/brev-cli_${V}_checksums.txt"
grep "darwin_${ARCH}.tar.gz" brev-cli_${V}_checksums.txt | shasum -a 256 -c -   # expect: OK
tar -xzf brev-cli_${V}_darwin_${ARCH}.tar.gz brev
mkdir -p ~/.local/bin && mv brev ~/.local/bin/ && chmod 755 ~/.local/bin/brev
```
</details>

**Log in:**

1. Run **`brev login`**. This is the browser flow, and its session refreshes itself.
   - **Don't use the `brev login --token …` line from the console.** That session
     expires after about 15 minutes and can't refresh.
   - Symptoms of an expired login: `WARN: malformed refresh token`, or
     `Could not resolve hostname permanent-tan-marmot` when you SSH. Fix it by
     running `brev login` again.
   - `brev login --skip-browser` prints the login URL instead of opening a browser.
2. Run `brev ls`. You should see `permanent-tan-marmot` (H200). If you're in several
   orgs, switch with `brev set <org-name>`.

**SSH to the box:**

```bash
brev refresh                                   # writes ~/.brev/ssh_config, included from ~/.ssh/config
ssh permanent-tan-marmot 'nvidia-smi -L'       # expect: GPU 0: NVIDIA H200
```

If that fails with `Permission denied (publickey)`, check whether your Mac's SSH
policy is skipping Brev's key:

```bash
ssh -v permanent-tan-marmot true 2>&1 | grep -i skipping
```

Brev issues **ed25519** certificates, which managed Macs reject. If you see
`Skipping ssh-ed25519 ...`, allow ed25519 for Brev's gateway **only** (not globally)
and retry:

```bash
cat >> ~/.ssh/config <<'EOF'

# Brev: its SSH certs are ed25519; allow that key type for the Brev gateway only
Match host global.prd.ga.run.brev.nvidia.com
  PubkeyAcceptedAlgorithms +ssh-ed25519,ssh-ed25519-cert-v01@openssh.com
EOF
```

**Useful commands:**

| Command | What it does |
| --- | --- |
| `brev shell permanent-tan-marmot` | Interactive shell (`ssh permanent-tan-marmot` also works) |
| `brev open permanent-tan-marmot code` | VS Code over SSH (`cursor` also works) |
| `brev exec permanent-tan-marmot "nvidia-smi"` | Run one command |
| `brev port-forward permanent-tan-marmot -p 8001:8001` | Forward a port (local:remote), e.g. the grounder's vLLM server |
| `brev copy ./file permanent-tan-marmot:/home/ubuntu/` | Copy files |
| `brev start permanent-tan-marmot` / `brev stop permanent-tan-marmot` | Start / stop. **Stop when idle**: stopped instances only pay for storage |

## 4. API keys (keep them out of git)

Create `.env` in the repo root (gitignored) from the template:

```bash
cp .env.example .env    # then fill in values
```

- **`NVIDIA_API_KEY`:** sign in at <https://build.nvidia.com>, open the Nemotron 3.5
  Lightning model page, and click "Get API Key". This is used for hosted Lightning
  (`nvidia/nemotron-3.5-lightning-30b-a3b` at `https://integrate.api.nvidia.com/v1`).
- **`HF_TOKEN`:** <https://huggingface.co/settings/tokens>. A read token is enough to
  download. Whoever publishes models or datasets needs a write token.

Smoke test for hosted Lightning:

```bash
source .env && curl -s https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"nvidia/nemotron-3.5-lightning-30b-a3b","messages":[{"role":"user","content":"Say OK"}],"max_tokens":256,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"])'
```

## 5. Shared paths on the Brev box

The box has one shared `ubuntu` user. Keep your own git clone so you don't overwrite
each other's working tree:

| Path | Contents |
| --- | --- |
| `/home/ubuntu/<your-name>/nvidia-hack-26` | Your clone (`git clone https://github.com/pballou/nvidia-hack-26.git`). Push from your laptop or set up your own credentials. |
| `/home/ubuntu/data` | Shared datasets: OS-Atlas slice, ScreenSpot-v2 |
| `/home/ubuntu/hf` | Shared Hugging Face cache. Set `export HF_HOME=/home/ubuntu/hf`. Don't use `~/.cache`, which is a symlink to `/ephemeral`. |
| `/home/ubuntu/ckpt` | Checkpoints |
| `/home/ubuntu/venvs/train` | Shared training venv ([docs/gpu-box.md](docs/gpu-box.md) §5). Gym venvs live in each clone (`.venv-gym`). |

Run long jobs in tmux (`tmux new -s <your-name>`), and log GPU jobs in
[docs/runs.md](docs/runs.md).

## 6. Python environments

[uv](https://docs.astral.sh/uv/) manages Python, on both the laptop and the box. Gym
also shells out to `uv`, so it must be on your PATH.

```bash
curl -LsSf https://astral.sh/uv/0.13.1/install.sh | sh && source $HOME/.local/bin/env
uv --version                                   # 0.13.1
```

**NeMo Gym** (`nemo-gym==0.7.0` needs Python ≥ 3.13.14) goes in its own venv in the
repo root. **Always run `gym` from the repo root:** its config lookup, server venvs,
`./cache` and `./results` all depend on the current directory. Gym is CPU-only, so it
runs on a laptop against remote models.

```bash
cd nvidia-hack-26
uv venv --python 3.13.14 .venv-gym && source .venv-gym/bin/activate   # always pin the patch version
uv pip install "nemo-gym==0.7.0"
export NEMO_GYM_EXTRA_ROOTS="$(git rev-parse --show-toplevel)"        # finds our resources_servers/ from anywhere
gym --version                                                         # NeMo Gym v0.7.0 / Python 3.13.14
cp env.yaml.example env.yaml                                          # gitignored; set the model endpoint
```

**Gym hello world.** This proves the install, the tunnel and the model wiring. With
the grounder served on the box ([docs/gpu-box.md](docs/gpu-box.md) §3) and a tunnel
open for port 8001 (§7 there):

```bash
# env.yaml: policy_base_url: http://127.0.0.1:8001/v1, policy_api_key: EMPTY, policy_model_name: grounder
gym env start --resources-server gui_coordinate --model-type vllm_model ++server_spinup_timeout_seconds=1800
#   first start builds ~4 server venvs; wait for "servers ready". Later starts: add ++skip_venv_if_present=true
# second terminal (same dir, venv active):
gym eval run --no-serve --agent gui_coordinate_simple_agent \
  --input resources_servers/gui_coordinate/data/example.jsonl --output results/hello_gui.jsonl --limit 5 --concurrency 5
cat results/hello_gui_aggregate_metrics.json
```

If the GPU box is stopped, smoke-test Gym against hosted Lightning with a text env:
`gym env start --resources-server mcqa --model-type vllm_model --model-url https://integrate.api.nvidia.com/v1 --model-api-key "$NVIDIA_API_KEY" --model nvidia/nemotron-3.5-lightning-30b-a3b`,
then `gym eval run --no-serve --agent mcqa_simple_agent --input resources_servers/mcqa/data/example.jsonl --output results/hello_mcqa.jsonl --limit 5`.

**Other tools:**

| Tool | Install | Notes |
| --- | --- | --- |
| Hugging Face CLI | `uv tool install 'hf==2.2.0'`, then `hf auth login` | `export HF_XET_HIGH_PERFORMANCE=1` for fast downloads |
| Playwright (stream C) | `uv pip install 'playwright==1.62.0'`, then `playwright install chromium` (`--with-deps` on Ubuntu) | Gym's `interactive_browser` pins `>=1.62,<1.63`; matching the browser build matters |

**Training and serving stack on the GPU box:** [docs/gpu-box.md](docs/gpu-box.md)
(pinned vLLM image, Lightning and grounder serving commands, training venv, GPU
memory plans).

## 7. Coding agents

The repo ships agent context. Claude Code and Codex pick it up automatically:

| File | What's in it |
| --- | --- |
| `AGENTS.md` (imported by `CLAUDE.md`) | Rules, constraints, conventions |
| `docs/plan.md` | Plan, workstreams, contracts |
| `docs/research-notes.md` | Verified facts. Read this before researching. |
| `.agents/skills/` (also `.claude/skills/`) | NeMo Gym skills, including **BLADE** failure analysis (`nemo-gym-blade-analysis`). Point it at `gym eval run` output to get a failure taxonomy and next steps. |

## 8. Done-checks (milestone M0)

- [ ] `ssh -T git@github.com` authenticates, and you can clone and push the repo.
- [ ] `brev ls` shows `permanent-tan-marmot`, and `ssh permanent-tan-marmot 'nvidia-smi -L'` works.
- [ ] The hosted Lightning smoke test prints a message.
- [ ] `gym --version` works in `.venv-gym` (and, once the grounder is served, the Gym hello world runs).
