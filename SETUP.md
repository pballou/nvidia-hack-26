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
1. Brev console → your avatar → Settings → **CLI**. Copy the
   `brev login --token ...` line and run it in your own terminal. Don't paste the
   token into chat or a coding agent.
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
| `/home/ubuntu/venvs/{gym,train}` | Shared Python environments (see §6) |

Run long jobs in tmux (`tmux new -s <your-name>`), and log GPU jobs in
[docs/runs.md](docs/runs.md).

## 6. Python environments

[uv](https://docs.astral.sh/uv/) manages Python, on both the laptop and the box:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # then open a new shell
```

**NeMo Gym environment** (Python ≥ 3.13.14, required by `nemo-gym` 0.7.0). Gym is
CPU-only, so it also runs on your laptop against remote models.

```bash
uv venv --python 3.13.14 ~/venvs/gym && source ~/venvs/gym/bin/activate
uv pip install "nemo-gym==0.7.0"
gym --help
```

**Training and serving stack on the box.** This section is being finalized: exact
torch/transformers/TRL/vLLM pins and the vLLM serving commands for the grounder and
Lightning will land here. Until then, coordinate with workstream B (docs/plan.md)
before installing anything into the shared `/home/ubuntu/venvs/train`.

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
- [ ] `gym --help` works in `~/venvs/gym`.
