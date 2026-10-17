# Setup

Gets a teammate from a fresh Mac to working on the shared Brev H200. Do the steps in
order; each one ends with a check. Commands assume macOS with zsh, the default shell.
**You don't need Homebrew.**

## 0. Command-line tools

A fresh Mac has stub `git` and `python3` binaries that only prompt you to install
Apple's Command Line Tools. Install them first. It takes a few minutes, and a managed
Mac may need admin rights or IT.

```bash
xcode-select -p >/dev/null 2>&1 || xcode-select --install    # wait for the installer to finish
git --version && python3 --version
```

## 1. Get access (ask the team captain)

1. **GitHub:** ask to be added as a collaborator on `pballou/nvidia-hack-26`.
2. **Brev org:** ask for an invite link. The captain gets one from the Brev console
   Team tab → "Generate Invite Link", or by running `brev invite`. Open the link while
   signed in to <https://brev.nvidia.com>.
3. **SSH to the instance:** the instance owner adds you on the instance page →
   **Share SSH Access**. The instance must be running.

## 2. Git and GitHub

Set an identity first, so commits to this public repo don't show your laptop's
hostname:

```bash
git config --global user.name "Your Name"
git config --global user.email "<id>+<username>@users.noreply.github.com"   # GitHub → Settings → Emails
```

Clone to a fixed path; the rest of this guide assumes `~/nvidia-hack-26`:

```bash
git clone git@github.com:pballou/nvidia-hack-26.git ~/nvidia-hack-26
```

If you get `Permission denied (publickey)`, check what your SSH client offers:

```bash
ssh -vT git@github.com 2>&1 | grep -iE 'skipping|offering|authenticated'
```

Create an **ECDSA** key in either of these cases:
- Nothing is `Offering` a key (you don't have one yet).
- You see `Skipping ssh-ed25519 key ... not in PubkeyAcceptedAlgorithms` (your Mac's
  managed SSH policy only allows ECDSA).

ECDSA works in both cases. Add the `.pub` file to GitHub (Settings → SSH keys →
Authentication key), then retry:

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

If `~/.ssh/config` already has a `Host github.com` block, edit that one instead; SSH
uses the first match. You can also clone over HTTPS
(`git clone https://github.com/pballou/nvidia-hack-26.git ~/nvidia-hack-26`), but
pushing over HTTPS needs a personal access token.

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
( V=0.6.337; ARCH=$(uname -m | sed 's/x86_64/amd64/')   # arm64 on Apple Silicon
  cd "$(mktemp -d)" &&
  curl -fLO "https://github.com/brevdev/brev-cli/releases/download/v$V/brev-cli_${V}_darwin_${ARCH}.tar.gz" &&
  curl -fLO "https://github.com/brevdev/brev-cli/releases/download/v$V/brev-cli_${V}_checksums.txt" &&
  grep "darwin_${ARCH}.tar.gz" brev-cli_${V}_checksums.txt | shasum -a 256 -c - &&   # expect: OK
  tar -xzf brev-cli_${V}_darwin_${ARCH}.tar.gz brev &&
  mkdir -p ~/.local/bin && mv brev ~/.local/bin/ && chmod 755 ~/.local/bin/brev )
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc && brev --version
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

**SSH to the box.** The box is usually **stopped** to save credits. Ask Kyle
(stream B), or start it yourself with `brev start permanent-tan-marmot` if the team
agrees.

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
`Skipping ssh-ed25519 ...`, you can allow ed25519 for Brev's gateway **only**.

> **This overrides your Mac's managed SSH policy for one host. Only do it if your IT
> policy allows it.** Otherwise use a personal machine, or the instance's browser
> terminal / Jupyter shared through Brev's "Using secure links" setting.

```bash
cat >> ~/.ssh/config <<'EOF'

# Brev: its SSH certs are ed25519; allow that key type for the Brev gateway only
Match host global.prd.ga.run.brev.nvidia.com
  PubkeyAcceptedAlgorithms +ssh-ed25519,ssh-ed25519-cert-v01@openssh.com
EOF
ssh -G permanent-tan-marmot | grep -iE '^(hostname|pubkeyacceptedalgorithms) '
# expect: hostname global.prd.ga.run.brev.nvidia.com, and a list that includes ssh-ed25519-cert-v01@openssh.com
```

If `hostname` isn't the gateway, use
`Match originalhost permanent-tan-marmot,permanent-tan-marmot-host` instead.

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

```bash
cd ~/nvidia-hack-26
cp .env.example .env              # then fill in values; it's gitignored
set -a; source .env; set +a       # exports the values, so child processes (Python, hf, docker -e) see them
```

- **`NVIDIA_API_KEY`:** sign in at <https://build.nvidia.com>, open the Nemotron 3.5
  Lightning model page, and click "Get API Key". This is used for hosted Lightning
  (`nvidia/nemotron-3.5-lightning-30b-a3b` at `https://integrate.api.nvidia.com/v1`).
- **`HF_TOKEN`:** <https://huggingface.co/settings/tokens>.
  - A **read** token is enough for everyone.
  - Patrick's write token stays on his laptop.
  - Uploads from the Brev box use a separate fine-grained token scoped to write only `patrickballou/gui-assist-grounder` and `patrickballou/gui-assist-data`. Revoke it after the event.

Smoke test for hosted Lightning:

```bash
curl -s https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"nvidia/nemotron-3.5-lightning-30b-a3b","messages":[{"role":"user","content":"Say OK"}],"max_tokens":256,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"] if "choices" in d else d)'
```

## 5. Shared paths on the Brev box

The box has one shared `ubuntu` user, so:
- **Keep your own clone.** That way nobody overwrites anyone else's working tree.
- **Push from your laptop.** Don't store GitHub credentials there.
- **HF tokens on the box:** export a read-only token in your shell.
  - Uploads (Patrick/Kyle) use the fine-grained token scoped to write only `patrickballou/gui-assist-grounder` and `patrickballou/gui-assist-data`, exported only in the upload shell.
  - Never run `hf auth login` there: it saves the token into the shared `HF_HOME`.

| Path | Contents |
| --- | --- |
| `/home/ubuntu/<your-name>/nvidia-hack-26` | Your clone (`git clone https://github.com/pballou/nvidia-hack-26.git`; pull only) |
| `/home/ubuntu/data` | Shared datasets: OS-Atlas slice, ScreenSpot-v2, preprocessed images (`file://` URLs point here) |
| `/home/ubuntu/hf` | Shared Hugging Face cache (`HF_HOME`, set by `/home/ubuntu/serving.env`). Don't use `~/.cache`, which is a symlink to `/ephemeral`. |
| `/home/ubuntu/ckpt` | Checkpoints and LoRA adapters |
| `/home/ubuntu/venvs/train` | Shared training venv ([docs/gpu-box.md](docs/gpu-box.md) §5). Gym venvs live in each clone (`.venv-gym`). |

On the box, start every shell with `source /home/ubuntu/serving.env`. Run long jobs
in tmux (`tmux new -s <your-name>`), and log GPU jobs in [docs/runs.md](docs/runs.md).

## 6. Python environments

[uv](https://docs.astral.sh/uv/) manages Python, on both the laptop and the box. Gym
also shells out to `uv`, so it must be on your PATH. `~/.local/bin` is already on
your PATH from §3.

```bash
curl -LsSf https://astral.sh/uv/0.13.1/install.sh | sh; hash -r      # hash -r works in zsh and bash
uv --version                                   # 0.13.1
```

**NeMo Gym** (`nemo-gym==0.7.0` needs Python ≥ 3.13.14) goes in its own venv in the
repo root. **Always run `gym` from the repo root:** its config lookup, server venvs,
`./cache` and `./results` all depend on the current directory. Gym is CPU-only, so it
runs on a laptop against remote models.

```bash
cd ~/nvidia-hack-26
uv venv --python 3.13.14 .venv-gym && source .venv-gym/bin/activate   # always pin the patch version
uv pip install "nemo-gym==0.7.0"
export NEMO_GYM_EXTRA_ROOTS="$(git rev-parse --show-toplevel)"        # finds our resources_servers/ from anywhere
gym --version                                                         # NeMo Gym v0.7.0 / Python 3.13.14
cp env.yaml.example env.yaml                                          # gitignored; set the model endpoint
```

**Gym hello world.** This proves the install, the tunnel and the model wiring. It
needs the base VLM served on the box ([docs/gpu-box.md](docs/gpu-box.md) §3).
**Stream B (Kyle) runs that server; don't start your own.**

```bash
ssh permanent-tan-marmot 'curl -sf localhost:8001/v1/models' || echo "not up yet: ask Kyle"
ssh -N -o ControlPath=none -L 8001:localhost:8001 permanent-tan-marmot &     # tunnel (or brev port-forward)
# env.yaml: policy_base_url: http://127.0.0.1:8001/v1, policy_api_key: EMPTY, policy_model_name: base
gym env start --resources-server gui_coordinate --model-type vllm_model ++server_spinup_timeout_seconds=1800
#   first start builds ~4 server venvs; wait for "servers ready". Later starts: add ++skip_venv_if_present=true
# second terminal (cd ~/nvidia-hack-26, venv active):
gym eval run --no-serve --agent gui_coordinate_simple_agent \
  --input resources_servers/gui_coordinate/data/example.jsonl --output results/hello_gui.jsonl --limit 5 --concurrency 5
cat results/hello_gui_aggregate_metrics.json
```

`resources_servers/gui_coordinate` is Gym's built-in env; Gym resolves it from the
installed package. If `--input` isn't found, use the installed copy:
`--input "$(find .venv-gym -path '*resources_servers/gui_coordinate/data/example.jsonl' | head -1)"`.

If the GPU box is stopped, smoke-test Gym against hosted Lightning with a text env:

```bash
gym env start --resources-server mcqa --model-type vllm_model \
  --model-url https://integrate.api.nvidia.com/v1 --model-api-key "$NVIDIA_API_KEY" --model nvidia/nemotron-3.5-lightning-30b-a3b
gym eval run --no-serve --agent mcqa_simple_agent --input resources_servers/mcqa/data/example.jsonl \
  --output results/hello_mcqa.jsonl --limit 5
```

**Other tools:**

| Tool | Install | Notes |
| --- | --- | --- |
| Hugging Face CLI | `uv tool install 'hf==2.2.0'` | Laptop: `hf auth login`. Box: export `HF_TOKEN` in the shell (read-only, or the scoped upload token) instead. `export HF_XET_HIGH_PERFORMANCE=1` for fast downloads. |
| Playwright (stream C) | `uv pip install 'playwright==1.62.0'`, then `playwright install chromium` (`--with-deps` on Ubuntu) | Gym's `interactive_browser` pins `>=1.62,<1.63`; matching the browser build matters. On the box, `PLAYWRIGHT_BROWSERS_PATH` (in `serving.env`) keeps browsers off `/ephemeral`. |

**Training and serving stack on the GPU box:** [docs/gpu-box.md](docs/gpu-box.md)
(pinned vLLM image, Lightning and grounder serving commands, training venv, GPU
memory plans).

## 7. Coding agents

The repo ships agent context. Claude Code and Codex pick it up automatically:

| File | What's in it |
| --- | --- |
| `AGENTS.md` (imported by `CLAUDE.md`) | Rules, constraints, conventions |
| `docs/plan.md` | Plan, decisions, workstreams, contracts |
| `docs/research-notes.md` | Verified facts. Read this before researching. |
| `.agents/skills/` (also `.claude/skills/`) | NeMo Gym skills, including **BLADE** failure analysis (`nemo-gym-blade-analysis`). Point it at the output of a comparable pair of `gym eval run`s to get a failure taxonomy and next steps. |

## 8. Done-checks (milestone M0)

On your own:
- [ ] `git --version`, `python3 --version` and `uv --version` work.
- [ ] `ssh -T git@github.com` authenticates, and you can clone and push the repo.
- [ ] `brev ls` shows `permanent-tan-marmot`.
- [ ] The hosted Lightning smoke test prints a message.
- [ ] `gym --version` works in `.venv-gym`, and the hosted `mcqa` smoke test runs.

When Kyle has the box up and the base VLM served (do these together):
- [ ] `ssh permanent-tan-marmot 'nvidia-smi -L'` works.
- [ ] The Gym hello world (`gui_coordinate`) runs through the tunnel.
