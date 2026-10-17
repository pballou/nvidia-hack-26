"""Download pinned model weights and raw datasets onto the Brev box.

Run on the box, from a shell that has sourced serving.env (docs/gpu-box.md §1):

    uv run --no-project --python 3.12 --with huggingface_hub==2.2.0 download.py models
    uv run --no-project --python 3.12 --with huggingface_hub==2.2.0 download.py data
    uv run --no-project --python 3.12 --with huggingface_hub==2.2.0 download.py data-rest

- Models go to the shared HF cache ($HF_HOME), so vLLM and TRL find them by repo id.
- Raw datasets go to /home/ubuntu/data/raw/<name>; data_prep/ reads from there.
- Every repo is pinned to a commit, so reruns fetch the same bytes. All six repos
  are ungated, so no HF token is needed (a read-only one only raises rate limits).
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

DATA_ROOT = Path("/home/ubuntu/data/raw")

# Grounder candidates first (the M1 baselines need them). Lightning NVFP4 is only for
# self-hosting on Sunday, so it comes last.
MODELS = [
    ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"),
    ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "bee7596271d1495f6992ae224aefde4410e816b8"),
]

# OS-Atlas: only the subsets in the docs/plan.md data mix (~12.6 GB). Never linux_* or
# macos_* (contaminated, research-notes §5). Windows and FineWeb images are
# range-extracted by data_prep/, so only their annotations are fetched here.
OS_ATLAS_FILES = [
    "README.md",
    "prompts.json",
    "mobile_domain/aw_mobile.json",  # AndroidWorld
    "mobile_domain/mobile_images.zip",
    "mobile_domain/uibert_raw.json",
    "mobile_domain/UIBert.zip",
    "mobile_domain/widget_captioning.json",
    "mobile_domain/rico_imgs.zip",
    "desktop_domain/windows_splited.json",
    "web_domain/fineweb_3m.json",
    "web_domain/seeclick_web.json",  # fallback if FineWeb fails its SSv2 overlap check
]
assert not any("linux" in f or "macos" in f for f in OS_ATLAS_FILES)
OS_ATLAS_EXCLUDE = ["desktop_domain/linux_*", "desktop_domain/macos_*"]

# `data-rest`: the rest of OS-Atlas (~795 GB), for downselecting on the box. Fetched in
# this order so the plan's sources land first: Windows, the first FineWeb shard,
# SeeClick web (FineWeb's fallback), AMEX, IG_data, then the other FineWeb shards.
OS_ATLAS_REST = [
    ["desktop_domain/windows_image_*"],
    ["web_domain/fineweb_3m_s11.zip"],
    ["web_domain/seeclick_web_image_*"],
    ["mobile_domain/amex_*", "mobile_domain/ricosca.json"],
    ["IG_data/*"],
    ["web_domain/fineweb_3m_s*.zip"],
]

# (local name, repo, commit, allow_patterns)
DATASETS = [
    # Held-out test only; never train on it. Revision pinned per AGENTS.md.
    ("screenspot_v2", "OS-Copilot/ScreenSpot-v2", "5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c", None),
    ("os_atlas", "OS-Copilot/OS-Atlas-data", "e3a4c90c5f6129c25efdaa0671b08a11f7cb8f3f", OS_ATLAS_FILES),
    ("groundcua", "ServiceNow/GroundCUA", "5d6845b0116029d46ec762e734701c5b8ce207c3", None),  # ~22.5 GB
]


def download_models():
    if os.environ.get("HF_HOME") != "/home/ubuntu/hf":
        sys.exit("HF_HOME is not /home/ubuntu/hf: source /home/ubuntu/serving.env first")
    for repo, commit in MODELS:
        print(f"== {repo}@{commit[:10]}", flush=True)
        print(snapshot_download(repo, revision=commit, max_workers=16), flush=True)


def download_data():
    for name, repo, commit, patterns in DATASETS:
        local_dir = DATA_ROOT / name
        print(f"== {repo}@{commit[:10]} -> {local_dir}", flush=True)
        snapshot_download(
            repo, repo_type="dataset", revision=commit, allow_patterns=patterns,
            ignore_patterns=OS_ATLAS_EXCLUDE, local_dir=local_dir, max_workers=16,
        )
        # M1 baselines need the SSv2 screenshots on disk for file:// URLs right away,
        # so unzip before moving on to the bigger datasets.
        marker = local_dir / ".extracted"
        if name == "screenspot_v2" and not marker.exists():
            with zipfile.ZipFile(local_dir / "screenspotv2_image.zip") as zf:
                zf.extractall(local_dir)
            marker.touch()
    print("done", flush=True)


def download_data_rest():
    _, repo, commit, _ = next(d for d in DATASETS if d[0] == "os_atlas")
    for patterns in OS_ATLAS_REST:
        print(f"== {repo}@{commit[:10]} {patterns}", flush=True)
        snapshot_download(
            repo, repo_type="dataset", revision=commit, allow_patterns=patterns,
            ignore_patterns=OS_ATLAS_EXCLUDE, local_dir=DATA_ROOT / "os_atlas", max_workers=16,
        )
    print("done", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["models", "data", "data-rest"])
    commands = {"models": download_models, "data": download_data, "data-rest": download_data_rest}
    commands[parser.parse_args().what]()
