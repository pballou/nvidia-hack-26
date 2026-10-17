"""Process, serving and NeMo Gym helpers for the night loop and one-off adapter evals.

    # evaluate one adapter on dev (and optionally SSv2-clean, results table only)
    python -m training.gymctl eval --adapter /home/ubuntu/ckpt/sft-template/adapter --name sft-template [--ssv2]

- vLLM servers are started with scripts/serve_vlm.sh (one process: `base` + LoRA `grounder`).
  The loop's eval server is :8004 (container vllm-vlm-8004); :8001 is the team contract and
  only changes at the very end of the loop (docs/tasks/B-night-loop.md).
- One Gym env (gui_grounding + simple_agent + vllm_model policy) runs on its own head port
  (:11010) with the resources server pinned at :12010, so GRPO can POST to /verify and other
  envs on :11000/:11001 are untouched. It points at the eval server as `grounder`.
- Gym 0.7.0 facts this relies on (docs/research-notes.md §3): `--model` on `gym eval run
  --no-serve` is ignored (the env fixes the policy), overall accuracy is
  `[0].agent_metrics["mean/reward"]`, `quality_summary.json` / `compare_report.*` have fixed
  names (so every run gets its own directory), and `/readyz` on the head is 200 only once
  every server and the policy endpoint answer.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GYM = REPO / ".venv-gym/bin/gym"
DEV_ROWS = Path("/home/ubuntu/data/gym/os_atlas_dev.jsonl")
SSV2_ROWS = Path("/home/ubuntu/data/gym/ssv2_clean.jsonl")
AGENT = "gui_grounding_simple_agent"

EVAL_PORT = 8004
HEAD_PORT = 11010
VERIFY_PORT = 12010
VERIFY_URL = f"http://127.0.0.1:{VERIFY_PORT}/verify"
FINAL_PORT = 8001

SOURCES = ("os_atlas/linux", "os_atlas/macos", "os_atlas/androidworld", "os_atlas/uibert", "os_atlas/fineweb")
_PLATFORMS = ("desktop", "mobile", "web")
# Slices worth a column (app.py compute_metrics also emits per element-role slices; skip those).
SLICES = {*SOURCES, *_PLATFORMS, "text", "icon", *(f"{p}/{t}" for p in _PLATFORMS for t in ("text", "icon"))}

_LIVE: list["Proc"] = []  # every child we started, so a crash can stop them all


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {msg}", flush=True)


class StepError(RuntimeError):
    """A subprocess failed or timed out; the message carries the log tail."""


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


class Proc:
    """A child in its own process group (gym and vLLM spawn trees), logged to a file."""

    def __init__(self, cmd: list, log_path: Path, env: dict | None = None, cwd: Path = REPO):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.cmd = [str(c) for c in cmd]
        self.log_path = log_path
        self._f = open(log_path, "ab")
        self._f.write(f"\n$ {' '.join(self.cmd)}\n".encode())
        self._f.flush()
        self.p = subprocess.Popen(self.cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=self._f,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        _LIVE.append(self)

    def alive(self) -> bool:
        return self.p.poll() is None

    def wait(self, timeout_s: float) -> int:
        try:
            rc = self.p.wait(timeout=max(1.0, timeout_s))
        except subprocess.TimeoutExpired:
            self.kill()
            raise StepError(f"timed out after {timeout_s:.0f}s: {' '.join(self.cmd[:4])}\n{_tail(self.log_path)}")
        self._close()
        return rc

    def kill(self, grace_s: float = 20) -> None:
        """SIGINT the group (gym shuts its servers down on SIGINT, not SIGTERM), then SIGKILL."""
        if self.alive():
            for sig, wait in ((signal.SIGINT, grace_s), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
                try:
                    os.killpg(self.p.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.p.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        try:  # stragglers in the group (ray workers, uvicorn children)
            os.killpg(self.p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self._close()

    def _close(self) -> None:
        if self in _LIVE:
            _LIVE.remove(self)
        if not self._f.closed:
            self._f.close()


def run(cmd: list, log_path: Path, timeout_s: float, env: dict | None = None, ok=(0,)) -> int:
    """Run to completion; raise StepError (with the log tail) on a bad exit code or timeout."""
    proc = Proc(cmd, log_path, env=env)
    rc = proc.wait(timeout_s)
    if rc not in ok:
        raise StepError(f"exit {rc}: {' '.join(proc.cmd[:6])}\n{_tail(log_path)}")
    return rc


def kill_all() -> None:
    for proc in list(_LIVE):
        proc.kill()


def http_status(url: str, timeout: float = 5) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


# --------------------------------------------------------------------------- vLLM


def container_name(port: int) -> str:
    return "vllm-vlm" if port == 8001 else f"vllm-vlm-{port}"


def stop_server(port: int) -> None:
    subprocess.run(["docker", "rm", "-f", container_name(port)], capture_output=True, timeout=120)


def served_adapter(port: int) -> str | None:
    """The `grounder=` path a running container was started with (None: not running / no LoRA)."""
    r = subprocess.run(["docker", "inspect", "-f", "{{json .Args}}", container_name(port)],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None
    state = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container_name(port)],
                           capture_output=True, text=True, timeout=60).stdout.strip()
    if state != "true":
        return None
    for a in json.loads(r.stdout):
        if a.startswith("grounder="):
            return a.split("=", 1)[1]
    return None


def server_models(port: int) -> list[str]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as r:
            return [m["id"] for m in json.load(r)["data"]]
    except Exception:
        return []


def serve(port: int, adapter: Path | str, log_path: Path, gpu_util: float, timeout_s: float = 900) -> None:
    """(Re)start `base` + LoRA `grounder`=adapter on port; no-op if it already serves exactly that."""
    adapter = str(adapter)
    if served_adapter(port) == adapter and "grounder" in server_models(port):
        log(f":{port} already serves grounder={adapter}")
        return
    assert Path(adapter, "adapter_config.json").exists(), f"not a LoRA adapter dir: {adapter}"
    assert adapter.startswith(("/home/ubuntu/ckpt/", "/home/ubuntu/data/")), \
        f"{adapter}: the vLLM container only mounts /home/ubuntu/ckpt and /home/ubuntu/data"
    stop_server(port)
    log(f"starting :{port} with grounder={adapter} (gpu util {gpu_util})")
    env = os.environ | {"GPU_UTIL": str(gpu_util)}
    run([REPO / "scripts/serve_vlm.sh", "qwen3.5", port, "--enable-lora", "--max-lora-rank", "64",
         "--lora-modules", f"grounder={adapter}"], log_path, timeout_s, env=env)
    models = server_models(port)
    if "grounder" not in models:
        raise StepError(f":{port} came up without grounder: {models}")


# --------------------------------------------------------------------------- Gym


def gym_env_vars() -> dict:
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.local/bin'}:{REPO / '.venv-gym/bin'}:{env.get('PATH', '')}"
    env["UV_LINK_MODE"] = "copy"
    env["NEMO_GYM_EXTRA_ROOTS"] = str(REPO)
    env.pop("VIRTUAL_ENV", None)  # the driver runs in the train venv; gym has its own
    env.pop("PYTHONPATH", None)
    return env


class GymEnv:
    """`gym env start` for gui_grounding on HEAD_PORT, policy = `grounder` on the eval server."""

    def __init__(self, log_path: Path, policy_port: int = EVAL_PORT):
        self.log_path = log_path
        self.policy_port = policy_port
        self.proc: Proc | None = None

    def ready(self) -> bool:
        return (self.proc is not None and self.proc.alive()
                and http_status(f"http://127.0.0.1:{HEAD_PORT}/readyz") == 200
                and http_status(f"http://127.0.0.1:{VERIFY_PORT}/health") == 200)

    def ensure(self, timeout_s: float = 1800) -> None:
        if self.ready():
            return
        self.stop()
        for port in (HEAD_PORT, VERIFY_PORT):
            if http_status(f"http://127.0.0.1:{port}/health", timeout=2) != 0:
                raise StepError(f"port {port} is already in use by another process; stop it first")
        log(f"starting gym env (head :{HEAD_PORT}, verify :{VERIFY_PORT}, policy :{self.policy_port} grounder)")
        self.proc = Proc([GYM, "env", "start", "--resources-server", "gui_grounding", "--model-type", "vllm_model",
                          "++server_spinup_timeout_seconds=1800", "++skip_venv_if_present=true",
                          f"++policy_base_url=http://127.0.0.1:{self.policy_port}/v1", "++policy_model_name=grounder",
                          f"++head_server.port={HEAD_PORT}",
                          f"++gui_grounding.resources_servers.gui_grounding.port={VERIFY_PORT}"],
                         self.log_path, env=gym_env_vars())
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if not self.proc.alive():
                raise StepError(f"gym env exited during startup\n{_tail(self.log_path)}")
            if self.ready():
                log(f"gym env ready in {time.time() - t0:.0f}s")
                return
            time.sleep(3)
        self.stop()
        raise StepError(f"gym env not ready after {timeout_s:.0f}s\n{_tail(self.log_path)}")

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc = None


def eval_run(rows: Path, out_dir: Path, *, timeout_s: float, repeats: int = 1, temperature: float | None = None,
             concurrency: int = 64, tolerate_failures: bool = False) -> Path:
    """`gym eval run` into its own directory; returns the rollouts path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "rollouts.jsonl"
    cmd = [GYM, "eval", "run", "--no-serve", "--agent", AGENT, "--input", rows, "--output", out,
           "--concurrency", concurrency, f"++head_server.port={HEAD_PORT}"]
    if repeats > 1:
        cmd += ["--num-repeats", repeats, "+num_repeats_add_seed=true"]
    if temperature is not None:
        cmd += ["--temperature", temperature]
    if tolerate_failures:  # a failed rollout goes to *_failures.jsonl instead of aborting the run
        cmd += ["++route_failures_to_sidecar=true"]
    (out_dir / "command.txt").write_text(" ".join(map(str, cmd)) + "\n")
    run(cmd, out_dir / "eval.log", timeout_s, env=gym_env_vars())
    if not out.exists():
        raise StepError(f"gym eval run wrote no rollouts: {out}")
    return out


def profile(rollouts: Path, timeout_s: float = 900) -> Path:
    inputs = rollouts.with_name(rollouts.stem + "_materialized_inputs.jsonl")
    cmd = [GYM, "eval", "profile", "--inputs", inputs, "--rollouts", rollouts, "++allow_partial_rollouts=true"]
    run(cmd, rollouts.parent / "profile.log", timeout_s, env=gym_env_vars())
    out = rollouts.with_name(rollouts.stem + "_reward_profiling.jsonl")
    if not out.exists():
        raise StepError(f"gym eval profile wrote no {out.name}")
    return out


def compare(baseline: Path, candidate: Path, out_dir: Path, timeout_s: float = 600) -> Path:
    cmd = [GYM, "eval", "compare", "--baseline", baseline, "--candidates", candidate,
           "--output-dir", out_dir, "--report-format", "both"]
    out_dir.mkdir(parents=True, exist_ok=True)
    run(cmd, out_dir / "compare.log", timeout_s, env=gym_env_vars())
    return out_dir / "compare_report.json"


def metrics(rollouts: Path) -> dict:
    """Overall dev/SSv2 accuracy (0-100) and the per-slice accuracies from the aggregate file."""
    agg = json.loads(rollouts.with_name(rollouts.stem + "_aggregate_metrics.json").read_text())[0]["agent_metrics"]
    out = {"acc": round(100 * agg["mean/reward"], 2)}
    for k, v in agg.items():
        if k.removesuffix("/pass@1/accuracy") in SLICES:
            out[k.removesuffix("/pass@1/accuracy")] = round(v, 2)
    if "mean/parsed" in agg:
        out["parsed"] = round(100 * agg["mean/parsed"], 2)
    failures = rollouts.with_name(rollouts.stem + "_failures.jsonl")
    out["failures"] = sum(1 for _ in failures.open()) if failures.exists() else 0
    out["rollouts"] = sum(1 for _ in rollouts.open())
    return out


def headline(m: dict) -> str:
    keys = [k for k in (*SOURCES, "desktop", "mobile", "web", "text", "icon") if k in m]
    return f"{m['acc']:.2f}% (" + ", ".join(f"{k.split('/')[-1]} {m[k]:.1f}" for k in keys) + ")"


# --------------------------------------------------------------------------- CLI


def _eval_cli(args) -> None:
    """Dev (+ SSv2-clean for the results table) for one adapter through the gui_grounding env."""
    out = args.out_root / args.name
    out.mkdir(parents=True, exist_ok=True)
    env = GymEnv(out / "gym_env.log")
    try:
        serve(EVAL_PORT, args.adapter, out / "serve.log", gpu_util=args.gpu_util)
        env.ensure()
        results = {"adapter": str(args.adapter), "name": args.name}
        sets = [("dev", DEV_ROWS)] + ([("ssv2_clean", SSV2_ROWS)] if args.ssv2 else [])
        for name, rows in sets:
            r = eval_run(rows, out / f"{name}_{args.name}", timeout_s=1800, concurrency=64)
            results[name] = metrics(r)
            results[f"{name}_rollouts"] = str(r)
            log(f"{args.name} {name}: {headline(results[name])}")
        (out / "results.json").write_text(json.dumps(results, indent=2))
        print("RESULTS " + json.dumps(results))
    finally:
        env.stop()
        if not args.keep_server:
            stop_server(EVAL_PORT)
        kill_all()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="dev (+ --ssv2) eval of one adapter through Gym")
    e.add_argument("--adapter", type=Path, required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--ssv2", action="store_true", help="also ScreenSpot-v2-clean (results table only)")
    e.add_argument("--out-root", type=Path, default=Path("/home/ubuntu/night/evals"))
    e.add_argument("--gpu-util", type=float, default=0.25)
    e.add_argument("--keep-server", action="store_true")
    args = ap.parse_args()
    if args.cmd == "eval":
        _eval_cli(args)


if __name__ == "__main__":
    sys.exit(main())
