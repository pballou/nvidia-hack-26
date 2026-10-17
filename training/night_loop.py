"""Overnight NeMo Gym improvement loop for the grounder (docs/tasks/B-night-loop.md).

    training/night_loop.sh --init-adapter /home/ubuntu/ckpt/sft-<winner>/adapter \
        --train-file /home/ubuntu/data/os_atlas_20k/<winner's train file>.jsonl

Round r = 1..3:
 1. Read /home/ubuntu/night/knobs.json (the only thing the BLADE orchestrator may change);
    stop cleanly if /home/ubuntu/night/STOP exists.
 2. Profile: a fresh random 4,000-row train subset (seed r, never dev or ScreenSpot)
    through the gui_grounding Gym env, `gym eval run --num-repeats 8 --temperature 1.0`,
    then `gym eval profile`.
 3. Pool: rows with 0 < pass rate < 1, minus exclude_ids, sampled by source_weights,
    capped at pool_max -> round_<r>/pool.jsonl + pool_stats.json.
 4. training/grpo.py: TRL GRPO with the Gym /verify reward, continuing the current LoRA;
    its first 20 steps are gate D2 (<= 35 s/step). If the gate fails, run one
    rejection-sampled SFT round on the reward=1 profile replies instead, then stop.
 5. Dev eval (temperature 0) + `gym eval compare` against the previous dev rollouts.
 6. Keep the adapter if dev improves by >= 0.5 points, else revert and stop.
 7. round_<r>/summary.json + DONE, a runs.md row, then wait <= 10 min for knob changes
    (touch round_<r>/BLADE_DONE to continue at once).
Stops: 3 rounds, no new GRPO step after 02:40, no round starts after 01:30, no dev
gain, STOP, or any crash (reverts to the last kept adapter and records stop_reason).
End: the best kept adapter is served on :8001 as `grounder`, FINAL.json is written last,
and every process this loop started is stopped. The instance itself is left running.

Checkpoints are chosen on dev only; ScreenSpot is never touched here.
"""

import argparse
import datetime as dt
import hashlib
import json
import random
import shutil
import signal
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

from data_prep.gym_rows import gym_row
from data_prep.images import preresize
from training import gymctl
from training.gymctl import StepError, log

TZ = ZoneInfo("America/New_York")
PY = sys.executable

DEFAULT_KNOBS = {
    "pool_max": 2000, "num_generations": 6, "grpo_max_steps": 200, "round_time_budget_min": 75,
    "format_reward": 0.1,
    "source_weights": {"os_atlas/linux": 1.0, "os_atlas/macos": 1.0, "os_atlas/androidworld": 1.0,
                       "os_atlas/uibert": 1.0, "os_atlas/fineweb": 1.0},
    "exclude_ids": [],
}
# type, min, max for the scalar knobs; values outside are clamped, the change is recorded.
KNOB_LIMITS = {"pool_max": (int, 16, 4000), "num_generations": (int, 2, 16), "grpo_max_steps": (int, 1, 2000),
               "round_time_budget_min": (float, 30, 240), "format_reward": (float, 0.0, 1.0)}

PROFILE_REPEATS = 8
PROFILE_MAX_OUTPUT_TOKENS = 32  # = GRPO max_completion_length
MIN_ROLLOUTS_PER_TASK = 4  # pass rates from fewer (infra-failed) rollouts are too noisy for the pool
MIN_POOL = 32
KEEP_MIN_GAIN = 0.5  # dev points
GATE_S_PER_STEP = 35.0
SMOKE_STEPS = 20
DEV_RESERVE_MIN = 8  # round budget kept back for adapter save + server restart + dev eval
MIN_GRPO_WINDOW_MIN = 15  # GRPO always gets load + the 20-step smoke, even if the round budget is spent
EVAL_GPU_UTIL = 0.40  # :8004 during profile / dev (8 repeats x 4k rows)
FINAL_GPU_UTIL = 0.30  # :8001, as today
GRPO_VLLM_UTIL = 0.25  # colocated vLLM inside the trainer (TRL default 0.3, set explicitly)
GRPO_LR = 1e-5  # fixed before the night; not a knob
RS_PER_TASK, RS_MAX_ROWS, RS_LR, RS_EX_PER_S = 2, 6000, 5e-5, 6.0


def now() -> dt.datetime:
    return dt.datetime.now(TZ)


def clock(hhmm: str, start: dt.datetime) -> dt.datetime:
    """The next HH:MM (EDT) at or after start - 12 h, i.e. tonight's, even if it already passed."""
    h, m = map(int, hhmm.split(":"))
    t = start.replace(hour=h, minute=m, second=0, microsecond=0)
    return t + dt.timedelta(days=1) if t < start - dt.timedelta(hours=12) else t


def edt(t: dt.datetime | None) -> str:
    return t.strftime("%a %H:%M") if t else ""


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def reply_text(rollout: dict) -> str:
    texts = []
    for o in rollout.get("response", {}).get("output", []):
        if o.get("type") == "message":
            texts += [c.get("text", "") for c in o.get("content", []) if c.get("type") == "output_text"]
    return "\n".join(texts).strip()


# --------------------------------------------------------------------------- knobs


def load_knobs(path: Path, prev: dict) -> tuple[dict, list[str]]:
    """Validated knobs; anything unreadable keeps the previous value. Returns (knobs, notes)."""
    notes = []
    if not path.exists():
        write_json(path, DEFAULT_KNOBS)
        notes.append("knobs.json missing: wrote defaults")
    try:
        raw = json.loads(path.read_text())
        assert isinstance(raw, dict), "not a JSON object"
    except Exception as e:
        return dict(prev), [f"knobs.json unreadable ({e!r}); kept previous knobs"]
    knobs = dict(prev)
    for k, v in raw.items():
        if k in KNOB_LIMITS:
            typ, lo, hi = KNOB_LIMITS[k]
            try:
                val = typ(v)
            except (TypeError, ValueError):
                notes.append(f"{k}={v!r} invalid; kept {prev[k]}")
                continue
            if not lo <= val <= hi:
                notes.append(f"{k}={val} clamped to [{lo}, {hi}]")
                val = min(hi, max(lo, val))
            knobs[k] = val
        elif k == "source_weights" and isinstance(v, dict):
            weights = dict(prev["source_weights"])
            for src, w in v.items():
                try:
                    weights[str(src)] = max(0.0, float(w))
                except (TypeError, ValueError):
                    notes.append(f"source_weights[{src}]={w!r} invalid; ignored")
            knobs["source_weights"] = weights
        elif k == "exclude_ids" and isinstance(v, list):
            knobs["exclude_ids"] = [str(x) for x in v]
        else:
            notes.append(f"ignored knob {k!r} (not on the approved list or wrong type)")
    return knobs, notes


# --------------------------------------------------------------------------- data


def load_train_rows(train_file: Path, dev_ids: set[str]) -> list[dict]:
    rows = [r for r in read_jsonl(train_file) if r.get("split") == "train"]
    if bad := [r["id"] for r in rows if "screenspot" in r["source"].lower()]:
        raise SystemExit(f"refusing ScreenSpot rows in the train file: {bad[:5]}")
    if leak := {r["id"] for r in rows} & dev_ids:
        raise SystemExit(f"train file shares ids with dev: {sorted(leak)[:5]}")
    assert rows, f"no split=train rows in {train_file}"
    return rows


def pool_from_profile(profile_rows: list[dict], by_id: dict, knobs: dict, seed: int) -> tuple[list[str], dict]:
    """GRPO pool ids (0 < pass rate < 1, knobs applied) and per-source pass-rate stats."""
    excluded, weights = set(knobs["exclude_ids"]), knobs["source_weights"]
    stats: dict[str, Counter] = defaultdict(Counter)
    cands = []
    for p in profile_rows:
        rid, rate, n = p["sample"]["id"], float(p["mean/reward"]), int(p.get("num_rollouts", 0))
        src = by_id[rid]["source"]
        s = stats[src]
        s["tasks"] += 1
        s["rollouts"] += n
        s["pass_sum"] += rate
        s["always" if rate >= 1 else "never" if rate <= 0 else "sometimes"] += 1
        if not 0 < rate < 1:
            continue
        if n < MIN_ROLLOUTS_PER_TASK:
            s["dropped_few_rollouts"] += 1
        elif rid in excluded:
            s["dropped_exclude_ids"] += 1
        elif weights.get(src, 1.0) <= 0:
            s["dropped_weight_0"] += 1
        else:
            cands.append(rid)
    rng = random.Random(seed)
    w = {rid: weights.get(by_id[rid]["source"], 1.0) for rid in cands}
    cap = knobs["pool_max"]
    if len(cands) > cap:  # weighted sampling without replacement (Efraimidis-Spirakis keys)
        pool = sorted(cands, key=lambda rid: rng.random() ** (1 / w[rid]), reverse=True)[:cap]
    else:  # weight > 1 oversamples (repeats), < 1 subsamples
        pool = [rid for rid in cands for _ in range(int(w[rid]) + (rng.random() < w[rid] - int(w[rid])))]
        if len(pool) > cap:
            pool = rng.sample(pool, cap)
    rng.shuffle(pool)
    by_source = {}
    for src, s in sorted(stats.items()):
        by_source[src] = {"tasks": s["tasks"], "pass_rate_mean": round(s["pass_sum"] / max(1, s["tasks"]), 4),
                          "always": s["always"], "sometimes": s["sometimes"], "never": s["never"],
                          **{k: v for k, v in s.items() if k.startswith("dropped_")}}
    return pool, {"profile_by_source": by_source, "candidates": len(cands),
                  "pool_by_source": dict(Counter(by_id[rid]["source"] for rid in pool))}


def rs_sft_rows(rollouts: Path, by_id: dict, knobs: dict, seed: int, max_rows: int) -> tuple[list[dict], dict]:
    """Rejection sampling: up to RS_PER_TASK distinct reward=1 replies per task, as SFT targets."""
    excluded, weights = set(knobs["exclude_ids"]), knobs["source_weights"]
    wins: dict[str, list[str]] = defaultdict(list)
    for r in read_jsonl(rollouts):
        rid = r.get("id")
        if r.get("reward") != 1.0 or r.get("mask_sample") or rid not in by_id or rid in excluded:
            continue
        if weights.get(by_id[rid]["source"], 1.0) <= 0:
            continue
        text = reply_text(r)
        if text and text not in wins[rid]:
            wins[rid].append(text)
    rng = random.Random(seed)
    rows = []
    for rid in sorted(wins):
        for text in rng.sample(wins[rid], min(RS_PER_TASK, len(wins[rid]))):
            rows.append(by_id[rid] | {"target": text})
    rng.shuffle(rows)
    rows = rows[:max_rows]
    return rows, {"tasks_with_reward_1": len(wins), "rows": len(rows),
                  "by_source": dict(Counter(r["source"] for r in rows))}


# --------------------------------------------------------------------------- loop


class Loop:
    def __init__(self, a):
        self.a = a
        self.night: Path = a.night_dir
        self.start = now()
        self.run_id = self.start.strftime("%Y%m%d-%H%M%S")
        self.ckpt: Path = a.ckpt_dir / self.run_id  # per launch: a relaunch never overwrites or reuses adapters
        self.no_new_step = clock(a.no_new_step, self.start)
        self.last_round_start = clock(a.last_round_start, self.start)
        self.hard_end = clock(a.hard_end, self.start)  # every timeout is capped here so finish() runs by then
        self.env = gymctl.GymEnv(self.night / "gym_env.log")
        self.knobs = dict(DEFAULT_KNOBS)
        self.used_ids: set[str] = set()
        self.rounds: list[dict] = []
        self.current = str(a.init_adapter)  # last kept adapter
        self.current_round = 0
        self.dev: dict = {}  # metrics of the current adapter on dev
        self.dev_rollouts: Path | None = None
        self.dev_ids = {r["id"] for r in read_jsonl(gymctl.DEV_ROWS)}
        self.train_rows = load_train_rows(a.train_file, self.dev_ids)
        self.by_id = {r["id"]: r for r in self.train_rows}

    # ---- bookkeeping

    def status(self, phase: str, **extra) -> None:
        write_json(self.night / "STATUS.json", {"phase": phase, "updated": now().isoformat(),
                                                "current_adapter": self.current, "current_round": self.current_round,
                                                **extra})
        log(f"== {phase} {extra if extra else ''}")

    def save_state(self) -> None:
        write_json(self.night / "state.json", {
            "current_adapter": self.current, "current_round": self.current_round, "dev": self.dev,
            "dev_rollouts": str(self.dev_rollouts), "rounds": self.rounds, "used_profile_ids": len(self.used_ids)})

    def runs_row(self, cells: list[str]) -> None:
        line = "| " + " | ".join(c.replace("|", "/").replace("\n", " ") for c in cells) + " |\n"
        with (self.night / "runs_rows.md").open("a") as f:
            f.write(line)
        if self.a.runs_md:
            with self.a.runs_md.open("a") as f:
                f.write(line)

    def stop_requested(self) -> bool:
        return (self.night / "STOP").exists()

    def secs_to(self, t: dt.datetime) -> float:
        return (t - now()).total_seconds()

    def cap(self, seconds: float) -> float:
        """A step timeout that never runs past the hard end."""
        return max(30.0, min(seconds, self.secs_to(self.hard_end)))

    # ---- steps

    def serve_eval(self, adapter: str, log_path: Path) -> None:
        gymctl.serve(gymctl.EVAL_PORT, adapter, log_path, gpu_util=EVAL_GPU_UTIL, timeout_s=self.cap(900))
        self.env.ensure(self.cap(1800))

    def dev_eval(self, adapter: str, out_dir: Path) -> tuple[dict, Path]:
        if self.secs_to(self.hard_end) < 150:
            raise StepError(f"past the {edt(self.hard_end)} hard end: no time for a dev eval")
        self.serve_eval(adapter, out_dir.parent / "serve.log")
        try:
            rollouts = gymctl.eval_run(gymctl.DEV_ROWS, out_dir, timeout_s=self.cap(1200), concurrency=64)
        except StepError as e:  # one retry: a transient model/env failure aborts the whole gym run
            if self.secs_to(self.hard_end) < 300:
                raise
            log(f"dev eval failed once, retrying: {e}")
            self.env.ensure(self.cap(1800))
            rollouts = gymctl.eval_run(gymctl.DEV_ROWS, out_dir, timeout_s=self.cap(1200), concurrency=64)
        m = gymctl.metrics(rollouts)
        log(f"dev {adapter}: {gymctl.headline(m)}")
        return m, rollouts

    def baseline(self) -> None:
        rd = self.night / "round_0"
        rd.mkdir(parents=True, exist_ok=True)
        self.status("round 0: dev baseline of the SFT adapter", adapter=self.current)
        self.dev, self.dev_rollouts = self.dev_eval(self.current, rd / "dev_r0")
        summary = {"round": 0, "checkpoint": self.current, "dev_acc_new": self.dev["acc"], "dev": self.dev,
                   "dev_rollouts": str(self.dev_rollouts), "kept": True, "time": now().isoformat()}
        write_json(rd / "summary.json", summary)
        self.rounds.append(summary)
        self.save_state()

    def profile(self, r: int, rd: Path) -> tuple[Path, list[dict], dict]:
        avail = [row for row in self.train_rows if row["id"] not in self.used_ids]
        n = min(self.a.profile_rows, len(self.train_rows))
        if len(avail) < n:
            log(f"only {len(avail)} unused train rows; sampling from all {len(self.train_rows)}")
            avail = self.train_rows
        subset = random.Random(r).sample(avail, n)
        self.used_ids |= {s["id"] for s in subset}
        assert not ({s["id"] for s in subset} & self.dev_ids)
        cached = preresize({s["image_path"] for s in subset}, self.a.train_file.parent)
        rows = [gym_row(s, "file://" + cached[s["image_path"]], "qwen3_5", PROFILE_MAX_OUTPUT_TOKENS) for s in subset]
        write_jsonl(rd / "profile_input.jsonl", rows)
        input_by_id = {row["id"]: row for row in rows}

        self.status(f"round {r}: profile", rows=len(rows), repeats=PROFILE_REPEATS)
        self.serve_eval(self.current, rd / "serve.log")
        timeout = min(75 * 60, self.secs_to(self.no_new_step))
        if timeout < 10 * 60:
            raise StepError("too close to the 02:40 cutoff to profile")
        rollouts = gymctl.eval_run(rd / "profile_input.jsonl", rd / f"profile_r{r}", timeout_s=timeout,
                                   repeats=PROFILE_REPEATS, temperature=1.0, concurrency=192, tolerate_failures=True)
        prof = read_jsonl(gymctl.profile(rollouts))
        return rollouts, prof, input_by_id

    def grpo(self, r: int, rd: Path, knobs: dict, t_round: dt.datetime) -> tuple[str, dict]:
        """('ok' | 'gate_failed' | 'crashed_before_smoke' | 'crashed', info).

        info['adapter'] is set for 'ok' and for a crash after the smoke that left a checkpoint."""
        budget_end = t_round + dt.timedelta(minutes=knobs["round_time_budget_min"] - DEV_RESERVE_MIN)
        deadline = min(self.no_new_step, max(budget_end, now() + dt.timedelta(minutes=MIN_GRPO_WINDOW_MIN)))
        if self.secs_to(deadline) < 8 * 60:
            raise StepError(f"no time left for GRPO before the {edt(self.no_new_step)} cutoff")
        out = self.ckpt / f"round_{r}" / "grpo"
        shutil.rmtree(out, ignore_errors=True)
        self.env.ensure(self.cap(1800))  # /verify must be up before the trainer starts
        cmd = [PY, "-m", "training.grpo", "--init-adapter", self.current, "--pool", rd / "pool.jsonl", "--out", out,
               "--verify-url", gymctl.VERIFY_URL, "--num-generations", knobs["num_generations"],
               "--max-steps", knobs["grpo_max_steps"], "--deadline", deadline.timestamp(),
               "--stop-file", self.night / "STOP", "--format-reward", knobs["format_reward"], "--seed", r,
               "--smoke-steps", SMOKE_STEPS, "--gate-s-per-step", self.a.gate_s_per_step,
               "--vllm-gpu-util", GRPO_VLLM_UTIL, "--lr", GRPO_LR]
        (rd / "grpo_command.txt").write_text(" ".join(map(str, cmd)) + "\n")
        self.status(f"round {r}: GRPO", deadline=edt(deadline), pool=rd / "pool.jsonl")
        (rd / "grpo.log").unlink(missing_ok=True)
        proc = gymctl.Proc(cmd, rd / "grpo.log")
        try:
            rc = proc.wait(self.cap(self.secs_to(deadline) + 10 * 60))  # + last step, save
        except StepError as e:
            log(f"GRPO timed out: {e}")
            rc = None
        summary_file = out / "grpo_summary.json"
        info = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        info["exit_code"] = rc
        if rc == 10 or (info.get("gate_d2") and not info["gate_d2"]["passed"]):
            return "gate_failed", info
        if info.get("adapter") and Path(info["adapter"], "adapter_config.json").exists():
            return "ok", info
        gate = None
        for line in (rd / "grpo.log").read_text(errors="replace").splitlines():
            if line.startswith("GATE_D2 "):
                gate = json.loads(line[len("GATE_D2 "):])
        info["gate_d2"] = gate
        info["error_tail"] = gymctl._tail(rd / "grpo.log", 30)
        if gate is None:  # died before the smoke finished
            return "crashed_before_smoke", info
        ckpts = sorted(out.glob("checkpoint-*/adapter_config.json"), key=lambda p: int(p.parent.name.split("-")[1]))
        if gate["passed"] and ckpts:
            info["adapter"] = str(ckpts[-1].parent)
            info["recovered_from_checkpoint"] = True
        return "crashed", info

    def rs_sft(self, r: int, rd: Path, knobs: dict, rollouts: Path) -> dict:
        """Fallback round: SFT on the reward=1 profile replies, continuing the current LoRA."""
        secs = self.secs_to(self.no_new_step) - 6 * 60  # model load + save
        max_rows = min(RS_MAX_ROWS, int(secs * RS_EX_PER_S))
        if max_rows < 256:
            raise StepError("no time left for the rejection-sampled SFT round")
        rows, stats = rs_sft_rows(rollouts, self.by_id, knobs, seed=r, max_rows=max_rows)
        if len(rows) < 64:
            raise StepError(f"only {len(rows)} reward=1 replies for rejection-sampled SFT")
        data = rd / "rs_sft" / "train.jsonl"
        write_jsonl(data, rows)
        out = self.ckpt / f"round_{r}" / "rs_sft"
        shutil.rmtree(out, ignore_errors=True)
        cmd = [PY, "-m", "training.sft", "--model", "qwen3.5", "--train", data, "--image-root",
               self.a.train_file.parent, "--init-adapter", self.current, "--out", out, "--lr", RS_LR,
               "--seed", r, "--save-steps", 100000]
        (rd / "rs_sft" / "command.txt").write_text(" ".join(map(str, cmd)) + "\n")
        write_json(rd / "rs_sft" / "stats.json", stats | {"source_rollouts": str(rollouts)})
        self.status(f"round {r}: rejection-sampled SFT (GRPO fallback)", rows=len(rows))
        gymctl.run(cmd, rd / "rs_sft.log", timeout_s=self.cap(max(600, self.secs_to(self.no_new_step) + 15 * 60)))
        sft_summary = json.loads((out / "summary.json").read_text())
        return {"adapter": str(out / "adapter"), "data": str(data), **stats, "sft": sft_summary}

    def round(self, r: int) -> dict:
        rd = self.night / f"round_{r}"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "DONE").unlink(missing_ok=True)
        (rd / "BLADE_DONE").unlink(missing_ok=True)
        t0 = now()
        self.knobs, notes = load_knobs(self.night / "knobs.json", self.knobs)
        knobs = self.knobs
        write_json(rd / "knobs_used.json", {"knobs": knobs, "notes": notes})
        for n in notes:
            log(f"knobs: {n}")
        s = {"round": r, "start": t0.isoformat(), "checkpoint_in": self.current, "checkpoint": None,
             "method": None, "pool_size": 0, "pool_by_source": {}, "grpo_steps": 0, "s_per_step": None,
             "reward_mean_first": None, "reward_mean_last": None, "dev_acc_prev": self.dev.get("acc"),
             "dev_acc_new": None, "kept": False, "knobs": knobs}
        finished = False
        try:
            rollouts, prof, input_by_id = self.profile(r, rd)
            pool, stats = pool_from_profile(prof, self.by_id, knobs, seed=r)
            write_jsonl(rd / "pool.jsonl", (input_by_id[rid] for rid in pool))
            write_json(rd / "pool_stats.json", stats)
            s.update(pool_size=len(pool), pool_by_source=stats["pool_by_source"],
                     profile_rollouts=str(rollouts), profile_by_source=stats["profile_by_source"])
            log(f"pool {len(pool)} rows {stats['pool_by_source']} from {stats['candidates']} sometimes-pass tasks")
            if self.stop_requested():
                s["stop_reason"] = "STOP file"
                finished = True
                return s

            gymctl.stop_server(gymctl.EVAL_PORT)  # GPU for the trainer; :8001 stays up
            time.sleep(20)  # vLLM's startup profiler asserts if another process frees GPU memory meanwhile
            outcome, info = ("pool_too_small", {"note": f"pool too small ({len(pool)} < {MIN_POOL})"}) \
                if len(pool) < MIN_POOL else self.grpo(r, rd, knobs, t0)
            s["grpo"] = info
            s.update(grpo_steps=info.get("grpo_steps", 0), s_per_step=info.get("s_per_step"),
                     reward_mean_first=info.get("reward_mean_first"), reward_mean_last=info.get("reward_mean_last"))
            stop_after = None
            s["grpo_outcome"] = outcome
            if outcome in ("gate_failed", "crashed_before_smoke", "pool_too_small"):
                why = {"gate_failed": f"GRPO failed gate D2 ({info.get('gate_d2')})",
                       "crashed_before_smoke": "GRPO crashed before finishing its 20-step smoke",
                       "pool_too_small": f"GRPO skipped: {info.get('note')}"}[outcome]
                log(f"{why}; falling back to one rejection-sampled SFT round\n{info.get('error_tail') or ''}")
                rs = self.rs_sft(r, rd, knobs, rollouts)
                s["rs_sft"] = rs
                s["method"], candidate = "rejection_sampled_sft", rs["adapter"]
                stop_after = f"{why}: ran the one rejection-sampled SFT round"
            elif outcome == "crashed" and not info.get("adapter"):
                raise StepError(f"GRPO crashed after the smoke with no checkpoint:\n{info.get('error_tail')}")
            else:
                s["method"], candidate = "grpo", info["adapter"]
                if outcome == "crashed":
                    stop_after = f"GRPO crashed; evaluated its last checkpoint {candidate}"
                elif info.get("stop_reason") == "stop_file":
                    stop_after = "STOP file"
                elif info.get("stop_reason") == "deadline" and self.secs_to(self.no_new_step) <= 0:
                    stop_after = "02:40 cutoff reached during GRPO"

            self.status(f"round {r}: dev eval", candidate=candidate)
            m, dev_rollouts = self.dev_eval(candidate, rd / f"dev_r{r}")
            s.update(checkpoint=candidate, dev_acc_new=m["acc"], dev=m, dev_rollouts=str(dev_rollouts))
            try:
                s["compare_report"] = str(gymctl.compare(self.dev_rollouts, dev_rollouts, rd / "compare",
                                                         timeout_s=self.cap(600)))
            except Exception as e:  # the report is an artifact, not the gate
                log(f"gym eval compare failed: {e}")
            gain = m["acc"] - self.dev["acc"]
            s["dev_gain"] = round(gain, 2)
            if gain >= KEEP_MIN_GAIN:
                s["kept"] = True
                self.current, self.current_round = candidate, r
                self.dev, self.dev_rollouts = m, dev_rollouts
                log(f"round {r} KEPT: dev {s['dev_acc_prev']:.2f} -> {m['acc']:.2f} ({gain:+.2f})")
            else:
                stop_after = stop_after or f"no dev improvement ({gain:+.2f} pts < +{KEEP_MIN_GAIN})"
                log(f"round {r} REVERTED: dev {s['dev_acc_prev']:.2f} -> {m['acc']:.2f} ({gain:+.2f})")
            if stop_after:
                s["stop_reason"] = stop_after
            finished = True
            return s
        except Exception as e:
            s["stop_reason"] = f"crash in round {r}: {type(e).__name__}: {str(e)[:1500]}"
            s["traceback"] = traceback.format_exc()[-4000:]
            log(s["stop_reason"])
            finished = True
            return s
        finally:
            if not finished:
                s.setdefault("stop_reason", "interrupted")
            s["end"] = now().isoformat()
            s["dev_acc_current"] = self.dev.get("acc")
            s["checkpoint_current"] = self.current
            write_json(rd / "summary.json", s)
            (rd / "DONE").touch()
            self.rounds.append(s)
            self.save_state()
            dev_txt = (f"dev {s['dev_acc_prev']} → {s['dev_acc_new']} ({'kept' if s['kept'] else 'reverted'})"
                       if s["dev_acc_new"] is not None else "no dev eval")
            self.runs_row([edt(t0), edt(now()), "Kyle (night loop)",
                           f"Night loop round {r} ({s['method'] or 'n/a'}): profile {self.a.profile_rows}×"
                           f"{PROFILE_REPEATS} @T=1 → pool {s['pool_size']} → train → dev",
                           f"`training/night_loop.sh` round {r}; knobs {json.dumps({k: v for k, v in knobs.items() if k != 'exclude_ids'})}, "
                           f"exclude_ids {len(knobs['exclude_ids'])}; GRPO lr {GRPO_LR}",
                           f"{rd}, {self.ckpt / f'round_{r}'}",
                           f"{dev_txt}; {s['grpo_steps']} steps @ {s['s_per_step']} s/step; verify reward "
                           f"{s['reward_mean_first']} → {s['reward_mean_last']}"
                           + (f"; stop: {s['stop_reason'][:200]}" if s.get("stop_reason") else "")])

    def wait_for_blade(self, r: int) -> str | None:
        """Up to 10 min for the orchestrator's BLADE pass; returns a stop reason or None."""
        knobs = self.night / "knobs.json"
        digest = lambda: hashlib.sha256(knobs.read_bytes()).hexdigest() if knobs.exists() else None  # noqa: E731
        h0 = digest()
        end = min(now() + dt.timedelta(minutes=self.a.blade_wait_min), self.last_round_start)
        self.status(f"round {r} done: waiting for BLADE / knobs.json until {edt(end)}",
                    hint=f"touch {self.night / f'round_{r}' / 'BLADE_DONE'} to continue now")
        while now() < end:
            if self.stop_requested():
                return "STOP file"
            if (self.night / f"round_{r}" / "BLADE_DONE").exists():
                log("BLADE_DONE: continuing")
                return None
            if digest() != h0:
                log("knobs.json changed: continuing in 30 s")
                time.sleep(30)
                return None
            time.sleep(10)
        return None

    def archive_previous(self) -> None:
        """Move an earlier launch's artifacts aside so the orchestrator never sees a stale FINAL.json / DONE."""
        old = [p for p in self.night.iterdir()
               if p.name.startswith("round_") or p.name in ("FINAL.json", "STATUS.json", "state.json", "loop.json")]
        if old:
            dest = self.night / f"prev_{self.run_id}"
            dest.mkdir()
            for p in old:
                p.rename(dest / p.name)
            log(f"moved {len(old)} artifacts of an earlier launch to {dest}")

    def run(self) -> str:
        self.archive_previous()
        write_json(self.night / "loop.json", {
            "args": {k: str(v) for k, v in vars(self.a).items()}, "start": self.start.isoformat(),
            "no_new_grpo_step_after": self.no_new_step.isoformat(),
            "no_round_starts_after": self.last_round_start.isoformat(), "hard_end": self.hard_end.isoformat(),
            "run_id": self.run_id, "ckpt_dir": str(self.ckpt),
            "code_revision": (gymctl.REPO / "REVISION").read_text().strip() if (gymctl.REPO / "REVISION").exists() else None,
            "train_rows": len(self.train_rows)})
        if not (self.night / "knobs.json").exists():
            write_json(self.night / "knobs.json", DEFAULT_KNOBS)
        for port in self.a.stop_ports:
            log(f"stopping :{port} before the loop")
            gymctl.stop_server(port)
        self.baseline()
        for r in range(1, self.a.max_rounds + 1):
            if self.stop_requested():
                return "STOP file"
            if now() >= self.last_round_start:
                return f"no round starts after {edt(self.last_round_start)}"
            s = self.round(r)
            if s.get("stop_reason"):
                return s["stop_reason"]
            if r == self.a.max_rounds:
                return f"{r} rounds done"
            if reason := self.wait_for_blade(r):
                return reason
        return "no rounds run"

    def finish(self, stop_reason: str) -> None:
        self.status("finishing", stop_reason=stop_reason)
        for step in (gymctl.kill_all, self.env.stop, lambda: gymctl.stop_server(gymctl.EVAL_PORT)):
            try:
                step()
            except Exception as e:
                log(f"cleanup step failed: {e!r}")
        served, serve_error = None, None
        if self.a.final_port:
            for adapter in dict.fromkeys([self.current, str(self.a.init_adapter)]):
                try:
                    gymctl.serve(self.a.final_port, adapter, self.night / "serve_final.log", gpu_util=FINAL_GPU_UTIL,
                                 timeout_s=420)
                    served = adapter
                    break
                except Exception as e:
                    serve_error = f"{type(e).__name__}: {str(e)[:500]}"
                    log(f"serving {adapter} on :{self.a.final_port} failed: {serve_error}")
        table = [{"round": s["round"], "method": s.get("method", "sft"), "checkpoint": s.get("checkpoint"),
                  "dev_acc": s.get("dev_acc_new"), "kept": s.get("kept"),
                  "dev_slices": {k: v for k, v in (s.get("dev") or {}).items() if k != "acc"},
                  "stop_reason": s.get("stop_reason")} for s in self.rounds]
        final = {"best_round": self.current_round, "adapter_path": self.current, "dev_acc": self.dev.get("acc"),
                 "dev_table": table, "stop_reason": stop_reason,
                 "served": {"port": self.a.final_port, "name": "grounder", "adapter": served, "error": serve_error},
                 "start": self.start.isoformat(), "finished": now().isoformat()}
        self.runs_row([edt(self.start), edt(now()), "Kyle (night loop)", "Night loop end: serve best adapter",
                       f"`training/night_loop.sh`; FINAL.json", str(self.night / "FINAL.json"),
                       f"best round {self.current_round} ({self.current}) dev {self.dev.get('acc')}; "
                       f"served on :{self.a.final_port}: {served}; stop: {stop_reason[:200]}"])
        write_json(self.night / "FINAL.json", final)  # last: the orchestrator stops the box once this exists
        self.status("done", stop_reason=stop_reason, served=served)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-adapter", type=Path, required=True, help="round-0 adapter (picked on dev)")
    ap.add_argument("--train-file", type=Path, required=True, help="the round-0 run's train manifest")
    ap.add_argument("--night-dir", type=Path, default=Path("/home/ubuntu/night"))
    ap.add_argument("--ckpt-dir", type=Path, default=Path("/home/ubuntu/ckpt/night"),
                    help="adapters go to <dir>/<launch time>/round_<r>; must be under /home/ubuntu/ckpt (vLLM mounts)")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--profile-rows", type=int, default=4000)
    ap.add_argument("--no-new-step", default="02:40", help="EDT; no GRPO step starts after this")
    ap.add_argument("--last-round-start", default="01:30", help="EDT; no round starts after this")
    ap.add_argument("--hard-end", default="02:50", help="EDT; every step's timeout is capped here, then finish()")
    ap.add_argument("--blade-wait-min", type=float, default=10)
    ap.add_argument("--gate-s-per-step", type=float, default=GATE_S_PER_STEP, help="gate D2 (rehearsals on a busy GPU)")
    ap.add_argument("--stop-ports", type=int, nargs="*", default=[8002, 8003])
    ap.add_argument("--final-port", type=int, default=gymctl.FINAL_PORT, help="0 = don't touch any team server")
    ap.add_argument("--runs-md", type=lambda v: None if v.lower() in ("", "none") else Path(v),
                    default=gymctl.REPO / "docs/runs.md", help="append one row per round ('none' for rehearsals)")
    a = ap.parse_args()
    a.night_dir.mkdir(parents=True, exist_ok=True)
    a.ckpt_dir.mkdir(parents=True, exist_ok=True)
    gymctl.LOG_FILE = a.night_dir / "night_loop.log"
    if (a.night_dir / "STOP").exists():
        sys.exit(f"{a.night_dir / 'STOP'} exists; remove it to start the loop")
    assert str(a.ckpt_dir).startswith("/home/ubuntu/ckpt"), "adapters must live under /home/ubuntu/ckpt"
    assert Path(a.init_adapter, "adapter_config.json").exists(), a.init_adapter

    def on_signal(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)
    loop = Loop(a)
    reason = "crash before the first round"
    try:
        reason = loop.run()
    except KeyboardInterrupt as e:
        reason = f"interrupted ({e})"
    except Exception as e:
        reason = f"crash: {type(e).__name__}: {str(e)[:1500]}"
        log(traceback.format_exc())
    finally:
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):  # finish() must not be interrupted
            signal.signal(sig, signal.SIG_IGN)
        log(f"stopping: {reason}")
        loop.finish(reason)


if __name__ == "__main__":
    main()
