"""Run the interactive agent on sandbox tasks and verify each one from sandbox state.

This is a verifier caller: it reads /api/state and runs sandbox.checks.evaluate after
each run. The agent itself never does.

Examples:
  # loop mechanics, no GPU and no T1 pages: fake sandbox + DOM oracle grounder (debug only)
  python scripts/run_agent_tasks.py --fake-sandbox --use-oracle-grounder --tags easy
  # the real sandbox (T1) on :8090 with the base VLM on :8001
  python scripts/run_agent_tasks.py --tasks sandbox/tasks.json --ids text-size-large,high-contrast
  # an ablation arm: every task x 3, two at a time, guarded against an adapter swap and memory pressure
  python scripts/run_agent_tasks.py --tasks sandbox/tasks.json --grounder grounder --thinking on \
      --repeats 3 --concurrency 2 --pin-grounder-adapter --pause-on-memory-pressure

Writes <runs_dir>/summary_<timestamp>.json with per-task results (including the checkpoint
the Lightning endpoint reported serving); traces go to <runs_dir>/<run_id>/.
--pin-grounder-adapter records what the grounder served name maps to (VLM /v1/models) at the
start and stops launching runs if it changes (exit 3; rerun the arm). --pause-on-memory-pressure
waits while macOS reports memory pressure at warn (yellow) or worse.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

# Run as a file (python scripts/run_agent_tasks.py), so put the repo root on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402

from agent.__main__ import add_model_args, config_from_args, print_event  # noqa: E402
from agent.config import AgentConfig  # noqa: E402
from agent.core import run_task  # noqa: E402
from sandbox.checks import evaluate  # noqa: E402


def select_tasks(tasks: list[dict], ids: str | None, tags: str | None) -> list[dict]:
    if ids:
        wanted = [i.strip() for i in ids.split(",") if i.strip()]
        by_id = {t["id"]: t for t in tasks}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise SystemExit(f"unknown task id(s): {', '.join(missing)}")
        tasks = [by_id[i] for i in wanted]
    if tags:
        wanted_tags = {t.strip() for t in tags.split(",") if t.strip()}
        tasks = [t for t in tasks if wanted_tags & set(t.get("tags", []))]
    return tasks


def start_url_for(sandbox: str, start_path: str, sid: str) -> str:
    return sandbox.rstrip("/") + start_path + ("&" if "?" in start_path else "?") + "sid=" + sid


STAT_KEYS = ("n_tool_calls", "n_invalid_calls", "n_no_tool_call", "n_truncated", "n_failed_actions", "n_no_change",
             "n_recoveries", "n_thinking_calls")


def done_stats(trace_dir: str | None) -> tuple[str | None, dict]:
    """(reason, stats) from the run's done event."""
    try:
        last = json.loads((Path(trace_dir) / "events.jsonl").read_text().splitlines()[-1])
    except (OSError, ValueError, IndexError, TypeError):
        return None, {}
    if last.get("kind") != "done":
        return None, {}
    p = last["payload"]
    return p.get("reason"), {k: p.get("stats", {}).get(k, 0) for k in STAT_KEYS}


def lightning_root(trace_dir: str | None) -> str | None:
    """The checkpoint the Lightning endpoint reported serving (start event; agent/core.py)."""
    try:
        first = json.loads((Path(trace_dir) / "events.jsonl").read_text().splitlines()[0])
    except (OSError, ValueError, IndexError, TypeError):
        return None
    return ((first.get("payload") or {}).get("config") or {}).get("lightning", {}).get("root")


async def run_one(task: dict, cfg: AgentConfig, sandbox: str, http: httpx.AsyncClient, pool, quiet: bool) -> dict:
    sid = uuid.uuid4().hex
    row = {"task_id": task["id"], "tags": task.get("tags", []), "verified_success": False, "success_claimed": False,
           "steps": 0, "wall_s": None, "reason": None, "run_id": None, "trace_dir": None, "explanation": ""}
    try:
        r = await http.post(f"{sandbox}/api/reset", json={"sid": sid, "task_id": task["id"]})
        r.raise_for_status()
        t0 = time.perf_counter()
        result = await run_task(task["request"], start_url=start_url_for(sandbox, task["start_path"], sid), cfg=cfg,
                                on_event=None if quiet else print_event, pool=pool)
        row["wall_s"] = round(time.perf_counter() - t0, 1)
        reason, stats = done_stats(result.trace_dir)
        row.update(success_claimed=result.success_claimed, steps=result.steps, run_id=result.run_id,
                   trace_dir=result.trace_dir, summary=result.summary, reason=reason, stats=stats,
                   lightning_root=lightning_root(result.trace_dir))
        r = await http.get(f"{sandbox}/api/state", params={"sid": sid})
        r.raise_for_status()
        passed, explanation = evaluate(r.json(), task["check"])
        row.update(verified_success=bool(passed), explanation=explanation)
    except Exception as e:  # one broken task must not stop the batch
        row["explanation"] = f"harness error: {type(e).__name__}: {e}"
    return row


def totals(rows: list[dict]) -> dict:
    out = {k: sum(r.get("stats", {}).get(k, 0) for r in rows) for k in STAT_KEYS}
    n = out["n_tool_calls"]
    out["invalid_call_rate"] = round(out["n_invalid_calls"] / n, 3) if n else None
    return out


def print_table(rows: list[dict]) -> None:
    w = max([len("task"), *(len(r["task_id"]) for r in rows)])
    print(f"\n{'task':<{w}}  verified  claimed  steps  wall_s  calls  invalid  recov  reason     run_id")
    for r in rows:
        s = r.get("stats", {})
        wall = f"{r['wall_s']:.1f}" if r["wall_s"] is not None else "-"
        print(f"{r['task_id']:<{w}}  {'PASS' if r['verified_success'] else 'fail':<8}  "
              f"{'yes' if r['success_claimed'] else 'no':<7}  {r['steps']:>5}  {wall:>6}  "
              f"{s.get('n_tool_calls', 0):>5}  {s.get('n_invalid_calls', 0):>7}  {s.get('n_recoveries', 0):>5}  "
              f"{r['reason'] or '-':<9}  {r['run_id'] or '-'}")
        if not r["verified_success"]:
            print(f"{'':<{w}}  -> {r['explanation'][:160]}")
    n = sum(r["verified_success"] for r in rows)
    t = totals(rows)
    rate = f"{t['invalid_call_rate']:.1%}" if t["invalid_call_rate"] is not None else "-"
    print(f"\nverified {n}/{len(rows)} | tool calls {t['n_tool_calls']}, invalid {t['n_invalid_calls']} ({rate}), "
          f"no tool call {t['n_no_tool_call']} (cut off {t['n_truncated']}), recoveries {t['n_recoveries']}, "
          f"thinking calls {t['n_thinking_calls']}")
    print(f"Lightning served: {', '.join(lightning_roots(rows)) or 'unknown'}")


def lightning_roots(rows: list[dict]) -> list[str]:
    return sorted({str(r.get("lightning_root")) for r in rows if r.get("run_id")})


async def served_root(http: httpx.AsyncClient, cfg: AgentConfig, name: str) -> str | None:
    """What a VLM served name maps to (vLLM /v1/models: the LoRA path for `grounder`), or None."""
    try:
        r = await http.get(f"{cfg.vlm.base_url.rstrip('/')}/models", headers={"Authorization": f"Bearer {cfg.vlm.api_key}"})
        r.raise_for_status()
        return next((m.get("root") for m in r.json().get("data", []) if m.get("id") == name), None)
    except Exception:
        return None


def memory_pressure_level() -> int:
    """macOS kern.memorystatus_vm_pressure_level: 1 normal, 2 warn (yellow), 4 critical; 1 elsewhere."""
    if sys.platform != "darwin":
        return 1
    import subprocess

    try:
        return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True,
                                  text=True, timeout=5).stdout.strip() or 1)
    except Exception:
        return 1


async def wait_for_memory(poll_s: float = 30.0) -> None:
    if memory_pressure_level() < 2:
        return
    print("=== memory pressure is warn or worse: pausing until it is normal", flush=True)
    while memory_pressure_level() >= 2:
        await asyncio.sleep(poll_s)
    print("=== memory pressure normal again: resuming", flush=True)


async def amain(args: argparse.Namespace) -> int:
    from agent import browser

    cfg = config_from_args(args)
    server = None
    if args.fake_sandbox:
        from agent.testing.fake_sandbox import FAKE_TASKS, FakeSandboxServer

        server = FakeSandboxServer()
        sandbox, all_tasks = server.start(), FAKE_TASKS
    else:
        sandbox = (args.sandbox_url or cfg.sandbox_url).rstrip("/")
        all_tasks = json.loads(Path(args.tasks).read_text())
    tasks = select_tasks(all_tasks, args.ids, args.tags)
    if not tasks:
        raise SystemExit("no tasks selected")
    cfg = cfg.with_options(sandbox_url=sandbox)

    rows: list[dict] = []
    started = time.strftime("%Y%m%d-%H%M%S")
    jobs = [(rep, task) for rep in range(args.repeats) for task in tasks]
    adapter, aborted = None, None
    try:
        pool = await browser.get_pool(cfg.browser)
        async with httpx.AsyncClient(timeout=30, trust_env=False) as http:
            if args.pin_grounder_adapter:
                adapter = await served_root(http, cfg, cfg.grounder_model)
                if adapter is None:
                    raise SystemExit(f"--pin-grounder-adapter: {cfg.vlm.base_url} doesn't serve {cfg.grounder_model!r}")
                print(f"=== grounder {cfg.grounder_model!r} = {adapter}", flush=True)
            sem = asyncio.Semaphore(args.concurrency)

            async def run_job(i: int, rep: int, task: dict) -> None:
                nonlocal aborted
                async with sem:
                    if aborted:
                        return
                    if args.pause_file and Path(args.pause_file).exists():
                        print(f"=== {args.pause_file} exists: holding before the next run", flush=True)
                        while Path(args.pause_file).exists():
                            await asyncio.sleep(10)
                        print("=== pause file removed: resuming", flush=True)
                    if args.pause_on_memory_pressure:
                        await wait_for_memory()
                    if adapter is not None and (now := await served_root(http, cfg, cfg.grounder_model)) != adapter:
                        aborted = f"grounder adapter changed before job {i}: {adapter} -> {now}"
                        return
                    print(f"\n=== [{i}/{len(jobs)}] {task['id']} (repeat {rep + 1}): {task['request']}", flush=True)
                    row = await run_one(task, cfg, sandbox, http, pool, args.quiet)
                    row["repeat"] = rep
                    if adapter is not None and (now := await served_root(http, cfg, cfg.grounder_model)) != adapter:
                        row["adapter_changed"] = True  # served something else at some point: don't count it
                        aborted = f"grounder adapter changed during {task['id']} repeat {rep + 1}: {adapter} -> {now}"
                    rows.append(row)
                    print(f"=== {task['id']}: {'PASS' if row['verified_success'] else 'fail'} "
                          f"({row['explanation'][:120]})", flush=True)

            await asyncio.gather(*(run_job(i, rep, task) for i, (rep, task) in enumerate(jobs, 1)))
    finally:
        await browser.close_pool()
        if server is not None:
            server.stop()

    rows.sort(key=lambda r: (r.get("repeat", 0), [t["id"] for t in tasks].index(r["task_id"])))
    print_table(rows)
    out = cfg.run_dir_root / f"summary_{started}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "created": started, "sandbox_url": sandbox, "fake_sandbox": bool(args.fake_sandbox),
        # which checkpoint answered: the URL and served name are the same for every local build
        "config": cfg.public_dict(), "lightning_roots": lightning_roots(rows),
        "grounder_adapter": adapter, "repeats": args.repeats, "concurrency": args.concurrency, "aborted": aborted,
        "n_tasks": len(rows), "n_verified": sum(r["verified_success"] for r in rows),
        "totals": totals(rows), "tasks": rows,
    }, indent=2))
    print(f"summary: {out}")
    if aborted:
        print(f"ABORTED: {aborted}. Rerun this arm.", file=sys.stderr)
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group()
    src.add_argument("--tasks", default=str(REPO_ROOT / "sandbox" / "tasks.json"), help="tasks.json (sandbox schema)")
    src.add_argument("--fake-sandbox", action="store_true",
                     help="start agent/testing/fake_sandbox.py in-process and run its FAKE_TASKS")
    p.add_argument("--ids", help="comma-separated task ids (in this order)")
    p.add_argument("--tags", help="comma-separated tags; keeps tasks with any of them")
    p.add_argument("--sandbox-url", help="default: sandbox_url from the config")
    p.add_argument("--quiet", action="store_true", help="don't print per-event lines")
    p.add_argument("--repeats", type=int, default=1, help="run every selected task this many times")
    p.add_argument("--concurrency", type=int, default=1, help="runs in flight at once (Lightning is shared)")
    p.add_argument("--pin-grounder-adapter", action="store_true",
                   help="record the grounder's served adapter (VLM /v1/models) and abort (exit 3) if it changes")
    p.add_argument("--pause-on-memory-pressure", action="store_true",
                   help="macOS: wait before each run while memory pressure is warn (yellow) or worse")
    p.add_argument("--pause-file", help="hold before each run while this file exists (runs in flight finish)")
    add_model_args(p)
    args = p.parse_args(argv)
    if args.repeats < 1 or args.concurrency < 1:
        p.error("--repeats and --concurrency must be >= 1")
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
