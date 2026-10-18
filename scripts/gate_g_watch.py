"""Watchdog for the local grounder (Gate G): stop it if the Mac or Lightning suffers.

  python3 -I scripts/gate_g_watch.py --pidfile ~/models/logs/local-grounder-8011.pid --port 8011 \
      --log ~/models/gate_g/watch.jsonl &

Start it BEFORE scripts/run_local_grounder.sh: it measures Lightning's baseline while the
grounder is still off, writes <pidfile>.watch (the launcher refuses to start without a live
watchdog), then waits for the PID file. It stops the grounder (SIGTERM to the vllm process and
its children, SIGKILL after a grace period: 1 s for memory, --grace-s otherwise) when:
- macOS memory pressure leaves normal (kern.memorystatus_vm_pressure_level != 1, i.e. yellow
  or red; checked every second, never blocked by HTTP), or free memory drops below
  --min-free-pct;
- Lightning's warm latency jumps: the rolling median of a tiny fixed request (sent twice back to
  back every --probe-every s, the second one timed, so MLX re-wiring an idle Lightning doesn't
  count) exceeds both --probe-ratio x baseline and baseline + --probe-margin-s;
- Lightning's real decode slows down: mean inter-token latency over the last --window-s, from its
  /metrics counters, is --metrics-ratio x its pre-grounder average;
- Lightning stops answering, the grounder's API server dies leaving children, or the watchdog
  itself hits an unexpected error (fail-safe).
It writes --stop-file first (gate_g.py stops when it sees it), then signals, then exits.

It never signals Lightning: the target must be `vllm serve ... --port <port>`, and the :8000
listener, its descendants and its process group are always excluded. Standard library only.
"""

import argparse
import json
import os
import queue
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

PROBE = {"model": "lightning", "temperature": 0.0, "max_tokens": 4,
         "chat_template_kwargs": {"enable_thinking": False},
         "messages": [{"role": "user", "content": "Reply with the single word OK."}]}
METRICS = ("vllm:inter_token_latency_seconds_sum", "vllm:inter_token_latency_seconds_count",
           "vllm:num_requests_running", "vllm:num_requests_waiting")
PROBE_TIMEOUT_S = 15.0


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def pressure_level() -> int:
    return int(sh("sysctl", "-n", "kern.memorystatus_vm_pressure_level") or 0)


def free_pct() -> int | None:
    m = re.search(r"free percentage: (\d+)%", sh("memory_pressure", "-Q"))
    return int(m.group(1)) if m else None


def children(pid: int) -> list[int]:
    out = []
    for c in sh("pgrep", "-P", str(pid)).split():
        out += [int(c)] + children(int(c))
    return out


def protected_pids(lightning_port: int) -> set[int]:
    """Lightning's API server(s), their descendants, and everything in their process groups."""
    roots = {int(p) for p in sh("lsof", "-nP", f"-iTCP:{lightning_port}", "-sTCP:LISTEN", "-t").split()}
    pids = set(roots)
    for r in roots:
        pids.update(children(r))
    pgids = {sh("ps", "-o", "pgid=", "-p", str(p)) for p in pids} - {""}
    for g in pgids:
        pids.update(int(p) for p in sh("pgrep", "-g", g).split())
    return pids


def cmdline(pid: int | None) -> str:
    return sh("ps", "-ww", "-o", "command=", "-p", str(pid)) if pid else ""


def is_target(cmd: str, port: int) -> bool:
    return "vllm" in cmd and "serve" in cmd and f"--port {port}" in cmd


def scrape(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            text = r.read().decode()
    except Exception:
        return None
    vals = dict.fromkeys(METRICS, 0.0)
    for line in text.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in vals:  # one model on :8000, so summing over label sets is that model's value
            vals[name] += float(line.rsplit(" ", 1)[1])
    return vals


def probe(url: str) -> float | None:
    req = urllib.request.Request(url, data=json.dumps(PROBE).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S) as r:
            json.load(r)["choices"][0]
    except Exception:
        return None
    return time.perf_counter() - t0


def warm_probe(url: str) -> float | None:
    """Time the second of two back-to-back probes: the first re-wires an idle Lightning."""
    if probe(url) is None:
        return None
    return probe(url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pidfile", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8011, help="the grounder's port (the target must be serving it)")
    ap.add_argument("--lightning-port", type=int, default=8000)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--stop-file", type=Path, default=Path("~/models/gate_g/STOPPED"))
    ap.add_argument("--baseline-probes", type=int, default=5)
    ap.add_argument("--probe-every", type=float, default=20.0)
    ap.add_argument("--probe-ratio", type=float, default=2.0)
    ap.add_argument("--probe-margin-s", type=float, default=0.5)
    ap.add_argument("--window-s", type=float, default=60.0)
    ap.add_argument("--metrics-ratio", type=float, default=2.0)
    ap.add_argument("--min-tokens", type=float, default=200, help="decode tokens a window needs before ITL counts")
    ap.add_argument("--min-free-pct", type=int, default=14)
    ap.add_argument("--grace-s", type=float, default=8.0, help="SIGTERM -> SIGKILL for non-memory stops")
    args = ap.parse_args()
    pidfile, logp, stop_file = (p.expanduser() for p in (args.pidfile, args.log, args.stop_file))
    armed = pidfile.with_name(pidfile.name + ".watch")
    logp.parent.mkdir(parents=True, exist_ok=True)
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.unlink(missing_ok=True)
    base = f"http://127.0.0.1:{args.lightning_port}"
    logf = logp.open("a")

    def log(**kv):
        kv = {"t": time.strftime("%H:%M:%S"), **kv}
        logf.write(json.dumps(kv) + "\n")
        logf.flush()
        if kv.get("event"):
            print(json.dumps(kv), flush=True)

    protected = protected_pids(args.lightning_port)
    assert protected, f"nothing listens on :{args.lightning_port}; is Lightning up?"
    log(event="start", protected=sorted(protected))

    if pidfile.exists():
        txt = pidfile.read_text().strip()
        old = int(txt) if txt.isdigit() else None
        if is_target(cmdline(old), args.port):
            sys.exit(f"a grounder (pid {old}) already runs on :{args.port}; stop it before arming a new watchdog")
        pidfile.unlink()
        log(event="stale_pidfile_removed", pid=old)

    probes = [p for p in (warm_probe(f"{base}/v1/chat/completions") for _ in range(args.baseline_probes)) if p is not None]
    m0 = scrape(f"{base}/metrics")
    if len(probes) < max(1, args.baseline_probes - 1) or m0 is None:
        sys.exit("Lightning probes or /metrics failed; not arming")
    base_probe = statistics.median(probes)
    itl0 = m0["vllm:inter_token_latency_seconds_sum"] / max(1.0, m0["vllm:inter_token_latency_seconds_count"])
    log(event="baseline", probe_s=[round(p, 3) for p in probes], probe_median_s=round(base_probe, 3),
        itl_s=round(itl0, 4), pressure=pressure_level(), free_pct=free_pct())

    armed.write_text(f"{os.getpid()}\n")
    print(f"armed ({armed}); waiting for {pidfile} ...", flush=True)
    target = None
    while target is None:
        txt = pidfile.read_text().strip() if pidfile.exists() else ""
        target = int(txt) if txt.isdigit() else None
        time.sleep(0.5)
    deadline = time.time() + 15
    while not is_target(cmdline(target), args.port) and time.time() < deadline:
        time.sleep(0.5)  # the launcher exec's vllm right after writing the file
    cmd = cmdline(target)
    if not is_target(cmd, args.port) or target in protected | protected_pids(args.lightning_port):
        stop_file.write_text(f"{time.strftime('%H:%M:%S')} target {target} not verifiable: {cmd[:200]}\n")
        log(event="target_unverified", pid=target, cmd=cmd[:300])
        armed.unlink(missing_ok=True)
        sys.exit(3)
    log(event="target", pid=target, cmd=cmd[:300])

    known: set[int] = set()  # the target's descendants seen so far (they outlive a crashed parent)

    def ours(c: str) -> bool:
        return "VLLM_GROUNDER" in c or "vllm" in c or "multiprocessing" in c

    def victims_now() -> list[int]:
        now_protected = protected | protected_pids(args.lightning_port)
        alive = is_target(cmdline(target), args.port)
        live = [target] + children(target) if alive else []
        known.update(live[1:])
        # Earlier-seen children must still look like ours (guards against PID reuse).
        return [p for p in dict.fromkeys(live + sorted(known))
                if p not in now_protected and cmdline(p) and (p in live or ours(cmdline(p)))]

    def stop(reason: str, fast: bool, **kv):
        try:
            stop_file.write_text(f"{time.strftime('%H:%M:%S')} {reason} {json.dumps(kv)}\n")
        except OSError:
            pass
        victims = victims_now()
        for p in victims:
            try:
                os.kill(p, signal.SIGTERM)
            except ProcessLookupError:
                pass
        log(event="stop", reason=reason, victims=victims, **kv)
        deadline = time.time() + (1.0 if fast else args.grace_s)
        while time.time() < deadline and any(cmdline(p) for p in victims):
            time.sleep(0.25)
        for _ in range(5):  # rescan: a child spawned mid-shutdown is caught here
            left = [p for p in dict.fromkeys(victims + victims_now()) if cmdline(p)]
            if not left:
                break
            for p in left:
                try:
                    os.kill(p, signal.SIGKILL)
                    log(event="sigkill", pid=p)
                except ProcessLookupError:
                    pass
            time.sleep(0.5)
        left = [p for p in victims_now() if cmdline(p)]
        log(event="stopped", survivors=left, pressure=pressure_level(), free_pct=free_pct())
        armed.unlink(missing_ok=True)
        pidfile.unlink(missing_ok=True)
        sys.exit(1)

    # Probes and scrapes run in a thread, so the pressure check below never waits on HTTP.
    results: queue.Queue = queue.Queue()

    def prober():
        while True:
            results.put((time.time(), warm_probe(f"{base}/v1/chat/completions"), scrape(f"{base}/metrics")))
            time.sleep(args.probe_every)

    threading.Thread(target=prober, daemon=True).start()
    window = deque()  # (time, metrics)
    recent = deque(maxlen=3)
    fails = 0
    last_free = 0.0
    try:
        while True:
            if not is_target(cmdline(target), args.port):
                if victims_now():
                    stop("target_exited_leaving_children", fast=True)
                log(event="target_exited", pid=target, pressure=pressure_level(), free_pct=free_pct())
                armed.unlink(missing_ok=True)
                pidfile.unlink(missing_ok=True)
                return
            victims_now()  # keeps `known` current
            level = pressure_level()
            if level != 1:
                stop("memory_pressure", fast=True, level=level)
            if time.time() - last_free >= 5:
                last_free = time.time()
                fp = free_pct()
                if fp is not None and fp < args.min_free_pct:
                    stop("low_free_memory", fast=True, free_pct=fp, level=level)
            while not results.empty():
                now, p, m = results.get()
                fails = fails + 1 if p is None else 0
                if fails >= 2:
                    stop("lightning_not_answering", fast=False)
                if p is not None:
                    recent.append(p)
                med = statistics.median(recent) if len(recent) == recent.maxlen else None
                itl = None
                if m is not None:
                    window.append((now, m))
                    while len(window) > 1 and now - window[0][0] > args.window_s:
                        window.popleft()
                    old = window[0][1]
                    ntok = m["vllm:inter_token_latency_seconds_count"] - old["vllm:inter_token_latency_seconds_count"]
                    if ntok >= args.min_tokens:
                        itl = (m["vllm:inter_token_latency_seconds_sum"] - old["vllm:inter_token_latency_seconds_sum"]) / ntok
                        if itl > args.metrics_ratio * itl0:
                            stop("lightning_itl_slowdown", fast=False, itl_s=round(itl, 4), baseline_itl_s=round(itl0, 4))
                log(pressure=level, free_pct=free_pct(), probe_s=round(p, 3) if p else None,
                    probe_median_s=round(med, 3) if med else None, itl_s=round(itl, 4) if itl else None,
                    lightning_running=m and m["vllm:num_requests_running"], lightning_waiting=m and m["vllm:num_requests_waiting"])
                if med is not None and med > max(args.probe_ratio * base_probe, base_probe + args.probe_margin_s):
                    stop("lightning_probe_slowdown", fast=False, probe_median_s=round(med, 3), baseline_s=round(base_probe, 3))
            time.sleep(1.0)
    except SystemExit:
        raise
    except BaseException as e:  # fail safe: never leave the grounder unguarded
        stop("watchdog_error", fast=True, error=repr(e)[:200])


if __name__ == "__main__":
    main()
