"""Gate G (docs/local-mac.md): send the same dev rows to the local and the H200 grounder.

  # 1) The same requests, row by row: the local vllm-metal server first, then the H200
  #    `grounder` (base + sft-template LoRA, one vLLM process) through the tunnel on 8001.
  ~/.venv-merge/bin/python scripts/gate_g.py run \
      --manifest <os_atlas_final>/manifest.jsonl --out ~/models/gate_g/sft-template \
      --endpoint local=http://127.0.0.1:8011/v1 --endpoint h200=http://127.0.0.1:8001/v1
  # 2) The verdict (exit code 0 = pass)
  ~/.venv-merge/bin/python scripts/gate_g.py compare ~/models/gate_g/sft-template --a local --b h200

Rows: 200 of the OS-Atlas `dev` split (checkpoint-selection split, never SSv2), sampled with
random.Random(0) like data_prep/gym_rows.py --limit. Each screenshot goes through
common/grounding.py's preprocess_screenshot and is sent as a PNG data URL with the exact
qwen3_5 grounding prompt, so both servers get byte-identical requests. Sampling is greedy
(temperature 0, max_tokens 128, as in the Gym rows), and each server's own default chat-template
kwargs apply (enable_thinking false on both), so a template mismatch shows up as a prompt-token
mismatch. Scoring matches resources_servers/gui_grounding/app.py: strip think blocks,
parse_point, inclusive point-in-bbox on the original image's pixels.

Pass if usage.prompt_tokens is identical on every row, hit rates are within 2 points, and at
least 90% of hit/miss decisions agree. Results are appended per row (<out>/<name>.jsonl), so a
stopped run resumes where it left off. If the watchdog (scripts/gate_g_watch.py) writes its
STOPPED file, the run stops.
"""

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image

# Scripts run as `python scripts/gate_g.py` (no root pyproject, AGENTS.md): put the repo root
# on sys.path so the one grounding interface is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grounding import (  # noqa: E402
    MAX_PIXELS, MIN_PIXELS, build_grounding_messages, parse_point, preprocess_screenshot, to_data_url, to_pixels,
)

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S | re.I)
FAMILY = "qwen3_5"
GATE = {"max_hit_rate_gap_points": 2.0, "min_decision_agreement": 0.90}


def inside(pt_px: tuple[float, float], bbox: list[float]) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= pt_px[0] <= x2 and y1 <= pt_px[1] <= y2  # inclusive, as in gui_grounding


def select_rows(manifest: Path, n: int, seed: int) -> list[dict]:
    dev = [r for r in map(json.loads, manifest.open()) if r.get("split") == "dev"]
    assert len(dev) >= n, f"only {len(dev)} dev rows in {manifest}"
    return random.Random(seed).sample(dev, n)


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def ask(base_url: str, model: str, messages: list[dict], max_tokens: int, timeout: float) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
    t0 = time.perf_counter()
    data = post(f"{base_url}/chat/completions", body, timeout)
    latency = time.perf_counter() - t0
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return {
        "latency_s": round(latency, 4), "raw_output": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
    }


def lightning_busy(metrics_url: str) -> bool:
    """True while Lightning (:8000) has requests running or waiting; unknown counts as idle."""
    try:
        with urllib.request.urlopen(metrics_url, timeout=5) as r:
            text = r.read().decode()
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False
    busy = 0.0
    for line in text.splitlines():
        if line.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting")):
            busy += float(line.rsplit(" ", 1)[1])
    return busy > 0


def wait_for_idle(metrics_url: str, max_s: float) -> float:
    """Hold the next local request while Lightning is busy (it shares the Metal GPU), up to max_s."""
    t0 = time.perf_counter()
    while lightning_busy(metrics_url) and time.perf_counter() - t0 < max_s:
        time.sleep(0.5)
    return time.perf_counter() - t0


def build_request(row: dict, root: Path) -> tuple[list[dict], dict]:
    with Image.open(root / row["image_path"]) as im:
        img = preprocess_screenshot(im)
    url = to_data_url(img)
    w, h = img.size
    info = {"pre_w": w, "pre_h": h, "image_tokens": (w // 32) * (h // 32),
            "png_sha256": hashlib.sha256(url.encode()).hexdigest()[:16]}
    return build_grounding_messages(url, row["instruction"], FAMILY), info


def score(reply: str, row: dict) -> dict:
    pt = parse_point(_THINK_RE.sub("", reply))
    hit = pt is not None and inside(to_pixels(pt, row["img_w"], row["img_h"]), row["bbox"])
    return {"parsed": pt is not None, "pred_x": pt[0] if pt else None, "pred_y": pt[1] if pt else None, "hit": hit}


def cmd_run(args) -> int:
    assert (MAX_PIXELS, MIN_PIXELS) == (2_073_600, 65_536), "the pixel cap is frozen at D1"
    endpoints = []
    for spec in args.endpoint:
        name, url = spec.split("=", 1)
        endpoints.append((name, url.rstrip("/")))
    args.out.mkdir(parents=True, exist_ok=True)
    rows = select_rows(args.manifest, args.n, args.seed)
    root = args.manifest.parent
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    sel = {"manifest": str(args.manifest), "manifest_sha256": manifest_sha, "n": args.n, "seed": args.seed,
           "ids": [r["id"] for r in rows], "model": args.model,
           "max_tokens": args.max_tokens, "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS}
    sel_path = args.out / "selection.json"
    old = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    if old:
        assert old["ids"] == sel["ids"] and old["manifest_sha256"] == manifest_sha, "selection changed; use a new --out"
    # What each endpoint served (served names, roots, LoRA paths) at the time of its run.
    sel["endpoints"] = old.get("endpoints", {})
    for name, url in endpoints:
        with urllib.request.urlopen(f"{url}/models", timeout=30) as resp:
            served = [{k: m.get(k) for k in ("id", "root", "parent")} for m in json.load(resp)["data"]]
        assert any(m["id"] == args.model for m in served), f"{name} doesn't serve {args.model!r}: {served}"
        sel["endpoints"][name] = {"url": url, "served": served, "checked_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    sel_path.write_text(json.dumps(sel, indent=2) + "\n")

    done = {}
    for name, _ in endpoints:
        p = args.out / f"{name}.jsonl"
        # Rows that errored are retried; the newest record per id wins in compare.
        done[name] = {r["id"] for r in map(json.loads, p.open()) if "error" not in r} if p.exists() else set()
    files = {name: (args.out / f"{name}.jsonl").open("a") for name, _ in endpoints}
    stop_file = args.stop_file.expanduser() if args.stop_file else None

    if args.warmup:
        # Warm-up rows: dev rows outside the selection (the first local request compiles kernels).
        ids = set(sel["ids"])
        spare = [r for r in map(json.loads, args.manifest.open()) if r.get("split") == "dev" and r["id"] not in ids]
        for row in random.Random(args.seed + 1).sample(spare, args.warmup):
            msgs, _ = build_request(row, root)
            for name, url in endpoints:
                r = ask(url, args.model, msgs, args.max_tokens, args.timeout)
                print(f"warmup {name}: {r['latency_s']:.2f}s prompt_tokens={r['prompt_tokens']}", flush=True)

    t_start = time.time()
    for i, row in enumerate(rows):
        if stop_file and stop_file.exists():
            print(f"stop file {stop_file} exists: {stop_file.read_text().strip()}", flush=True)
            return 2
        todo = [(n, u) for n, u in endpoints if row["id"] not in done[n]]
        if not todo:
            continue
        msgs, info = build_request(row, root)
        for name, url in todo:
            waited = wait_for_idle(args.yield_to, args.yield_max_s) if args.yield_to and name in args.yield_endpoints else 0.0
            if waited > 1:
                print(f"{name}: waited {waited:.1f}s for Lightning to go idle", flush=True)
            rec = {"id": row["id"], "platform": row["platform"], "data_type": row.get("data_type"),
                   "source": row["source"], "instruction": row["instruction"], "bbox": row["bbox"],
                   "img_w": row["img_w"], "img_h": row["img_h"], "yield_wait_s": round(waited, 2), **info}
            for attempt in range(3):
                try:
                    rec.update(ask(url, args.model, msgs, args.max_tokens, args.timeout))
                    rec.update(score(rec["raw_output"], row))
                    rec.pop("error", None)
                    break
                except (urllib.error.URLError, ConnectionError, TimeoutError, KeyError, ValueError) as e:
                    rec["error"] = repr(e)
                    if isinstance(getattr(e, "reason", None), ConnectionRefusedError) or (stop_file and stop_file.exists()):
                        print(f"{name} at {url} refused the connection (server down?): stopping", flush=True)
                        return 2
                    time.sleep(2 * (attempt + 1))
            files[name].write(json.dumps(rec) + "\n")
            files[name].flush()
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(rows)} rows, {time.time() - t_start:.0f}s", flush=True)
    return 0


def pct(xs: list[float], q: float) -> float | None:
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts (as scripts/compare_grounding.py)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def cmd_compare(args) -> int:
    sel = json.loads((args.dir / "selection.json").read_text())
    runs = {}
    for name in (args.a, args.b):
        recs = {}
        for line in (args.dir / f"{name}.jsonl").open():
            r = json.loads(line)
            recs[r["id"]] = r  # a resumed row overwrites an earlier attempt
        runs[name] = recs
    a, b = runs[args.a], runs[args.b]
    ids = sel["ids"]
    # Both legs must be different servers serving different weights (guards against pointing
    # both at the Mac, e.g. with the local grounder on 8001 in demo mode).
    eps = sel.get("endpoints", {})
    roots = {n: sorted(m["root"] for m in eps.get(n, {}).get("served", []) if m["id"] == sel["model"]) for n in runs}
    distinct = (args.a in eps and args.b in eps and eps[args.a]["url"] != eps[args.b]["url"]
                and roots[args.a] != roots[args.b])
    missing = {n: [i for i in ids if i not in runs[n] or "error" in runs[n][i]] for n in runs}
    paired = [i for i in ids if i in a and i in b and "error" not in a[i] and "error" not in b[i]]
    n = len(paired)

    tok_mismatch = [i for i in paired if a[i]["prompt_tokens"] != b[i]["prompt_tokens"]]
    hit_a = sum(a[i]["hit"] for i in paired)
    hit_b = sum(b[i]["hit"] for i in paired)
    agree = sum(a[i]["hit"] == b[i]["hit"] for i in paired)
    only_a = sum(a[i]["hit"] and not b[i]["hit"] for i in paired)
    only_b = sum(b[i]["hit"] and not a[i]["hit"] for i in paired)
    same_text = sum(a[i]["raw_output"].strip() == b[i]["raw_output"].strip() for i in paired)
    dists = [math.dist((a[i]["pred_x"], a[i]["pred_y"]), (b[i]["pred_x"], b[i]["pred_y"]))
             for i in paired if a[i]["parsed"] and b[i]["parsed"]]
    per_platform = defaultdict(lambda: [0, 0, 0])
    for i in paired:
        p = per_platform[a[i]["platform"]]
        p[0] += a[i]["hit"]
        p[1] += b[i]["hit"]
        p[2] += 1

    rate_a = 100 * hit_a / n if n else 0.0
    rate_b = 100 * hit_b / n if n else 0.0
    checks = {
        "distinct_servers": distinct,
        "all_rows_answered": n == len(ids),
        "prompt_tokens_identical": n > 0 and not tok_mismatch,
        "hit_rate_within_2_points": n > 0 and abs(rate_a - rate_b) <= GATE["max_hit_rate_gap_points"],
        "decision_agreement_ge_90pct": n > 0 and agree / n >= GATE["min_decision_agreement"],
    }
    lat = {name: [runs[name][i]["latency_s"] for i in paired] for name in runs}
    summary = {
        "served_roots": roots, "rows": len(ids), "paired": n, "missing_or_error": {k: len(v) for k, v in missing.items()},
        "prompt_tokens_mismatches": len(tok_mismatch), "prompt_tokens_mismatch_ids": tok_mismatch[:10],
        "prompt_tokens_matches_image_tokens_plus_text": None,
        f"hit_rate_{args.a}": round(rate_a, 2), f"hit_rate_{args.b}": round(rate_b, 2),
        "hit_rate_gap_points": round(rate_a - rate_b, 2),
        "decision_agreement": round(agree / n, 4) if n else None,
        f"hit_only_{args.a}": only_a, f"hit_only_{args.b}": only_b, "mcnemar_p": round(mcnemar_p(only_a, only_b), 4),
        "identical_text": same_text, "parsed": {k: sum(runs[k][i]["parsed"] for i in paired) for k in runs},
        "point_distance_0_1000": {"median": pct(dists, 0.5), "p90": pct(dists, 0.9), "max": max(dists) if dists else None,
                                  "zero": sum(d == 0 for d in dists)},
        "per_platform_hit_rate": {p: {args.a: round(100 * v[0] / v[2], 1), args.b: round(100 * v[1] / v[2], 1), "n": v[2]}
                                  for p, v in sorted(per_platform.items())},
        "latency_s": {k: {"p50": round(pct(v, 0.5), 3), "p90": round(pct(v, 0.9), 3), "mean": round(statistics.mean(v), 3)}
                      for k, v in lat.items() if v},
        "checks": checks, "pass": all(checks.values()),
    }
    # Text tokens are the same for both; the prompt-token count is image tokens + a constant-ish text part.
    extra = {a[i]["prompt_tokens"] - a[i]["image_tokens"] for i in paired if a[i].get("prompt_tokens") is not None}
    summary["prompt_tokens_matches_image_tokens_plus_text"] = sorted(extra)[:5]
    (args.dir / f"summary_{args.a}_vs_{args.b}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("GATE G:", "PASS" if summary["pass"] else "FAIL")
    return 0 if summary["pass"] else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--manifest", type=Path, required=True, help="finalized OS-Atlas manifest (has the dev split)")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--endpoint", action="append", required=True, help="name=base_url (requests go in this order)")
    r.add_argument("--model", default="grounder")
    r.add_argument("--n", type=int, default=200)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--max-tokens", type=int, default=128)
    r.add_argument("--timeout", type=float, default=300.0)
    r.add_argument("--warmup", type=int, default=2, help="unscored warm-up requests on dev rows outside the selection")
    r.add_argument("--stop-file", type=Path, default=Path("~/models/gate_g/STOPPED"))
    r.add_argument("--yield-to", default="http://127.0.0.1:8000/metrics",
                   help="Lightning's /metrics: local requests wait while it's busy ('' to disable)")
    r.add_argument("--yield-endpoints", nargs="*", default=["local"], help="endpoints that share the GPU with Lightning")
    r.add_argument("--yield-max-s", type=float, default=120.0)
    c = sub.add_parser("compare")
    c.add_argument("dir", type=Path)
    c.add_argument("--a", default="local")
    c.add_argument("--b", default="h200")
    args = ap.parse_args()
    if args.cmd == "run":
        args.manifest = args.manifest.expanduser()
        args.out = args.out.expanduser()
        sys.exit(cmd_run(args))
    args.dir = args.dir.expanduser()
    sys.exit(cmd_compare(args))


if __name__ == "__main__":
    main()
