"""Rewrite raw OS-Atlas labels into click instructions with Nemotron 3.5 Lightning (batched, validated, resumable).

Run from the repo root (the hosted key is read from $NVIDIA_API_KEY or ./.env):
  python data_prep/rewrite/rewrite.py --manifest ~/data/os_atlas_20k/manifest.jsonl --out ~/data/rewrite_v1 --dry-run
  python data_prep/rewrite/rewrite.py --manifest ~/data/os_atlas_20k/manifest.jsonl --out ~/data/rewrite_v1
"""

import argparse
import asyncio
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_prep.rewrite import cache as cache_mod
from data_prep.rewrite import client as client_mod
from data_prep.rewrite import meta, method, prompts, validate

REPO = Path(__file__).resolve().parents[2]
SCRIPT = "data_prep/rewrite/rewrite.py"
DRY_RUN_LIMIT = 200
MAX_ATTEMPTS = 3  # the batch attempt plus 2 single-row retries
FIRST_TRY_VALID_MIN, DROP_RATE_MAX, FAITHFUL_MIN = 0.95, 0.25, 0.90
PUBLIC_FIELDS = ("id", "source", "original", "rewritten", "style", "keep", "drop_reason", "prompt_version", "model",
                 "ts")
RESPONSE_FORMATS = ("json_schema", "json_object", "none")
COMPLETED = ("ok", "partial", "invalid_json")  # request statuses that returned a completion
PROGRESS_EVERY_S = 15.0


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _csv(text: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in text.split(",") if s.strip())


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, required=True, help="canonical manifest.jsonl (sample_os_atlas.py output)")
    ap.add_argument("--out", type=Path, required=True, help="output dir: rewrites.jsonl, stats.json, METHOD.md, cache")
    ap.add_argument("--endpoint", choices=sorted(client_mod.ENDPOINTS), default="hosted")
    ap.add_argument("--base-url", help="override the endpoint's base URL")
    ap.add_argument("--model", help="override the endpoint's model name")
    ap.add_argument("--model-label", help="model id recorded in rewrites.jsonl and METHOD.md (default: the model name)")
    ap.add_argument("--limit", type=int, help="only the first N selected rows, ordered by sha256('sample:' + id)")
    ap.add_argument("--dry-run", action="store_true", help=f"quality-gate run; implies --limit {DRY_RUN_LIMIT}")
    ap.add_argument("--batch-size", type=_positive, default=20)
    ap.add_argument("--concurrency", type=_positive, default=6)
    ap.add_argument("--cache", type=Path, help="sqlite cache keyed by (prompt_version, id) (default OUT/cache.sqlite)")
    ap.add_argument("--response-format", choices=RESPONSE_FORMATS, default="json_object")
    ap.add_argument("--retry-errors", action="store_true", help="re-request rows cached as llm_error")
    ap.add_argument("--show-prompt", action="store_true", help="print the first batch's messages and exit (no network)")
    ap.add_argument("--raw-sources", type=_csv, default=(), help="extra raw-label sources, comma-separated")
    ap.add_argument("--natural-sources", type=_csv, default=(), help="extra natural sources, comma-separated")
    ap.add_argument("--max-consecutive-transport-errors", type=_positive, default=5)
    return ap


def sample_key(row_id: str) -> str:
    return hashlib.sha256(("sample:" + row_id).encode("utf-8")).hexdigest()


def select_targets(rows, limit=None, extra_raw=(), extra_natural=()) -> tuple[list[dict], list[dict]]:
    """(selected, targets): selected raw-label rows, then the first `limit` of them by sample_key, in manifest order."""
    selected = [r for r in rows if meta.is_selected(r, extra_raw, extra_natural)]
    if limit is None or limit >= len(selected):
        return selected, list(selected)
    keep = {r["id"] for r in sorted(selected, key=lambda r: sample_key(r["id"]))[:max(0, limit)]}
    return selected, [r for r in selected if r["id"] in keep]


def is_current(row: dict, rec: dict | None) -> bool:
    """A cached record still matches the manifest row (same label and source)."""
    return rec is not None and rec.get("original") == row["instruction"] and rec.get("source") == row["source"]


def make_record(row, m, norm, *, attempts, errors, warnings, ctx) -> dict:
    keep = bool(norm["keep"] and norm.get("rewritten"))
    return {"id": row["id"], "source": row["source"], "original": row["instruction"],
            "rewritten": norm["rewritten"] if keep else None, "style": m["style"], "keep": keep,
            "drop_reason": None if keep else norm.get("drop_reason"), "prompt_version": ctx["pv"],
            "model": ctx["model_label"], "ts": ctx["now"](), "endpoint": ctx["endpoint"], "attempts": attempts,
            "first_try_valid": attempts == 1,
            "warnings": list(warnings), "errors": list(errors)}


async def process(pending, metas, client, store, ctx, *, batch_size, concurrency, max_transport_errors,
                  state=None) -> dict:
    """Send pending rows in batches; invalid rows are re-asked one at a time. All sqlite writes happen on this loop.

    Counters go into `state` (pass a dict to keep them if the run is cancelled)."""
    queue, stop, seq = asyncio.PriorityQueue(), asyncio.Event(), [0]
    state = {} if state is None else state
    state.update({"abort": None, "abort_code": 0, "consecutive": 0, "transport_errors": 0, "rows_done": 0, "kept": 0,
                  "dropped": 0, "requests": 0, "models": set(), "total": len(pending), "last_print": time.monotonic()})

    def push(priority, job):
        seq[0] += 1
        queue.put_nowait((priority, seq[0], job))

    def abort(reason, code):
        if state["abort"] is None:
            state["abort"], state["abort_code"] = reason, code
        stop.set()

    def log(entry):
        state["requests"] += 1
        store.log_request(ctx["pv"], ctx["run_id"], entry)

    def save(records):
        if records:
            store.put_many(ctx["pv"], records)
        for r in records:
            state["rows_done"] += 1
            state["kept" if r["keep"] else "dropped"] += 1
        if time.monotonic() - state["last_print"] >= PROGRESS_EVERY_S:
            state["last_print"] = time.monotonic()
            print(f"  {state['rows_done']}/{state['total']} rows | kept {state['kept']} dropped {state['dropped']} | "
                  f"{state['requests']} requests, {state['transport_errors']} transport errors", flush=True)

    async def handle(job):
        kind, rows, attempt, prior = job
        ms = [metas[r["id"]] for r in rows]
        entry = {"ts": ctx["now"](), "kind": kind, "n_rows": len(rows), "attempt": attempt}
        t0 = time.monotonic()
        try:
            comp = await client.complete(prompts.build_messages(ms), json_schema=prompts.RESPONSE_SCHEMA)
        except client_mod.TransportError as e:
            state["consecutive"] += 1
            state["transport_errors"] += 1
            log({**entry, "status": "transport_error", "payload_ok": False, "prompt_tokens": 0, "completion_tokens": 0,
                 "latency_s": round(time.monotonic() - t0, 3), "http_attempts": getattr(e, "attempts", None),
                 "finish_reason": None, "error": str(e)[:500]})
            print(f"  transport error ({kind}, {len(rows)} rows stay pending): {str(e)[:300]}", file=sys.stderr,
                  flush=True)
            if state["consecutive"] >= max_transport_errors:
                abort(f"{state['consecutive']} consecutive transport errors (last: {str(e)[:300]})", 2)
            return
        except client_mod.FatalHTTPError as e:
            log({**entry, "status": "fatal_http_error", "payload_ok": False, "prompt_tokens": 0, "completion_tokens": 0,
                 "latency_s": round(time.monotonic() - t0, 3), "http_attempts": None, "finish_reason": None,
                 "error": str(e)[:500]})
            abort(f"fatal HTTP error: {e}", 1)
            return
        state["consecutive"] = 0
        state["models"].add(comp.model)
        if comp.finish_reason == "length":  # cut off: never trust a partial batch
            ok, warns, payload_ok = {}, {}, False
            errors = {k: "truncated: finish_reason=length" for k in range(1, len(rows) + 1)}
        else:
            ok, errors, warns, payload_ok = validate.validate_batch(comp.text, ms)
        status = "invalid_json" if not payload_ok else "partial" if errors else "ok"
        first_error = next(iter(errors.values()), None)
        log({**entry, "status": status, "payload_ok": payload_ok, "prompt_tokens": comp.prompt_tokens,
             "completion_tokens": comp.completion_tokens, "latency_s": comp.latency_s,
             "http_attempts": comp.http_attempts,
             "finish_reason": comp.finish_reason,
             "error": None if status == "ok" else f"{len(errors)}/{len(rows)} invalid; first: {first_error}"[:500],
             **({} if status == "ok" else {"sample_text": comp.text[:1000]})})
        records = []
        for k, row in enumerate(rows, 1):
            past = prior.get(row["id"], [])
            if k in ok:
                records.append(make_record(row, ms[k - 1], ok[k], attempts=attempt, errors=past,
                                           warnings=warns.get(k, []), ctx=ctx))
                continue
            errs = [*past, str(errors.get(k, "missing"))[:300]]
            if attempt < MAX_ATTEMPTS:
                push(0, ("single", [row], attempt + 1, {row["id"]: errs}))
            else:
                norm = {"keep": False, "rewritten": None, "drop_reason": validate.LLM_ERROR}
                records.append(make_record(row, ms[k - 1], norm, attempts=attempt, errors=errs,
                                           warnings=warns.get(k, []), ctx=ctx))
        save(records)

    async def worker():
        while True:
            _, _, job = await queue.get()
            try:
                if state["abort"] is None:
                    await handle(job)
            except Exception as e:  # a bug, not an API failure: stop cleanly and keep what is cached
                traceback.print_exc()
                abort(f"internal error: {type(e).__name__}: {e}", 1)
            finally:
                queue.task_done()

    for i in range(0, len(pending), batch_size):
        push(1, ("batch", pending[i:i + batch_size], 1, {}))
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    waiters = [asyncio.ensure_future(queue.join()), asyncio.ensure_future(stop.wait())]
    try:
        await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in workers + waiters:
            t.cancel()
        await asyncio.gather(*workers, *waiters, return_exceptions=True)
    return state


def _rate(num, den, nd=4):
    return round(num / den, nd) if den else None


def _percentile(values, p):
    values = sorted(values)
    return values[max(0, math.ceil(p / 100 * len(values)) - 1)] if values else None


def _counts(values) -> dict:
    return dict(sorted(Counter(values).items()))


def summarize_requests(reqs: list[dict]) -> dict:
    lat = [r["latency_s"] for r in reqs if r.get("status") in COMPLETED and r.get("latency_s") is not None]
    attempts = [r.get("http_attempts") or 0 for r in reqs]
    pt, ct = sum(r.get("prompt_tokens") or 0 for r in reqs), sum(r.get("completion_tokens") or 0 for r in reqs)
    return {"n": len(reqs), "by_kind": _counts(r.get("kind") for r in reqs),
            "by_status": _counts(r.get("status") for r in reqs), "http_attempts": sum(attempts),
            "retries": sum(max(0, a - 1) for a in attempts), "prompt_tokens": pt, "completion_tokens": ct,
            "total_tokens": pt + ct, "latency_p50_s": _percentile(lat, 50), "latency_p95_s": _percentile(lat, 95)}


def rewrite_counts(targets, records) -> dict:
    by_source, by_style = {}, {}
    for t in targets:
        rec = records.get(t["id"])
        src = by_source.setdefault(t["source"], {"targets": 0, "kept": 0, "dropped": 0})
        sty = by_style.setdefault(meta.assign_style(t["id"]), {"targets": 0, "kept": 0})
        src["targets"] += 1
        sty["targets"] += 1
        if rec is not None:
            src["kept" if rec["keep"] else "dropped"] += 1
            sty["kept"] += bool(rec["keep"])
    for s in by_source.values():
        s["drop_rate"] = _rate(s["dropped"], s["kept"] + s["dropped"])
    recs = list(records.values())
    return {"total": len(recs), "kept": sum(r["keep"] for r in recs), "dropped": sum(not r["keep"] for r in recs),
            "by_source": dict(sorted(by_source.items())),
            "by_style": {s: by_style[s] for s, _ in meta.STYLE_SPLIT if s in by_style},
            "drop_reasons": _counts(r.get("drop_reason") for r in recs if not r["keep"]),
            "warnings": _counts(w.split(":")[0] for r in recs for w in r.get("warnings") or ())}


def quality(records, req_all) -> dict:
    recs = list(records.values())
    batches = [r for r in req_all if r.get("kind") == "batch" and r.get("status") in COMPLETED]
    model_drops = sum(not r["keep"] and r.get("drop_reason") != validate.LLM_ERROR for r in recs)
    return {"first_try_valid_rate": _rate(sum(bool(r.get("first_try_valid")) for r in recs), len(recs)),
            "batch_json_ok_rate": _rate(sum(bool(r.get("payload_ok")) for r in batches), len(batches)),
            "rows_retried": sum((r.get("attempts") or 1) > 1 for r in recs),
            "llm_error_rows": sum(r.get("drop_reason") == validate.LLM_ERROR for r in recs),
            "drop_rate_raw": _rate(model_drops, len(recs)),
            "validation_errors": _counts(str(e).split(":")[0] for r in recs for e in r.get("errors") or ())}


def git_info() -> tuple[str | None, bool | None]:
    try:
        run = lambda *a: subprocess.run(["git", "--no-optional-locks", "-C", str(REPO), *a], capture_output=True,
                                        text=True, check=True, timeout=10).stdout.strip()
        return run("rev-parse", "HEAD"), bool(run("status", "--porcelain", "--", "data_prep/rewrite"))
    except (OSError, subprocess.SubprocessError):
        return None, None


def file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build_stats(*, ctx, args, rows, selected, targets, records, req_this, req_all, client_stats, state, started,
                finished, wall_s, limit) -> dict:
    sha, dirty = git_info()
    extra_raw, extra_natural = set(args.raw_sources or ()), set(args.natural_sources or ())
    this = summarize_requests(req_this)
    this["downgrades"] = list(client_stats.get("downgrades") or [])
    this["rate_limit_headers"] = dict(sorted((client_stats.get("rate_limit") or {}).items()))
    this["transport_errors"] = state.get("transport_errors", 0)
    q = quality(records, req_all)
    pending = sum(t["id"] not in records for t in targets)
    minutes = wall_s / 60
    exit_code = state.get("abort_code") or (2 if pending else 0)
    return {
        "run": {"run_id": ctx["run_id"], "started": started, "finished": finished, "wall_s": round(wall_s, 3),
                "command": ctx["command"], "git_sha": sha, "git_dirty": dirty, "endpoint": ctx["endpoint"],
                "base_url": ctx["base_url"], "model": ctx["model"], "model_label": ctx["model_label"],
                "models_reported": sorted(state.get("models") or ()), "prompt_version": ctx["pv"],
                "sampling": client_mod.SAMPLING, "batch_size": args.batch_size, "concurrency": args.concurrency,
                "response_format_requested": args.response_format,
                "response_format_final": ctx.get("response_format_final") or args.response_format,
                "dry_run": bool(args.dry_run), "limit": limit,
                "raw_sources": sorted(meta.RAW_LABEL_SOURCES - extra_natural | extra_raw),
                "natural_sources": sorted(meta.NATURAL_SOURCES - extra_raw | extra_natural),
                "select_pct": meta.SELECT_PCT, "style_split": [list(s) for s in meta.STYLE_SPLIT]},
        "manifest": {"path": str(Path(args.manifest).resolve()), "sha256": file_sha256(args.manifest), "rows": len(rows),
                     "rows_by_source": _counts(r["source"] for r in rows),
                     "selected_by_source": _counts(r["source"] for r in selected), "targets": len(targets)},
        "rewrites": rewrite_counts(targets, records),
        "quality": q,
        "requests": {"this_run": this, "all_runs": summarize_requests(req_all)},
        "throughput": {"rows_per_min": round(state.get("rows_done", 0) / minutes, 2) if minutes else None,
                       "requests_per_min": round(this["n"] / minutes, 2) if minutes else None,
                       "completion_tokens_per_s": round(this["completion_tokens"] / wall_s, 2) if wall_s else None},
        "pending_rows": pending,
        "gate": {"first_try_valid_ok": (q["first_try_valid_rate"] or 0) >= FIRST_TRY_VALID_MIN,
                 "drop_rate_ok": q["drop_rate_raw"] is not None and q["drop_rate_raw"] < DROP_RATE_MAX,
                 "first_try_valid_min": FIRST_TRY_VALID_MIN, "drop_rate_max": DROP_RATE_MAX,
                 "faithful_min": FAITHFUL_MIN,
                 "faithfulness": "manual: review_sample.py --n 50"},
        "aborted": state.get("abort"),
        "exit_code": exit_code,
    }


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_outputs(out: Path, targets, records, stats) -> None:
    """rewrites.jsonl is a pure function of the cache (manifest order, fixed keys), so reruns are byte-identical."""
    lines = [json.dumps({k: records[t["id"]].get(k) for k in PUBLIC_FIELDS}, ensure_ascii=False) + "\n"
             for t in targets if t["id"] in records]
    _write_text(out / "rewrites.jsonl", "".join(lines))
    _write_text(out / "stats.json", json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    method.write_method(out, stats)


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def print_summary(stats: dict, out: Path) -> None:
    run, man, rw, q, g = stats["run"], stats["manifest"], stats["rewrites"], stats["quality"], stats["gate"]
    this = stats["requests"]["this_run"]
    limit = f", limit {run['limit']}" if run["limit"] is not None else ""
    print(f"rewrite {run['prompt_version']} | {run['model_label']} via {run['endpoint']} ({run['base_url']})")
    n_sel = sum(man["selected_by_source"].values())
    print(f"targets {man['targets']} (selected {n_sel} of {man['rows']} rows{limit}) | "
          f"done {rw['total']} | kept {rw['kept']} | dropped {rw['dropped']} | pending {stats['pending_rows']}")
    if rw["drop_reasons"]:
        print("drop reasons: " + ", ".join(f"{k} {v}" for k, v in rw["drop_reasons"].items()))
    for src, s in rw["by_source"].items():
        print(f"  {src}: {s['targets']} targets, kept {s['kept']}, dropped {s['dropped']} ({_pct(s['drop_rate'])})")
    print(f"first-try valid {_pct(q['first_try_valid_rate'])} (gate >= {FIRST_TRY_VALID_MIN:.0%}: "
          f"{'PASS' if g['first_try_valid_ok'] else 'FAIL'}) | drop rate {_pct(q['drop_rate_raw'])} (gate < "
          f"{DROP_RATE_MAX:.0%}: {'PASS' if g['drop_rate_ok'] else 'FAIL'}) | "
          f"batch JSON ok {_pct(q['batch_json_ok_rate'])}")
    tp = stats["throughput"]
    print(f"this run: {this['n']} requests {this['by_status']}, {this['http_attempts']} HTTP attempts, "
          f"{this['prompt_tokens']} prompt + {this['completion_tokens']} completion tokens, wall {run['wall_s']}s, "
          f"{tp['rows_per_min']} rows/min")
    if this["rate_limit_headers"]:
        print(f"rate-limit headers: {this['rate_limit_headers']}")
    if stats["aborted"]:
        print(f"ABORTED: {stats['aborted']}", file=sys.stderr)
    if stats["pending_rows"]:
        print(f"{stats['pending_rows']} rows pending: rerun the same command to resume (cached rows are not re-sent)")
    print(f"wrote {out / 'rewrites.jsonl'}, {out / 'stats.json'}, {out / 'METHOD.md'}")
    print(f"next: python data_prep/rewrite/review_sample.py --out {shlex.quote(str(out))}")


def _fail(msg: str, code: int = 1) -> dict:
    print(f"error: {msg}", file=sys.stderr)
    return {"exit_code": code, "error": msg}


async def run(args, *, transport=None, sleep=asyncio.sleep, now=None) -> dict:
    """Rewrite pending targets, then write rewrites.jsonl, stats.json and METHOD.md. Returns stats (with exit_code)."""
    now = now or utc_now
    started, t0 = now(), time.monotonic()
    extra_raw, extra_natural = tuple(args.raw_sources or ()), tuple(args.natural_sources or ())
    try:
        rows = meta.load_manifest(args.manifest, extra_raw, extra_natural)
    except (OSError, ValueError) as e:  # ManifestError and ContaminationError are ValueErrors
        return _fail(f"manifest rejected, nothing was sent: {e}")
    limit = args.limit if args.limit is not None else (DRY_RUN_LIMIT if args.dry_run else None)
    selected, targets = select_targets(rows, limit, extra_raw, extra_natural)
    by_image = meta.index_by_image(rows)
    metas = {r["id"]: meta.row_meta(r, by_image) for r in targets}

    if args.show_prompt:
        first = [metas[r["id"]] for r in targets[:args.batch_size]]
        if not first:
            return _fail("no rows selected for rewriting; nothing to show")
        print(json.dumps(prompts.build_messages(first), indent=2, ensure_ascii=False))
        return {"exit_code": 0, "show_prompt": True}

    endpoint = client_mod.ENDPOINTS[args.endpoint]
    model = args.model or endpoint.model
    argv = getattr(args, "argv", None)
    ctx = {"pv": prompts.PROMPT_VERSION, "run_id": f"{started.replace('-', '').replace(':', '')}-{uuid.uuid4().hex[:6]}",
           "model": model, "model_label": args.model_label or model, "endpoint": endpoint.name,
           "base_url": (args.base_url or endpoint.base_url).rstrip("/"), "now": now,
           "command": shlex.join(["python", SCRIPT, *(sys.argv[1:] if argv is None else argv)])}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state, client_stats = {}, {}
    with cache_mod.RewriteCache(args.cache or out / "cache.sqlite") as store:
        cached = store.get_many(ctx["pv"], [r["id"] for r in targets])
        pending = [r for r in targets if not is_current(r, cached.get(r["id"]))
                   or (args.retry_errors and cached[r["id"]].get("drop_reason") == validate.LLM_ERROR)]
        if pending:
            api_key = client_mod.load_api_key(endpoint.api_key_env) if endpoint.api_key_env else None
            if endpoint.api_key_env and not api_key:
                return _fail(f"{endpoint.api_key_env} is not set (environment or ./.env; run from the repo root)")
            print(f"{len(pending)} of {len(targets)} targets pending; {ctx['endpoint']} {ctx['base_url']} "
                  f"model {model}, "
                  f"batch {args.batch_size}, concurrency {args.concurrency}", flush=True)
            client = client_mod.LightningClient(endpoint, api_key=api_key, base_url=args.base_url, model=args.model,
                                                concurrency=args.concurrency, response_format=args.response_format,
                                                transport=transport, sleep=sleep)
            try:
                await process(pending, metas, client, store, ctx, batch_size=args.batch_size,
                              concurrency=args.concurrency, max_transport_errors=args.max_consecutive_transport_errors,
                              state=state)
            except asyncio.CancelledError:  # Ctrl-C: cached rows are safe; write outputs for them
                state.update(abort="interrupted", abort_code=130)
            finally:
                ctx["response_format_final"] = client.response_format
                client_stats = client.stats
                await client.aclose()
        current = store.get_many(ctx["pv"], [r["id"] for r in targets])
        records = {r["id"]: current[r["id"]] for r in targets if is_current(r, current.get(r["id"]))}
        req_this, req_all = store.requests(ctx["pv"], ctx["run_id"]), store.requests(ctx["pv"])
    stats = build_stats(ctx=ctx, args=args, rows=rows, selected=selected, targets=targets, records=records,
                        req_this=req_this, req_all=req_all, client_stats=client_stats, state=state, started=started,
                        finished=now(), wall_s=time.monotonic() - t0, limit=limit)
    write_outputs(out, targets, records, stats)
    print_summary(stats, out)
    return stats


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(argv)
    args.argv = argv
    return asyncio.run(run(args))["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
