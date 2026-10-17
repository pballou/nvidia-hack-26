"""Rewrite raw OS-Atlas labels into click instructions with Nemotron 3.5 Lightning (batched, validated, resumable).

Numeric labels (validate.is_numeric_label) are dropped by rule before batching and never sent.

Run from the repo root (the hosted key is read from $NVIDIA_API_KEY or ./.env):
  python data_prep/rewrite/rewrite.py --manifest ~/data/os_atlas_20k/manifest.jsonl --out ~/data/rewrite_v3 --dry-run
  python data_prep/rewrite/rewrite.py --manifest ~/data/os_atlas_20k/manifest.jsonl --out ~/data/rewrite_v3
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
from urllib.parse import urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_prep.rewrite import cache as cache_mod
from data_prep.rewrite import client as client_mod
from data_prep.rewrite import meta, method, prompts, validate

REPO = Path(__file__).resolve().parents[2]
SCRIPT = "data_prep/rewrite/rewrite.py"
DRY_RUN_LIMIT = 200
MAX_ATTEMPTS = 3  # retry budget per invocation: the batch attempt plus 2 single-row retries
MAX_EMPTY_REPLIES = 5  # output breaker: consecutive replies with no valid item before the run stops
RULE_MODEL = validate.RULE_MODEL  # `model` of rows dropped by validate.is_numeric_label before any request
FIRST_TRY_VALID_MIN, DROP_RATE_MAX, FAITHFUL_MIN = 0.95, 0.25, 0.90
PUBLIC_FIELDS = ("id", "source", "original", "rewritten", "style", "keep", "drop_reason", "prompt_version", "model",
                 "ts")
RESPONSE_FORMATS = ("json_schema", "json_object", "none")
COMPLETED = ("ok", "partial", "invalid_json")  # request statuses that returned a completion
PROGRESS_EVERY_S = 15.0
HOSTED_LABEL = "NVIDIA-hosted API"  # default model label on build.nvidia.com (it reports no checkpoint root)


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
    ap.add_argument("--model-label", help="model id recorded in rewrites.jsonl and METHOD.md (default: the served "
                    "checkpoint from GET /models, else the model name)")
    ap.add_argument("--note", default="", help="free-text provenance note recorded in stats.json and METHOD.md")
    ap.add_argument("--splits", type=_csv, default=(),
                    help="only rewrite rows whose split is listed, e.g. train (dev keeps raw labels); default: all")
    ap.add_argument("--limit", type=int, help="only the first N selected rows, ordered by sha256('sample:' + id)")
    ap.add_argument("--dry-run", action="store_true", help=f"quality-gate run; implies --limit {DRY_RUN_LIMIT}")
    ap.add_argument("--batch-size", type=_positive, default=20)
    ap.add_argument("--concurrency", type=_positive, default=None,
                    help="concurrent requests (default per endpoint: " +
                         ", ".join(f"{k} {v}" for k, v in client_mod.DEFAULT_CONCURRENCY.items()) + ")")
    ap.add_argument("--cache", type=Path, help="sqlite cache keyed by (prompt_version, id) (default OUT/cache.sqlite)")
    ap.add_argument("--response-format", choices=RESPONSE_FORMATS, default="json_object")
    ap.add_argument("--retry-errors", action="store_true", help="re-request rows cached as llm_error")
    ap.add_argument("--show-prompt", action="store_true", help="print the first batch's messages and exit (no network)")
    ap.add_argument("--raw-sources", type=_csv, default=(), help="extra raw-label sources, comma-separated")
    ap.add_argument("--natural-sources", type=_csv, default=(), help="extra natural sources, comma-separated")
    ap.add_argument("--max-consecutive-transport-errors", type=_positive, default=5)
    ap.add_argument("--max-consecutive-empty-replies", type=_positive, default=MAX_EMPTY_REPLIES,
                    help="stop (exit 2, rows stay pending) after this many replies in a row with no valid item")
    ap.add_argument("--allow-unofficial-model", action="store_true",
                    help="PLUMBING ONLY: run against a local checkpoint that is not the official NVIDIA one; "
                         "outputs are marked NOT FOR PUBLICATION")
    return ap


def sample_key(row_id: str) -> str:
    return hashlib.sha256(("sample:" + row_id).encode("utf-8")).hexdigest()


def select_targets(rows, limit=None, extra_raw=(), extra_natural=(), splits=()) -> tuple[list[dict], list[dict]]:
    """(selected, targets): selected raw-label rows (in `splits` if given), then the first `limit` of them by
    sample_key, in manifest order. Neighbors still come from every row of the screenshot."""
    selected = [r for r in rows if meta.is_selected(r, extra_raw, extra_natural)
                and (not splits or r.get("split") in splits)]
    if limit is None or limit >= len(selected):
        return selected, list(selected)
    keep = {r["id"] for r in sorted(selected, key=lambda r: sample_key(r["id"]))[:max(0, limit)]}
    return selected, [r for r in selected if r["id"] in keep]


def is_current(row: dict, rec: dict | None, allow_unofficial: bool = False) -> bool:
    """A cached record still matches the manifest row (same label and source) and, unless allow_unofficial, was not
    generated by an unofficial checkpoint (so plumbing rows never leak into a publishable run)."""
    return (rec is not None and rec.get("original") == row["instruction"] and rec.get("source") == row["source"]
            and (allow_unofficial or not rec.get("unofficial_model")))


def is_nvidia_hosted(base_url: str) -> bool:
    """NVIDIA's own hosted API, official by definition (the bounty rule allows it)."""
    host = (urlparse(base_url).hostname or "").lower()
    return host == "api.nvidia.com" or host.endswith(".api.nvidia.com")


def make_record(row, m, norm, *, attempts, errors, warnings, ctx) -> dict:
    """`attempts` counts every LLM attempt for the row across invocations and `errors` all of their errors, so a row
    that failed before (an earlier invocation, --retry-errors) is never first_try_valid."""
    keep = bool(norm["keep"] and norm.get("rewritten"))
    return {"id": row["id"], "source": row["source"], "original": row["instruction"],
            "rewritten": norm["rewritten"] if keep else None, "style": m["style"], "keep": keep,
            "drop_reason": None if keep else norm.get("drop_reason"), "prompt_version": ctx["pv"],
            "model": ctx["model_label"], "ts": ctx["now"](), "endpoint": ctx["endpoint"], "attempts": attempts,
            "first_try_valid": attempts == 1, "warnings": list(warnings), "errors": list(errors),
            "served_model_root": ctx.get("served_model_root"), "unofficial_model": bool(ctx.get("unofficial"))}


def rule_record(row, m, ctx) -> dict:
    """A numeric label (validate.is_numeric_label) is dropped by rule before batching: never sent, no model."""
    return {"id": row["id"], "source": row["source"], "original": row["instruction"], "rewritten": None,
            "style": m["style"], "keep": False, "drop_reason": "numeric", "prompt_version": ctx["pv"],
            "model": RULE_MODEL, "ts": ctx["now"](), "endpoint": None, "attempts": 0, "first_try_valid": None,
            "warnings": [], "errors": [], "served_model_root": None, "unofficial_model": False}


def is_rule(rec: dict | None) -> bool:
    return bool(rec) and rec.get("model") == RULE_MODEL


def history_record(row, attempts, errors, ctx) -> dict:
    return {"id": row["id"], "original": row["instruction"], "source": row["source"],
            "unofficial_model": bool(ctx.get("unofficial")), "attempts": attempts, "errors": list(errors)}


def prior_attempts(row, history, cached, allow_unofficial=False) -> tuple[int, list[str]]:
    """(attempts, errors) of earlier LLM attempts for this row and label: its attempt history (written on every
    failed attempt, so it survives aborts mid-retry) or a cached llm_error record being re-sent, whichever has more."""
    found = [h for h in (history, cached) if is_current(row, h, allow_unofficial) and not is_rule(h)]
    best = max(found, key=lambda h: h.get("attempts") or 0, default=None)
    return (int(best.get("attempts") or 0), list(best.get("errors") or [])) if best else (0, [])


async def process(pending, metas, client, store, ctx, *, batch_size, concurrency, max_transport_errors,
                  max_empty_replies=MAX_EMPTY_REPLIES, prior=None, state=None) -> dict:
    """Send pending rows in batches; invalid rows are re-asked one at a time. All sqlite writes happen on this loop.

    prior: {id: (attempts, errors)} from earlier invocations (see prior_attempts); MAX_ATTEMPTS is a per-invocation
    retry budget, `attempts` in the records is the total. Output breaker: a reply with no valid item extends a streak
    (a single-row retry that parsed but whose one item failed only extends a running streak: stubborn rows are
    expected among retries); any valid item resets it. While a streak runs, rows that use up their retries are held
    back instead of becoming llm_error; at max_empty_replies the run stops (exit 2) and those rows stay pending.

    Counters go into `state` (pass a dict to keep them if the run is cancelled)."""
    queue, stop, seq, prior = asyncio.PriorityQueue(), asyncio.Event(), [0], prior or {}
    state = {} if state is None else state
    state.update({"abort": None, "abort_code": 0, "consecutive": 0, "transport_errors": 0, "rows_done": 0, "kept": 0,
                  "dropped": 0, "requests": 0, "models": set(), "total": len(pending), "last_print": time.monotonic(),
                  "empty_streak": 0, "held": []})

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

    def release_held():
        held, state["held"] = state["held"], []
        save(held)

    async def handle(job):
        kind, rows, attempt, past_by_id = job
        ms = [metas[r["id"]] for r in rows]
        entry = {"ts": ctx["now"](), "kind": kind, "n_rows": len(rows), "attempt": attempt,
                 "unofficial_model": bool(ctx.get("unofficial"))}
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
        if ok:
            state["empty_streak"] = 0
        elif kind == "batch" or not payload_ok or state["empty_streak"]:
            state["empty_streak"] += 1
        records, failed = [], []
        for k, row in enumerate(rows, 1):
            n_before, past = past_by_id.get(row["id"], (0, []))
            if k in ok:
                records.append(make_record(row, ms[k - 1], ok[k], attempts=n_before + 1, errors=past,
                                           warnings=warns.get(k, []), ctx=ctx))
                continue
            errs = [*past, str(errors.get(k, "missing"))[:300]]
            failed.append(history_record(row, n_before + 1, errs, ctx))
            if attempt < MAX_ATTEMPTS:
                push(0, ("single", [row], attempt + 1, {row["id"]: (n_before + 1, errs)}))
            else:
                norm = {"keep": False, "rewritten": None, "drop_reason": validate.LLM_ERROR}
                rec = make_record(row, ms[k - 1], norm, attempts=n_before + 1, errors=errs,
                                  warnings=warns.get(k, []), ctx=ctx)
                (state["held"] if state["empty_streak"] else records).append(rec)
        if failed:
            store.put_history(ctx["pv"], failed)
        save(records)
        if not state["empty_streak"] and state["held"]:
            release_held()
        if state["empty_streak"] >= max_empty_replies:
            abort(f"{state['empty_streak']} consecutive replies with no valid item (last: {status}, finish_reason "
                  f"{comp.finish_reason}, first error: {first_error}); the model output or the validator is broken. "
                  f"{len(state['held'])} rows held back from llm_error stay pending", 2)

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
        chunk = pending[i:i + batch_size]
        push(1, ("batch", chunk, 1, {r["id"]: prior[r["id"]] for r in chunk if prior.get(r["id"], (0,))[0]}))
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    waiters = [asyncio.ensure_future(queue.join()), asyncio.ensure_future(stop.wait())]
    try:
        await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in workers + waiters:
            t.cancel()
        await asyncio.gather(*workers, *waiters, return_exceptions=True)
    if state["abort"] is None and state["held"]:  # the streak ended with the queue, below the breaker: finalize
        release_held()
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
    """Counts over the finished records. Provenance (llm_models, served_model_roots, endpoints) counts only rows a
    model produced; rule drops are counted separately (rule_drops, model RULE_MODEL)."""
    by_source, by_style = {}, {}
    for t in targets:
        rec = records.get(t["id"])
        src = by_source.setdefault(t["source"], {"targets": 0, "kept": 0, "dropped": 0, "rule_dropped": 0})
        sty = by_style.setdefault(meta.assign_style(t["id"]), {"targets": 0, "kept": 0})
        src["targets"] += 1
        sty["targets"] += 1
        if rec is not None:
            src["kept" if rec["keep"] else "dropped"] += 1
            src["rule_dropped"] += is_rule(rec)
            sty["kept"] += bool(rec["keep"])
    for s in by_source.values():
        s["drop_rate"] = _rate(s["dropped"], s["kept"] + s["dropped"])
    recs = list(records.values())
    llm = [r for r in recs if not is_rule(r)]
    return {"total": len(recs), "kept": sum(r["keep"] for r in recs), "dropped": sum(not r["keep"] for r in recs),
            "rule_drops": len(recs) - len(llm), "rule_model": RULE_MODEL,
            "by_source": dict(sorted(by_source.items())),
            "by_style": {s: by_style[s] for s, _ in meta.STYLE_SPLIT if s in by_style},
            "drop_reasons": _counts(r.get("drop_reason") for r in recs if not r["keep"]),
            "warnings": _counts(w.split(":")[0] for r in recs for w in r.get("warnings") or ()),
            "models": _counts(str(r.get("model")) for r in recs),
            "llm_models": _counts(str(r.get("model")) for r in llm),
            "served_model_roots": _counts(r.get("served_model_root") or "(not reported)" for r in llm),
            "endpoints": _counts(str(r.get("endpoint")) for r in llm),
            "unofficial_records": sum(bool(r.get("unofficial_model")) for r in recs)}


def quality(records, req_all) -> dict:
    """first_try_valid_rate and drop_rate_model cover the rows sent to the model; drop_rate_raw (the gate) counts
    every non-llm_error drop, rule drops included, over all finished rows."""
    recs = list(records.values())
    llm = [r for r in recs if not is_rule(r)]
    batches = [r for r in req_all if r.get("kind") == "batch" and r.get("status") in COMPLETED]
    drops = [r for r in recs if not r["keep"] and r.get("drop_reason") != validate.LLM_ERROR]
    return {"first_try_valid_rate": _rate(sum(r.get("first_try_valid") is True for r in llm), len(llm)),
            "batch_json_ok_rate": _rate(sum(bool(r.get("payload_ok")) for r in batches), len(batches)),
            "llm_rows": len(llm), "rule_drops": len(recs) - len(llm),
            "rows_retried": sum((r.get("attempts") or 1) > 1 for r in llm),
            "llm_error_rows": sum(r.get("drop_reason") == validate.LLM_ERROR for r in recs),
            "drop_rate_raw": _rate(len(drops), len(recs)),
            "drop_rate_model": _rate(sum(not is_rule(r) for r in drops), len(llm)),
            "validation_errors": _counts(str(e).split(":")[0] for r in recs for e in r.get("errors") or ())}


def throughput(runs: list[dict]) -> dict:
    """Over every logged invocation that sent >= 1 request (sum of rows / sum of wall time), so a resume that sends
    nothing never replaces the numbers of the runs that did the work."""
    runs = [r for r in runs if (r.get("requests") or 0) >= 1]
    rows, reqs = sum(r.get("rows_done") or 0 for r in runs), sum(r.get("requests") or 0 for r in runs)
    wall, ct = sum(r.get("wall_s") or 0.0 for r in runs), sum(r.get("completion_tokens") or 0 for r in runs)
    return {"scope": "all invocations that sent >= 1 request", "invocations": len(runs), "rows": rows,
            "requests": reqs, "wall_s": round(wall, 3), "rows_per_min": round(rows / wall * 60, 2) if wall else None,
            "requests_per_min": round(reqs / wall * 60, 2) if wall else None,
            "completion_tokens_per_s": round(ct / wall, 2) if wall else None}


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


def build_stats(*, ctx, args, rows, selected, targets, records, req_this, req_all, runs, client_stats, state, started,
                finished, wall_s, limit) -> dict:
    sha, dirty = git_info()
    extra_raw, extra_natural = set(args.raw_sources or ()), set(args.natural_sources or ())
    this = summarize_requests(req_this)
    this["downgrades"] = list(client_stats.get("downgrades") or [])
    this["rate_limit_headers"] = dict(sorted((client_stats.get("rate_limit") or {}).items()))
    this["transport_errors"] = state.get("transport_errors", 0)
    q = quality(records, req_all)
    pending = sum(t["id"] not in records for t in targets)
    exit_code = state.get("abort_code") or (2 if pending else 0)
    unofficial_roots = {r.get("served_model_root") or "unverified" for r in records.values()
                        if r.get("unofficial_model")}
    if ctx.get("unofficial") and args.allow_unofficial_model:
        unofficial_roots.add(ctx.get("served_model_root") or "unverified")
    return {
        "run": {"run_id": ctx["run_id"], "started": started, "finished": finished, "wall_s": round(wall_s, 3),
                "rows_done": state.get("rows_done", 0), "rule_drops": state.get("rule_drops", 0),
                "rows_per_min": round(state.get("rows_done", 0) / wall_s * 60, 2) if this["n"] and wall_s else None,
                "command": ctx["command"], "git_sha": sha, "git_dirty": dirty, "endpoint": ctx["endpoint"],
                "base_url": ctx["base_url"], "model": ctx["model"], "model_label": ctx["model_label"],
                "models_reported": sorted(state.get("models") or ()),
                "served_model_root": ctx.get("served_model_root"), "unofficial_model": bool(unofficial_roots),
                "unofficial_roots": sorted(unofficial_roots),
                "allow_unofficial_model": bool(args.allow_unofficial_model),
                "prompt_version": ctx["pv"], "sampling": client_mod.SAMPLING, "batch_size": args.batch_size,
                "concurrency": ctx["concurrency"], "max_attempts": MAX_ATTEMPTS,
                "max_consecutive_empty_replies": args.max_consecutive_empty_replies,
                "max_consecutive_transport_errors": args.max_consecutive_transport_errors,
                "response_format_requested": args.response_format,
                "response_format_final": ctx.get("response_format_final") or args.response_format,
                "dry_run": bool(args.dry_run), "limit": limit, "splits": list(args.splits or ()), "note": args.note or "",
                "raw_sources": sorted(meta.RAW_LABEL_SOURCES - extra_natural | extra_raw),
                "natural_sources": sorted(meta.NATURAL_SOURCES - extra_raw | extra_natural),
                "select_pct": meta.SELECT_PCT, "style_split": [list(s) for s in meta.STYLE_SPLIT]},
        "manifest": {"path": str(Path(args.manifest).resolve()), "sha256": file_sha256(args.manifest), "rows": len(rows),
                     "rows_by_source": _counts(r["source"] for r in rows),
                     "selected_by_source": _counts(r["source"] for r in selected), "targets": len(targets)},
        "rewrites": rewrite_counts(targets, records),
        "quality": q,
        "requests": {"this_run": this, "all_runs": summarize_requests(req_all)},
        "throughput": throughput(runs),
        "runs": runs,
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
    print(f"rewrite {run['prompt_version']} | {run['model_label']} via {run['endpoint']} ({run['base_url']}) | "
          f"served checkpoint {run['served_model_root']}")
    if run["unofficial_model"]:
        print(f"NOT FOR PUBLICATION: unofficial/community checkpoint ({', '.join(run['unofficial_roots'])}); "
              "plumbing only", file=sys.stderr)
    n_sel = sum(man["selected_by_source"].values())
    print(f"targets {man['targets']} (selected {n_sel} of {man['rows']} rows{limit}) | "
          f"done {rw['total']} | kept {rw['kept']} | dropped {rw['dropped']} | pending {stats['pending_rows']}")
    if rw["drop_reasons"]:
        print("drop reasons: " + ", ".join(f"{k} {v}" for k, v in rw["drop_reasons"].items())
              + f" (numeric includes {rw['rule_drops']} dropped by rule, never sent)")
    for src, s in rw["by_source"].items():
        print(f"  {src}: {s['targets']} targets, kept {s['kept']}, dropped {s['dropped']} ({_pct(s['drop_rate'])})")
    print(f"first-try valid {_pct(q['first_try_valid_rate'])} of {q['llm_rows']} rows sent (gate >= "
          f"{FIRST_TRY_VALID_MIN:.0%}: {'PASS' if g['first_try_valid_ok'] else 'FAIL'}) | drop rate "
          f"{_pct(q['drop_rate_raw'])} (gate < {DROP_RATE_MAX:.0%}: {'PASS' if g['drop_rate_ok'] else 'FAIL'}; model "
          f"only {_pct(q['drop_rate_model'])}) | batch JSON ok {_pct(q['batch_json_ok_rate'])}")
    tp = stats["throughput"]
    print(f"this run: {this['n']} requests {this['by_status']}, {this['http_attempts']} HTTP attempts, "
          f"{this['prompt_tokens']} prompt + {this['completion_tokens']} completion tokens, wall {run['wall_s']}s, "
          f"{run['rows_done']} rows ({run['rows_per_min']} rows/min), {run['rule_drops']} rule drops")
    print(f"all {tp['invocations']} invocations with requests: {tp['rows']} rows, {tp['requests']} requests in "
          f"{tp['wall_s']}s, {tp['rows_per_min']} rows/min")
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
    selected, targets = select_targets(rows, limit, extra_raw, extra_natural, tuple(args.splits or ()))
    by_image = meta.index_by_image(rows)
    metas = {r["id"]: meta.row_meta(r, by_image) for r in targets}
    rule_ids = {r["id"] for r in targets if validate.is_numeric_label(metas[r["id"]]["label"])}  # never sent

    if args.show_prompt:
        first = [metas[r["id"]] for r in targets if r["id"] not in rule_ids][:args.batch_size]
        if not first:
            return _fail("no rows selected for rewriting (or all are numeric rule drops); nothing to show")
        print(json.dumps(prompts.build_messages(first), indent=2, ensure_ascii=False))
        return {"exit_code": 0, "show_prompt": True}

    endpoint = client_mod.ENDPOINTS[args.endpoint]
    model = args.model or endpoint.model
    argv = getattr(args, "argv", None)
    ctx = {"pv": prompts.PROMPT_VERSION, "run_id": f"{started.replace('-', '').replace(':', '')}-{uuid.uuid4().hex[:6]}",
           "model": model, "model_label": args.model_label or model, "endpoint": endpoint.name,
           "base_url": (args.base_url or endpoint.base_url).rstrip("/"), "now": now,
           "concurrency": args.concurrency or client_mod.DEFAULT_CONCURRENCY.get(endpoint.name, 6),
           "command": shlex.join(["python", SCRIPT, *(sys.argv[1:] if argv is None else argv)])}
    allow_unofficial = bool(args.allow_unofficial_model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state, client_stats = {}, {}
    with cache_mod.RewriteCache(args.cache or out / "cache.sqlite") as store:
        cached = store.get_many(ctx["pv"], [r["id"] for r in targets])
        # The rule is authoritative: numeric labels always get the rule record (even over an older LLM record), and
        # a rule record whose label no longer matches the rule is redone by the model.
        rule_new = [r for r in targets if r["id"] in rule_ids
                    and not (is_rule(cached.get(r["id"])) and is_current(r, cached.get(r["id"])))]
        pending = [r for r in targets if r["id"] not in rule_ids
                   and (not is_current(r, cached.get(r["id"]), allow_unofficial) or is_rule(cached.get(r["id"]))
                        or (args.retry_errors and cached[r["id"]].get("drop_reason") == validate.LLM_ERROR))]
        history = store.history(ctx["pv"], [r["id"] for r in pending])
        prior = {r["id"]: prior_attempts(r, history.get(r["id"]), cached.get(r["id"]), allow_unofficial)
                 for r in pending}
        api_key = client_mod.load_api_key(endpoint.api_key_env) if endpoint.api_key_env else None
        if pending and endpoint.api_key_env and not api_key:
            return _fail(f"{endpoint.api_key_env} is not set (environment or ./.env; run from the repo root)")
        client = client_mod.LightningClient(endpoint, api_key=api_key, base_url=args.base_url, model=args.model,
                                            concurrency=ctx["concurrency"], response_format=args.response_format,
                                            transport=transport, sleep=sleep)
        try:
            root = client_mod.served_root(await client.list_models(), model)
            hosted = is_nvidia_hosted(ctx["base_url"])
            official = hosted or client_mod.is_official_root(root)
            label = f"{HOSTED_LABEL}, {model}" if hosted else root or model
            ctx.update(served_model_root=root, unofficial=not official, model_label=args.model_label or label)
            if pending and not official:
                models_url = f"GET {ctx['base_url']}/models"
                why = (f"{models_url} reports checkpoint {root!r} for {model!r}" if root else
                       f"could not verify the served checkpoint ({models_url} has no root for {model!r})")
                if not allow_unofficial:
                    return _fail(f"{why}. Only {client_mod.OFFICIAL_CHECKPOINT} (served by its HF id) or NVIDIA's "
                                 "hosted endpoint may produce data we report. Pass --allow-unofficial-model for "
                                 "plumbing only (outputs are marked NOT FOR PUBLICATION).", 3)
                print(f"WARNING: {why}; --allow-unofficial-model: outputs are NOT FOR PUBLICATION", file=sys.stderr)
            elif pending and root and root != client_mod.OFFICIAL_CHECKPOINT and not is_nvidia_hosted(ctx["base_url"]):
                print(f"warning: served checkpoint {root!r} is an nvidia/ id but not {client_mod.OFFICIAL_CHECKPOINT}",
                      file=sys.stderr)
            if rule_new:
                store.put_many(ctx["pv"], [rule_record(r, metas[r["id"]], ctx) for r in rule_new])
                state["rule_drops"] = len(rule_new)
                print(f"{len(rule_new)} numeric labels dropped by rule (validate.is_numeric_label), never sent",
                      flush=True)
            if pending:
                print(f"{len(pending)} of {len(targets)} targets pending; {ctx['endpoint']} {ctx['base_url']} model "
                      f"{model} ({ctx['model_label']}), batch {args.batch_size}, concurrency {ctx['concurrency']}",
                      flush=True)
                await process(pending, metas, client, store, ctx, batch_size=args.batch_size,
                              concurrency=ctx["concurrency"],
                              max_transport_errors=args.max_consecutive_transport_errors,
                              max_empty_replies=args.max_consecutive_empty_replies, prior=prior, state=state)
        except asyncio.CancelledError:  # Ctrl-C: cached rows are safe; write outputs for them
            state.update(abort="interrupted", abort_code=130)
        finally:
            ctx["response_format_final"] = client.response_format
            client_stats = client.stats
            await client.aclose()
        current = store.get_many(ctx["pv"], [r["id"] for r in targets])
        records = {r["id"]: current[r["id"]] for r in targets
                   if is_current(r, current.get(r["id"]), allow_unofficial)
                   and is_rule(current[r["id"]]) == (r["id"] in rule_ids)}
        req_this, req_all = store.requests(ctx["pv"], ctx["run_id"]), store.requests(ctx["pv"])
        if not allow_unofficial:  # plumbing requests never count toward a publishable run's totals
            req_all = [r for r in req_all if not r.get("unofficial_model")]
        finished, wall_s = now(), time.monotonic() - t0
        if req_this:  # an invocation that sent nothing (a resume) never replaces the runs that did the work
            sent = summarize_requests(req_this)
            store.log_run(ctx["pv"], {
                "run_id": ctx["run_id"], "started": started, "finished": finished, "wall_s": round(wall_s, 3),
                "rows_done": state.get("rows_done", 0), "rule_drops": state.get("rule_drops", 0),
                "requests": sent["n"], "prompt_tokens": sent["prompt_tokens"],
                "completion_tokens": sent["completion_tokens"], "endpoint": ctx["endpoint"],
                "base_url": ctx["base_url"], "model_label": ctx["model_label"],
                "served_model_root": ctx.get("served_model_root"), "unofficial_model": bool(ctx.get("unofficial")),
                "command": ctx["command"]})
        runs = [r for r in store.runs(ctx["pv"]) if allow_unofficial or not r["unofficial_model"]]
    stats = build_stats(ctx=ctx, args=args, rows=rows, selected=selected, targets=targets, records=records,
                        req_this=req_this, req_all=req_all, runs=runs, client_stats=client_stats, state=state,
                        started=started, finished=finished, wall_s=wall_s, limit=limit)
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
