"""End-to-end rewrite.py runs against the fake Lightning server (httpx.MockTransport, fake sleep, fixed clock)."""

import asyncio
import hashlib
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from data_prep.rewrite import cache as cache_mod
from data_prep.rewrite import meta, method, prompts, rewrite, validate
from data_prep.rewrite.tests import fake_lightning as fake
from data_prep.rewrite.tests import synth

TS = "2026-10-17T12:00:00Z"
PV = prompts.PROMPT_VERSION
REPO = Path(__file__).resolve().parents[3]
KEY = "nvapi-test-not-a-real-key"


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # load_api_key's default ./.env is never the repo's


@pytest.fixture
def manifest(tmp_path) -> Path:
    return synth.make_manifest(tmp_path / "data")


def run_rewrite(manifest, out, handler, *extra, sleeps=None, endpoint="local") -> dict:
    sleeps = [] if sleeps is None else sleeps

    async def sleep(s):
        sleeps.append(s)

    args = rewrite.build_parser().parse_args(["--manifest", str(manifest), "--out", str(out), "--endpoint", endpoint,
                                              *map(str, extra)])
    return asyncio.run(rewrite.run(args, transport=httpx.MockTransport(handler), sleep=sleep, now=lambda: TS))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def selected(manifest) -> list[dict]:
    return [r for r in read_jsonl(manifest) if meta.is_selected(r)]


def label_of(row) -> str:
    return meta.normalize_label(row["instruction"])


def is_rule_label(row) -> bool:
    return validate.is_numeric_label(label_of(row))


def sent_rows(manifest) -> list[dict]:
    """Selected rows that go to the model; numeric labels are dropped by rule before any request."""
    return [r for r in selected(manifest) if not is_rule_label(r)]


def rule_ids(manifest) -> list[str]:
    return [r["id"] for r in selected(manifest) if is_rule_label(r)]


def unique_label_rows(rows) -> list[dict]:
    labels = [label_of(r) for r in rows]
    return [r for r, lab in zip(rows, labels) if labels.count(lab) == 1]


def cached(out, ids=None) -> dict[str, dict]:
    with cache_mod.RewriteCache(Path(out) / "cache.sqlite") as c:
        return c.get_many(PV, ids if ids is not None else [r["id"] for r in read_jsonl(Path(out) / "rewrites.jsonl")])


def history(out, ids) -> dict[str, dict]:
    with cache_mod.RewriteCache(Path(out) / "cache.sqlite") as c:
        return c.history(PV, ids)


def logged(out) -> list[dict]:
    with cache_mod.RewriteCache(Path(out) / "cache.sqlite") as c:
        return c.requests(PV)


def invalid_reply(body) -> httpx.Response:
    """A parseable reply in which every item fails validation (empty instruction)."""
    items = [{"k": it["k"], "keep": True, "instruction": ""} for it in fake.batch_items(body)]
    return fake.completion(json.dumps({"items": items}))


def test_fake_answers_validate_under_the_v3_rules(tmp_path):
    """The fake answers like a well-behaved Lightning: every plausible() reply to an item exactly as sent passes
    validate_item with no error, for every style, including look-alike ("Search" / "Search 14") and duplicate labels."""
    m = synth.make_manifest(tmp_path / "all", n_images=30, per_image=16, write_images=False,
                            sources=("os_atlas/androidworld", "os_atlas/fineweb", "os_atlas/windows", "os_atlas/linux",
                                     "os_atlas/macos"))
    rows = read_jsonl(m)
    by_image, checked = meta.index_by_image(rows), 0
    for r in rows:
        for style in validate.STYLES:
            m_ = meta.row_meta(r, by_image, style=style)
            [item] = fake.batch_items({"messages": [{"role": "user", "content": prompts.format_batch([m_])}]})
            assert item["neighbors"] == m_["neighbors"] and all("target_is" in n for n in item["neighbors"])
            answer = fake.plausible(item)
            _, error, _ = validate.validate_item(answer, m_)
            assert error is None, (m_, answer, error)
            checked += answer["keep"]
    assert checked > 500


def test_happy_path_writes_outputs_with_expected_counts(manifest, tmp_path):
    out, sel, llm, rules = tmp_path / "out", selected(manifest), sent_rows(manifest), set(rule_ids(manifest))
    assert sel and rules and len(llm) == len(sel) - len(rules)
    handler = fake.make_handler()
    stats = run_rewrite(manifest, out, handler, "--batch-size", 5)

    assert stats["exit_code"] == 0 and stats["pending_rows"] == 0 and stats["aborted"] is None
    recs = read_jsonl(out / "rewrites.jsonl")
    assert [r["id"] for r in recs] == [r["id"] for r in sel]
    for rec, row in zip(recs, sel):
        assert list(rec) == list(rewrite.PUBLIC_FIELDS)
        assert rec["original"] == row["instruction"] and rec["source"] == row["source"]
        assert rec["style"] == meta.assign_style(row["id"]) and rec["prompt_version"] == PV and rec["ts"] == TS
        # --model-label defaults to the served root; rule drops name the rule instead of a model
        assert rec["model"] == (rewrite.RULE_MODEL if row["id"] in rules else fake.OFFICIAL_ROOT)
        assert (rec["rewritten"] is not None) == rec["keep"]
        assert rec["keep"] or rec["drop_reason"] in validate.DROP_REASONS
        if fake.NUMERIC.fullmatch(row["instruction"]):
            assert (rec["keep"], rec["drop_reason"]) == (False, "numeric") and row["id"] in rules
    assert len(handler.calls) == math.ceil(len(llm) / 5)
    sent = [it["label"] for c in handler.calls for it in c["items"]]
    assert sent == [label_of(r) for r in llm]  # rule drops never reach the model
    body = handler.calls[0]["body"]
    assert body["model"] == "lightning" and body["temperature"] == 0.7 and body["top_p"] == 0.95
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0] == {"role": "system", "content": prompts.SYSTEM_PROMPT}

    rw, q = stats["rewrites"], stats["quality"]
    n_drop = sum(not r["keep"] for r in recs)
    assert rw["total"] == len(sel) == rw["kept"] + rw["dropped"] == stats["manifest"]["targets"]
    assert rw["rule_drops"] == q["rule_drops"] == stats["run"]["rule_drops"] == len(rules) and q["llm_rows"] == len(llm)
    assert sum(s["targets"] for s in rw["by_source"].values()) == len(sel)
    assert sum(s["rule_dropped"] for s in rw["by_source"].values()) == len(rules)
    assert sum(rw["drop_reasons"].values()) == rw["dropped"] and rw["drop_reasons"]["numeric"] >= len(rules)
    assert q["first_try_valid_rate"] == 1.0 and q["batch_json_ok_rate"] == 1.0 and q["rows_retried"] == 0
    assert q["drop_rate_raw"] == round(n_drop / len(sel), 4)
    assert q["drop_rate_model"] == round((n_drop - len(rules)) / len(llm), 4) < q["drop_rate_raw"]
    assert set(stats["gate"]) >= {"first_try_valid_ok", "drop_rate_ok"} and stats["gate"]["first_try_valid_ok"]
    this = stats["requests"]["this_run"]
    assert this["n"] == len(handler.calls) == len(logged(out)) and this["by_kind"] == {"batch": len(handler.calls)}
    assert this["prompt_tokens"] == 900 * len(handler.calls) and this["total_tokens"] == 1020 * len(handler.calls)
    assert stats["manifest"]["rows"] == len(read_jsonl(manifest))
    assert stats["manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert stats["run"]["models_reported"] == [fake.MODEL] and stats["run"]["prompt_version"] == PV
    assert stats["run"]["served_model_root"] == fake.OFFICIAL_ROOT and stats["run"]["unofficial_model"] is False
    assert stats["run"]["concurrency"] == 4 and len(handler.model_calls) == 1  # local default; one GET /models
    assert rw["unofficial_records"] == 0
    assert rw["served_model_roots"] == rw["llm_models"] == {fake.OFFICIAL_ROOT: len(llm)}
    assert rw["models"] == {fake.OFFICIAL_ROOT: len(llm), rewrite.RULE_MODEL: len(rules)}
    assert rw["endpoints"] == {"local": len(llm)}
    [run] = stats["runs"]
    assert list(run) == list(cache_mod.RUN_FIELDS) and run["run_id"] == stats["run"]["run_id"]
    assert (run["requests"], run["rows_done"], run["rule_drops"]) == (len(handler.calls), len(llm), len(rules))
    assert run["prompt_tokens"] == 900 * len(handler.calls) and run["completion_tokens"] == 120 * len(handler.calls)
    assert (run["endpoint"], run["served_model_root"], run["unofficial_model"]) == ("local", fake.OFFICIAL_ROOT, False)
    assert run["wall_s"] == stats["run"]["wall_s"] and run["command"] == stats["run"]["command"]
    tp = stats["throughput"]
    assert (tp["invocations"], tp["rows"], tp["requests"], tp["wall_s"]) == (1, len(llm), len(handler.calls),
                                                                              run["wall_s"])
    assert set(stats) >= {"run", "manifest", "rewrites", "quality", "requests", "throughput", "runs", "pending_rows",
                          "gate"}
    assert json.loads((out / "stats.json").read_text()) == json.loads(json.dumps(stats))
    text = (out / "METHOD.md").read_text(encoding="utf-8")
    assert prompts.SYSTEM_PROMPT in text and '% 100 < 60' in text
    assert all(m["content"] in text for m in prompts.FEW_SHOT)
    assert f"`{fake.OFFICIAL_ROOT}` ({len(llm)} rows)" in text and f"`local` ({len(llm)} rows)" in text
    assert f"{len(rules)} row{'s' * (len(rules) != 1)} (`model` = `{rewrite.RULE_MODEL}`)" in text
    assert inspect.getsource(validate.is_numeric_label).rstrip() in text
    entry = cached(out)[llm[0]["id"]]
    assert entry["attempts"] == 1 and entry["first_try_valid"] and entry["endpoint"] == "local"
    assert entry["served_model_root"] == fake.OFFICIAL_ROOT and entry["unofficial_model"] is False
    rule_entry = cached(out)[sorted(rules)[0]]
    assert {k: rule_entry[k] for k in ("keep", "rewritten", "drop_reason", "model", "attempts", "first_try_valid",
                                       "warnings", "errors", "endpoint", "served_model_root", "unofficial_model")} == {
        "keep": False, "rewritten": None, "drop_reason": "numeric", "model": rewrite.RULE_MODEL, "attempts": 0,
        "first_try_valid": None, "warnings": [], "errors": [], "endpoint": None, "served_model_root": None,
        "unofficial_model": False}
    assert history(out, [r["id"] for r in sel]) == {}  # nothing ever failed
    assert "NOT FOR PUBLICATION" not in text


def test_resume_makes_zero_requests_and_keeps_the_runs_numbers(manifest, tmp_path):
    out = tmp_path / "out"
    first = run_rewrite(manifest, out, fake.make_handler(), "--batch-size", 4)
    before = (out / "rewrites.jsonl").read_bytes()
    again = fake.make_handler()
    second = run_rewrite(manifest, out, again, "--batch-size", 4, "--concurrency", 1)
    assert again.calls == [] and second["exit_code"] == 0
    assert (out / "rewrites.jsonl").read_bytes() == before
    assert second["requests"]["this_run"]["n"] == 0
    assert second["requests"]["all_runs"]["n"] == first["requests"]["this_run"]["n"]
    assert second["quality"] == first["quality"] and second["rewrites"] == first["rewrites"]
    # the resume sent nothing: it is not logged and does not replace the first invocation's throughput
    assert second["throughput"] == first["throughput"] and second["runs"] == first["runs"] and len(first["runs"]) == 1
    assert second["run"]["rule_drops"] == 0 and second["run"]["rows_done"] == 0 and second["run"]["rows_per_min"] is None
    assert first["runs"][0]["run_id"] in (out / "METHOD.md").read_text(encoding="utf-8")


def test_malformed_batch_rows_are_requeued_individually(manifest, tmp_path):
    out, llm = tmp_path / "out", sent_rows(manifest)
    handler = fake.make_handler(malformed_first=1)
    stats = run_rewrite(manifest, out, handler, "--batch-size", 4, "--concurrency", 1)
    assert stats["exit_code"] == 0 and stats["pending_rows"] == 0
    first4 = [r["id"] for r in llm[:4]]
    recs = cached(out, [r["id"] for r in llm])
    assert all(recs[i]["attempts"] == 2 and not recs[i]["first_try_valid"] for i in first4)
    assert all(recs[i]["errors"][0].startswith("invalid_json") for i in first4)
    assert all(recs[r["id"]]["attempts"] == 1 for r in llm[4:])
    assert [len(c["items"]) for c in handler.calls[1:5]] == [1, 1, 1, 1]  # retries jump the queue
    q, this = stats["quality"], stats["requests"]["this_run"]
    assert q["rows_retried"] == len(first4)
    assert q["first_try_valid_rate"] == round((len(llm) - 4) / len(llm), 4)
    n_batches = math.ceil(len(llm) / 4)
    assert q["batch_json_ok_rate"] == round((n_batches - 1) / n_batches, 4)
    assert this["by_kind"] == {"batch": n_batches, "single": 4} and this["by_status"]["invalid_json"] == 1
    assert logged(out)[0]["sample_text"] == fake.MALFORMED
    assert {i: h["attempts"] for i, h in history(out, first4).items()} == dict.fromkeys(first4, 1)


def test_missing_item_is_requeued(manifest, tmp_path):
    out, llm = tmp_path / "out", sent_rows(manifest)
    stats = run_rewrite(manifest, out, fake.make_handler(drop_k=2), "--batch-size", 5)
    recs = cached(out, [r["id"] for r in llm])
    second_of_batch = [llm[i]["id"] for i in range(1, len(llm), 5)]
    assert all(recs[i]["attempts"] == 2 and recs[i]["errors"] == ["missing"] for i in second_of_batch)
    assert sum(r["attempts"] == 2 for r in recs.values()) == len(second_of_batch)
    assert stats["requests"]["this_run"]["by_status"]["partial"] == len(second_of_batch)
    assert stats["exit_code"] == 0


def test_truncated_reply_counts_as_invalid(manifest, tmp_path):
    out, llm = tmp_path / "out", sent_rows(manifest)
    stats = run_rewrite(manifest, out, fake.make_handler(length_first=1), "--batch-size", 4, "--concurrency", 1)
    recs = cached(out, [r["id"] for r in llm])
    assert all(recs[r["id"]]["attempts"] == 2 for r in llm[:4])
    assert logged(out)[0]["status"] == "invalid_json" and logged(out)[0]["finish_reason"] == "length"
    assert stats["exit_code"] == 0


def test_persistently_invalid_row_becomes_llm_error_and_keeps_its_history_when_retried(manifest, tmp_path):
    out, llm = tmp_path / "out", sent_rows(manifest)
    victim = unique_label_rows(llm)[0]
    label = label_of(victim)
    handler = fake.make_handler(invalid_labels={label})
    stats = run_rewrite(manifest, out, handler, "--batch-size", 5)

    with_victim = [c for c in handler.calls if any(it["label"] == label for it in c["items"])]
    assert len(with_victim) == 3 and [len(c["items"]) for c in with_victim[1:]] == [1, 1]
    rec = cached(out, [victim["id"]])[victim["id"]]
    assert (rec["keep"], rec["rewritten"], rec["drop_reason"]) == (False, None, validate.LLM_ERROR)
    assert rec["attempts"] == 3 and rec["errors"] == ["missing_instruction"] * 3 and not rec["first_try_valid"]
    public = {r["id"]: r for r in read_jsonl(out / "rewrites.jsonl")}[victim["id"]]
    assert public["drop_reason"] == validate.LLM_ERROR
    assert stats["quality"]["llm_error_rows"] == 1 and stats["rewrites"]["drop_reasons"][validate.LLM_ERROR] == 1
    assert stats["exit_code"] == 0 and stats["pending_rows"] == 0 and stats["aborted"] is None

    idle = fake.make_handler()
    assert run_rewrite(manifest, out, idle, "--batch-size", 5)["exit_code"] == 0 and idle.calls == []
    healed = fake.make_handler()  # --retry-errors re-sends only the llm_error row (never the rule drops)
    assert run_rewrite(manifest, out, healed, "--batch-size", 5, "--retry-errors")["exit_code"] == 0
    assert len(healed.calls) == 1 and [it["label"] for it in healed.calls[0]["items"]] == [label]
    rec = cached(out, [victim["id"]])[victim["id"]]
    assert rec["drop_reason"] != validate.LLM_ERROR and rec["model"] == fake.OFFICIAL_ROOT
    assert rec["attempts"] == 4 and rec["first_try_valid"] is False and rec["errors"] == ["missing_instruction"] * 3
    stats = run_rewrite(manifest, out, fake.make_handler(), "--batch-size", 5)
    assert stats["quality"]["llm_error_rows"] == 0 and stats["quality"]["rows_retried"] == 1
    assert stats["quality"]["first_try_valid_rate"] == round((len(llm) - 1) / len(llm), 4)


def test_row_that_failed_before_an_abort_is_never_first_try_valid(manifest, tmp_path):
    """The row fails validation in the batch, then its single-row retry never gets a reply (transport abort). The
    failed attempt is kept in the row history, so the next invocation records attempts=2, first_try_valid=False."""
    out, llm = tmp_path / "out", sent_rows(manifest)
    victim = unique_label_rows(llm)[0]
    label = label_of(victim)

    def flaky(request, body, n):
        if n == 0:
            items = [fake.plausible(it) if it["label"] != label else {"k": it["k"], "keep": True, "instruction": ""}
                     for it in fake.batch_items(body)]
            return fake.completion(json.dumps({"items": items}))
        raise httpx.ConnectError("connection refused", request=request)

    stats = run_rewrite(manifest, out, fake.make_handler(flaky), "--batch-size", 50, "--concurrency", 1,
                        "--max-consecutive-transport-errors", 1)
    assert stats["exit_code"] == 2 and "consecutive transport errors" in stats["aborted"]
    assert stats["pending_rows"] == 1 and victim["id"] not in cached(out, [victim["id"]])
    assert history(out, [victim["id"]])[victim["id"]] == {
        "id": victim["id"], "original": victim["instruction"], "source": victim["source"], "unofficial_model": False,
        "attempts": 1, "errors": ["missing_instruction"]}

    healthy = fake.make_handler()
    stats = run_rewrite(manifest, out, healthy, "--batch-size", 50)
    assert [it["label"] for c in healthy.calls for it in c["items"]] == [label]
    assert healthy.calls[0]["body"]["messages"][-1]["content"].count('{"k":') == 1
    rec = cached(out, [victim["id"]])[victim["id"]]
    assert rec["model"] == fake.OFFICIAL_ROOT and rec["attempts"] == 2 and rec["first_try_valid"] is False
    assert rec["errors"] == ["missing_instruction"]
    assert stats["exit_code"] == 0 and stats["quality"]["first_try_valid_rate"] == round((len(llm) - 1) / len(llm), 4)
    assert len(stats["runs"]) == 2 and stats["throughput"]["rows"] == len(llm)


@pytest.mark.parametrize("fault", ["malformed", "all_invalid"])
def test_output_breaker_stops_when_nothing_validates_and_finalizes_nothing(manifest, tmp_path, fault):
    out, llm = tmp_path / "out", sent_rows(manifest)
    ids = [r["id"] for r in llm]
    handler = (fake.make_handler(malformed_first=10_000) if fault == "malformed" else
               fake.make_handler(invalid_labels={label_of(r) for r in llm}))
    stats = run_rewrite(manifest, out, handler, "--batch-size", 3, "--concurrency", 1)
    # batch 1 (3 rows), its 3 single retries, then row 1's last retry: 5 replies in a row with no valid item
    assert len(handler.calls) == 5 and [len(c["items"]) for c in handler.calls] == [3, 1, 1, 1, 1]
    assert stats["exit_code"] == 2 and "5 consecutive replies with no valid item" in stats["aborted"]
    assert "1 rows held back from llm_error" in stats["aborted"]
    assert stats["pending_rows"] == len(llm) and stats["quality"]["llm_error_rows"] == 0
    assert cached(out, ids) == {} and set(cached(out)) == set(rule_ids(manifest))
    assert {i: h["attempts"] for i, h in history(out, ids).items()} == {ids[0]: 3, ids[1]: 2, ids[2]: 2}

    stats = run_rewrite(manifest, out, fake.make_handler(), "--batch-size", 3)
    recs = cached(out, ids)
    assert stats["exit_code"] == 0 and [recs[i]["attempts"] for i in ids[:4]] == [4, 3, 3, 1]
    assert not any(recs[i]["first_try_valid"] for i in ids[:3])
    assert not any(recs[i]["drop_reason"] == validate.LLM_ERROR for i in ids)
    assert stats["quality"]["first_try_valid_rate"] == round((len(llm) - 3) / len(llm), 4)


def test_stubborn_single_retries_do_not_trip_the_breaker(manifest, tmp_path):
    out, llm = tmp_path / "out", sent_rows(manifest)
    victims = unique_label_rows(llm)[:4]
    handler = fake.make_handler(invalid_labels={label_of(r) for r in victims})
    stats = run_rewrite(manifest, out, handler, "--batch-size", 50, "--concurrency", 1,
                        "--max-consecutive-empty-replies", 2)
    # the batch has valid items; the 8 failing single-row retries after it are not counted against the server
    assert len(handler.calls) == 1 + 2 * len(victims)
    assert stats["exit_code"] == 0 and stats["aborted"] is None
    assert stats["quality"]["llm_error_rows"] == len(victims)


@pytest.mark.parametrize("limit,tripped", [(4, False), (3, True)])
def test_rows_that_fail_during_a_streak_are_held_back(manifest, tmp_path, limit, tripped):
    """Batch size 1: the first row fails 3 times in a row (streak 3). Limit 4: the next valid reply ends the streak
    and the row becomes llm_error. Limit 3: the breaker trips and the row stays pending."""
    out, llm = tmp_path / "out", sent_rows(manifest)
    handler = fake.make_handler(lambda request, body, n: invalid_reply(body) if n < 3 else None)
    stats = run_rewrite(manifest, out, handler, "--batch-size", 1, "--concurrency", 1,
                        "--max-consecutive-empty-replies", limit)
    first = llm[0]["id"]
    if tripped:
        assert stats["exit_code"] == 2 and "3 consecutive replies" in stats["aborted"] and len(handler.calls) == 3
        assert cached(out, [first]) == {} and history(out, [first])[first]["attempts"] == 3
        assert stats["pending_rows"] == len(llm)
    else:
        assert stats["exit_code"] == 0 and len(handler.calls) == 3 + len(llm) - 1
        rec = cached(out, [first])[first]
        assert rec["drop_reason"] == validate.LLM_ERROR and rec["attempts"] == 3
        assert stats["quality"]["llm_error_rows"] == 1


def test_rows_held_back_at_the_end_of_the_queue_are_finalized(manifest, tmp_path):
    out, llm = tmp_path / "out", sent_rows(manifest)
    last = len(llm) - 1  # batch size 1: the last batch and its two retries are the last three calls
    handler = fake.make_handler(lambda request, body, n: invalid_reply(body) if n >= last else None)
    stats = run_rewrite(manifest, out, handler, "--batch-size", 1, "--concurrency", 1)
    assert stats["exit_code"] == 0 and len(handler.calls) == len(llm) + 2
    rec = cached(out, [llm[-1]["id"]])[llm[-1]["id"]]
    assert rec["drop_reason"] == validate.LLM_ERROR and rec["attempts"] == 3 and stats["pending_rows"] == 0


@pytest.mark.parametrize("status", [429, 503])
def test_rate_limit_and_5xx_bursts_recover(manifest, tmp_path, status):
    out, llm, sleeps = tmp_path / "out", sent_rows(manifest), []
    handler = fake.make_handler(fail_first=3, fail_status=status, retry_after=1)
    stats = run_rewrite(manifest, out, handler, "--batch-size", 5, sleeps=sleeps)
    n_batches = math.ceil(len(llm) / 5)
    assert stats["exit_code"] == 0 and stats["pending_rows"] == 0
    assert len(handler.calls) == n_batches + 3 and len(sleeps) == 3 and min(sleeps) >= 1
    this = stats["requests"]["this_run"]
    assert this["n"] == n_batches and this["http_attempts"] == n_batches + 3 and this["retries"] == 3
    assert stats["quality"]["first_try_valid_rate"] == 1.0


def test_fatal_401_aborts_without_crashing_or_leaking_the_key(manifest, tmp_path, capsys):
    out, sel, rules = tmp_path / "out", selected(manifest), rule_ids(manifest)
    handler = fake.make_handler("unauthorized")
    stats = run_rewrite(manifest, out, handler, "--batch-size", 3, "--concurrency", 2, endpoint="hosted")
    assert stats["exit_code"] != 0 and "401" in stats["aborted"]
    assert 1 <= len(handler.calls) <= 2
    assert handler.calls[0]["headers"]["authorization"] == f"Bearer {KEY}"
    assert handler.calls[0]["body"]["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    # only the rule drops (no model involved) are done; everything sent to the model stays pending
    assert [r["id"] for r in read_jsonl(out / "rewrites.jsonl")] == rules
    assert stats["pending_rows"] == len(sel) - len(rules) and set(cached(out, [r["id"] for r in sel])) == set(rules)
    captured = capsys.readouterr()
    assert "NVIDIA_API_KEY" in captured.out + captured.err
    for blob in (captured.out, captured.err, (out / "stats.json").read_text(), (out / "METHOD.md").read_text(),
                 json.dumps(logged(out))):
        assert KEY not in blob


def test_missing_api_key_aborts_before_any_request(manifest, tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    handler = fake.make_handler()
    stats = run_rewrite(manifest, tmp_path / "out", handler, endpoint="hosted")
    assert stats["exit_code"] == 1 and "NVIDIA_API_KEY" in stats["error"] and handler.calls == []


def test_full_outage_aborts_and_leaves_rows_pending_then_resumes(manifest, tmp_path):
    out, sel, llm, rules = tmp_path / "out", selected(manifest), sent_rows(manifest), rule_ids(manifest)
    handler = fake.make_handler("offline")
    stats = run_rewrite(manifest, out, handler, "--batch-size", 2, "--concurrency", 2,
                        "--max-consecutive-transport-errors", 3)
    assert stats["exit_code"] == 2 and "consecutive transport errors" in stats["aborted"]
    assert stats["pending_rows"] == len(llm) and [r["id"] for r in read_jsonl(out / "rewrites.jsonl")] == rules
    assert set(cached(out, [r["id"] for r in sel])) == set(rules)
    assert history(out, [r["id"] for r in llm]) == {}  # no reply, so no attempt is counted
    statuses = stats["requests"]["this_run"]["by_status"]
    assert list(statuses) == ["transport_error"] and 3 <= statuses["transport_error"] < math.ceil(len(llm) / 2)

    stats = run_rewrite(manifest, out, fake.make_handler(), "--batch-size", 2)
    assert stats["exit_code"] == 0 and stats["pending_rows"] == 0
    assert [r["id"] for r in read_jsonl(out / "rewrites.jsonl")] == [r["id"] for r in sel]
    assert stats["quality"]["first_try_valid_rate"] == 1.0


def test_partial_outage_keeps_failed_batch_pending_until_rerun(manifest, tmp_path):
    out, sel, llm = tmp_path / "out", selected(manifest), sent_rows(manifest)
    first4 = [r["id"] for r in llm[:4]]
    handler = fake.make_handler(fail_first=6, fail_status=503)  # the client's 6 attempts all fail for batch 1
    stats = run_rewrite(manifest, out, handler, "--batch-size", 4, "--concurrency", 1)
    assert stats["exit_code"] == 2 and stats["aborted"] is None and stats["pending_rows"] == 4
    assert [r["id"] for r in read_jsonl(out / "rewrites.jsonl")] == [r["id"] for r in sel if r["id"] not in first4]
    assert stats["requests"]["this_run"]["by_status"]["transport_error"] == 1
    first_run = stats["run"]["run_id"]

    rerun = fake.make_handler()
    stats = run_rewrite(manifest, out, rerun, "--batch-size", 4)
    assert len(rerun.calls) == 1 and [it["label"] for it in rerun.calls[0]["items"]] == [label_of(r) for r in llm[:4]]
    assert stats["exit_code"] == 0 and stats["pending_rows"] == 0
    assert stats["requests"]["all_runs"]["by_status"]["transport_error"] == 1
    # run history: both invocations sent requests, so both count toward the published throughput
    assert [r["run_id"] for r in stats["runs"]] == [first_run, stats["run"]["run_id"]]
    assert [r["rows_done"] for r in stats["runs"]] == [len(llm) - 4, 4]
    tp = stats["throughput"]
    assert tp["invocations"] == 2 and tp["rows"] == len(llm) and tp["requests"] == stats["requests"]["all_runs"]["n"]
    assert tp["wall_s"] == round(sum(r["wall_s"] for r in stats["runs"]), 3)
    text = (out / "METHOD.md").read_text(encoding="utf-8")
    assert first_run in text and stats["run"]["run_id"] in text
    assert stats["quality"]["first_try_valid_rate"] == 1.0  # a transport failure is not an attempt


def test_dry_run_limits_to_200_rows_by_sample_hash(tmp_path):
    manifest = synth.make_manifest(tmp_path / "big", n_images=70, per_image=8, write_images=False,
                                   sources=("os_atlas/androidworld", "os_atlas/fineweb"))
    sel = selected(manifest)
    assert len(sel) > 200
    handler = fake.make_handler()
    stats = run_rewrite(manifest, tmp_path / "dry", handler, "--dry-run")
    order = sorted(sel, key=lambda r: hashlib.sha256(("sample:" + r["id"]).encode()).hexdigest())
    first200 = {r["id"] for r in order[:200]}
    ids = [r["id"] for r in read_jsonl(tmp_path / "dry" / "rewrites.jsonl")]
    assert ids == [r["id"] for r in sel if r["id"] in first200]
    assert stats["run"]["dry_run"] and stats["run"]["limit"] == 200 and stats["manifest"]["targets"] == 200
    assert sum(stats["manifest"]["selected_by_source"].values()) == len(sel)
    n_sent = sum(not is_rule_label(r) for r in order[:200])
    assert stats["rewrites"]["rule_drops"] == 200 - n_sent > 0
    assert len(handler.calls) == math.ceil(n_sent / 20) and stats["exit_code"] == 0

    small = run_rewrite(manifest, tmp_path / "dry", fake.make_handler(), "--limit", 7)
    assert small["requests"]["this_run"]["n"] == 0  # the 7 are a prefix of the 200, already cached
    assert set(r["id"] for r in read_jsonl(tmp_path / "dry" / "rewrites.jsonl")) <= first200
    assert small["manifest"]["targets"] == 7


def test_show_prompt_prints_first_batch_without_requests(manifest, tmp_path, capsys):
    out = tmp_path / "out"
    handler = fake.make_handler()
    stats = run_rewrite(manifest, out, handler, "--show-prompt", "--batch-size", 3)
    assert stats["exit_code"] == 0 and handler.calls == [] and not out.exists()
    messages = json.loads(capsys.readouterr().out)
    by_image = meta.index_by_image(read_jsonl(manifest))
    assert messages == prompts.build_messages([meta.row_meta(r, by_image) for r in sent_rows(manifest)[:3]])
    items = fake.batch_items({"messages": messages})
    assert len(items) == 3 and all(set(n) == {"label", "target_is"} for it in items for n in it["neighbors"])
    assert '"direction"' not in messages[-1]["content"]
    assert rewrite.main(["--manifest", str(manifest), "--out", str(out), "--show-prompt", "--endpoint", "hosted"]) == 0


@pytest.mark.parametrize("bad", [
    {"id": "screenspot:web_1:0", "source": "screenspot/web", "image_path": "images/screenspot/web_1.png"},
    {"id": "groundui:shot_1:0", "source": "groundui_18k", "image_path": "images/groundui/shot_1.png"},
])
def test_contaminated_manifest_aborts_before_any_request(manifest, tmp_path, bad):
    row = {"img_w": 640, "img_h": 400, "instruction": "Settings", "bbox": [1, 1, 50, 20], "platform": "desktop",
           "data_type": None, "role": None, "split": None, **bad}
    with open(manifest, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    handler = fake.make_handler()
    out = tmp_path / "out"
    stats = run_rewrite(manifest, out, handler)
    assert stats["exit_code"] == 1 and handler.calls == [] and not out.exists()
    assert rewrite.main(["--manifest", str(manifest), "--out", str(out), "--endpoint", "local"]) == 1


def _relabel(manifest, row_id, label) -> None:
    rows = read_jsonl(manifest)
    next(r for r in rows if r["id"] == row_id)["instruction"] = label
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_changed_label_invalidates_only_that_cached_row(manifest, tmp_path):
    out, llm = tmp_path / "out", sent_rows(manifest)
    run_rewrite(manifest, out, fake.make_handler())
    target = unique_label_rows(llm)[0]
    _relabel(manifest, target["id"], "Download statement")
    handler = fake.make_handler()
    stats = run_rewrite(manifest, out, handler)
    assert [[it["label"] for it in c["items"]] for c in handler.calls] == [["Download statement"]]
    assert {r["id"]: r for r in read_jsonl(out / "rewrites.jsonl")}[target["id"]]["original"] == "Download statement"
    assert stats["exit_code"] == 0

    _relabel(manifest, target["id"], "73 minutes ago")  # now numeric: rule drop, nothing sent
    handler = fake.make_handler()
    stats = run_rewrite(manifest, out, handler)
    assert handler.calls == [] and cached(out, [target["id"]])[target["id"]]["model"] == rewrite.RULE_MODEL
    assert stats["run"]["rule_drops"] == 1 and stats["rewrites"]["rule_drops"] == len(rule_ids(manifest))

    rule_row = next(r for r in selected(manifest) if r["id"] != target["id"] and is_rule_label(r))
    _relabel(manifest, rule_row["id"], "Battery saver")  # no longer numeric: sent to the model
    handler = fake.make_handler()
    run_rewrite(manifest, out, handler)
    assert [[it["label"] for it in c["items"]] for c in handler.calls] == [["Battery saver"]]
    assert cached(out, [rule_row["id"]])[rule_row["id"]]["model"] == fake.OFFICIAL_ROOT


def test_numeric_rule_is_authoritative_over_the_cache(manifest, tmp_path, monkeypatch):
    out, llm, n_rule = tmp_path / "out", sent_rows(manifest), len(rule_ids(manifest))
    run_rewrite(manifest, out, fake.make_handler())
    target = unique_label_rows(llm)[0]
    real = validate.is_numeric_label
    monkeypatch.setattr(validate, "is_numeric_label", lambda s: real(s) or s == label_of(target))
    handler = fake.make_handler()
    stats = run_rewrite(manifest, out, handler, "--retry-errors")  # the rule now matches a row the model rewrote
    assert handler.calls == [] and cached(out, [target["id"]])[target["id"]]["model"] == rewrite.RULE_MODEL
    assert stats["rewrites"]["rule_drops"] == n_rule + 1 and stats["run"]["rule_drops"] == 1

    monkeypatch.setattr(validate, "is_numeric_label", real)  # the rule no longer matches: the model redoes it
    handler = fake.make_handler()
    stats = run_rewrite(manifest, out, handler)
    assert [it["label"] for c in handler.calls for it in c["items"]] == [label_of(target)]
    assert cached(out, [target["id"]])[target["id"]]["model"] == fake.OFFICIAL_ROOT
    assert stats["rewrites"]["rule_drops"] == n_rule


def test_json_schema_mode_sends_the_response_schema(manifest, tmp_path):
    handler = fake.make_handler()
    stats = run_rewrite(manifest, tmp_path / "out", handler, "--response-format", "json_schema", "--model-label",
                        "nvidia/nemotron-3.5-lightning-30b-a3b (local MLX)")
    rf = handler.calls[0]["body"]["response_format"]
    assert rf == {"type": "json_schema", "json_schema": {"name": "rewrites", "schema": prompts.RESPONSE_SCHEMA}}
    assert stats["run"]["response_format_final"] == "json_schema"
    assert {r["model"] for r in read_jsonl(tmp_path / "out" / "rewrites.jsonl")} == \
        {"nvidia/nemotron-3.5-lightning-30b-a3b (local MLX)", rewrite.RULE_MODEL}


def test_method_md_is_deterministic_and_complete(manifest, tmp_path):
    out = tmp_path / "out"
    stats = json.loads(json.dumps(run_rewrite(manifest, out, fake.make_handler(), "--dry-run")))
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    assert method.write_method(a, stats).read_bytes() == method.write_method(b, stats).read_bytes()
    text = (a / "METHOD.md").read_text(encoding="utf-8")
    example = method.example_meta()
    assert (example["label"], example["control"], example["position"], example["size"]) == \
        ("More options", "button", "top-right", "small")
    assert example["neighbors"] == [{"label": "Search", "target_is": "right of"},
                                    {"label": "Inbox", "target_is": "above"}]
    for needle in (prompts.SYSTEM_PROMPT, *(m["content"] for m in prompts.FEW_SHOT), f"prompt {PV}",
                   'int(hashlib.sha256(row["id"].encode("utf-8")).hexdigest(), 16) % 100 < 60',
                   '"enable_thinking": false', "never sees the screenshot", stats["manifest"]["sha256"],
                   stats["run"]["command"], "OS-Atlas", "Apache-2.0", "limited to",
                   prompts.format_batch([example]), '"target_is":"right of"', method.control_table(), "`element`",
                   "U+200E", f"longer than {prompts.LABEL_MAX_CHARS} characters", rewrite.RULE_MODEL,
                   *method.numeric_rule_blocks(), "Output breaker", "Throughput over all",
                   *(f"`{code}`" for code, _ in method.ERROR_DOCS), *(f"`{w}`" for w, _ in method.WARNING_DOCS)):
        assert needle in text, needle
    assert '"direction"' not in text
    for noun in set(meta.CONTROL_NOUNS.values()):
        assert f"`{noun}`" in method.control_table()


def test_method_docs_name_real_error_codes():
    """Every documented error/warning code is produced somewhere (validate.py, or rewrite.py for truncated)."""
    source = inspect.getsource(validate) + inspect.getsource(rewrite)
    for code, _ in [*method.ERROR_DOCS, *method.WARNING_DOCS]:
        name = code.split(":")[0]
        if name in ("invented_color", "invented_shape"):  # built as f"invented_{kind}:..."
            assert 'f"invented_{kind}:' in source and f'("{name.split("_")[1]}", ' in source
        else:
            assert f'"{name}' in source or f"'{name}" in source, code
    if hasattr(validate, "NUMERIC_LABEL_RE"):
        assert any(validate.NUMERIC_LABEL_RE.pattern in block for block in method.numeric_rule_blocks())


def test_method_md_lists_every_generator_when_records_are_mixed(manifest, tmp_path):
    out, llm = tmp_path / "out", sent_rows(manifest)
    run_rewrite(manifest, out, fake.make_handler())  # local official checkpoint
    _relabel(manifest, unique_label_rows(llm)[0]["id"], "Download statement")
    stats = run_rewrite(manifest, out, fake.make_handler(served_root=None), endpoint="hosted")  # 1 row via hosted
    rw, hosted = stats["rewrites"], f"NVIDIA-hosted API, {fake.MODEL}"
    n = len(llm) - 1
    assert rw["llm_models"] == {fake.OFFICIAL_ROOT: n, hosted: 1} and rw["endpoints"] == {"hosted": 1, "local": n}
    assert rw["served_model_roots"] == {fake.OFFICIAL_ROOT: n, "(not reported)": 1}
    text = (out / "METHOD.md").read_text(encoding="utf-8")
    assert f"`{fake.OFFICIAL_ROOT}` ({n} rows); `{hosted}` (1 row)" in text
    assert f"`local` ({n} rows); `hosted` (1 row)" in text
    assert f"Rewrites were generated by `{fake.OFFICIAL_ROOT}` ({n} rows), `{hosted}` (1 row)" in text
    assert "| endpoint | hosted (`https://integrate.api.nvidia.com/v1`)" in text  # the invocation, shown separately
    assert len(stats["runs"]) == 2 and stats["throughput"]["rows"] == len(llm) + 1  # the relabeled row was done twice


def test_cli_runs_as_a_script_from_the_repo_root(manifest, tmp_path):
    cmd = [sys.executable, "data_prep/rewrite/rewrite.py", "--manifest", str(manifest), "--out", str(tmp_path / "o"),
           "--endpoint", "local", "--show-prompt", "--batch-size", "2"]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)[0]["content"] == prompts.SYSTEM_PROMPT


def test_linux_macos_rows_are_selected_and_rewritten(tmp_path):
    """Eval option C: OS-Atlas Linux/macOS are raw-label training sources like the others."""
    manifest = synth.make_manifest(tmp_path / "desk", n_images=6, per_image=6, write_images=False,
                                   sources=("os_atlas/linux", "os_atlas/macos"))
    sel = selected(manifest)
    assert {r["source"] for r in sel} == {"os_atlas/linux", "os_atlas/macos"}
    stats = run_rewrite(manifest, tmp_path / "out", fake.make_handler())
    assert stats["exit_code"] == 0 and stats["manifest"]["targets"] == len(sel)
    assert set(stats["rewrites"]["by_source"]) == {"os_atlas/linux", "os_atlas/macos"}
    assert {r["source"] for r in read_jsonl(tmp_path / "out" / "rewrites.jsonl")} == {"os_atlas/linux", "os_atlas/macos"}
    assert stats["run"]["raw_sources"] == sorted(meta.RAW_LABEL_SOURCES)


# --- served-checkpoint provenance (only the official NVIDIA checkpoint or NVIDIA's hosted API may be reported) ---

@pytest.mark.parametrize("root,models_status", [(fake.COMMUNITY_ROOT, 200), ("someone/Lightning-30B-gguf", 200),
                                                ("nvidia/Lightning-AWQ", 200), ("/Users/me/models/lightning", 200),
                                                (None, 200), (fake.OFFICIAL_ROOT, 404)])
def test_unofficial_or_unverified_checkpoint_is_refused(manifest, tmp_path, capsys, root, models_status):
    out = tmp_path / "out"
    handler = fake.make_handler(served_root=root, models_status=models_status)
    stats = run_rewrite(manifest, out, handler)
    assert stats["exit_code"] == 3 and "--allow-unofficial-model" in stats["error"]
    assert handler.calls == [] and len(handler.model_calls) == 1
    assert not (out / "rewrites.jsonl").exists() and not (out / "METHOD.md").exists()
    assert cached(out, [r["id"] for r in selected(manifest)]) == {}  # refused before anything, rule drops included
    assert fake.OFFICIAL_ROOT in stats["error"] and "--allow-unofficial-model" in capsys.readouterr().err
    args = ["--manifest", str(manifest), "--out", str(out), "--endpoint", "h200"]
    assert asyncio.run(rewrite.run(rewrite.build_parser().parse_args(args), transport=httpx.MockTransport(handler),
                                   now=lambda: TS))["exit_code"] == 3


def test_allow_unofficial_model_marks_outputs_and_never_leaks_into_an_official_run(manifest, tmp_path):
    out, llm = tmp_path / "out", sent_rows(manifest)
    plumb = fake.make_handler(served_root=fake.COMMUNITY_ROOT)
    stats = run_rewrite(manifest, out, plumb, "--allow-unofficial-model")
    assert stats["exit_code"] == 0 and plumb.calls
    run = stats["run"]
    assert run["unofficial_model"] is True and run["unofficial_roots"] == [fake.COMMUNITY_ROOT]
    assert run["served_model_root"] == run["model_label"] == fake.COMMUNITY_ROOT
    text = (out / "METHOD.md").read_text(encoding="utf-8")
    assert text.startswith(f"**NOT FOR PUBLICATION: generated with an unofficial/community checkpoint "
                           f"({fake.COMMUNITY_ROOT}); plumbing only**")
    assert {r["model"] for r in read_jsonl(out / "rewrites.jsonl")} == {fake.COMMUNITY_ROOT, rewrite.RULE_MODEL}
    assert all(r["unofficial_model"] != rewrite.is_rule(r) for r in cached(out).values())
    assert stats["runs"][0]["unofficial_model"] is True

    # Same cache, official checkpoint, no flag: every plumbing row is stale and is re-generated; rule drops (no model)
    # are reused.
    official = fake.make_handler()
    stats = run_rewrite(manifest, out, official)
    assert stats["exit_code"] == 0 and stats["run"]["unofficial_model"] is False
    assert sum(len(c["items"]) for c in official.calls) == len(llm)
    assert {r["model"] for r in read_jsonl(out / "rewrites.jsonl")} == {fake.OFFICIAL_ROOT, rewrite.RULE_MODEL}
    assert not any(r["unofficial_model"] for r in cached(out).values())
    assert stats["requests"]["all_runs"]["n"] == stats["requests"]["this_run"]["n"] == len(official.calls)
    assert len(logged(out)) == len(plumb.calls) + len(official.calls)  # the log keeps both, flagged
    assert [r["run_id"] for r in stats["runs"]] == [stats["run"]["run_id"]]  # the plumbing run is not counted
    assert stats["throughput"]["invocations"] == 1 and stats["throughput"]["rows"] == len(llm)
    assert "NOT FOR PUBLICATION" not in (out / "METHOD.md").read_text(encoding="utf-8")


def test_unofficial_server_with_complete_official_cache_sends_nothing_and_stays_publishable(manifest, tmp_path):
    out = tmp_path / "out"
    run_rewrite(manifest, out, fake.make_handler())
    before = (out / "rewrites.jsonl").read_bytes()
    plumb = fake.make_handler(served_root=fake.COMMUNITY_ROOT)
    stats = run_rewrite(manifest, out, plumb)  # nothing pending: no refusal, no requests
    assert stats["exit_code"] == 0 and plumb.calls == [] and stats["run"]["unofficial_model"] is False
    assert (out / "rewrites.jsonl").read_bytes() == before
    assert stats["run"]["served_model_root"] == fake.COMMUNITY_ROOT
    assert "NOT FOR PUBLICATION" not in (out / "METHOD.md").read_text(encoding="utf-8")


def test_hosted_is_official_without_a_root_and_uses_hosted_concurrency(manifest, tmp_path):
    handler = fake.make_handler(served_root=None)
    stats = run_rewrite(manifest, tmp_path / "out", handler, endpoint="hosted")
    run = stats["run"]
    assert stats["exit_code"] == 0 and run["unofficial_model"] is False and run["served_model_root"] is None
    assert run["model_label"] == f"NVIDIA-hosted API, {fake.MODEL}" and run["concurrency"] == 16
    assert stats["rewrites"]["endpoints"] == {"hosted": len(sent_rows(manifest))}
    assert handler.model_calls[0]["url"] == "https://integrate.api.nvidia.com/v1/models"
    assert handler.model_calls[0]["headers"]["authorization"] == f"Bearer {KEY}"
    stats = run_rewrite(manifest, tmp_path / "o2", fake.make_handler(served_root=None), "--concurrency", 3,
                        "--model-label", "custom-label", endpoint="hosted")
    assert stats["run"]["concurrency"] == 3 and stats["run"]["model_label"] == "custom-label"


def test_hosted_endpoint_pointed_at_a_local_server_is_checked(manifest, tmp_path):
    handler = fake.make_handler(served_root=fake.COMMUNITY_ROOT)
    stats = run_rewrite(manifest, tmp_path / "out", handler, "--base-url", "http://127.0.0.1:8000/v1",
                        endpoint="hosted")
    assert stats["exit_code"] == 3 and handler.calls == []


def test_h200_default_concurrency(manifest, tmp_path):
    stats = run_rewrite(manifest, tmp_path / "out", fake.make_handler(), endpoint="h200")
    assert stats["exit_code"] == 0 and stats["run"]["concurrency"] == 8


def test_select_targets_splits_filter():
    rows = [{"id": f"fineweb:p{i}:0", "source": "os_atlas/fineweb", "split": "train" if i % 2 else "dev"}
            for i in range(200)]
    selected, targets = rewrite.select_targets(rows, None, splits=("train",))
    assert selected and {r["split"] for r in selected} == {"train"}
    all_selected, _ = rewrite.select_targets(rows)
    assert len(all_selected) > len(selected) and {r["id"] for r in selected} <= {r["id"] for r in all_selected}


def test_group_retries_reasks_a_failed_batch_in_one_request(manifest, tmp_path):
    out, llm = tmp_path / "out", sent_rows(manifest)
    handler = fake.make_handler(malformed_first=1)
    stats = run_rewrite(manifest, out, handler, "--batch-size", 4, "--concurrency", 1, "--group-retries")
    assert stats["exit_code"] == 0 and stats["pending_rows"] == 0 and stats["run"]["group_retries"] is True
    recs = cached(out, [r["id"] for r in llm])
    assert all(recs[r["id"]]["attempts"] == 2 and not recs[r["id"]]["first_try_valid"] for r in llm[:4])
    assert [len(c["items"]) for c in handler.calls[1:2]] == [4]  # one retry request for the 4 failed rows, k = 1..4
    assert stats["requests"]["this_run"]["by_kind"] == {"batch": math.ceil(len(llm) / 4), "retry": 1}
    assert "together, in one request per failed reply" in (out / "METHOD.md").read_text(encoding="utf-8")


def test_method_md_reports_the_human_audit_and_drop_policy(manifest, tmp_path):
    out = tmp_path / "out"
    stats = run_rewrite(manifest, out, fake.make_handler())
    cards = ([{"kind": "kept", "verdict": "faithful"}] * 42 + [{"kind": "kept", "verdict": "unfaithful"}]
             + [{"kind": "kept", "verdict": "unsure"}] * 6 + [{"kind": "kept", "verdict": None}]
             + [{"kind": "drop", "verdict": "no", "original": "EXPLORER", "drop_reason": "unintelligible"}] * 13
             + [{"kind": "drop", "verdict": "yes"}] * 3)
    (out / "review_verdicts_x.json").write_text(json.dumps(
        {"sheet": {"pv": stats["run"]["prompt_version"], "seed": "0", "runId": "r1"}, "cards": cards}))
    assert method.main(["--out", str(out)]) == 0
    text = (out / "METHOD.md").read_text(encoding="utf-8")
    assert "42/43 = 97.7%" in text and "42/49 = 85.7%" in text and "3 of 16 sampled drops" in text
    assert "lightning_drop_reason" in text and '"EXPLORER" (unintelligible)' in text
