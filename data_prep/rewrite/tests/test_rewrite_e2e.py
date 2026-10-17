"""End-to-end rewrite.py runs against the fake Lightning server (httpx.MockTransport, fake sleep, fixed clock)."""

import asyncio
import hashlib
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


def cached(out, ids=None) -> dict[str, dict]:
    with cache_mod.RewriteCache(Path(out) / "cache.sqlite") as c:
        rows = c.get_many(PV, ids if ids is not None else [r["id"] for r in read_jsonl(Path(out) / "rewrites.jsonl")])
        return rows


def logged(out) -> list[dict]:
    with cache_mod.RewriteCache(Path(out) / "cache.sqlite") as c:
        return c.requests(PV)


def test_happy_path_writes_outputs_with_expected_counts(manifest, tmp_path):
    out, sel = tmp_path / "out", selected(manifest)
    assert sel
    handler = fake.make_handler()
    stats = run_rewrite(manifest, out, handler, "--batch-size", 5)

    assert stats["exit_code"] == 0 and stats["pending_rows"] == 0 and stats["aborted"] is None
    recs = read_jsonl(out / "rewrites.jsonl")
    assert [r["id"] for r in recs] == [r["id"] for r in sel]
    for rec, row in zip(recs, sel):
        assert list(rec) == list(rewrite.PUBLIC_FIELDS)
        assert rec["original"] == row["instruction"] and rec["source"] == row["source"]
        assert rec["style"] == meta.assign_style(row["id"]) and rec["prompt_version"] == PV
        assert rec["model"] == fake.OFFICIAL_ROOT and rec["ts"] == TS  # --model-label defaults to the served root
        assert (rec["rewritten"] is not None) == rec["keep"]
        assert rec["keep"] or rec["drop_reason"] in validate.DROP_REASONS
        if fake.NUMERIC.fullmatch(row["instruction"]):
            assert (rec["keep"], rec["drop_reason"]) == (False, "numeric")
    assert len(handler.calls) == math.ceil(len(sel) / 5)
    sent = {it["label"] for c in handler.calls for it in c["items"]}
    assert sent == {meta.normalize_label(r["instruction"]) for r in sel}
    body = handler.calls[0]["body"]
    assert body["model"] == "lightning" and body["temperature"] == 0.7 and body["top_p"] == 0.95
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0] == {"role": "system", "content": prompts.SYSTEM_PROMPT}

    rw, q = stats["rewrites"], stats["quality"]
    assert rw["total"] == len(sel) == rw["kept"] + rw["dropped"] == stats["manifest"]["targets"]
    assert sum(s["targets"] for s in rw["by_source"].values()) == len(sel)
    assert sum(rw["drop_reasons"].values()) == rw["dropped"]
    assert q["first_try_valid_rate"] == 1.0 and q["batch_json_ok_rate"] == 1.0 and q["rows_retried"] == 0
    assert set(stats["gate"]) >= {"first_try_valid_ok", "drop_rate_ok"} and stats["gate"]["first_try_valid_ok"]
    this = stats["requests"]["this_run"]
    assert this["n"] == len(handler.calls) == len(logged(out)) and this["by_kind"] == {"batch": len(handler.calls)}
    assert this["prompt_tokens"] == 900 * len(handler.calls) and this["total_tokens"] == 1020 * len(handler.calls)
    assert stats["manifest"]["rows"] == len(read_jsonl(manifest))
    assert stats["manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert stats["run"]["models_reported"] == [fake.MODEL] and stats["run"]["prompt_version"] == PV
    assert stats["run"]["served_model_root"] == fake.OFFICIAL_ROOT and stats["run"]["unofficial_model"] is False
    assert stats["run"]["concurrency"] == 4 and len(handler.model_calls) == 1  # local default; one GET /models
    assert stats["rewrites"]["unofficial_records"] == 0
    assert stats["rewrites"]["served_model_roots"] == {fake.OFFICIAL_ROOT: len(sel)}
    assert set(stats) >= {"run", "manifest", "rewrites", "quality", "requests", "throughput", "pending_rows", "gate"}
    assert json.loads((out / "stats.json").read_text()) == json.loads(json.dumps(stats))
    text = (out / "METHOD.md").read_text(encoding="utf-8")
    assert prompts.SYSTEM_PROMPT in text and '% 100 < 60' in text
    assert all(m["content"] in text for m in prompts.FEW_SHOT)
    entry = cached(out)[sel[0]["id"]]
    assert entry["attempts"] == 1 and entry["first_try_valid"] and entry["endpoint"] == "local"
    assert entry["served_model_root"] == fake.OFFICIAL_ROOT and entry["unofficial_model"] is False
    assert "NOT FOR PUBLICATION" not in text


def test_resume_makes_zero_requests_and_identical_output(manifest, tmp_path):
    out = tmp_path / "out"
    first = run_rewrite(manifest, out, fake.make_handler(), "--batch-size", 4)
    before = (out / "rewrites.jsonl").read_bytes()
    again = fake.make_handler()
    second = run_rewrite(manifest, out, again, "--batch-size", 4)
    assert again.calls == [] and second["exit_code"] == 0
    assert (out / "rewrites.jsonl").read_bytes() == before
    assert second["requests"]["this_run"]["n"] == 0
    assert second["requests"]["all_runs"]["n"] == first["requests"]["this_run"]["n"]
    assert second["quality"] == first["quality"]


def test_malformed_batch_rows_are_requeued_individually(manifest, tmp_path):
    out, sel = tmp_path / "out", selected(manifest)
    handler = fake.make_handler(malformed_first=1)
    stats = run_rewrite(manifest, out, handler, "--batch-size", 4, "--concurrency", 1)
    assert stats["exit_code"] == 0 and stats["pending_rows"] == 0
    first4 = [r["id"] for r in sel[:4]]
    recs = cached(out, [r["id"] for r in sel])
    assert all(recs[i]["attempts"] == 2 and not recs[i]["first_try_valid"] for i in first4)
    assert all(recs[i]["errors"][0].startswith("invalid_json") for i in first4)
    assert all(recs[r["id"]]["attempts"] == 1 for r in sel[4:])
    assert [len(c["items"]) for c in handler.calls[1:5]] == [1, 1, 1, 1]  # retries jump the queue
    q, this = stats["quality"], stats["requests"]["this_run"]
    assert q["rows_retried"] == len(first4)
    assert q["first_try_valid_rate"] == round((len(sel) - 4) / len(sel), 4)
    n_batches = math.ceil(len(sel) / 4)
    assert q["batch_json_ok_rate"] == round((n_batches - 1) / n_batches, 4)
    assert this["by_kind"] == {"batch": n_batches, "single": 4} and this["by_status"]["invalid_json"] == 1
    assert logged(out)[0]["sample_text"] == fake.MALFORMED


def test_missing_item_is_requeued(manifest, tmp_path):
    out, sel = tmp_path / "out", selected(manifest)
    stats = run_rewrite(manifest, out, fake.make_handler(drop_k=2), "--batch-size", 5)
    recs = cached(out, [r["id"] for r in sel])
    second_of_batch = [sel[i]["id"] for i in range(1, len(sel), 5)]
    assert all(recs[i]["attempts"] == 2 and recs[i]["errors"] == ["missing"] for i in second_of_batch)
    assert sum(r["attempts"] == 2 for r in recs.values()) == len(second_of_batch)
    assert stats["requests"]["this_run"]["by_status"]["partial"] == len(second_of_batch)
    assert stats["exit_code"] == 0


def test_truncated_reply_counts_as_invalid(manifest, tmp_path):
    out, sel = tmp_path / "out", selected(manifest)
    stats = run_rewrite(manifest, out, fake.make_handler(length_first=1), "--batch-size", 4, "--concurrency", 1)
    recs = cached(out, [r["id"] for r in sel])
    assert all(recs[r["id"]]["attempts"] == 2 for r in sel[:4])
    assert logged(out)[0]["status"] == "invalid_json" and logged(out)[0]["finish_reason"] == "length"
    assert stats["exit_code"] == 0


def test_persistently_invalid_row_becomes_llm_error_after_three_attempts(manifest, tmp_path):
    out, sel = tmp_path / "out", selected(manifest)
    labels = [meta.normalize_label(r["instruction"]) for r in sel]
    victim = next(r for r, lab in zip(sel, labels) if labels.count(lab) == 1 and not fake.NUMERIC.fullmatch(lab))
    label = meta.normalize_label(victim["instruction"])
    handler = fake.make_handler(invalid_labels={label})
    stats = run_rewrite(manifest, out, handler, "--batch-size", 5)

    with_victim = [c for c in handler.calls if any(it["label"] == label for it in c["items"])]
    assert len(with_victim) == 3 and [len(c["items"]) for c in with_victim[1:]] == [1, 1]
    rec = cached(out, [victim["id"]])[victim["id"]]
    assert (rec["keep"], rec["rewritten"], rec["drop_reason"]) == (False, None, validate.LLM_ERROR)
    assert rec["attempts"] == 3 and len(rec["errors"]) == 3 and not rec["first_try_valid"]
    public = {r["id"]: r for r in read_jsonl(out / "rewrites.jsonl")}[victim["id"]]
    assert public["drop_reason"] == validate.LLM_ERROR
    assert stats["quality"]["llm_error_rows"] == 1 and stats["rewrites"]["drop_reasons"][validate.LLM_ERROR] == 1
    assert stats["exit_code"] == 0 and stats["pending_rows"] == 0

    idle = fake.make_handler()
    assert run_rewrite(manifest, out, idle, "--batch-size", 5)["exit_code"] == 0 and idle.calls == []
    healed = fake.make_handler()
    stats = run_rewrite(manifest, out, healed, "--batch-size", 5, "--retry-errors")
    assert len(healed.calls) == 1 and [it["label"] for it in healed.calls[0]["items"]] == [label]
    assert cached(out, [victim["id"]])[victim["id"]]["drop_reason"] != validate.LLM_ERROR
    assert stats["quality"]["llm_error_rows"] == 0


@pytest.mark.parametrize("status", [429, 503])
def test_rate_limit_and_5xx_bursts_recover(manifest, tmp_path, status):
    out, sel, sleeps = tmp_path / "out", selected(manifest), []
    handler = fake.make_handler(fail_first=3, fail_status=status, retry_after=1)
    stats = run_rewrite(manifest, out, handler, "--batch-size", 5, sleeps=sleeps)
    n_batches = math.ceil(len(sel) / 5)
    assert stats["exit_code"] == 0 and stats["pending_rows"] == 0
    assert len(handler.calls) == n_batches + 3 and len(sleeps) == 3 and min(sleeps) >= 1
    this = stats["requests"]["this_run"]
    assert this["n"] == n_batches and this["http_attempts"] == n_batches + 3 and this["retries"] == 3
    assert stats["quality"]["first_try_valid_rate"] == 1.0


def test_fatal_401_aborts_without_crashing_or_leaking_the_key(manifest, tmp_path, capsys):
    out, sel = tmp_path / "out", selected(manifest)
    handler = fake.make_handler("unauthorized")
    stats = run_rewrite(manifest, out, handler, "--batch-size", 3, "--concurrency", 2, endpoint="hosted")
    assert stats["exit_code"] != 0 and "401" in stats["aborted"]
    assert 1 <= len(handler.calls) <= 2
    assert handler.calls[0]["headers"]["authorization"] == f"Bearer {KEY}"
    assert handler.calls[0]["body"]["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert (out / "rewrites.jsonl").read_text() == "" and stats["pending_rows"] == len(sel)
    assert cached(out, [r["id"] for r in sel]) == {}
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
    out, sel = tmp_path / "out", selected(manifest)
    handler = fake.make_handler("offline")
    stats = run_rewrite(manifest, out, handler, "--batch-size", 2, "--concurrency", 2,
                        "--max-consecutive-transport-errors", 3)
    assert stats["exit_code"] == 2 and "consecutive transport errors" in stats["aborted"]
    assert stats["pending_rows"] == len(sel) and (out / "rewrites.jsonl").read_text() == ""
    assert cached(out, [r["id"] for r in sel]) == {}
    statuses = stats["requests"]["this_run"]["by_status"]
    assert list(statuses) == ["transport_error"] and 3 <= statuses["transport_error"] < math.ceil(len(sel) / 2)

    stats = run_rewrite(manifest, out, fake.make_handler(), "--batch-size", 2)
    assert stats["exit_code"] == 0 and stats["pending_rows"] == 0
    assert [r["id"] for r in read_jsonl(out / "rewrites.jsonl")] == [r["id"] for r in sel]


def test_partial_outage_keeps_failed_batch_pending_until_rerun(manifest, tmp_path):
    out, sel = tmp_path / "out", selected(manifest)
    handler = fake.make_handler(fail_first=6, fail_status=503)  # the client's 6 attempts all fail for batch 1
    stats = run_rewrite(manifest, out, handler, "--batch-size", 4, "--concurrency", 1)
    assert stats["exit_code"] == 2 and stats["aborted"] is None and stats["pending_rows"] == 4
    assert [r["id"] for r in read_jsonl(out / "rewrites.jsonl")] == [r["id"] for r in sel[4:]]
    assert stats["requests"]["this_run"]["by_status"]["transport_error"] == 1

    rerun = fake.make_handler()
    stats = run_rewrite(manifest, out, rerun, "--batch-size", 4)
    assert len(rerun.calls) == 1 and [it["label"] for it in rerun.calls[0]["items"]] == \
        [meta.normalize_label(r["instruction"]) for r in sel[:4]]
    assert stats["exit_code"] == 0 and stats["pending_rows"] == 0
    assert stats["requests"]["all_runs"]["by_status"]["transport_error"] == 1


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
    assert len(handler.calls) == 10 and stats["exit_code"] == 0

    small = run_rewrite(manifest, tmp_path / "dry", fake.make_handler(), "--limit", 7)
    assert small["requests"]["this_run"]["n"] == 0  # the 7 are a prefix of the 200, already cached
    assert set(r["id"] for r in read_jsonl(tmp_path / "dry" / "rewrites.jsonl")) <= first200
    assert small["manifest"]["targets"] == 7


def test_show_prompt_prints_first_batch_without_requests(manifest, tmp_path, capsys):
    out, sel = tmp_path / "out", selected(manifest)
    handler = fake.make_handler()
    stats = run_rewrite(manifest, out, handler, "--show-prompt", "--batch-size", 3)
    assert stats["exit_code"] == 0 and handler.calls == [] and not out.exists()
    messages = json.loads(capsys.readouterr().out)
    by_image = meta.index_by_image(read_jsonl(manifest))
    assert messages == prompts.build_messages([meta.row_meta(r, by_image) for r in sel[:3]])
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


def test_changed_label_invalidates_only_that_cached_row(manifest, tmp_path):
    out, sel = tmp_path / "out", selected(manifest)
    run_rewrite(manifest, out, fake.make_handler())
    rows = read_jsonl(manifest)
    target = next(r for r in rows if r["id"] == sel[0]["id"])
    target["instruction"] = "Download statement"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    handler = fake.make_handler()
    stats = run_rewrite(manifest, out, handler)
    assert [[it["label"] for it in c["items"]] for c in handler.calls] == [["Download statement"]]
    assert read_jsonl(out / "rewrites.jsonl")[0]["original"] == "Download statement" and stats["exit_code"] == 0


def test_json_schema_mode_sends_the_response_schema(manifest, tmp_path):
    handler = fake.make_handler()
    stats = run_rewrite(manifest, tmp_path / "out", handler, "--response-format", "json_schema", "--model-label",
                        "nvidia/nemotron-3.5-lightning-30b-a3b (local MLX)")
    rf = handler.calls[0]["body"]["response_format"]
    assert rf == {"type": "json_schema", "json_schema": {"name": "rewrites", "schema": prompts.RESPONSE_SCHEMA}}
    assert stats["run"]["response_format_final"] == "json_schema"
    assert {r["model"] for r in read_jsonl(tmp_path / "out" / "rewrites.jsonl")} == \
        {"nvidia/nemotron-3.5-lightning-30b-a3b (local MLX)"}


def test_method_md_is_deterministic_and_complete(manifest, tmp_path):
    out = tmp_path / "out"
    stats = json.loads(json.dumps(run_rewrite(manifest, out, fake.make_handler(), "--dry-run")))
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    assert method.write_method(a, stats).read_bytes() == method.write_method(b, stats).read_bytes()
    text = (a / "METHOD.md").read_text(encoding="utf-8")
    for needle in (prompts.SYSTEM_PROMPT, *(m["content"] for m in prompts.FEW_SHOT), f"prompt {PV}",
                   'int(hashlib.sha256(row["id"].encode("utf-8")).hexdigest(), 16) % 100 < 60',
                   '"enable_thinking": false', "never sees the screenshot", stats["manifest"]["sha256"],
                   stats["run"]["command"], "OS-Atlas", "Apache-2.0", "limited to"):
        assert needle in text, needle


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
    assert fake.OFFICIAL_ROOT in stats["error"] and "--allow-unofficial-model" in capsys.readouterr().err
    args = ["--manifest", str(manifest), "--out", str(out), "--endpoint", "h200"]
    assert asyncio.run(rewrite.run(rewrite.build_parser().parse_args(args), transport=httpx.MockTransport(handler),
                                   now=lambda: TS))["exit_code"] == 3


def test_allow_unofficial_model_marks_outputs_and_never_leaks_into_an_official_run(manifest, tmp_path):
    out, sel = tmp_path / "out", selected(manifest)
    plumb = fake.make_handler(served_root=fake.COMMUNITY_ROOT)
    stats = run_rewrite(manifest, out, plumb, "--allow-unofficial-model")
    assert stats["exit_code"] == 0 and plumb.calls
    run = stats["run"]
    assert run["unofficial_model"] is True and run["unofficial_roots"] == [fake.COMMUNITY_ROOT]
    assert run["served_model_root"] == run["model_label"] == fake.COMMUNITY_ROOT
    text = (out / "METHOD.md").read_text(encoding="utf-8")
    assert text.startswith(f"**NOT FOR PUBLICATION: generated with an unofficial/community checkpoint "
                           f"({fake.COMMUNITY_ROOT}); plumbing only**")
    assert {r["model"] for r in read_jsonl(out / "rewrites.jsonl")} == {fake.COMMUNITY_ROOT}
    assert all(r["unofficial_model"] for r in cached(out).values())

    # Same cache, official checkpoint, no flag: every plumbing row is stale and is re-generated.
    official = fake.make_handler()
    stats = run_rewrite(manifest, out, official)
    assert stats["exit_code"] == 0 and stats["run"]["unofficial_model"] is False
    assert sum(len(c["items"]) for c in official.calls) == len(sel)
    assert {r["model"] for r in read_jsonl(out / "rewrites.jsonl")} == {fake.OFFICIAL_ROOT}
    assert not any(r["unofficial_model"] for r in cached(out).values())
    assert stats["requests"]["all_runs"]["n"] == stats["requests"]["this_run"]["n"] == len(official.calls)
    assert len(logged(out)) == len(plumb.calls) + len(official.calls)  # the log keeps both, flagged
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
