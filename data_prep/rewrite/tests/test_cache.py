"""RewriteCache: roundtrip, prompt_version isolation, upsert, persistence, get_many, request log."""

import sqlite3

import pytest

from data_prep.rewrite.cache import RewriteCache


def rec(row_id, **kw):
    return {"id": row_id, "source": "os_atlas/androidworld", "original": "More options", "rewritten": "open more options",
            "style": "functional", "keep": True, "drop_reason": None, "prompt_version": "v1", "model": "m",
            "ts": "2026-10-17T18:00:00+00:00", **kw}


@pytest.fixture
def cache(tmp_path):
    with RewriteCache(tmp_path / "sub" / "cache.sqlite") as c:
        yield c


def test_roundtrip_and_missing(cache):
    r = rec("aw:1:0", warnings=["length"], original="Öffnen 设置 \"quoted\"")
    cache.put("v1", r)
    assert cache.get("v1", "aw:1:0") == r
    assert list(cache.get("v1", "aw:1:0")) == list(r)  # key order preserved
    assert cache.get("v1", "aw:1:1") is None


def test_prompt_version_isolation(cache):
    cache.put("v1", rec("a"))
    cache.put("v2", rec("a", rewritten="the More options button", prompt_version="v2"))
    assert cache.get("v1", "a")["rewritten"] == "open more options"
    assert cache.get("v2", "a")["rewritten"] == "the More options button"
    assert cache.get("v3", "a") is None
    assert cache.get_many("v2", ["a"]) == {"a": cache.get("v2", "a")}


def test_upsert_replaces(cache):
    cache.put("v1", rec("a", keep=False, rewritten=None, drop_reason="llm_error"))
    cache.put("v1", rec("a"))
    assert cache.get("v1", "a")["keep"] is True
    n = cache._db.execute("SELECT COUNT(*) FROM rewrites").fetchone()[0]
    assert n == 1


def test_persistence_across_reopen_and_wal(tmp_path):
    path = tmp_path / "cache.sqlite"
    c = RewriteCache(path)
    c.put("v1", rec("a"))
    c.log_request("v1", "run1", {"kind": "batch", "n_rows": 20})
    c.close()
    with RewriteCache(path) as c2:
        assert c2.get("v1", "a") == rec("a")
        assert [r["kind"] for r in c2.requests("v1")] == ["batch"]
        assert c2._db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_get_many_order_missing_and_chunking(cache):
    ids = [f"fineweb:img{i}:{i % 8}" for i in range(1203)]
    cache.put_many("v1", [rec(i) for i in ids[::2]])
    query = list(reversed(ids)) + ["nope", ids[0]]
    got = cache.get_many("v1", query)
    assert list(got) == [i for i in reversed(ids) if i in set(ids[::2])]
    assert len(got) == 602 and "nope" not in got and got[ids[0]]["id"] == ids[0]
    assert cache.get_many("v1", []) == {}


def test_put_requires_id(cache):
    with pytest.raises(ValueError):
        cache.put("v1", {"source": "x"})
    with pytest.raises(ValueError):
        cache.put_many("v1", [rec("ok"), {"id": ""}])
    assert cache.get("v1", "ok") is None  # failed batch is not partially written


def test_requests_log_filter_by_run_and_version(cache):
    cache.log_request("v1", "r1", {"kind": "batch", "n_rows": 20, "status": "ok"})
    cache.log_request("v1", "r2", {"kind": "single", "n_rows": 1, "status": "partial", "ts": "2026-10-17T19:00:00+00:00"})
    cache.log_request("v1", "r1", {"kind": "single", "n_rows": 1, "status": "invalid_json"})
    cache.log_request("v2", "r1", {"kind": "batch", "n_rows": 5, "status": "ok"})
    all_v1 = cache.requests("v1")
    assert [r["status"] for r in all_v1] == ["ok", "partial", "invalid_json"]
    assert [r["status"] for r in cache.requests("v1", "r1")] == ["ok", "invalid_json"]
    assert [r["n_rows"] for r in cache.requests("v2")] == [5]
    assert cache.requests("v1", "missing") == []
    assert all(r["ts"] and r["run_id"] for r in all_v1)
    assert all_v1[1]["ts"] == "2026-10-17T19:00:00+00:00" and all_v1[1]["run_id"] == "r2"


def test_log_request_does_not_mutate_input(cache):
    record = {"kind": "batch"}
    cache.log_request("v1", "r1", record)
    assert record == {"kind": "batch"}


def test_close_via_context_manager(tmp_path):
    with RewriteCache(tmp_path / "c.sqlite") as c:
        c.put("v1", rec("a"))
    with pytest.raises(sqlite3.ProgrammingError):
        c.get("v1", "a")
