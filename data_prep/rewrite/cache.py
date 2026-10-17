"""Resumable sqlite cache: rewrite records keyed by (prompt_version, id), per-row attempt history, an append-only
request log and one row per invocation (run history)."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS rewrites (
  prompt_version TEXT NOT NULL, id TEXT NOT NULL, record_json TEXT NOT NULL, PRIMARY KEY (prompt_version, id));
CREATE TABLE IF NOT EXISTS requests (prompt_version TEXT NOT NULL, run_id TEXT NOT NULL, ts TEXT NOT NULL, record_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS requests_by_run ON requests (prompt_version, run_id);
CREATE TABLE IF NOT EXISTS row_history (
  prompt_version TEXT NOT NULL, id TEXT NOT NULL, record_json TEXT NOT NULL, PRIMARY KEY (prompt_version, id));
CREATE TABLE IF NOT EXISTS runs (
  prompt_version TEXT NOT NULL, run_id TEXT NOT NULL, started TEXT, finished TEXT, wall_s REAL, rows_done INTEGER,
  rule_drops INTEGER, requests INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER, endpoint TEXT, base_url TEXT,
  model_label TEXT, served_model_root TEXT, unofficial_model INTEGER, command TEXT, PRIMARY KEY (prompt_version, run_id));
"""
RUN_FIELDS = ("run_id", "started", "finished", "wall_s", "rows_done", "rule_drops", "requests", "prompt_tokens",
              "completion_tokens", "endpoint", "base_url", "model_label", "served_model_root", "unofficial_model",
              "command")
CHUNK = 500  # ids per IN (...) query


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _dumps(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False)


class RewriteCache:
    """sqlite3 in WAL mode; every write commits immediately, so a crash loses at most the in-flight request."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self._db.close()

    def _get_many(self, table: str, prompt_version: str, ids) -> dict[str, dict]:
        ids = list(dict.fromkeys(ids))
        found = {}
        for i in range(0, len(ids), CHUNK):
            chunk = ids[i:i + CHUNK]
            q = f"SELECT id, record_json FROM {table} WHERE prompt_version = ? AND id IN ({','.join('?' * len(chunk))})"
            found.update((rid, json.loads(rec)) for rid, rec in self._db.execute(q, (prompt_version, *chunk)))
        return {i: found[i] for i in ids if i in found}

    def _put_many(self, table: str, prompt_version: str, records) -> None:
        rows = []
        for r in records:
            if not r.get("id"):
                raise ValueError(f"{table} record has no 'id': {r!r}"[:200])
            rows.append((prompt_version, str(r["id"]), _dumps(r)))
        with self._db:
            self._db.executemany(f"INSERT OR REPLACE INTO {table} (prompt_version, id, record_json) VALUES (?, ?, ?)",
                                 rows)

    def get(self, prompt_version: str, row_id: str) -> dict | None:
        return self._get_many("rewrites", prompt_version, [row_id]).get(row_id)

    def get_many(self, prompt_version: str, ids) -> dict[str, dict]:
        """Cached records for `ids`, in the order given; ids without a record are absent."""
        return self._get_many("rewrites", prompt_version, ids)

    def put(self, prompt_version: str, record: dict) -> None:
        self.put_many(prompt_version, [record])

    def put_many(self, prompt_version: str, records) -> None:
        """Upsert records (each has "id") in one transaction."""
        self._put_many("rewrites", prompt_version, records)

    def history(self, prompt_version: str, ids) -> dict[str, dict]:
        """Attempt history for `ids`, in the order given: {"id", "original", "source", "unofficial_model", "attempts",
        "errors"}. rewrite.py writes it on every failed LLM attempt, so it survives aborts, resumes and re-sends."""
        return self._get_many("row_history", prompt_version, ids)

    def put_history(self, prompt_version: str, records) -> None:
        """Upsert attempt-history records (each has "id") in one transaction."""
        self._put_many("row_history", prompt_version, records)

    def log_request(self, prompt_version: str, run_id: str, record: dict) -> None:
        """Append one request record; "ts" (UTC ISO8601) and "run_id" are filled in when absent."""
        record = dict(record)
        record.setdefault("ts", _now())
        record.setdefault("run_id", run_id)
        with self._db:
            self._db.execute("INSERT INTO requests (prompt_version, run_id, ts, record_json) VALUES (?, ?, ?, ?)",
                             (prompt_version, run_id, str(record["ts"]), _dumps(record)))

    def requests(self, prompt_version: str, run_id: str | None = None) -> list[dict]:
        """Logged request records for this prompt_version (optionally one run), in insertion order."""
        q, args = "SELECT record_json FROM requests WHERE prompt_version = ?", [prompt_version]
        if run_id is not None:
            q += " AND run_id = ?"
            args.append(run_id)
        return [json.loads(r[0]) for r in self._db.execute(q + " ORDER BY rowid", args)]

    def log_run(self, prompt_version: str, record: dict) -> None:
        """Upsert one invocation's summary by run_id (RUN_FIELDS; fields missing from `record` are stored as NULL)."""
        if not record.get("run_id"):
            raise ValueError(f"run record has no 'run_id': {record!r}"[:200])
        values = {k: record.get(k) for k in RUN_FIELDS}
        values["unofficial_model"] = int(bool(values["unofficial_model"]))
        cols = ("prompt_version", *RUN_FIELDS)
        with self._db:
            self._db.execute(f"INSERT OR REPLACE INTO runs ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                             (prompt_version, *values.values()))

    def runs(self, prompt_version: str) -> list[dict]:
        """Logged invocations for this prompt_version, oldest first (by started, then insertion)."""
        q = f"SELECT {', '.join(RUN_FIELDS)} FROM runs WHERE prompt_version = ? ORDER BY started, rowid"
        out = [dict(zip(RUN_FIELDS, row)) for row in self._db.execute(q, (prompt_version,))]
        for r in out:
            r["unofficial_model"] = bool(r["unofficial_model"])
        return out
