"""Lightweight SQLite history — turns the tool into a workspace with a record trail.

One row per analysis: enough metadata to list/filter/re-open/delete, plus the full
result JSON so a past analysis re-renders without re-running the model. Demo-grade
persistence (single local file), deliberately not a multi-tenant Postgres — see the
MVP roadmap. No external dependency; standard-library sqlite3 only.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "finwise.db"

_COLUMNS = ("id", "ts", "filename", "doc_type", "source", "sample", "model",
            "mode", "status", "confidence", "needs_review", "cost_usd", "latency_s")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                filename TEXT,
                doc_type TEXT,
                source TEXT,          -- 'upload' | 'sample'
                sample TEXT,          -- sample stem (re-runnable) or NULL
                model TEXT,
                mode TEXT,            -- fast | balanced | accurate | free
                status TEXT,          -- ok | review | error
                confidence REAL,      -- 0-1 mean field confidence
                needs_review INTEGER,
                cost_usd REAL,
                latency_s REAL,
                payload TEXT          -- full JSON result for re-render
            )""")


def save(record: dict) -> int:
    """Persist one analysis; `payload` is the full /api/analyze response dict."""
    init_db()
    payload = json.dumps(record.get("payload", {}), ensure_ascii=False)
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO analyses
               (ts, filename, doc_type, source, sample, model, mode, status,
                confidence, needs_review, cost_usd, latency_s, payload)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (record.get("ts") or datetime.now().isoformat(timespec="seconds"),
             record.get("filename"), record.get("doc_type"), record.get("source"),
             record.get("sample"), record.get("model"), record.get("mode"),
             record.get("status"), record.get("confidence"), record.get("needs_review"),
             record.get("cost_usd"), record.get("latency_s"), payload))
        return int(cur.lastrowid)


def list_records(doc_type: Optional[str] = None, status: Optional[str] = None,
                 limit: int = 200) -> list[dict]:
    init_db()
    q = f"SELECT {', '.join(_COLUMNS)} FROM analyses"
    where, args = [], []
    if doc_type:
        where.append("doc_type = ?"); args.append(doc_type)
    if status:
        where.append("status = ?"); args.append(status)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY id DESC LIMIT ?"; args.append(limit)
    with _conn() as c:
        return [dict(r) for r in c.execute(q, args).fetchall()]


def get(rec_id: int) -> Optional[dict]:
    init_db()
    with _conn() as c:
        r = c.execute("SELECT * FROM analyses WHERE id = ?", (rec_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["payload"] = json.loads(d["payload"]) if d.get("payload") else {}
    return d


def delete(rec_id: int) -> bool:
    init_db()
    with _conn() as c:
        cur = c.execute("DELETE FROM analyses WHERE id = ?", (rec_id,))
        return cur.rowcount > 0


def clear() -> None:
    init_db()
    with _conn() as c:
        c.execute("DELETE FROM analyses")
