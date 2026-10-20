"""SQLite persistence for MemoryBridge.

Design notes
------------
* One flat table (``memories``) holds the records; the row ``id`` is a short
  UUID hex string so it is safe to hand to an LLM and type back in.
* An FTS5 external-content index (``memories_fts``) mirrors the ``content``
  column and is kept in sync with SQL triggers, so keyword search survives
  inserts, updates and deletes without application bookkeeping.
* Every statement filters by ``owner``; the owner string comes from the
  server configuration (``--user`` / ``MEMORYBRIDGE_USER``), never from a
  tool argument, so one user cannot address another user's rows.
* Connections are created per-thread (SQLite default) and the schema is
  applied idempotently on first use.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

__all__ = ["SCHEMA_SQL", "connect", "ensure_schema"]

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    owner       TEXT NOT NULL,
    content     TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'general',
    source      TEXT NOT NULL DEFAULT 'unknown',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_owner ON memories(owner);
CREATE INDEX IF NOT EXISTS idx_memories_owner_category ON memories(owner, category);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (and create if needed) the database at *db_path*.

    ``:memory:`` is supported for tests. Parent directories are created for
    file-backed databases so first run never fails on a missing folder.
    """
    path_str = str(db_path)
    if path_str != ":memory:":
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path_str, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables, indexes, the FTS index and its sync triggers."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()
