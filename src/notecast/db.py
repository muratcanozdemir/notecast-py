"""SQLite storage — notes, themes (DAG), config, vectors as JSON blobs."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any

DB_PATH_DEFAULT = "./notecast.db"


def _db_path() -> str:
    return os.environ.get("NOTECAST_DB", DB_PATH_DEFAULT)


def connect(path: str | None = None) -> sqlite3.Connection:
    p = path or _db_path()
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── schema ──────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    summary     TEXT,
    topics      TEXT,           -- JSON list[str]
    vector      TEXT,           -- JSON list[float]
    status      TEXT NOT NULL DEFAULT 'pending',
    source_path TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS themes (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    is_base     INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS theme_edges (
    child_id    TEXT NOT NULL REFERENCES themes(id) ON DELETE CASCADE,
    parent_id   TEXT NOT NULL REFERENCES themes(id) ON DELETE CASCADE,
    PRIMARY KEY (child_id, parent_id)
);

CREATE TABLE IF NOT EXISTS note_themes (
    note_id     TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    theme_id    TEXT NOT NULL REFERENCES themes(id) ON DELETE CASCADE,
    PRIMARY KEY (note_id, theme_id)
);

CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_counters (
    name  TEXT PRIMARY KEY,
    count INTEGER NOT NULL DEFAULT 0
);
"""

FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
USING fts5(title, content, summary, content='notes', content_rowid='rowid');

CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, title, content, summary)
    VALUES (new.rowid, new.title, new.content, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content, summary)
    VALUES ('delete', old.rowid, old.title, old.content, old.summary);
    INSERT INTO notes_fts(rowid, title, content, summary)
    VALUES (new.rowid, new.title, new.content, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content, summary)
    VALUES ('delete', old.rowid, old.title, old.content, old.summary);
END;
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.executescript(FTS)
    # seed pipeline counters
    for name in ("classify_runs", "organize_runs", "consolidate_runs"):
        conn.execute(
            "INSERT OR IGNORE INTO pipeline_counters(name, count) VALUES (?, 0)",
            (name,),
        )
    conn.commit()


# ── helpers ─────────────────────────────────────────────────────────

def new_id() -> str:
    return uuid.uuid4().hex[:12]


# ── notes ───────────────────────────────────────────────────────────

@dataclass
class Note:
    id: str
    title: str
    content: str
    summary: str | None = None
    topics: list[str] = field(default_factory=list)
    vector: list[float] = field(default_factory=list)
    status: str = "pending"
    source_path: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Note:
        return cls(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            summary=row["summary"],
            topics=json.loads(row["topics"]) if row["topics"] else [],
            vector=json.loads(row["vector"]) if row["vector"] else [],
            status=row["status"],
            source_path=row["source_path"],
        )


def add_note(conn: sqlite3.Connection, title: str, content: str,
             source_path: str | None = None) -> str:
    nid = new_id()
    conn.execute(
        "INSERT INTO notes(id, title, content, source_path) VALUES (?,?,?,?)",
        (nid, title, content, source_path),
    )
    conn.commit()
    return nid


def get_note(conn: sqlite3.Connection, nid: str) -> Note | None:
    row = conn.execute("SELECT * FROM notes WHERE id=?", (nid,)).fetchone()
    return Note.from_row(row) if row else None


def _fts_escape(query: str) -> str:
    """Turn free-text input into a literal FTS5 match expression.

    User queries may contain FTS5 operators/syntax (colons, quotes,
    AND/OR/NOT, unbalanced parens) that would otherwise raise a syntax
    error from SQLite. Quoting each token neutralizes all of that.
    """
    tokens = query.split()
    if not tokens:
        return '""'
    return " ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def search_notes(conn: sqlite3.Connection, query: str) -> list[Note]:
    rows = conn.execute(
        "SELECT n.* FROM notes n JOIN notes_fts f ON n.rowid=f.rowid "
        "WHERE notes_fts MATCH ? ORDER BY rank",
        (_fts_escape(query),),
    ).fetchall()
    return [Note.from_row(r) for r in rows]


def list_notes(conn: sqlite3.Connection, status: str | None = None) -> list[Note]:
    if status:
        rows = conn.execute(
            "SELECT * FROM notes WHERE status=? ORDER BY created_at", (status,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM notes ORDER BY created_at").fetchall()
    return [Note.from_row(r) for r in rows]


def update_note(conn: sqlite3.Connection, nid: str, **fields: Any) -> None:
    allowed = {"summary", "topics", "vector", "status", "title", "content"}
    sets, vals = [], []
    for k, v in fields.items():
        if k not in allowed:
            continue
        if k in ("topics", "vector"):
            v = json.dumps(v)
        sets.append(f"{k}=?")
        vals.append(v)
    if not sets:
        return
    sets.append("updated_at=datetime('now')")
    vals.append(nid)
    conn.execute(f"UPDATE notes SET {','.join(sets)} WHERE id=?", vals)
    conn.commit()


def delete_note(conn: sqlite3.Connection, nid: str) -> None:
    conn.execute("DELETE FROM notes WHERE id=?", (nid,))
    conn.commit()


def status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT status, count(*) as cnt FROM notes GROUP BY status"
    ).fetchall()
    return {r["status"]: r["cnt"] for r in rows}


# ── themes ──────────────────────────────────────────────────────────

@dataclass
class Theme:
    id: str
    name: str
    description: str | None
    is_base: bool

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Theme:
        return cls(
            id=row["id"], name=row["name"],
            description=row["description"], is_base=bool(row["is_base"]),
        )


def add_theme(conn: sqlite3.Connection, name: str,
              description: str | None = None, is_base: bool = False,
              parent_id: str | None = None) -> str:
    tid = new_id()
    conn.execute(
        "INSERT INTO themes(id, name, description, is_base) VALUES (?,?,?,?)",
        (tid, name, description, int(is_base)),
    )
    if parent_id:
        conn.execute(
            "INSERT INTO theme_edges(child_id, parent_id) VALUES (?,?)",
            (tid, parent_id),
        )
    conn.commit()
    return tid


def list_themes(conn: sqlite3.Connection) -> list[Theme]:
    rows = conn.execute("SELECT * FROM themes ORDER BY name").fetchall()
    return [Theme.from_row(r) for r in rows]


def get_theme_by_name(conn: sqlite3.Connection, name: str) -> Theme | None:
    row = conn.execute(
        "SELECT * FROM themes WHERE lower(name)=lower(?)", (name,)
    ).fetchone()
    return Theme.from_row(row) if row else None


def get_theme(conn: sqlite3.Connection, tid: str) -> Theme | None:
    row = conn.execute("SELECT * FROM themes WHERE id=?", (tid,)).fetchone()
    return Theme.from_row(row) if row else None


def delete_theme(conn: sqlite3.Connection, tid: str) -> None:
    conn.execute("DELETE FROM themes WHERE id=?", (tid,))
    conn.commit()


def theme_note_count(conn: sqlite3.Connection, tid: str) -> int:
    row = conn.execute(
        "SELECT count(*) as cnt FROM note_themes WHERE theme_id=?", (tid,)
    ).fetchone()
    return row["cnt"]


def get_theme_notes(conn: sqlite3.Connection, tid: str) -> list[Note]:
    rows = conn.execute(
        "SELECT n.* FROM notes n JOIN note_themes nt ON n.id=nt.note_id "
        "WHERE nt.theme_id=?", (tid,)
    ).fetchall()
    return [Note.from_row(r) for r in rows]


def theme_parents(conn: sqlite3.Connection, tid: str) -> list[str]:
    rows = conn.execute(
        "SELECT parent_id FROM theme_edges WHERE child_id=?", (tid,)
    ).fetchall()
    return [r["parent_id"] for r in rows]


def theme_children(conn: sqlite3.Connection, tid: str) -> list[str]:
    rows = conn.execute(
        "SELECT child_id FROM theme_edges WHERE parent_id=?", (tid,)
    ).fetchall()
    return [r["child_id"] for r in rows]


def all_edges(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Return (child_id, parent_id) pairs."""
    rows = conn.execute("SELECT child_id, parent_id FROM theme_edges").fetchall()
    return [(r["child_id"], r["parent_id"]) for r in rows]


# ── note <-> theme ──────────────────────────────────────────────────

def assign_note_theme(conn: sqlite3.Connection, note_id: str, theme_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO note_themes(note_id, theme_id) VALUES (?,?)",
        (note_id, theme_id),
    )
    conn.commit()


def unassign_note_theme(conn: sqlite3.Connection, note_id: str, theme_id: str) -> None:
    conn.execute(
        "DELETE FROM note_themes WHERE note_id=? AND theme_id=?",
        (note_id, theme_id),
    )
    conn.commit()


def note_theme_ids(conn: sqlite3.Connection, note_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT theme_id FROM note_themes WHERE note_id=?", (note_id,)
    ).fetchall()
    return [r["theme_id"] for r in rows]


# ── config ──────────────────────────────────────────────────────────

DEFAULTS: dict[str, str] = {
    "ollama_url": "http://localhost:11434",
    "gen_model": "llama3.2:1b",
    "embed_model": "nomic-embed-text",
    "classify_every": "10",
    "organize_after": "2",
    "consolidate_after": "3",
    "split_threshold": "15",
    "language": "english",
    "context": "",
}


def get_config(conn: sqlite3.Connection, key: str) -> str:
    row = conn.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
    if row:
        return row["value"]
    return DEFAULTS.get(key, "")


def set_config(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO config(key, value) VALUES (?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def all_config(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT key, value FROM config").fetchall()
    merged = dict(DEFAULTS)
    merged.update({r["key"]: r["value"] for r in rows})
    return merged


# ── pipeline counters ───────────────────────────────────────────────

def get_counter(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute(
        "SELECT count FROM pipeline_counters WHERE name=?", (name,)
    ).fetchone()
    return row["count"] if row else 0


def increment_counter(conn: sqlite3.Connection, name: str) -> int:
    conn.execute(
        "UPDATE pipeline_counters SET count=count+1 WHERE name=?", (name,)
    )
    conn.commit()
    return get_counter(conn, name)


def reset_counter(conn: sqlite3.Connection, name: str) -> None:
    conn.execute(
        "UPDATE pipeline_counters SET count=0 WHERE name=?", (name,)
    )
    conn.commit()
