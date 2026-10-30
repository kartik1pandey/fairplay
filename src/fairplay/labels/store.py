"""SQLite store for the private side of the project: usernames, UUIDs and label snapshots.

Lives in private/ (git-ignored). Everything that leaves this file is keyed by account_id (UUID).
Tables are append-only: a snapshot is never overwritten, a later snapshot is a new snapshot_id.
"""

import json
import sqlite3
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_DB = Path("private/labels.sqlite")
BATCH_SIZE = 300  # Lichess POST /api/users accepts up to 300 ids

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    username   TEXT PRIMARY KEY,          -- lowercase Lichess id
    account_id TEXT NOT NULL UNIQUE,      -- random UUID, the only id used outside private/
    added_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    n_accounts  INTEGER NOT NULL,
    note        TEXT
);
CREATE TABLE IF NOT EXISTS batches (
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    batch_no    INTEGER NOT NULL,
    usernames   TEXT NOT NULL,            -- JSON list, frozen when the snapshot is created
    n_ids       INTEGER NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',   -- pending | done | http_error | abandoned
    http_status INTEGER,
    attempts    INTEGER NOT NULL DEFAULT 0,
    fetched_at  TEXT,
    response    TEXT,                     -- minimal projection of the response, for audit
    PRIMARY KEY (snapshot_id, batch_no)
);
CREATE TABLE IF NOT EXISTS labels (
    snapshot_id   TEXT NOT NULL,
    account_id    TEXT NOT NULL REFERENCES accounts(account_id),
    outcome       TEXT NOT NULL,          -- ok | not_returned | http_error
    tos_violation INTEGER,                -- NULL unless outcome = ok
    disabled      INTEGER,
    created_at_ms INTEGER,
    fetched_at    TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, account_id)
);
"""


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: Path = DEFAULT_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")  # a crash mid-write cannot corrupt committed rows
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def add_accounts(conn: sqlite3.Connection, usernames: Iterable[str]) -> int:
    """Give each new username a random UUID. Returns how many were new.

    Random, not a hash: anyone can hash a known username and match it; nobody can invert uuid4.
    """
    added = now_utc()
    before = conn.total_changes
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO accounts VALUES (?, ?, ?)",
            ((u.lower(), str(uuid.uuid4()), added) for u in usernames),
        )
    return conn.total_changes - before


def create_snapshot(
    conn: sqlite3.Connection, snapshot_id: str, usernames: Iterable[str], note: str = ""
) -> int:
    """Freeze the account list into fixed batches. Sorted, so batches are reproducible."""
    names = sorted({u.lower() for u in usernames})
    add_accounts(conn, names)
    with conn:
        conn.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?)", (snapshot_id, now_utc(), len(names), note)
        )
        conn.executemany(
            "INSERT INTO batches (snapshot_id, batch_no, usernames, n_ids) VALUES (?, ?, ?, ?)",
            (
                (snapshot_id, i // BATCH_SIZE, json.dumps(names[i : i + BATCH_SIZE]),
                 len(names[i : i + BATCH_SIZE]))
                for i in range(0, len(names), BATCH_SIZE)
            ),
        )
    return len(names)


def close_snapshot(conn: sqlite3.Connection, snapshot_id: str, reason: str) -> int:
    """Freeze a snapshot: unfinished batches become 'abandoned' and are never fetched, so the
    label set cannot change after results exist. Their accounts get no label row and are
    excluded (counted) downstream. Returns the number of abandoned batches."""
    with conn:
        n = conn.execute(
            "UPDATE batches SET status = 'abandoned' WHERE snapshot_id = ? "
            "AND status IN ('pending', 'http_error')", (snapshot_id,)).rowcount
        conn.execute(
            "UPDATE snapshots SET note = COALESCE(note, '') || ? WHERE snapshot_id = ?",
            (f" | CLOSED: {reason}", snapshot_id))
    return n


def summary(conn: sqlite3.Connection, snapshot_id: str) -> dict:
    batches = dict(
        conn.execute(
            "SELECT status, COUNT(*) FROM batches WHERE snapshot_id = ? GROUP BY status",
            (snapshot_id,),
        ).fetchall()
    )
    outcomes = dict(
        conn.execute(
            "SELECT outcome, COUNT(*) FROM labels WHERE snapshot_id = ? GROUP BY outcome",
            (snapshot_id,),
        ).fetchall()
    )
    flagged, disabled = conn.execute(
        "SELECT COALESCE(SUM(tos_violation), 0), COALESCE(SUM(disabled), 0) FROM labels "
        "WHERE snapshot_id = ? AND outcome = 'ok'",
        (snapshot_id,),
    ).fetchone()
    ok = outcomes.get("ok", 0)
    total = sum(outcomes.values())
    return {
        "snapshot_id": snapshot_id,
        "batches": batches,
        "outcomes": outcomes,
        "flagged": flagged,
        "disabled": disabled,
        "ok_share": round(ok / total, 4) if total else None,
        "flagged_share_of_ok": round(flagged / ok, 4) if ok else None,
    }
