"""Online state database (design §11): Postgres in deployment, SQLite for tests and local runs.

One SQL dialect for both: ON CONFLICT upserts and RETURNING exist in SQLite >= 3.35 and Postgres,
so every correctness rule (idempotency, version guards, leases) is a single atomic statement and
is tested on SQLite in CI. Timestamps are ISO-8601 UTC text and JSON is text, the common subset.
Writes in SQLite take the write lock up front (BEGIN IMMEDIATE) so two writers never deadlock.
"""

import contextlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_key TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    tombstone INTEGER NOT NULL DEFAULT 0,
    received_at TEXT NOT NULL,
    moves TEXT,
    clocks TEXT
);
CREATE TABLE IF NOT EXISTS player_games (
    account_id TEXT NOT NULL,
    game_key TEXT NOT NULL,
    version INTEGER NOT NULL,
    color TEXT NOT NULL,
    start_ts TEXT NOT NULL,
    tc_base INTEGER NOT NULL,
    tc_inc INTEGER NOT NULL,
    own_elo INTEGER NOT NULL,
    opp_elo INTEGER NOT NULL,
    own_rating_diff INTEGER,
    score REAL NOT NULL,
    termination TEXT NOT NULL,
    ended_in_mate INTEGER NOT NULL,
    event_kind TEXT NOT NULL,
    n_plies INTEGER NOT NULL,
    own_clocks TEXT NOT NULL,
    clock_valid INTEGER NOT NULL,
    PRIMARY KEY (account_id, game_key)
);
CREATE INDEX IF NOT EXISTS player_games_account_ts ON player_games (account_id, start_ts);
CREATE TABLE IF NOT EXISTS accounts (
    account_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS account_features (
    account_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL,
    as_of TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    eligible INTEGER NOT NULL,
    features TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    job_key TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    cutoff TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    lease_owner TEXT,
    lease_until TEXT,
    not_before TEXT NOT NULL,
    last_error TEXT,
    result TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, cutoff);
CREATE TABLE IF NOT EXISTS scores (
    account_id TEXT NOT NULL,
    cutoff TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    score REAL,
    display_score REAL,
    games INTEGER NOT NULL,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (account_id, cutoff, model_version)
);
CREATE TABLE IF NOT EXISTS queues (
    queue_id TEXT PRIMARY KEY,
    cutoff TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    policy TEXT NOT NULL,
    features_as_of TEXT NOT NULL,
    created_at TEXT NOT NULL,
    summary TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS queue_entries (
    queue_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    account_id TEXT NOT NULL,
    score REAL NOT NULL,
    display_score REAL NOT NULL,
    games INTEGER NOT NULL,
    PRIMARY KEY (queue_id, account_id)
);
CREATE TABLE IF NOT EXISTS decisions (
    queue_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    PRIMARY KEY (queue_id, account_id)
);
"""


class Tx:
    """Thin cursor wrapper: `?` placeholders everywhere, converted for Postgres."""

    def __init__(self, cur, postgres: bool):
        self.cur, self.pg = cur, postgres

    def run(self, sql: str, params: tuple = ()) -> "Tx":
        self.cur.execute(sql.replace("?", "%s") if self.pg else sql, params)
        return self

    def all(self) -> list[tuple]:
        return self.cur.fetchall()

    def one(self) -> tuple | None:
        return self.cur.fetchone()

    @property
    def rowcount(self) -> int:
        return self.cur.rowcount


class Database:
    def __init__(self, url: str):
        """url: 'sqlite:///path/to.db' or 'postgresql://user:pw@host/db'."""
        self.url = url
        self.pg = url.startswith("postgresql")
        if not self.pg:
            self.path = Path(url.removeprefix("sqlite:///"))
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.tx() as t:
            for stmt in filter(str.strip, SCHEMA.split(";")):
                t.run(stmt)

    def _connect(self):
        if self.pg:
            import psycopg  # deployment only; tests run on SQLite

            return psycopg.connect(self.url)
        con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA busy_timeout = 30000")
        return con

    @contextlib.contextmanager
    def tx(self, write: bool = True) -> Iterator[Tx]:
        """One transaction: commit on success, roll back on any exception."""
        con = self._connect()
        try:
            cur = con.cursor()
            if not self.pg:
                cur.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            elif not write:  # one snapshot for every read in the transaction
                cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            yield Tx(cur, self.pg)
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()
