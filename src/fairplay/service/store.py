"""Game history store (design §0.18): upsert by game id with a version, tombstones, 60-day keep.

Every write is one transaction that also bumps the revision of each affected account. Feature
recomputation records the revision it read, and a stored feature row is replaced only by one
computed from the same or a newer revision, so a slow recompute can never overwrite a fresher one
(two concurrent games for one account end in the same state as sequential ones).

The version guard is inside the upsert statement itself (ON CONFLICT ... WHERE version is newer),
so a repeated request is a no-op even when two copies race each other.
"""

import datetime as dt
import hashlib
import json

from fairplay.service.db import Database, Tx
from fairplay.service.schema import GameIn

KEEP_DAYS = 60


def game_key(secret: str, game_id: str) -> str:
    """Same key as the offline cleaner: sha256(secret || game_id), first 16 hex characters."""
    return hashlib.sha256((secret + game_id).encode()).hexdigest()[:16]


def iso(ts: dt.datetime) -> str:
    """Naive-UTC ISO text, the storage format (sorts correctly as text)."""
    if ts.tzinfo is not None:
        ts = ts.astimezone(dt.UTC).replace(tzinfo=None)
    return ts.isoformat(sep=" ", timespec="seconds")


def player_rows(g: GameIn, key: str) -> list[tuple]:
    """Both sides of one game, derived exactly as the offline cleaner does."""
    clock_valid = g.clocks is not None and len(g.clocks) == len(g.moves)
    clocks = g.clocks or []
    rows = []
    for color, own, opp, win, first in (("w", g.white, g.black, "1-0", 0),
                                        ("b", g.black, g.white, "0-1", 1)):
        score = 1.0 if g.result == win else 0.5 if g.result == "1/2-1/2" else 0.0
        rows.append((
            str(own.account_id), key, g.version, color, iso(g.start_ts), g.tc_base, g.tc_inc,
            own.rating, opp.rating, own.rating_diff, score, g.termination,
            int(g.moves[-1].endswith("#")), g.event_kind, len(g.moves),
            json.dumps([float(c) for c in clocks[first::2]]), int(clock_valid)))
    return rows


def _bump(t: Tx, accounts: set[str]) -> None:
    for a in sorted(accounts):  # fixed order: no lock-order deadlock between writers
        t.run("INSERT INTO accounts (account_id, revision) VALUES (?, 1) "
              "ON CONFLICT (account_id) DO UPDATE SET revision = accounts.revision + 1", (a,))


def _accounts_of(t: Tx, key: str) -> set[str]:
    return {r[0] for r in t.run("SELECT account_id FROM player_games WHERE game_key = ?",
                                (key,)).all()}


def upsert_game(db: Database, g: GameIn, secret: str, now: dt.datetime) -> tuple[str, set[str]]:
    """Returns (status, accounts whose history changed)."""
    key = game_key(secret, g.game_id)
    with db.tx() as t:
        won = t.run("""
            INSERT INTO games (game_key, version, tombstone, received_at, moves, clocks)
            VALUES (?, ?, 0, ?, ?, ?)
            ON CONFLICT (game_key) DO UPDATE SET
                version = excluded.version, tombstone = 0, received_at = excluded.received_at,
                moves = excluded.moves, clocks = excluded.clocks
            WHERE games.version < excluded.version
            RETURNING game_key
        """, (key, g.version, iso(now), json.dumps(g.moves), json.dumps(g.clocks))).one()
        if won is None:  # same or newer version already stored: no-op
            current = t.run("SELECT version FROM games WHERE game_key = ?", (key,)).one()[0]
            return ("duplicate" if current == g.version else "stale_version"), set()
        affected = _accounts_of(t, key)  # a correction may change who played
        t.run("DELETE FROM player_games WHERE game_key = ?", (key,))
        for row in player_rows(g, key):
            t.run("INSERT INTO player_games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
        affected |= {str(g.white.account_id), str(g.black.account_id)}
        _bump(t, affected)
    return "stored", affected


def delete_game(db: Database, game_id: str, version: int, secret: str,
                now: dt.datetime) -> tuple[str, set[str]]:
    """Tombstone: the game id stays known (a late re-send of an old version stays a no-op)."""
    key = game_key(secret, game_id)
    with db.tx() as t:
        won = t.run("""
            INSERT INTO games (game_key, version, tombstone, received_at) VALUES (?, ?, 1, ?)
            ON CONFLICT (game_key) DO UPDATE SET version = excluded.version, tombstone = 1,
                received_at = excluded.received_at, moves = NULL, clocks = NULL
            WHERE games.version < excluded.version
            RETURNING game_key
        """, (key, version, iso(now))).one()
        if won is None:
            return "stale_version", set()
        affected = _accounts_of(t, key)
        t.run("DELETE FROM player_games WHERE game_key = ?", (key,))
        _bump(t, affected)
    return "deleted", affected


def prune(db: Database, now: dt.datetime, keep_days: int = KEEP_DAYS) -> int:
    """Design §0.10: keep 60 days of game rows per account online."""
    with db.tx() as t:
        t.run("DELETE FROM player_games WHERE start_ts < ?",
              (iso(now - dt.timedelta(days=keep_days)),))
        return t.rowcount


def histories(db: Database, account_ids: list[str],
              since: dt.datetime) -> dict[str, tuple[int, list[tuple]]]:
    """{account: (revision, rows)} read in ONE transaction, so each revision describes exactly
    the rows returned with it."""
    out = {}
    with db.tx(write=False) as t:
        for a in account_ids:
            rev = t.run("SELECT revision FROM accounts WHERE account_id = ?", (a,)).one()
            rows = t.run("SELECT * FROM player_games WHERE account_id = ? AND start_ts >= ? "
                         "ORDER BY start_ts", (a, iso(since))).all()
            out[a] = ((rev[0] if rev else 0), rows)
    return out


def history(db: Database, account_id: str, since: dt.datetime) -> tuple[int, list[tuple]]:
    return histories(db, [account_id], since)[account_id]
