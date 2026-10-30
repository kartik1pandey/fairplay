"""Online feature recomputation (design §0.10): the account's stored game rows run through the
SAME SQL as the offline pipeline (game_features_sql -> cutoff_features_from -> apply_peer_z),
in an in-memory DuckDB. There is no second implementation that could drift from the offline one.

as_of plays the role of the cutoff: features describe games started in
[as_of - 30 days, as_of - 30 minutes), exactly as at a weekly cutoff.
Account age needs createdAt from a label snapshot and is joined by the weekly scoring job.
"""

import datetime as dt
import json
import threading
from pathlib import Path

import duckdb
import pyarrow as pa

from fairplay.features.account_features import apply_peer_z, cutoff_features_from
from fairplay.features.game_features import FEATURE_VERSION, game_features_sql
from fairplay.service import store
from fairplay.service.db import Database

MIN_GAMES = 10  # eligibility (design §4)
PLAYER_GAMES_SCHEMA = pa.schema([
    ("account_id", pa.string()), ("game_key", pa.string()), ("month", pa.string()),
    ("color", pa.string()), ("start_ts", pa.timestamp("us")), ("tc_base", pa.int32()),
    ("tc_inc", pa.int32()), ("own_elo", pa.int32()), ("opp_elo", pa.int32()),
    ("own_rating_diff", pa.int32()), ("score", pa.float64()), ("termination", pa.string()),
    ("ended_in_mate", pa.bool_()), ("event_kind", pa.string()), ("n_plies", pa.int64()),
    ("own_clocks", pa.list_(pa.float64())), ("clock_valid", pa.bool_()),
])


def to_arrow(rows: list[tuple]) -> pa.Table:
    """Stored rows -> the offline player_games schema."""
    cols = {n: [] for n in PLAYER_GAMES_SCHEMA.names}
    for (acc, key, _version, color, ts, base, inc, own, opp, diff, score, term, mate, kind,
         plies, clocks, valid) in rows:
        start = dt.datetime.fromisoformat(ts)
        for name, v in (("account_id", acc), ("game_key", key), ("month", ts[:7]),
                        ("color", color), ("start_ts", start), ("tc_base", base),
                        ("tc_inc", inc), ("own_elo", own), ("opp_elo", opp),
                        ("own_rating_diff", diff), ("score", score), ("termination", term),
                        ("ended_in_mate", bool(mate)), ("event_kind", kind), ("n_plies", plies),
                        ("own_clocks", json.loads(clocks)), ("clock_valid", bool(valid))):
            cols[name].append(v)
    return pa.table(cols, schema=PLAYER_GAMES_SCHEMA)


_local = threading.local()  # one DuckDB connection per thread (connect costs ~50 ms)


def _connection(peers: Path) -> duckdb.DuckDBPyConnection:
    """Reused per thread; the frozen peer table is loaded once per connection."""
    if getattr(_local, "peers", None) != peers:
        _local.con = duckdb.connect()
        _local.con.execute("CREATE TEMP TABLE peers AS SELECT * FROM "
                           f"read_parquet('{peers.as_posix()}')")
        _local.peers = peers
    return _local.con


def compute(rows: list[tuple], account_ids: list[str], as_of: dt.datetime,
            peers: Path) -> dict[str, dict]:
    """Feature rows at as_of (naive UTC) for several accounts in ONE pass of the shared SQL
    (query planning, not data, dominates the cost). Accounts without a game in the window are
    absent from the result."""
    con = _connection(peers)
    con.register("pg_src", to_arrow(rows))
    con.execute(f"CREATE OR REPLACE TEMP TABLE gf AS {game_features_sql('pg_src')}")
    con.execute("CREATE OR REPLACE TEMP TABLE acc AS SELECT unnest(?::VARCHAR[]) AS account_id, "
                "?::TIMESTAMP AS cutoff", [account_ids, as_of])
    rel = cutoff_features_from(con, "gf", "acc", as_of)
    con.execute(f"CREATE OR REPLACE TEMP TABLE f AS {rel.sql_query()}")
    rel = con.sql(f"SELECT * FROM {apply_peer_z(con, 'f', 'peers')}")
    out = {}
    for r in rel.fetchall():
        d = dict(zip(rel.columns, r, strict=True))
        d["cutoff"] = d["cutoff"].isoformat(sep=" ")
        out[d["account_id"]] = d
    con.unregister("pg_src")
    return out


def recompute(db: Database, account_ids: list[str], as_of: dt.datetime, peers: Path,
              now: dt.datetime) -> dict[str, dict]:
    """Read histories + revisions in one snapshot, compute together, store each if not stale."""
    as_of = as_of.astimezone(dt.UTC).replace(tzinfo=None) if as_of.tzinfo else as_of
    hist = store.histories(db, account_ids, as_of - dt.timedelta(days=store.KEEP_DAYS))
    feats = compute([r for _, rows in hist.values() for r in rows], account_ids, as_of, peers)
    out = {}
    for a in account_ids:
        f = feats.get(a, {})
        eligible = bool(f) and f["games_in_window"] >= MIN_GAMES
        write(db, a, hist[a][0], as_of, now, f, eligible)
        out[a] = {"revision": hist[a][0], "eligible": eligible, **f}
    return out


def write(db: Database, account_id: str, revision: int, as_of: dt.datetime, now: dt.datetime,
          feats: dict, eligible: bool) -> bool:
    """Upsert guarded by revision: an older computation never replaces a newer one."""
    with db.tx() as t:
        won = t.run("""
            INSERT INTO account_features
                (account_id, revision, as_of, computed_at, feature_version, eligible, features)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (account_id) DO UPDATE SET
                revision = excluded.revision, as_of = excluded.as_of,
                computed_at = excluded.computed_at, feature_version = excluded.feature_version,
                eligible = excluded.eligible, features = excluded.features
            WHERE account_features.revision <= excluded.revision
            RETURNING account_id
        """, (account_id, revision, store.iso(as_of), store.iso(now), FEATURE_VERSION,
              int(eligible), json.dumps(feats, default=str))).one()
    return won is not None


def stored(db: Database, account_id: str) -> dict | None:
    with db.tx(write=False) as t:
        r = t.run("SELECT revision, as_of, feature_version, eligible, features "
                  "FROM account_features WHERE account_id = ?", (account_id,)).one()
    if r is None:
        return None
    return {"revision": r[0], "as_of": r[1], "feature_version": r[2], "eligible": bool(r[3]),
            "features": json.loads(r[4])}
