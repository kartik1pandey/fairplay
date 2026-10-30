"""Online service (design §11, §0.10, §0.18, §12 API and Jobs rows). SQLite; no network."""

import datetime as dt
import json
import os
import threading
import uuid

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from fairplay.features import account_features as af
from fairplay.features import game_features as gf
from fairplay.service import jobs, online, store
from fairplay.service.app import create_app
from fairplay.service.db import Database

SECRET = "test-secret"
T0 = dt.datetime(2026, 9, 1, 12, 0, tzinfo=dt.UTC)
A, B, C = (str(uuid.UUID(int=i)) for i in (1, 2, 3))


@pytest.fixture
def peers(tmp_path):
    t = pa.table({"rating_band": pa.array([1400, None], pa.int64()),
                  "tc_group": pa.array(["180+0", None]), "med": [0.30, 0.25],
                  "mad": [0.10, 0.12], "level": ["cell", "global"]})
    path = tmp_path / "peers.parquet"
    pq.write_table(t, path)
    return path


PG = os.environ.get("FAIRPLAY_TEST_PG")  # opt-in: same suite against real Postgres


@pytest.fixture(params=["sqlite", "postgres"])
def db(request, tmp_path):
    if request.param == "sqlite":
        return Database(f"sqlite:///{(tmp_path / 'online.db').as_posix()}")
    if not PG:
        pytest.skip("set FAIRPLAY_TEST_PG to run against Postgres (docker compose up -d db)")
    import psycopg

    with psycopg.connect(PG, autocommit=True) as con:
        for table in ("games", "player_games", "accounts", "account_features", "jobs"):
            con.execute(f"DROP TABLE IF EXISTS {table}")
    return Database(PG)


class Clock:
    def __init__(self, now=T0 + dt.timedelta(days=20)):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def client(db, peers):
    clock = Clock()
    c = TestClient(create_app(db, peers, SECRET, allow_writes=True, model_version="m-test",
                              clock=clock))
    c.clock = clock
    return c


def make_game(i: int, white=A, black=B, start=None, plies=60, result="1-0", version=1,
              clocks=True, tc=(180, 0), seed=None) -> dict:
    rng = np.random.default_rng(i if seed is None else seed)
    clk, out = {0: tc[0], 1: tc[0]}, []
    for k in range(plies):
        side = k % 2
        if k >= 2:
            clk[side] = max(0, clk[side] - int(rng.integers(0, 6)) + tc[1])
        out.append(clk[side])
    return {"game_id": f"g{i:07d}", "version": version,
            "start_ts": (start or T0 + dt.timedelta(hours=i)).isoformat(),
            "tc_base": tc[0], "tc_inc": tc[1],
            "white": {"account_id": white, "rating": 1450 + i % 7, "rating_diff": 5},
            "black": {"account_id": black, "rating": 1500, "rating_diff": -5},
            "result": result, "termination": "normal", "moves": ["e4"] * plies,
            "clocks": out if clocks else None}


def post(client, g):
    r = client.post("/v1/games", json=g)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("bad", [
    {"result": "2-0"},
    {"black": {"account_id": A, "rating": 1500}},  # same account on both sides
    {"start_ts": "2026-09-01T12:00:00"},  # no UTC offset
    {"surprise": 1},  # unknown field
    {"clocks": [-1] * 60},
    {"white": {"account_id": "not-a-uuid", "rating": 1500}},
])
def test_malformed_input_is_422_and_writes_nothing(client, db, bad):
    assert client.post("/v1/games", json=make_game(1) | bad).status_code == 422
    with db.tx(write=False) as t:
        assert t.run("SELECT count(*) FROM games").one()[0] == 0


@pytest.mark.parametrize("change,reason", [
    ({"tc_base": 60}, "not_blitz"), ({"tc_base": 600}, "not_blitz"), ({"rated": False}, "unrated"),
    ({"termination": "abandoned"}, "abandoned"), ({"moves": ["e4"] * 19, "clocks": None},
                                                  "too_short"),
])
def test_out_of_scope_games_are_skipped_like_offline(client, change, reason):
    out = post(client, make_game(1) | change)
    assert out["status"] == "skipped" and out["reason"] == reason


def test_bot_game_skipped(client):
    g = make_game(1)
    g["white"]["is_bot"] = True
    assert post(client, g)["reason"] == "bot_game"


def test_game_key_matches_offline_cleaner():
    assert store.game_key(SECRET, "abcdEFGH") == duckdb.connect().execute(
        f"SELECT sha256('{SECRET}' || 'abcdEFGH')[:16]").fetchone()[0]


def test_repeat_is_a_noop_and_correction_replaces(client, db):
    g = make_game(1)
    first = post(client, g)
    assert first["status"] == "stored" and first["accounts_updated"] == sorted([A, B])
    assert first["model_version"] == "m-test"
    rev = store.history(db, A, T0 - dt.timedelta(days=1))[0]
    assert post(client, g)["status"] == "duplicate"
    assert store.history(db, A, T0 - dt.timedelta(days=1))[0] == rev  # nothing recomputed
    fixed = make_game(1, black=C, version=2, result="0-1")
    assert post(client, fixed)["accounts_updated"] == sorted([A, B, C])  # B lost the game
    assert post(client, g)["status"] == "stale_version"
    _, rows_b = store.history(db, B, T0 - dt.timedelta(days=1))
    _, rows_a = store.history(db, A, T0 - dt.timedelta(days=1))
    assert rows_b == [] and rows_a[0][10] == 0.0  # A's row now shows the corrected loss


def test_tombstone_removes_rows_and_blocks_old_versions(client, db):
    post(client, make_game(1))
    r = client.delete("/v1/games/g0000001", params={"version": 2}).json()
    assert r["status"] == "deleted" and r["accounts_updated"] == sorted([A, B])
    assert store.history(db, A, T0 - dt.timedelta(days=1))[1] == []
    assert post(client, make_game(1))["status"] == "stale_version"  # late re-send of v1


def test_player_rows_split_clocks_and_flag_invalid_clocks(client, db):
    post(client, make_game(1, plies=40))
    post(client, make_game(2, plies=40) | {"clocks": [180] * 39})  # one clock missing
    _, rows = store.history(db, A, T0 - dt.timedelta(days=1))
    assert len(json.loads(rows[0][15])) == 20 and rows[0][16] == 1
    assert rows[1][16] == 0  # stored and flagged, excluded from timing features (§0.18)


def test_online_features_equal_offline_pipeline(client, db, peers, tmp_path):
    for i in range(14):
        post(client, make_game(i, start=T0 + dt.timedelta(hours=8 * i)))
    as_of = client.clock.now.replace(tzinfo=None)
    online_row = online.stored(db, A)["features"]
    assert online.stored(db, A)["eligible"]

    _, rows = store.history(db, A, as_of - dt.timedelta(days=60))
    d = tmp_path / "player_games"
    d.mkdir()
    pq.write_table(online.to_arrow(rows), d / "month=x.parquet")
    con = duckdb.connect()
    out = gf.build_month(con, "x", processed=tmp_path)
    con.execute("CREATE TEMP TABLE acc AS SELECT ?::VARCHAR AS account_id, ?::TIMESTAMP AS cutoff",
                [A, as_of])
    rel = af.cutoff_features(con, [out], "acc", as_of)  # the offline, file-based entry point
    con.execute(f"CREATE TEMP TABLE f AS {rel.sql_query()}")
    con.execute(f"CREATE TEMP TABLE peers AS SELECT * FROM read_parquet('{peers.as_posix()}')")
    rel = con.sql(f"SELECT * FROM {af.apply_peer_z(con, 'f', 'peers')}")
    offline_row = dict(zip(rel.columns, rel.limit(1).fetchall()[0], strict=True))
    for name in af.FEATURES:
        if name == "account_age_days":
            continue  # joined from the label snapshot by the scoring job
        a, b = online_row[name], offline_row[name]
        assert (a is None and b is None) or a == pytest.approx(b), name
    assert online_row["think_time_cv"] is not None  # timing features really computed


def test_late_game_enters_later_computation(client, db):
    for i in range(10):
        post(client, make_game(i, start=T0 + dt.timedelta(days=i)))
    before = online.stored(db, A)["features"]["games_in_window"]
    post(client, make_game(99, start=T0 - dt.timedelta(days=2)))  # arrives late, played earlier
    assert online.stored(db, A)["features"]["games_in_window"] == before + 1


def test_stale_recompute_cannot_overwrite_fresher_one(client, db, peers):
    post(client, make_game(1))
    now = client.clock.now
    rev_old, rows_old = store.history(db, A, now - dt.timedelta(days=60))  # slow worker reads
    post(client, make_game(2))  # a newer game lands and is recomputed meanwhile
    stale = online.compute(rows_old, [A], now.replace(tzinfo=None), peers)[A]
    assert not online.write(db, A, rev_old, now, now, stale, False)  # rejected
    assert online.stored(db, A)["features"]["games_in_window"] == 2


def test_concurrent_games_equal_sequential(tmp_path, peers):
    games = [make_game(i, start=T0 + dt.timedelta(hours=i)) for i in range(12)]

    def run(name, parallel):
        db = Database(f"sqlite:///{(tmp_path / name).as_posix()}")
        c = TestClient(create_app(db, peers, SECRET, allow_writes=True, clock=Clock()))
        if parallel:
            threads = [threading.Thread(target=post, args=(c, g)) for g in games]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            for g in games:
                post(c, g)
        return {a: online.stored(db, a)["features"] for a in (A, B)}

    assert run("par.db", True) == run("seq.db", False)


def test_prune_keeps_sixty_days(client, db):
    post(client, make_game(1, start=T0 - dt.timedelta(days=70)))
    post(client, make_game(2, start=T0))
    assert store.prune(db, T0) == 2  # both sides of the old game
    assert len(store.history(db, A, T0 - dt.timedelta(days=365))[1]) == 1


def test_read_only_app_has_no_write_or_lookup_endpoints(db, peers):
    c = TestClient(create_app(db, peers, SECRET))
    assert c.get("/health").json()["ok"]
    assert c.post("/v1/games", json=make_game(1)).status_code in (404, 405)
    assert c.get(f"/v1/accounts/{A}/features").status_code == 404


# ---------------------------------------------------------------- jobs
def test_jobs_duplicate_message_and_bound(db):
    assert jobs.enqueue(db, A, "2026-09-06", T0) == "queued"
    assert jobs.enqueue(db, A, "2026-09-06", T0) == "duplicate"
    assert jobs.enqueue(db, B, "2026-09-06", T0, limit=1) == "deferred"
    assert jobs.counts(db) == {"queued": 1}


def test_expired_lease_is_retaken_and_late_result_not_double_counted(db):
    jobs.enqueue(db, A, "2026-09-06", T0)
    first = jobs.lease(db, "w1", T0)  # w1 crashes: never completes
    assert jobs.lease(db, "w2", T0 + dt.timedelta(minutes=5)) is None  # lease still valid
    second = jobs.lease(db, "w2", T0 + dt.timedelta(minutes=11))
    assert second["job_key"] == first["job_key"] and second["attempts"] == 2
    assert jobs.fail(db, first["job_key"], "w1", "timeout", "late", T0) == "ignored"  # stale
    assert jobs.complete(db, second["job_key"], {"x": 1}, T0)
    assert not jobs.complete(db, first["job_key"], {"x": 2}, T0)  # duplicate run ignored
    assert jobs.counts(db) == {"done": 1}


def test_timeouts_retry_with_backoff_then_fail(db):
    jobs.enqueue(db, A, "2026-09-06", T0)
    now = T0
    for attempt in (1, 2):
        j = jobs.lease(db, "w", now)
        assert j["attempts"] == attempt
        assert jobs.fail(db, j["job_key"], "w", "timeout", "engine hang", now) == "retry"
        assert jobs.lease(db, "w", now) is None  # backoff: not runnable yet
        now += jobs.BACKOFF * 2 ** (attempt - 1)
    j = jobs.lease(db, "w", now)
    assert jobs.fail(db, j["job_key"], "w", "engine_crash", "x", now) == "failed"
    assert jobs.counts(db) == {"failed": 1}


def test_bad_input_fails_immediately_and_crash_loop_stops(db):
    jobs.enqueue(db, A, "c1", T0)
    j = jobs.lease(db, "w", T0)
    assert jobs.fail(db, j["job_key"], "w", "bad_input", "no games", T0) == "failed"
    jobs.enqueue(db, B, "c1", T0)
    now = T0
    for _ in range(jobs.MAX_ATTEMPTS):  # the worker crashes every time
        assert jobs.lease(db, "w", now) is not None
        now += jobs.LEASE + dt.timedelta(seconds=1)
    assert jobs.lease(db, "w", now) is None
    assert jobs.counts(db) == {"failed": 2}
