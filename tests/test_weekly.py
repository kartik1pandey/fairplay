"""Weekly queue job, drift monitoring and the load test (design §7, §11, §12)."""

import datetime as dt
import json
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from fairplay.evaluation.queue import Month, evaluate, rank_order, select_week, week_limits
from fairplay.monitoring.drift import PSI_ALERT, drift_report, psi
from fairplay.service import loadtest, online, store, weekly
from fairplay.service.app import create_app
from fairplay.service.db import Database
from fairplay.service.schema import GameIn

SECRET = "s"
CUT1 = dt.datetime(2026, 9, 6)  # a Sunday
ACC = [str(uuid.UUID(int=i)) for i in range(1, 13)]
OPP = [str(uuid.UUID(int=1000 + i)) for i in range(40)]


@pytest.fixture
def peers(tmp_path):
    return loadtest.neutral_peers(tmp_path / "peers.parquet")


@pytest.fixture
def db(tmp_path):
    return Database(f"sqlite:///{(tmp_path / 'w.db').as_posix()}")


def add_games(db, account, n, end, rating=1500, gid=[0]):  # noqa: B006 (shared id counter)
    for i in range(n):
        gid[0] += 1
        start = end - dt.timedelta(days=1 + i)
        g = GameIn.model_validate({
            "game_id": f"w{gid[0]:07d}", "start_ts": start.replace(tzinfo=dt.UTC).isoformat(),
            "tc_base": 180, "tc_inc": 0,
            "white": {"account_id": account, "rating": rating, "rating_diff": 4},
            "black": {"account_id": OPP[gid[0] % 40], "rating": 1500, "rating_diff": -4},
            "result": "1-0", "termination": "normal", "moves": ["e4"] * 40,
            "clocks": [180 - k for k in range(40)]})
        store.upsert_game(db, g, SECRET, end)


class FakeModel:
    """score = rank of the account in ACC (later = higher), optional overrides; NaN allowed."""

    name = "fake"

    def __init__(self, override=None):
        self.override = override or {}

    def score(self, split):
        return np.array([self.override.get(a, ACC.index(a) / 10 if a in ACC else 0.0)
                         for a in split.account_id], dtype=float)


def week(db, peers, cutoff, model=None, capacity=0.25, version="m1"):
    return weekly.run_weekly(db, cutoff, model or FakeModel(), version, peers, {}, cutoff,
                             weekly.Policy(capacity=capacity))


def entries(db, summary):
    return [e["account_id"] for e in weekly.queue(db, summary["queue_id"])["entries"]]


@pytest.mark.parametrize("seed", range(20))
def test_production_selection_equals_offline_evaluation(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(50, 400))
    score = np.round(rng.random(n), 2)  # rounding forces ties
    games = rng.integers(10, 40, n)
    capped = rng.random(n) < 0.15
    m = Month.from_rows(np.arange(n), np.zeros(n), games, np.zeros(n), capped)
    order = rank_order(m, score, seed)[0]
    offline = evaluate(m, [order], score, 0.05, cap_multiplier=1.0)["queued_rows"][0]
    k, cap = week_limits(n, int(capped.sum()), 0.05, 1.0)
    assert list(select_week(order, capped, k, cap)) == list(offline)


def test_queue_is_immutable_and_versioned(db, peers):
    for a in ACC[:8]:
        add_games(db, a, 12, CUT1)
    s = week(db, peers, CUT1)
    assert s["eligible"] == 8 and s["k"] == 2 and s["tier"] == "tier1_only"
    assert entries(db, s) == [ACC[7], ACC[6]]  # highest scores first
    add_games(db, ACC[0], 5, CUT1)  # late games after the queue was made
    assert week(db, peers, CUT1, FakeModel({ACC[0]: 99.0})) == s  # same queue, not regenerated
    assert entries(db, s) == [ACC[7], ACC[6]]
    with db.tx(write=False) as t:
        v = t.run("SELECT DISTINCT model_version, feature_version FROM scores").all()
    assert v == [("m1", online.FEATURE_VERSION)]


def test_ineligible_and_unscorable_accounts_fail_closed(db, peers):
    for a in ACC[:6]:
        add_games(db, a, 12, CUT1)
    add_games(db, ACC[11], 5, CUT1)  # too few games: not eligible
    s = week(db, peers, CUT1, FakeModel({ACC[5]: float("nan")}), capacity=0.5)
    assert s["eligible"] == 6 and s["insufficient_data"] == 1 and s["scorable"] == 5
    assert ACC[5] not in entries(db, s) and ACC[11] not in entries(db, s)


def test_high_rated_cap_applies(db, peers):
    for i, a in enumerate(ACC[:10]):
        add_games(db, a, 12, CUT1, rating=2300 if i >= 7 else 1500)  # top 3 scores are 2200+
    s = week(db, peers, CUT1, capacity=0.3)  # k = 3; 2200+ share 0.3 -> cap = ceil(0.9) = 1
    assert s["k"] == 3 and s["cap_2200"] == 1 and s["capped_in_queue"] == 1
    assert entries(db, s) == [ACC[9], ACC[6], ACC[5]]


def test_reviewer_workflow_over_weeks(db, peers):
    for a in ACC[:8]:
        add_games(db, a, 72, CUT1 + dt.timedelta(days=42))  # daily from CUT1 - 30 d to + 41 d
    w1 = week(db, peers, CUT1, capacity=0.5)  # k = 4: ACC 7, 6, 5, 4
    q1 = w1["queue_id"]
    now = CUT1 + dt.timedelta(days=2)
    weekly.record_decision(db, q1, ACC[7], "clear", "timing normal for peers", now)
    weekly.record_decision(db, q1, ACC[6], "watch", "re-check next month", now)
    weekly.record_decision(db, q1, ACC[5], "escalate", "needs senior review", now)
    # ACC[4]: no decision yet -> pending

    w2 = week(db, peers, CUT1 + dt.timedelta(days=7), capacity=0.5)
    assert w2["blocked"] == {"cleared": 1, "escalated": 1, "pending": 1, "watch": 1}
    assert entries(db, w2) == [ACC[3], ACC[2], ACC[1], ACC[0]]

    rose = FakeModel({ACC[7]: 0.7 + 0.1})  # cleared account's score rose by >= 0.1
    w3 = week(db, peers, CUT1 + dt.timedelta(days=14), rose, capacity=0.5, version="m2")
    assert ACC[7] in entries(db, w3)

    w6 = week(db, peers, CUT1 + dt.timedelta(days=35), capacity=0.5, version="m3")
    queued6 = entries(db, w6)
    assert ACC[6] in queued6 and ACC[5] not in queued6  # watch expired; escalated stays out


def test_decisions_are_validated(db, peers):
    for a in ACC[:4]:
        add_games(db, a, 12, CUT1)
    q = week(db, peers, CUT1, capacity=0.5)["queue_id"]
    with pytest.raises(ValueError):
        weekly.record_decision(db, q, ACC[3], "ban", "x", CUT1)
    with pytest.raises(ValueError):
        weekly.record_decision(db, q, ACC[3], "escalate", "clearly cheated", CUT1)
    with pytest.raises(KeyError):
        weekly.record_decision(db, q, ACC[0], "clear", "fine", CUT1)  # not in this queue


def test_queue_endpoints(db, peers):
    for a in ACC[:4]:
        add_games(db, a, 12, CUT1)
    q = week(db, peers, CUT1, capacity=0.5)
    public = TestClient(create_app(db, peers, SECRET))
    r = public.get("/v1/queue", params={"cutoff": "2026-09-06"}).json()
    assert [e["account_id"] for e in r["entries"]] == [ACC[3], ACC[2]]
    assert "not evidence of wrongdoing" in r["notice"]
    body = {"queue_id": q["queue_id"], "account_id": ACC[3], "decision": "clear", "reason": "ok"}
    assert public.post("/v1/queue/decisions", json=body).status_code in (404, 405)
    local = TestClient(create_app(db, peers, SECRET, allow_writes=True,
                                  clock=lambda: dt.datetime(2026, 9, 7, tzinfo=dt.UTC)))
    assert local.post("/v1/queue/decisions", json=body).json() == {"ok": True}
    assert local.post("/v1/queue/decisions", json=body | {"reason": "he cheated"}
                      ).status_code == 422


def test_psi_behaviour():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20_000)
    assert psi(ref, rng.normal(size=5_000)) < 0.01
    assert psi(ref, rng.normal(0.8, 1, 5_000)) > PSI_ALERT
    gone = rng.normal(size=5_000)
    gone[:2_500] = np.nan  # half the values suddenly missing
    assert psi(ref, gone) > PSI_ALERT
    rep = drift_report(ref, rng.normal(size=5_000), {"f": ref}, {"f": rng.normal(1, 1, 5_000)})
    assert rep["alerts"] == [f"f PSI {rep['feature_psi']['f']:.3f}"]
    assert "no automatic retraining" in rep["action"]


def test_load_test_runs_and_reports_percentiles(tmp_path, peers):
    games = loadtest.synthetic_week(40, 8, dt.datetime(2026, 9, 6, tzinfo=dt.UTC))
    r = loadtest.run(games, peers, f"sqlite:///{(tmp_path / 'l.db').as_posix()}")
    assert r["requests"] == 40 and r["statuses"] == {"stored": 40}
    assert r["p50_ms"] <= r["p95_ms"] <= r["p99_ms"] <= r["max_ms"]


def test_real_week_conversion_round_trips(tmp_path, db):
    add_games(db, ACC[0], 3, CUT1)
    _, rows = store.history(db, ACC[0], CUT1 - dt.timedelta(days=30))
    path = tmp_path / "pg.parquet"
    pq.write_table(online.to_arrow(rows), path)
    games = loadtest.real_week(path, dt.date(2026, 9, 1))
    assert len(games) == 3
    for g, row in zip(games, sorted(rows, key=lambda r: r[4]), strict=True):
        own = [r for r in store.player_rows(GameIn.model_validate(g), "k")
               if r[0] == ACC[0]][0]
        assert own[10] == row[10] and json.loads(own[15]) == json.loads(row[15])  # score, clocks
        assert own[16] == row[16]


def test_privacy_no_usernames_in_service_tables(db):
    """Only UUIDs and game keys reach the service: every id column parses as a UUID."""
    add_games(db, ACC[0], 2, CUT1)
    with db.tx(write=False) as t:
        ids = [r[0] for r in t.run("SELECT account_id FROM player_games").all()]
    assert all(uuid.UUID(i) for i in ids)
    _ = pa  # pyarrow imported for parquet fixtures above
