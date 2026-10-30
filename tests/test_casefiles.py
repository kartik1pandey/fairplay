"""Case-file template, peer reference and privacy scan (design §10, §0 item 17, §12)."""

import datetime as dt

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fairplay.casefiles.peers import PeerReference
from fairplay.casefiles.template import build_case, extreme_game, record_decision
from fairplay.evaluation.queue import Month, evaluate, rank_order
from fairplay.features.account_features import FEATURES
from fairplay.privacy import scan

VERSIONS = {"model": "lgbm-test", "features": "f-1.0.0", "policy": "c0.5-m1.0"}


def reference(bands: dict[int, int], tc="300+0", seed=0) -> PeerReference:
    """bands = {rating_band: n_accounts}; one row per account, features ~ U(0, 1)."""
    rng = np.random.default_rng(seed)
    n = sum(bands.values())
    cols = {"account_id": np.array([f"a{i}" for i in range(n)]),
            "rating_band": np.repeat(list(bands), list(bands.values())),
            "tc_group": np.array([tc] * n), "weight": np.ones(n)}
    cols |= {f: rng.uniform(size=n) for f in FEATURES}
    return PeerReference(cols)


def account(**overrides) -> dict:
    row = {f: 0.5 for f in FEATURES}  # the median of U(0, 1): neither tail
    row |= {"account_id": "acc-1", "cutoff": dt.datetime(2025, 8, 10), "rating_band": 1600,
            "tc_group": "300+0", "games_in_window": 25}
    return row | overrides


def test_small_cell_merges_neighbouring_bands():
    ref = reference({1400: 120, 1600: 50, 1800: 120})
    g = ref.group(1600, "300+0")
    assert g.n_accounts == 290 and g.sufficient and g.label == "1400-1999 blitz 300+0"
    assert ref.group(1400, "300+0").label == "1400-1999 blitz 300+0"  # 170 at +-200, so +-400


def test_percentile_uses_mid_ranks_and_weights():
    cols = {"account_id": np.array([f"a{i}" for i in range(200)]),
            "rating_band": np.full(200, 1600), "tc_group": np.array(["300+0"] * 200),
            "weight": np.ones(200)}
    cols |= {f: np.arange(200, dtype=float) for f in FEATURES}
    ref = PeerReference(cols)
    g = ref.group(1600, "300+0")
    assert ref.percentile(g, "think_time_cv", 10.0) == pytest.approx(100 * 10.5 / 200)
    cols["weight"] = np.where(np.arange(200) < 100, 3.0, 1.0)  # heavier low half
    assert PeerReference(cols).percentile(g, "think_time_cv", 100.0) == pytest.approx(75.125)


def test_evidence_ranked_by_extremeness_and_context_never_cited():
    ref = reference({1600: 400})
    case = build_case(account(think_time_cv=0.001, performance_rating_gap=0.97,
                              upset_win_share=0.999, games_per_active_day=0.9999,
                              time_loss_share=0.2), 0.83, ref, VERSIONS)
    feats = [e["feature"] for e in case["evidence"]]
    assert feats[0] in ("think_time_cv", "upset_win_share") and "performance_rating_gap" in feats
    assert "games_per_active_day" not in feats  # context only
    assert case["status"] == "review"
    assert any(c["feature"] == "time_loss_share" for c in case["counter_evidence"])
    e = case["evidence"][0]
    assert e["peer_n"] == 400 and e["reference"] == "July 2025" and e["query"].startswith("q_")
    assert e["peer_p05"] < e["peer_median"] < e["peer_p95"]


def test_at_most_three_evidence_items():
    ref = reference({1600: 400})
    extreme = {f: (0.0001 if d == "low" else 0.9999) for f, d in FEATURES.items()
               if d != "context"}
    assert len(build_case(account(**extreme), 0.9, ref, VERSIONS)["evidence"]) == 3


def test_insufficient_evidence_rules():
    ref = reference({1600: 400})
    one = build_case(account(think_time_cv=0.001), 0.5, ref, VERSIONS)
    assert one["status"] == "insufficient_evidence" and len(one["evidence"]) == 1
    small = reference({1600: 150})
    case = build_case(account(think_time_cv=0.001, upset_win_share=0.999), 0.5, small, VERSIONS)
    assert case["status"] == "insufficient_evidence" and case["evidence"] == []


def test_both_tails_and_nan_skipped():
    ref = reference({1600: 400})
    case = build_case(account(instant_move_share=0.0001, clock_at_30_z=0.9999,
                              think_time_cv=float("nan")), 0.5, ref, VERSIONS)
    assert {e["feature"] for e in case["evidence"]} == {"instant_move_share", "clock_at_30_z"}
    assert all(c["feature"] != "think_time_cv" for c in case["counter_evidence"])


def test_case_id_is_deterministic_and_reviewer_rules():
    ref = reference({1600: 400})
    a = build_case(account(), 0.4, ref, VERSIONS)
    assert a["case_id"] == build_case(account(), 0.4, ref, VERSIONS)["case_id"]
    assert a["case_id"] != build_case(account(), 0.4, ref, VERSIONS | {"model": "x"})["case_id"]
    done = record_decision(a, "watch", "re-check timing in 30 days")
    assert done["reviewer"] == {"decision": "watch", "reason": "re-check timing in 30 days"}
    assert a["reviewer"]["decision"] is None  # the original is not mutated
    with pytest.raises(ValueError):
        record_decision(a, "ban", "x")
    with pytest.raises(ValueError):
        record_decision(a, "escalate", "this proves he cheated")


def test_extreme_game_respects_cutoff(tmp_path):
    cut = dt.datetime(2025, 8, 10)
    rows = [{"account_id": "acc-1", "game_key": k, "start_ts": cut - dt.timedelta(minutes=m),
             "score": s, "expected": 0.5}
            for k, m, s in [("old", 60 * 24 * 31, 1.0), ("ok", 600, 1.0), ("meh", 120, 0.5),
                            ("late", 29, 1.0)]]
    f = tmp_path / "g.parquet"
    pq.write_table(pa.Table.from_pylist(rows), f)
    con = duckdb.connect()
    assert extreme_game(con, [f], "acc-1", cut, "score_minus_expected", 99.0) == "ok"


def test_queue_reports_queued_rows_per_week():
    m = Month.from_rows(["a", "b", "c", "a", "b", "c"], [1, 1, 1, 2, 2, 2], [10] * 6,
                        [1, 0, 0, 1, 0, 0])
    s = np.array([0.9, 0.5, 0.1, 0.95, 0.6, 0.2])
    r = evaluate(m, rank_order(m, s), s, capacity=0.34)  # k = 2 per week
    assert [list(w) for w in r["queued_rows"]] == [[0, 1], [5]]  # a, b already queued in week 2


def test_privacy_scan_finds_names_without_noise(tmp_path):
    (tmp_path / "ok.json").write_text('{"account": "3f1c", "move": "e4"}', encoding="utf-8")
    (tmp_path / "bad.md").write_text("line one\nreviewed Knight_Rider99 today\n",
                                     encoding="utf-8")
    (tmp_path / "skip.parquet").write_bytes(b"knight_rider99")  # binary: not scanned
    hits = scan([tmp_path], {"knight_rider99", "e4"})
    assert hits == [((tmp_path / "bad.md").as_posix(), 2)]
