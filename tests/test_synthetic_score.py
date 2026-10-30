"""Synthetic scoring (design §6): shared feature SQL, frozen real threshold, recall vs controls."""

import datetime as dt

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_synthetic import SAMPLER, FakeEngine, FakeHuman

from fairplay.models.data import FEATURE_COLUMNS, Split
from fairplay.service.loadtest import neutral_peers
from fairplay.synthetic import run_synthetic
from fairplay.synthetic import score as sc
from fairplay.synthetic.generator import STRATEGIES, generate_account

CUT = dt.datetime(2025, 8, 10)


def test_synthetic_accounts_go_through_the_shared_feature_sql(tmp_path):
    files = []
    for strat in ("control", "A"):
        rows = []
        for i in range(2):
            r, _ = generate_account(f"syn-{strat}-{i:04d}", STRATEGIES[strat], 1500, 300,
                                    [((180, 0), 1.0)], CUT, 10 * i + len(strat), FakeHuman(),
                                    FakeEngine(), SAMPLER, n_games=12)
            rows += r
        f = tmp_path / f"strategy={strat}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), f)
        files.append(f)
    feats = sc.synthetic_features(files, CUT, neutral_peers(tmp_path / "p.parquet"))
    assert [f["account_id"] for f in feats] == ["syn-A-0000", "syn-A-0001", "syn-control-0000",
                                                "syn-control-0001"]
    assert all(f["games_in_window"] == 12 for f in feats)
    split = sc.to_split(feats, {"syn-A-0000": 300}, CUT)
    assert split.X.shape == (4, len(FEATURE_COLUMNS))
    age = FEATURE_COLUMNS.index("account_age_days")
    assert split.X[0, age] == 300 and np.isnan(split.X[1, age])  # unknown age stays NaN


def test_recall_table_against_frozen_threshold():
    ids = np.array([f"syn-A-{i}" for i in range(4)] + [f"syn-control-{i}" for i in range(4)])
    score = np.array([0.9, 0.8, 0.2, np.nan, 0.95, 0.1, 0.1, 0.1])
    rating = np.array([1500, 1700, 1500, 1500, 1500, 1500, 1700, 1700], dtype=float)
    t = sc.recall_table(ids, score, rating, threshold=0.5)
    assert t["A"]["synthetic_recall"] == 0.5 and t["A"]["unscorable"] == 1  # NaN never counts
    assert t["control"]["control_false_alarm"] == 0.25
    assert t["A"]["by_band"]["1600-1799"] == {"share_above": 1.0, "n": 1,
                                              "wilson_95": sc.wilson(1, 1)}
    lo, hi = t["A"]["wilson_95"]
    assert lo < 0.5 < hi


class ColumnModel:
    def score(self, split):
        return split.column("performance_rating_gap")


def test_threshold_is_the_median_weekly_queue_cut_from_the_real_month():
    rng = np.random.default_rng(0)
    n = 400  # 200 accounts x 2 weeks
    X = np.full((n, len(FEATURE_COLUMNS)), np.nan)
    X[:, FEATURE_COLUMNS.index("performance_rating_gap")] = rng.normal(size=n)
    val = Split("test", X, rng.integers(0, 2, n).astype(np.int8), np.full(n, 0.5),
                np.tile(np.arange(200), 2).astype(object), np.repeat([1, 2], 200),
                np.full(n, 20), {"rating_at_cutoff": np.full(n, 1500.0)})
    th = sc.frozen_threshold(ColumnModel(), val, capacity=0.05)  # k = 10 per week
    s = X[:, FEATURE_COLUMNS.index("performance_rating_gap")]
    week1_cut = np.sort(s[:200])[-10]  # 10th best score of week 1
    assert th["weekly"][0] == pytest.approx(week1_cut)
    assert th["threshold"] == pytest.approx(np.median(th["weekly"]))


def test_held_out_strategy_needs_explicit_flag():
    with pytest.raises(SystemExit, match="held out"):
        run_synthetic.score(["control", "C"], include_held_out=False)


def test_realism_compares_controls_with_real_unflagged_only():
    rng = np.random.default_rng(1)
    names = [f for f in FEATURE_COLUMNS if f != "account_age_days"]
    controls = [{"account_id": f"syn-control-{i}", "rating_at_cutoff": 1500,
                 **{f: float(rng.normal()) for f in names}} for i in range(600)]
    cheats = [{"account_id": f"syn-A-{i}", "rating_at_cutoff": 1500,
               **{f: 50.0 for f in names}} for i in range(600)]  # must be ignored
    n = 1200
    X = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    y = np.r_[np.zeros(600), np.ones(600)].astype(np.int8)
    X[600:] += 10  # flagged real accounts are far away; must be ignored too
    real = Split("train", X, y, np.ones(n), np.arange(n).astype(object), np.ones(n),
                 np.full(n, 20), {})
    rep = sc.realism(controls + cheats, real, np.full(n, 1400.0), np.full(n, 1500.0))
    assert rep["features_ks_above_limit"] == []
    shifted = [c | {"think_time_cv": c["think_time_cv"] + 1} for c in controls]
    assert sc.realism(shifted, real, np.full(n, 1400.0),
                      np.full(n, 1500.0))["features_ks_above_limit"] == ["think_time_cv"]


def test_small_bands_are_not_judged():
    from fairplay.synthetic.realism import realism_table

    rng = np.random.default_rng(2)
    rep = realism_table({"f": rng.normal(size=60)}, {"f": rng.normal(size=60)},
                        np.full(60, 1400.0), np.full(60, 1400.0))
    assert rep["rows"][0]["ks"] is None and rep["features_ks_above_limit"] == []


def test_auc_vs_control_ranks_each_strategy_against_controls_only():
    from fairplay.synthetic.score import auc_vs_control

    ids = np.array(["syn-control-0", "syn-control-1", "syn-A-0", "syn-A-1", "syn-B-0", "syn-B-1"])
    score = np.array([0.1, 0.2, 0.9, 0.8, 0.15, 0.05])
    out = auc_vs_control(ids, score)
    assert out["A"] == 1.0  # both A above both controls
    assert out["B"] == 0.25  # B wins one of four pairs (0.15 > 0.1)
