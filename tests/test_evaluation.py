import datetime as dt

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fairplay.dataset.assemble import assemble, missing_label_balance
from fairplay.evaluation.bootstrap import bootstrap, compare
from fairplay.evaluation.queue import Month, evaluate, rank_order

W0, W1 = "2025-07-06", "2025-07-13"


def hand_month():
    """Week 0: A .9 B .8 C .7 D .6.  Week 1: A .99 E .95 B .8 C .7 D .6.  Flagged: A, C, E."""
    rows = [("A", W0, .9), ("B", W0, .8), ("C", W0, .7), ("D", W0, .6),
            ("A", W1, .99), ("E", W1, .95), ("B", W1, .8), ("C", W1, .7), ("D", W1, .6)]
    flagged = {"A", "C", "E"}
    month = Month.from_rows([r[0] for r in rows], [r[1] for r in rows], [10] * len(rows),
                            [int(r[0] in flagged) for r in rows])
    return month, np.array([r[2] for r in rows])


def test_queue_contract_by_hand():
    month, score = hand_month()
    r = evaluate(month, rank_order(month, score), score, capacity=0.4)
    # week 0: k = ceil(0.4 x 4) = 2 -> A, B; precision 1/2; base 2/4
    # week 1: k = ceil(0.4 x 5) = 2; A and B already queued -> E, C; precision 2/2; base 2/3
    assert r["k_per_week"] == [2, 2]
    assert r["weekly_precision"] == [0.5, 1.0]
    assert r["flag_precision"] == pytest.approx(0.75)
    assert r["base_rate"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert r["lift"] == pytest.approx(0.75 / ((0.5 + 2 / 3) / 2))
    assert r["flag_recall"] == pytest.approx(1.0)  # A, C, E all queued during the month
    # PR-AUC on max score per account: A .99(1) E .95(1) B .8(0) C .7(1) D .6(0)
    assert r["flag_pr_auc"] == pytest.approx((1 + 1 + 3 / 4) / 3)
    assert r["unique_accounts_queued"] == 4


def test_ties_prefer_more_games_then_seeded_random():
    month = Month.from_rows(["A", "B", "C"], [W0] * 3, [10, 50, 20], [0, 0, 0])
    order = rank_order(month, np.array([0.5, 0.5, 0.5]))
    assert list(month.acc[order[0]]) == [1, 2, 0]  # B (50 games), C (20), A (10)


def test_queue_never_padded_when_list_runs_out():
    month = Month.from_rows(["A", "A"], [W0, W1], [10, 10], [1, 1])
    r = evaluate(month, rank_order(month, np.array([.5, .5])), np.array([.5, .5]), capacity=1.0)
    assert r["queue_size_per_week"] == [1]  # week 1 has nobody left; no empty-week precision


def test_multiplicity_equals_duplicated_account():
    month, score = hand_month()
    mult = np.array([1, 1, 2, 1, 1])  # account C drawn twice
    via_mult = evaluate(month, rank_order(month, score), score, 0.4, mult)
    rows = [("A", W0, .9), ("B", W0, .8), ("C", W0, .7), ("C2", W0, .7), ("D", W0, .6),
            ("A", W1, .99), ("E", W1, .95), ("B", W1, .8), ("C", W1, .7), ("C2", W1, .7),
            ("D", W1, .6)]
    dup = Month.from_rows([r[0] for r in rows], [r[1] for r in rows], [10] * 11,
                          [int(r[0] in {"A", "C", "C2", "E"}) for r in rows])
    s2 = np.array([r[2] for r in rows])
    explicit = evaluate(dup, rank_order(dup, s2), s2, 0.4)
    for m in ("flag_precision", "base_rate", "flag_recall", "flag_pr_auc"):
        assert via_mult[m] == pytest.approx(explicit[m])


def synthetic(n=3000, weeks=4, seed=1):
    rng = np.random.default_rng(seed)
    label = rng.random(n) < 0.05
    acc, wk, good, noise = [], [], [], []
    for a in range(n):
        for w in range(weeks):
            if rng.random() < 0.7:
                acc.append(a)
                wk.append(w)
                good.append(label[a] * 2 + rng.normal())
                noise.append(rng.normal())
    month = Month.from_rows(acc, wk, np.full(len(acc), 10), label[np.array(acc)].astype(int))
    return month, np.array(good), np.array(noise)


def test_identical_models_have_zero_paired_difference():
    month, good, _ = synthetic(n=600)
    res = bootstrap(month, {"a": good, "b": good.copy()}, capacity=0.02, n_resamples=50)
    c = compare(res, "a", "b")
    assert c["diff"] == 0 and c["ci95"] == (0.0, 0.0)
    assert c["verdict"] == "no evidence of improvement"


def test_informative_model_beats_noise_and_interval_contains_point():
    month, good, noise = synthetic()
    res = bootstrap(month, {"good": good, "noise": noise}, capacity=0.02, n_resamples=200)
    assert compare(res, "good", "noise")["verdict"] == "good beats noise"
    lo, hi = np.percentile(res["draws"]["good"]["flag_precision"], [2.5, 97.5])
    assert lo <= res["point"]["good"]["flag_precision"] <= hi


def test_assemble_excludes_unknown_and_disabled_unflagged(tmp_path):
    c = dt.datetime(2025, 7, 13)
    feats = pa.table({"account_id": ["a", "a", "b", "c", "d", "e"],
                      "split": ["train"] * 6, "cutoff": [c, c + dt.timedelta(days=7), c, c, c, c],
                      "games_in_window": [10] * 6})
    pq.write_table(feats, tmp_path / "f.parquet")
    created = int(dt.datetime(2025, 7, 3, tzinfo=dt.UTC).timestamp() * 1000)  # UTC, not local
    labels = pa.table({"account_id": ["a", "b", "c", "d"],
                       "outcome": ["ok", "ok", "not_returned", "ok"],
                       "tos_violation": [1, 0, None, 0], "disabled": [0, 1, None, 0],
                       "created_at_ms": [created, created, None, created]})
    con = duckdb.connect()
    report = assemble(con, tmp_path / "f.parquet", labels)
    reasons = {(e["reason"], e["accounts"]) for e in report["excluded"]}
    assert reasons == {("disabled_unflagged", 1), ("label_not_returned", 1), ("no_label_row", 1)}
    rows = con.execute("SELECT account_id, label, weight, account_age_days FROM model_table "
                       "ORDER BY account_id, cutoff").fetchall()
    assert rows == [("a", 1, 0.5, 10), ("a", 1, 0.5, 17), ("d", 0, 1.0, 10)]


def test_missing_label_balance_flags_a_shifted_feature(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    n = 400
    ids = [f"u{i}" for i in range(n)]
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    y[300:] += 1.0  # the unlabelled accounts differ on y only
    pq.write_table(pa.table({"account_id": ids, "split": ["train"] * n,
                             "cutoff": [dt.datetime(2025, 7, 13)] * n,
                             "games_in_window": [10] * n, "x": x, "y": y}), tmp_path / "f.parquet")
    labels = pa.table({"account_id": ids[:300], "outcome": ["ok"] * 300,
                       "tos_violation": [0] * 300, "disabled": [0] * 300,
                       "created_at_ms": [0] * 300})
    con = duckdb.connect()
    assemble(con, tmp_path / "f.parquet", labels)
    bal = missing_label_balance(con, ["x", "y"])["train"]
    assert bal["above_limit"] == ["y"] and abs(bal["smd"]["x"]) < 0.3
