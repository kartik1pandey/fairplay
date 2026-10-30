import datetime as dt

import numpy as np
import pytest

from fairplay.evaluation.queue import Month, evaluate, rank_order
from fairplay.fairness import audit as fa
from fairplay.models.data import FEATURE_COLUMNS, Split
from fairplay.runs import claim_evaluation

W0 = "2025-08-03"


def test_cap_walks_the_list_and_skips_high_rated_once_full():
    # 10 eligible, k = ceil(0.5 x 10) = 5; 2200+ share s = 4/10 -> cap = ceil(1.0 x 5 x 0.4) = 2
    names = list("ABCDEFGHIJ")
    high = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # A-D are 2200+ and rank highest
    month = Month.from_rows(names, [W0] * 10, [10] * 10, [1, 0, 1, 0, 1, 0, 0, 1, 0, 0], high)
    score = np.arange(10, 0, -1, dtype=float)  # A best ... J worst
    order = rank_order(month, score)
    uncapped = evaluate(month, order, score, capacity=0.5)
    capped = evaluate(month, order, score, capacity=0.5, cap_multiplier=1.0)
    assert np.flatnonzero(uncapped["queued"]).tolist() == [0, 1, 2, 3, 4]  # A B C D E
    assert np.flatnonzero(capped["queued"]).tolist() == [0, 1, 4, 5, 6]  # A B, skip C D, E F G
    assert capped["capped_share_per_week"] == [0.4]
    assert capped["queue_size_per_week"] == [5]  # still full: cap never shortens the queue
    assert uncapped["flag_precision"] == pytest.approx(3 / 5)
    assert capped["flag_precision"] == pytest.approx(2 / 5)


def test_cap_shortens_queue_only_when_list_runs_out():
    month = Month.from_rows(list("ABC"), [W0] * 3, [10] * 3, [0, 0, 0], [1, 1, 0])
    score = np.array([3.0, 2.0, 1.0])
    r = evaluate(month, rank_order(month, score), score, capacity=1.0, cap_multiplier=1.0)
    # k = 3, cap = ceil(3 x 2/3) = 2 -> A, B, C all fit
    assert r["queue_size_per_week"] == [3]


def test_groups_follow_design_boundaries():
    assert [fa.rating_group(r) for r in (1199, 1200, 1599, 1600, 2199, 2200)] == [
        "<1200", "1200-1599", "1200-1599", "1600-1999", "2000-2199", "2200+"]
    assert [fa.age_group(d) for d in (29, 30, 365, 366, np.nan)] == [
        "<30d", "30-365d", "30-365d", ">1y", "unknown"]
    assert [fa.activity_group(g) for g in (10, 19, 20, 49, 50)] == [
        "10-19", "10-19", "20-49", "20-49", "50+"]
    th = {1400.0: 100.0}
    assert fa.improver_group(np.nan, 1400.0, th) == "insufficient_history"
    assert fa.improver_group(150, 1400.0, th) == "fast_improver"
    assert fa.improver_group(100, 1400.0, th) == "not_fast"


def audit_split(n=800, seed=2):
    """2200+ accounts get high scores regardless of label: the uncapped queue over-selects them."""
    rng = np.random.default_rng(seed)
    acc, cut, y, X, rating = [], [], [], [], []
    for a in range(n):
        strong = rng.random() < 0.15
        flagged = rng.random() < 0.05
        for w in range(2):
            acc.append(f"a{a}")
            cut.append(dt.datetime(2025, 8, 3) + dt.timedelta(weeks=w))
            y.append(int(flagged))
            x = rng.normal(size=len(FEATURE_COLUMNS))
            x[FEATURE_COLUMNS.index("account_age_days")] = 400
            x[FEATURE_COLUMNS.index("games_in_window")] = 25
            X.append(x)
            rating.append(2300 if strong else 1500)
    rating = np.array(rating, dtype=float)
    split = Split("validation", np.array(X), np.array(y, dtype=np.int8), np.ones(len(y)),
                  np.array(acc), np.array(cut), np.full(len(y), 25),
                  {"rating_at_cutoff": rating, "rating_band": (rating // 200) * 200,
                   "gain_last50": rng.normal(0, 50, len(y))})
    score = np.array(y) * 2 + (rating >= 2200) * 3 + rng.normal(size=len(y))
    return split, score


def test_audit_flags_overselected_group_and_cap_fixes_it():
    split, score = audit_split()
    th = {1400.0: 80.0, 2200.0: 80.0}
    r = fa.audit(split, score, th, capacity=0.05, n_resamples=30)
    assert "rating:2200+" in r["uncapped"]["failing_groups"]
    assert "rating:2200+" not in r["capped"]["failing_groups"]
    capped_share = r["capped"]["share_2200plus_per_week"][0]
    assert capped_share <= 0.15 * 1.5  # near the population share (cap rounds up)
    lo, hi = r["capped"]["selection_rate_among_unflagged"]["overall"]["ci95"]
    assert lo <= r["capped"]["selection_rate_among_unflagged"]["overall"]["rate"] <= hi


def test_diagnostic_entries_do_not_use_validation_budget(tmp_path):
    log = tmp_path / "e.jsonl"
    for i in range(8):
        claim_evaluation("validation", f"m{i}", log=log)
    claim_evaluation("validation", "fairness", log=log, diagnostic=True)  # still allowed
