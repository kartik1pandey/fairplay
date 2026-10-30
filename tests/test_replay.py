"""Monitoring replay (design §11): weekly operations without labels."""

import numpy as np

from fairplay.models.data import FEATURE_COLUMNS, Split
from fairplay.monitoring.replay import weekly_ops


def week(n, high_rated):
    X = np.zeros((n, len(FEATURE_COLUMNS)))
    rating = np.where(np.arange(n) < high_rated, 2300.0, 1500.0)
    return Split("test", X, np.zeros(n, np.int8), np.ones(n),
                 np.array([f"a{i}" for i in range(n)], dtype=object),
                 np.full(n, np.datetime64("2025-08-03")), np.full(n, 10),
                 extra={"rating_at_cutoff": rating})


def test_cap_holds_back_high_rated_accounts_ranked_first():
    n = 1000  # k = ceil(0.5% x 1000) = 5; 2200+ share 10% -> cap ceil(1.0 x 5 x 0.1) = 1
    split = week(n, high_rated=100)
    score = -np.arange(n, dtype=float)  # the 100 high-rated accounts rank first
    ops = weekly_ops(split, score, 0.005)
    assert ops == {"eligible": n, "k": 5, "queued": 5, "high_rated_held_back": 4}


def test_no_high_rated_accounts_nothing_held_back():
    ops = weekly_ops(week(1000, high_rated=0), np.arange(1000.0), 0.005)
    assert ops["high_rated_held_back"] == 0 and ops["queued"] == 5
