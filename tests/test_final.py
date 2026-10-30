"""Split restructure (DECISIONS 2026-10-29): July OOF for choices, August scored once."""

import numpy as np
import pytest

from fairplay.dataset.eligibility import SPLITS
from fairplay.models.data import FEATURE_COLUMNS, Split
from fairplay.models.run_baselines import load_oof, row_digest, save_oof, take
from fairplay.models.run_final import reliability
from fairplay.runs import BudgetExceeded, claim_evaluation


def toy(n=6, name="train"):
    X = np.zeros((n, len(FEATURE_COLUMNS)))
    return Split(name, X, np.zeros(n, np.int8), np.ones(n), np.array([f"a{i}" for i in range(n)],
                 dtype=object), np.repeat([1, 2], n // 2), np.full(n, 10))


def test_only_train_and_test_months():
    assert SPLITS == {"train": "2025-07", "test": "2025-08"}


def test_oof_scores_refuse_a_different_row_set(tmp_path):
    s = toy()
    save_oof(tmp_path, s, {"rule": np.arange(6.0)})
    assert load_oof(tmp_path, s)["rule"].tolist() == list(range(6))
    shuffled = take(s, np.array([1, 0, 2, 3, 4, 5]))
    assert row_digest(shuffled) != row_digest(s)
    with pytest.raises(SystemExit, match="OOF rows differ"):
        load_oof(tmp_path, shuffled)


def test_reliability_is_account_weighted():
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 1, 1, 1])
    w = np.array([3.0, 1.0, 1.0, 1.0])
    r = reliability(p, y, w, bins=2)
    assert r["curve"][0]["observed_flag_share"] == pytest.approx(0.25)
    assert r["brier"] == pytest.approx((3 * 0.01 + 0.81 + 2 * 0.01) / 6)


def test_test_month_is_scored_once(tmp_path):
    log = tmp_path / "e.jsonl"
    with pytest.raises(BudgetExceeded):
        claim_evaluation("test", "early", log=log)
    claim_evaluation("test", "final", final=True, log=log)
    with pytest.raises(BudgetExceeded, match="re-run"):
        claim_evaluation("test", "again", final=True, log=log)


def test_integer_group_codes_give_the_same_folds_as_string_ids():
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(0)
    ids = np.array([f"acc-{i:05d}" for i in rng.integers(0, 400, 2000)], dtype=object)
    s = toy(2000)
    s.account_id = ids
    X = np.zeros((2000, 1))
    a = [p for _, p in GroupKFold(5, shuffle=True, random_state=3).split(X, groups=ids)]
    b = [p for _, p in GroupKFold(5, shuffle=True, random_state=3).split(X, groups=s.groups())]
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_models_never_see_excluded_features():
    from fairplay.models.data import EXCLUDED_FROM_MODELS

    s = toy()
    s.X[:] = 7.0
    age = FEATURE_COLUMNS.index("account_age_days")
    assert "account_age_days" in EXCLUDED_FROM_MODELS
    assert np.isnan(s.model_X[:, age]).all() and (s.X[:, age] == 7.0).all()  # data kept
