import numpy as np
import pytest

from fairplay.models.bakeoff import Fold, stop_split
from fairplay.models.run_bakeoff import rank_average


def toy_fold(n_accounts=60, rows=4, seed=0) -> Fold:
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_accounts), rows)
    X = rng.normal(size=(len(groups), 16))
    y = (X[:, 0] > 1.0).astype(np.int8)
    return Fold(X, y, np.full(len(groups), 1 / rows), groups)


def test_stop_split_keeps_each_account_on_one_side():
    fit, stop = stop_split(toy_fold(), seed=0)
    assert not set(fit.groups) & set(stop.groups)
    assert len(fit.y) + len(stop.y) == 240


def test_rank_average_is_scale_free():
    a = np.array([0.1, 0.5, 0.9])
    scores = {"a": a, "b": a * 1000 - 7}
    assert np.allclose(rank_average(scores, ["a", "b"]), rank_average({"a": a}, ["a"]))


def test_rank_average_breaks_disagreement_by_mean_rank():
    scores = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([3.0, 2.0, 1.0])}
    assert np.allclose(rank_average(scores, ["a", "b"]), 2 / 3)


@pytest.mark.parametrize("name", ["xgb", "mlp"])
def test_challenger_learns_an_obvious_signal(name):
    pytest.importorskip("xgboost" if name == "xgb" else "torch")
    from fairplay.models.bakeoff import SPECS

    params = {"xgb": {"eta": 0.3, "max_depth": 3, "min_child_weight": 0.01, "subsample": 1.0,
                      "colsample_bytree": 1.0, "lambda": 1.0},
              "mlp": {"width": 16, "depth": 1, "dropout": 0.0, "weight_decay": 0.0,
                      "lr": 1e-2}}[name]
    fold = toy_fold(n_accounts=200)
    test = toy_fold(n_accounts=50, seed=1)
    score = SPECS[name].fit_predict(params, fold, test.X, 0)
    assert score[test.y == 1].mean() > score[test.y == 0].mean()
