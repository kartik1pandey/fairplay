import numpy as np
import pytest

from fairplay.models import lgbm
from fairplay.models.data import FEATURE_COLUMNS, Split
from fairplay.models.run_tier1 import choose


def interaction_split(n_accounts=1200, seed=5) -> Split:
    """Flag depends on an INTERACTION (fast clock AND big rating gain), which a linear model
    cannot express; 25% of think_time_cv is missing, and missingness itself is informative."""
    rng = np.random.default_rng(seed)
    acc, X, y = [], [], []
    for a in range(n_accounts):
        base = rng.normal(size=len(FEATURE_COLUMNS))
        for _ in range(rng.integers(1, 5)):
            x = base + rng.normal(scale=0.3, size=len(FEATURE_COLUMNS))
            acc.append(a)
            X.append(x)
    X = np.array(X)
    i, j = FEATURE_COLUMNS.index("instant_move_share"), FEATURE_COLUMNS.index("largest_daily_gain")
    logit = 2.5 * ((X[:, i] > 0.8) & (X[:, j] > 0.8)) - 3
    acc = np.array(acc)
    per_acc = {a: rng.random() < 1 / (1 + np.exp(-logit[acc == a].mean())) for a in set(acc)}
    y = np.array([per_acc[a] for a in acc], dtype=np.int8)
    k = FEATURE_COLUMNS.index("think_time_cv")
    X[(rng.random(len(X)) < 0.25), k] = np.nan
    counts = np.bincount(acc)
    return Split("train", X, y, 1 / counts[acc], acc.astype(str), np.zeros(len(y)),
                 np.full(len(y), 20))


@pytest.fixture(scope="module")
def trained():
    split = interaction_split()
    plain, weighted, info = lgbm.train_tier1(split, n_trials=3, seed=1)
    return split, plain, weighted, info


def test_tuning_records_trials_and_best_params(trained):
    _, plain, _, info = trained
    assert info["n_trials"] == 3 and len(info["trials"]) == 3
    assert {"learning_rate", "num_leaves", "min_child_samples"} <= set(info["best_params"])
    assert plain.rounds >= 1 and len(plain.info["fold_best_iterations"]) == 5


def test_native_nan_scoring_and_calibrated_probabilities(trained):
    split, plain, _, _ = trained
    assert np.isnan(split.X).any()
    s, p = plain.score(split), plain.probability(split)
    assert np.isfinite(s).all()
    assert ((p >= 0) & (p <= 1)).all()


def test_calibration_is_monotone_in_the_raw_score(trained):
    split, plain, _, _ = trained
    raw = plain.booster.predict(split.X)
    order = np.argsort(raw)
    assert np.all(np.diff(plain.calibrator.predict(raw[order])) >= -1e-12)


def test_class_weight_variant_differs(trained):
    split, plain, weighted, _ = trained
    assert "scale_pos_weight" in weighted.params
    assert weighted.params["scale_pos_weight"] == pytest.approx(lgbm.pos_weight(split))
    assert not np.allclose(plain.score(split), weighted.score(split))


def test_contributions_sum_to_raw_score(trained):
    split, plain, _, _ = trained
    contrib = plain.contributions(split)
    assert contrib.shape == (len(split.y), len(FEATURE_COLUMNS) + 1)
    assert np.allclose(contrib.sum(axis=1), plain.score(split))


def test_training_is_deterministic():
    split = interaction_split(n_accounts=300)
    params = lgbm.base_params(7) | {"learning_rate": 0.1, "num_leaves": 15}
    a = lgbm.fit_final(split, params, seed=7)
    b = lgbm.fit_final(split, params, seed=7)
    assert np.array_equal(a.score(split), b.score(split))


def test_stop_rule_freezes_logreg_unless_lgbm_beats_it():
    tie = {"lgbm_vs_logreg": {"verdict": "no evidence of improvement", "diff": 0.01},
           "lgbm_cw_vs_logreg": {"verdict": "no evidence of improvement", "diff": 0.0}}
    assert choose(tie) == "logreg"
    win = {"lgbm_vs_logreg": {"verdict": "lgbm beats logreg", "diff": 0.03},
           "lgbm_cw_vs_logreg": {"verdict": "lgbm_cw beats logreg", "diff": 0.05},
           "lgbm_cw_vs_lgbm": {"verdict": "no evidence of improvement", "diff": 0.02}}
    assert choose(win) == "lgbm"  # bigger point gain is not a win; the ablation must beat lgbm
    win["lgbm_cw_vs_lgbm"]["verdict"] = "lgbm_cw beats lgbm"
    assert choose(win) == "lgbm_cw"



def test_tuning_resumes_from_saved_trials(tmp_path):
    split = interaction_split(n_accounts=300)
    url = f"sqlite:///{(tmp_path / 'o.sqlite3').as_posix()}"
    lgbm.tune(split, 2, seed=0, storage=url, study_name="s")
    _, resumed = lgbm.tune(split, 3, seed=0, storage=url, study_name="s")
    assert len(resumed.trials) == 3  # 2 saved + 1 new, not 3 new
    _, again = lgbm.tune(split, 3, seed=0, storage=url, study_name="s")
    assert len(again.trials) == 3  # already complete: nothing re-run
