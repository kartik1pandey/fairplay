"""Tier 1: LightGBM on the 16 features (design §7, §0 item 8).

- Native missing values (NaN), no imputation.
- Hyperparameters: Optuna TPE, `n_trials` trials, each scored by grouped 5-fold CV within July,
  objective = weighted out-of-fold PR-AUC (same procedure as logistic regression).
- Early stopping without peeking: inside each training fold, 10% of the ACCOUNTS are held out
  only to choose the number of trees; the outer fold that gets scored never influences training.
  The final model uses the mean best iteration of the folds, trained on all of July.
- Class weights are a variant (scale_pos_weight = negatives / positives), not the default:
  for ranking at a fixed capacity they mostly change calibration, not order.
- Isotonic calibration fitted on July out-of-fold scores, used only to report P(flag by S0).
  The queue ranks by the raw score: isotonic is a step function and would create ties.
- deterministic=True, fixed thread count and seeds: the same run gives the same model.
"""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from fairplay.models.data import FEATURE_COLUMNS, Split, group_codes

THREADS = 8
MAX_ROUNDS = 3000
EARLY_STOP = 100


def base_params(seed: int) -> dict:
    return {"objective": "binary", "metric": "average_precision", "verbosity": -1,
            "deterministic": True, "force_row_wise": True, "num_threads": THREADS,
            "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed}


def suggest(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    }


def pos_weight(split: Split) -> float:
    w_pos = split.weight[split.y == 1].sum()
    return float(split.weight[split.y == 0].sum() / w_pos) if w_pos else 1.0


def _fit(params: dict, X, y, w, groups, seed: int, rounds: int | None = None) -> lgb.Booster:
    """Fixed `rounds`, or early stopping on a 10% account holdout drawn from (X, y)."""
    if rounds is not None:
        return lgb.train(params, lgb.Dataset(X, y, weight=w, feature_name=FEATURE_COLUMNS),
                         num_boost_round=rounds)
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    fit_i, stop_i = next(inner.split(X, y, group_codes(groups)))
    train_set = lgb.Dataset(X[fit_i], y[fit_i], weight=w[fit_i], feature_name=FEATURE_COLUMNS)
    stop_set = lgb.Dataset(X[stop_i], y[stop_i], weight=w[stop_i], reference=train_set)
    return lgb.train(params, train_set, num_boost_round=MAX_ROUNDS, valid_sets=[stop_set],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def grouped_cv(params: dict, split: Split, seed: int, n_folds: int = 5) -> tuple[np.ndarray, list]:
    """Out-of-fold scores (folds grouped by account) and each fold's best iteration."""
    oof = np.empty(len(split.y))
    iters = []
    folds = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X = split.model_X
    for fit_idx, pred_idx in folds.split(X, split.y, groups=split.groups()):
        booster = _fit(params, X[fit_idx], split.y[fit_idx], split.weight[fit_idx],
                       split.account_id[fit_idx], seed)
        iters.append(booster.best_iteration or booster.current_iteration())
        oof[pred_idx] = booster.predict(X[pred_idx], num_iteration=iters[-1])
    return oof, iters


@dataclass
class LgbmModel:
    booster: lgb.Booster
    params: dict
    rounds: int
    calibrator: IsotonicRegression
    name: str = "lgbm"
    info: dict = field(default_factory=dict)

    def score(self, split: Split) -> np.ndarray:
        """Raw score used for ranking."""
        return self.booster.predict(split.model_X, raw_score=True)

    def probability(self, split: Split) -> np.ndarray:
        """Calibrated P(flagged at S0), for display and reliability plots only."""
        return self.calibrator.predict(self.booster.predict(split.model_X))

    def contributions(self, split: Split) -> np.ndarray:
        """Per-row feature contributions (TreeSHAP, built into LightGBM): what the model USED,
        never a cause (design §5). Last column is the bias."""
        return self.booster.predict(split.model_X, pred_contrib=True)


def tune(train: Split, n_trials: int, seed: int = 0, storage: str | None = None,
         study_name: str | None = None) -> tuple[dict, optuna.Study]:
    """With `storage` (an SQLite URL) every finished trial is saved, and a re-run with the same
    study_name continues from there instead of starting over. A resumed TPE sampler sees the
    same past trials, but its random stream restarts, so a resumed search is not bit-identical
    to an uninterrupted one; the trial log records exactly what was tried."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = base_params(seed) | suggest(trial)
        oof, iters = grouped_cv(params, train, seed)
        trial.set_user_attr("mean_best_iteration", float(np.mean(iters)))
        return float(average_precision_score(train.y, oof, sample_weight=train.weight))

    study = optuna.create_study(direction="maximize", storage=storage, study_name=study_name,
                                load_if_exists=storage is not None,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done)
    return base_params(seed) | study.best_params | {"bagging_freq": 1}, study


def fit_final(train: Split, params: dict, seed: int = 0, name: str = "lgbm") -> LgbmModel:
    oof, iters = grouped_cv(params, train, seed)
    rounds = max(1, int(round(np.mean(iters))))
    booster = _fit(params, train.model_X, train.y, train.weight, train.account_id, seed,
                   rounds=rounds)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
        oof, train.y, sample_weight=train.weight)
    info = {"oof_pr_auc": float(average_precision_score(train.y, oof, sample_weight=train.weight)),
            "fold_best_iterations": iters, "rounds": rounds}
    return LgbmModel(booster, params, rounds, calibrator, name=name, info=info)


def train_tier1(train: Split, n_trials: int = 50, seed: int = 0, storage: str | None = None,
                study_name: str | None = None) -> tuple[LgbmModel, LgbmModel, dict]:
    """Returns (default model, class-weighted variant, tuning info)."""
    params, study = tune(train, n_trials, seed, storage, study_name)
    plain = fit_final(train, params, seed, name="lgbm")
    weighted = fit_final(train, params | {"scale_pos_weight": pos_weight(train)}, seed,
                         name="lgbm_cw")
    info = {"best_params": params, "best_cv_pr_auc": study.best_value,
            "n_trials": len(study.trials),
            "trials": [{"value": t.value, "params": t.params} for t in study.trials]}
    return plain, weighted, info
