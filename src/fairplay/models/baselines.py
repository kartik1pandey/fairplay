"""Baselines (design §7). Every later model must beat these on the same rows.

Rule: rank by one feature. Candidates are the performance-rating gap (defined for every eligible
row) and a timing rule (low think-time CV = machine-like). The better one on the TRAINING month's
queue metric is frozen; validation data never chooses it.

Logistic regression: median imputation + missing-indicator columns, standardisation, L2.
C is chosen by 5-fold cross-validation grouped by account within the training month (an account's
weekly rows never sit on both sides of a fold), objective = out-of-fold weighted PR-AUC.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fairplay.evaluation.queue import evaluate, rank_order
from fairplay.models.data import EXCLUDED_FROM_MODELS, FEATURE_COLUMNS, Split

LR_FEATURES = [f for f in FEATURE_COLUMNS if f not in EXCLUDED_FROM_MODELS]

RULE_CANDIDATES = {
    "perf_gap": ("performance_rating_gap", +1),
    "timing_cv": ("think_time_cv", -1),  # lower CV = more suspicious
}
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


@dataclass
class RuleModel:
    feature: str
    sign: int
    name: str = "rule"

    def score(self, split: Split) -> np.ndarray:
        s = self.sign * split.column(self.feature)
        return np.where(np.isnan(s), -np.inf, s)  # undefined -> back of the queue


def choose_rule(train: Split, capacity: float) -> tuple[RuleModel, dict]:
    month = train.month()
    results = {}
    for name, (feature, sign) in RULE_CANDIDATES.items():
        s = RuleModel(feature, sign).score(train)
        results[name] = evaluate(month, rank_order(month, s), s, capacity)["flag_precision"]
    best = max(results, key=results.get)
    return RuleModel(*RULE_CANDIDATES[best]), {"train_flag_precision": results, "chosen": best}


def logistic_pipeline(C: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, l1_ratio=0, max_iter=2000)),
    ])


def lr_inputs(split: Split) -> np.ndarray:
    """Model inputs without the excluded columns. Dropped rather than NaN-filled, so the imputer
    never sees an empty column (LightGBM gets NaN columns instead: same effect, same shape)."""
    return split.X[:, [FEATURE_COLUMNS.index(f) for f in LR_FEATURES]]


def grouped_oof(make_model, split: Split, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold scores with folds grouped by account."""
    oof = np.empty(len(split.y))
    folds = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X = lr_inputs(split)
    for fit_idx, pred_idx in folds.split(X, split.y, groups=split.groups()):
        model = make_model()
        model.fit(X[fit_idx], split.y[fit_idx],
                  lr__sample_weight=split.weight[fit_idx])
        oof[pred_idx] = model.predict_proba(X[pred_idx])[:, 1]
    return oof


@dataclass
class LogisticModel:
    C: float
    pipeline: Pipeline
    name: str = "logreg"

    def score(self, split: Split) -> np.ndarray:
        return self.pipeline.predict_proba(lr_inputs(split))[:, 1]


def fit_logistic(train: Split, seed: int = 0) -> tuple[LogisticModel, dict]:
    cv = {}
    for C in C_GRID:
        oof = grouped_oof(lambda C=C: logistic_pipeline(C), train, seed=seed)
        cv[C] = float(average_precision_score(train.y, oof, sample_weight=train.weight))
    best_C = max(cv, key=cv.get)
    final = logistic_pipeline(best_C).fit(lr_inputs(train), train.y,
                                          lr__sample_weight=train.weight)
    return LogisticModel(best_C, final), {"cv_oof_pr_auc": cv, "chosen_C": best_C,
                                          "vif": variance_inflation(lr_inputs(train))}


def variance_inflation(X: np.ndarray) -> list[float | None]:
    """VIF per feature on median-imputed, standardised data (design §5: correlated 7, 8, 9).

    VIF_j = 1 / (1 - R^2 of feature j regressed on the others). Above ~10 means the LR coefficient
    of that feature is unstable; its ranking contribution is still fine.
    """
    Z = SimpleImputer(strategy="median").fit_transform(X)
    keep = Z.std(axis=0) > 0
    Z = (Z[:, keep] - Z[:, keep].mean(0)) / Z[:, keep].std(0)
    out: list[float | None] = [None] * X.shape[1]
    for j_full, j in zip(np.flatnonzero(keep), range(Z.shape[1]), strict=True):
        others = np.delete(Z, j, axis=1)
        coef, *_ = np.linalg.lstsq(others, Z[:, j], rcond=None)
        r2 = 1 - np.var(Z[:, j] - others @ coef) / np.var(Z[:, j])
        out[j_full] = float(1 / max(1 - r2, 1e-12))
    return out
