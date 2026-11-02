"""Challenger models for the tabular bake-off (research, not part of the locked design).

Each challenger is a `Spec`: a search space and a fit-and-predict function. All of them go through
the same `grouped_oof` as LightGBM did: grouped 5-fold CV by account, early stopping (where the
model has it) on a 10% ACCOUNT holdout carved out of the training fold, so the scored fold never
influences training. Tuning uses folds `seed`, the comparison folds `seed + 1` (as in run_tier1).

- XGBoost and CatBoost: the two other gradient-boosted tree libraries. Native missing values.
- MLP: a small PyTorch network on the logistic-regression inputs (median imputation, missing
  indicators, quantile-normalised). Tests whether a non-tree learner closes the gap.
- TabPFN: a pretrained transformer that does "in-context" prediction on tables; no tuning, no
  sample weights, and a context of at most `TABPFN_CONTEXT` rows (sampled by account), which is
  a handicap the results must state.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import optuna
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from fairplay.models.baselines import LR_FEATURES
from fairplay.models.data import FEATURE_COLUMNS, Split, group_codes

THREADS = 8
MAX_ROUNDS = 3000
EARLY_STOP = 100
TABPFN_CONTEXT = 10_000
LR_COLS = [FEATURE_COLUMNS.index(f) for f in LR_FEATURES]


@dataclass
class Fold:
    X: np.ndarray
    y: np.ndarray
    w: np.ndarray
    groups: np.ndarray


def stop_split(fold: Fold, seed: int) -> tuple[Fold, Fold]:
    """10% of the fold's accounts, held out only to choose when to stop."""
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    a, b = next(inner.split(fold.X, fold.y, fold.groups))
    take = lambda i: Fold(fold.X[i], fold.y[i], fold.w[i], fold.groups[i])  # noqa: E731
    return take(a), take(b)


# --- XGBoost -------------------------------------------------------------------------------------

def xgb_suggest(trial: optuna.Trial) -> dict:
    return {
        "eta": trial.suggest_float("eta", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
    }


def xgb_fit_predict(params: dict, fold: Fold, X_pred: np.ndarray, seed: int) -> np.ndarray:
    import xgboost as xgb

    fit, stop = stop_split(fold, seed)
    full = {"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
            "nthread": THREADS, "seed": seed} | params
    d_fit = xgb.DMatrix(fit.X, fit.y, weight=fit.w)
    d_stop = xgb.DMatrix(stop.X, stop.y, weight=stop.w)
    booster = xgb.train(full, d_fit, MAX_ROUNDS, evals=[(d_stop, "stop")],
                        early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    return booster.predict(xgb.DMatrix(X_pred),
                           iteration_range=(0, booster.best_iteration + 1))


# --- CatBoost ------------------------------------------------------------------------------------

def cat_suggest(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-2, 10.0, log=True),
    }


def cat_fit_predict(params: dict, fold: Fold, X_pred: np.ndarray, seed: int) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool

    fit, stop = stop_split(fold, seed)
    model = CatBoostClassifier(iterations=MAX_ROUNDS, eval_metric="PRAUC",
                               early_stopping_rounds=EARLY_STOP, thread_count=THREADS,
                               random_seed=seed, verbose=False, allow_writing_files=False,
                               **params)
    model.fit(Pool(fit.X, fit.y, weight=fit.w), eval_set=Pool(stop.X, stop.y, weight=stop.w),
              use_best_model=True)
    return model.predict_proba(X_pred)[:, 1]


# --- MLP -----------------------------------------------------------------------------------------

def mlp_suggest(trial: optuna.Trial) -> dict:
    return {
        "width": trial.suggest_categorical("width", [32, 64, 128, 256]),
        "depth": trial.suggest_int("depth", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    }


def mlp_inputs() -> Pipeline:
    """Median imputation with missing indicators (as logistic regression), then a quantile
    transform to a normal shape: several features are heavy-tailed (games per active day,
    rating gains) and a network trains badly on raw tails; trees do not care, networks do."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("quantile", QuantileTransformer(output_distribution="normal", subsample=100_000,
                                         random_state=0)),
    ])


def mlp_fit_predict(params: dict, fold: Fold, X_pred: np.ndarray, seed: int) -> np.ndarray:
    """A small PyTorch MLP trained on account-weighted binary cross-entropy. Early stopping on
    the account holdout (weighted PR-AUC, patience 5 epochs); the best epoch's weights are kept."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    fit, stop = stop_split(fold, seed)
    pre = mlp_inputs().fit(fit.X[:, LR_COLS])
    as_t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    Xf, Xs = as_t(pre.transform(fit.X[:, LR_COLS])), as_t(pre.transform(stop.X[:, LR_COLS]))
    yf, wf = as_t(fit.y), as_t(fit.w / fit.w.mean())

    layers, d = [], Xf.shape[1]
    for _ in range(params["depth"]):
        layers += [nn.Linear(d, params["width"]), nn.ReLU(), nn.Dropout(params["dropout"])]
        d = params["width"]
    net = nn.Sequential(*layers, nn.Linear(d, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=params["lr"],
                            weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    gen = torch.Generator().manual_seed(seed)
    best, best_state, stale = -1.0, None, 0
    for _ in range(100):
        net.train()
        for i in torch.randperm(len(yf), generator=gen).split(1024):
            opt.zero_grad()
            loss = (loss_fn(net(Xf[i]).squeeze(1), yf[i]) * wf[i]).mean()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            ap = average_precision_score(stop.y, net(Xs).squeeze(1).numpy(),
                                         sample_weight=stop.w)
        if ap > best:
            best, stale = ap, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= 5:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(as_t(pre.transform(X_pred[:, LR_COLS]))).squeeze(1).numpy()


# --- TabPFN --------------------------------------------------------------------------------------

def tabpfn_fit_predict(params: dict, fold: Fold, X_pred: np.ndarray, seed: int) -> np.ndarray:
    """Context = every row of a random sample of accounts, up to TABPFN_CONTEXT rows. Sampling
    accounts (not rows) keeps histories whole. No sample weights: TabPFN does not take them."""
    from tabpfn import TabPFNClassifier

    rng = np.random.default_rng(seed)
    accounts = rng.permutation(np.unique(fold.groups))
    sizes = np.bincount(fold.groups)[accounts]
    keep = accounts[np.cumsum(sizes) <= TABPFN_CONTEXT]
    idx = np.flatnonzero(np.isin(fold.groups, keep))
    model = TabPFNClassifier(device="cpu", random_state=seed, n_estimators=params.get("n", 4),
                             ignore_pretraining_limits=True)
    model.fit(fold.X[idx][:, LR_COLS], fold.y[idx])
    out = np.empty(len(X_pred))
    for start in range(0, len(X_pred), 5000):
        out[start:start + 5000] = model.predict_proba(X_pred[start:start + 5000, LR_COLS])[:, 1]
    return out


@dataclass(frozen=True)
class Spec:
    name: str
    fit_predict: Callable[[dict, Fold, np.ndarray, int], np.ndarray]
    suggest: Callable[[optuna.Trial], dict] | None  # None = no tuning


SPECS = {
    "xgb": Spec("xgb", xgb_fit_predict, xgb_suggest),
    "catboost": Spec("catboost", cat_fit_predict, cat_suggest),
    "mlp": Spec("mlp", mlp_fit_predict, mlp_suggest),
    "tabpfn": Spec("tabpfn", tabpfn_fit_predict, None),
}


def grouped_oof(spec: Spec, params: dict, split: Split, seed: int, n_folds: int = 5,
                X: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold scores, folds grouped by account: the same folds lgbm.grouped_cv makes."""
    X = split.model_X if X is None else X
    oof = np.empty(len(split.y))
    codes = group_codes(split.account_id)
    folds = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fit_idx, pred_idx in folds.split(X, split.y, groups=split.groups()):
        fold = Fold(X[fit_idx], split.y[fit_idx], split.weight[fit_idx], codes[fit_idx])
        oof[pred_idx] = spec.fit_predict(params, fold, X[pred_idx], seed)
    return oof


def tune(spec: Spec, train: Split, n_trials: int, seed: int, storage: str | None,
         study_name: str | None) -> tuple[dict, dict]:
    """Optuna TPE on weighted out-of-fold PR-AUC, resumable from `storage` (as lgbm.tune)."""
    if spec.suggest is None:
        return {}, {"n_trials": 0}
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        oof = grouped_oof(spec, spec.suggest(trial), train, seed)
        return float(average_precision_score(train.y, oof, sample_weight=train.weight))

    study = optuna.create_study(direction="maximize", storage=storage, study_name=study_name,
                                load_if_exists=storage is not None,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done)
    return study.best_params, {"best_cv_pr_auc": study.best_value, "n_trials": len(study.trials),
                               "trials": [{"value": t.value, "params": t.params}
                                          for t in study.trials]}
