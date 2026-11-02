"""Do Maia humanness features add signal? LightGBM with vs without them (research, July).

Rows: the case-control sample from humanness.sample (all flagged accounts, NEG_RATE of the
others). Both models use the tier-1 LightGBM hyperparameters (no new tuning, so neither side
gets an extra search) and the same grouped folds (seed + 1). Weights = tier-1 account weights x
inverse sampling probability, so every metric estimates the full July population.

Metric: inverse-probability-weighted PR-AUC, and the weighted flag-precision in the top
CAPACITY of the weighted population. Interval: paired bootstrap over accounts (1,000 draws).
The weekly queue metric is not used: it needs every eligible row, and the sample has a tenth
of the negatives.

uv run python -m fairplay.deep.run_humanness
"""

import argparse
import json

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from fairplay.deep import humanness as hm
from fairplay.models.data import FEATURE_COLUMNS, group_codes, load_split
from fairplay.models.lgbm import EARLY_STOP, MAX_ROUNDS
from fairplay.models.run_baselines import TABLE
from fairplay.models.run_tier1 import latest_run
from fairplay.runs import new_run, save_metrics

CAPACITY = 0.005


def top_precision(y: np.ndarray, w: np.ndarray, s: np.ndarray, capacity: float = CAPACITY
                  ) -> float:
    """Weighted share of flagged rows among the top `capacity` of the weighted population."""
    order = np.argsort(-s, kind="stable")
    cw = np.cumsum(w[order])
    top = order[: np.searchsorted(cw, capacity * cw[-1]) + 1]
    return float((w[top] * y[top]).sum() / w[top].sum())


def oof(X: np.ndarray, y, w, groups, params: dict, seed: int, names: list[str]) -> np.ndarray:
    out = np.empty(len(y))
    for fit, pred in GroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        rng = np.random.default_rng(seed)
        fit_groups = np.unique(groups[fit])
        stop_g = rng.choice(fit_groups, max(1, len(fit_groups) // 10), replace=False)
        stop = np.isin(groups[fit], stop_g)
        a, b = fit[~stop], fit[stop]
        ds = lgb.Dataset(X[a], y[a], weight=w[a], feature_name=names)
        booster = lgb.train(params, ds, MAX_ROUNDS,
                            valid_sets=[lgb.Dataset(X[b], y[b], weight=w[b], reference=ds)],
                            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        out[pred] = booster.predict(X[pred], num_iteration=booster.best_iteration)
    return out


def paired_bootstrap(y, w, groups, a, b, resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    accounts = np.unique(groups)
    rows_of = np.split(np.argsort(groups, kind="stable"),
                       np.cumsum(np.bincount(groups))[:-1])
    draws = {"pr_auc": [], "top_precision": []}
    for _ in range(resamples):
        idx = np.concatenate([rows_of[g] for g in rng.choice(accounts, len(accounts))])
        yy, ww = y[idx], w[idx]
        draws["pr_auc"].append(average_precision_score(yy, b[idx], sample_weight=ww)
                               - average_precision_score(yy, a[idx], sample_weight=ww))
        draws["top_precision"].append(top_precision(yy, ww, b[idx])
                                      - top_precision(yy, ww, a[idx]))
    point = {"pr_auc": average_precision_score(y, b, sample_weight=w)
             - average_precision_score(y, a, sample_weight=w),
             "top_precision": top_precision(y, w, b) - top_precision(y, w, a)}
    return {k: {"diff": float(point[k]),
                "ci95": [float(x) for x in np.percentile(v, [2.5, 97.5])]}
            for k, v in draws.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train = load_split(TABLE, "train")
    rows, ipw = hm.sample(train)
    games = hm.row_games(train, rows)
    H = hm.row_features(train, rows, games)
    covered = ~np.isnan(H[:, 0])
    tier1 = latest_run("tier1")
    params = json.loads((tier1 / "metrics.json").read_text())["tuning"]["best_params"]
    y, w = train.y[rows], train.weight[rows] * ipw
    groups = group_codes(train.account_id[rows])
    base_X = train.model_X[rows]
    with_X = np.column_stack([base_X, H])
    names = FEATURE_COLUMNS + list(hm.ROW_FEATURES)
    run = new_run("humanness", vars(args) | {"tier1_run": tier1.name, "neg_rate": hm.NEG_RATE,
                                             "games": hm.GAMES, "moves": [hm.FIRST_MOVE,
                                                                          hm.LAST_MOVE]},
                  [TABLE])
    a = oof(base_X, y, w, groups, params, args.seed + 1, FEATURE_COLUMNS)
    b = oof(with_X, y, w, groups, params, args.seed + 1, names)
    metrics = {
        "rows": int(len(rows)), "accounts": int(len(np.unique(groups))),
        "positive_rows": int(y.sum()), "rows_with_humanness": int(covered.sum()),
        "lgbm": {"pr_auc": float(average_precision_score(y, a, sample_weight=w)),
                 "top_precision": top_precision(y, w, a)},
        "lgbm_plus_humanness": {"pr_auc": float(average_precision_score(y, b, sample_weight=w)),
                                "top_precision": top_precision(y, w, b)},
        "difference": paired_bootstrap(y, w, groups, a, b, args.resamples, args.seed),
        "feature_means": {
            "flagged": dict(zip(hm.ROW_FEATURES, np.nanmean(H[y == 1], 0).round(4).tolist(),
                                strict=True)),
            "unflagged": dict(zip(hm.ROW_FEATURES, np.nanmean(H[y == 0], 0).round(4).tolist(),
                                  strict=True))},
    }
    save_metrics(run, metrics)
    print(json.dumps({"run": run.name} | metrics, indent=2))


if __name__ == "__main__":
    main()
