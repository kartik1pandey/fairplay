"""Fairness mitigation, step 1-2 of the pre-registered order (design §9), on July OOF only.

The July audit failed the 2x criterion for young accounts (< 30 d: 7.9x; 30-365 d: 2.5x) and
fast improvers (3.1x). The order is: (1) find the feature driving it, (2) remove it if it has no
behavioural meaning, (3) only then group thresholds, which is a policy choice for someone else.

Each variant refits LightGBM with the frozen tier-1 hyperparameters and the comparison folds
(seed + 1), with the dropped features set to NaN (a constant column, so LightGBM never splits on
it). Reported per variant: flag-precision@k vs the full model (paired), and the capped audit.
Dropping the trajectory features is a diagnostic only: they carry behavioural signal.

uv run python -m fairplay.fairness.run_mitigation
"""

import argparse
import dataclasses
import json

import joblib
import numpy as np

from fairplay.evaluation.bootstrap import bootstrap, compare
from fairplay.fairness.audit import AUDIT_COLUMNS, audit, fast_improver_thresholds
from fairplay.models.data import FEATURE_COLUMNS, Split, load_split
from fairplay.models.lgbm import grouped_cv
from fairplay.models.run_baselines import TABLE, load_oof
from fairplay.models.run_tier1 import latest_run
from fairplay.runs import new_run, save_metrics

VARIANTS = {
    "drop_account_age": ["account_age_days"],
    "drop_trajectory (diagnostic)": ["window_rating_change", "largest_daily_gain"],
}
WATCH = ("account_age:<30d", "account_age:30-365d", "improver:fast_improver")


def without(split: Split, features: list[str]) -> Split:
    X = split.X.copy()
    X[:, [FEATURE_COLUMNS.index(f) for f in features]] = np.nan
    return dataclasses.replace(split, X=X)


def brief(a: dict) -> dict:
    sel = a["capped"]["selection_rate_among_unflagged"]
    return {"flag_precision_capped": a["capped"]["flag_precision"],
            "failing_groups": a["capped"]["failing_groups"],
            "ratio_to_overall": {g: sel[g]["ratio_to_overall"] for g in WATCH}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--capacity", type=float, default=0.005)
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train = load_split(TABLE, "train", AUDIT_COLUMNS)
    tier1_dir = latest_run("tier1")
    params = joblib.load(tier1_dir / "models.joblib")["lgbm"].params
    oof = {"full": load_oof(tier1_dir, train)["lgbm"]}
    run = new_run("mitigation", vars(args) | {"tier1_run": tier1_dir.name, "variants": VARIANTS},
                  [TABLE])
    for name, feats in VARIANTS.items():
        oof[name] = grouped_cv(params, without(train, feats), args.seed + 1)[0]
        print("fitted", name, flush=True)

    thresholds = fast_improver_thresholds(train)
    res = bootstrap(train.month(), oof, args.capacity, args.resamples, args.seed)
    report = {"scores": "July out-of-fold, frozen tier-1 hyperparameters", "variants": {}}
    for name, s in oof.items():
        report["variants"][name] = {
            "dropped": VARIANTS.get(name, []),
            "vs_full": None if name == "full" else compare(res, name, "full"),
            "audit": brief(audit(train, s, thresholds, args.capacity, args.resamples, args.seed)),
        }
    save_metrics(run, report)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
