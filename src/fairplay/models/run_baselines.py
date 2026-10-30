"""Fit both baselines on July (train) and compare them on July out-of-fold scores.

There is no validation month (September was dropped, DECISIONS 2026-10-29): every model choice is
made by account-grouped cross-validation inside July, and August (test) is scored once, by
run_final. The comparison OOF uses folds with seed + 1, so they differ from the folds that chose C.
The rule has no fitted parameters beyond picking one of two variants, so its July score is used
as is (that small in-sample choice only favours the baseline).

uv run python -m fairplay.models.run_baselines --capacity 0.005
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from fairplay.evaluation.bootstrap import bootstrap, compare, summary
from fairplay.models.baselines import (
    LR_FEATURES,
    choose_rule,
    fit_logistic,
    grouped_oof,
    logistic_pipeline,
)
from fairplay.models.data import EXCLUDED_FROM_MODELS, FEATURE_COLUMNS, Split, load_split
from fairplay.runs import new_run, save_metrics

TABLE = Path("data/processed/model_table_S0.parquet")


def take(split: Split, mask: np.ndarray) -> Split:
    return Split(split.name, split.X[mask], split.y[mask], split.weight[mask],
                 split.account_id[mask], split.cutoff[mask], split.games[mask],
                 {k: v[mask] for k, v in split.extra.items()})


def scorable_subset(split: Split, scores: dict[str, np.ndarray]) -> tuple[Split, dict, int]:
    """Design §0.11: rows any model cannot score are excluded from every paired comparison."""
    ok = np.logical_and.reduce([np.isfinite(s) for s in scores.values()])
    return take(split, ok), {k: v[ok] for k, v in scores.items()}, int((~ok).sum())


def row_digest(split: Split) -> str:
    """Identifies the exact rows (and order) OOF scores belong to, so runs can share them."""
    h = hashlib.sha256()
    for a, c in zip(split.account_id, split.cutoff, strict=True):
        h.update(f"{a}|{c}\n".encode())
    return h.hexdigest()


def save_oof(run: Path, split: Split, scores: dict[str, np.ndarray]) -> None:
    np.savez(run / "oof.npz", digest=row_digest(split), **scores)


def load_oof(run: Path, split: Split) -> dict[str, np.ndarray]:
    z = np.load(run / "oof.npz")
    if str(z["digest"]) != row_digest(split):
        raise SystemExit(f"{run.name}: OOF rows differ from the current model table; rerun it")
    return {k: z[k] for k in z.files if k != "digest"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--capacity", type=float, default=0.005)
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train = load_split(TABLE, "train")
    run = new_run("baselines", vars(args) | {"features": FEATURE_COLUMNS,
                                             "excluded_from_models": list(EXCLUDED_FROM_MODELS)},
                  [TABLE])
    rule, rule_info = choose_rule(train, args.capacity)
    logreg, lr_info = fit_logistic(train, seed=args.seed)

    oof = {"rule": rule.score(train),
           "logreg": grouped_oof(lambda: logistic_pipeline(logreg.C), train, seed=args.seed + 1)}
    save_oof(run, train, oof)
    ok, scores, n_excluded = scorable_subset(train, oof)
    res = bootstrap(ok.month(), scores, args.capacity, args.resamples, args.seed)
    metrics = {
        "rule": rule_info | {"feature": rule.feature, "sign": rule.sign},
        "logreg": {k: v for k, v in lr_info.items() if k != "vif"},
        "vif": dict(zip(LR_FEATURES, lr_info["vif"], strict=True)),
        "july_oof_rows_excluded_unscorable": n_excluded,
        "july_oof": summary(res),
        "weekly_precision": {m: res["point"][m]["weekly_precision"] for m in scores},
        "k_per_week": res["point"]["rule"]["k_per_week"],
        "comparison": compare(res, "logreg", "rule"),
    }
    joblib.dump({"rule": rule, "logreg": logreg}, run / "models.joblib")
    save_metrics(run, metrics)  # last: marks the run complete
    print(json.dumps({"run": run.name, "july_oof": metrics["july_oof"],
                      "comparison": metrics["comparison"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
