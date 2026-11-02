"""Tabular bake-off: can another model family beat tier-1 LightGBM on July? (research)

Same rows, same folds, same metric as run_tier1:
- each challenger is tuned by Optuna on grouped folds `seed` (50 trials, LightGBM's budget),
- then scored out-of-fold on grouped folds `seed + 1`, the folds LightGBM, logistic regression
  and the rule were compared on (their OOF scores are loaded from the latest tier1 run),
- every challenger is compared with LightGBM by the paired account bootstrap (design §8 rule:
  CI above zero and a gain of at least 2 points).

August was already scored once and cannot pick a winner (design §4), so this is July-only and
labelled exploratory. A challenger that beats LightGBM here would be frozen and tested once on
Experiment P, the next untouched data.

uv run --group research python -m fairplay.models.run_bakeoff --models xgb catboost mlp
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
from scipy.stats import rankdata

from fairplay.evaluation.bootstrap import bootstrap, compare, summary
from fairplay.models.bakeoff import SPECS, grouped_oof, tune
from fairplay.models.data import EXCLUDED_FROM_MODELS, FEATURE_COLUMNS, load_split
from fairplay.models.run_baselines import TABLE, load_oof, row_digest, save_oof, scorable_subset
from fairplay.models.run_tier1 import latest_run
from fairplay.runs import RUNS, new_run, save_metrics

TREES = ("lgbm", "xgb", "catboost")


def rank_average(scores: dict[str, np.ndarray], names) -> np.ndarray:
    """Mean of each model's rank: a scale-free ensemble that needs no extra fitting."""
    return np.mean([rankdata(scores[n]) / len(scores[n]) for n in names], axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["xgb", "catboost", "mlp", "tabpfn"],
                   choices=sorted(SPECS))
    p.add_argument("--capacity", type=float, default=0.005)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train = load_split(TABLE, "train")
    tier1_dir = latest_run("tier1")
    oof = load_oof(tier1_dir, train)
    run = new_run("bakeoff", vars(args) | {"features": FEATURE_COLUMNS,
                                           "excluded_from_models": list(EXCLUDED_FROM_MODELS),
                                           "tier1_run": tier1_dir.name}, [TABLE])
    storage = f"sqlite:///{(RUNS / 'optuna.sqlite3').as_posix()}"
    digest = row_digest(train)[:16]
    models, skipped = {}, {}
    for name in args.models:
        if name == "tabpfn" and not os.environ.get("TABPFN_TOKEN"):
            skipped[name] = "weights need a licence accepted by the user (TABPFN_TOKEN unset)"
            continue
        spec = SPECS[name]
        t0 = time.time()
        params, info = tune(spec, train, args.trials, args.seed, storage,
                            f"bakeoff_{name}_{digest}_seed{args.seed}")
        t1 = time.time()
        # comparison OOF cached per (rows, seed, params): a run killed after this step resumes
        key = hashlib.sha256(json.dumps([digest, args.seed + 1, params], sort_keys=True,
                                        default=str).encode()).hexdigest()[:16]
        cache = RUNS / "bakeoff_oof_cache" / f"{name}_{key}.npy"
        if cache.exists():
            oof[name] = np.load(cache)
        else:
            oof[name] = grouped_oof(spec, params, train, args.seed + 1)
            cache.parent.mkdir(exist_ok=True)
            np.save(cache.with_suffix(".tmp.npy"), oof[name])
            os.replace(cache.with_suffix(".tmp.npy"), cache)
        models[name] = {"best_params": params, "tune_seconds_this_session": round(t1 - t0),
                        "oof_seconds": round(time.time() - t1)} | info
        print(f"{name}: tuned ({info['n_trials']} trials), OOF in "
              f"{models[name]['oof_seconds']} s", flush=True)

    trees = [n for n in TREES if n in oof]
    if len(trees) > 1:
        oof["trees_rank_avg"] = rank_average(oof, trees)
    save_oof(run, train, oof)
    ok, scores, n_excluded = scorable_subset(train, oof)
    res = bootstrap(ok.month(), scores, args.capacity, args.resamples, args.seed)
    challengers = [m for m in scores if m not in ("rule", "logreg", "lgbm", "lgbm_cw")]
    comparisons = {f"{m}_vs_lgbm": compare(res, m, "lgbm") for m in challengers}
    comparisons |= {f"{m}_vs_logreg": compare(res, m, "logreg") for m in challengers}
    winners = [m for m in challengers
               if comparisons[f"{m}_vs_lgbm"]["verdict"] == f"{m} beats lgbm"]
    metrics = {
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "trials"}
                   for k, v in models.items()},
        "skipped": skipped,
        "ensemble": {"trees_rank_avg": trees} if "trees_rank_avg" in oof else {},
        "july_oof_rows_excluded_unscorable": n_excluded,
        "july_oof": summary(res),
        "comparisons": comparisons,
        "beats_lgbm": winners,
        "kept_tier1_model": winners[0] if len(winners) == 1 else ("lgbm" if not winners
                                                                   else "tie: see comparisons"),
    }
    (run / "trials.json").write_text(json.dumps({k: v.get("trials", []) for k, v in
                                                 models.items()}, indent=2))
    save_metrics(run, metrics)  # last: marks the run complete
    print(json.dumps({"run": run.name, "skipped": skipped,
                      "precision": {m: v["flag_precision"]["value"]
                                    for m, v in metrics["july_oof"].items()},
                      "comparisons": {k: v["verdict"] for k, v in comparisons.items()}},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
