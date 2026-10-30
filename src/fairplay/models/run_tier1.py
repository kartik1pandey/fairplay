"""Tier 1 on July, compared with the baselines on July out-of-fold scores (design §8).

Tuning (50 Optuna trials) uses grouped folds with `seed`; the comparison OOF is rebuilt with
folds `seed + 1`, the same folds the baseline run used, so every model is compared on the same
rows and no model is judged on the folds that chose its hyperparameters. Some optimism remains
(the hyperparameters were still chosen on July), which is why only August (run_final) carries a
claim. If neither LightGBM configuration beats logistic regression, logistic regression is frozen
as the tier-1 model (design §13 stop rule).

uv run python -m fairplay.models.run_tier1 --trials 50
"""

import argparse
import json
from pathlib import Path

import joblib

from fairplay.evaluation.bootstrap import bootstrap, compare, summary
from fairplay.models.data import EXCLUDED_FROM_MODELS, FEATURE_COLUMNS, load_split
from fairplay.models.lgbm import grouped_cv, train_tier1
from fairplay.models.run_baselines import TABLE, load_oof, row_digest, save_oof, scorable_subset
from fairplay.runs import RUNS, new_run, save_metrics


def feature_set(run: Path) -> list:
    return json.loads((run / "config.json").read_text())["config"].get("excluded_from_models", [])


def latest_run(name: str) -> Path:
    """Latest COMPLETED run built on the CURRENT model feature set. metrics.json is written last,
    so a run killed midway is skipped; a run from before a feature change is skipped too, so a
    new baseline can never be paired with an old tier-1 model (or the reverse)."""
    runs = sorted(r for r in RUNS.glob(f"*_{name}") if (r / "metrics.json").exists()
                  and feature_set(r) == list(EXCLUDED_FROM_MODELS))
    if not runs:
        raise SystemExit(f"no completed '{name}' run on the current feature set; run it first")
    return runs[-1]


def choose(comparisons: dict) -> str:
    """Smallest model that survives (design §7): logistic regression unless LightGBM beats it;
    class weights are an ablation, used only if lgbm_cw beats plain lgbm on its own paired test
    (a larger point gain over logreg is not a win: the paired interval decides)."""
    beats = [m for m in ("lgbm", "lgbm_cw")
             if comparisons[f"{m}_vs_logreg"]["verdict"] == f"{m} beats logreg"]
    if not beats:
        return "logreg"
    if beats == ["lgbm_cw"]:
        return "lgbm_cw"
    cw_wins = comparisons["lgbm_cw_vs_lgbm"]["verdict"] == "lgbm_cw beats lgbm"
    return "lgbm_cw" if "lgbm_cw" in beats and cw_wins else "lgbm"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--capacity", type=float, default=0.005)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train = load_split(TABLE, "train")
    baseline_dir = latest_run("baselines")
    baseline_oof = load_oof(baseline_dir, train)
    run = new_run("tier1", vars(args) | {"features": FEATURE_COLUMNS,
                                         "excluded_from_models": list(EXCLUDED_FROM_MODELS),
                                         "baseline_run": baseline_dir.name}, [TABLE])
    # Trials are saved as they finish; a run stopped midway resumes the same study. The study name
    # pins the exact rows and seed, so a changed model table can never resume an old search.
    study = f"tier1_{row_digest(train)[:16]}_seed{args.seed}"
    storage = f"sqlite:///{(RUNS / 'optuna.sqlite3').as_posix()}"
    lgbm, lgbm_cw, tune_info = train_tier1(train, args.trials, args.seed, storage, study)
    tune_info["study_name"] = study
    # The study was tuned with all 16 features; after account age was excluded (fairness step 2)
    # the same hyperparameters are reused, not re-tuned (DECISIONS 2026-10-30).
    tune_info["tuned_with_excluded"] = []

    oof = baseline_oof | {m.name: grouped_cv(m.params, train, args.seed + 1)[0]
                          for m in (lgbm, lgbm_cw)}
    save_oof(run, train, oof)
    ok, scores, n_excluded = scorable_subset(train, oof)
    res = bootstrap(ok.month(), scores, args.capacity, args.resamples, args.seed)
    comparisons = {
        "lgbm_vs_logreg": compare(res, "lgbm", "logreg"),
        "lgbm_vs_rule": compare(res, "lgbm", "rule"),
        "lgbm_cw_vs_logreg": compare(res, "lgbm_cw", "logreg"),
        "lgbm_cw_vs_lgbm": compare(res, "lgbm_cw", "lgbm"),
    }
    chosen = choose(comparisons)
    importance = dict(zip(FEATURE_COLUMNS,
                          lgbm.booster.feature_importance("gain").round(1).tolist(), strict=True))
    metrics = {
        "tuning": {k: v for k, v in tune_info.items() if k != "trials"},
        "lgbm": lgbm.info, "lgbm_cw": lgbm_cw.info,
        "july_oof_rows_excluded_unscorable": n_excluded,
        "july_oof": summary(res),
        "comparisons": comparisons,
        "chosen_tier1_model": chosen,
        "gain_importance": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
    }
    joblib.dump({"lgbm": lgbm, "lgbm_cw": lgbm_cw, "chosen": chosen}, run / "models.joblib")
    (run / "optuna_trials.json").write_text(json.dumps(tune_info["trials"], indent=2))
    save_metrics(run, metrics)  # last: marks the run complete
    print(json.dumps({"run": run.name, "comparisons": comparisons, "chosen": chosen},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
