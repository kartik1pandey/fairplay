"""The locked test: August scored ONCE with every frozen model (design §4 frozen test, §8).

Everything here was fixed before this run: models (July only), capacity, the 2200+ cap, the 2x
fairness criterion and the comparison list. claim_evaluation refuses a second run unless a reason
is given, and a re-run after a bug fix must report both results (design §4).

Pre-registered comparisons (design §8): LightGBM vs rule, LightGBM vs logistic regression, and
the same two on the stricter unseen-account evaluation (test accounts never seen in July).
lgbm_cw vs lgbm and the chosen model vs the rule are reported as exploratory.

uv run python -m fairplay.models.run_final
"""

import argparse
import json

import joblib
import numpy as np

from fairplay.evaluation.bootstrap import bootstrap, compare, summary
from fairplay.fairness.audit import AUDIT_COLUMNS, audit, fast_improver_thresholds
from fairplay.models.data import FEATURE_COLUMNS, Split, load_split
from fairplay.models.run_baselines import TABLE, scorable_subset, take
from fairplay.models.run_tier1 import latest_run
from fairplay.runs import claim_evaluation, new_run, save_metrics

PRE_REGISTERED = (("lgbm", "rule"), ("lgbm", "logreg"))


def reliability(p: np.ndarray, y: np.ndarray, w: np.ndarray, bins: int = 10) -> dict:
    """Account-weighted reliability curve over score-quantile bins, plus the Brier score."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    curve = [{"mean_predicted": float(np.average(p[m], weights=w[m])),
              "observed_flag_share": float(np.average(y[m], weights=w[m])),
              "accounts_weight": float(w[m].sum())}
             for b in range(len(edges) - 1) if (m := idx == b).any()]
    return {"brier": float(np.average((p - y) ** 2, weights=w)), "curve": curve}


def comparisons(split: Split, scores: dict, chosen: str, capacity: float, resamples: int,
                seed: int) -> dict:
    res = bootstrap(split.month(), scores, capacity, resamples, seed)
    pairs = [*PRE_REGISTERED, ("lgbm_cw", "lgbm")]
    if chosen not in ("lgbm", "rule"):
        pairs.append((chosen, "rule"))
    return {"summary": summary(res), "k_per_week": res["point"]["rule"]["k_per_week"],
            "comparisons": {f"{a}_vs_{b}": compare(res, a, b) | {
                "pre_registered": (a, b) in PRE_REGISTERED} for a, b in pairs}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--capacity", type=float, default=0.005)
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reason", default="", help="required for any re-run of the test")
    args = p.parse_args()

    base_dir, tier1_dir = latest_run("baselines"), latest_run("tier1")
    base, tier1 = joblib.load(base_dir / "models.joblib"), joblib.load(tier1_dir / "models.joblib")
    models = {"rule": base["rule"], "logreg": base["logreg"],
              "lgbm": tier1["lgbm"], "lgbm_cw": tier1["lgbm_cw"]}
    chosen = tier1["chosen"]
    train = load_split(TABLE, "train", AUDIT_COLUMNS)
    test = load_split(TABLE, "test", AUDIT_COLUMNS)
    run = new_run("final", vars(args) | {"features": FEATURE_COLUMNS, "chosen": chosen,
                                         "baseline_run": base_dir.name,
                                         "tier1_run": tier1_dir.name}, [TABLE])
    claim_evaluation("test", f"{run.name}:final", final=True, reason=args.reason)

    ok, scores, n_excluded = scorable_subset(test, {n: m.score(test) for n, m in models.items()})
    unseen = ~np.isin(ok.account_id, np.unique(train.account_id))
    thresholds = fast_improver_thresholds(train)
    model = models[chosen]
    prob = model.probability(ok) if hasattr(model, "probability") else None
    metrics = {
        "chosen_model": chosen,
        "rows": {"scored": len(ok.y), "excluded_unscorable": n_excluded,
                 "accounts": int(np.unique(ok.account_id).size),
                 "unseen_accounts": int(np.unique(ok.account_id[unseen]).size),
                 "positive_accounts": int(np.unique(ok.account_id[ok.y == 1]).size)},
        "all_accounts": comparisons(ok, scores, chosen, args.capacity, args.resamples, args.seed),
        "unseen_accounts": comparisons(take(ok, unseen), {n: s[unseen] for n, s in scores.items()},
                                       chosen, args.capacity, args.resamples, args.seed),
        "calibration": None if prob is None else reliability(prob, ok.y, ok.weight),
        "fairness": {n: audit(ok, scores[n], thresholds, args.capacity, args.resamples)
                     for n in dict.fromkeys(("rule", "logreg", chosen))},
    }
    save_metrics(run, metrics)
    print(json.dumps({"run": run.name, "chosen": chosen, "rows": metrics["rows"],
                      "all": metrics["all_accounts"]["comparisons"],
                      "unseen": metrics["unseen_accounts"]["comparisons"]},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
