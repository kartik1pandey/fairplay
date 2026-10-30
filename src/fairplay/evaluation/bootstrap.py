"""Account-level (cluster) bootstrap with paired model comparisons (design §0.7 step 7, §8).

Resampling ACCOUNTS, not rows: an account contributes up to 5 weekly rows that share one label and
very similar features. Treating those rows as independent would make intervals too narrow.
Each resample draws n accounts with replacement (multiplicity vector), rebuilds every weekly queue
with the same capacity, and recomputes the metrics. All models see the SAME resamples, so their
difference is "paired": shared noise cancels, which makes the comparison far more sensitive.

Decision rule (§8): A beats B only if the 95% interval of (A - B) flag-precision is above zero AND
the point gain is at least 2 percentage points. An interval that includes zero is reported as
"no evidence of improvement".
"""

import numpy as np

from fairplay.evaluation.queue import DEFAULT_CAPACITY, Month, evaluate, rank_order

METRICS = ("flag_precision", "lift", "flag_recall", "flag_pr_auc", "base_rate")
MIN_GAIN = 0.02


def bootstrap(month: Month, scores: dict[str, np.ndarray], capacity: float = DEFAULT_CAPACITY,
              n_resamples: int = 1000, seed: int = 0) -> dict:
    orders = {name: rank_order(month, s, seed=seed) for name, s in scores.items()}
    point = {name: evaluate(month, orders[name], s, capacity) for name, s in scores.items()}
    rng = np.random.default_rng(seed)
    draws = {name: {m: np.empty(n_resamples) for m in METRICS} for name in scores}
    for b in range(n_resamples):
        mult = np.bincount(rng.integers(0, month.n_accounts, month.n_accounts),
                           minlength=month.n_accounts)
        for name, s in scores.items():
            r = evaluate(month, orders[name], s, capacity, mult)
            for m in METRICS:
                draws[name][m][b] = r[m]
    return {"point": point, "draws": draws, "capacity": capacity, "n_resamples": n_resamples}


def interval(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.nanpercentile(x, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def compare(result: dict, a: str, b: str, metric: str = "flag_precision") -> dict:
    """Paired difference a - b with its interval and the pre-registered verdict."""
    diff = result["draws"][a][metric] - result["draws"][b][metric]
    point = result["point"][a][metric] - result["point"][b][metric]
    lo, hi = interval(diff)
    if lo > 0 and point >= MIN_GAIN:
        verdict = f"{a} beats {b}"
    elif lo > 0:
        verdict = f"{a} better than {b} but below the {MIN_GAIN:.0%} minimum gain"
    elif hi < 0:
        verdict = f"{b} beats {a}" if -point >= MIN_GAIN else f"{b} better, below minimum gain"
    else:
        verdict = "no evidence of improvement"
    return {"metric": metric, "a": a, "b": b, "diff": point, "ci95": (lo, hi), "verdict": verdict}


def summary(result: dict) -> dict:
    return {name: {m: {"value": result["point"][name][m],
                       "ci95": interval(result["draws"][name][m])} for m in METRICS}
            for name in result["point"]}
