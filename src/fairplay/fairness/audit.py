"""Fairness control and audit (design §0 item 9, §9).

Control (decision time, observable data only): at most ceil(m x k_w x s_w) accounts rated 2200+
per weekly queue, m = 1.0 pre-registered. Applied identically to every model; precision, queue
size and composition are reported before and after.

Audit (after labels): selection rate among UNFLAGGED accounts per group, vs the overall rate.
- selected = queued at any cutoff of the month.
- an account's group = its value at its first eligible cutoff of the month.
- criterion: group rate <= 2 x overall rate; failures trigger the §9 mitigation order.
- an unflagged account is not proven honest, so this is a proxy for "unnecessary investigation".
Groups with fewer than 50 unflagged accounts are shown with counts and intervals, never ranked.
"""

import numpy as np

from fairplay.evaluation.queue import DEFAULT_CAPACITY, Month, evaluate, rank_order
from fairplay.models.data import FEATURE_COLUMNS as F
from fairplay.models.data import Split

HIGH_RATED = 2200
CAP_MULTIPLIER = 1.0
MAX_RATIO = 2.0
MIN_GROUP = 50
AUDIT_COLUMNS = ("rating_at_cutoff", "gain_last50", "rating_band")


def rating_group(r: float) -> str:
    if r < 1200:
        return "<1200"
    if r < 1600:
        return "1200-1599"
    if r < 2000:
        return "1600-1999"
    if r < 2200:
        return "2000-2199"
    return "2200+"


def age_group(days: float) -> str:
    if np.isnan(days):
        return "unknown"
    return "<30d" if days < 30 else ("30-365d" if days <= 365 else ">1y")


def activity_group(games: float) -> str:
    return "10-19" if games < 20 else ("20-49" if games < 50 else "50+")


def fast_improver_thresholds(train: Split) -> dict[float, float]:
    """95th percentile of gain_last50 per rating band, from the training split only."""
    band, gain = train.extra["rating_band"], train.extra["gain_last50"]
    return {b: float(np.nanpercentile(gain[band == b], 95))
            for b in np.unique(band[~np.isnan(gain)])}


def improver_group(gain: float, band: float, thresholds: dict[float, float]) -> str:
    if np.isnan(gain):
        return "insufficient_history"
    t = thresholds.get(band)
    return "fast_improver" if t is not None and gain > t else "not_fast"


def account_groups(split: Split, month: Month, thresholds: dict[float, float]) -> dict:
    """Group label per account (first eligible cutoff of the month) for each dimension."""
    first = {}
    for row in np.lexsort((split.cutoff, month.acc)):
        first.setdefault(month.acc[row], row)
    rows = np.array([first[a] for a in range(month.n_accounts)])
    r, x = split.extra, split.X
    return {
        "rating": np.array([rating_group(v) for v in r["rating_at_cutoff"][rows]]),
        "account_age": np.array([age_group(v) for v in x[rows, F.index("account_age_days")]]),
        "activity": np.array([activity_group(v) for v in x[rows, F.index("games_in_window")]]),
        "improver": np.array([improver_group(g, b, thresholds) for g, b in
                              zip(r["gain_last50"][rows], r["rating_band"][rows], strict=True)]),
    }


def selection_rates(queued: np.ndarray, label: np.ndarray, groups: dict,
                    mult: np.ndarray) -> dict:
    unflagged = (label == 0) * mult
    overall = float((queued * unflagged).sum() / unflagged.sum())
    out = {"overall": overall}
    for dim, g in groups.items():
        for name in np.unique(g):
            m = unflagged * (g == name)
            out[f"{dim}:{name}"] = float((queued * m).sum() / m.sum()) if m.sum() else np.nan
    return out


def audit(split: Split, score: np.ndarray, thresholds: dict[float, float],
          capacity: float = DEFAULT_CAPACITY, n_resamples: int = 1000, seed: int = 0) -> dict:
    """Before/after the cap: precision, composition, and selection rates with 95% intervals."""
    capped = split.extra["rating_at_cutoff"] >= HIGH_RATED
    month = split.month(capped=capped)
    order = rank_order(month, score, seed=seed)
    groups = account_groups(split, month, thresholds)
    counts = {f"{d}:{n}": int(((g == n) & (month.label == 0)).sum())
              for d, g in groups.items() for n in np.unique(g)}
    result = {"unflagged_accounts_per_group": counts}
    def resamples():
        """The same resamples for both policies (paired), regenerated from the seed instead of
        held in memory: 1,000 x 73k accounts would be ~0.6 GB on August."""
        rng = np.random.default_rng(seed)
        for _ in range(n_resamples):
            yield np.bincount(rng.integers(0, month.n_accounts, month.n_accounts),
                              minlength=month.n_accounts)

    for policy, m in (("uncapped", None), ("capped", CAP_MULTIPLIER)):
        point = evaluate(month, order, score, capacity, cap_multiplier=m)
        rates = selection_rates(point["queued"], month.label, groups, np.ones(month.n_accounts))
        draws = {k: [] for k in rates}
        for mult in resamples():
            r = evaluate(month, order, score, capacity, mult, cap_multiplier=m)
            for k, v in selection_rates(r["queued"], month.label, groups, mult).items():
                draws[k].append(v)
        table = {}
        for k, v in rates.items():
            lo, hi = np.nanpercentile(draws[k], [2.5, 97.5])
            ratio = v / rates["overall"] if rates["overall"] else np.nan
            table[k] = {"rate": v, "ci95": (float(lo), float(hi)), "ratio_to_overall": ratio,
                        "n_unflagged": counts.get(k), "fails_2x": bool(ratio > MAX_RATIO)
                        if k != "overall" and counts.get(k, 0) >= MIN_GROUP else None}
        result[policy] = {
            "flag_precision": point["flag_precision"],
            "queue_size_per_week": point["queue_size_per_week"],
            "share_2200plus_per_week": point["capped_share_per_week"],
            "selection_rate_among_unflagged": table,
            "failing_groups": [k for k, t in table.items() if t["fails_2x"]],
        }
    result["precision_cost_of_cap"] = (result["uncapped"]["flag_precision"]
                                       - result["capped"]["flag_precision"])
    return result
