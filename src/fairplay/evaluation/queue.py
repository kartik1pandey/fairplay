"""Weekly review-queue simulation: the single place where every model is judged (design §0.7).

For each weekly cutoff w of a month, in order:
  1. eligible_w = the account-week rows at w; remove accounts already queued earlier this month.
  2. Rank the rest by score; ties: more games first, then a seeded random order (design §7).
  3. queue_w = top k_w, k_w = ceil(c x eligible_w), c = capacity share (0.5% by default).
  4. precision_w = flagged share of queue_w; base_w = flagged share of the remaining rows.
Primary metric = mean of precision_w over weeks (equal weights). Lift = that / mean of base_w.
Flag-recall = flagged accounts queued in the month / flagged accounts eligible at any cutoff.
PR-AUC = one row per account: its max score over the month's cutoffs, against its label.

Fairness cap (design §0 item 9), optional: rows flagged in `month.capped` (2200+ at that cutoff)
may fill at most ceil(m x k_w x s_w) places, s_w = capped share of eligible rows at w. Walking the
ranked list and skipping capped rows once the cap is full is the same as: keep all uncapped rows
plus the first `cap` capped rows, then take the first k_w of those.

Bootstrap support: `mult[a]` = how many times account a was drawn. An account drawn twice counts
as two reviewable copies with identical rows; with mult = 1 everywhere this is the plain metric.
"""

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score

DEFAULT_CAPACITY = 0.005


@dataclass
class Month:
    """One evaluation month. Rows are account-weeks; accounts are re-indexed 0..n_accounts-1."""

    acc: np.ndarray  # account index per row
    week: np.ndarray  # week index per row (0 = first cutoff)
    games: np.ndarray  # games_in_window per row (tie-break)
    label: np.ndarray  # label per ACCOUNT (0/1), length n_accounts
    n_accounts: int
    n_weeks: int
    capped: np.ndarray | None = None  # per row: subject to the fairness cap (2200+)

    @classmethod
    def from_rows(cls, account_ids, cutoffs, games, labels, capped=None) -> "Month":
        """labels are per row; they must be constant within an account (one S0 label)."""
        acc_keys, acc = np.unique(np.asarray(account_ids), return_inverse=True)
        week_keys, week = np.unique(np.asarray(cutoffs), return_inverse=True)
        label = np.zeros(len(acc_keys), dtype=np.int8)
        label[acc] = np.asarray(labels, dtype=np.int8)
        return cls(acc=acc, week=week, games=np.asarray(games), label=label,
                   n_accounts=len(acc_keys), n_weeks=len(week_keys),
                   capped=None if capped is None else np.asarray(capped, dtype=bool))


def rank_order(month: Month, score: np.ndarray, seed: int = 0) -> list[np.ndarray]:
    """Per week, row indices sorted best-first: score desc, games desc, then seeded random."""
    rng = np.random.default_rng(seed)
    tiebreak = rng.random(len(score))
    out = []
    for w in range(month.n_weeks):
        rows = np.flatnonzero(month.week == w)
        order = np.lexsort((tiebreak[rows], -month.games[rows], -score[rows]))
        out.append(rows[order])
    return out


def week_limits(n_eligible: int, n_capped: int, capacity: float = DEFAULT_CAPACITY,
                cap_multiplier: float | None = None) -> tuple[int, int | None]:
    """k_w = ceil(c x eligible_w); 2200+ cap = ceil(m x k_w x s_w) (design §0.7, §0.9)."""
    k = math.ceil(capacity * n_eligible)
    cap = None if cap_multiplier is None else math.ceil(cap_multiplier * k * n_capped / n_eligible)
    return k, cap


def select_week(order: np.ndarray, capped: np.ndarray, k: int, cap: int | None) -> np.ndarray:
    """Production queue for one week: walk the ranked rows, skip a capped (2200+) row once the
    cap is reached, stop at k. Same rule as `evaluate` (tested equal), without bootstrap copies."""
    chosen, n_capped = [], 0
    for i in order:
        if len(chosen) == k:
            break
        if cap is not None and capped[i]:
            if n_capped >= cap:
                continue
            n_capped += 1
        chosen.append(i)
    return np.array(chosen, dtype=np.int64)


def evaluate(month: Month, order: list[np.ndarray], score: np.ndarray,
             capacity: float = DEFAULT_CAPACITY, mult: np.ndarray | None = None,
             cap_multiplier: float | None = None) -> dict:
    mult = np.ones(month.n_accounts, dtype=np.int64) if mult is None else mult
    use_cap = cap_multiplier is not None and month.capped is not None
    queued = np.zeros(month.n_accounts, dtype=bool)
    eligible_any = np.zeros(month.n_accounts, dtype=bool)
    precisions, bases, ks, queue_sizes, capped_shares = [], [], [], [], []
    queued_rows = []  # per week: row indices that entered the queue, best-first
    for rows in order:
        a = month.acc[rows]
        w = mult[a]
        c = month.capped[rows] if month.capped is not None else np.zeros(len(rows), bool)
        eligible_any[a] = True
        n_eligible = int(w.sum())
        if n_eligible == 0:
            continue
        k = math.ceil(capacity * n_eligible)
        cap = math.ceil(cap_multiplier * k * (w * c).sum() / n_eligible) if use_cap else None
        keep = ~queued[a] & (w > 0)
        a, w, c, kept_rows = a[keep], w[keep], c[keep], rows[keep]
        if w.sum() == 0:
            continue
        lab = month.label[a]
        avail = w
        if use_cap:
            capped_w = w * c
            capped_before = np.cumsum(capped_w) - capped_w
            avail = np.where(c, np.clip(cap - capped_before, 0, w), w)
        before = np.cumsum(avail) - avail
        take = np.clip(k - before, 0, avail)  # copies of each row that fit in the queue
        chosen = take > 0
        precisions.append(float((take * lab).sum() / take.sum()))
        bases.append(float((w * lab).sum() / w.sum()))
        ks.append(k)
        queue_sizes.append(int(take.sum()))
        capped_shares.append(float((take * c).sum() / take.sum()))
        queued[a[chosen]] = True
        queued_rows.append(kept_rows[chosen])

    flagged_eligible = float((mult * month.label * eligible_any).sum())
    flagged_queued = float((mult * month.label * queued).sum())
    precision = float(np.mean(precisions))
    base = float(np.mean(bases))
    acc_max = np.full(month.n_accounts, -np.inf)
    np.maximum.at(acc_max, month.acc, score)
    present = mult > 0
    finite = np.isfinite(acc_max)
    if not finite.all():  # -inf = "back of the queue"; keep the order, satisfy sklearn
        floor = acc_max[finite].min() - 1 if finite.any() else 0.0
        acc_max = np.where(finite, acc_max, floor)
    return {
        "flag_precision": precision,
        "base_rate": base,
        "lift": precision / base if base > 0 else math.nan,
        "flag_recall": flagged_queued / flagged_eligible if flagged_eligible else math.nan,
        "flag_pr_auc": float(average_precision_score(
            month.label[present], acc_max[present], sample_weight=mult[present]))
        if month.label[present].any() else math.nan,  # undefined without positives
        "weekly_precision": precisions,
        "k_per_week": ks,
        "queue_size_per_week": queue_sizes,
        "unique_accounts_queued": int((mult * queued).sum()),
        "capped_share_per_week": capped_shares,
        "queued": queued,
        "queued_rows": queued_rows,
    }
