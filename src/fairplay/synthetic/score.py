"""Scoring the synthetic benchmark (design §6, §0.12). Evaluation only: nothing is trained here.

1. Features: the synthetic player_games rows go through the SAME shared SQL as real accounts
   (game_features_sql -> cutoff_features_from -> apply_peer_z with the frozen July peers).
2. Threshold: frozen from the REAL August queue of the frozen model, never from synthetic data.
   Per August week, the score of the last account-week admitted to the capped queue; the
   threshold is their median (one number, robust to a single odd week). The cut uses scores and
   ratings only, never labels, exactly like a live weekly queue.
3. Synthetic recall = share of synthetic engine-using accounts scoring >= threshold, per strategy
   and rating band, with Wilson 95% intervals. Shown beside the share of synthetic CONTROLS above
   the threshold (the simulation's false-alarm rate): the comparison that cancels generator
   artefacts. Real-world recall is never claimed.
4. Realism: synthetic controls vs real unflagged training accounts per feature and band (KS);
   features over the limit are listed and no synthetic claim may rest on them.
"""

import datetime as dt
import math
from pathlib import Path

import duckdb
import numpy as np

from fairplay.evaluation.queue import DEFAULT_CAPACITY, evaluate, rank_order
from fairplay.fairness.audit import CAP_MULTIPLIER, HIGH_RATED
from fairplay.features.account_features import FEATURES, apply_peer_z, cutoff_features_from
from fairplay.features.game_features import game_features_sql
from fairplay.models.data import FEATURE_COLUMNS, Split
from fairplay.synthetic.realism import realism_table

HELD_OUT = "C"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return mid - half, mid + half


def synthetic_features(files: list[Path], cutoff: dt.datetime, peers: Path) -> list[dict]:
    con = duckdb.connect()
    src = "read_parquet([" + ", ".join(f"'{f.as_posix()}'" for f in files) + "])"
    con.execute(f"CREATE TEMP TABLE gf AS {game_features_sql(src)}")
    con.execute("CREATE TEMP TABLE acc AS SELECT DISTINCT account_id, ?::TIMESTAMP AS cutoff "
                "FROM gf", [cutoff])
    rel = cutoff_features_from(con, "gf", "acc", cutoff)
    con.execute(f"CREATE TEMP TABLE f AS {rel.sql_query()}")
    con.execute(f"CREATE TEMP TABLE peers AS SELECT * FROM read_parquet('{peers.as_posix()}')")
    rel = con.sql(f"SELECT * FROM {apply_peer_z(con, 'f', 'peers')} ORDER BY account_id")
    return [dict(zip(rel.columns, r, strict=True)) for r in rel.fetchall()]


def to_split(feats: list[dict], ages: dict[str, int], cutoff: dt.datetime) -> Split:
    def value(f: dict, name: str) -> float:
        v = ages.get(f["account_id"]) if name == "account_age_days" else f[name]
        return np.nan if v is None else float(v)

    n = len(feats)
    return Split(
        name="synthetic",
        X=np.array([[value(f, c) for c in FEATURE_COLUMNS] for f in feats], dtype=float)
        .reshape(n, len(FEATURE_COLUMNS)),
        y=np.zeros(n, np.int8), weight=np.ones(n),
        account_id=np.array([f["account_id"] for f in feats], dtype=object),
        cutoff=np.full(n, cutoff), games=np.array([f["games_in_window"] for f in feats]),
        extra={"rating_at_cutoff": np.array([float(f["rating_at_cutoff"]) for f in feats])})


def frozen_threshold(model, real: Split, capacity: float = DEFAULT_CAPACITY) -> dict:
    """Median over the month's weeks of the lowest score admitted to the capped real queue."""
    score = np.asarray(model.score(real), dtype=float)
    ok = np.isfinite(score)
    capped = real.extra["rating_at_cutoff"] >= HIGH_RATED
    month = real.month(capped)
    r = evaluate(month, rank_order(month, np.where(ok, score, -np.inf)),
                 np.where(ok, score, -np.inf), capacity, cap_multiplier=CAP_MULTIPLIER)
    weekly = [float(score[rows].min()) for rows in r["queued_rows"] if len(rows)]
    return {"threshold": float(np.median(weekly)), "weekly": weekly, "capacity": capacity}


def strategy_of(account_id: str) -> str:
    return account_id.split("-")[1]  # syn-<strategy>-<n>


def recall_table(account_ids: np.ndarray, score: np.ndarray, rating: np.ndarray,
                 threshold: float) -> dict:
    strategy = np.array([strategy_of(a) for a in account_ids])
    band = (rating // 200) * 200
    above = np.isfinite(score) & (score >= threshold)
    out = {}
    for s in sorted(set(strategy)):
        m = strategy == s
        k, n = int(above[m].sum()), int(m.sum())
        lo, hi = wilson(k, n)
        by_band = {}
        for b in sorted(set(band[m])):
            mb = m & (band == b)
            kb, nb = int(above[mb].sum()), int(mb.sum())
            by_band[f"{int(b)}-{int(b) + 199}"] = {"share_above": kb / nb, "n": nb,
                                                   "wilson_95": wilson(kb, nb)}
        key = "control_false_alarm" if s == "control" else "synthetic_recall"
        out[s] = {key: k / n, "n": n, "wilson_95": (lo, hi), "by_band": by_band,
                  "unscorable": int((~np.isfinite(score[m])).sum())}
    return out


def auc_vs_control(account_ids: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Per strategy: P(a random engine-using account outranks a random synthetic control).
    Ranks against controls from the same generator, so artefacts shared by both cancel; the
    threshold recall above does not have that protection."""
    from sklearn.metrics import roc_auc_score

    strategy = np.array([strategy_of(a) for a in account_ids])
    ok = np.isfinite(score)
    out = {}
    for s in sorted(set(strategy) - {"control"}):
        m = ok & ((strategy == s) | (strategy == "control"))
        out[s] = float(roc_auc_score(strategy[m] == s, score[m]))
    return out


def realism(synth_feats: list[dict], real: Split, real_band: np.ndarray,
            real_rating: np.ndarray | None = None) -> dict:
    """Synthetic CONTROLS vs real unflagged accounts (design §0.12): per band where both sides
    have >= 150 accounts, and pooled over Maia's rating range (synthetic ratings are drawn from
    the real distribution in that range, so the pooled comparison is like for like)."""
    controls = [f for f in synth_feats if strategy_of(f["account_id"]) == "control"]
    names = [f for f in FEATURES if f != "account_age_days"]
    synth = {n: np.array([np.nan if f[n] is None else float(f[n]) for f in controls])
             for n in names}
    unflagged = real.y == 0
    real_cols = {n: real.column(n)[unflagged] for n in names}
    synth_band = np.array([(f["rating_at_cutoff"] // 200) * 200 for f in controls], dtype=float)
    by_band = realism_table(synth, real_cols, synth_band, real_band[unflagged])
    rating = real_band if real_rating is None else real_rating
    in_range = (rating[unflagged] >= 1050) & (rating[unflagged] < 1950)
    pooled = realism_table(synth, {n: v[in_range] for n, v in real_cols.items()},
                           np.zeros(len(controls)), np.zeros(int(in_range.sum())))
    return {"by_band": by_band, "pooled": pooled,
            "features_ks_above_limit": sorted(set(by_band["features_ks_above_limit"])
                                              | set(pooled["features_ks_above_limit"]))}
