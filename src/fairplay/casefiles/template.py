"""Deterministic v1 case file (design §10): fixed queries, no LLM, so it cannot hallucinate.

- Evidence: up to 3 non-context features at or beyond the 5th percentile in their suspicious
  direction (low: pct <= 5; high: pct >= 95; both: either tail), most extreme first.
- Counter-evidence: non-context features in the non-suspicious half (low: pct > 50; high: pct < 50;
  both: the middle half, 25-75).
- Status insufficient_evidence: fewer than 2 features meet the evidence rule, or the peer group
  still has < 200 accounts after merging bands.
- Extreme games: per evidence feature, the account's most extreme game in the window (fixed SQL).
- The text never says an account cheated; a blocklist check enforces it on every string.
"""

import datetime as dt
import math
import uuid
from pathlib import Path

import duckdb

from fairplay.casefiles.peers import PeerReference
from fairplay.features.account_features import FEATURES

EVIDENCE_PERCENTILE = 5.0
MAX_EVIDENCE = 3
DECISIONS = ("clear", "watch", "escalate")
DISCLAIMER = "Research prototype. A score is a prioritisation signal, not evidence of wrongdoing."
BLOCKLIST = ("cheat", "proves", "proof", "used an engine", "guilty", "engine user")
CASE_NAMESPACE = uuid.UUID("5d0c7a3e-0f1b-4c55-9a77-3f6c1b2a9e10")

UNITS = {
    "think_time_cv": "ratio", "even_timing_share": "share", "time_trend_strength": "|rho|",
    "instant_move_share": "share", "clock_at_30_z": "robust z", "time_loss_share": "share",
    "score_minus_expected": "points per game", "performance_rating_gap": "rating points",
    "upset_win_share": "share", "window_rating_change": "rating points",
    "largest_daily_gain": "rating points", "early_resignation_share": "share",
    "best_single_surplus": "points",
}
QUERY = {f: f"q_{fam}_v1" for fam, fs in {
    "timing": ("think_time_cv", "even_timing_share", "time_trend_strength", "instant_move_share"),
    "clock": ("clock_at_30_z", "time_loss_share"),
    "results": ("score_minus_expected", "performance_rating_gap", "upset_win_share",
                "best_single_surplus"),
    "trajectory": ("window_rating_change", "largest_daily_gain"),
    "behaviour": ("early_resignation_share",)}.items() for f in fs}

# Per-game expression behind each feature, for "the most extreme game" (NULL = not applicable).
GAME_EXPR = {
    "think_time_cv": "list_aggregate(think_norm_after10, 'stddev_samp')"
                     " / nullif(list_avg(think_norm_after10), 0)",
    "even_timing_share": "list_aggregate(think_norm_after10, 'stddev_samp')"
                         " / nullif(list_avg(think_norm_after10), 0)",
    "time_trend_strength": "abs_spearman_15_40",
    "instant_move_share": "n_instant_after10 / nullif(len(think_norm_after10), 0)",
    "clock_at_30_z": "clock_at_30_norm",
    "time_loss_share": "CASE WHEN lost_on_time THEN 1 END",
    "score_minus_expected": "score - expected",
    "performance_rating_gap": "score - expected",
    "best_single_surplus": "score - expected",
    "upset_win_share": "CASE WHEN score = 1 AND opp_elo >= own_elo + 200"
                       " THEN opp_elo - own_elo END",
    "window_rating_change": "post_rating - own_elo",
    "largest_daily_gain": "post_rating - own_elo",
    "early_resignation_share": "CASE WHEN early_resignation THEN 1 END",
}
LOW_GAME = {"think_time_cv", "even_timing_share", "time_trend_strength"}  # lower per game = extreme


def suspicion_distance(direction: str, pct: float) -> float:
    """Percentile points from the suspicious tail (0 = most extreme)."""
    return {"low": pct, "high": 100 - pct, "both": min(pct, 100 - pct)}[direction]


def is_counter(direction: str, pct: float) -> bool:
    return {"low": pct > 50, "high": pct < 50, "both": 25 <= pct <= 75}[direction]


def extreme_game(con: duckdb.DuckDBPyConnection, game_files: list[Path], account_id: str,
                 cutoff: dt.datetime, feature: str, pct: float) -> str | None:
    """Most extreme game for one feature among the last 30 games of the window before cutoff."""
    ascending = feature in LOW_GAME or (FEATURES[feature] == "both" and pct < 50)
    files = ", ".join(f"'{f.as_posix()}'" for f in game_files)
    row = con.execute(f"""
        WITH w AS (
            SELECT * FROM read_parquet([{files}])
            WHERE account_id = $acc AND start_ts >= $cut - INTERVAL 30 DAY
              AND start_ts < $cut - INTERVAL 30 MINUTE
            ORDER BY start_ts DESC, game_key LIMIT 30)
        SELECT game_key FROM w WHERE ({GAME_EXPR[feature]}) IS NOT NULL
        ORDER BY {GAME_EXPR[feature]} {'ASC' if ascending else 'DESC'}, start_ts DESC LIMIT 1
    """, {"acc": account_id, "cut": cutoff}).fetchone()
    return row[0] if row else None


def _r(x: float, nd: int = 4) -> float:
    return float(round(x, nd))


def build_case(row: dict, score: float, ref: PeerReference, versions: dict,
               games_for=None) -> dict:
    """row: one account-cutoff row (features + rating_band, tc_group, games_in_window).
    games_for(feature, pct) -> game key or None (extreme_game bound to this account/cutoff)."""
    cutoff = str(row["cutoff"])[:10]
    group = ref.group(int(row["rating_band"]), str(row["tc_group"]))
    evidence, counter = [], []
    if group.sufficient:
        for feature, direction in FEATURES.items():
            x = row.get(feature)
            if direction == "context" or x is None or math.isnan(float(x)):
                continue
            pct = ref.percentile(group, feature, float(x))
            if math.isnan(pct):
                continue
            if suspicion_distance(direction, pct) <= EVIDENCE_PERCENTILE:
                s = ref.summary(group, feature)
                evidence.append({
                    "feature": feature, "value": _r(float(x)), "unit": UNITS[feature],
                    "direction": direction, "peer_p05": _r(s["peer_p05"]),
                    "peer_median": _r(s["peer_median"]), "peer_p95": _r(s["peer_p95"]),
                    "account_percentile": _r(pct, 1), "peer_group": group.label,
                    "peer_n": s["peer_n"], "reference": ref.reference,
                    "games": int(row["games_in_window"]), "query": QUERY[feature],
                    "_distance": suspicion_distance(direction, pct)})
            elif is_counter(direction, pct):
                counter.append({"feature": feature, "value": _r(float(x)),
                                "account_percentile": _r(pct, 1)})
    evidence.sort(key=lambda e: (e["_distance"], e["feature"]))
    n_qualifying = len(evidence)
    evidence = evidence[:MAX_EVIDENCE]
    extreme = []
    for e in evidence:
        del e["_distance"]
        g = games_for(e["feature"], e["account_percentile"]) if games_for else None
        if g is not None and g not in extreme:
            extreme.append(g)
    status = "review" if group.sufficient and n_qualifying >= 2 else "insufficient_evidence"
    case = {
        "case_id": str(uuid.uuid5(CASE_NAMESPACE, f"{row['account_id']}|{cutoff}|"
                                                  f"{sorted(versions.items())}")),
        "account": str(row["account_id"]), "cutoff": cutoff, "versions": dict(versions),
        "score": _r(float(score)), "games_used": int(row["games_in_window"]),
        "peer_group": group.label, "peer_group_accounts": group.n_accounts,
        "evidence": evidence, "counter_evidence": counter, "extreme_games": extreme,
        "status": status, "reviewer": {"decision": None, "reason": None},
        "disclaimer": DISCLAIMER,
    }
    check_language(case)
    return case


def record_decision(case: dict, decision: str, reason: str) -> dict:
    """Reviewer outcome. Stored apart from the score; 'clear' feeds the 4-week suppression."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    if not reason.strip():
        raise ValueError("a reason is required")
    out = {**case, "reviewer": {"decision": decision, "reason": reason.strip()}}
    check_language(out)
    return out


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def check_language(case: dict) -> None:
    """Fail closed if any string in the case file uses accusation wording."""
    for s in _strings({k: v for k, v in case.items() if k != "disclaimer"}):
        low = s.lower()
        hit = next((b for b in BLOCKLIST if b in low), None)
        if hit:
            raise ValueError(f"blocked wording {hit!r} in case file")
