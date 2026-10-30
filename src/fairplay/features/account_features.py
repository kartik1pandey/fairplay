"""Tier-1 features per (account, cutoff) (design §5), from per-game summaries.

Window = games that started in [cutoff - 30 days, cutoff - 30 minutes) (same rule as eligibility).
"Last 30" = the 30 most recent games in that window. Coverage rules give NULL, never a guess.
Feature 14 (account age) needs createdAt from the S0 snapshot and is joined in the model step.
Feature 5 is a robust z-score against peer statistics frozen from the training split only.

uv run python -m fairplay.features.account_features
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import duckdb

from fairplay import duck
from fairplay.dataset.eligibility import END_MARGIN, WINDOW
from fairplay.duck import copy_atomic
from fairplay.features.game_features import FEATURE_VERSION

PROCESSED = Path("data/processed")
LAST_N = 30
MIN_TIMING_MOVES = 100
PEER_MIN_ACCOUNTS = 500
MAIN_BLITZ_POOLS = ("180+0", "180+2", "300+0", "300+3")

# name -> suspicious direction (design §5 table); "context" features are never cited as evidence
FEATURES = {
    "think_time_cv": "low",
    "even_timing_share": "high",
    "time_trend_strength": "low",
    "instant_move_share": "both",
    "clock_at_30_z": "both",
    "time_loss_share": "high",
    "score_minus_expected": "high",
    "performance_rating_gap": "high",
    "upset_win_share": "high",
    "window_rating_change": "high",
    "largest_daily_gain": "high",
    "early_resignation_share": "high",
    "games_per_active_day": "context",
    "account_age_days": "context",
    "games_in_window": "context",
    "best_single_surplus": "high",
}

CUTOFF_SQL = """
WITH w AS (
    SELECT g.*,
        row_number() OVER (PARTITION BY g.account_id ORDER BY g.start_ts DESC, g.game_key)
            AS recency
    FROM {source} g
    JOIN (SELECT account_id FROM {accounts} WHERE cutoff = TIMESTAMP '{cutoff}') e
      USING (account_id)
    WHERE g.start_ts >= TIMESTAMP '{cutoff}' - INTERVAL {window_days} DAY
      AND g.start_ts <  TIMESTAMP '{cutoff}' - INTERVAL {margin_min} MINUTE
),
w60 AS (  -- fairness audit only: rating gain over the last 50 games in 60 days (design §0.3)
    SELECT g.account_id, g.own_elo, g.post_rating,
        row_number() OVER (PARTITION BY g.account_id ORDER BY g.start_ts DESC, g.game_key) AS r
    FROM {source} g
    JOIN (SELECT account_id FROM {accounts} WHERE cutoff = TIMESTAMP '{cutoff}') e
      USING (account_id)
    WHERE g.start_ts >= TIMESTAMP '{cutoff}' - INTERVAL 60 DAY
      AND g.start_ts <  TIMESTAMP '{cutoff}' - INTERVAL {margin_min} MINUTE
),
improve AS (
    SELECT account_id,
        CASE WHEN count(*) >= 50 THEN max(post_rating) FILTER (r = 1) - max(own_elo) FILTER (r = 50)
        END AS gain_last50
    FROM w60 WHERE r <= 50 GROUP BY account_id
),
daily AS (
    SELECT account_id, max(gain) AS largest_daily_gain FROM (
        SELECT account_id, day,
               arg_max(post_rating, start_ts) - arg_min(own_elo, start_ts) AS gain
        FROM w GROUP BY account_id, day)
    GROUP BY account_id
),
agg AS (
    SELECT account_id,
        count(*) AS games_in_window,
        avg(lost_on_time::INT) AS time_loss_share,
        avg(early_resignation::INT) AS early_resignation_share,
        count(*) / count(DISTINCT day) AS games_per_active_day,
        arg_max(post_rating, start_ts) - arg_min(own_elo, start_ts) AS window_rating_change,
        arg_max(post_rating, start_ts) AS rating_at_cutoff,
        mode(tc) AS modal_tc,
        -- last-30 statistics
        count(*) FILTER (recency <= {n}) AS n_last,
        avg(score - expected) FILTER (recency <= {n}) AS score_minus_expected,
        max(score - expected) FILTER (recency <= {n}) AS best_single_surplus,
        avg(opp_elo) FILTER (recency <= {n})
          + 400.0 * (count(*) FILTER (recency <= {n} AND score = 1)
                     - count(*) FILTER (recency <= {n} AND score = 0))
            / count(*) FILTER (recency <= {n})
          - avg(own_elo) FILTER (recency <= {n}) AS perf_gap_raw,
        count(*) FILTER (recency <= {n} AND opp_elo >= own_elo + 200) AS n_upset_games,
        count(*) FILTER (recency <= {n} AND opp_elo >= own_elo + 200 AND score = 1)
            AS n_upset_wins,
        flatten(list(think_norm_after10) FILTER (recency <= {n} AND clock_valid)) AS thinks,
        sum(n_instant_after10) FILTER (recency <= {n} AND clock_valid) AS n_instant,
        sum(n_clipped_after10) FILTER (recency <= {n} AND clock_valid) AS n_clipped,
        avg(abs_spearman_15_40) FILTER (recency <= {n}) AS spearman_mean,
        count(abs_spearman_15_40) FILTER (recency <= {n}) AS n_spearman_games,
        median(clock_at_30_norm) FILTER (recency <= {n}) AS clock30_median,
        count(clock_at_30_norm) FILTER (recency <= {n}) AS n_clock30_games
    FROM w GROUP BY account_id
),
timing AS (
    SELECT *, list_median(thinks) AS think_median, len(thinks) AS n_think_moves FROM agg
)
SELECT
    t.account_id, TIMESTAMP '{cutoff}' AS cutoff,
    CASE WHEN n_think_moves >= {min_moves}
         THEN list_stddev_samp(thinks) / nullif(list_avg(thinks), 0) END AS think_time_cv,
    CASE WHEN n_think_moves >= {min_moves}
         THEN len(list_filter(thinks, lambda x: abs(x - think_median) <= 0.2 * think_median))
              / n_think_moves END AS even_timing_share,
    CASE WHEN n_spearman_games >= 3 THEN spearman_mean END AS time_trend_strength,
    CASE WHEN n_think_moves >= {min_moves} THEN n_instant / n_think_moves END
        AS instant_move_share,
    CASE WHEN n_clock30_games >= 3 THEN clock30_median END AS clock_at_30_raw,
    time_loss_share, score_minus_expected,
    greatest(-800, least(800, perf_gap_raw)) AS performance_rating_gap,
    CASE WHEN n_upset_games >= 3 THEN n_upset_wins / n_upset_games END AS upset_win_share,
    window_rating_change, d.largest_daily_gain, early_resignation_share,
    games_per_active_day, games_in_window, best_single_surplus,
    -- diagnostics and peer-group keys (not model inputs)
    rating_at_cutoff, (rating_at_cutoff // 200) * 200 AS rating_band,
    CASE WHEN modal_tc IN {pools} THEN modal_tc ELSE 'other' END AS tc_group,
    n_last, n_think_moves, n_clipped, i.gain_last50
FROM timing t JOIN daily d USING (account_id) LEFT JOIN improve i USING (account_id)
"""


def cutoff_features(
    con: duckdb.DuckDBPyConnection, files: list[Path], accounts_table: str, cutoff: dt.datetime
) -> duckdb.DuckDBPyRelation:
    source = "read_parquet([" + ", ".join(f"'{f.as_posix()}'" for f in files) + "])"
    return cutoff_features_from(con, source, accounts_table, cutoff)


def cutoff_features_from(
    con: duckdb.DuckDBPyConnection, source: str, accounts_table: str, cutoff: dt.datetime
) -> duckdb.DuckDBPyRelation:
    """source = any relation with the game_features schema (file scan offline, table online)."""
    return con.sql(CUTOFF_SQL.format(
        source=source,
        accounts=accounts_table,
        cutoff=cutoff.isoformat(sep=" "),
        window_days=WINDOW.days,
        margin_min=int(END_MARGIN.total_seconds() // 60),
        n=LAST_N,
        min_moves=MIN_TIMING_MOVES,
        pools=str(MAIN_BLITZ_POOLS),
    ))


def add_peer_z(con: duckdb.DuckDBPyConnection, table: str, peer_split: str = "train") -> str:
    """Robust z of clock_at_30_raw within rating band x time-control group.

    Peer median and MAD come from `peer_split` rows only (frozen, no future data). A cell with
    fewer than PEER_MIN_ACCOUNTS accounts falls back to the band, then to all accounts.
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE peer_clock30 AS
        WITH src AS (SELECT * FROM {table} WHERE split = '{peer_split}'
                       AND clock_at_30_raw IS NOT NULL),
        cell AS (SELECT rating_band, tc_group, median(clock_at_30_raw) AS med,
                        mad(clock_at_30_raw) AS mad, count(DISTINCT account_id) AS n
                 FROM src GROUP BY ALL),
        band AS (SELECT rating_band, median(clock_at_30_raw) AS med,
                        mad(clock_at_30_raw) AS mad, count(DISTINCT account_id) AS n
                 FROM src GROUP BY ALL),
        all_peers AS (SELECT median(clock_at_30_raw) AS med, mad(clock_at_30_raw) AS mad FROM src)
        SELECT b.rating_band, t.tc_group,
               CASE WHEN c.n >= {PEER_MIN_ACCOUNTS} THEN c.med
                    WHEN b.n >= {PEER_MIN_ACCOUNTS} THEN b.med ELSE g.med END AS med,
               CASE WHEN c.n >= {PEER_MIN_ACCOUNTS} THEN c.mad
                    WHEN b.n >= {PEER_MIN_ACCOUNTS} THEN b.mad ELSE g.mad END AS mad,
               CASE WHEN c.n >= {PEER_MIN_ACCOUNTS} THEN 'cell'
                    WHEN b.n >= {PEER_MIN_ACCOUNTS} THEN 'band' ELSE 'global' END AS level
        FROM band b CROSS JOIN (SELECT DISTINCT tc_group FROM src) t CROSS JOIN all_peers g
        LEFT JOIN cell c ON c.rating_band = b.rating_band AND c.tc_group = t.tc_group
        UNION ALL  -- fallback row (NULL keys) for bands never seen in training
        SELECT NULL, NULL, med, mad, 'global' FROM all_peers
    """)
    return apply_peer_z(con, table)


def apply_peer_z(con: duckdb.DuckDBPyConnection, table: str, peers: str = "peer_clock30") -> str:
    """clock_at_30_z from FROZEN peer stats (offline: just built; online: the saved file)."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table}_z AS
        SELECT f.*, (f.clock_at_30_raw - coalesce(p.med, g.med))
                    / nullif(1.4826 * coalesce(p.mad, g.mad), 0) AS clock_at_30_z
        FROM {table} f
        LEFT JOIN {peers} p ON p.rating_band = f.rating_band AND p.tc_group = f.tc_group
        CROSS JOIN (SELECT med, mad FROM {peers} WHERE rating_band IS NULL) g
    """)
    return f"{table}_z"


def build(processed: Path = PROCESSED) -> dict:
    con = duck.connect()
    files = sorted((processed / "game_features").glob("month=*.parquet"))
    con.execute(f"CREATE TEMP TABLE eligible AS "
                f"SELECT * FROM '{(processed / 'eligible_rows.parquet').as_posix()}'")
    cutoffs = [r[0] for r in con.execute(
        "SELECT DISTINCT cutoff FROM eligible ORDER BY 1").fetchall()]
    con.execute("CREATE TEMP TABLE feats AS SELECT NULL::VARCHAR AS split, * FROM "
                f"({cutoff_features(con, files, 'eligible', cutoffs[0]).sql_query()}) LIMIT 0")
    for c in cutoffs:
        rel = cutoff_features(con, files, "eligible", c)
        con.execute(f"INSERT INTO feats SELECT e.split, f.* FROM ({rel.sql_query()}) f "
                    f"JOIN eligible e USING (account_id, cutoff)")
        print("features for cutoff", c.date(), flush=True)
    final = add_peer_z(con, "feats")
    con.execute("COPY peer_clock30 TO "
                f"'{(processed / 'peer_stats_clock30.parquet').as_posix()}' (FORMAT parquet)")
    out = processed / "features_tier1.parquet"
    copy_atomic(con, f"SELECT *, '{FEATURE_VERSION}' AS feature_version FROM {final} "
                     f"ORDER BY split, cutoff, account_id", out)
    names = [f for f in FEATURES if f != "account_age_days"]
    values = con.execute(f"""
        SELECT {", ".join(f"round(avg(({f} IS NOT NULL)::INT), 4)" for f in names)} FROM {final}
    """).fetchone()
    coverage = dict(zip(names, values, strict=True))
    summary = {"rows": con.execute(f"SELECT count(*) FROM {final}").fetchone()[0],
               "non_null_share": coverage}
    (processed / "features_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    argparse.ArgumentParser().parse_args()
    print(json.dumps(build(), indent=2))


if __name__ == "__main__":
    main()
