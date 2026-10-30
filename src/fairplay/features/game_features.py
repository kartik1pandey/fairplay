"""Per-game summaries (design §0 item 10), computed once per (account, game), cutoff-independent.

Definitions (design §5 base quantities), with own move k = 1, 2, ... and c_k = clock after move k:
- think_k = c_{k-1} - c_k + increment, for k >= 2 (move 1 has no prior clock). Negative values
  (lag compensation) are clipped to 0 and counted.
- normalised think = think / estimated duration, where duration = base + 40 x increment.
  (The design says "÷ base time"; base can be 0 in blitz, e.g. 0+5, so we use the duration.)
- Timing features use own moves k >= 11 ("after move 10"), only from games with valid clocks.
- post-game rating = pre-game rating + RatingDiff; if RatingDiff is missing (~0.4%) we use the
  pre-game rating. The design's fallback (next game's pre-game rating) can reach past a cutoff,
  which would leak future information into a feature.

uv run python -m fairplay.features.game_features 2025-06 2025-07 2025-08 2025-09
"""

import argparse
from pathlib import Path

import duckdb

from fairplay import duck
from fairplay.duck import copy_atomic

PROCESSED = Path("data/processed")
FEATURE_VERSION = "f-1.1.0"  # 1.1: deterministic tie-breaks

GAME_FEATURES_SQL = """
WITH base AS (
    SELECT *,
        (tc_base + 40 * tc_inc)::DOUBLE AS duration,
        CASE WHEN color = 'w' THEN (n_plies + 1) // 2 ELSE n_plies // 2 END AS own_moves,
        CASE WHEN clock_valid THEN list_transform(
            range(2, len(own_clocks) + 1),
            lambda k: own_clocks[k - 1] - own_clocks[k] + tc_inc) END AS raw_think  -- moves 2..n
    FROM {source}
),
timed AS (
    SELECT *,
        raw_think[10:] AS raw_after10,                  -- raw_think[1] is move 2 -> [10:] = 11..n
        list_transform(raw_think, lambda t: greatest(t, 0)) AS think,
        raw_think[14:39] AS raw_15_40                   -- moves 15..40
    FROM base
),
ranked AS (
    SELECT *,
        CASE WHEN len(raw_15_40) >= 10 THEN list_transform(raw_15_40, lambda v:
            len(list_filter(raw_15_40, lambda w: greatest(w, 0) < greatest(v, 0)))
            + (len(list_filter(raw_15_40, lambda w: greatest(w, 0) = greatest(v, 0))) + 1) / 2.0)
        END AS r1540                                    -- average ranks (ties share a rank)
    FROM timed
)
SELECT
    account_id, game_key, month, color, start_ts, start_ts::DATE AS day,
    tc_base, tc_inc, tc_base || '+' || tc_inc AS tc,
    own_elo, opp_elo,
    own_elo + coalesce(own_rating_diff, 0) AS post_rating,
    own_rating_diff IS NULL AS rating_diff_missing,
    score,
    1.0 / (1.0 + power(10.0, (opp_elo - own_elo) / 400.0)) AS expected,
    termination = 'time_forfeit' AND score = 0 AS lost_on_time,
    termination = 'normal' AND score = 0 AND NOT ended_in_mate AND own_moves < 20
        AS early_resignation,
    clock_valid,
    list_transform(raw_after10, lambda t: greatest(t, 0) / duration) AS think_norm_after10,
    len(list_filter(raw_after10, lambda t: t < 0)) AS n_clipped_after10,
    len(list_filter(raw_after10, lambda t: greatest(t, 0) < 1)) AS n_instant_after10,
    CASE WHEN clock_valid AND len(own_clocks) >= 30 THEN own_clocks[30] / duration END
        AS clock_at_30_norm,
    CASE WHEN r1540 IS NOT NULL THEN
        abs(list_sum(list_transform(r1540, lambda r, i: (r - (len(r1540) + 1) / 2.0)
                                                       * (i - (len(r1540) + 1) / 2.0)))
            / nullif(sqrt(list_sum(list_transform(r1540, lambda r:
                                       power(r - (len(r1540) + 1) / 2.0, 2)))
                          * (len(r1540) * (power(len(r1540), 2) - 1) / 12.0)), 0))
    END AS abs_spearman_15_40,                          -- NULL when times are constant
    '{version}' AS feature_version
FROM ranked
"""


def game_features_sql(source: str) -> str:
    """source = any relation with the player_games schema: a Parquet file offline, a registered
    in-memory table online. One code path, so online and offline features cannot drift."""
    return GAME_FEATURES_SQL.format(source=source, version=FEATURE_VERSION)


def build_month(con: duckdb.DuckDBPyConnection, month: str, processed: Path = PROCESSED) -> Path:
    src = (processed / "player_games" / f"month={month}.parquet").as_posix()
    out_dir = processed / "game_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"month={month}.parquet"
    return copy_atomic(con, game_features_sql(f"read_parquet('{src}')"), out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("months", nargs="+")
    args = p.parse_args()
    con = duck.connect()
    for m in args.months:
        out = PROCESSED / "game_features" / f"month={m}.parquet"
        if out.exists():  # written atomically, so existing means finished
            print("skip (done)", out)
            continue
        print(build_month(con, m), flush=True)


if __name__ == "__main__":
    main()
