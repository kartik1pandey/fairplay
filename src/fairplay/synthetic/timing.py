"""Human think times for synthetic games (design §6), from real honest training-month accounts.

Cell = (200-point rating band, move-number bucket, clock-remaining bucket). Values are think time
normalised by the game's estimated duration, so one table serves every time control. Seeds come
from July (training) accounts unflagged at S0 only; evaluation months are never used.
"""

from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

MOVE_EDGES = (11, 21, 31, 41)  # buckets: 2-10, 11-20, 21-30, 31-40, 41+
CLOCK_EDGES = (0.1, 0.25, 0.5)  # share of duration left before the move
PER_CELL = 5000
BANDS = tuple(range(1000, 2001, 200))


def move_bucket(k: int) -> int:
    return int(np.searchsorted(MOVE_EDGES, k, side="right"))


def clock_bucket(frac: float) -> int:
    return int(np.searchsorted(CLOCK_EDGES, frac, side="right"))


def band_of(rating: float) -> int:
    return int(min(max((rating // 200) * 200, BANDS[0]), BANDS[-1]))


def build_table(con: duckdb.DuckDBPyConnection, player_games: Path, honest_accounts: str,
                out: Path, seed: float = 0.42) -> Path:
    """honest_accounts: a table/view name with one column account_id.

    Sampling per cell orders moves by hash(game, move, seed): deterministic for a given seed and
    data, unlike random() (which DuckDB 2.0 also rejects in a window ORDER BY)."""
    con.execute(f"""
        COPY (
        WITH g AS (
            SELECT least(greatest((own_elo // 200) * 200, {BANDS[0]}), {BANDS[-1]}) AS band,
                   (tc_base + 40 * tc_inc)::DOUBLE AS dur, tc_inc, own_clocks, game_key,
                   unnest(range(2, len(own_clocks) + 1)) AS k
            FROM read_parquet('{player_games.as_posix()}')
            WHERE clock_valid AND account_id IN (SELECT account_id FROM {honest_accounts})
        ), m AS (
            SELECT band, k, game_key,
                CASE WHEN k >= 41 THEN 4 WHEN k >= 31 THEN 3 WHEN k >= 21 THEN 2
                     WHEN k >= 11 THEN 1 ELSE 0 END AS move_b,
                CASE WHEN own_clocks[k - 1] / dur >= 0.5 THEN 3
                     WHEN own_clocks[k - 1] / dur >= 0.25 THEN 2
                     WHEN own_clocks[k - 1] / dur >= 0.1 THEN 1 ELSE 0 END AS clock_b,
                greatest(own_clocks[k - 1] - own_clocks[k] + tc_inc, 0) / dur AS think_norm
            FROM g
        )
        SELECT band, move_b, clock_b, think_norm FROM (
            SELECT *, row_number() OVER (PARTITION BY band, move_b, clock_b
                                         ORDER BY hash(game_key, k, '{seed}')) AS r
            FROM m) WHERE r <= {PER_CELL}
        ) TO '{out.as_posix()}' (FORMAT parquet)
    """)
    return out


class ThinkSampler:
    """Draws a human think time (seconds) for a move; falls back to coarser cells if empty."""

    def __init__(self, table: Path | None = None, cells: dict | None = None):
        if cells is None:
            t = pq.read_table(table).to_pydict()
            cells = {}
            for b, mb, cb, v in zip(t["band"], t["move_b"], t["clock_b"], t["think_norm"],
                                    strict=True):
                cells.setdefault((b, mb, cb), []).append(v)
        self.cells = {k: np.asarray(v) for k, v in cells.items()}
        self.by_band_move: dict = {}
        for (b, mb, _), v in self.cells.items():
            self.by_band_move.setdefault((b, mb), []).append(v)
        self.by_band_move = {k: np.concatenate(v) for k, v in self.by_band_move.items()}
        self.all = np.concatenate(list(self.cells.values()))

    def sample(self, rng: np.random.Generator, rating: float, move_no: int, clock_left: float,
               duration: float) -> float:
        key = (band_of(rating), move_bucket(move_no), clock_bucket(clock_left / duration))
        pool = self.cells.get(key)
        if pool is None or len(pool) == 0:
            pool = self.by_band_move.get(key[:2], self.all)
        return float(pool[rng.integers(len(pool))] * duration)
