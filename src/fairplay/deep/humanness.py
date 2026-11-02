"""Maia "humanness" features on a case-control sample of July rows (research).

Question: does "how human are this player's move CHOICES at their rating" add signal beyond
tier 1? Maia-1 (lc0, nodes=1, the raw policy) gives the probability a human at that rating
plays each move. Per game, over own moves FIRST_MOVE..LAST_MOVE:
  mean_logp   mean log P_maia(played move)            (engine-like play is often less human)
  top1_share  share of moves equal to Maia's top choice
  rare_share  share of played moves with P_maia < RARE (choices few humans at that level make)
Per account row: mean of each over its last GAMES games in the tier-1 window, and the minimum
mean_logp (the least human game). Maia covers ratings 1100-1900; other ratings use the nearest
network and set `maia_in_range` = 0.

Cost: ~150-600 positions/s per lc0 process, so full July coverage would take days. The sample
is case-control, as design §7 samples tier-2 training: every row of every flagged account plus
the rows of a random NEG_RATE share of unflagged accounts, weighted 1 / NEG_RATE. Results are
cached per game (data/processed/humanness/), so a killed run resumes.

uv run python -m fairplay.deep.humanness --workers 4
"""

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fairplay import duck
from fairplay.dataset.eligibility import END_MARGIN, WINDOW
from fairplay.features.account_features import PROCESSED
from fairplay.models.data import Split

OUT = PROCESSED / "humanness"
GAMES = 5
FIRST_MOVE, LAST_MOVE = 11, 30
RARE = 0.02
NEG_RATE = 0.10
CHUNK = 500  # games per cached part file
BLOCK = 5000  # games whose moves are held in memory at once
ROW_FEATURES = ("maia_mean_logp", "maia_min_game_logp", "maia_top1_share", "maia_rare_share",
                "maia_in_range")


def sample(split: Split, neg_rate: float = NEG_RATE, seed: int = 0
           ) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of the case-control sample and their inverse-probability weights."""
    accounts = np.unique(split.account_id)
    flagged = set(split.account_id[split.y == 1])
    rng = np.random.default_rng(seed)
    keep_neg = {a for a in accounts if a not in flagged and rng.random() < neg_rate}
    rows = np.flatnonzero([a in flagged or a in keep_neg for a in split.account_id])
    ipw = np.where(split.y[rows] == 1, 1.0, 1 / neg_rate)
    return rows, ipw


def game_stats(sans: list[str], color: str, rating: int, maia) -> dict:
    from fairplay.synthetic.engines import MAIA_RATINGS, maia_for_play

    net = maia_for_play(rating)
    board, logps, top1, rare = chess.Board(), [], 0, 0
    own_parity = 0 if color == "w" else 1
    for ply, san in enumerate(sans):
        move = board.parse_san(san)
        move_no = ply // 2 + 1
        if ply % 2 == own_parity and FIRST_MOVE <= move_no <= LAST_MOVE:
            policy = maia.policy(board, net)
            p = policy.get(move, 0.0)
            logps.append(math.log(max(p, 1e-4)))
            top1 += move == max(policy, key=policy.get)
            rare += p < RARE
        if move_no > LAST_MOVE:
            break
        board.push(move)
    n = len(logps)
    return {"n_moves": n, "mean_logp": float(np.mean(logps)) if n else None,
            "top1_share": top1 / n if n else None, "rare_share": rare / n if n else None,
            "in_range": int(MAIA_RATINGS[0] - 50 <= rating < MAIA_RATINGS[-1] + 50)}


def _work(batch: list[tuple]) -> list[dict]:
    from fairplay.synthetic.engines import Maia

    maia = Maia()
    try:
        return [{"game_key": k, "color": c} | game_stats(list(s), c, r, maia)
                for k, c, r, s in batch]
    finally:
        maia.close()


def row_games(split: Split, rows: np.ndarray) -> pa.Table:
    """(row, game_key, color, own_elo) for the last GAMES games of each sampled row."""
    con = duck.connect()
    files = sorted((PROCESSED / "game_features").glob("month=*.parquet"))
    src = "read_parquet([" + ", ".join(f"'{f.as_posix()}'" for f in files) + "])"
    con.register("r", pa.table({"row": rows, "account_id": split.account_id[rows],
                                "cutoff": split.cutoff[rows]}))
    t = con.sql(f"""
        SELECT row, game_key, color, own_elo FROM (
            SELECT r.row, g.game_key, g.color, g.own_elo,
                   row_number() OVER (PARTITION BY r.row ORDER BY g.start_ts DESC, g.game_key)
                       AS recency
            FROM r JOIN {src} g USING (account_id)
            WHERE g.start_ts >= r.cutoff - INTERVAL {WINDOW.days} DAY
              AND g.start_ts <  r.cutoff - INTERVAL {int(END_MARGIN.total_seconds() // 60)} MINUTE)
        WHERE recency <= {GAMES}
    """).arrow()
    return t.read_all() if hasattr(t, "read_all") else t


def done_keys(out: Path = OUT) -> set[tuple[str, str]]:
    if not out.exists():
        return set()
    return {(k, c) for f in out.glob("part-*.parquet")
            for k, c in zip(*pq.read_table(f, columns=["game_key", "color"]).to_pydict().values(),
                            strict=True)}


def compute(games: pa.Table, workers: int, out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    have = done_keys(out)
    todo = {(k, c): e for k, c, e in zip(games["game_key"].to_pylist(),
                                         games["color"].to_pylist(),
                                         games["own_elo"].to_pylist(), strict=True)
            if (k, c) not in have}
    print(f"{len(have)} games cached, {len(todo)} to compute", flush=True)
    if not todo:
        return
    files = sorted((PROCESSED / "game_moves").glob("month=*.parquet"))
    src = "read_parquet([" + ", ".join(f"'{f.as_posix()}'" for f in files) + "])"
    con = duck.connect()
    keys = list(todo)
    part = len(list(out.glob("part-*.parquet")))
    with ProcessPoolExecutor(workers) as pool:
        for b0 in range(0, len(keys), BLOCK):  # moves are fetched per block, not all at once
            block = keys[b0:b0 + BLOCK]
            con.register("t", pa.table({"game_key": sorted({k for k, _ in block})}))
            sans = dict(con.execute(f"SELECT game_key, any_value(sans) FROM {src} "
                                    "JOIN t USING (game_key) GROUP BY 1").fetchall())
            con.unregister("t")
            items = [(k, c, todo[(k, c)], sans[k]) for k, c in block if k in sans]
            batches = [items[i:i + CHUNK] for i in range(0, len(items), CHUNK)]
            for res in pool.map(_work, batches):
                path = out / f"part-{part:06d}.parquet"
                pq.write_table(pa.Table.from_pylist(res), path.with_suffix(".tmp"))
                os.replace(path.with_suffix(".tmp"), path)
                part += 1
            print(f"{min(b0 + BLOCK, len(keys))}/{len(keys)} games", flush=True)


def row_features(split: Split, rows: np.ndarray, games: pa.Table, out: Path = OUT) -> np.ndarray:
    """[len(rows), len(ROW_FEATURES)], NaN where a row has no scored game."""
    con = duck.connect()
    con.register("g", games)
    stats = f"read_parquet('{(out / 'part-*.parquet').as_posix()}')"
    agg = con.sql(f"""
        SELECT g.row, avg(s.mean_logp), min(s.mean_logp), avg(s.top1_share), avg(s.rare_share),
               min(s.in_range)
        FROM g JOIN {stats} s USING (game_key, color)
        WHERE s.n_moves > 0 GROUP BY g.row
    """).fetchall()
    X = np.full((len(rows), len(ROW_FEATURES)), np.nan)
    where = {r: i for i, r in enumerate(rows)}
    for r, *vals in agg:
        X[where[r]] = vals
    return X


def main() -> None:
    from fairplay.models.data import load_split
    from fairplay.models.run_baselines import TABLE

    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    train = load_split(TABLE, "train")
    rows, _ = sample(train)
    games = row_games(train, rows)
    print(f"{len(rows)} sampled rows, {games.num_rows} row-games", flush=True)
    compute(games, args.workers)


if __name__ == "__main__":
    main()
