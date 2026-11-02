"""Bags of games: the input of the attention and sequence models (research).

One bag per model-table row (account, cutoff) = the account's last 30 games in the tier-1 window,
[cutoff - 30 days, cutoff - 30 minutes), ordered by recency exactly as account_features orders
them (start time, then game key). So the deep models see the same games tier 1 summarised, and
nothing after the cutoff.

Per game: GAME_FEATURES (results, rating, clock and timing summaries, all from that game's own
headers and clocks), and the sequence of its own think times after move 10 (log seconds, first
SEQ_LEN moves), for the move-sequence model.

Arrays, aligned with load_split(TABLE, split) row order:
  games [rows, 30, F] float32 (0 where padded), mask [rows, 30] bool, seq [rows, 30, SEQ_LEN]
  float16 (NaN where no move). Saved to data/processed/bags/<split>.npz (git-ignored).

uv run python -m fairplay.deep.bags --split train
"""

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pyarrow as pa

from fairplay import duck
from fairplay.dataset.eligibility import END_MARGIN, WINDOW
from fairplay.features.account_features import LAST_N, PROCESSED
from fairplay.models.data import Split

OUT = PROCESSED / "bags"
SEQ_LEN = 40

# name -> SQL over one game row g (think list = g.thinks, log seconds); NULL-free by construction
GAME_FEATURES = {
    "surplus": "score - expected",
    "score": "score::DOUBLE",
    "opp_gap": "(opp_elo - own_elo) / 400.0",
    "own_elo": "(own_elo - 1500) / 400.0",
    "rating_change": "coalesce(post_rating - own_elo, 0) / 20.0",
    "lost_on_time": "lost_on_time::DOUBLE",
    "early_resignation": "early_resignation::DOUBLE",
    "clock_valid": "clock_valid::DOUBLE",
    "clock30": "coalesce(clock_at_30_norm, 0)",
    "has_clock30": "(clock_at_30_norm IS NOT NULL)::DOUBLE",
    "spearman": "coalesce(abs_spearman_15_40, 0)",
    "has_spearman": "(abs_spearman_15_40 IS NOT NULL)::DOUBLE",
    "n_moves": "len(thinks) / 50.0",
    "think_mean": "coalesce(list_avg(thinks), 0)",
    "think_sd": "coalesce(list_stddev_samp(thinks), 0)",
    "instant_share": "coalesce(n_instant_after10 / nullif(len(thinks), 0), 0)",
    "clipped_share": "coalesce(n_clipped_after10 / nullif(len(thinks), 0), 0)",
    "even_share": "coalesce(len(list_filter(raw, lambda x: abs(x - list_median(raw)) "
                  "<= 0.2 * list_median(raw))) / nullif(len(raw), 0), 0)",
    "log_base": "ln(1 + tc_base) / 6.0",  # 0+N games are blitz too
    "increment": "tc_inc / 5.0",
    "gap_hours": "ln(1 + coalesce(epoch(start_ts - prev_ts) / 3600.0, 720))",
    "recency_frac": "recency / 30.0",
}

BAG_SQL = """
WITH w AS (
    SELECT g.*,
        row_number() OVER (PARTITION BY g.account_id ORDER BY g.start_ts DESC, g.game_key)
            AS recency,
        lag(g.start_ts) OVER (PARTITION BY g.account_id ORDER BY g.start_ts, g.game_key)
            AS prev_ts
    FROM {source} g
    JOIN (SELECT account_id FROM rows WHERE cutoff = TIMESTAMP '{cutoff}') r USING (account_id)
    WHERE g.start_ts >= TIMESTAMP '{cutoff}' - INTERVAL {window_days} DAY
      AND g.start_ts <  TIMESTAMP '{cutoff}' - INTERVAL {margin_min} MINUTE
),
last AS (
    SELECT *,
        CASE WHEN clock_valid THEN think_norm_after10 ELSE []::DOUBLE[] END AS raw,
        CASE WHEN clock_valid
             THEN list_transform(think_norm_after10, lambda x: ln(x * tc_base + 0.1))
             ELSE []::DOUBLE[] END AS thinks
    FROM w WHERE recency <= {n}
)
SELECT account_id, game_key, recency, {features},
       thinks[1:{seq_len}]::FLOAT[] AS seq
FROM last
"""


def bag_sql(source: str, cutoff: dt.datetime) -> str:
    feats = ", ".join(f"({sql})::FLOAT AS {name}" for name, sql in GAME_FEATURES.items())
    return BAG_SQL.format(source=source, cutoff=cutoff.isoformat(sep=" "),
                          window_days=WINDOW.days,
                          margin_min=int(END_MARGIN.total_seconds() // 60),
                          n=LAST_N, features=feats, seq_len=SEQ_LEN)


def pad_lists(col: pa.ChunkedArray, width: int) -> np.ndarray:
    """List column -> [n, width] float array, NaN where a list is shorter."""
    arr = col.combine_chunks()
    lengths = np.minimum(np.diff(arr.offsets.to_numpy()), width)
    values = arr.flatten().to_numpy(zero_copy_only=False)
    out = np.full((len(arr), width), np.nan, dtype=np.float32)
    starts = arr.offsets.to_numpy()[:-1] - arr.offsets[0].as_py()
    rows = np.repeat(np.arange(len(arr)), lengths)
    cols = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    out[rows, cols] = values[np.repeat(starts, lengths) + cols]
    return out


def build(split: Split, processed: Path = PROCESSED) -> dict[str, np.ndarray]:
    files = sorted((processed / "game_features").glob("month=*.parquet"))
    source = "read_parquet([" + ", ".join(f"'{f.as_posix()}'" for f in files) + "])"
    n, F = len(split.y), len(GAME_FEATURES)
    games = np.zeros((n, LAST_N, F), dtype=np.float32)
    seq = np.full((n, LAST_N, SEQ_LEN), np.nan, dtype=np.float16)
    mask = np.zeros((n, LAST_N), dtype=bool)
    con = duck.connect()
    cutoffs = np.unique(split.cutoff)
    for cutoff in cutoffs:
        at = np.flatnonzero(split.cutoff == cutoff)
        row_of = dict(zip(split.account_id[at], at, strict=True))
        con.register("rows", pa.table({"account_id": split.account_id[at],
                                       "cutoff": np.full(len(at), cutoff)}))
        c = cutoff.astype("datetime64[us]").astype(dt.datetime)
        t = con.sql(bag_sql(source, c)).arrow()
        if hasattr(t, "read_all"):
            t = t.read_all()
        r = np.array([row_of[a] for a in t["account_id"].to_pylist()])
        slot = t["recency"].to_numpy() - 1
        games[r, slot] = np.column_stack([t[f].to_numpy() for f in GAME_FEATURES])
        seq[r, slot] = pad_lists(t["seq"], SEQ_LEN)
        mask[r, slot] = True
        con.unregister("rows")
        print("bags for cutoff", c.date(), len(at), "rows", flush=True)
    return {"games": games, "mask": mask, "seq": seq}


def save(split: Split, arrays: dict[str, np.ndarray], out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{split.name}.npz"
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, account_id=split.account_id, cutoff=split.cutoff, **arrays)
    tmp.replace(path)
    return path


def load(split: Split, out: Path = OUT) -> dict[str, np.ndarray]:
    z = np.load(out / f"{split.name}.npz", allow_pickle=True)
    if not (np.array_equal(z["account_id"], split.account_id)
            and np.array_equal(z["cutoff"], split.cutoff)):
        raise SystemExit(f"bags/{split.name}.npz rows differ from the model table; rebuild it")
    return {k: z[k] for k in ("games", "mask", "seq")}


def main() -> None:
    from fairplay.models.data import load_split
    from fairplay.models.run_baselines import TABLE

    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train")
    args = p.parse_args()
    split = load_split(TABLE, args.split)
    arrays = build(split)
    games_in_window = np.minimum(split.games, LAST_N)
    assert np.array_equal(arrays["mask"].sum(1), games_in_window), "bag sizes != tier-1 counts"
    print("saved", save(split, arrays), {k: v.shape for k, v in arrays.items()})


if __name__ == "__main__":
    main()
