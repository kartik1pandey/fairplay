"""Every tier-1 feature against hand-computed values (design §5: "a toy-data unit test with a
hand-computed answer"), plus the point-in-time rule: games at/after cutoff - 30 min change nothing.
"""

import datetime as dt
import math

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fairplay.features import account_features as af
from fairplay.features import game_features as gf

CUTOFF = dt.datetime(2025, 7, 13)
E_200 = 1 / (1 + 10**0.5)  # expected score against an opponent rated 200 higher = 0.2402...


def clocks(start: float, thinks: list[float]) -> list[float]:
    """Own clock after each own move, given think times of moves 2..n (no increment)."""
    out = [start]
    for t in thinks:
        out.append(out[-1] - t)
    return out


def game(i, start_ts, *, score=1.0, opp=1500, diff=0, term="normal", plies=60, mate=False,
         thinks=None, own=1500, account="A", tc=(180, 0)):
    own_moves = (plies + 1) // 2
    thinks = thinks if thinks is not None else [2.0] * (own_moves - 1)
    return {"account_id": account, "game_key": f"k{i}", "month": "2025-07", "color": "w",
            "start_ts": start_ts, "tc_base": tc[0], "tc_inc": tc[1], "own_elo": own,
            "opp_elo": opp, "own_rating_diff": diff, "score": score, "termination": term,
            "ended_in_mate": mate, "n_plies": plies,
            "own_clocks": clocks(float(tc[0] + 40 * tc[1]), thinks), "clock_valid": True}


def build(tmp_path, games, accounts=("A",)):
    d = tmp_path / "player_games"
    d.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist(games), d / "month=2025-07.parquet")
    con = duckdb.connect()
    out = gf.build_month(con, "2025-07", processed=tmp_path)
    con.register("acc", pa.table({"account_id": list(accounts),
                                  "cutoff": [CUTOFF] * len(accounts)}))
    rel = af.cutoff_features(con, [out], "acc", CUTOFF)
    cols = rel.columns
    return {r[0]: dict(zip(cols, r, strict=True)) for r in rel.fetchall()}, con, out


def ten_games():
    """Day 1 (Jul 10): games 0-4 vs 1500: W W D L L. Day 2 (Jul 11): games 5-9 vs 1700: W L L L L.
    Game 6 is lost on time; game 8 is an early resignation (15 own moves)."""
    d1, d2 = dt.datetime(2025, 7, 10, 12), dt.datetime(2025, 7, 11, 12)
    res1 = [1.0, 1.0, 0.5, 0.0, 0.0]
    gs = [game(i, d1 + dt.timedelta(minutes=10 * i), score=s, diff=30 if i == 4 else 0)
          for i, s in enumerate(res1)]
    for j, s in enumerate([1.0, 0.0, 0.0, 0.0, 0.0]):
        i = 5 + j
        gs.append(game(i, d2 + dt.timedelta(minutes=10 * j), score=s, opp=1700,
                       diff=10 if i == 9 else 0,
                       term="time_forfeit" if i == 6 else "normal",
                       plies=30 if i == 8 else 60))
    return gs


def test_results_and_trajectory_features_by_hand(tmp_path):
    feats, *_ = build(tmp_path, ten_games())
    f = feats["A"]
    assert f["games_in_window"] == 10
    # S - E: day 1 = .5 + .5 + 0 - .5 - .5 = 0; day 2 = (1 - E) - 4E = 1 - 5E
    assert f["score_minus_expected"] == pytest.approx((1 - 5 * E_200) / 10)
    assert f["best_single_surplus"] == pytest.approx(1 - E_200)
    # perf = mean opp 1600 + 400 * (W 3 - L 6) / 10 = 1480; minus own 1500
    assert f["performance_rating_gap"] == pytest.approx(-20)
    assert f["upset_win_share"] == pytest.approx(1 / 5)  # 1 win in 5 games vs +200
    assert f["window_rating_change"] == 10  # post of last game 1510 - pre of first 1500
    assert f["largest_daily_gain"] == 30  # day 1: 1530 - 1500; day 2: 1510 - 1500
    assert f["time_loss_share"] == pytest.approx(0.1)
    assert f["early_resignation_share"] == pytest.approx(0.1)
    assert f["games_per_active_day"] == pytest.approx(5)


def test_timing_features_constant_think_time(tmp_path):
    f = build(tmp_path, ten_games())[0]["A"]
    # every think time is 2 s: 9 games x 20 moves (11..30) + game 8 with moves 11..15 = 185
    assert f["n_think_moves"] == 185
    assert f["think_time_cv"] == pytest.approx(0)
    assert f["even_timing_share"] == pytest.approx(1)
    assert f["instant_move_share"] == pytest.approx(0)
    assert f["time_trend_strength"] is None  # constant times: Spearman undefined in every game
    assert f["clock_at_30_raw"] == pytest.approx((180 - 29 * 2) / 180)  # game 8 never reaches 30


def test_timing_features_two_value_think_times(tmp_path):
    # moves 2-10 take 2 s; moves 11-30 alternate 0.5 s and 3.5 s -> 20 values per game
    thinks = [2.0] * 9 + [0.5, 3.5] * 10
    gs = [game(i, dt.datetime(2025, 7, 10) + dt.timedelta(hours=i), thinks=thinks)
          for i in range(10)]
    f = build(tmp_path, gs)[0]["A"]
    n = 200
    assert f["n_think_moves"] == n
    # mean 2, deviations +-1.5 -> sample sd = 1.5 * sqrt(n / (n - 1)); cv = sd / mean
    assert f["think_time_cv"] == pytest.approx(0.75 * math.sqrt(n / (n - 1)), rel=1e-5)
    assert f["instant_move_share"] == pytest.approx(0.5)  # every 0.5 s move is < 1 s
    assert f["even_timing_share"] == pytest.approx(0)  # median 2; nothing within +-0.4


def test_timing_needs_100_moves(tmp_path):
    gs = [game(i, dt.datetime(2025, 7, 10) + dt.timedelta(hours=i), plies=30) for i in range(10)]
    f = build(tmp_path, gs)[0]["A"]  # 10 games x moves 11..15 = 50 moves
    assert f["n_think_moves"] == 50
    assert f["think_time_cv"] is None and f["instant_move_share"] is None


def test_games_at_or_after_cutoff_minus_30_minutes_change_nothing(tmp_path):
    before, *_ = build(tmp_path, ten_games())
    late = [game(100, CUTOFF - dt.timedelta(minutes=29), opp=2500, diff=200),
            game(101, CUTOFF + dt.timedelta(hours=1), opp=2500, diff=200)]
    after, *_ = build(tmp_path, ten_games() + late)
    assert after["A"] == before["A"]


def test_fast_improver_gain_needs_50_games_in_60_days(tmp_path):
    start = CUTOFF - dt.timedelta(days=55)
    gs = [game(i, start + dt.timedelta(days=i), own=1500 + 4 * i, diff=4) for i in range(50)]
    feats = build(tmp_path, gs)[0]
    # last 50 games: first pre-game 1500, last post-game 1500 + 4*49 + 4 = 1700
    assert feats["A"]["gain_last50"] == 200
    feats = build(tmp_path, gs[1:])[0]  # 49 games -> insufficient history
    assert feats["A"]["gain_last50"] is None


def test_negative_think_times_are_clipped_and_counted(tmp_path):
    thinks = [2.0] * 9 + [-1.0] + [2.0] * 19  # move 11 shows the clock going up 1 s
    gs = [game(i, dt.datetime(2025, 7, 10) + dt.timedelta(hours=i), thinks=thinks)
          for i in range(10)]
    _, con, out = build(tmp_path, gs)
    row = con.execute(f"SELECT think_norm_after10, n_clipped_after10, n_instant_after10 "
                      f"FROM '{out.as_posix()}' LIMIT 1").fetchone()
    assert row[0][0] == 0 and row[1] == 1 and row[2] == 1  # clipped to 0 s, which is instant


def python_abs_spearman(x: list[float]) -> float:
    """Independent reference: average ranks for ties, then Pearson against 1..m."""
    m = len(x)
    ranks = [sum(v < xi for v in x) + (sum(v == xi for v in x) + 1) / 2 for xi in x]
    mu = (m + 1) / 2
    cov = sum((r - mu) * (i + 1 - mu) for i, r in enumerate(ranks))
    var_r = sum((r - mu) ** 2 for r in ranks)
    return abs(cov / math.sqrt(var_r * (m * (m * m - 1) / 12)))


def test_spearman_15_40_matches_reference_with_ties(tmp_path):
    middle = [1.0, 3.0, 2.0, 2.0, 5.0, 1.0, 4.0, 4.0, 6.0, 2.0, 7.0, 3.0, 8.0,
              2.0, 9.0, 5.0, 1.0, 6.0, 3.0, 2.0, 4.0, 1.0, 2.0, 3.0, 1.0, 2.0]  # moves 15..40
    thinks = [1.0] * 13 + middle + [1.0] * 4  # moves 2..14, 15..40, 41..44
    _, con, out = build(tmp_path, [game(0, dt.datetime(2025, 7, 10), plies=88, thinks=thinks)])
    (rho,) = con.execute(f"SELECT abs_spearman_15_40 FROM '{out.as_posix()}'").fetchone()
    assert rho == pytest.approx(python_abs_spearman(middle))


def test_post_rating_falls_back_to_pre_rating_without_leaking(tmp_path):
    g = game(0, dt.datetime(2025, 7, 10))
    g["own_rating_diff"] = None
    _, con, out = build(tmp_path, [g])
    assert con.execute(f"SELECT post_rating, rating_diff_missing FROM '{out.as_posix()}'"
                       ).fetchone() == (1500, True)


def test_duration_normalisation_handles_zero_base(tmp_path):
    g = game(0, dt.datetime(2025, 7, 10), tc=(0, 5))  # 0+5 is blitz: 0 + 40 x 5 = 200 s
    _, con, out = build(tmp_path, [g])
    (first,) = con.execute(f"SELECT think_norm_after10[1] FROM '{out.as_posix()}'").fetchone()
    assert first == pytest.approx((2 + 5) / 200)  # think = 2 s used + 5 s increment


def test_peer_z_uses_training_rows_only():
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT * FROM (VALUES
        ('a', 'train', 1400, '180+0', 0.4), ('b', 'train', 1400, '180+0', 0.5),
        ('c', 'train', 1400, '180+0', 0.6), ('d', 'test', 1400, '180+0', 0.9),
        ('e', 'test', 1400, '180+0', 5.0))
        AS v(account_id, split, rating_band, tc_group, clock_at_30_raw)""")
    z = dict(con.execute(f"SELECT account_id, clock_at_30_z FROM {af.add_peer_z(con, 't')}"
                         ).fetchall())
    # train median 0.5, MAD 0.1 (fewer than 500 accounts -> global train stats); test rows unused
    assert z["d"] == pytest.approx((0.9 - 0.5) / (1.4826 * 0.1))
    assert z["b"] == pytest.approx(0)


def test_ties_in_time_control_and_start_time_resolve_the_same_way(tmp_path):
    """5 games at 300+0 and 5 at 180+0: the modal time control is a tie, broken
    alphabetically; two last games share a start time, broken by game key."""
    t0 = dt.datetime(2025, 7, 10, 12)
    gs = [game(i, t0 + dt.timedelta(minutes=10 * i), tc=(300, 0) if i < 5 else (180, 0),
               own=1500 + i, diff=1) for i in range(9)]
    gs.append(game(9, gs[-1]["start_ts"], own=1600, diff=1))  # same start as game 8
    f, _, _ = build(tmp_path, gs)
    assert f["A"]["tc_group"] == "180+0"
    assert f["A"]["rating_at_cutoff"] == 1508 + 1  # k8 sorts before k9 at the same start time
