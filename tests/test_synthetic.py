"""Synthetic generator tests with fake engines (CI never needs Stockfish or lc0, design §12)."""

import datetime as dt

import chess
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fairplay.features import game_features as gf
from fairplay.synthetic import engines, glicko2
from fairplay.synthetic.generator import STRATEGIES, generate_account, play_game
from fairplay.synthetic.realism import realism_table
from fairplay.synthetic.timing import ThinkSampler, clock_bucket, move_bucket


class FakeHuman:
    """Uniform policy over legal moves (stands in for Maia)."""

    def policy(self, board, rating):
        return {m: 1.0 for m in board.legal_moves}


class FakeEngine:
    """'Best' = first legal move in UCI order; gap alternates 15 / 5 win% points."""

    def __init__(self):
        self.calls = 0

    def new_game(self):
        pass

    def best(self, board):
        return sorted(board.legal_moves, key=lambda m: m.uci())[0]

    def top2_gap(self, board):
        self.calls += 1
        return self.best(board), 15.0 if self.calls % 2 else 5.0


SAMPLER = ThinkSampler(cells={(b, m, c): np.array([0.004, 0.008, 0.012])
                              for b in range(1000, 2001, 200) for m in range(5)
                              for c in range(4)})


def test_policy_parser_reads_lc0_verbose_stats():
    board = chess.Board()
    lines = ["d2d4  (293 ) N:       0 (+ 0) (P: 22.43%) (WL:  -.-----) (D: -.---)",
             "e2e4  (322 ) N:       0 (+ 0) (P: 30.10%) (WL:  -.-----)",
             "node  ( 20) N:       1 (+ 0) (P: 100%)"]
    pol = engines.parse_policy(lines, board)
    assert pol == {chess.Move.from_uci("d2d4"): pytest.approx(0.2243),
                   chess.Move.from_uci("e2e4"): pytest.approx(0.3010)}


def test_critical_gap_is_100cp_at_equality_and_shrinks_when_decided():
    assert engines.CRITICAL_GAP_WIN == pytest.approx(9.10, abs=0.01)
    assert engines.win_percent(800) - engines.win_percent(500) < engines.CRITICAL_GAP_WIN
    assert engines.win_percent(5000) == engines.win_percent(1000)  # clipped like Lichess


def test_maia_rating_mapping_and_range():
    assert engines.nearest_maia_rating(1449) == 1400
    assert engines.nearest_maia_rating(1451) == 1500
    assert engines.nearest_maia_rating(1049) is None and engines.nearest_maia_rating(1950) is None


def test_sampling_is_reproducible_with_seed():
    pol = {m: 1.0 for m in chess.Board().legal_moves}
    a = [engines.sample_move(pol, np.random.default_rng(3)) for _ in range(5)]
    b = [engines.sample_move(pol, np.random.default_rng(3)) for _ in range(5)]
    assert a == b


def test_glicko2_matches_glickman_worked_example():
    r, rd, vol = glicko2.update(1500, 200, 0.06, [(1400, 30, 1), (1550, 100, 0), (1700, 300, 0)])
    assert r == pytest.approx(1464.06, abs=0.01)
    assert rd == pytest.approx(151.52, abs=0.01)
    assert vol == pytest.approx(0.05999, abs=1e-5)


def game(strategy, seed=0, tc=(180, 0)):
    return play_game(np.random.default_rng(seed), FakeHuman(), FakeEngine(), SAMPLER, 1500, 1500,
                     True, tc, STRATEGIES[strategy])


def test_games_are_legal_and_clocks_are_whole_seconds_and_decreasing():
    for seed in range(5):
        g = game("A", seed)  # play_game asserts legality of every move
        c = g["own_clocks"]
        assert all(x == int(x) for x in c)
        assert c[0] == 180 and all(b <= a for a, b in zip(c, c[1:], strict=False))


def test_no_substitution_in_first_ten_moves_and_share_close_to_p():
    for name in ("A", "B"):
        subs = after10 = 0
        for seed in range(40):
            g = game(name, seed)
            subs += g["n_substituted"]
            after10 += g["n_own_after10"]
            assert g["n_own_after10"] <= max(0, (g["n_plies"] + 1) // 2 - 10)
        assert abs(subs / after10 - STRATEGIES[name].p) <= 0.05  # design gate: within ±2 points
        # at full scale; ±5 here because 40 fake games give ~1,000 moves


def test_critical_strategy_never_exceeds_p():
    g = game("C", 1)
    assert g["n_substituted"] <= STRATEGIES["C"].p * g["n_own_after10"] + 1


def test_control_never_substitutes():
    assert all(game("control", s)["n_substituted"] == 0 for s in range(5))


def test_running_out_of_time_loses():
    slow = ThinkSampler(cells={(b, m, c): np.array([0.2]) for b in range(1000, 2001, 200)
                               for m in range(5) for c in range(4)})  # 36 s per move in 3+0
    g = play_game(np.random.default_rng(0), FakeHuman(), FakeEngine(), slow, 1500, 1500, True,
                  (180, 0), STRATEGIES["control"])
    assert g["termination"] == "time_forfeit" and g["n_plies"] < 20


def test_account_rows_run_through_the_real_feature_code(tmp_path):
    rows, summary = generate_account("syn-A-0000", STRATEGIES["A"], 1500, 400,
                                     [((180, 0), 0.7), ((300, 3), 0.3)],
                                     dt.datetime(2025, 8, 10), 7, FakeHuman(), FakeEngine(),
                                     SAMPLER, n_games=12)
    assert len(rows) == 12 and summary["strategy"] == "A"
    assert all(r["start_ts"] < dt.datetime(2025, 8, 10) for r in rows)
    assert rows[1]["own_elo"] == rows[0]["own_elo"] + rows[0]["own_rating_diff"]  # Glicko chain
    d = tmp_path / "player_games"
    d.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), d / "month=synthetic.parquet")
    out = gf.build_month(duckdb.connect(), "synthetic", processed=tmp_path)
    assert pq.read_table(out).num_rows == 12
    again, _ = generate_account("syn-A-0000", STRATEGIES["A"], 1500, 400,
                                [((180, 0), 0.7), ((300, 3), 0.3)], dt.datetime(2025, 8, 10), 7,
                                FakeHuman(), FakeEngine(), SAMPLER, n_games=12)
    assert again == rows  # same seed -> identical account


def test_buckets_and_realism_flags_shifted_feature():
    assert [move_bucket(k) for k in (2, 10, 11, 20, 21, 41)] == [0, 0, 1, 1, 2, 4]
    assert [clock_bucket(f) for f in (0.05, 0.1, 0.3, 0.9)] == [0, 1, 2, 3]
    rng = np.random.default_rng(0)
    synth = {"same": rng.normal(size=300), "shifted": rng.normal(1, 1, 300)}
    real = {"same": rng.normal(size=300), "shifted": rng.normal(size=300)}
    band = np.full(300, 1400.0)
    rep = realism_table(synth, real, band, band)
    assert rep["features_ks_above_limit"] == ["shifted"]


def test_play_model_clamps_while_account_selection_keeps_the_range():
    from fairplay.synthetic.engines import maia_for_play, nearest_maia_rating

    assert nearest_maia_rating(900) is None and nearest_maia_rating(2100) is None  # not selected
    assert maia_for_play(900) == 1100 and maia_for_play(2100) == 1900  # but playable in a game
    assert maia_for_play(1449) == nearest_maia_rating(1449) == 1400
