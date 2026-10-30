"""Hand-review evidence (design §0.15): rubric helpers, never a verdict."""

import datetime as dt

import chess
import pytest

from fairplay.review import evidence as ev


def test_obvious_moves_neither_extend_nor_break_a_run():
    m, o, x = (False, True), (True, False), (False, False)  # matched / obvious / missed
    assert ev.longest_match_run([m, m, o, m, x, m]) == 3
    assert ev.longest_match_run([o, o, o]) == 0


def test_think_times_clip_lag_and_skip_first_move():
    assert ev.think_times([180, 178, 179, 170], 0) == [2, 0, 9]
    assert ev.think_times([180, 178], 2) == [4]


def test_early_loss_counts_resignations_and_flags_not_mates():
    g = {"n_plies": 20, "color": "w", "score": 0.0, "termination": "normal",
         "ended_in_mate": False}
    assert ev.is_early_loss(g)
    assert not ev.is_early_loss(g | {"ended_in_mate": True})
    assert ev.is_early_loss(g | {"termination": "time_forfeit", "ended_in_mate": False})
    assert not ev.is_early_loss(g | {"n_plies": 40})  # own move 20: too late
    assert not ev.is_early_loss(g | {"score": 0.5})


def test_manipulation_pattern_needs_losses_and_recovery():
    t0 = dt.datetime(2025, 7, 1)
    loss = {"n_plies": 10, "color": "w", "score": 0.0, "termination": "normal",
            "ended_in_mate": False, "own_rating_diff": -10}
    win = {"n_plies": 60, "color": "w", "score": 1.0, "termination": "normal",
           "ended_in_mate": True, "own_rating_diff": 40}
    games = [loss | {"start_ts": t0 + dt.timedelta(hours=i), "own_elo": 1500 - 10 * i}
             for i in range(5)]
    games += [win | {"start_ts": t0 + dt.timedelta(days=8, hours=i), "own_elo": 1450 + 40 * i}
              for i in range(5)]
    r = ev.manipulation_pattern(games)
    assert r["max_early_losses_7d"] == 5 and r["recovery_after"] >= 150 and r["pattern"]
    assert not ev.manipulation_pattern(games[:5])["pattern"]  # no recovery


class FakeEngine:
    """Best move = first legal move in UCI order; gap 100 when only one capture exists."""
    def top2_gap(self, board):
        legal = sorted(board.legal_moves, key=lambda m: m.uci())
        return legal[0], (float("inf") if len(legal) == 1 else 5.0)


def test_engine_steps_only_own_moves_after_move_10():
    board, sans = chess.Board(), []
    for _ in range(30):
        mv = sorted(board.legal_moves, key=lambda m: m.uci())[0]
        sans.append(board.san(mv))
        board.push(mv)
    steps = ev.engine_steps(sans, True, FakeEngine())
    assert [s["move_no"] for s in steps] == list(range(11, 16))
    assert all(s["matched"] for s in steps)  # the game was played with the engine's choice


def test_kappa():
    assert ev.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert ev.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
