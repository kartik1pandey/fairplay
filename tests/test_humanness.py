import math

import chess
import numpy as np

from fairplay.deep.humanness import game_stats, sample
from fairplay.models.data import Split

SANS = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5",
        "Bb3", "d6", "c3", "O-O", "h3", "Nb8", "d4", "Nbd7", "Nbd2", "Bb7", "Bc2", "Re8",
        "Nf1", "Bf8", "Ng3", "g6"]


class FakeMaia:
    """Puts probability 0.5 on the first legal move in UCI order, the rest spread evenly."""

    def __init__(self):
        self.calls = 0

    def policy(self, board: chess.Board, rating: int) -> dict:
        self.calls += 1
        legal = sorted(board.legal_moves, key=lambda m: m.uci())
        rest = 0.5 / (len(legal) - 1)
        return {m: (0.5 if i == 0 else rest) for i, m in enumerate(legal)}


def test_game_stats_scores_only_own_moves_in_range():
    maia = FakeMaia()
    s = game_stats(SANS, "w", 1500, maia)
    assert maia.calls == 4 and s["n_moves"] == 4  # white moves 11-14 of a 14-move game
    assert s["in_range"] == 1 and s["mean_logp"] < math.log(0.5) + 1e-9
    assert 0 <= s["top1_share"] <= 1 and 0 <= s["rare_share"] <= 1


def test_out_of_range_rating_is_marked():
    assert game_stats(SANS, "b", 2400, FakeMaia())["in_range"] == 0


def test_case_control_sample_keeps_every_flagged_account():
    acc = np.repeat([f"a{i}" for i in range(200)], 3)
    y = np.isin(acc, ["a1", "a2"]).astype(np.int8)
    split = Split("t", np.zeros((600, 16)), y, np.ones(600), acc, np.zeros(600), np.zeros(600))
    rows, ipw = sample(split, neg_rate=0.1, seed=0)
    assert set(np.flatnonzero(y)) <= set(rows)
    assert np.all(ipw[y[rows] == 1] == 1) and np.all(ipw[y[rows] == 0] == 10)
    kept_neg = len(set(acc[rows]) - {"a1", "a2"})
    assert 5 <= kept_neg <= 40  # about 10% of 198


def test_top_precision_counts_the_weighted_top():
    from fairplay.deep.run_humanness import top_precision

    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.arange(10, 0, -1, dtype=float)  # row 0 ranked first
    w = np.ones(10)
    assert top_precision(y, w, s, capacity=0.1) == 1.0  # top 1 row
    assert top_precision(y, w, s, capacity=0.3) == 2 / 3
    w2 = w.copy()
    w2[1] = 10  # a heavy negative in second place fills the top 50% of weight
    assert top_precision(y, w2, s, capacity=0.5) < 0.5
