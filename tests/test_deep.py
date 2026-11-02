import numpy as np
import pytest

from fairplay.deep.bags import pad_lists
from fairplay.deep.mil import Bags, shuffle_moves


def test_pad_lists_pads_with_nan_and_truncates():
    pa = pytest.importorskip("pyarrow")
    col = pa.chunked_array([pa.array([[1.0, 2.0], [], [3.0, 4.0, 5.0, 6.0]])])
    out = pad_lists(col, 3)
    assert np.array_equal(out[0, :2], [1, 2]) and np.isnan(out[0, 2])
    assert np.isnan(out[1]).all()
    assert np.array_equal(out[2], [3, 4, 5])


def test_shuffle_keeps_each_games_moves_and_padding():
    seq = np.full((2, 3, 6), np.nan, dtype=np.float16)
    seq[0, 0, :4] = [1, 2, 3, 4]
    seq[1, 2, :6] = [6, 5, 4, 3, 2, 1]
    out = shuffle_moves(seq, seed=0)
    assert np.array_equal(np.sort(out[0, 0, :4]), [1, 2, 3, 4])
    assert np.isnan(out[0, 0, 4:]).all() and np.isnan(out[0, 1]).all()
    assert sorted(out[1, 2]) == [1, 2, 3, 4, 5, 6]
    assert not np.array_equal(out[1, 2], seq[1, 2])  # order changed (fixed seed)


def toy_bags(n=600, seed=0):
    """Positives have ONE planted game with a high value of game feature 0; the rest is noise."""
    rng = np.random.default_rng(seed)
    games = rng.normal(size=(n, 30, 22)).astype(np.float32)
    mask = np.zeros((n, 30), dtype=bool)
    sizes = rng.integers(10, 31, size=n)
    for i, s in enumerate(sizes):
        mask[i, :s] = True
    y = (rng.random(n) < 0.3).astype(np.int8)
    planted = np.array([rng.integers(0, s) for s in sizes])
    games[y == 1, planted[y == 1], 0] = 6.0
    seq = rng.normal(size=(n, 30, 8)).astype(np.float16)
    tab = rng.normal(size=(n, 16))
    return Bags(games, mask, seq, tab), y, planted


def test_attention_finds_the_planted_game():
    pytest.importorskip("torch")
    from fairplay.deep.mil import VARIANTS, deletion_test, fit

    b, y, planted = toy_bags()
    cfg = {"d": 16, "channels": 4, "dropout": 0.0, "lr": 1e-2, "weight_decay": 0.0,
           "batch": 64, "max_epochs": 15, "patience": 15}
    m = fit(VARIANTS["mil"], b, y, np.ones(len(y)), np.arange(len(y)), seed=0, epochs=15,
            cfg=cfg)
    score, att = m.score(b)
    assert score[y == 1].mean() > score[y == 0].mean()
    assert (att[y == 1].argmax(1) == planted[y == 1]).mean() > 0.8
    d = deletion_test(m, b, np.flatnonzero(y == 1), seed=0)
    assert d["mean_logit_drop_top"] > d["mean_logit_drop_random"]


def test_numpy_forward_matches_torch():
    pytest.importorskip("torch")
    from fairplay.deep.export import forward, to_numpy
    from fairplay.deep.mil import VARIANTS, fit

    b, y, _ = toy_bags(n=200)
    cfg = {"d": 8, "channels": 4, "dropout": 0.1, "lr": 1e-2, "weight_decay": 0.0,
           "batch": 64, "max_epochs": 2, "patience": 2}
    m = fit(VARIANTS["mil"], b, y, np.ones(len(y)), np.arange(len(y)), seed=0, epochs=2, cfg=cfg)
    logit_t, att_t = m.score(b)
    logit_n, att_n = forward(to_numpy(m), b.games, b.mask)
    assert np.allclose(logit_t, logit_n, atol=1e-4)
    assert np.allclose(att_t, att_n, atol=1e-5)
