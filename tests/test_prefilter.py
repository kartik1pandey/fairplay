import pytest

from fairplay.ingest.pgn_stream import iter_games
from fairplay.ingest.prefilter import (
    FilterStats,
    filter_games,
    in_sample,
    is_blitz,
    parse_time_control,
    sample_bucket,
)


def game(white: str, black: str, tc: str, moves: str = "1. e4 { [%clk 0:03:00] } 1-0") -> bytes:
    return (
        f'[Event "Rated Blitz game"]\n[White "{white}"]\n[Black "{black}"]\n'
        f'[TimeControl "{tc}"]\n\n{moves}\n\n'
    ).encode()


@pytest.mark.parametrize(
    "tc, expected",
    [(b"180+2", (180, 2)), (b"60+0", (60, 0)), (b"-", None), (b"", None), (b"abc+1", None)],
)
def test_parse_time_control(tc, expected):
    assert parse_time_control(tc) == expected


@pytest.mark.parametrize(
    "tc, expected",
    [
        (b"179+0", False),  # 179 s: bullet
        (b"180+0", True),  # 180 s: first blitz value
        (b"479+0", True),  # last blitz value
        (b"480+0", False),  # rapid
        (b"120+1", False),  # 120 + 40 = 160: bullet despite the increment
        (b"60+3", True),  # 60 + 120 = 180: blitz although the base is one minute
        (b"300+3", True),  # 300 + 120 = 420
        (b"300+5", False),  # 300 + 200 = 500: rapid
        (b"-", False),  # correspondence
    ],
)
def test_is_blitz_duration_rule(tc, expected):
    assert is_blitz(tc) is expected


def test_sampling_is_case_insensitive_and_stable():
    assert sample_bucket(b"MagnusFan") == sample_bucket(b"magnusfan")
    assert sample_bucket(b"magnusfan") == sample_bucket(b"magnusfan")


def test_samples_nest():
    names = [f"user{i}".encode() for i in range(5000)]
    one_pct = {n for n in names if in_sample(n, 100)}
    ten_pct = {n for n in names if in_sample(n, 1000)}
    assert one_pct <= ten_pct


def test_sample_rate_is_roughly_right():
    names = [f"user{i}".encode() for i in range(20000)]
    share = sum(in_sample(n, 1000) for n in names) / len(names)
    assert 0.09 < share < 0.11


def test_iter_games_splits_and_keeps_exact_text():
    text = game("a", "b", "180+0") + game("c", "d", "60+0")
    games = list(iter_games(text.splitlines(keepends=True)))
    assert len(games) == 2
    assert games[0][0][b"White"] == b"a"
    assert games[1][0][b"TimeControl"] == b"60+0"
    assert b"".join(b"".join(raw) for _, raw in games) == text


def test_filter_keeps_blitz_game_when_either_player_sampled():
    sampled = next(f"u{i}" for i in range(1000) if in_sample(f"u{i}".encode(), 1000))
    unsampled = [f"u{i}" for i in range(1000) if not in_sample(f"u{i}".encode(), 1000)]
    text = (
        game(sampled, unsampled[0], "180+0")  # kept: white sampled
        + game(unsampled[1], sampled, "300+0")  # kept: black sampled
        + game(sampled, unsampled[2], "60+0")  # dropped: bullet
        + game(unsampled[3], unsampled[4], "180+0")  # dropped: nobody sampled
    )
    out: list[bytes] = []
    stats = filter_games(text.splitlines(keepends=True), 1000, out.append)
    assert stats == FilterStats(games_seen=4, dropped_not_blitz=1, dropped_not_sampled=1, kept=2)
    assert len(out) == 2
