import datetime as dt
import json

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fairplay.ingest import clean
from fairplay.ingest.prefilter import in_sample
from fairplay.labels import store


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    clean.install_parse_macros(c)
    return c


def sans(con, mt):
    return con.execute("SELECT parse_sans(?)", [mt]).fetchone()[0]


def test_parse_sans_handles_castling_promotion_disambiguation(con):
    mt = "1. O-O-O { [%clk 0:03:00] } 1... O-O 2. exd8=Q+ 2... Nbd7 3. R1e2# 3... Qh4xe1 1-0"
    assert sans(con, mt) == ["O-O-O", "O-O", "exd8=Q+", "Nbd7", "R1e2#", "Qh4xe1"]


def test_parse_sans_ignores_eval_comments_and_analysis_glyphs(con):
    mt = "1. e4 { [%eval 0.2] [%clk 0:03:00] } 1... f6? { Nf3 was best. [%eval 1.1] } 2. Qh5+?! 1-0"
    assert sans(con, mt) == ["e4", "f6", "Qh5+"]  # no 'Nf3' from the comment, no '?' glyphs


def test_parse_clocks_to_seconds(con):
    mt = "1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 1:02:03] } 2. Nf3 { [%clk 0:00:05.5] }"
    assert con.execute("SELECT parse_clocks(?)", [mt]).fetchone()[0] == [180.0, 3723.0, 5.5]


def movetext(n_plies, clock_every=True):
    parts = []
    for i in range(n_plies):
        num = f"{i // 2 + 1}." if i % 2 == 0 else f"{i // 2 + 1}..."
        mv = ("Nf3", "Nf6", "Ng1", "Ng8")[i % 4]
        clk = f" {{ [%clk 0:02:{59 - i // 2:02d}] }}" if clock_every or i < 5 else ""
        parts.append(f"{num} {mv}{clk}")
    return " ".join(parts) + " 1-0"


def test_clean_month_end_to_end(tmp_path, monkeypatch):
    names = [f"p{i}" for i in range(400)]
    s = next(n for n in names if in_sample(n.encode(), 1000))  # a sampled player
    u = [n for n in names if not in_sample(n.encode(), 1000)]  # unsampled opponents
    rows = [  # (game_id, white, black, white_title, termination, plies, clocks_ok)
        ("g1", s, u[0], None, "Normal", 40, True),  # kept
        ("g1", s, u[0], None, "Normal", 40, True),  # duplicate
        ("g2", u[1], s, "BOT", "Normal", 40, True),  # bot game
        ("g3", s, u[2], None, "Abandoned", 40, True),  # abandoned
        ("g4", s, u[3], None, "Normal", 19, True),  # too short
        ("g5", u[4], s, None, "Time forfeit", 30, False),  # kept, clocks invalid
    ]
    month = "2025-07"
    src_dir = tmp_path / "sampled" / f"month={month}"
    src_dir.mkdir(parents=True)
    def write(part, name):
        n = len(part)
        pq.write_table(pa.table({
            "game_id": [r[0] for r in part], "Event": ["Rated Blitz game"] * n,
            "White": [r[1] for r in part], "Black": [r[2] for r in part],
            "WhiteTitle": [r[3] for r in part], "BlackTitle": pa.array([None] * n, pa.string()),
            "Result": ["1-0"] * n, "WhiteElo": [1500] * n, "BlackElo": [1600] * n,
            "WhiteRatingDiff": [8] * n, "BlackRatingDiff": [-8] * n,
            "UTCDate": [dt.date(2025, 7, 1)] * n, "UTCTime": [dt.time(12, 0)] * n,
            "Termination": [r[4] for r in part], "TimeControl": ["180+0"] * n,
            "movetext": [movetext(r[5], r[6]) for r in part],
        }), src_dir / name)

    write([rows[0], *rows[2:]], "part-a.parquet")  # files are cleaned one at a time;
    write([rows[1]], "part-b.parquet")  # the duplicate sits in ANOTHER file
    monkeypatch.setattr(clean, "SAMPLED_ROOT", tmp_path / "sampled")
    monkeypatch.setattr(clean, "PROCESSED", tmp_path / "processed")

    con = duckdb.connect()
    clean.install_parse_macros(con)
    clean.register_account_map(con, [month], 1000, conn=store.connect(tmp_path / "l.sqlite"))
    m = clean.clean_month(con, month, 1000, secret="test-secret")

    assert m["drops"] == {"kept": 2, "duplicate_game_id": 1, "bot_game": 1, "abandoned": 1,
                          "too_short": 1}
    assert m["kept_clock_invalid"] == 1
    pg = pq.read_table(tmp_path / "processed/player_games" / f"month={month}.parquet").to_pylist()
    assert len(pg) == 2  # only the sampled side of each kept game
    w = next(r for r in pg if r["color"] == "w")
    assert w["score"] == 1.0 and w["own_elo"] == 1500 and w["opp_elo"] == 1600
    assert w["own_clocks"][:3] == [179.0, 178.0, 177.0] and len(w["own_clocks"]) == 20
    b = next(r for r in pg if r["color"] == "b")
    assert b["score"] == 0.0 and b["clock_valid"] is False and b["termination"] == "time_forfeit"
    # no usernames and no Lichess game ids leave private/
    dumped = json.dumps(pg, default=str)
    assert s not in dumped and "g1" not in dumped
