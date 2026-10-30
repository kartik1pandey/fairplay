import datetime as dt

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from fairplay.dataset import eligibility as el


def test_weekly_cutoffs_are_sundays_in_month():
    july = el.weekly_cutoffs("2025-07")
    assert [c.day for c in july] == [6, 13, 20, 27]
    assert all(c.weekday() == 6 and c.hour == 0 for c in july)
    assert [c.day for c in el.weekly_cutoffs("2025-08")] == [3, 10, 17, 24, 31]


def write_games(tmp_path, month, account, starts):
    d = tmp_path / "player_games"
    d.mkdir(exist_ok=True)
    pq.write_table(pa.table({"account_id": [account] * len(starts), "start_ts": starts}),
                   d / f"month={month}.parquet")


def eligible(tmp_path):
    con = duckdb.connect()
    el.eligible_rows(con, processed=tmp_path)
    return con.execute("SELECT account_id, cutoff, n_games FROM eligible ORDER BY 1, 2").fetchall()


def test_game_at_t_minus_29_minutes_is_excluded(tmp_path):
    cutoff = dt.datetime(2025, 7, 13)
    nine_days_before = [cutoff - dt.timedelta(days=d) for d in range(1, 10)]  # 9 games
    write_games(tmp_path, "2025-07", "a", nine_days_before + [cutoff - dt.timedelta(minutes=29)])
    assert all(c != cutoff for _, c, _ in eligible(tmp_path))  # only 9 games count


def test_game_at_t_minus_31_minutes_is_included(tmp_path):
    cutoff = dt.datetime(2025, 7, 13)
    nine_days_before = [cutoff - dt.timedelta(days=d) for d in range(1, 10)]
    write_games(tmp_path, "2025-07", "a", nine_days_before + [cutoff - dt.timedelta(minutes=31)])
    assert ("a", cutoff, 10) in eligible(tmp_path)


def test_window_is_30_days_and_uses_lookback_month(tmp_path):
    cutoff = dt.datetime(2025, 7, 6)  # first July cutoff: window reaches back into June
    june = [dt.datetime(2025, 6, 10) + dt.timedelta(hours=h) for h in range(10)]
    write_games(tmp_path, "2025-06", "a", june)
    write_games(tmp_path, "2025-07", "b", [dt.datetime(2025, 7, 1)])
    assert ("a", cutoff, 10) in eligible(tmp_path)
    old = [dt.datetime(2025, 6, 5) + dt.timedelta(hours=h) for h in range(10)]  # > 30 days
    write_games(tmp_path, "2025-06", "a", old)
    assert all(c != cutoff for _, c, _ in eligible(tmp_path))
