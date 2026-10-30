"""Weekly cutoffs and eligible account-week rows (design §4, §0 items 4 and 19).

A row (account, cutoff) is eligible if the account has >= 10 kept blitz games that started in
[cutoff - 30 days, cutoff - 30 minutes). The 30-minute margin covers the longest blitz game
(the export gives only the start time, design §19).

Cutoffs are Sundays 00:00 UTC inside each split month. Experiment R (2025, DECISIONS.md):
June = lookback only, July = train, August = validation, September = test.

uv run python -m fairplay.dataset.eligibility
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import duckdb
import pyarrow as pa

from fairplay.labels import store

PROCESSED = Path("data/processed")
SPLITS = {"train": "2025-07", "test": "2025-08"}  # Sept dropped: S0 came early (DECISIONS)
LOOKBACK_MONTHS = ["2025-06"]
MIN_GAMES = 10
WINDOW = dt.timedelta(days=30)
END_MARGIN = dt.timedelta(minutes=30)


def weekly_cutoffs(month: str) -> list[dt.datetime]:
    """Every Sunday 00:00 UTC within the month (naive datetimes are UTC throughout)."""
    year, mm = map(int, month.split("-"))
    day = dt.date(year, mm, 1)
    out = []
    while day.month == mm:
        if day.weekday() == 6:
            out.append(dt.datetime(day.year, day.month, day.day))
        day += dt.timedelta(days=1)
    return out


def cutoff_table() -> pa.Table:
    rows = [(split, c) for split, month in SPLITS.items() for c in weekly_cutoffs(month)]
    return pa.table({"split": [r[0] for r in rows], "cutoff": [r[1] for r in rows]})


def eligible_rows(con: duckdb.DuckDBPyConnection, processed: Path = PROCESSED) -> str:
    """Creates temp table `eligible(account_id, split, cutoff, n_games)`; returns its name."""
    months = LOOKBACK_MONTHS + list(SPLITS.values())
    files = [processed / "player_games" / f"month={m}.parquet" for m in months]
    file_list = ", ".join(f"'{f.as_posix()}'" for f in files if f.exists())
    con.register("cutoffs", cutoff_table())
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE eligible AS
        SELECT g.account_id, c.split, c.cutoff, count(*) AS n_games
        FROM read_parquet([{file_list}]) g
        JOIN cutoffs c
          ON g.start_ts >= c.cutoff - INTERVAL {WINDOW.days} DAY
         AND g.start_ts <  c.cutoff - INTERVAL {int(END_MARGIN.total_seconds() // 60)} MINUTE
        GROUP BY ALL
        HAVING count(*) >= {MIN_GAMES}
    """)
    return "eligible"


def main() -> None:
    argparse.ArgumentParser().parse_args()
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    eligible_rows(con)
    out = PROCESSED / "eligible_rows.parquet"
    con.execute(f"COPY (SELECT * FROM eligible ORDER BY split, cutoff, account_id) "
                f"TO '{out.as_posix()}' (FORMAT parquet)")
    per_cutoff = con.execute(
        "SELECT split, cutoff::DATE::VARCHAR, count(*) FROM eligible GROUP BY ALL ORDER BY 2"
    ).fetchall()
    per_split = dict(con.execute(
        "SELECT split, count(DISTINCT account_id) FROM eligible GROUP BY 1"
    ).fetchall())
    ids = [r[0] for r in con.execute("SELECT DISTINCT account_id FROM eligible").fetchall()]

    # Usernames are needed only for the label API; they are written to private/ only.
    conn = store.connect()
    names = [r[0] for r in conn.execute(
        "SELECT username FROM accounts WHERE account_id IN (SELECT value FROM json_each(?))",
        (json.dumps(ids),),
    ).fetchall()]
    Path("private/accounts_eligible.txt").write_text("\n".join(sorted(names)))
    summary = {
        "eligible_rows": sum(n for *_, n in per_cutoff),
        "per_cutoff": per_cutoff,
        "unique_accounts_per_split": per_split,
        "unique_accounts_any_split": len(ids),
    }
    (PROCESSED / "eligibility_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
