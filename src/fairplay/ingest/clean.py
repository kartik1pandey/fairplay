"""Sampled games -> cleaned, parsed, pseudonymised Parquet (design §2 exclusions, §9 privacy).

Input:  private/games_sampled/month=M/*.parquet   (usernames, Lichess game ids)
Output: data/processed/player_games/month=M.parquet  one row per (sampled account, game)
        data/processed/game_moves/month=M.parquet    one row per game: SAN list + clock list
        data/processed/clean_manifest_M.json         every dropped game counted by reason

Pseudonymisation:
- account_id: random UUID from the private map (labels/store.py).
- game_key: sha256(secret || game_id). A Lichess game id is a re-identification key
  (lichess.org/<id> shows both usernames); without the secret in private/ it cannot be recomputed.
- Opponent identity is not kept at all, only the opponent's rating.

Parsing (all in DuckDB/RE2):
- Clocks come from `[%clk h:mm:ss]`, extracted before comments are removed.
- SAN moves come from the movetext with every `{...}` comment removed, so `%eval` never enters.
  Lichess's analysis glyphs (`Bb4?`) are not part of the SAN pattern and are dropped: tier 1 uses
  no engine information, and those glyphs are engine judgements.
"""

import argparse
import json
import secrets
import shutil
from collections import Counter
from pathlib import Path

import duckdb
import pyarrow as pa

from fairplay.ingest.hf_month import OUT_ROOT as SAMPLED_ROOT
from fairplay.ingest.hf_month import install_macros
from fairplay.ingest.prefilter import MIN_PLIES
from fairplay.labels import store

PROCESSED = Path("data/processed")
SECRET_PATH = Path("private/game_key_secret.txt")


SAN_RE = r"(?:O-O-O|O-O|[NBRQK][a-h]?[1-8]?x?[a-h][1-8]|[a-h](?:x[a-h])?[1-8](?:=[NBRQ])?)[+#]?"
CLK_RE = r"%clk (\d+):(\d\d):(\d\d(?:\.\d+)?)"


def game_key_secret() -> str:
    if not SECRET_PATH.exists():
        SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
        SECRET_PATH.write_text(secrets.token_hex(32))
    return SECRET_PATH.read_text().strip()


def install_parse_macros(con: duckdb.DuckDBPyConnection) -> None:
    install_macros(con)
    con.execute(f"""
        CREATE OR REPLACE MACRO parse_sans(mt) AS
            regexp_extract_all(regexp_replace(mt, '\\{{[^}}]*\\}}', '', 'g'), '{SAN_RE}');
        CREATE OR REPLACE MACRO parse_clocks(mt) AS
            list_transform(
                regexp_extract_all(mt, '%clk \\d+:\\d\\d:\\d\\d(?:\\.\\d+)?'),
                lambda x: (split_part(x[6:], ':', 1)::INTEGER * 3600
                      + split_part(x[6:], ':', 2)::INTEGER * 60
                      + split_part(x[6:], ':', 3)::DOUBLE)::FLOAT);
    """)


def sampled_usernames(con: duckdb.DuckDBPyConnection, months: list[str], rate_bp: int) -> list[str]:
    globs = ", ".join(f"'{(SAMPLED_ROOT / f'month={m}' / '*.parquet').as_posix()}'" for m in months)
    return [r[0] for r in con.execute(f"""
        SELECT DISTINCT lower(name) FROM (
            SELECT White AS name FROM read_parquet([{globs}])
            UNION ALL SELECT Black FROM read_parquet([{globs}])
        ) WHERE sample_bucket(name) < {rate_bp}
    """).fetchall()]


def register_account_map(
    con: duckdb.DuckDBPyConnection, months: list[str], rate_bp: int, conn=None
) -> int:
    """Add every sampled username to the private UUID map; expose the map to DuckDB."""
    conn = conn or store.connect()
    new = store.add_accounts(conn, sampled_usernames(con, months, rate_bp))
    rows = conn.execute("SELECT username, account_id FROM accounts").fetchall()
    con.register("account_map", pa.table({"username": [r[0] for r in rows],
                                           "account_id": [r[1] for r in rows]}))
    return new


def find_duplicates(con: duckdb.DuckDBPyConnection, src_files: list[Path]) -> None:
    """Headers-only pre-pass over the whole month: for each game id seen more than once, the copy
    to keep = earliest (date, time, file, row). Cheap (no move text), so chunks can be cleaned
    one at a time and still agree on duplicates that span two input files."""
    files = ", ".join(f"'{f.as_posix()}'" for f in src_files)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE dup_first AS
        SELECT game_id, min({{'d': UTCDate, 't': UTCTime, 'f': parse_filename(filename),
                              'r': file_row_number}}) AS first
        FROM read_parquet([{files}], filename = true, file_row_number = true)
        GROUP BY game_id HAVING count(*) > 1
    """)


def clean_chunk(con: duckdb.DuckDBPyConnection, src: Path, month: str, rate_bp: int, secret: str,
                out_dir: Path, part: int) -> tuple[dict, tuple[int, int]]:
    """Clean one input file (~125k games): memory stays bounded by the chunk, not the month."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE parsed AS
        SELECT
            sha256('{secret}' || s.game_id)[:16] AS game_key,
            CASE WHEN d.game_id IS NOT NULL
                  AND d.first <> {{'d': s.UTCDate, 't': s.UTCTime, 'f': parse_filename(s.filename),
                                  'r': s.file_row_number}} THEN 2 ELSE 1 END AS dup_rank,
            White, Black, WhiteTitle, BlackTitle, WhiteElo, BlackElo,
            WhiteRatingDiff, BlackRatingDiff, Result, Termination, Event,
            UTCDate + UTCTime::TIME AS start_ts,
            split_part(TimeControl, '+', 1)::INTEGER AS tc_base,
            split_part(TimeControl, '+', 2)::INTEGER AS tc_inc,
            parse_sans(movetext) AS sans,
            parse_clocks(movetext) AS clocks
        FROM read_parquet('{src.as_posix()}', filename = true, file_row_number = true) s
        LEFT JOIN dup_first d ON d.game_id = s.game_id
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE judged AS
        SELECT *, CASE
            WHEN dup_rank > 1 THEN 'duplicate_game_id'
            WHEN WhiteTitle = 'BOT' OR BlackTitle = 'BOT' THEN 'bot_game'
            WHEN Termination = 'Abandoned' THEN 'abandoned'
            WHEN len(sans) < {MIN_PLIES} THEN 'too_short'
            WHEN WhiteElo IS NULL OR BlackElo IS NULL THEN 'missing_rating'
            ELSE NULL END AS drop_reason
        FROM parsed
    """)
    drops = dict(con.execute(
        "SELECT coalesce(drop_reason, 'kept'), count(*) FROM judged GROUP BY 1"
    ).fetchall())
    flags = con.execute("""
        SELECT count(*) FILTER (WHERE Termination = 'Rules infraction'),
               count(*) FILTER (WHERE len(clocks) != len(sans))
        FROM judged WHERE drop_reason IS NULL
    """).fetchone()
    side = """
        SELECT m.account_id, j.game_key, '{month}' AS month, '{c}' AS color, j.start_ts,
               j.tc_base, j.tc_inc,
               j.{own}Elo AS own_elo, j.{opp}Elo AS opp_elo, j.{own}RatingDiff AS own_rating_diff,
               CASE j.Result WHEN '{win}' THEN 1.0 WHEN '1/2-1/2' THEN 0.5 ELSE 0.0 END AS score,
               CASE j.Termination WHEN 'Normal' THEN 'normal'
                    WHEN 'Time forfeit' THEN 'time_forfeit'
                    WHEN 'Rules infraction' THEN 'rules_infraction'
                    ELSE 'other' END AS termination,
               ends_with(j.sans[-1], '#') AS ended_in_mate,
               CASE WHEN j.Event ILIKE '%swiss%' THEN 'swiss'
                    WHEN j.Event ILIKE '%tournament%' THEN 'arena' ELSE 'game' END AS event_kind,
               len(j.sans) AS n_plies,
               list_slice(j.clocks, {first}, -1, 2) AS own_clocks,
               len(j.clocks) = len(j.sans) AS clock_valid
        FROM judged j JOIN account_map m ON m.username = lower(j.{own})
        WHERE j.drop_reason IS NULL AND sample_bucket(j.{own}) < {rate_bp}
    """
    white = side.format(month=month, c="w", own="White", opp="Black", win="1-0", first=1,
                        rate_bp=rate_bp)
    black = side.format(month=month, c="b", own="Black", opp="White", win="0-1", first=2,
                        rate_bp=rate_bp)
    con.execute(f"COPY ({white} UNION ALL {black}) TO "
                f"'{(out_dir / f'pg-{part:03d}.parquet').as_posix()}' (FORMAT parquet)")
    con.execute(f"""
        COPY (SELECT game_key, '{month}' AS month, sans, clocks FROM judged
              WHERE drop_reason IS NULL)
        TO '{(out_dir / f'gm-{part:03d}.parquet').as_posix()}' (FORMAT parquet)
    """)
    return drops, flags


def clean_month(con: duckdb.DuckDBPyConnection, month: str, rate_bp: int, secret: str,
                processed: Path | None = None) -> dict:
    processed = processed or PROCESSED  # read at call time: tests patch PROCESSED
    src_files = sorted((SAMPLED_ROOT / f"month={month}").glob("*.parquet"))
    find_duplicates(con, src_files)
    chunks = processed / "_chunks" / month
    chunks.mkdir(parents=True, exist_ok=True)  # kept across restarts: finished chunks resume
    drops: Counter = Counter()
    flags = [0, 0]
    for part, src in enumerate(src_files):
        done = chunks / f"done-{part:03d}.json"  # written after both chunk files are complete
        if done.exists():
            d, f = json.loads(done.read_text())
        else:
            d, f = clean_chunk(con, src, month, rate_bp, secret, chunks, part)
            done.write_text(json.dumps([d, list(f)]))
        drops.update(d)
        flags = [flags[0] + f[0], flags[1] + f[1]]
        print(f"{month} chunk {part + 1}/{len(src_files)}", flush=True)

    (processed / "player_games").mkdir(parents=True, exist_ok=True)
    (processed / "game_moves").mkdir(parents=True, exist_ok=True)
    pg_path = (processed / "player_games" / f"month={month}.parquet").as_posix()
    gm_path = (processed / "game_moves" / f"month={month}.parquet").as_posix()
    # streaming merge, no global sort: nothing downstream depends on row order
    for prefix, out in (("pg", pg_path), ("gm", gm_path)):
        con.execute(f"COPY (SELECT * FROM read_parquet('{(chunks / prefix).as_posix()}-*.parquet'))"
                    f" TO '{out}.tmp' (FORMAT parquet, COMPRESSION zstd)")
        Path(f"{out}.tmp").replace(out)
    shutil.rmtree(chunks)
    (player_rows, accounts) = con.execute(
        f"SELECT count(*), count(DISTINCT account_id) FROM '{pg_path}'"
    ).fetchone()
    manifest = {
        "month": month, "rate_bp": rate_bp, "games_in": sum(drops.values()),
        "drops": dict(sorted(drops.items())), "kept_rules_infraction": flags[0],
        "kept_clock_invalid": flags[1], "player_game_rows": player_rows, "accounts": accounts,
        "input_files": len(src_files),
    }
    (processed / f"clean_manifest_{month}.json").write_text(json.dumps(manifest, indent=2),
                                                           encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("months", nargs="+")
    p.add_argument("--rate-bp", type=int, default=1000)
    args = p.parse_args()
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false; SET memory_limit = '2GB'; SET threads = 2; "
                "SET preserve_insertion_order = false; SET temp_directory = 'private/tmp/duckdb'")
    install_parse_macros(con)
    print("new accounts in map:", register_account_map(con, args.months, args.rate_bp))
    secret = game_key_secret()
    for month in args.months:
        done = PROCESSED / f"clean_manifest_{month}.json"  # written last, after the atomic merge
        if done.exists():
            print(month, "already cleaned, skipped", flush=True)
            continue
        print(json.dumps(clean_month(con, month, args.rate_bp, secret), indent=2), flush=True)


if __name__ == "__main__":
    main()
