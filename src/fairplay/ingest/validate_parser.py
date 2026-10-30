"""Design §2 gate: our SQL parser must agree with python-chess on 1,000 random games.

python-chess replays every move on a real board, so agreement also proves the SAN sequence is
legal and complete. Gate: 100% agreement on moves; every clock mismatch explained.

uv run python -m fairplay.ingest.validate_parser 2025-06 --n 1000
"""

import argparse
import io
import json
from pathlib import Path

import chess.pgn
import duckdb

from fairplay.ingest.clean import install_parse_macros
from fairplay.ingest.hf_month import OUT_ROOT as SAMPLED_ROOT


def chess_parse(movetext: str) -> tuple[list[str], list[float | None]]:
    game = chess.pgn.read_game(io.StringIO(movetext))
    board = game.board()
    sans, clocks = [], []
    for node in game.mainline():
        sans.append(board.san(node.move))
        board.push(node.move)
        clocks.append(node.clock())
    if game.errors:
        raise ValueError(game.errors[0])
    return sans, clocks


def validate(month: str, n: int, seed: int = 7) -> dict:
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    install_parse_macros(con)
    src = (SAMPLED_ROOT / f"month={month}" / "*.parquet").as_posix()
    rows = con.execute(f"""
        SELECT movetext, parse_sans(movetext), parse_clocks(movetext)
        FROM read_parquet('{src}') USING SAMPLE {n} ROWS (reservoir, {seed})
    """).fetchall()
    move_mismatch, clock_mismatch, examples = 0, 0, []
    for movetext, sql_sans, sql_clocks in rows:
        ref_sans, ref_clocks = chess_parse(movetext)
        if sql_sans != ref_sans:
            move_mismatch += 1
            examples.append({"kind": "moves", "sql": sql_sans[:12], "chess": ref_sans[:12]})
        ref = [c for c in ref_clocks if c is not None]
        if len(ref) != len(sql_clocks) or any(
            abs(a - b) > 1e-3 for a, b in zip(ref, sql_clocks, strict=True)
        ):
            clock_mismatch += 1
            examples.append({"kind": "clocks", "sql": sql_clocks[:6], "chess": ref[:6]})
    return {
        "month": month, "games": len(rows), "move_mismatches": move_mismatch,
        "clock_mismatches": clock_mismatch, "examples": examples[:5],
        "passed": move_mismatch == 0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("month")
    p.add_argument("--n", type=int, default=1000)
    args = p.parse_args()
    report = validate(args.month, args.n)
    out = Path("data/processed") / f"parser_validation_{args.month}.json"  # data inventory item
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
