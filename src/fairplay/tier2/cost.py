"""Tier 2 engine budget (design §0.13, §7): measure first, then decide whether tier 2 can exist.

Tier 2 analyses own moves 11-60 of an account's last 10 games (at most 500 positions) with the
pinned Stockfish, single thread, multipv 2 (the features need the gap to the second-best move).
Budget: at most 4 CPU-hours per weekly cutoff. Per cutoff the engine must cover
  candidates  = ceil(q x eligible_w), q in {1%, 3%, 10%}
  audit       = 2% of the non-candidates (design §7 "missed cases")
The node limit is one of the two benchmarked limits (100k, 1M). If no (q, nodes) pair fits,
design §7 removes tier 2 and the queue stays tier 1.

  positions   positions per account, measured at every July cutoff (no engine needed)
  benchmark   seconds per position on real July positions; records machine load
  decide      combines both into the cost table and the keep/remove decision

uv run python -m fairplay.tier2.cost positions|benchmark|decide
"""

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import chess
import numpy as np

from fairplay import duck

OUT = Path("data/tier2")
TABLE = Path("data/processed/model_table_S0.parquet")
PG = Path("data/processed/player_games")
GM = Path("data/processed/game_moves")
BUDGET_CPU_HOURS = 4.0
QS = (0.01, 0.03, 0.10)
AUDIT = 0.02
NODES = (100_000, 1_000_000)
LAST_GAMES, FIRST_MOVE, LAST_MOVE, MAX_POSITIONS = 10, 11, 60, 500


def own_moves(n_plies: int, white: bool) -> int:
    return (n_plies + (1 if white else 0)) // 2


def positions_in_game(n_plies: int, white: bool) -> int:
    """Own moves numbered 11..60 that were actually played."""
    return max(0, min(own_moves(n_plies, white), LAST_MOVE) - FIRST_MOVE + 1)


def positions() -> dict:
    """Per eligible account-cutoff in July: positions in its last 10 games before the cutoff
    (same 30-minute start rule as the features), capped at 500."""
    con = duck.connect()
    months = ", ".join(f"'{(PG / f'month={m}.parquet').as_posix()}'"
                       for m in ("2025-06", "2025-07"))
    rows = con.execute(f"""
        WITH rows AS (SELECT account_id, cutoff FROM '{TABLE.as_posix()}' WHERE split = 'train'),
        g AS (
            SELECT r.cutoff, r.account_id,
                   greatest(0, least((p.n_plies + CASE WHEN p.color = 'w' THEN 1 ELSE 0 END) // 2,
                                     {LAST_MOVE}) - {FIRST_MOVE} + 1) AS pos,
                   row_number() OVER (PARTITION BY r.cutoff, r.account_id
                                      ORDER BY p.start_ts DESC, p.game_key) AS rk
            FROM rows r JOIN read_parquet([{months}]) p USING (account_id)
            WHERE p.start_ts < r.cutoff - INTERVAL 30 MINUTE)
        SELECT cutoff, count(*) AS accounts, avg(least(s, {MAX_POSITIONS})) AS mean_pos,
               quantile_cont(least(s, {MAX_POSITIONS}), 0.5) AS median_pos
        FROM (SELECT cutoff, account_id, sum(pos) AS s FROM g WHERE rk <= {LAST_GAMES}
              GROUP BY ALL) GROUP BY cutoff ORDER BY cutoff""").fetchall()
    out = {"per_cutoff": [{"cutoff": str(c)[:10], "eligible": int(n), "mean_positions": float(m),
                           "median_positions": float(md)} for c, n, m, md in rows]}
    out["mean_positions"] = float(np.mean([r["mean_positions"] for r in out["per_cutoff"]]))
    out["mean_eligible"] = float(np.mean([r["eligible"] for r in out["per_cutoff"]]))
    return save("positions", out)


def sample_positions(n: int, seed: int) -> list[str]:
    """Real July positions: own moves 11-60, one random position per random game."""
    con = duck.connect()
    games = con.execute(f"""
        SELECT sans FROM '{(GM / 'month=2025-07.parquet').as_posix()}'
        WHERE len(sans) >= 2 * {FIRST_MOVE + 1}
        ORDER BY hash(game_key, '{seed}') LIMIT {n}""").fetchall()
    rng = np.random.default_rng(seed)
    fens = []
    for (sans,) in games:
        board = chess.Board()
        stop = int(rng.integers(2 * (FIRST_MOVE - 1), min(len(sans), 2 * LAST_MOVE)))
        for san in sans[:stop]:
            board.push_san(san)
        if not board.is_game_over():
            fens.append(board.fen())
    return fens


def busy_engines() -> int:
    """Other engine processes running during the benchmark (results are slower under load)."""
    try:
        out = subprocess.run(["tasklist"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return -1
    return sum(line.lower().startswith(("stockfish", "lc0")) for line in out.splitlines())


def benchmark(n: int = 200, n_slow: int = 40, seed: int = 7) -> dict:
    from fairplay.synthetic.engines import Stockfish

    fens = sample_positions(n, seed)
    out = {"cpu_count": os.cpu_count(), "other_engine_processes": busy_engines(), "nodes": {}}
    for nodes, m in zip(NODES, (len(fens), n_slow), strict=True):
        sf = Stockfish(nodes=nodes)
        sf.new_game()
        out["stockfish"] = sf.version
        times = []
        try:
            for f in fens[:m]:
                t0 = time.perf_counter()
                sf.top2(chess.Board(f))
                times.append(time.perf_counter() - t0)
        finally:
            sf.close()
        t = np.array(times)
        out["nodes"][str(nodes)] = {"positions": len(t), "sec_per_position": float(t.mean()),
                                    "p95_sec": float(np.quantile(t, 0.95)),
                                    "positions_per_sec": float(1 / t.mean())}
    return save("benchmark", out)


def cost_table(eligible: float, mean_positions: float, sec_per_position: dict[str, float]) -> list:
    rows = []
    for nodes, sec in sec_per_position.items():
        for q in QS:
            cand = math.ceil(q * eligible)
            audit = math.ceil(AUDIT * (eligible - cand))
            hours = (cand + audit) * mean_positions * sec / 3600
            rows.append({"nodes": int(nodes), "q": q, "candidates": cand, "audit": audit,
                         "cpu_hours_per_cutoff": round(hours, 1),
                         "cpu_hours_candidates_only": round(cand * mean_positions * sec / 3600, 1),
                         "fits_budget": hours <= BUDGET_CPU_HOURS})
    return rows


def decide() -> dict:
    pos = json.loads((OUT / "positions.json").read_text())
    bench = json.loads((OUT / "benchmark.json").read_text())
    sec = {k: v["sec_per_position"] for k, v in bench["nodes"].items()}
    rows = cost_table(pos["mean_eligible"], pos["mean_positions"], sec)
    fits = [r for r in rows if r["fits_budget"]]
    cheapest = min(rows, key=lambda r: r["cpu_hours_per_cutoff"])
    needed = (BUDGET_CPU_HOURS * 3600 /
              ((cheapest["candidates"] + cheapest["audit"]) * pos["mean_positions"]))
    report = {"budget_cpu_hours_per_cutoff": BUDGET_CPU_HOURS, "table": rows,
              "feasible": fits,
              "decision": ("keep: tune q among the feasible rows on July" if fits else
                           "remove: no benchmarked (q, nodes) fits the engine budget (design §7);"
                           " the queue stays tier 1"),
              "cheapest_row": cheapest,
              "sec_per_position_needed_for_cheapest": needed,
              "speedup_needed_for_cheapest": cheapest["cpu_hours_per_cutoff"] / BUDGET_CPU_HOURS,
              "benchmark_load": bench["other_engine_processes"],
              "provisional": bench["other_engine_processes"] != 0}
    return save("decision", report)


def save(name: str, obj: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{name}.json.tmp"
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(OUT / f"{name}.json")
    return obj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["positions", "benchmark", "decide"])
    args = p.parse_args()
    print(json.dumps({"positions": positions, "benchmark": benchmark,
                      "decide": decide}[args.cmd](), indent=2))


if __name__ == "__main__":
    main()
