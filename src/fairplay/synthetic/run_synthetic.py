"""Synthetic benchmark commands.

  pilot       Phase 0a: tool checksums and 5 timed games
  critical    critical-position agreement at 100k vs 1M nodes (300 positions, gate 90%)
  timing      build the human think-time table from July honest accounts
  generate    --strategy control|A|B|C --accounts N --workers W
  score       --strategies control A B [--include-held-out]   (needs the frozen real models)

Synthetic data is evaluation-only: nothing here is ever used for training (design §6).
"""

import argparse
import dataclasses
import datetime as dt
import json
import time
from multiprocessing import Pool
from pathlib import Path

import chess
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fairplay.synthetic import engines
from fairplay.synthetic.engines import CRITICAL_GAP_WIN, win_percent
from fairplay.synthetic.generator import STRATEGIES, generate_account, play_game
from fairplay.synthetic.timing import ThinkSampler, build_table

OUT = Path("data/synthetic")
TIMING_TABLE = OUT / "think_table.parquet"
MODEL_TABLE = Path("data/processed/model_table_S0.parquet")
CUTOFF = dt.datetime(2025, 8, 10)
SEED_BASE = {"control": 1_000_000, "A": 2_000_000, "B": 3_000_000, "C": 4_000_000}


def manifest(sf: engines.Stockfish, maia: engines.Maia) -> dict:
    return {
        "move_model": "Maia-1 (CSSLab) in lc0, nodes=1, policy sampled with seeded RNG",
        "maia_weights_sha256": {r: engines.sha256(maia.weights(r))
                                for r in engines.MAIA_RATINGS if maia.weights(r).exists()},
        "lc0_sha256": engines.sha256(maia.lc0),
        "rating_mapping": "nearest of 1100..1900; accounts outside 1050-1950 not generated",
        "stockfish": sf.version, "stockfish_sha256": engines.sha256(sf.path),
        "stockfish_nodes": sf.nodes, "threads": 1,
        "clock": "clock_after = floor(clock_before - think + inc); human thinks from July honest "
                 "accounts by band x move bucket x clock bucket",
        "rating_update": "Glicko-2 after each game; opponents uniform within +-150",
        "seeds": SEED_BASE,
    }


def pilot() -> dict:
    maia, sf = engines.Maia(), engines.Stockfish(nodes=100_000)
    sampler = ThinkSampler(cells={(b, m, c): np.array([0.01, 0.02, 0.03])
                                  for b in range(1000, 2001, 200) for m in range(5)
                                  for c in range(4)}) if not TIMING_TABLE.exists() \
        else ThinkSampler(TIMING_TABLE)
    rng = np.random.default_rng(0)
    timings = []
    for i, strat in enumerate(["control", "A", "A", "B", "C"]):
        t0 = time.perf_counter()
        g = play_game(rng, maia, sf, sampler, 1500, 1520, i % 2 == 0, (180, 0), STRATEGIES[strat])
        timings.append({"strategy": strat, "seconds": round(time.perf_counter() - t0, 1),
                        "plies": g["n_plies"], "substituted": g["n_substituted"]})
    report = {"manifest": manifest(sf, maia), "games": timings}
    for e in (maia, sf):
        e.close()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pilot.json").write_text(json.dumps(report, indent=2))
    return report


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(mid - half), float(mid + half)


def critical_agreement(n_positions: int = 300, seed: int = 1) -> dict:
    """Design §0.12 gate: 'critical' must agree on >= 90% of positions at 100k vs 1M nodes.
    Positions: own moves > 10 from Maia-vs-Maia games at 1100/1500/1900, one per 3 plies so
    neighbouring positions are not near-duplicates. Every gap is saved for later analysis."""
    maia = engines.Maia()
    lo, hi = engines.Stockfish(nodes=100_000), engines.Stockfish(nodes=1_000_000)
    rng = np.random.default_rng(seed)
    positions, game = [], 0
    while len(positions) < n_positions:
        rating = (1100, 1500, 1900)[game % 3]
        board, game = chess.Board(), game + 1
        while not board.is_game_over() and board.ply() < 160 and len(positions) < n_positions:
            board.push(engines.sample_move(maia.policy(board, rating), rng))
            if board.ply() > 20 and board.ply() % 3 == 0 and not board.is_game_over():
                positions.append(board.fen())
    gaps = []
    for f in positions:
        (_, a1, a2), (_, b1, b2) = lo.top2(chess.Board(f)), hi.top2(chess.Board(f))
        gaps.append({"fen": f, "s1_100k": a1, "s2_100k": a2, "s1_1m": b1, "s2_1m": b2,
                     "gap_100k": float("inf") if a2 is None else win_percent(a1) - win_percent(a2),
                     "gap_1m": float("inf") if b2 is None else win_percent(b1) - win_percent(b2)})
    for e in (maia, lo, hi):
        e.close()
    a = np.array([g["gap_100k"] >= CRITICAL_GAP_WIN for g in gaps])
    b = np.array([g["gap_1m"] >= CRITICAL_GAP_WIN for g in gaps])
    k, n = int((a == b).sum()), len(gaps)
    report = {"seed": seed, "n_positions": n, "games": game, "agreement": k / n,
              "wilson_95": wilson(k, n),
              "critical_share_100k": float(a.mean()), "critical_share_1m": float(b.mean()),
              "only_100k": int((a & ~b).sum()), "only_1m": int((~a & b).sum()),
              "gate_passed": k / n >= 0.9}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"critical_agreement_seed{seed}.json").write_text(json.dumps(report, indent=2))
    (OUT / f"critical_gaps_seed{seed}.json").write_text(json.dumps(gaps))
    return report


def build_timing() -> Path:
    con = duckdb.connect()
    con.execute(f"CREATE TEMP TABLE honest AS SELECT DISTINCT account_id FROM "
                f"'{MODEL_TABLE.as_posix()}' WHERE split = 'train' AND label = 0")
    OUT.mkdir(parents=True, exist_ok=True)
    return build_table(con, Path("data/processed/player_games/month=2025-07.parquet"), "honest",
                       TIMING_TABLE)


def account_specs(strategy: str, n: int) -> tuple[list[dict], list]:
    """Ratings, ages and time controls drawn from real training-month accounts in Maia's range."""
    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT rating_at_cutoff, account_age_days FROM '{MODEL_TABLE.as_posix()}'
        WHERE split = 'train' AND label = 0 AND rating_at_cutoff BETWEEN 1050 AND 1949
    """).fetchall()
    tcs = con.execute("""
        SELECT tc_base, tc_inc, count(*) FROM 'data/processed/player_games/month=2025-07.parquet'
        GROUP BY ALL ORDER BY 3 DESC LIMIT 8""").fetchall()
    rng = np.random.default_rng(SEED_BASE[strategy])
    picks = rng.choice(len(rows), n, replace=False)
    specs = [{"account_id": f"syn-{strategy}-{i:04d}", "rating": rows[j][0],
              "age": int(rows[j][1]), "seed": SEED_BASE[strategy] + i}
             for i, j in enumerate(picks)]
    return specs, [((b, inc), c) for b, inc, c in tcs]


_worker: dict = {}


def _init_worker() -> None:
    _worker["maia"], _worker["sf"] = engines.Maia(), engines.Stockfish(nodes=100_000)
    _worker["sampler"] = ThinkSampler(TIMING_TABLE)


def _one(args) -> tuple[list[dict], dict]:
    spec, strategy, tcs = args
    return generate_account(spec["account_id"], STRATEGIES[strategy], spec["rating"], spec["age"],
                            tcs, CUTOFF, spec["seed"], _worker["maia"], _worker["sf"],
                            _worker["sampler"])


def generate(strategy: str, n: int, workers: int) -> dict:
    """Resumable: each finished account is saved at once under parts/<strategy>/ (games parquet +
    summary json, written atomically), and a re-run skips accounts already there. Every account
    has its own seed, so a resumed run produces exactly the accounts an uninterrupted one would."""
    specs, tcs = account_specs(strategy, n)
    parts = OUT / "parts" / strategy
    parts.mkdir(parents=True, exist_ok=True)
    todo = [sp for sp in specs if not (parts / f"{sp['account_id']}.json").exists()]
    t0 = time.perf_counter()
    # multiprocessing.Pool, not ProcessPoolExecutor: its exit terminates the workers. With the
    # executor, shutdown waited for workers that could not exit while their engine processes were
    # open, and the run hung after the last account (DECISIONS 2026-10-30).
    with Pool(workers, initializer=_init_worker) as pool:
        for done, (r, s) in enumerate(pool.imap(_one, [(sp, strategy, tcs) for sp in todo]), 1):
            pq.write_table(pa.Table.from_pylist(r), parts / f"{s['account_id']}.parquet")
            tmp = parts / f"{s['account_id']}.json.tmp"
            tmp.write_text(json.dumps(s))
            tmp.replace(parts / f"{s['account_id']}.json")  # json last: marks the account done
            if done % 10 == 0:
                print(f"{strategy}: {done}/{len(todo)} new accounts", flush=True)
    rows, summaries = [], []
    for sp in specs:
        rows += pq.read_table(parts / f"{sp['account_id']}.parquet").to_pylist()
        summaries.append(json.loads((parts / f"{sp['account_id']}.json").read_text()))
    d = OUT / "player_games"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), d / f"strategy={strategy}.parquet")
    (OUT / f"accounts_{strategy}.json").write_text(json.dumps(summaries, indent=2))
    shares = [s["achieved_share"] for s in summaries]
    return {"strategy": strategy, "accounts": len(summaries), "games": len(rows),
            "seconds": round(time.perf_counter() - t0),
            "achieved_share_mean": float(np.mean(shares))}


def score(strategies: list[str], include_held_out: bool) -> dict:
    """Synthetic recall at the frozen real-queue threshold + realism (score.py)."""
    import joblib

    from fairplay.casefiles.run_casefiles import chosen_model
    from fairplay.models.data import load_split
    from fairplay.models.run_tier1 import latest_run
    from fairplay.runs import new_run, save_metrics
    from fairplay.synthetic import score as sc

    if sc.HELD_OUT in strategies and not include_held_out:
        raise SystemExit("strategy C is held out: score it last, with --include-held-out")
    files = [OUT / "player_games" / f"strategy={s}.parquet" for s in strategies]
    ages = {a["account_id"]: a["account_age_days"] for s in strategies
            for a in json.loads((OUT / f"accounts_{s}.json").read_text(encoding="utf-8"))}
    model = chosen_model()
    # The queue cut needs scores and ratings only. Labels are blanked before anything runs, so
    # this reads no August outcome and needs no test claim.
    test = load_split(MODEL_TABLE, "test", ("rating_at_cutoff",))
    test = dataclasses.replace(test, y=np.zeros_like(test.y))
    train = load_split(MODEL_TABLE, "train", ("rating_at_cutoff",))
    run = new_run("synthetic_score", {"strategies": strategies,
                                      "include_held_out": include_held_out}, [MODEL_TABLE, *files])
    threshold = sc.frozen_threshold(model, test)
    feats = sc.synthetic_features(files, CUTOFF, Path("data/processed/peer_stats_clock30.parquet"))
    split = sc.to_split(feats, ages, CUTOFF)
    scores = np.asarray(model.score(split), dtype=float)
    # The rule baseline, frozen the same way, is the comparator: does the model react to the
    # simulated pattern at least as much as "rank by performance-rating gap" does?
    rule = joblib.load(latest_run("baselines") / "models.joblib")["rule"]
    rule_threshold = sc.frozen_threshold(rule, test)
    rule_scores = np.asarray(rule.score(split), dtype=float)
    report = {"threshold": threshold,
              "recall": sc.recall_table(split.account_id, scores,
                                        split.extra["rating_at_cutoff"], threshold["threshold"]),
              "auc_vs_control": sc.auc_vs_control(split.account_id, scores),
              "rule": {"threshold": rule_threshold,
                       "recall": sc.recall_table(split.account_id, rule_scores,
                                                 split.extra["rating_at_cutoff"],
                                                 rule_threshold["threshold"]),
                       "auc_vs_control": sc.auc_vs_control(split.account_id, rule_scores)},
              "realism": sc.realism(feats, train, (train.extra["rating_at_cutoff"] // 200) * 200,
                                    train.extra["rating_at_cutoff"])}
    save_metrics(run, report)
    return {"run": run.name, "threshold": threshold["threshold"],
            "recall": {s: {k: v for k, v in r.items() if k != "by_band"}
                       for s, r in report["recall"].items()},
            "auc_vs_control": report["auc_vs_control"],
            "rule_recall": {s: next(v for k, v in r.items() if k.endswith(("recall", "alarm")))
                            for s, r in report["rule"]["recall"].items()},
            "rule_auc_vs_control": report["rule"]["auc_vs_control"],
            "features_ks_above_limit": report["realism"]["features_ks_above_limit"]}


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pilot")
    sub.add_parser("timing")
    c = sub.add_parser("critical")
    c.add_argument("--seed", type=int, default=1)
    g = sub.add_parser("generate")
    g.add_argument("--strategy", choices=list(STRATEGIES), required=True)
    g.add_argument("--accounts", type=int, default=200)
    g.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("score")
    s.add_argument("--strategies", nargs="+", default=["control", "A", "B"])
    s.add_argument("--include-held-out", action="store_true")
    args = p.parse_args()
    if args.cmd == "pilot":
        print(json.dumps(pilot(), indent=2))
    elif args.cmd == "critical":
        print(json.dumps(critical_agreement(seed=args.seed), indent=2))
    elif args.cmd == "timing":
        print(build_timing())
    elif args.cmd == "score":
        print(json.dumps(score(args.strategies, args.include_held_out), indent=2, default=str))
    else:
        print(json.dumps(generate(args.strategy, args.accounts, args.workers), indent=2))


if __name__ == "__main__":
    main()
