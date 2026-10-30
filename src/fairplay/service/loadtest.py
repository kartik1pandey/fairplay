"""Replay load test (design §11 scale table): one week of games through POST /v1/games, timed.

Measures per-request latency (validation + DB write + recomputing both accounts' features with
the shared SQL) in-process, so it measures the service, not the network. Results state the
machine, the database and the data source; anything larger is an estimate, never a measurement.

  uv run python -m fairplay.service.loadtest --source synthetic --games 3000 --accounts 300
  uv run python -m fairplay.service.loadtest --source real --week 2025-08-03   (after cleaning)

Real replay: our player rows hold one side per sampled account; the opponent becomes a
placeholder UUID derived from the game key (no identity), and move text is a placeholder of
the right length (tier-1 features never read SAN).
"""

import argparse
import datetime as dt
import json
import os
import platform
import tempfile
import threading
import time
import uuid
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi.testclient import TestClient

from fairplay.service.app import PEERS, create_app
from fairplay.service.db import Database

OPPONENT_NS = uuid.UUID("0b5e7a52-6c1e-4e8f-9d3c-2a4f7e1b9c08")


def neutral_peers(path: Path) -> Path:
    """Stand-in peer stats when the real frozen file does not exist yet (latency is unaffected)."""
    pq.write_table(pa.table({"rating_band": pa.array([None], pa.int64()),
                             "tc_group": pa.array([None], pa.string()), "med": [0.3],
                             "mad": [0.1], "level": ["global"]}), path)
    return path


def synthetic_week(n_games: int, n_accounts: int, start: dt.datetime, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    accounts = [str(uuid.UUID(int=int(i) + 1)) for i in range(n_accounts)]
    games = []
    for i in range(n_games):
        w, b = rng.choice(n_accounts, 2, replace=False)
        plies = int(rng.integers(20, 120))
        base, inc = [(180, 0), (180, 2), (300, 0), (300, 3)][int(rng.integers(0, 4))]
        clk, clocks = {0: base, 1: base}, []
        for k in range(plies):
            if k >= 2:
                clk[k % 2] = max(0, clk[k % 2] - int(rng.integers(0, 8)) + inc)
            clocks.append(clk[k % 2])
        games.append({
            "game_id": f"L{i:07d}", "tc_base": base, "tc_inc": inc,
            "start_ts": (start + dt.timedelta(seconds=float(rng.uniform(0, 7 * 86400))))
            .isoformat(),
            "white": {"account_id": accounts[w], "rating": int(rng.integers(900, 2400)),
                      "rating_diff": int(rng.integers(-8, 9))},
            "black": {"account_id": accounts[b], "rating": int(rng.integers(900, 2400)),
                      "rating_diff": int(rng.integers(-8, 9))},
            "result": ["1-0", "0-1", "1/2-1/2"][int(rng.integers(0, 3))],
            "termination": "normal", "moves": ["e4"] * plies, "clocks": clocks})
    return sorted(games, key=lambda g: g["start_ts"])


def real_week(player_games: Path, week_start: dt.date) -> list[dict]:
    """One week of stored player rows as API requests (own side real, opponent placeholder)."""
    lo = dt.datetime.combine(week_start, dt.time())
    rows = duckdb.connect().execute(f"""
        SELECT * FROM read_parquet('{player_games.as_posix()}')
        WHERE start_ts >= ? AND start_ts < ? ORDER BY start_ts
    """, [lo, lo + dt.timedelta(days=7)]).fetchall()
    cols = [c for c in pq.read_schema(player_games).names]
    games = []
    for r in rows:
        g = dict(zip(cols, r, strict=True))
        own = {"account_id": g["account_id"], "rating": g["own_elo"],
               "rating_diff": g["own_rating_diff"]}
        opp = {"account_id": str(uuid.uuid5(OPPONENT_NS, g["game_key"] + g["color"])),
               "rating": g["opp_elo"]}
        n, mine = g["n_plies"], list(g["own_clocks"] or [])
        first = 0 if g["color"] == "w" else 1
        clocks = [int(mine[min(k // 2, len(mine) - 1)]) if mine else 0 for k in range(n)]
        win = "1-0" if (g["color"] == "w") == (g["score"] == 1.0) else "0-1"
        games.append({
            "game_id": f"R{len(games):07d}", "tc_base": g["tc_base"], "tc_inc": g["tc_inc"],
            "start_ts": g["start_ts"].replace(tzinfo=dt.UTC).isoformat(),
            "white": own if first == 0 else opp, "black": opp if first == 0 else own,
            "result": "1/2-1/2" if g["score"] == 0.5 else win,
            "termination": g["termination"] if g["termination"] != "other" else "normal",
            "event_kind": g["event_kind"],
            "moves": ["e4"] * (n - 1) + (["Qh7#"] if g["ended_in_mate"] else ["e4"]),
            "clocks": clocks if g["clock_valid"] else None})
    return games


def run(games: list[dict], peers: Path, db_url: str, workers: int = 1) -> dict:
    db = Database(db_url)
    now = {"t": dt.datetime.now(dt.UTC)}
    client = TestClient(create_app(db, peers, "loadtest-secret", allow_writes=True,
                                   clock=lambda: now["t"]))
    latencies, statuses, lock = [], {}, threading.Lock()

    def send(batch: list[dict]) -> None:
        for g in batch:
            now["t"] = dt.datetime.fromisoformat(g["start_ts"]) + dt.timedelta(minutes=30)
            t0 = time.perf_counter()
            r = client.post("/v1/games", json=g)
            ms = (time.perf_counter() - t0) * 1000
            with lock:
                latencies.append(ms)
                s = r.json().get("status", str(r.status_code))
                statuses[s] = statuses.get(s, 0) + 1

    t0 = time.perf_counter()
    threads = [threading.Thread(target=send, args=(games[w::workers],)) for w in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    lat = np.array(latencies)
    return {"requests": len(lat), "statuses": statuses, "workers": workers,
            "p50_ms": round(float(np.percentile(lat, 50)), 1),
            "p95_ms": round(float(np.percentile(lat, 95)), 1),
            "p99_ms": round(float(np.percentile(lat, 99)), 1),
            "max_ms": round(float(lat.max()), 1),
            "throughput_rps": round(len(lat) / wall, 1), "wall_s": round(wall, 1),
            "under_1s_share": float((lat < 1000).mean()),
            "machine": {"cpu": platform.processor(), "cores": os.cpu_count(),
                        "os": platform.platform()},
            "database": db_url.split(":")[0]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--games", type=int, default=3000)
    p.add_argument("--accounts", type=int, default=300)
    p.add_argument("--week", default="2025-08-03")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--db", default=None)
    args = p.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="fairplay_load_"))
    peers = PEERS if PEERS.exists() else neutral_peers(tmp / "peers.parquet")
    if args.source == "synthetic":
        games = synthetic_week(args.games, args.accounts, dt.datetime(2026, 9, 6, tzinfo=dt.UTC))
    else:
        week = dt.date.fromisoformat(args.week)
        games = real_week(Path(f"data/processed/player_games/month={week:%Y-%m}.parquet"), week)
    url = args.db or f"sqlite:///{(tmp / 'load.db').as_posix()}"
    report = run(games, peers, url, args.workers) | {
        "source": args.source, "peers": "frozen" if PEERS.exists() else "neutral stand-in"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
