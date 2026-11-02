"""Slim, aggregate-only bundle for the public PGN demo (DECISIONS 2026-10-31).

What ships (demo_bundle/):
- model.txt          the frozen tier-1 LightGBM trees (split thresholds only, no rows)
- calibration.json   isotonic step points (score -> P(flag at S0)), display only
- bundle.json        frozen queue threshold (synthetic_score run), score quantiles over August
                     eligible account-weeks (no labels used), peer quantile grids per rating band
                     x time control (July reference), manifest of source runs
- peers_clock30.parquet  the frozen per-cell clock medians (already aggregate)

What never ships: account ids, account-level rows, labels. `check_aggregate_only` scans the
bundle for UUID-shaped strings and fails closed.

Percentiles from a 201-point quantile grid differ from the exact case-file percentiles by at most
0.5 points; the demo says "about".

uv run python -m fairplay.demo.bundle
"""

import json
import re
import shutil
from pathlib import Path

import numpy as np

from fairplay.casefiles.peers import PEER_MIN, REFERENCE, PeerReference
from fairplay.casefiles.run_casefiles import chosen_model
from fairplay.demo.peergrid import GRID, OUT, TC_GROUPS
from fairplay.features.account_features import FEATURES
from fairplay.features.game_features import FEATURE_VERSION
from fairplay.models.data import load_split
from fairplay.models.run_baselines import TABLE
from fairplay.models.run_tier1 import latest_run
from fairplay.runs import RUNS

PEERS = Path("data/processed/peer_stats_clock30.parquet")
SAMPLES_PER_STRATEGY = 3
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def weighted_quantiles(v: np.ndarray, w: np.ndarray, q: np.ndarray = GRID) -> list[float]:
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = (np.cumsum(w) - w / 2) / w.sum()
    return [round(float(x), 6) for x in np.interp(q, cdf, v)]


def peer_grids(ref: PeerReference) -> dict:
    bands = sorted({int(b) for b in ref.c["rating_band"]})
    out = {}
    for tc in TC_GROUPS:
        for band in bands:
            g = ref.group(band, tc)
            entry = {"label": g.label, "n_accounts": int(g.n_accounts), "features": {}}
            if g.sufficient:
                for f, direction in FEATURES.items():
                    if direction == "context":
                        continue
                    v, w, acc = ref._values(g, f)
                    if len(v):
                        entry["features"][f] = {"q": weighted_quantiles(v, w),
                                                **ref.summary(g, f)}
            out[f"{band}|{tc}"] = entry
    return out


def synthetic_samples(model, per_strategy: int = SAMPLES_PER_STRATEGY, seed: int = 0) -> list:
    """Synthetic players for the "try a sample" button: `per_strategy` per strategy drawn at
    random with a fixed seed (not picked by score). Each carries its rows in stored-row form and
    the score the OFFLINE synthetic path gives it, which tests/test_demo_bundle.py reproduces
    through the demo path."""
    import duckdb

    from fairplay.synthetic import run_synthetic as rs
    from fairplay.synthetic import score as sc

    rng = np.random.default_rng(seed)
    samples = []
    for strategy in ("control", "A", "B", "C"):
        f = rs.OUT / "player_games" / f"strategy={strategy}.parquet"
        con = duckdb.connect()
        ids = sorted(r[0] for r in con.execute(
            f"SELECT DISTINCT account_id FROM read_parquet('{f.as_posix()}')").fetchall())
        for acc in sorted(rng.choice(ids, per_strategy, replace=False)):
            rows = con.execute(f"""
                SELECT account_id, game_key, 1, color, strftime(start_ts, '%Y-%m-%d %H:%M:%S'),
                       tc_base, tc_inc, own_elo, opp_elo, own_rating_diff, score::DOUBLE,
                       termination, ended_in_mate::INT, event_kind, n_plies,
                       to_json(own_clocks)::VARCHAR, clock_valid::INT
                FROM read_parquet('{f.as_posix()}') WHERE account_id = ? ORDER BY start_ts
            """, [acc]).fetchall()
            feats = sc.synthetic_features([f], rs.CUTOFF, PEERS)
            feats = [x for x in feats if x["account_id"] == acc]
            offline = float(model.score(sc.to_split(feats, {}, rs.CUTOFF))[0])
            samples.append({"strategy": strategy, "account_id": acc, "rows": rows,
                            "offline_score": offline})
    return samples


def check_aggregate_only(folder: Path) -> None:
    for p in folder.rglob("*"):
        if p.suffix in (".json", ".txt") and UUID_RE.search(p.read_text(encoding="utf-8")):
            raise SystemExit(f"{p}: UUID-shaped string in the demo bundle; refusing to ship")


def build(out: Path = OUT) -> dict:
    model = chosen_model()
    if not hasattr(model, "booster"):
        raise SystemExit("the demo expects the LightGBM tier-1 model")
    synth = sorted(RUNS.glob("*_synthetic_score"))[-1]
    threshold = json.loads((synth / "metrics.json").read_text())["threshold"]
    test = load_split(TABLE, "test")  # scores only; labels are not read
    scores = model.score(test)
    ref = PeerReference.from_model_table(TABLE, "train")
    out.mkdir(exist_ok=True)
    model.booster.save_model(str(out / "model.txt"))  # binary-safe newlines on Windows
    cal = model.calibrator
    (out / "calibration.json").write_text(json.dumps({
        "x": [float(x) for x in cal.X_thresholds_], "y": [float(y) for y in cal.y_thresholds_]}))
    shutil.copyfile(PEERS, out / "peers_clock30.parquet")
    bundle = {
        "versions": {"model": f"tier1-{model.name}", "features": FEATURE_VERSION,
                     "tier1_run": latest_run("tier1").name, "synthetic_run": synth.name},
        "threshold": threshold["threshold"], "capacity": threshold["capacity"],
        "score_quantiles": weighted_quantiles(scores, test.weight, np.linspace(0, 1, 1001)),
        "score_reference": "August 2025 eligible account-weeks (scores only, no labels)",
        "peer_reference": REFERENCE, "peer_min": PEER_MIN,
        "peers": peer_grids(ref),
        "samples": synthetic_samples(model),
    }
    (out / "bundle.json").write_text(json.dumps(bundle))
    check_aggregate_only(out)
    return {"files": sorted(p.name for p in out.iterdir()),
            "bytes": sum(p.stat().st_size for p in out.iterdir())}


def main() -> None:
    print(json.dumps(build(), indent=2))


if __name__ == "__main__":
    main()
