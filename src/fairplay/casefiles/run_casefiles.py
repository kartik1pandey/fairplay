"""Case files for the real weekly queue of the frozen chosen model. LOCAL ONLY (design §9).

Real case files are written under private/casefiles/ and nowhere else: a score next to an account
is an accusation risk even keyed by UUID. The default is the locked August queue, which is only
allowed after run_final has scored it (logged as a diagnostic: frozen model, nothing chosen from
it). --split train builds July queues for development, from the final model's in-sample scores.

uv run python -m fairplay.casefiles.run_casefiles --split test
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pyarrow.parquet as pq

from fairplay.casefiles.peers import PeerReference
from fairplay.casefiles.template import build_case, extreme_game
from fairplay.evaluation.queue import DEFAULT_CAPACITY, evaluate, rank_order
from fairplay.fairness.audit import CAP_MULTIPLIER, HIGH_RATED
from fairplay.features.game_features import FEATURE_VERSION
from fairplay.models.data import load_split
from fairplay.models.run_baselines import TABLE
from fairplay.models.run_tier1 import latest_run
from fairplay.runs import RUNS, claim_evaluation, new_run, save_metrics

OUT = Path("private/casefiles")
GAME_FEATURES = Path("data/processed/game_features")
MONTHS = {"train": ["2025-06", "2025-07"], "test": ["2025-07", "2025-08"]}  # lookback + month


def chosen_model():
    base = joblib.load(latest_run("baselines") / "models.joblib")
    tier1 = joblib.load(latest_run("tier1") / "models.joblib")
    return {**base, **tier1}[tier1["chosen"]]


def display_score(model, split) -> np.ndarray:
    """Calibrated probability where the model has one (isotonic, display only), else its score."""
    return model.probability(split) if hasattr(model, "probability") else model.score(split)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--capacity", type=float, default=DEFAULT_CAPACITY)
    args = p.parse_args()
    if args.split == "test" and not list(RUNS.glob("*_final")):
        raise SystemExit("August is locked until run_final has scored it")

    model = chosen_model()
    split = load_split(TABLE, args.split, ("rating_at_cutoff",))
    rows = pq.read_table(TABLE, filters=[("split", "=", args.split)]).to_pylist()  # same order
    run = new_run("casefiles", vars(args), [TABLE])
    if args.split == "test":
        claim_evaluation("test", f"{run.name}:casefiles", final=True, diagnostic=True,
                         reason="case files for the already-scored final queue")

    score = model.score(split)
    shown = display_score(model, split)
    capped = split.extra["rating_at_cutoff"] >= HIGH_RATED
    month = split.month(capped)
    result = evaluate(month, rank_order(month, score), score, args.capacity,
                      cap_multiplier=CAP_MULTIPLIER)

    ref = PeerReference.from_model_table(TABLE)
    con = duckdb.connect()
    files = [GAME_FEATURES / f"month={m}.parquet" for m in MONTHS[args.split]]
    versions = {"model": f"{model.name}-{run.name}", "features": FEATURE_VERSION,
                "policy": f"c{args.capacity * 100:g}-m{CAP_MULTIPLIER}"}
    counts = {"review": 0, "insufficient_evidence": 0}
    for week_rows in result["queued_rows"]:
        for i in week_rows:
            row = rows[i]
            cutoff = row["cutoff"] if isinstance(row["cutoff"], dt.datetime) else \
                dt.datetime.fromisoformat(str(row["cutoff"]))
            case = build_case(
                row, shown[i], ref, versions,
                games_for=lambda f, pct, a=row["account_id"], c=cutoff:
                    extreme_game(con, files, a, c, f, pct))
            d = OUT / args.split / case["cutoff"]
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{case['case_id']}.json").write_text(json.dumps(case, indent=2),
                                                       encoding="utf-8")
            counts[case["status"]] += 1
    save_metrics(run, {"cases": counts, "weeks": len(result["queued_rows"]),
                       "out": (OUT / args.split).as_posix()})  # aggregate counts only
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
