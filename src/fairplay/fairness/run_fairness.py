"""Fairness cap + audit on July out-of-fold scores: a development check before the locked test.

The cap (m = 1.0) and the 2x criterion were fixed before any result. This run uses the OOF scores
saved by run_tier1, so no model sees its own training rows and August stays untouched; the
binding audit on August is part of run_final.

uv run python -m fairplay.fairness.run_fairness
"""

import argparse
import json

from fairplay.fairness.audit import AUDIT_COLUMNS, audit, fast_improver_thresholds
from fairplay.models.data import load_split
from fairplay.models.run_baselines import TABLE, load_oof, scorable_subset
from fairplay.models.run_tier1 import latest_run
from fairplay.runs import new_run, save_metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--capacity", type=float, default=0.005)
    p.add_argument("--resamples", type=int, default=1000)
    args = p.parse_args()
    train = load_split(TABLE, "train", AUDIT_COLUMNS)
    tier1_dir = latest_run("tier1")
    oof = load_oof(tier1_dir, train)
    run = new_run("fairness", vars(args) | {"tier1_run": tier1_dir.name}, [TABLE])

    thresholds = fast_improver_thresholds(train)
    ok, scores, n_excluded = scorable_subset(train, oof)
    report = {"scores": "July out-of-fold", "fast_improver_thresholds_by_band": thresholds,
              "rows_excluded_unscorable": n_excluded,
              "models": {name: audit(ok, s, thresholds, args.capacity, args.resamples)
                         for name, s in scores.items()}}
    save_metrics(run, report)
    print(json.dumps({name: {"precision_uncapped": r["uncapped"]["flag_precision"],
                             "precision_capped": r["capped"]["flag_precision"],
                             "failing_groups_uncapped": r["uncapped"]["failing_groups"],
                             "failing_groups_capped": r["capped"]["failing_groups"]}
                      for name, r in report["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
