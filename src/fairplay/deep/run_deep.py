"""Attention-over-games and move-sequence models on July, judged like every other model.

Same rows and folds as run_tier1 / run_bakeoff (grouped by account, folds `seed + 1`), same
metric and paired bootstrap against LightGBM. July-only and exploratory: August is spent.

Also reports, per variant:
- deletion test: does masking the top-attention game move the score more than a random game?
- the move-order test (design §7): mil_seq against mil_seq_shuf, the same model with every
  game's moves shuffled. Only a win there says move ORDER carries signal.

uv run --group research python -m fairplay.deep.run_deep --variants mil mil_tab mil_seq mil_seq_shuf
"""

import argparse
import json
import time

import numpy as np

from fairplay.deep import bags
from fairplay.deep.mil import CONFIG, VARIANTS, Bags, deletion_test, grouped_oof
from fairplay.evaluation.bootstrap import bootstrap, compare, summary
from fairplay.models.data import EXCLUDED_FROM_MODELS, FEATURE_COLUMNS, load_split
from fairplay.models.run_bakeoff import rank_average
from fairplay.models.run_baselines import TABLE, load_oof, save_oof, scorable_subset
from fairplay.models.run_tier1 import latest_run
from fairplay.runs import new_run, save_metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--capacity", type=float, default=0.005)
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train = load_split(TABLE, "train")
    arrays = bags.load(train)
    b = Bags(arrays["games"], arrays["mask"], arrays["seq"], train.X)
    tier1_dir = latest_run("tier1")
    oof = load_oof(tier1_dir, train)
    run = new_run("deep", vars(args) | {"config": CONFIG,
                                        "game_features": list(bags.GAME_FEATURES),
                                        "seq_len": bags.SEQ_LEN, "features": FEATURE_COLUMNS,
                                        "excluded_from_models": list(EXCLUDED_FROM_MODELS),
                                        "tier1_run": tier1_dir.name}, [TABLE])
    info, attention = {}, {}
    for name in args.variants:
        t0 = time.time()
        print(name, flush=True)
        oof[name], attention[name], fitted = grouped_oof(VARIANTS[name], b, train, args.seed + 1)
        positives = [deletion_test(m, b, rows[train.y[rows] == 1], args.seed)
                     for m, rows in fitted]
        n = sum(d["rows"] for d in positives)
        info[name] = {"best_epochs": [m.epochs for m, _ in fitted],
                      "seconds": round(time.time() - t0),
                      "deletion_test_positive_rows": {
                          k: sum(d[k] * d["rows"] for d in positives) / n
                          for k in positives[0] if k != "rows"} | {"rows": n}}
        print(json.dumps(info[name]), flush=True)
    for v in args.variants:
        oof[f"lgbm+{v}"] = rank_average(oof, ["lgbm", v])
    save_oof(run, train, oof)
    np.savez_compressed(run / "attention.npz", **attention)
    ok, scores, n_excluded = scorable_subset(train, oof)
    res = bootstrap(ok.month(), scores, args.capacity, args.resamples, args.seed)
    new = [m for m in scores if m not in ("rule", "logreg", "lgbm", "lgbm_cw")]
    comparisons = {f"{m}_vs_lgbm": compare(res, m, "lgbm") for m in new}
    if {"mil_seq", "mil_seq_shuf"} <= set(args.variants):
        comparisons["mil_seq_vs_mil_seq_shuf"] = compare(res, "mil_seq", "mil_seq_shuf")
    if {"mil", "mil_seq"} <= set(args.variants):
        comparisons["mil_seq_vs_mil"] = compare(res, "mil_seq", "mil")
    metrics = {"variants": info, "july_oof_rows_excluded_unscorable": n_excluded,
               "july_oof": summary(res), "comparisons": comparisons,
               "beats_lgbm": [m for m in new
                              if comparisons[f"{m}_vs_lgbm"]["verdict"] == f"{m} beats lgbm"]}
    save_metrics(run, metrics)  # last: marks the run complete
    print(json.dumps({"run": run.name,
                      "precision": {m: v["flag_precision"]["value"]
                                    for m, v in metrics["july_oof"].items()},
                      "comparisons": {k: v["verdict"] for k, v in comparisons.items()}},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
