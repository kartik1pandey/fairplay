"""Monitoring replay (design §11): the frozen model run week by week over July and August.

Each weekly cutoff gets what the monitoring job would see that week:
- PSI of the score distribution and of the model's top-10 features, against July;
- operations: eligible accounts, queue size k_w, how many 2200+ accounts the cap held back;
- the retraining decision, by rule: PSI > 0.2 -> investigation note, never automatic retraining;
  promotion needs a newer matured month beaten by the paired rule, and the test month is never
  used for promotion.

August labels are NOT read: August was scored once by run_final and stays locked. The reference
is the deployed model's scores on its own training month (what a production monitor has).

uv run python -m fairplay.monitoring.replay
"""

import json

import joblib
import numpy as np

from fairplay.evaluation.queue import DEFAULT_CAPACITY, evaluate, rank_order
from fairplay.fairness.audit import CAP_MULTIPLIER, HIGH_RATED
from fairplay.models.data import load_split
from fairplay.models.run_baselines import TABLE, take
from fairplay.models.run_tier1 import latest_run
from fairplay.monitoring.drift import PSI_ALERT, drift_report
from fairplay.runs import new_run, save_metrics

TOP_FEATURES = 10


def weekly_ops(split, score: np.ndarray, capacity: float) -> dict:
    capped = split.extra["rating_at_cutoff"] >= HIGH_RATED
    month = split.month(capped)  # queue selection reads scores and ratings, never labels
    order = rank_order(month, score)
    with_cap = evaluate(month, order, score, capacity, cap_multiplier=CAP_MULTIPLIER)
    without = evaluate(month, order, score, capacity)

    def high(r: dict) -> int:
        return round(r["capped_share_per_week"][0] * r["queue_size_per_week"][0])

    return {"eligible": int(len(split.y)), "k": with_cap["k_per_week"][0],
            "queued": with_cap["queue_size_per_week"][0],
            "high_rated_held_back": high(without) - high(with_cap)}


def main() -> None:
    tier1_dir = latest_run("tier1")
    t1 = joblib.load(tier1_dir / "models.joblib")
    model = t1[t1["chosen"]]
    metrics = json.loads((tier1_dir / "metrics.json").read_text())
    top = [f for f in metrics["gain_importance"]][:TOP_FEATURES]
    run = new_run("monitoring_replay", {"tier1_run": tier1_dir.name, "top_features": top},
                  [TABLE])
    july = load_split(TABLE, "train", ("rating_at_cutoff",))
    ref_scores = model.score(july)
    ref_feats = {f: july.column(f) for f in top}
    weeks = []
    for name in ("train", "test"):
        split = load_split(TABLE, name, ("rating_at_cutoff",))
        if name == "test":  # locked month: queue and drift only, labels blanked
            split.y = np.zeros_like(split.y)
        scores = model.score(split)
        for c in np.unique(split.cutoff):
            m = split.cutoff == c
            wk = take(split, m)
            rep = drift_report(ref_scores, scores[m], ref_feats,
                               {f: wk.column(f) for f in top},
                               ops=weekly_ops(wk, scores[m], DEFAULT_CAPACITY))
            weeks.append({"cutoff": str(c)[:10], "month": "July" if name == "train" else "August",
                          **rep,
                          "retraining": "no: drift alerts trigger an investigation note only; "
                                        "promotion needs a newer matured month"})
            print(weeks[-1]["cutoff"], f"score PSI {rep['score_psi']:.3f}",
                  f"alerts {len(rep['alerts'])}", rep["ops"], flush=True)
    report = {"psi_alert": PSI_ALERT, "weeks": weeks,
              "weeks_with_alerts": [w["cutoff"] for w in weeks if w["alerts"]],
              "max_score_psi": max(w["score_psi"] for w in weeks)}
    save_metrics(run, report)
    print(json.dumps({k: v for k, v in report.items() if k != "weeks"}, indent=2))


if __name__ == "__main__":
    main()
