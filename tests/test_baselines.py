import datetime as dt
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fairplay.models.baselines import choose_rule, fit_logistic, grouped_oof, logistic_pipeline
from fairplay.models.data import FEATURE_COLUMNS, load_split
from fairplay.runs import BudgetExceeded, claim_evaluation


def model_table(tmp_path: Path, n_accounts=1500, seed=3) -> Path:
    """Synthetic table: flagged accounts have a higher perf gap; think_time_cv is pure noise
    and missing for 30% of rows."""
    rng = np.random.default_rng(seed)
    rows = {c: [] for c in ["account_id", "split", "cutoff", "label", "weight",
                            *FEATURE_COLUMNS]}
    for a in range(n_accounts):
        flagged = rng.random() < 0.08
        weeks = [w for w in range(4) if rng.random() < 0.8] or [0]
        for w in weeks:
            rows["account_id"].append(f"acc{a}")
            rows["split"].append("train")
            rows["cutoff"].append(dt.datetime(2025, 7, 6) + dt.timedelta(weeks=w))
            rows["label"].append(int(flagged))
            rows["weight"].append(1 / len(weeks))
            for c in FEATURE_COLUMNS:
                rows[c].append(float(rng.normal()))
            rows["performance_rating_gap"][-1] = float(rng.normal(150 if flagged else 0, 100))
            rows["games_in_window"][-1] = float(rng.integers(10, 60))
            if rng.random() < 0.3:
                rows["think_time_cv"][-1] = None
    path = tmp_path / "model_table.parquet"
    pq.write_table(pa.table(rows), path)
    return path


def test_rule_choice_uses_training_data_and_picks_signal(tmp_path):
    train = load_split(model_table(tmp_path), "train")
    rule, info = choose_rule(train, capacity=0.05)
    assert info["chosen"] == "perf_gap" and rule.feature == "performance_rating_gap"
    assert set(info["train_flag_precision"]) == {"perf_gap", "timing_cv"}


def test_rule_puts_undefined_values_last(tmp_path):
    train = load_split(model_table(tmp_path), "train")
    from fairplay.models.baselines import RuleModel
    s = RuleModel("think_time_cv", -1).score(train)
    assert np.all(s[np.isnan(train.column("think_time_cv"))] == -np.inf)


def test_grouped_cv_never_splits_an_account(tmp_path, monkeypatch):
    train = load_split(model_table(tmp_path, n_accounts=300), "train")
    seen = []
    import fairplay.models.baselines as b
    real = b.GroupKFold.split

    def spy(self, X, y, groups):
        for fit_idx, pred_idx in real(self, X, y, groups):
            seen.append(set(groups[fit_idx]) & set(groups[pred_idx]))
            yield fit_idx, pred_idx

    monkeypatch.setattr(b.GroupKFold, "split", spy)
    grouped_oof(lambda: logistic_pipeline(1.0), train)
    assert len(seen) == 5 and all(not overlap for overlap in seen)


def test_logistic_handles_missing_values_and_learns_signal(tmp_path):
    train = load_split(model_table(tmp_path), "train")
    model, info = fit_logistic(train)
    assert set(info["cv_oof_pr_auc"]) == {0.001, 0.01, 0.1, 1.0, 10.0}
    assert info["chosen_C"] in info["cv_oof_pr_auc"]
    s = model.score(train)
    assert np.isfinite(s).all()
    imputer = model.pipeline.named_steps["impute"]
    assert imputer.indicator_ is not None  # a "was missing" column exists for think_time_cv
    coef = model.pipeline.named_steps["lr"].coef_[0]
    assert np.argmax(coef[: len(FEATURE_COLUMNS)]) == FEATURE_COLUMNS.index(
        "performance_rating_gap")


def test_validation_budget_and_test_lock(tmp_path):
    log = tmp_path / "eval.jsonl"
    for i in range(8):
        claim_evaluation("validation", f"model{i}", log=log)
    with pytest.raises(BudgetExceeded):
        claim_evaluation("validation", "ninth", log=log)
    with pytest.raises(BudgetExceeded):
        claim_evaluation("test", "peek", log=log)
    claim_evaluation("test", "final", final=True, log=log)
    with pytest.raises(BudgetExceeded):
        claim_evaluation("test", "again", final=True, log=log)
    claim_evaluation("test", "after fix", final=True, reason="bug fix in feature 3", log=log)
    assert len(log.read_text().splitlines()) == 10
