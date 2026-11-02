import datetime as dt
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fairplay.labels import store
from fairplay.prospective import run_p


def test_outcome_classes_censor_instead_of_counting_negatives():
    # status tuples: (outcome, tos_violation, disabled, fetched_at)
    index = {"a": ("ok", 0, 0, "t"), "b": ("ok", 1, 0, "t"), "c": ("ok", 0, 1, "t"),
             "d": ("not_returned", None, None, "t"), "e": ("ok", 0, 0, "t"),
             "f": ("ok", 0, 0, "t"), "g": ("ok", 0, 0, "t")}
    outcome = {"a": ("ok", 1, 1, "t"), "e": ("not_returned", None, None, "t"),
               "f": ("ok", 0, 1, "t"), "g": ("ok", 0, 0, "t")}
    ids = np.array(list("abcdefgh"))
    cls, counts = run_p.outcome_classes(ids, index, outcome)
    assert cls.tolist() == [
        "positive",                      # unflagged at index, flagged at outcome
        "excluded_flagged_at_index",     # outcome already known
        "excluded_closed_at_index",
        "excluded_no_index_status",
        "censored_missing_or_renamed",   # never a negative
        "censored_closed_without_flag",  # closed without a flag: unknown, not honest
        "negative",
        "excluded_no_index_status",      # absent from P0 entirely
    ]
    assert counts["positive"] == 1 and counts["negative"] == 1


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path / "p_test"
    root.mkdir()
    runs = tmp_path / "runs"
    monkeypatch.setattr(run_p, "RUNS", runs)
    scores = run_p.scores_path(root)
    pq.write_table(pa.table({"account_id": ["a"], "score_lgbm": [0.5]}), scores)
    freeze = runs / "20261105T000000Z_p_freeze_p_test"
    freeze.mkdir(parents=True)
    (freeze / "metrics.json").write_text(json.dumps({
        "scores_sha256": run_p.file_sha256(scores), "frozen_at": "2026-11-05T00:00:00+00:00"}))
    db = tmp_path / "labels.sqlite"
    real = store.connect
    monkeypatch.setattr(run_p.store, "connect", lambda path=db: real(path))
    return root, scores, db


def test_check_freeze_passes_when_scores_are_untouched(frozen):
    root, _, _ = frozen
    assert run_p.check_freeze(root)["index_created_at"] is None


def test_check_freeze_rejects_edited_scores(frozen):
    root, scores, _ = frozen
    pq.write_table(pa.table({"account_id": ["a"], "score_lgbm": [0.9]}), scores)
    with pytest.raises(SystemExit, match="differs from the frozen hash"):
        run_p.check_freeze(root)


def test_check_freeze_rejects_an_index_snapshot_older_than_the_scores(frozen):
    root, _, db = frozen
    conn = store.connect(db)
    with conn:
        conn.execute("INSERT INTO snapshots VALUES ('P0', '2026-11-04T00:00:00+00:00', 0, '')")
    with pytest.raises(SystemExit, match="before the scores were frozen"):
        run_p.check_freeze(root)


def test_evaluate_end_to_end_with_fake_snapshots(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 2000
    ids = [f"acc{i:04d}" for i in range(n)]
    positive = rng.random(n) < 0.05
    good = np.where(positive, 2.0, 0.0) + rng.normal(size=n)  # informative score
    root = tmp_path / "p_e2e"
    root.mkdir()
    pq.write_table(pa.table({
        "account_id": ids,
        "cutoff": pa.array([dt.datetime(2026, 11, 1)] * n, pa.timestamp("us")),
        "games_in_window": rng.integers(10, 60, n), "rating_at_cutoff": rng.normal(1600, 300, n),
        "score_rule": rng.normal(size=n), "score_logreg": rng.normal(size=n),
        "score_lgbm": good, "score_mil": good + rng.normal(scale=0.1, size=n)}),
        run_p.scores_path(root))
    runs = tmp_path / "runs"
    monkeypatch.setattr(run_p, "RUNS", runs)
    real_new_run = run_p.new_run
    monkeypatch.setattr(run_p, "new_run",
                        lambda name, cfg, inputs: real_new_run(name, cfg, inputs, root=runs))
    freeze = runs / "20261105T000000Z_p_freeze_p_e2e"
    freeze.mkdir(parents=True)
    (freeze / "metrics.json").write_text(json.dumps({
        "scores_sha256": run_p.file_sha256(run_p.scores_path(root)),
        "frozen_at": "2026-11-05T00:00:00+00:00"}))
    db = tmp_path / "labels.sqlite"
    real = store.connect
    monkeypatch.setattr(run_p.store, "connect", lambda path=db: real(path))
    conn = store.connect(db)
    t0, t60 = "2026-11-06T00:00:00+00:00", "2027-01-05T00:00:00+00:00"
    with conn:
        conn.execute("INSERT INTO snapshots VALUES ('P0', ?, ?, '')", (t0, n))
        conn.execute("INSERT INTO snapshots VALUES ('P60', ?, ?, '')", (t60, n))
        conn.executemany("INSERT INTO accounts VALUES (?, ?, ?)",
                         [(a, a, t0) for a in ids])
        p0 = [("P0", a, "ok", int(i < 10), 0, 0, t0) for i, a in enumerate(ids)]
        p60 = [("P60", a, "not_returned" if i % 100 == 50 else "ok",
                None if i % 100 == 50 else int(positive[i]), 0, 0, t60)
               for i, a in enumerate(ids)]
        conn.executemany("INSERT INTO labels VALUES (?, ?, ?, ?, ?, ?, ?)", p0 + p60)
    early = run_p.evaluate("P60", interim=False, capacity=0.02, resamples=50, seed=0,
                           root=root)
    assert early["days_after_index"] == 60 and not early["descriptive_only"]
    counts = early["population_counts"]
    assert counts["excluded_flagged_at_index"] == 10
    assert counts["censored_missing_or_renamed"] == 20
    assert early["population"] == n - 30 == counts["positive"] + counts["negative"]
    s = early["uncapped"]["summary"]
    assert s["lgbm"]["flag_precision"]["value"] > s["rule"]["flag_precision"]["value"]
    assert early["uncapped"]["comparisons"]["lgbm_vs_rule"]["verdict"] == "lgbm beats rule"


def test_evaluate_refuses_an_immature_outcome_unless_interim(tmp_path, monkeypatch):
    monkeypatch.setattr(run_p, "check_freeze", lambda root, index: {})
    conn = store.connect(tmp_path / "l.sqlite")
    monkeypatch.setattr(run_p.store, "connect", lambda: conn)
    monkeypatch.setattr(run_p, "status", lambda c, sid: {
        "a": ("ok", 0, 0, "2026-11-06T00:00:00+00:00" if sid == "P0"
              else "2026-11-20T00:00:00+00:00")})
    with pytest.raises(SystemExit, match="14 days after P0"):
        run_p.evaluate("P14", interim=False, capacity=0.005, resamples=10, seed=0)
