"""Published-output guard, results table, pipeline cache (design §12 reproducibility/privacy)."""

import json
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fairplay import pipeline, report
from fairplay.privacy import check_published


def test_published_guard_blocks_account_level_outputs(tmp_path):
    (tmp_path / "ok.md").write_text("flag-precision 0.31, lift 6.2x; account syn-A-0003\n",
                                    encoding="utf-8")
    (tmp_path / "leak.md").write_text("queued 9f1c2d3e-4b5a-4c6d-8e7f-0123456789ab\n",
                                      encoding="utf-8")
    (tmp_path / "rows.csv").write_text("account_id,score\nx,0.9\n", encoding="utf-8")
    pq.write_table(pa.table({"username": ["x"]}), tmp_path / "t.parquet")
    (tmp_path / "agg.json").write_text('{"band": "1600", "precision": 0.3}', encoding="utf-8")
    problems = check_published([tmp_path])
    assert len(problems) == 3
    assert any("leak.md:1" in p for p in problems)
    assert any("rows.csv" in p and "account_id" in p for p in problems)
    assert any("t.parquet" in p and "username" in p for p in problems)


def test_repo_published_files_are_clean():
    paths = [p for p in ("README.md", "docs", "results", "DECISIONS.md") if os.path.exists(p)]
    from pathlib import Path

    assert check_published([Path(p) for p in paths]) == []


def test_report_table_and_readme_check(tmp_path):
    runs, results = tmp_path / "runs", tmp_path / "results"
    (runs / "r1").mkdir(parents=True)
    (runs / "r1" / "metrics.json").write_text(json.dumps(
        {"models": {"lgbm": {"flag_precision": 0.3123, "ci": [0.25, 0.37]}}}), encoding="utf-8")
    results.mkdir()
    (results / "runs.json").write_text(json.dumps({"rows": [{
        "experiment": "R", "split": "test", "claim": "flag-precision@k, LightGBM", "run": "r1",
        "value": "models.lgbm.flag_precision", "low": "models.lgbm.ci.0",
        "high": "models.lgbm.ci.1"}]}), encoding="utf-8")
    md = report.build(results, runs)
    assert "| R | test | flag-precision@k, LightGBM | 0.312 | 0.250 – 0.370 | `r1` |" in md
    readme = tmp_path / "README.md"
    readme.write_text(f"# x\n{report.START}\nold numbers\n{report.END}\ntail\n", encoding="utf-8")
    assert not report.check(readme, results)  # a hand-typed number fails CI
    report.sync(readme, results)
    assert report.check(readme, results)
    assert readme.read_text(encoding="utf-8").endswith("tail\n")


@pytest.mark.skipif(not os.path.exists("README.md"), reason="no README in this checkout")
def test_committed_readme_matches_generated_table():
    from pathlib import Path

    expected = (Path("results/results.md").read_text(encoding="utf-8")
                if Path("results/results.md").exists() else report.EMPTY + "\n")
    assert report.readme_block(Path("README.md")) == expected


def test_pipeline_cache_reruns_changed_step_and_everything_after(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("1")
    for out in ("b.txt", "c.txt"):
        (tmp_path / out).write_text("x")
    steps = [pipeline.Step("first", "fairplay.report", inputs=("a.txt",), outputs=("b.txt",)),
             pipeline.Step("second", "fairplay.report", inputs=("b.txt",), outputs=("c.txt",))]
    monkeypatch.setattr(pipeline, "TARGETS", {"t": steps})
    monkeypatch.setattr(pipeline, "STATE", tmp_path / "state.json")
    assert pipeline.adopt("t") == ["first", "second"]
    assert [a for _, a, _ in pipeline.plan("t", pipeline.load_state())] == ["cached", "cached"]
    time.sleep(1.1)  # mtime resolution
    (tmp_path / "a.txt").write_text("22")  # input of the FIRST step changed
    acts = [(s.name, a) for s, a, _ in pipeline.plan("t", pipeline.load_state())]
    assert acts == [("first", "run"), ("second", "run")]  # downstream reruns too
    (tmp_path / "c.txt").unlink()
    pipeline.adopt("t")
    assert [a for _, a, _ in pipeline.plan("t", pipeline.load_state())][1] == "run"  # missing


def test_reproduce_synthetic_never_needs_labels():
    assert not any(s.needs_labels for s in pipeline.TARGETS["reproduce-synthetic"])
    assert all("labels" not in s.module for s in pipeline.STEPS)  # no live API step at all


@pytest.mark.skipif(not os.path.exists("README.md"), reason="no README in this checkout")
@pytest.mark.parametrize("bad", ["README.md"])
def test_readme_has_no_cheating_claims(bad):
    from pathlib import Path

    text = Path(bad).read_text(encoding="utf-8").lower()
    for phrase in ("detects cheaters", "cheating recall", "false-positive rate", "% of cheaters"):
        assert phrase not in text


def test_copy_atomic_leaves_nothing_on_failure(tmp_path):
    import duckdb

    from fairplay.duck import copy_atomic

    con = duckdb.connect()
    out = tmp_path / "x.parquet"
    with pytest.raises(duckdb.Error):
        copy_atomic(con, "SELECT 1 / 0::INT AS a FROM range(3) WHERE error('boom') IS NULL", out)
    assert not out.exists()
    copy_atomic(con, "SELECT range AS a FROM range(3)", out)
    assert pq.read_table(out).num_rows == 3 and not list(tmp_path.glob("*.tmp"))
