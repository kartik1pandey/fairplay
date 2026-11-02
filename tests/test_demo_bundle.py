import json
from pathlib import Path

import numpy as np
import pytest

from fairplay.demo.peergrid import GRID, PeerGrid

BUNDLE = Path("demo_bundle")


def grid_with(values: np.ndarray) -> PeerGrid:
    q = np.quantile(values, GRID).tolist()
    return PeerGrid({"1400|180+0": {"label": "1400-1599 blitz 180+0", "n_accounts": 500,
                                    "features": {"think_time_cv": {
                                        "q": q, "peer_p05": q[10], "peer_median": q[100],
                                        "peer_p95": q[190], "peer_n": 500}}}})


def test_peer_grid_percentiles_match_the_distribution():
    g = grid_with(np.arange(1000, dtype=float))
    group = g.group(1450, "180+0")
    assert group.sufficient
    assert abs(g.percentile(group, "think_time_cv", 250.0) - 25) < 0.6
    assert g.percentile(group, "think_time_cv", -5.0) == 0
    assert g.percentile(group, "think_time_cv", 5000.0) == 100
    assert np.isnan(g.percentile(group, "time_loss_share", 0.1))  # feature not in the grid


def test_peer_grid_ties_take_the_middle():
    g = grid_with(np.r_[np.zeros(500), np.ones(500)])
    assert abs(g.percentile(g.group(1400, "180+0"), "think_time_cv", 0.0) - 25) < 0.6


def test_unknown_band_falls_back_and_missing_cell_fails_closed():
    g = grid_with(np.arange(10, dtype=float))
    assert g.group(2900, "180+0").label == "1400-1599 blitz 180+0"  # nearest band
    missing = g.group(1400, "600+5")  # -> "other", absent here
    assert not missing.sufficient and "other" in missing.label


@pytest.mark.skipif(not BUNDLE.exists(), reason="demo bundle not built")
def test_bundle_is_aggregate_only_and_samples_are_synthetic():
    from fairplay.demo.bundle import check_aggregate_only

    check_aggregate_only(BUNDLE)
    b = json.loads((BUNDLE / "bundle.json").read_text())
    assert all(s["account_id"].startswith("syn-") for s in b["samples"])
    assert {s["strategy"] for s in b["samples"]} == {"control", "A", "B", "C"}


@pytest.mark.skipif(not BUNDLE.exists(), reason="demo bundle not built")
def test_demo_path_reproduces_offline_scores():
    from fairplay.demo.analyse import Scorer, analyse_sample

    scorer = Scorer(BUNDLE)
    for s in json.loads((BUNDLE / "bundle.json").read_text())["samples"]:
        r = analyse_sample([tuple(t) for t in s["rows"]], scorer)
        assert r["account"]["score"] == pytest.approx(s["offline_score"], abs=1e-3)
        assert "cheat" not in json.dumps(r).lower()


@pytest.mark.skipif(not (BUNDLE / "attention_mil.npz").exists(), reason="attention not exported")
def test_attention_weights_cover_the_window_games():
    from fairplay.demo.analyse import Scorer, analyse_sample

    scorer = Scorer(BUNDLE)
    s = json.loads((BUNDLE / "bundle.json").read_text())["samples"][0]
    r = analyse_sample([tuple(t) for t in s["rows"]], scorer)
    weights = [g["attention"] for g in r["games"] if g.get("attention") is not None]
    assert len(weights) == min(30, r["games_in_30_day_window"])
    assert abs(sum(weights) - 1) < 0.01 and np.isfinite(r["research_model"]["logit"])
    assert all("_key" not in g for g in r["games"])
