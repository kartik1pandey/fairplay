import json
import zipfile
from pathlib import Path

import pytest

from fairplay.demo import browser

BUNDLE = Path("demo_bundle")


def test_ids_request_refuses_profiles_and_builds_one_export_call():
    assert "error" in json.loads(browser.ids_request("https://lichess.org/@/someone"))
    assert "error" in json.loads(browser.ids_request("no links here"))
    r = json.loads(browser.ids_request("lichess.org/AbCdEfGh\nhttps://lichess.org/ZZZZzzzz/black"))
    assert r["body"] == "AbCdEfGh,ZZZZzzzz"
    assert r["url"].startswith("https://lichess.org/api/games/export/_ids?clocks=true")


def test_empty_pgn_is_an_error_not_a_label():
    assert json.loads(browser.run_pgn("   "))["error"]


@pytest.mark.skipif(not BUNDLE.exists(), reason="demo bundle not built")
def test_every_synthetic_player_matches_its_offline_score():
    for s in browser.scorer().b["samples"]:
        r = json.loads(browser.run_sample(s["strategy"]))
        assert r["synthetic"] and r["strategy_note"]
    from fairplay.demo.analyse import analyse_sample
    for s in browser.scorer().b["samples"]:
        got = analyse_sample([tuple(t) for t in s["rows"]], browser.scorer())["account"]["score"]
        assert got == pytest.approx(s["offline_score"], abs=5e-4)


@pytest.mark.skipif(not BUNDLE.exists(), reason="demo bundle not built")
def test_static_space_ships_code_and_bundle_only(tmp_path):
    from fairplay.demo.make_space import build_static

    out = build_static(tmp_path / "space")
    assert sorted(p.name for p in out.iterdir()) == ["README.md", "app.zip", "index.html"]
    names = zipfile.ZipFile(out / "app.zip").namelist()
    assert "fairplay/demo/browser.py" in names and "demo_bundle/model.txt" in names
    assert not [n for n in names if n.startswith(("data/", "private/", "runs/"))
                or n.endswith(".pyc")]
    # the only table is the frozen aggregate peer reference (rating band x time control)
    assert [n for n in names if n.endswith(".parquet")] == ["demo_bundle/peers_clock30.parquet"]
    assert "sdk: static" in (out / "README.md").read_text(encoding="utf-8")
