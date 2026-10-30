"""Public demo (design §9): synthetic accounts only, fail closed."""

import pytest

from fairplay.demo.build import check_synthetic_only, strategies_in


def test_demo_refuses_any_non_synthetic_account():
    check_synthetic_only({"queue": [{"account": "syn-A-0001", "case": {"account": "syn-A-0001"}}]})
    with pytest.raises(ValueError):
        check_synthetic_only({"queue": [{"account": "9f1c2d3e-4b5a-4c6d-8e7f-0123456789ab"}]})


def test_held_out_strategy_appears_only_once_scored():
    assert strategies_in({"recall": {"B": {}, "control": {}, "A": {}}}) == ["control", "A", "B"]
    assert strategies_in({"recall": {"C": {}, "A": {}, "control": {}}}) == ["control", "A", "C"]
