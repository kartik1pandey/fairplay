"""Tier 2 engine budget (design §0.13): positions per account and the cost table."""

import pytest

from fairplay.tier2.cost import BUDGET_CPU_HOURS, cost_table, positions_in_game


@pytest.mark.parametrize(("plies", "white", "expected"), [
    (20, True, 0),    # 10 own moves: none after move 10
    (21, True, 1),    # white played move 11
    (22, False, 1),   # black played move 11
    (200, True, 50),  # capped at move 60
])
def test_positions_in_game(plies, white, expected):
    assert positions_in_game(plies, white) == expected


def test_cost_table_counts_candidates_and_audit():
    rows = cost_table(10_000, 100, {"100000": 0.1})
    r = rows[0]  # q = 1%: 100 candidates, audit ceil(2% x 9900) = 198
    assert (r["candidates"], r["audit"]) == (100, 198)
    assert r["cpu_hours_per_cutoff"] == pytest.approx(298 * 100 * 0.1 / 3600, abs=0.05)
    assert r["fits_budget"] == (r["cpu_hours_per_cutoff"] <= BUDGET_CPU_HOURS)
