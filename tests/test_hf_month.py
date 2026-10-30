"""The SQL filter must make exactly the same decisions as the Python pre-filter."""

import duckdb
import pytest

from fairplay.ingest.hf_month import install_macros
from fairplay.ingest.prefilter import is_blitz, sample_bucket


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    install_macros(c)
    return c


def test_sql_bucket_matches_python(con):
    names = [f"Player_{i}" for i in range(2000)] + ["MagnusFan", "a", "ÜberPlayer"]
    sql = dict(con.execute(
        "SELECT n, sample_bucket(n) FROM unnest(?::VARCHAR[]) t(n)", [names]
    ).fetchall())
    assert all(sql[n] == sample_bucket(n.encode()) for n in names if n.isascii())


@pytest.mark.parametrize(
    "tc", ["179+0", "180+0", "479+0", "480+0", "120+1", "60+3", "300+3", "300+5", "-", "", "x+1"]
)
def test_sql_blitz_matches_python(con, tc):
    (sql,) = con.execute("SELECT is_blitz(?)", [tc]).fetchone()
    assert sql is is_blitz(tc.encode())
