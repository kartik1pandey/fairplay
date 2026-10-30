"""Features + S0 labels -> the model table (design §3, §4).

Label = tosViolation at snapshot S0 (Experiment R). Rows are excluded, and counted, when:
- the account's S0 outcome is not "ok" (not_returned / http_error): unknown is not negative;
- the account is disabled without a flag: closed for an unknown reason (design §3), reported as
  a separate group, never used as a negative.
Adds feature 14 (account age at the cutoff) and the training weight 1 / rows per account, so very
active accounts do not dominate training (design §4).

uv run python -m fairplay.dataset.assemble --snapshot S0
"""

import argparse
import json
from pathlib import Path

import duckdb
import pyarrow as pa

from fairplay import duck
from fairplay.duck import copy_atomic
from fairplay.features.account_features import FEATURES
from fairplay.labels import store

PROCESSED = Path("data/processed")


def label_table(conn, snapshot_id: str) -> pa.Table:
    rows = conn.execute(
        "SELECT account_id, outcome, tos_violation, disabled, created_at_ms FROM labels "
        "WHERE snapshot_id = ?", (snapshot_id,)).fetchall()
    cols = ["account_id", "outcome", "tos_violation", "disabled", "created_at_ms"]
    return pa.table({c: [r[i] for r in rows] for i, c in enumerate(cols)})


def assemble(con: duckdb.DuckDBPyConnection, features_path: Path, labels: pa.Table) -> dict:
    con.register("labels_s", labels)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE joined AS
        SELECT f.*, l.outcome, l.tos_violation, l.disabled, l.created_at_ms,
            CASE WHEN l.outcome IS NULL THEN 'no_label_row'
                 WHEN l.outcome != 'ok' THEN 'label_' || l.outcome
                 WHEN l.disabled = 1 AND l.tos_violation = 0 THEN 'disabled_unflagged'
                 END AS exclusion
        FROM read_parquet('{features_path.as_posix()}') f
        LEFT JOIN labels_s l USING (account_id)
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE model_table AS
        SELECT * EXCLUDE (outcome, disabled, created_at_ms, exclusion, tos_violation),
            tos_violation::INTEGER AS label,
            date_diff('day', make_timestamp(created_at_ms * 1000), cutoff) AS account_age_days,
            1.0 / count(*) OVER (PARTITION BY split, account_id) AS weight
        FROM joined WHERE exclusion IS NULL
    """)
    excluded = con.execute(
        "SELECT split, exclusion, count(*), count(DISTINCT account_id) FROM joined "
        "WHERE exclusion IS NOT NULL GROUP BY ALL ORDER BY ALL").fetchall()
    per_split = con.execute("""
        SELECT split, count(*) AS rows, count(DISTINCT account_id) AS accounts,
               count(DISTINCT account_id) FILTER (WHERE label = 1) AS positive_accounts,
               round(avg(label), 5) AS positive_row_share
        FROM model_table GROUP BY split ORDER BY split""").fetchall()
    return {
        "excluded": [{"split": s, "reason": r, "rows": n, "accounts": a}
                     for s, r, n, a in excluded],
        "per_split": [{"split": s, "rows": n, "accounts": a, "positive_accounts": p,
                       "positive_row_share": sh} for s, n, a, p, sh in per_split],
    }


MISSING_LABEL = ("no_label_row", "label_http_error")
SMD_LIMIT = 0.1  # common balance threshold for standardised mean differences


def missing_label_balance(con: duckdb.DuckDBPyConnection,
                          cols: list[str] | None = None) -> dict:
    """Accounts without a label vs labelled ones, on label-free columns (DECISIONS 2026-10-29).

    SMD = (mean_missing - mean_labelled) / sqrt((var_missing + var_labelled) / 2), per split. If
    every |SMD| stays below 0.1 the exclusion is treated as ignorable; otherwise it is a limit.
    """
    cols = cols or [c for c in FEATURES if c != "account_age_days"] + ["rating_at_cutoff"]
    out = {}
    for split, in con.execute("SELECT DISTINCT split FROM joined ORDER BY 1").fetchall():
        rows = {}
        for c in cols:
            m1, v1, n1, m0, v0 = con.execute(f"""
                SELECT avg({c}) FILTER (WHERE exclusion IN {MISSING_LABEL}),
                       var_samp({c}) FILTER (WHERE exclusion IN {MISSING_LABEL}),
                       count({c}) FILTER (WHERE exclusion IN {MISSING_LABEL}),
                       avg({c}) FILTER (WHERE exclusion IS NULL),
                       var_samp({c}) FILTER (WHERE exclusion IS NULL)
                FROM joined WHERE split = ?""", [split]).fetchone()
            pooled = ((v1 or 0) + (v0 or 0)) / 2
            smd = None if n1 < 2 or not pooled else (m1 - m0) / pooled ** 0.5
            rows[c] = None if smd is None else round(smd, 4)
        out[split] = {"smd": rows, "above_limit": sorted(
            c for c, v in rows.items() if v is not None and abs(v) > SMD_LIMIT)}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", default="S0")
    args = p.parse_args()
    con = duck.connect()
    report = assemble(con, PROCESSED / "features_tier1.parquet",
                      label_table(store.connect(), args.snapshot))
    report["missing_label_balance"] = missing_label_balance(con)
    out = PROCESSED / f"model_table_{args.snapshot}.parquet"
    copy_atomic(con, "SELECT * FROM model_table ORDER BY split, cutoff, account_id", out)
    (PROCESSED / f"model_table_{args.snapshot}_report.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
