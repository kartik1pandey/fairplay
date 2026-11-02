"""Experiment P: score accounts on October 2026 games, freeze the scores, then watch for flags.

Design §0 item 1, with the approved index-date change (DECISIONS 2026-10-28): the index is the
time the first status snapshot (P0) is fetched, after the October file is published.

Steps, each a subcommand, in this order:
  build     clean October, build game features, eligible rows at cutoff 2026-11-01 00:00
            (>= 10 blitz games in the 30 days before), tier-1 features with the FROZEN July 2025
            peer table. Outputs go to data/processed/p/, apart from Experiment R's tables.
  score     score every eligible row with the frozen models (rule, logistic regression,
            LightGBM, and the research attention model), write scores_P.parquet, and record its
            SHA256 in a `p_freeze` run. Also writes the account list for P0 (private/).
  [labels]  check-freeze, then
            uv run python -m fairplay.labels.cli create P0 --accounts private/accounts_p.txt
            uv run python -m fairplay.labels.cli run P0
  evaluate  --outcome P60 (60 days after P0) or an interim snapshot with --interim.
            Population: eligible, P0 status ok and not flagged/closed at P0. Outcome: flagged at
            the outcome snapshot. Missing, renamed or closed-without-flag at the outcome =
            censored: excluded from the denominator and counted, never negatives.
            The queue is the one-cutoff version of Experiment R's: k = ceil(0.5% x population),
            same tie-breaks, same paired account bootstrap and verdict rule.

uv run --group research python -m fairplay.prospective.run_p build
uv run --group research python -m fairplay.prospective.run_p score
uv run python -m fairplay.prospective.run_p check-freeze
uv run python -m fairplay.prospective.run_p evaluate --outcome P60

Rehearsal on Experiment R data (must reproduce R's features and scores at that cutoff):
uv run --group research python -m fairplay.prospective.run_p --root data/processed/p_rehearsal
    build --month 2025-08 --cutoff 2025-08-31
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fairplay import duck
from fairplay.dataset.eligibility import END_MARGIN, MIN_GAMES, WINDOW
from fairplay.duck import copy_atomic
from fairplay.features.account_features import apply_peer_z, cutoff_features
from fairplay.features.game_features import FEATURE_VERSION, game_features_sql
from fairplay.ingest.clean import (
    clean_month,
    game_key_secret,
    install_parse_macros,
    register_account_map,
)
from fairplay.labels import store
from fairplay.models.data import FEATURE_COLUMNS, Split
from fairplay.runs import RUNS, file_sha256, new_run, save_metrics

MONTH = "2026-10"
CUTOFF = dt.datetime(2026, 11, 1)
HORIZON_DAYS = 60
P_ROOT = Path("data/processed/p")
GF_PARTS = 8
PEERS = Path("data/processed/peer_stats_clock30.parquet")  # frozen July 2025 reference
MODELS = ("rule", "logreg", "lgbm", "mil")
PRE_REGISTERED = [("lgbm", "rule"), ("lgbm", "logreg"), ("mil", "lgbm")]


def features_path(root: Path) -> Path:
    return root / "features_P.parquet"


def scores_path(root: Path) -> Path:
    return root / "scores_P.parquet"


def accounts_path(root: Path) -> Path:
    """Usernames for the P0 status fetch: private/, never next to the UUID-keyed outputs."""
    return Path("private") / f"accounts_{root.name}.txt"


def game_feature_parts(con: duckdb.DuckDBPyConnection, player_games: str, month: str,
                       root: Path) -> list[Path]:
    """Game features in GF_PARTS files split by account, each written atomically and skipped
    if present: one ~30 min query becomes resumable pieces on a laptop where memory pressure
    stops long jobs. The SQL is per row (no windows), so the rows equal one build_month call.
    Files are named month=M-pNN.parquet, so the month=*.parquet globs (deep/bags) find them."""
    out_dir = root / "game_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for i in range(GF_PARTS):
        out = out_dir / f"month={month}-p{i:02d}.parquet"
        if not out.exists():
            source = (f"(SELECT * FROM read_parquet('{player_games}') "
                      f"WHERE hash(account_id) % {GF_PARTS} = {i})")
            copy_atomic(con, game_features_sql(source), out)
        parts.append(out)
    return parts


def build(month: str = MONTH, cutoff: dt.datetime = CUTOFF, root: Path = P_ROOT,
          rate_bp: int = 1000) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    clean_con = duckdb.connect()  # the cleaning step's own settings (clean.main)
    clean_con.execute("SET enable_progress_bar = false; SET memory_limit = '2GB'; "
                      "SET threads = 2; SET preserve_insertion_order = false; "
                      "SET temp_directory = 'private/tmp/duckdb'")
    install_parse_macros(clean_con)
    new = register_account_map(clean_con, [month], rate_bp)
    # each step is skipped when its output exists, so a killed build resumes where it stopped
    manifest = root / f"clean_manifest_{month}.json"
    clean = (json.loads(manifest.read_text()) if manifest.exists()
             else clean_month(clean_con, month, rate_bp, game_key_secret(), root))
    clean_con.close()
    con = duck.connect()
    pg = (root / "player_games" / f"month={month}.parquet").as_posix()
    gf = game_feature_parts(con, pg, month, root)
    con.execute(f"""
        CREATE TEMP TABLE eligible AS
        SELECT account_id, TIMESTAMP '{cutoff:%Y-%m-%d %H:%M:%S}' AS cutoff, count(*) AS n_games
        FROM read_parquet('{pg}')
        WHERE start_ts >= TIMESTAMP '{cutoff - WINDOW:%Y-%m-%d %H:%M:%S}'
          AND start_ts <  TIMESTAMP '{cutoff - END_MARGIN:%Y-%m-%d %H:%M:%S}'
        GROUP BY account_id HAVING count(*) >= {MIN_GAMES}
    """)
    rel = cutoff_features(con, gf, "eligible", cutoff)
    con.execute(f"CREATE TEMP TABLE feats AS {rel.sql_query()}")
    con.execute(f"CREATE TEMP TABLE peer_clock30 AS SELECT * FROM '{PEERS.as_posix()}'")
    final = apply_peer_z(con, "feats")
    out = features_path(root)
    tmp = out.with_suffix(".tmp")
    con.execute(f"COPY (SELECT 'prospective' AS split, *, NULL::INTEGER AS account_age_days, "
                f"'{FEATURE_VERSION}' AS feature_version FROM {final} ORDER BY account_id) "
                f"TO '{tmp.as_posix()}' (FORMAT parquet)")
    tmp.replace(out)
    rows, accounts = con.execute(f"SELECT count(*), count(DISTINCT account_id) "
                                 f"FROM '{out.as_posix()}'").fetchone()
    summary = {"month": month, "cutoff": cutoff.isoformat(), "new_accounts_in_map": new,
               "clean": clean, "eligible_rows": rows, "eligible_accounts": accounts}
    (root / "build_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def load_features(path: Path) -> Split:
    """Features as a Split with all-zero labels: nothing about outcomes exists at score time."""
    t = pq.read_table(path)
    X = np.column_stack([t[c].cast(pa.float64()).to_numpy(zero_copy_only=False)
                         for c in FEATURE_COLUMNS])
    n = len(t)
    return Split(name="prospective", X=X, y=np.zeros(n, dtype=np.int8), weight=np.ones(n),
                 account_id=t["account_id"].to_numpy(zero_copy_only=False),
                 cutoff=t["cutoff"].to_numpy(zero_copy_only=False),
                 games=t["games_in_window"].to_numpy(),
                 extra={"rating_at_cutoff": t["rating_at_cutoff"].cast(pa.float64())
                        .to_numpy(zero_copy_only=False)})


def mil_scores(split: Split, bundle: Path, root: Path, batch: int = 4096) -> np.ndarray:
    """The research attention model, exactly as exported for the demo (numpy forward)."""
    from fairplay.deep import bags
    from fairplay.deep.export import ATTENTION_FILE, forward

    w = dict(np.load(bundle / ATTENTION_FILE))
    arrays = bags.build(split, root)
    out = np.empty(len(split.y))
    for s in range(0, len(out), batch):
        out[s:s + batch] = forward(w, arrays["games"][s:s + batch],
                                   arrays["mask"][s:s + batch])[0]
    return out


def score(root: Path = P_ROOT) -> dict:
    import joblib

    from fairplay.demo.peergrid import OUT as BUNDLE
    from fairplay.models.run_tier1 import latest_run

    scores_file = scores_path(root)
    if scores_file.exists():
        raise SystemExit(f"{scores_file} exists: scores are frozen once; delete only with a "
                         "logged reason before any outcome snapshot exists")
    base_dir, tier1_dir = latest_run("baselines"), latest_run("tier1")
    base, tier1 = joblib.load(base_dir / "models.joblib"), joblib.load(tier1_dir / "models.joblib")
    split = load_features(features_path(root))
    scores = {"rule": base["rule"].score(split), "logreg": base["logreg"].score(split),
              "lgbm": tier1["lgbm"].score(split), "mil": mil_scores(split, BUNDLE, root)}
    tmp = scores_file.with_suffix(".tmp")
    pq.write_table(pa.table({"account_id": split.account_id, "cutoff": split.cutoff,
                             "games_in_window": split.games,
                             "rating_at_cutoff": split.extra["rating_at_cutoff"],
                             **{f"score_{m}": scores[m] for m in MODELS}}), tmp)
    tmp.replace(scores_file)
    digest = file_sha256(scores_file)
    built = json.loads((root / "build_summary.json").read_text())
    run = new_run(f"p_freeze_{root.name}", {
        "root": str(root), "month": built["month"], "cutoff": built["cutoff"],
        "baseline_run": base_dir.name, "tier1_run": tier1_dir.name,
        "attention_export": json.loads((BUNDLE / "attention_mil.json").read_text())},
        [features_path(root)])
    conn = store.connect()
    by_id = dict(conn.execute("SELECT account_id, username FROM accounts").fetchall())
    accounts = accounts_path(root)
    accounts.write_text("\n".join(sorted(by_id[a] for a in np.unique(split.account_id))) + "\n")
    metrics = {"scores_sha256": digest, "frozen_at": store.now_utc(), "rows": len(split.y),
               "non_finite": {m: int((~np.isfinite(s)).sum()) for m, s in scores.items()},
               "accounts_file": str(accounts)}
    save_metrics(run, metrics)
    return {"run": run.name} | metrics


def latest_freeze(root: Path) -> dict:
    runs = sorted(RUNS.glob(f"*_p_freeze_{root.name}"))
    if not runs or not (runs[-1] / "metrics.json").exists():
        raise SystemExit("no p_freeze run: run `score` before any P snapshot")
    return json.loads((runs[-1] / "metrics.json").read_text()) | {"run": runs[-1].name}


def check_freeze(root: Path = P_ROOT, index_id: str = "P0") -> dict:
    """Scores on disk must be byte-identical to the frozen hash, and frozen before P0."""
    freeze = latest_freeze(root)
    if file_sha256(scores_path(root)) != freeze["scores_sha256"]:
        raise SystemExit("scores_P.parquet differs from the frozen hash")
    conn = store.connect()
    p0 = conn.execute("SELECT created_at FROM snapshots WHERE snapshot_id = ?",
                      (index_id,)).fetchone()
    if p0 and p0[0] < freeze["frozen_at"]:
        raise SystemExit(f"{index_id} was created before the scores were frozen")
    return {"freeze_run": freeze["run"], "sha256": freeze["scores_sha256"],
            "frozen_at": freeze["frozen_at"], "index_created_at": p0[0] if p0 else None}


def status(conn, snapshot_id: str) -> dict[str, tuple]:
    rows = conn.execute("SELECT account_id, outcome, tos_violation, disabled, fetched_at "
                        "FROM labels WHERE snapshot_id = ?", (snapshot_id,)).fetchall()
    return {r[0]: r[1:] for r in rows}


def outcome_classes(account_ids: np.ndarray, index: dict, outcome: dict
                    ) -> tuple[np.ndarray, dict]:
    """-> (class per row, counts). Classes: excluded_* (not in the population), censored_*
    (in the population, outcome unknown), positive, negative."""
    cls = []
    for a in account_ids:
        i, o = index.get(a), outcome.get(a)
        if i is None or i[0] != "ok":
            cls.append("excluded_no_index_status")
        elif i[1] == 1:
            cls.append("excluded_flagged_at_index")
        elif i[2] == 1:
            cls.append("excluded_closed_at_index")
        elif o is None or o[0] != "ok":
            cls.append("censored_missing_or_renamed")
        elif o[1] == 1:
            cls.append("positive")
        elif o[2] == 1:
            cls.append("censored_closed_without_flag")
        else:
            cls.append("negative")
    cls = np.array(cls)
    names, counts = np.unique(cls, return_counts=True)
    return cls, dict(zip(names.tolist(), counts.tolist(), strict=True))


def evaluate(outcome_id: str, interim: bool, capacity: float, resamples: int, seed: int,
             root: Path = P_ROOT, index_id: str = "P0") -> dict:
    from fairplay.evaluation.bootstrap import bootstrap, compare, summary
    from fairplay.fairness.audit import HIGH_RATED
    from fairplay.models.run_baselines import take

    freeze = check_freeze(root, index_id)
    conn = store.connect()
    index, outcome = status(conn, index_id), status(conn, outcome_id)
    index_time = max(v[3] for v in index.values())
    outcome_time = min(v[3] for v in outcome.values())
    days = (dt.datetime.fromisoformat(outcome_time) - dt.datetime.fromisoformat(index_time)).days
    matured = days >= HORIZON_DAYS - 2
    if not interim and not matured:
        raise SystemExit(f"{outcome_id} is {days} days after {index_id}; the outcome needs "
                         f"{HORIZON_DAYS} ± 2. Use --interim for a descriptive look")
    t = pq.read_table(scores_path(root))
    ids = t["account_id"].to_numpy(zero_copy_only=False)
    cls, counts = outcome_classes(ids, index, outcome)
    split = Split(name="prospective", X=np.zeros((len(ids), 0)),
                  y=(cls == "positive").astype(np.int8), weight=np.ones(len(ids)),
                  account_id=ids, cutoff=t["cutoff"].to_numpy(zero_copy_only=False),
                  games=t["games_in_window"].to_numpy(),
                  extra={"rating_at_cutoff": t["rating_at_cutoff"].to_numpy()})
    raw = {m: t[f"score_{m}"].to_numpy() for m in MODELS}
    finite = np.logical_and.reduce([np.isfinite(s) for s in raw.values()])
    keep = np.isin(cls, ["positive", "negative"]) & finite
    ok, scores = take(split, keep), {m: s[keep] for m, s in raw.items()}
    result = {}
    for policy, capped in (("uncapped", None),
                           ("capped_2200", ok.extra["rating_at_cutoff"] >= HIGH_RATED)):
        res = bootstrap(ok.month(capped), scores, capacity, resamples, seed)
        result[policy] = {
            "summary": summary(res), "k": res["point"]["rule"]["k_per_week"],
            "comparisons": {f"{a}_vs_{b}": compare(res, a, b) for a, b in PRE_REGISTERED}}
    metrics = {"freeze": freeze, "index_snapshot": index_id, "outcome_snapshot": outcome_id,
               "days_after_index": days, "descriptive_only": interim or not matured,
               "population_counts": counts,
               "rows_unscorable_in_population": int((np.isin(cls, ["positive", "negative"])
                                                     & ~finite).sum()),
               "population": int(len(ok.y)), "positives": int(ok.y.sum()),
               "background_rate": float(ok.y.mean()) if len(ok.y) else None,
               "capacity": capacity, **result}
    run = new_run(f"p_eval_{outcome_id}", {"outcome": outcome_id, "index": index_id,
                                           "interim": interim, "capacity": capacity,
                                           "resamples": resamples, "seed": seed,
                                           "root": str(root)}, [scores_path(root)])
    save_metrics(run, metrics)
    return {"run": run.name} | metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=P_ROOT,
                   help="output folder; a rehearsal uses its own (data/processed/p_rehearsal)")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--month", default=MONTH)
    b.add_argument("--cutoff", type=dt.datetime.fromisoformat, default=CUTOFF)
    sub.add_parser("score")
    c = sub.add_parser("check-freeze")
    c.add_argument("--index", default="P0")
    e = sub.add_parser("evaluate")
    e.add_argument("--outcome", required=True)
    e.add_argument("--index", default="P0")
    e.add_argument("--interim", action="store_true")
    e.add_argument("--capacity", type=float, default=0.005)
    e.add_argument("--resamples", type=int, default=1000)
    e.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.cmd == "build":
        out = build(args.month, args.cutoff, args.root)
    elif args.cmd == "score":
        out = score(args.root)
    elif args.cmd == "check-freeze":
        out = check_freeze(args.root, args.index)
    else:
        out = evaluate(args.outcome, args.interim, args.capacity, args.resamples, args.seed,
                       args.root, args.index)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
