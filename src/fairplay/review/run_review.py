"""Blind hand review of 60 S0-flagged + 60 S0-unflagged accounts (design §0.15). LOCAL ONLY.

  sample     pick and shuffle the 120 accounts, assign blind ids R001..R120; the key (id ->
             account, label) goes to private/review/key.json, which reviewers never open
  evidence   per account: last 10 games before its last eligible cutoff, Stockfish match runs,
             timing, the rating-manipulation pattern -> private/review/packets/R###.json
             (resumable: one packet per account); blank forms for two reviewers
  score      reads both forms (+ resolved.csv after discussion): Cohen's kappa, raw and resolved
             shares with Wilson 95% intervals, and the design's gate on engine-like positives

It is a descriptive audit of public evidence, not ground truth. Packets contain moves, so they
stay in private/ like case files; only aggregate numbers leave it.

uv run python -m fairplay.review.run_review sample|evidence|score
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fairplay import duck
from fairplay.review import evidence as ev
from fairplay.runs import new_run, save_metrics
from fairplay.synthetic.score import wilson

OUT = Path("private/review")
TABLE = Path("data/processed/model_table_S0.parquet")
MONTHS = ["2025-06", "2025-07", "2025-08"]
N_PER_CLASS, SEED, N_GAMES = 60, 20261030, 10
LABELS = ("engine_like", "rating_manipulation", "unclear")


def sample() -> dict:
    con = duck.connect()
    acc = con.execute(f"""
        SELECT account_id, max(label) AS label, max(cutoff) AS last_cutoff
        FROM '{TABLE.as_posix()}' GROUP BY account_id ORDER BY account_id""").fetchall()
    rng = np.random.default_rng(SEED)
    chosen = []
    for lab in (1, 0):
        pool = [a for a in acc if a[1] == lab]
        chosen += [pool[i] for i in rng.choice(len(pool), N_PER_CLASS, replace=False)]
    order = rng.permutation(len(chosen))
    key = {f"R{i + 1:03d}": {"account_id": chosen[j][0], "s0_flagged": int(chosen[j][1]),
                             "last_cutoff": str(chosen[j][2])} for i, j in enumerate(order)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "key.json").write_text(json.dumps(key, indent=2))
    return {"accounts": len(key), "flagged": sum(v["s0_flagged"] for v in key.values())}


def last_games(con, account_id: str, cutoff: str) -> list[dict]:
    pg = ", ".join(f"'data/processed/player_games/month={m}.parquet'" for m in MONTHS)
    gm = ", ".join(f"'data/processed/game_moves/month={m}.parquet'" for m in MONTHS)
    rel = con.execute(f"""
        SELECT p.*, m.sans FROM read_parquet([{pg}]) p JOIN read_parquet([{gm}]) m USING (game_key)
        WHERE p.account_id = ? AND p.start_ts < ?::TIMESTAMP - INTERVAL 30 MINUTE
        ORDER BY p.start_ts""", [account_id, cutoff])
    cols = [d[0] for d in rel.description]
    return [dict(zip(cols, r, strict=True)) for r in rel.fetchall()]


def build_packet(games: list[dict], engine) -> dict:
    out = []
    for g in games[-N_GAMES:]:
        steps = ev.engine_steps(list(g["sans"]), g["color"] == "w", engine)
        timing = ev.timing_summary(ev.think_times(list(g["own_clocks"]), g["tc_inc"]))
        out.append({
            "date": str(g["start_ts"].date()), "time_control": f"{g['tc_base']}+{g['tc_inc']}",
            "own_rating": g["own_elo"], "opp_rating": g["opp_elo"], "score": float(g["score"]),
            "termination": g["termination"], "plies": g["n_plies"],
            "longest_non_obvious_match_run": ev.longest_match_run(
                [(s["obvious"], s["matched"]) for s in steps]),
            "non_obvious_moves": sum(not s["obvious"] for s in steps),
            "matched_non_obvious": sum(s["matched"] and not s["obvious"] for s in steps),
            "timing_after_move_10": timing, "moves": list(g["sans"])})
    runs = [x["longest_non_obvious_match_run"] for x in out]
    return {"games": out,
            "games_with_run_ge_8": sum(r >= ev.RUN_LENGTH for r in runs),
            "manipulation": ev.manipulation_pattern(games),
            "rubric": {"engine_like": "run >= 8 in >= 2 games AND even timing in those games "
                                      "(judge from timing_after_move_10)",
                       "rating_manipulation": ">= 5 early losses in 7 days, then +150",
                       "unclear": "neither"}}


def evidence() -> dict:
    from fairplay.synthetic.engines import Stockfish

    key = json.loads((OUT / "key.json").read_text())
    (OUT / "packets").mkdir(parents=True, exist_ok=True)
    con, sf, made = duck.connect(), Stockfish(nodes=100_000), 0
    try:
        for rid, k in key.items():
            out = OUT / "packets" / f"{rid}.json"
            if out.exists():
                continue
            packet = build_packet(last_games(con, k["account_id"], k["last_cutoff"]), sf)
            tmp = out.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"review_id": rid, **packet}, indent=1))
            tmp.replace(out)
            made += 1
            print(f"{rid} done ({made} new)", flush=True)
    finally:
        sf.close()
    for r in ("reviewer1", "reviewer2"):
        form = OUT / f"form_{r}.csv"
        if not form.exists():
            with open(form, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["review_id", "label", "notes"])
                w.writerows([[rid, "", ""] for rid in key])
    return {"packets": len(list((OUT / "packets").glob("R*.json"))), "new": made}


def read_form(path: Path) -> dict[str, str]:
    with open(path, newline="") as fh:
        rows = {r["review_id"]: r["label"].strip() for r in csv.DictReader(fh)}
    bad = {v for v in rows.values() if v not in LABELS}
    if bad:
        raise SystemExit(f"{path.name}: labels must be one of {LABELS}; found {sorted(bad)}")
    return rows


def shares(labels: list[str]) -> dict:
    n = len(labels)
    return {c: {"share": labels.count(c) / n, "wilson_95": wilson(labels.count(c), n)}
            for c in LABELS} | {"n": n}


def score() -> dict:
    key = json.loads((OUT / "key.json").read_text())
    a, b = read_form(OUT / "form_reviewer1.csv"), read_form(OUT / "form_reviewer2.csv")
    ids = sorted(key)
    resolved_path = OUT / "resolved.csv"
    resolved = read_form(resolved_path) if resolved_path.exists() else {
        i: a[i] for i in ids if a[i] == b[i]}
    report = {"kappa": ev.cohen_kappa([a[i] for i in ids], [b[i] for i in ids]),
              "raw_agreement": float(np.mean([a[i] == b[i] for i in ids])),
              "disagreements": sum(a[i] != b[i] for i in ids),
              "resolved_after_discussion": resolved_path.exists()}
    for name, flag in (("s0_flagged", 1), ("s0_unflagged", 0)):
        group = [i for i in ids if key[i]["s0_flagged"] == flag]
        report[name] = {"reviewer1": shares([a[i] for i in group]),
                        "reviewer2": shares([b[i] for i in group]),
                        "resolved": shares([resolved[i] for i in group if i in resolved])}
    pos = report["s0_flagged"]["resolved"]
    report["gate_mixed_flags"] = pos["engine_like"]["share"] < 0.5
    mixed = ("fewer than half of resolved positives are engine-like: all text must describe "
             "the target as mixed moderator flags")
    report["gate_text"] = mixed if report["gate_mixed_flags"] else "engine-like majority"
    run = new_run("hand_review", {"seed": SEED, "n_per_class": N_PER_CLASS}, [])
    save_metrics(run, report)  # aggregates only
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["sample", "evidence", "score"])
    args = p.parse_args()
    fn = {"sample": sample, "evidence": evidence, "score": score}[args.cmd]
    print(json.dumps(fn(), indent=2, default=str))


if __name__ == "__main__":
    main()
