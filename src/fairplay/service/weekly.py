"""Weekly queue job (design §7 queue policy, §0.7, §0.9, §0.18, §11 versions).

At cutoff t (Sunday 00:00 UTC):
1. Features for every account with stored games, by the shared feature SQL at cutoff t.
   Eligible = >= 10 games in the window. Account age is joined from the label snapshot.
2. Score with the frozen model. A non-finite score fails closed: "insufficient data", never
   queued, counted (design §12 failure policy). Every score is stored with its versions.
3. k_w and the 2200+ cap come from the scorable eligible accounts (same as the evaluation).
4. Remove accounts a reviewer is handling or has handled:
     pending   queued earlier, no decision yet            (still in someone's queue)
     clear     cleared within 4 weeks, unless the display score rose by >= 0.1 since
     watch     "re-check in 30 days": back after 30 days
     escalate  with a senior moderator: not re-queued
5. Rank (score, more games, seeded random) and select with the SAME function the offline
   evaluation uses; store the queue once. A queue is immutable: re-running returns the stored one.
Tier 2 does not exist yet, so every queue is tier-1 only (stored as such in the summary).
"""

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from fairplay.casefiles.template import DECISIONS, check_language
from fairplay.evaluation.queue import DEFAULT_CAPACITY, select_week, week_limits
from fairplay.fairness.audit import CAP_MULTIPLIER, HIGH_RATED
from fairplay.features.account_features import apply_peer_z, cutoff_features_from
from fairplay.features.game_features import FEATURE_VERSION, game_features_sql
from fairplay.models.data import FEATURE_COLUMNS, Split
from fairplay.service import store
from fairplay.service.db import Database
from fairplay.service.online import MIN_GAMES, to_arrow

CLEAR_DAYS = 28
WATCH_DAYS = 30
RESCORE_RISE = 0.1


@dataclass(frozen=True)
class Policy:
    capacity: float = DEFAULT_CAPACITY
    cap_multiplier: float = CAP_MULTIPLIER
    seed: int = 0

    @property
    def tag(self) -> str:
        return f"c{self.capacity * 100:g}-m{self.cap_multiplier:g}"


DEFAULT_POLICY = Policy()


def features_at(db: Database, cutoff: dt.datetime, peers: Path) -> list[dict]:
    """All accounts with games in the window, one feature row each (shared SQL)."""
    with db.tx(write=False) as t:
        rows = t.run("SELECT * FROM player_games WHERE start_ts >= ? AND start_ts < ?",
                      (store.iso(cutoff - dt.timedelta(days=store.KEEP_DAYS)),
                       store.iso(cutoff))).all()
    if not rows:
        return []
    con = duckdb.connect()
    con.register("pg_src", to_arrow(rows))
    con.execute(f"CREATE TEMP TABLE gf AS {game_features_sql('pg_src')}")
    con.execute("CREATE TEMP TABLE acc AS SELECT DISTINCT account_id, ?::TIMESTAMP AS cutoff "
                "FROM pg_src", [cutoff])
    rel = cutoff_features_from(con, "gf", "acc", cutoff)
    con.execute(f"CREATE TEMP TABLE f AS {rel.sql_query()}")
    con.execute(f"CREATE TEMP TABLE peers AS SELECT * FROM read_parquet('{peers.as_posix()}')")
    rel = con.sql(f"SELECT * FROM {apply_peer_z(con, 'f', 'peers')} ORDER BY account_id")
    return [dict(zip(rel.columns, r, strict=True)) for r in rel.fetchall()]


def to_split(feats: list[dict], cutoff: dt.datetime, created_at: dict[str, dt.datetime]) -> Split:
    def value(f: dict, name: str) -> float:
        if name == "account_age_days":
            c = created_at.get(f["account_id"])
            return float((cutoff - c).days) if c is not None else np.nan
        v = f[name]
        return np.nan if v is None else float(v)

    n = len(feats)
    return Split(
        name="online", X=np.array([[value(f, c) for c in FEATURE_COLUMNS] for f in feats],
                                  dtype=float).reshape(n, len(FEATURE_COLUMNS)),
        y=np.zeros(n, np.int8), weight=np.ones(n),
        account_id=np.array([f["account_id"] for f in feats], dtype=object),
        cutoff=np.full(n, cutoff), games=np.array([f["games_in_window"] for f in feats]),
        extra={"rating_at_cutoff": np.array([float(f["rating_at_cutoff"]) for f in feats])})


def blocked(db: Database, cutoff: dt.datetime, display: dict[str, float]) -> dict[str, str]:
    """account -> reason it is not re-queued this week (reviewer workflow, design §7)."""
    with db.tx(write=False) as t:
        queued = t.run("""
            SELECT e.account_id, q.cutoff, e.display_score, d.decision, d.decided_at
            FROM queue_entries e JOIN queues q USING (queue_id)
            LEFT JOIN decisions d ON d.queue_id = e.queue_id AND d.account_id = e.account_id
            WHERE q.cutoff < ? ORDER BY q.cutoff
        """, (store.iso(cutoff),)).all()
    out: dict[str, str] = {}
    for acc, _q_cutoff, shown_then, decision, decided_at in queued:  # later entries win
        if decision is None:
            out[acc] = "pending"
            continue
        age = cutoff - dt.datetime.fromisoformat(decided_at)
        if decision == "escalate":
            out[acc] = "escalated"
        elif decision == "watch" and age < dt.timedelta(days=WATCH_DAYS):
            out[acc] = "watch"
        elif (decision == "clear" and age < dt.timedelta(days=CLEAR_DAYS)
              and display.get(acc, -np.inf) < shown_then + RESCORE_RISE):
            out[acc] = "cleared"
        else:
            out.pop(acc, None)
    return out


def run_weekly(db: Database, cutoff: dt.datetime, model, model_version: str, peers: Path,
               created_at: dict[str, dt.datetime], now: dt.datetime,
               policy: Policy = DEFAULT_POLICY) -> dict:
    cutoff = cutoff.replace(tzinfo=None)
    queue_id = f"{cutoff.date()}|{model_version}|{policy.tag}"
    existing = queue(db, queue_id)
    if existing is not None:
        return existing["summary"]  # immutable: never regenerated

    feats = [f for f in features_at(db, cutoff, peers) if f["games_in_window"] >= MIN_GAMES]
    summary = {"queue_id": queue_id, "cutoff": str(cutoff.date()), "tier": "tier1_only",
               "eligible": len(feats)}
    if feats:
        split = to_split(feats, cutoff, created_at)
        score = np.asarray(model.score(split), dtype=float)
        shown = np.asarray(model.probability(split) if hasattr(model, "probability")
                           else score, dtype=float)
        ok = np.isfinite(score) & np.isfinite(shown)
        with db.tx() as t:
            for i, f in enumerate(feats):
                t.run("""INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                         ON CONFLICT DO NOTHING""",
                      (f["account_id"], str(cutoff.date()), model_version, FEATURE_VERSION,
                       float(score[i]) if ok[i] else None, float(shown[i]) if ok[i] else None,
                       int(split.games[i]), store.iso(now)))
        idx = np.flatnonzero(ok)
        capped = split.extra["rating_at_cutoff"] >= HIGH_RATED
        k, cap = week_limits(len(idx), int(capped[idx].sum()), policy.capacity,
                             policy.cap_multiplier)
        reasons = blocked(db, cutoff, {split.account_id[i]: shown[i] for i in idx})
        open_idx = np.array([i for i in idx if split.account_id[i] not in reasons], dtype=np.int64)
        rng = np.random.default_rng(policy.seed)
        tiebreak = rng.random(len(feats))
        order = open_idx[np.lexsort((tiebreak[open_idx], -split.games[open_idx],
                                     -score[open_idx]))] if len(open_idx) else open_idx
        chosen = select_week(order, capped, k, cap)
        summary |= {"scorable": int(ok.sum()), "insufficient_data": int((~ok).sum()), "k": k,
                    "cap_2200": cap, "queued": len(chosen), "queue_fill": len(chosen) / k
                    if k else None, "blocked": {r: list(reasons.values()).count(r)
                                                for r in sorted(set(reasons.values()))},
                    "capped_in_queue": int(capped[chosen].sum()) if len(chosen) else 0}
        entries = [(queue_id, pos, split.account_id[i], float(score[i]), float(shown[i]),
                    int(split.games[i])) for pos, i in enumerate(chosen, 1)]
    else:
        summary |= {"scorable": 0, "insufficient_data": 0, "k": 0, "queued": 0}
        entries = []
    with db.tx() as t:
        stored_now = t.run("""
            INSERT INTO queues VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (queue_id) DO NOTHING RETURNING queue_id
        """, (queue_id, str(cutoff.date()), model_version, FEATURE_VERSION, policy.tag,
              store.iso(cutoff), store.iso(now), json.dumps(summary))).one()
        if stored_now is None:  # a concurrent run stored it first: keep theirs
            return queue(db, queue_id)["summary"]
        for e in entries:
            t.run("INSERT INTO queue_entries VALUES (?, ?, ?, ?, ?, ?)", e)
    return summary


def queue(db: Database, queue_id: str) -> dict | None:
    with db.tx(write=False) as t:
        q = t.run("SELECT summary FROM queues WHERE queue_id = ?", (queue_id,)).one()
        if q is None:
            return None
        rows = t.run("SELECT position, account_id, score, display_score, games FROM queue_entries"
                     " WHERE queue_id = ? ORDER BY position", (queue_id,)).all()
    return {"summary": json.loads(q[0]),
            "entries": [dict(zip(("position", "account_id", "score", "display_score", "games"),
                                 r, strict=True)) for r in rows]}


def latest_queue(db: Database, cutoff: str) -> dict | None:
    with db.tx(write=False) as t:
        q = t.run("SELECT queue_id FROM queues WHERE cutoff = ? ORDER BY created_at DESC",
                  (cutoff,)).one()
    return queue(db, q[0]) if q else None


def record_decision(db: Database, queue_id: str, account_id: str, decision: str, reason: str,
                    now: dt.datetime) -> None:
    """Stored apart from scores (design §11). One decision per queue entry; wording checked."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    if not reason.strip():
        raise ValueError("a reason is required")
    check_language({"reason": reason})
    with db.tx() as t:
        if t.run("SELECT 1 FROM queue_entries WHERE queue_id = ? AND account_id = ?",
                 (queue_id, account_id)).one() is None:
            raise KeyError("account is not in that queue")
        t.run("INSERT INTO decisions VALUES (?, ?, ?, ?, ?)",
              (queue_id, account_id, decision, reason.strip(), store.iso(now)))


def scores_table(db: Database, cutoff: str, model_version: str) -> pa.Table:
    with db.tx(write=False) as t:
        rows = t.run("SELECT account_id, score FROM scores WHERE cutoff = ? AND model_version = ?",
                     (cutoff, model_version)).all()
    return pa.table({"account_id": [r[0] for r in rows], "score": [r[1] for r in rows]})
