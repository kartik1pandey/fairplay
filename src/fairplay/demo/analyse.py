"""One demo request: PGN text -> per-game table, account features, queue label, case file.

The account features come from the SAME shared SQL as training and the online service
(service/online.compute). The label is relative to the frozen real queue:

- "would enter the review queue": score at or above the frozen threshold, the score of the last
  account admitted to a capped August queue at capacity 0.5% (median over weeks).
- "would not be queued": below it.
- "not enough games": fewer than 10 rated blitz games in the 30 days before the newest game.
  Per-game numbers are still shown, but no account score.

Nothing here says a player cheated: the case file passes the template's language check, and
every response carries the disclaimer. Nothing is stored; names never leave pgn.parse.
"""

import datetime as dt
import json
import secrets
from pathlib import Path

import lightgbm as lgb
import numpy as np

from fairplay.casefiles.template import DISCLAIMER, build_case, check_language
from fairplay.deep.export import ATTENTION_FILE
from fairplay.demo.peergrid import OUT, PeerGrid
from fairplay.demo.pgn import parse
from fairplay.models.data import EXCLUDED_FROM_MODELS, FEATURE_COLUMNS
from fairplay.service import online, store

MIN_GAMES = online.MIN_GAMES


class Scorer:
    def __init__(self, folder: Path = OUT):
        self.folder = folder
        self.booster = lgb.Booster(model_file=str(folder / "model.txt"))
        self.cal = json.loads((folder / "calibration.json").read_text())
        self.b = json.loads((folder / "bundle.json").read_text())
        self.peers = PeerGrid(self.b["peers"], self.b["peer_reference"])
        self.attention = None  # research model; optional, the demo works without it
        if (folder / ATTENTION_FILE).exists():
            from fairplay.demo.attention import Attention

            self.attention = Attention(folder)

    def score(self, row: dict) -> tuple[float, float, float]:
        """-> (raw score, calibrated P(flag at S0), percentile among August account-weeks)."""
        x = np.array([[np.nan if f in EXCLUDED_FROM_MODELS or row.get(f) is None
                       else float(row[f]) for f in FEATURE_COLUMNS]])
        raw = float(self.booster.predict(x, raw_score=True)[0])
        prob = float(np.interp(1 / (1 + np.exp(-raw)), self.cal["x"], self.cal["y"]))
        q = np.array(self.b["score_quantiles"])
        pct = float(100 * np.interp(raw, q, np.linspace(0, 1, len(q))))
        return raw, prob, pct


def think_times(clocks: list[float], inc: int) -> list[float]:
    """Own think times after own move 10 (same base rule as the pipeline: clock before - clock
    after + increment, clipped at 0; the first own move has no prior clock)."""
    t = [max(0.0, a - b + inc) for a, b in zip(clocks[:-1], clocks[1:], strict=False)]
    return t[9:]


def game_summary(r: tuple) -> dict:
    """One stored player row (service/store.player_rows layout) -> a table row for the page."""
    (_acc, key, _v, color, ts, base, inc, own, opp, _diff, score, term, _mate, _kind, plies,
     clocks, valid) = r
    expected = 1 / (1 + 10 ** ((opp - own) / 400))
    out = {"_key": key, "date": str(ts)[:16], "colour": color, "time_control": f"{base}+{inc}",
           "result": score, "own_rating": own, "opponent_gap": opp - own,
           "score_minus_expected": round(score - expected, 3), "moves": plies // 2,
           "termination": term}
    t = np.array(think_times(json.loads(clocks), inc)) if valid else np.array([])
    if len(t) >= 5:
        out |= {"think_median_s": round(float(np.median(t)), 1),
                "think_cv": round(float(t.std(ddof=1) / t.mean()), 2) if t.mean() else None,
                "instant_share": round(float((t < 1).mean()), 2)}
    return out


def score_rows(rows: list[tuple], focus: str, scorer: Scorer, out: dict) -> dict:
    """Stored player rows (any accounts) -> the focus account's games, features, label, case."""
    newest = max(dt.datetime.fromisoformat(str(r[4])) for r in rows if r[0] == focus)
    as_of = newest + dt.timedelta(minutes=31)
    feats = online.compute(rows, [focus], as_of, scorer.folder / "peers_clock30.parquet")
    out["games"] = sorted((game_summary(r) for r in rows if r[0] == focus),
                          key=lambda g: g["date"], reverse=True)
    row = feats.get(focus)
    if scorer.attention is not None and row is not None:
        logit, weights = scorer.attention.per_game(focus, as_of,
                                                   scorer.folder / "peers_clock30.parquet")
        for g in out["games"]:
            w = weights.get(g["_key"])
            g["attention"] = None if w is None else round(w, 3)
        out["research_model"] = {
            "name": "attention over games (July out-of-fold only, not yet tested out of sample)",
            "logit": round(logit, 3),
            "note": "attention = how much this model's score leaned on each of the last 30 "
                    "games; it is not evidence about any single game"}
    for g in out["games"]:
        g.pop("_key", None)
    n_window = int(row["games_in_window"]) if row else 0
    out["games_in_30_day_window"] = n_window
    if n_window < MIN_GAMES:
        out["label"] = (f"not enough games for an account score: {n_window} rated blitz games "
                        f"in the 30 days before the newest game, {MIN_GAMES} needed")
        return out
    raw, prob, pct = scorer.score(row)
    queued = raw >= scorer.b["threshold"]
    out["label"] = ("would enter the review queue" if queued else "would not be queued")
    out["account"] = {
        "score": round(raw, 3), "queue_threshold": round(scorer.b["threshold"], 3),
        "capacity": scorer.b["capacity"],
        "p_flag_at_s0": round(prob, 3),
        "score_percentile": round(pct, 1), "score_reference": scorer.b["score_reference"],
        "features": {f: (None if row.get(f) is None else round(float(row[f]), 4))
                     for f in FEATURE_COLUMNS if f not in EXCLUDED_FROM_MODELS},
    }
    row = dict(row) | {"account_id": "request", "games_in_window": n_window}
    case = build_case(row, prob, scorer.peers, scorer.b["versions"])
    case.pop("account", None)
    case.pop("case_id", None)
    check_language(case)
    out["case_file"] = case
    return out


def analyse(text: str, scorer: Scorer, side: str | None = None) -> dict:
    focus, parsed, unreadable = parse(text, side)
    games = [p for p in parsed if p.game]
    skipped = {}
    for p in parsed:
        if p.problem:
            skipped[p.problem] = skipped.get(p.problem, 0) + 1
    if unreadable:
        skipped["unreadable PGN"] = unreadable
    out = {"disclaimer": DISCLAIMER, "games_read": len(parsed), "games_used": len(games),
           "skipped": skipped, "games": [], "label": None, "account": None}
    if not games:
        out["label"] = "no usable games"
        return out
    secret = secrets.token_hex(16)  # per request: game keys cannot be linked across requests
    rows = [r for p in games
            for r in store.player_rows(p.game, store.game_key(secret, p.game.game_id))]
    return score_rows(rows, str(focus), scorer, out)


def analyse_sample(rows: list[tuple], scorer: Scorer) -> dict:
    """A synthetic player from the bundle (already in stored-row form)."""
    out = {"disclaimer": DISCLAIMER, "games_read": len(rows), "games_used": len(rows),
           "skipped": {}, "games": [], "label": None, "account": None, "synthetic": True}
    return score_rows(rows, rows[0][0], scorer, out)
