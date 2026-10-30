"""Evidence for the blind hand review (design §0.15). Computes what the rubric looks at; it never
decides. Reviewers judge from these numbers and the moves.

Rubric (design §0.15):
- Engine-like: in >= 2 of the account's last 10 games, a run of >= 8 consecutive own moves after
  move 10 matching the pinned Stockfish top move in positions that are not the only legal or an
  obvious move, with even timing in those games.
- Rating manipulation: >= 5 resignations or time losses before own move 15 within 7 days,
  followed by a rating recovery of >= 150.
- Unclear: neither.

Operational choices (DECISIONS 2026-10-30):
- "Obvious" = the best move is >= 30 win% points ahead of the second (e.g. recapturing a queen),
  or the move is forced. Obvious positions neither extend nor break a run: they say nothing.
- Recovery = highest post-game rating in the 14 days after the window, minus the rating at the
  window's end.
"""

import datetime as dt
import math

import chess
import numpy as np

OBVIOUS_GAP_WIN = 30.0
RUN_LENGTH = 8
EARLY_MOVE = 15
WINDOW_DAYS = 7
EARLY_LOSSES = 5
RECOVERY = 150
RECOVERY_DAYS = 14


def longest_match_run(steps: list[tuple[bool, bool]]) -> int:
    """steps: (obvious, matched) per own move after move 10, in order. Obvious moves are skipped:
    they neither extend nor break a run."""
    best = run = 0
    for obvious, matched in steps:
        if obvious:
            continue
        run = run + 1 if matched else 0
        best = max(best, run)
    return best


def think_times(own_clocks: list[float], inc: float) -> list[float]:
    """Own think times from consecutive own clocks (first move excluded; lag clipped to 0)."""
    return [max(a - b + inc, 0.0) for a, b in zip(own_clocks, own_clocks[1:], strict=False)]


def timing_summary(thinks: list[float], first_own_move: int = 11) -> dict:
    """Think times of own moves after move 10. thinks[i] is own move i + 2 (move 1 has none)."""
    after = [t for i, t in enumerate(thinks) if i + 2 >= first_own_move]
    if len(after) < 5:
        return {"n": len(after), "cv": None, "instant_share": None}
    a = np.asarray(after)
    return {"n": len(after), "cv": float(a.std() / a.mean()) if a.mean() > 0 else None,
            "instant_share": float((a < 1).mean())}


def engine_steps(sans: list[str], subject_white: bool, engine) -> list[dict]:
    """Replays the game; for each own move after move 10 asks the engine for its top two moves."""
    board = chess.Board()
    out = []
    for ply, san in enumerate(sans):
        move = board.parse_san(san)
        own = (ply % 2 == 0) == subject_white
        move_no = ply // 2 + 1
        if own and move_no > 10:
            best, gap = engine.top2_gap(board)
            out.append({"move_no": move_no, "obvious": gap >= OBVIOUS_GAP_WIN,
                        "matched": move == best,
                        "gap": None if math.isinf(gap) else round(gap, 1)})
        board.push(move)
    return out


def is_early_loss(game: dict) -> bool:
    own_moves = (game["n_plies"] + (1 if game["color"] == "w" else 0)) // 2
    lost = float(game["score"]) == 0.0
    resign_or_time = game["termination"] == "time_forfeit" or (
        game["termination"] == "normal" and not game["ended_in_mate"])
    return lost and resign_or_time and own_moves < EARLY_MOVE


def manipulation_pattern(games: list[dict]) -> dict:
    """games sorted by start_ts with keys start_ts, own_elo, own_rating_diff + is_early_loss
    fields. Finds the 7-day window with the most early losses and the recovery after it."""
    ts = [g["start_ts"] for g in games]
    early = [is_early_loss(g) for g in games]
    post = [g["own_elo"] + g["own_rating_diff"] for g in games]
    best = {"max_early_losses_7d": 0, "recovery_after": None, "pattern": False}
    for i in range(len(games)):
        end = ts[i] + dt.timedelta(days=WINDOW_DAYS)
        idx = [j for j in range(i, len(games)) if ts[j] < end]
        n = sum(early[j] for j in idx)
        if n < best["max_early_losses_7d"]:
            continue
        last = idx[-1]
        later = [post[j] for j in range(last + 1, len(games))
                 if ts[j] < ts[last] + dt.timedelta(days=RECOVERY_DAYS)]
        recovery = (max(later) - post[last]) if later else None
        pattern = n >= EARLY_LOSSES and recovery is not None and recovery >= RECOVERY
        if n > best["max_early_losses_7d"] or pattern:
            best = {"max_early_losses_7d": n, "recovery_after": recovery,
                    "pattern": pattern or best["pattern"]}
    return best


def cohen_kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
