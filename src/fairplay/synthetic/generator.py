"""Generate whole synthetic accounts: 30 Maia-vs-Maia games each, optional engine substitution.

Games are played forward on a real board, so every move is legal by construction (asserted).
Per own move of the synthetic account (the "subject"):
  - own moves 1-10 are never substituted (opening book territory, design §6);
  - strategy A/B: substitute with probability p; strategy C: only in critical positions (best move
    >= 9.1 win% points ahead of the second = 100 cp at equality), keeping the share at or below p;
  - think time: human sampler, or the strategy's engine-move style for substituted moves.
Clocks: clock_after = clock_before - think + increment, floored to whole seconds like Lichess %clk;
running out of time loses (draw if the opponent has insufficient material, as on Lichess).
After each game the subject's rating is updated with Glicko-2; opponents are drawn within +-150.
Output rows use the player_games schema, so the real feature code runs on them unchanged.
"""

import datetime as dt
import math
from dataclasses import dataclass

import chess
import numpy as np

from fairplay.synthetic import glicko2
from fairplay.synthetic.engines import (
    CRITICAL_GAP_WIN,
    EngineModel,
    HumanModel,
    maia_for_play,
    sample_move,
)
from fairplay.synthetic.timing import ThinkSampler

MAX_PLIES = 200  # adjudicated draw after 100 moves each
OPENING_MOVES = 10
OPPONENT_RANGE = 150
GAMES_PER_ACCOUNT = 30


@dataclass(frozen=True)
class Strategy:
    name: str
    p: float
    select: str  # "none" | "random" | "critical"
    timing: str  # "human" | "instant" | "constant"


STRATEGIES = {
    "control": Strategy("control", 0.0, "none", "human"),
    "A": Strategy("A", 0.5, "random", "instant"),
    "B": Strategy("B", 0.2, "random", "constant"),
    "C": Strategy("C", 0.1, "critical", "human"),  # held out: generated and scored last
}
CONSTANT_DELAY = 3.0


def engine_think(strategy: Strategy, rng, sampler, rating, move_no, clock, duration) -> float:
    if strategy.timing == "instant":
        return float(rng.uniform(0.1, 0.9))
    if strategy.timing == "constant":
        return CONSTANT_DELAY
    return sampler.sample(rng, rating, move_no, clock, duration)


def play_game(rng: np.random.Generator, human: HumanModel, engine: EngineModel,
              sampler: ThinkSampler, subject_rating: float, opp_rating: float,
              subject_white: bool, tc: tuple[int, int], strategy: Strategy) -> dict:
    base, inc = tc
    duration = base + 40 * inc
    board = chess.Board()
    engine.new_game()
    clocks = {chess.WHITE: float(base), chess.BLACK: float(base)}
    subject = chess.WHITE if subject_white else chess.BLACK
    ratings = {subject: subject_rating, not subject: opp_rating}
    maia = {c: maia_for_play(r) for c, r in ratings.items()}
    own_clocks, own_move_no, n_after10, n_sub, flagged = [], 0, 0, 0, None
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        side = board.turn
        move_no = board.fullmove_number
        substituted = False
        move = None
        candidate = side == subject and own_move_no + 1 > OPENING_MOVES  # counts once played
        if candidate and strategy.select != "none":
            if strategy.select == "random" and rng.random() < strategy.p:
                move, substituted = engine.best(board), True
            elif strategy.select == "critical" and n_sub < strategy.p * (n_after10 + 1):
                best, gap = engine.top2_gap(board)
                if gap >= CRITICAL_GAP_WIN:
                    move, substituted = best, True
        if move is None:
            move = sample_move(human.policy(board, maia[side]), rng)
        assert move in board.legal_moves  # legality by construction
        if move_no == 1:
            think = 0.0  # Lichess clocks start after each side's first move
        elif substituted:
            think = engine_think(strategy, rng, sampler, ratings[side], move_no, clocks[side],
                                 duration)
        else:
            think = sampler.sample(rng, ratings[side], move_no, clocks[side], duration)
        if think >= clocks[side]:
            flagged = side
            break
        clocks[side] = math.floor(clocks[side] - think + (inc if move_no > 1 else 0))
        board.push(move)
        n_sub += substituted
        if side == subject:
            own_move_no += 1
            n_after10 += candidate and strategy.select != "none"
            own_clocks.append(float(clocks[side]))

    if flagged is not None:
        winner = not flagged
        insufficient = board.has_insufficient_material(winner)
        score = 0.5 if insufficient else (1.0 if winner == subject else 0.0)
        termination = "time_forfeit"
        mate = False
    else:
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            score = 0.5
        else:
            score = 1.0 if outcome.winner == subject else 0.0
        termination = "normal"
        mate = outcome is not None and outcome.termination == chess.Termination.CHECKMATE
    return {"n_plies": board.ply(), "own_clocks": own_clocks, "score": score,
            "termination": termination, "ended_in_mate": mate,
            "n_own_after10": n_after10, "n_substituted": n_sub}


def generate_account(account_id: str, strategy: Strategy, rating: float, account_age_days: int,
                     tcs: list[tuple[tuple[int, int], float]], cutoff: dt.datetime, seed: int,
                     human: HumanModel, engine: EngineModel, sampler: ThinkSampler,
                     n_games: int = GAMES_PER_ACCOUNT) -> tuple[list[dict], dict]:
    """Returns (player_games rows, account summary). tcs = [(time control, probability)]."""
    rng = np.random.default_rng(seed)
    r, rd, vol = float(rating), 60.0, 0.06
    start = cutoff - dt.timedelta(days=28)
    times = sorted(start + dt.timedelta(seconds=float(s)) for s in rng.uniform(0, 27 * 86400,
                                                                                n_games))
    probs = np.array([p for _, p in tcs])
    rows, subs, after10 = [], 0, 0
    for i, t in enumerate(times):
        tc = tcs[rng.choice(len(tcs), p=probs / probs.sum())][0]
        opp = float(rng.uniform(r - OPPONENT_RANGE, r + OPPONENT_RANGE))
        white = bool(rng.random() < 0.5)
        g = play_game(rng, human, engine, sampler, r, opp, white, tc, strategy)
        new_r, rd, vol = glicko2.update(r, rd, vol, [(opp, 60.0, g["score"])])
        rows.append({
            "account_id": account_id, "game_key": f"{account_id}-{i:02d}", "month": "synthetic",
            "color": "w" if white else "b", "start_ts": t, "tc_base": tc[0], "tc_inc": tc[1],
            "own_elo": int(round(r)), "opp_elo": int(round(opp)),
            "own_rating_diff": int(round(new_r)) - int(round(r)), "score": g["score"],
            "termination": g["termination"], "ended_in_mate": g["ended_in_mate"],
            "event_kind": "game", "n_plies": g["n_plies"], "own_clocks": g["own_clocks"],
            "clock_valid": True, "is_synthetic": True,
        })
        r = new_r
        subs += g["n_substituted"]
        after10 += g["n_own_after10"]
    summary = {"account_id": account_id, "strategy": strategy.name, "p": strategy.p,
               "start_rating": rating, "end_rating": r, "account_age_days": account_age_days,
               "achieved_share": subs / after10 if after10 else 0.0, "seed": seed}
    return rows, summary
