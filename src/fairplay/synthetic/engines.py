"""Move sources for the generator, behind two small interfaces so tests can use fakes.

Human-like moves: Maia-1 networks (CSSLab, ratings 1100-1900 in steps of 100) run in lc0 with
nodes=1, i.e. the raw policy head. We read the policy (lc0 --verbose-move-stats) and SAMPLE a move
with our own seeded RNG: taking the top move would replay identical games, and lc0's internal
temperature cannot be seeded.

Engine moves: Stockfish, single thread, fixed node limit, hash cleared per game -> deterministic.

Paths (override with environment variables):
  FAIRPLAY_LC0        D:/tools/lc0/lc0.exe
  FAIRPLAY_MAIA_DIR   D:/tools/maia            (maia-1100.pb.gz ... maia-1900.pb.gz)
  FAIRPLAY_STOCKFISH  D:/tools/stockfish/stockfish.exe
"""

import hashlib
import math
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Protocol

import chess
import chess.engine
import numpy as np

MAIA_RATINGS = tuple(range(1100, 2000, 100))
MAX_LIVE = 4  # lc0 processes kept open per worker
MATE_CAP_CP = 1000
WIN_PERCENT_K = 0.00368208  # Lichess accuracy scale: win% = 50 + 50 (2 / (1 + e^(-k cp)) - 1)


def win_percent(cp: float) -> float:
    """Expected score (0-100) for the side to move; cp clipped to +-1000 like Lichess."""
    cp = max(-MATE_CAP_CP, min(MATE_CAP_CP, cp))
    return 50 + 50 * (2 / (1 + math.exp(-WIN_PERCENT_K * cp)) - 1)


# "Critical" = best move >= 100 cp better than the second, measured where 100 cp matters: at
# equality. On the win% scale that is a fixed 9.1-point gap, so a 300-cp gap at +2000 (any move
# still wins) is not critical. Derived from the design's 100 cp, not tuned (DECISIONS.md).
CRITICAL_GAP_WIN = win_percent(0) - win_percent(-100)
POLICY_LINE = re.compile(r"^(\S+)\s+\(\s*\d+\s*\)\s+N:.*?\(P:\s*([\d.]+)%\)")


def tool_path(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest_maia_rating(rating: float) -> int | None:
    """Nearest supported Maia rating; None outside the supported range (account not generated)."""
    if rating < MAIA_RATINGS[0] - 50 or rating >= MAIA_RATINGS[-1] + 50:
        return None
    return min(MAIA_RATINGS, key=lambda r: abs(r - rating))


def maia_for_play(rating: float) -> int:
    """Model used for one game. The range rule above selects ACCOUNTS by starting rating; during
    play, opponents (drawn within +-150) and drifting ratings (engine users climb) can leave the
    range, and then play the nearest edge model instead of failing."""
    return min(MAIA_RATINGS, key=lambda r: abs(r - rating))


def parse_policy(info_strings: list[str], board: chess.Board) -> dict[chess.Move, float]:
    """lc0 verbose move stats -> {legal move: prior probability}."""
    out = {}
    for s in info_strings:
        m = POLICY_LINE.match(s.strip())
        if m and m.group(1) != "node":
            move = chess.Move.from_uci(m.group(1))
            if move in board.legal_moves:
                out[move] = float(m.group(2)) / 100
    return out


class HumanModel(Protocol):
    def policy(self, board: chess.Board, rating: int) -> dict[chess.Move, float]: ...


class EngineModel(Protocol):
    def best(self, board: chess.Board) -> chess.Move: ...
    def top2_gap(self, board: chess.Board) -> tuple[chess.Move, float]: ...
    def new_game(self) -> None: ...


def sample_move(policy: dict[chess.Move, float], rng: np.random.Generator) -> chess.Move:
    moves = sorted(policy, key=lambda m: m.uci())  # fixed order -> reproducible with the seed
    p = np.array([policy[m] for m in moves])
    return moves[rng.choice(len(moves), p=p / p.sum())]


class Maia:
    def __init__(self, lc0: Path | None = None, weights_dir: Path | None = None):
        self.lc0 = lc0 or tool_path("FAIRPLAY_LC0", "D:/tools/lc0/lc0.exe")
        self.weights_dir = weights_dir or tool_path("FAIRPLAY_MAIA_DIR", "D:/tools/maia")
        self._engines: OrderedDict[int, chess.engine.SimpleEngine] = OrderedDict()

    def weights(self, rating: int) -> Path:
        return self.weights_dir / f"maia-{rating}.pb.gz"

    def _engine(self, rating: int) -> chess.engine.SimpleEngine:
        """At most MAX_LIVE lc0 processes per worker (~100 MB each); the least recently used is
        closed first. One account spans <= 4 models (its rating +-150, plus drift). With nodes=1
        the output depends only on weights and position, so reopening changes no move."""
        if rating in self._engines:
            self._engines.move_to_end(rating)
        else:
            while len(self._engines) >= MAX_LIVE:
                self._engines.popitem(last=False)[1].quit()
            self._engines[rating] = chess.engine.SimpleEngine.popen_uci(
                [str(self.lc0), f"--weights={self.weights(rating)}", "--verbose-move-stats",
                 "--threads=1"])
        return self._engines[rating]

    def policy(self, board: chess.Board, rating: int) -> dict[chess.Move, float]:
        strings = []
        with self._engine(rating).analysis(board, chess.engine.Limit(nodes=1)) as analysis:
            for info in analysis:
                if "string" in info:
                    strings.append(info["string"])
        return parse_policy(strings, board) or {m: 1.0 for m in board.legal_moves}

    def close(self) -> None:
        for e in self._engines.values():
            e.quit()


class Stockfish:
    def __init__(self, path: Path | None = None, nodes: int = 100_000):
        self.path = path or tool_path("FAIRPLAY_STOCKFISH", "D:/tools/stockfish/stockfish.exe")
        self.nodes = nodes
        self.engine = chess.engine.SimpleEngine.popen_uci(str(self.path))
        self.engine.configure({"Threads": 1, "Hash": 16})
        self._game = object()

    @property
    def version(self) -> str:
        return self.engine.id.get("name", "unknown")

    def new_game(self) -> None:
        self._game = object()  # python-chess sends ucinewgame when the game key changes

    def best(self, board: chess.Board) -> chess.Move:
        return self.engine.play(board, chess.engine.Limit(nodes=self.nodes), game=self._game).move

    def top2(self, board: chess.Board) -> tuple[chess.Move, float, float | None]:
        """Best move, its score and the second-best score (side to move, cp, mates capped).
        Second score is None for a forced move."""
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("top2 on a finished position")
        if len(legal) == 1:
            return legal[0], 0.0, None  # forced: no engine call needed
        infos = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes), multipv=2,
                                    game=self._game)
        scored = sorted(((float(i["score"].pov(board.turn).score(mate_score=MATE_CAP_CP)),
                          i["pv"][0]) for i in infos if i.get("pv")),
                        key=lambda t: -t[0])  # multipv lines can arrive out of score order
        return scored[0][1], scored[0][0], scored[1][0]

    def top2_gap(self, board: chess.Board) -> tuple[chess.Move, float]:
        """Best move and its advantage over the second-best in win% points (inf if forced)."""
        best, s1, s2 = self.top2(board)
        return best, float("inf") if s2 is None else win_percent(s1) - win_percent(s2)

    def close(self) -> None:
        self.engine.quit()
