"""Header-only pre-filter (design §0 item 2, §2): blitz games involving a sampled account.

Not filtered here, on purpose:
- "Rated": the standard_rated files contain only rated games, and swiss events are named
  "Blitz swiss" without the word "Rated", so an Event check would silently drop rated games.
- BOTs, aborts, short games: removed in the cleaning step, where every drop is counted by reason.
"""

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from fairplay.ingest.pgn_stream import Headers, iter_games

# Lichess estimated duration = base seconds + 40 x increment; blitz is [180, 480).
BLITZ_MIN_SECONDS = 180
BLITZ_MAX_SECONDS = 479
SAMPLE_BUCKETS = 10_000  # sample rate is given in basis points: 100 = 1%, 1000 = 10%
MIN_PLIES = 20  # applied in cleaning (at least 10 moves per side); here with the other scope rules


def parse_time_control(tc: bytes) -> tuple[int, int] | None:
    """'180+2' -> (180, 2). Correspondence ('-') or malformed values -> None."""
    base, plus, inc = tc.partition(b"+")
    if not plus or not base.isdigit() or not inc.isdigit():
        return None
    return int(base), int(inc)


def is_blitz(tc: bytes) -> bool:
    parsed = parse_time_control(tc)
    if parsed is None:
        return False
    base, inc = parsed
    return BLITZ_MIN_SECONDS <= base + 40 * inc <= BLITZ_MAX_SECONDS


def sample_bucket(username: bytes) -> int:
    """Stable bucket in [0, 10000). Lowercased because Lichess usernames are case-insensitive."""
    digest = hashlib.sha256(username.lower()).digest()
    return int.from_bytes(digest[:8], "big") % SAMPLE_BUCKETS


def in_sample(username: bytes, rate_bp: int) -> bool:
    """bucket < rate, so a 1% sample is an exact subset of a 10% sample (samples nest)."""
    return sample_bucket(username) < rate_bp


@dataclass
class FilterStats:
    games_seen: int = 0
    dropped_not_blitz: int = 0
    dropped_not_sampled: int = 0
    kept: int = 0


def keep_game(headers: Headers, rate_bp: int, stats: FilterStats) -> bool:
    stats.games_seen += 1
    if not is_blitz(headers.get(b"TimeControl", b"")):
        stats.dropped_not_blitz += 1
        return False
    if not (
        in_sample(headers.get(b"White", b""), rate_bp)
        or in_sample(headers.get(b"Black", b""), rate_bp)
    ):
        stats.dropped_not_sampled += 1
        return False
    stats.kept += 1
    return True


def filter_games(
    lines: Iterable[bytes],
    rate_bp: int,
    write: Callable[[bytes], object],
    stats: FilterStats | None = None,
) -> FilterStats:
    """Stream games from `lines`, write the raw text of kept games, return counts.

    Pass `stats` to keep partial counts if the stream fails mid-way (e.g. a truncated pilot).
    """
    stats = stats if stats is not None else FilterStats()
    for headers, raw in iter_games(lines):
        if keep_game(headers, rate_bp, stats):
            write(b"".join(raw))
    return stats
