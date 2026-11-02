"""PGN text -> GameIn, for the public demo (research add-on; DECISIONS 2026-10-31).

Header rules mirror the offline cleaner (ingest/clean.py) so the demo scores games the way the
models were trained: Event decides rated / arena / swiss, Termination maps to the same four
values, a BOT title or an abandoned game is out of scope (GameIn.skip_reason).

Privacy: player names are read only to find the one player shared by the uploaded games, then
replaced by random UUIDs for this request. Names are never returned, logged or stored.
"""

import datetime as dt
import io
import re
import secrets
import uuid
from collections import Counter
from dataclasses import dataclass

import chess.pgn

from fairplay.service.schema import GameIn, PlayerIn

MAX_GAMES = 60
LICHESS_ID = re.compile(r"lichess\.org/([A-Za-z0-9]{8})")
TERMINATION = {"Normal": "normal", "Time forfeit": "time_forfeit",
               "Rules infraction": "rules_infraction", "Abandoned": "abandoned"}


@dataclass
class Parsed:
    game: GameIn | None
    color: str | None  # side of the focus player: "w" / "b"
    problem: str | None  # why the game cannot be used, in plain words


def read_games(text: str, limit: int = MAX_GAMES) -> list[chess.pgn.Game]:
    games, stream = [], io.StringIO(text)
    while len(games) < limit and (g := chess.pgn.read_game(stream)) is not None:
        if g.errors:
            continue  # unparseable movetext: dropped, counted by the caller
        games.append(g)
    return games


def focus_player(games: list[chess.pgn.Game]) -> str | None:
    """The one player present in every game. None if there is no such player, or if two are
    (games between the same two people): guessing would score the wrong player, so the caller
    asks for a side instead."""
    names = Counter(n.lower() for g in games for n in {g.headers.get("White", "?"),
                                                       g.headers.get("Black", "?")})
    common = [n for n, c in names.items() if c == len(games) and n != "?"]
    return common[0] if len(common) == 1 else None


def _event_kind(event: str) -> str:
    e = event.lower()
    return "swiss" if "swiss" in e else "arena" if "tournament" in e else "game"


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "", "?") else None
    except ValueError:
        return None


def to_game_in(g: chess.pgn.Game, color: str, ids: dict[str, uuid.UUID]) -> Parsed:
    h = g.headers
    site = LICHESS_ID.search(h.get("Site", ""))
    if not site:
        return Parsed(None, None, "not a Lichess game (the model only knows Lichess blitz)")
    tc = h.get("TimeControl", "-")
    if "+" not in tc:
        return Parsed(None, None, "no clock time control (correspondence or unlimited)")
    base, inc = (_int(x) for x in tc.split("+", 1))
    w, b = _int(h.get("WhiteElo")), _int(h.get("BlackElo"))
    if None in (base, inc, w, b):
        return Parsed(None, None, "missing rating or time control")
    try:
        start = dt.datetime.strptime(f"{h['UTCDate']} {h['UTCTime']}", "%Y.%m.%d %H:%M:%S")
    except (KeyError, ValueError):
        return Parsed(None, None, "missing UTCDate/UTCTime")
    sans, clocks, board = [], [], g.board()
    for node in g.mainline():
        sans.append(board.san(node.move))
        board.push(node.move)
        c = node.clock()
        clocks.append(None if c is None else int(round(c)))
    clocks = None if any(c is None for c in clocks) else clocks
    try:
        game = GameIn(
            game_id=site.group(1), start_ts=start.replace(tzinfo=dt.UTC), tc_base=base,
            tc_inc=inc, rated="rated" in h.get("Event", "").lower(),
            event_kind=_event_kind(h.get("Event", "")),
            white=PlayerIn(account_id=ids["w"], rating=w,
                           rating_diff=_int(h.get("WhiteRatingDiff")),
                           is_bot=h.get("WhiteTitle") == "BOT"),
            black=PlayerIn(account_id=ids["b"], rating=b,
                           rating_diff=_int(h.get("BlackRatingDiff")),
                           is_bot=h.get("BlackTitle") == "BOT"),
            result=h.get("Result"), termination=TERMINATION.get(h.get("Termination"), "other"),
            moves=sans, clocks=clocks)
    except ValueError:
        return Parsed(None, None, "header values out of range")
    if reason := game.skip_reason():
        return Parsed(None, None, reason.replace("_", " "))
    return Parsed(game, color, None)


def parse(text: str, side: str | None = None) -> tuple[uuid.UUID, list[Parsed], int]:
    """-> (focus player's request UUID, one Parsed per readable game, unreadable count).

    side = "w"/"b" analyses that colour in every game (one pasted game, or games of mixed
    players); otherwise the player common to all games is found by name."""
    games = read_games(text)
    unreadable = max(0, text.count("[Event ") - len(games))
    focus = uuid.uuid4()
    name = None if side else focus_player(games)
    out = []
    for g in games:
        if side:
            color = side
        elif name:
            color = "w" if g.headers.get("White", "").lower() == name else "b"
        else:
            out.append(Parsed(None, None, "no single player is common to every game; pick a side"))
            continue
        opponent = uuid.UUID(bytes=secrets.token_bytes(16), version=4)
        ids = {color: focus, "b" if color == "w" else "w": opponent}
        out.append(to_game_in(g, color, ids))
    return focus, out, unreadable
