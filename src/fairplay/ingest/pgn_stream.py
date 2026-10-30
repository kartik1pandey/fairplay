"""Split a PGN byte stream into games without parsing moves.

Lichess PGN layout per game: header lines `[Tag "value"]`, a blank line, one movetext line,
a blank line. We only look at the first character of each line, so this is far cheaper
than a real PGN parser; move parsing happens later, only for games we keep.
"""

from collections.abc import Iterable, Iterator

Headers = dict[bytes, bytes]


def iter_games(lines: Iterable[bytes]) -> Iterator[tuple[Headers, list[bytes]]]:
    """Yield (headers, raw_lines) for each game; raw_lines is the game's exact original text."""
    headers: Headers = {}
    raw: list[bytes] = []
    in_moves = False
    for line in lines:
        if line.startswith(b"["):
            if in_moves:  # a header after movetext starts the next game
                yield headers, raw
                headers, raw, in_moves = {}, [], False
            space = line.find(b" ")
            headers[line[1:space]] = line[space + 2 : line.rfind(b'"')]
        elif line.strip():
            in_moves = True
        raw.append(line)
    if headers:
        yield headers, raw
