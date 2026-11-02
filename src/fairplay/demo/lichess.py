"""Game links -> PGN with clocks, for the public demo.

Only GAME links are accepted (lichess.org/<8-char id>, with or without /white, /black or a move
anchor). Profile links (lichess.org/@/<name>) are refused: the demo never looks an account up
by name (design §9). Up to MAX_IDS games go out in ONE request to the export-by-ids endpoint;
Lichess asks for one request at a time and a full minute's pause after HTTP 429, so a 429 is
reported to the user instead of retried.
"""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # httpx is imported only when fetching: the browser demo fetches in JS
    import httpx

MAX_IDS = 60
EXPORT = "https://lichess.org/api/games/export/_ids"
GAME_LINK = re.compile(r"lichess\.org/([A-Za-z0-9]{8})(?:[A-Za-z0-9]{4})?(?![A-Za-z0-9])")
PROFILE_LINK = re.compile(r"lichess\.org/@/")
EXPORT_PARAMS = {"clocks": "true", "evals": "false", "opening": "false"}


class FetchError(Exception):
    pass


def game_ids(text: str) -> list[str]:
    if PROFILE_LINK.search(text):
        raise FetchError("profile links are not accepted; paste links to individual games")
    ids = list(dict.fromkeys(GAME_LINK.findall(text)))  # unique, in order
    if not ids:
        raise FetchError("no Lichess game links found")
    if len(ids) > MAX_IDS:
        raise FetchError(f"at most {MAX_IDS} games per request")
    return ids


def fetch_pgn(ids: list[str], client: "httpx.Client | None" = None) -> str:
    import httpx

    client = client or httpx.Client(timeout=30)
    r = client.post(EXPORT, content=",".join(ids), params=EXPORT_PARAMS,
                    headers={"Accept": "application/x-chess-pgn"})
    if r.status_code == 429:
        raise FetchError("Lichess is rate-limiting requests; wait one minute and try again")
    if r.status_code != 200:
        raise FetchError(f"Lichess returned HTTP {r.status_code}")
    return r.text
