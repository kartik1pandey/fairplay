"""Entry points for the in-browser demo (Pyodide on a static Hugging Face Space).

The page (demo/static/index.html) loads this package and the aggregate-only bundle into the
browser and calls these functions. They run the same code as the Gradio app: pgn.parse, the
shared feature SQL (service/online.compute) and the frozen LightGBM. Each returns a JSON string,
so the page never needs Python objects. Fetching Lichess games happens in JavaScript (Lichess
allows cross-origin requests), so no HTTP library is needed here.
"""

import json
import random
from pathlib import Path

from fairplay.demo.analyse import Scorer, analyse, analyse_sample
from fairplay.demo.lichess import EXPORT, EXPORT_PARAMS, FetchError, game_ids

NOTES = {"control": "plays like the human-move model, no engine moves",
         "A": "half of its moves after move 10 from Stockfish, played instantly",
         "B": "a fifth of its moves from Stockfish, with a constant delay",
         "C": "a tenth of its moves from Stockfish, only in critical positions, human timing"}

_scorer: Scorer | None = None


def scorer() -> Scorer:
    global _scorer
    if _scorer is None:
        _scorer = Scorer(Path("demo_bundle"))
    return _scorer


def ids_request(links: str) -> str:
    """-> {"url": ..., "body": "id1,id2"} for the page to POST, or {"error": ...}."""
    try:
        ids = game_ids(links or "")
    except FetchError as e:
        return json.dumps({"error": str(e)})
    query = "&".join(f"{k}={v}" for k, v in EXPORT_PARAMS.items())
    return json.dumps({"url": f"{EXPORT}?{query}", "body": ",".join(ids)})


def run_pgn(text: str, side: str = "") -> str:
    if not text or not text.strip():
        return json.dumps({"error": "paste at least one PGN game"})
    return json.dumps(analyse(text, scorer(), side or None), default=str)


def run_sample(strategy: str) -> str:
    pool = [s for s in scorer().b["samples"] if s["strategy"] == strategy]
    s = random.choice(pool)
    r = analyse_sample([tuple(t) for t in s["rows"]], scorer())
    r |= {"synthetic": True, "strategy": strategy, "strategy_note": NOTES[strategy]}
    return json.dumps(r, default=str)
