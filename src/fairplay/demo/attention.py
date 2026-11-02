"""Per-game attention weights for the demo, from the exported research model (deep/export.py).

The bag is built by the SAME query as training (deep/bags.bag_sql) over the request's games,
already loaded by online.compute as table `gf` on this thread's DuckDB connection. The weight
says how much the research model's score leaned on each game; it is not proof about any game.
The queue label still comes from the tested LightGBM model.
"""

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

from fairplay.deep.bags import GAME_FEATURES, bag_sql
from fairplay.deep.export import ATTENTION_FILE, forward
from fairplay.service import online


class Attention:
    def __init__(self, folder: Path):
        self.weights = dict(np.load(folder / ATTENTION_FILE))
        self.meta = json.loads((folder / "attention_mil.json").read_text())
        if self.meta["game_features"] != list(GAME_FEATURES):
            raise ValueError("attention model was exported with other game features")

    def per_game(self, focus: str, as_of: dt.datetime, peers: Path) -> tuple[float, dict]:
        """-> (research-model logit, {game_key: attention weight}) for the focus account."""
        con = online._connection(peers)  # holds `gf` from the online.compute just run
        con.register("rows", pa.table({"account_id": [focus], "cutoff": [as_of]}))
        try:
            t = con.sql(bag_sql("gf", as_of)).fetchall()
        finally:
            con.unregister("rows")
        n = len(GAME_FEATURES)
        games = np.zeros((1, 30, n), dtype=np.float32)
        mask = np.zeros((1, 30), dtype=bool)
        keys = {}
        for r in t:
            slot = r[2] - 1
            games[0, slot] = r[3:3 + n]
            mask[0, slot] = True
            keys[slot] = r[1]
        logit, att = forward(self.weights, games, mask)
        return float(logit[0]), {keys[s]: float(att[0, s]) for s in keys}
