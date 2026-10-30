"""Peer reference for case files (design §0 item 17, §10).

Peer group = rating band x time-control group from the TRAINING month only, frozen. A cell with
fewer than 200 accounts widens to neighbouring bands (same time control), +-200 at a time, until it
has 200; if the whole time-control group is still too small the case is insufficient_evidence.
Percentiles are account-weighted (each account's July rows sum to 1, like the training weights),
using mid-ranks so ties sit in the middle: pct = 100 x (W(< x) + W(= x) / 2) / W.
"""

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np

from fairplay.features.account_features import FEATURES

PEER_MIN = 200
BAND_WIDTH = 200
REFERENCE = "July 2025"


@dataclass(frozen=True)
class PeerGroup:
    label: str
    rows: np.ndarray  # indices into the reference arrays
    n_accounts: int

    @property
    def sufficient(self) -> bool:
        return self.n_accounts >= PEER_MIN


class PeerReference:
    def __init__(self, columns: dict[str, np.ndarray], reference: str = REFERENCE):
        self.c = columns
        self.reference = reference
        self._cache: dict[tuple[int, str], PeerGroup] = {}

    @classmethod
    def from_model_table(cls, path: Path, split: str = "train") -> "PeerReference":
        cols = ["account_id", "rating_band", "tc_group", "weight", *FEATURES]
        rel = duckdb.connect().execute(
            f"SELECT {', '.join(cols)} FROM read_parquet('{path.as_posix()}') WHERE split = ?",
            [split])
        data = rel.fetchnumpy()
        return cls({k: np.asarray(v) for k, v in data.items()})

    def group(self, band: int, tc_group: str) -> PeerGroup:
        key = (int(band), tc_group)
        if key not in self._cache:
            same_tc = self.c["tc_group"] == tc_group
            bands = self.c["rating_band"]
            span = int(np.abs(bands[same_tc] - band).max()) if same_tc.any() else 0
            for radius in range(0, span + BAND_WIDTH, BAND_WIDTH):
                rows = np.flatnonzero(same_tc & (np.abs(bands - band) <= radius))
                n = len(np.unique(self.c["account_id"][rows]))
                if n >= PEER_MIN:
                    break
            lo = int(bands[rows].min()) if len(rows) else band  # bands actually present
            hi = (int(bands[rows].max()) if len(rows) else band) + BAND_WIDTH - 1
            self._cache[key] = PeerGroup(f"{lo}-{hi} blitz {tc_group}", rows, n)
        return self._cache[key]

    def _values(self, group: PeerGroup, feature: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        v = self.c[feature][group.rows].astype(float)
        w = self.c["weight"][group.rows].astype(float)
        ok = ~np.isnan(v)
        return v[ok], w[ok], self.c["account_id"][group.rows][ok]

    def percentile(self, group: PeerGroup, feature: str, x: float) -> float:
        v, w, _ = self._values(group, feature)
        if not len(v) or np.isnan(x):
            return float("nan")
        return float(100 * (w[v < x].sum() + w[v == x].sum() / 2) / w.sum())

    def summary(self, group: PeerGroup, feature: str) -> dict:
        """Weighted p05 / median / p95 and the number of peer accounts with a value."""
        v, w, acc = self._values(group, feature)
        order = np.argsort(v)
        v, w = v[order], w[order]
        cdf = (np.cumsum(w) - w / 2) / w.sum()
        p05, p50, p95 = np.interp([0.05, 0.5, 0.95], cdf, v)
        return {"peer_p05": float(p05), "peer_median": float(p50), "peer_p95": float(p95),
                "peer_n": int(len(np.unique(acc)))}
