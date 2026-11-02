"""Peer percentiles from the demo bundle's aggregate quantile grids (runtime side of bundle.py).

Kept apart from bundle.py so the public demo imports no training code.
"""

from pathlib import Path

import numpy as np

from fairplay.casefiles.peers import REFERENCE, PeerGroup

OUT = Path("demo_bundle")
GRID = np.linspace(0, 1, 201)
TC_GROUPS = ("180+0", "180+2", "300+0", "300+3", "other")


class PeerGrid:
    """Same interface as PeerReference (group / percentile / summary), from the bundle grids,
    so the demo builds case files with the unchanged template."""

    def __init__(self, peers: dict, reference: str = REFERENCE):
        self.peers = peers
        self.reference = reference
        self.bands = sorted({int(k.split("|")[0]) for k in peers})

    def group(self, band: int, tc_group: str) -> PeerGroup:
        band = min(self.bands, key=lambda b: abs(b - band))  # outside the July range: nearest
        tc = tc_group if tc_group in TC_GROUPS else "other"
        key = f"{band}|{tc}"
        if key not in self.peers:  # fail closed: no peers -> insufficient_evidence
            return PeerGroup(f"{band}-{band + 199} blitz {tc}", np.array([key]), 0)
        e = self.peers[key]
        return PeerGroup(e["label"], np.array([key]), e["n_accounts"])

    def _feature(self, group: PeerGroup, feature: str) -> dict | None:
        return self.peers.get(str(group.rows[0]), {}).get("features", {}).get(feature)

    def percentile(self, group: PeerGroup, feature: str, x: float) -> float:
        f = self._feature(group, feature)
        if f is None or np.isnan(x):
            return float("nan")
        q = np.array(f["q"])
        lo = np.searchsorted(q, x, side="left")
        hi = np.searchsorted(q, x, side="right")
        return float(100 * (lo + hi) / 2 / (len(q) - 1)) if hi > lo else \
            float(100 * np.interp(x, q, GRID))

    def summary(self, group: PeerGroup, feature: str) -> dict:
        f = self._feature(group, feature) or {}
        return {k: f.get(k, float("nan")) for k in ("peer_p05", "peer_median", "peer_p95")} | {
            "peer_n": f.get("peer_n", 0)}
