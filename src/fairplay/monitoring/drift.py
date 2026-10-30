"""Weekly drift checks (design §11 monitoring). Diagnostics only: nothing here retrains a model.

PSI = sum over bins of (cur - ref) x ln(cur / ref), with bins = deciles of the REFERENCE
(training month), so each reference bin holds ~10%. Missing values get their own bin: a feature
that suddenly goes missing (a broken clock parser, a changed export) is drift too.
Rule of thumb used by the design: PSI > 0.2 -> write an investigation note. No automatic
retraining: only flag-precision on matured labels can justify that.
"""

import numpy as np

PSI_ALERT = 0.2
BINS = 10
EPS = 1e-4  # empty bins would make ln(0); floor both shares


def psi(reference: np.ndarray, current: np.ndarray, bins: int = BINS) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref_ok, cur_ok = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref_ok) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref_ok, np.linspace(0, 1, bins + 1))[1:-1])  # inner edges

    def shares(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
        counts = np.bincount(np.searchsorted(edges, ok, side="right"), minlength=len(edges) + 1)
        return np.append(counts, len(x) - len(ok)) / len(x)  # last bin = missing

    r, c = np.maximum(shares(ref, ref_ok), EPS), np.maximum(shares(cur, cur_ok), EPS)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(ref_scores: np.ndarray, cur_scores: np.ndarray,
                 ref_features: dict[str, np.ndarray], cur_features: dict[str, np.ndarray],
                 ops: dict | None = None) -> dict:
    """ref_features: the model's top features (by importance) in the training month."""
    feature_psi = {f: psi(ref_features[f], cur_features[f]) for f in ref_features}
    score_psi = psi(ref_scores, cur_scores)
    alerts = ([f"score PSI {score_psi:.3f}"] if score_psi > PSI_ALERT else []) + [
        f"{f} PSI {v:.3f}" for f, v in sorted(feature_psi.items(), key=lambda kv: -kv[1])
        if v > PSI_ALERT]
    return {"score_psi": score_psi, "feature_psi": feature_psi, "alerts": alerts,
            "action": "write an investigation note (no automatic retraining)" if alerts
            else "none", "ops": ops or {}}
