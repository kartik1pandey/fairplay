"""Realism check (design §0 item 12, §6): synthetic controls vs real unflagged accounts.

Per feature and rating band: two-sample KS statistic and mean difference. Features with KS > 0.2
are listed; synthetic claims must not rely on them. Done BEFORE trusting any synthetic result.

Minimum 150 accounts per side: the 5% critical KS for two samples of n is ~1.36 x sqrt(2 / n),
which is 0.25 at n = 60 and 0.16 at n = 150. Below that, sampling noise alone exceeds the 0.2
limit and the check would flag identical distributions (found by a test).
"""

import numpy as np
from scipy.stats import ks_2samp

KS_LIMIT = 0.2
MIN_N = 150


def realism_table(synth: dict[str, np.ndarray], real: dict[str, np.ndarray],
                  synth_band: np.ndarray, real_band: np.ndarray, min_n: int = MIN_N) -> dict:
    rows, unreliable = [], set()
    for feature in synth:
        for band in np.unique(synth_band):
            s = synth[feature][(synth_band == band) & ~np.isnan(synth[feature])]
            r = real[feature][(real_band == band) & ~np.isnan(real[feature])]
            if len(s) < min_n or len(r) < min_n:
                rows.append({"feature": feature, "band": float(band), "n_synth": len(s),
                             "n_real": len(r), "ks": None, "mean_diff": None})
                continue
            ks = float(ks_2samp(s, r).statistic)
            rows.append({"feature": feature, "band": float(band), "n_synth": len(s),
                         "n_real": len(r), "ks": ks, "mean_diff": float(s.mean() - r.mean())})
            if ks > KS_LIMIT:
                unreliable.add(feature)
    return {"rows": rows, "features_ks_above_limit": sorted(unreliable), "limit": KS_LIMIT}
