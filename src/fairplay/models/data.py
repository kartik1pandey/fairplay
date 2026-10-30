"""Load one split of the model table as numpy arrays, with features in a fixed order."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from fairplay.evaluation.queue import Month
from fairplay.features.account_features import FEATURES

FEATURE_COLUMNS = list(FEATURES)  # 16 tier-1 features, design §5 order
# Kept in the data (audit groups and context need them) but never seen by a learned model.
# account_age_days: removed by fairness mitigation step 2 (DECISIONS 2026-10-30).
EXCLUDED_FROM_MODELS = ("account_age_days",)


@dataclass
class Split:
    name: str
    X: np.ndarray  # rows x 16, float64, NaN = missing
    y: np.ndarray  # 0/1 label per row
    weight: np.ndarray  # 1 / rows per account
    account_id: np.ndarray
    cutoff: np.ndarray
    games: np.ndarray
    extra: dict[str, np.ndarray] = field(default_factory=dict)  # non-model columns (audit)

    def month(self, capped: np.ndarray | None = None) -> Month:
        return Month.from_rows(self.account_id, self.cutoff, self.games, self.y, capped)

    def column(self, name: str) -> np.ndarray:
        return self.X[:, FEATURE_COLUMNS.index(name)]

    @property
    def model_X(self) -> np.ndarray:
        """X as the models see it: excluded features are a constant NaN column, so LightGBM
        never splits on them and logistic regression's imputer drops them."""
        X = self.X.copy()
        X[:, [FEATURE_COLUMNS.index(f) for f in EXCLUDED_FROM_MODELS]] = np.nan
        return X

    def groups(self) -> np.ndarray:
        """Account ids as integer codes for grouped CV. scikit-learn's shuffled GroupKFold is
        ~138 s per fold on 71k string ids and 0.03 s for all folds on codes, with identical
        folds (codes follow the sorted order of the ids, as sklearn's own np.unique does)."""
        return group_codes(self.account_id)


def group_codes(ids: np.ndarray) -> np.ndarray:
    return np.unique(ids, return_inverse=True)[1]


def load_split(path: Path, split: str, extra: tuple[str, ...] = ()) -> Split:
    t = pq.read_table(path, filters=[("split", "=", split)])
    X = np.column_stack([
        pc.cast(t[c], "float64").to_numpy(zero_copy_only=False) for c in FEATURE_COLUMNS
    ])
    return Split(
        name=split, X=X,
        y=t["label"].to_numpy().astype(np.int8),
        weight=t["weight"].to_numpy(),
        account_id=t["account_id"].to_numpy(zero_copy_only=False),
        cutoff=t["cutoff"].to_numpy(zero_copy_only=False),
        games=t["games_in_window"].to_numpy(),
        extra={c: pc.cast(t[c], "float64").to_numpy(zero_copy_only=False) for c in extra},
    )
