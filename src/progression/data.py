"""Dataset loading and the fixed train / holdout split.

The holdout set is carved off once with a fixed seed and never used for model
selection or tuning. Every model choice is made with cross-validation on the
training split only; the holdout is scored once per candidate, for reporting.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

FEATURES = [
    # name, label, unit
    ("age", "Age", "years"),
    ("sex", "Sex", ""),
    ("bmi", "Body mass index", "kg/m²"),
    ("bp", "Blood pressure", "mmHg"),
    ("s1", "Total cholesterol", "mg/dL"),
    ("s2", "LDL cholesterol", "mg/dL"),
    ("s3", "HDL cholesterol", "mg/dL"),
    ("s4", "Total / HDL ratio", ""),
    ("s5", "Serum triglycerides (log)", ""),
    ("s6", "Blood glucose", "mg/dL"),
]
FEATURE_NAMES = [f[0] for f in FEATURES]

HOLDOUT_FRACTION = 0.2
SPLIT_SEED = 42


@dataclass(frozen=True)
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    @property
    def fingerprint(self) -> str:
        """Short hash of the exact split, stored with every registry entry."""
        h = hashlib.sha256()
        for a in (self.X_train, self.X_test, self.y_train, self.y_test):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        return h.hexdigest()[:12]


def load() -> Dataset:
    """Load the Efron et al. (2004) diabetes data in original clinical units."""
    X, y = load_diabetes(return_X_y=True, scaled=False)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=HOLDOUT_FRACTION, random_state=SPLIT_SEED
    )
    return Dataset(X_tr, X_te, y_tr, y_te)
