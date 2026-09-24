"""Candidate models, trained in this order and registered as v1, v2, ...

Every candidate runs through the same evaluation and the same promotion gate,
so a later, more complex model only becomes champion if it clearly earns it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0


@dataclass
class Candidate:
    name: str
    family: str  # "linear" or "gbm" / "forest" (what the web exporter understands)
    description: str
    build: Callable[[], RegressorMixin]
    params: dict = field(default_factory=dict)


def _linear(model) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def _tuned_gbm() -> GridSearchCV:
    """Gradient boosting tuned by an inner 5-fold grid search on training data only."""
    grid = {
        "n_estimators": [200, 400],
        "learning_rate": [0.01, 0.03],
        "max_depth": [1, 2],
        "min_samples_leaf": [15],
        "subsample": [0.7],
    }
    return GridSearchCV(
        GradientBoostingRegressor(random_state=SEED),
        grid,
        cv=KFold(5, shuffle=True, random_state=SEED),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )


CANDIDATES: list[Candidate] = [
    Candidate(
        "linear_regression", "linear",
        "Ordinary least squares on standardised features. The baseline every other model must beat.",
        lambda: _linear(LinearRegression()),
    ),
    Candidate(
        "ridge", "linear",
        "Ridge regression with the penalty chosen by internal cross-validation.",
        lambda: _linear(RidgeCV(alphas=[0.1, 0.3, 1, 3, 10, 30, 100])),
    ),
    Candidate(
        "lasso", "linear",
        "Lasso regression; the L1 penalty can drop weak features entirely.",
        lambda: _linear(LassoCV(cv=5, random_state=SEED, max_iter=20000)),
    ),
    Candidate(
        "random_forest", "forest",
        "500-tree random forest. Captures non-linearity but tends to overfit 350 patients.",
        lambda: RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
    ),
    Candidate(
        "gradient_boosting_default", "gbm",
        "Gradient boosting with scikit-learn defaults (depth 3, 100 trees).",
        lambda: GradientBoostingRegressor(random_state=SEED),
    ),
    Candidate(
        "gradient_boosting_tuned", "gbm",
        "Gradient boosting tuned by nested grid search: shallow trees, low learning rate, subsampling.",
        _tuned_gbm,
    ),
]
