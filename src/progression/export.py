"""Export the champion to a JSON payload the static web page executes.

Only plain numbers go into the payload: coefficients for linear models, or
every tree's split feature, threshold, child pointers and node values for
gradient boosting. The page re-runs the model locally with no backend.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .data import FEATURES, Dataset

N_EXAMPLES = 8


def _unwrap(est):
    return est.best_estimator_ if isinstance(est, GridSearchCV) else est


def export_linear(pipe: Pipeline, X_train: np.ndarray) -> dict:
    scaler, lin = pipe.named_steps["scale"], pipe.named_steps["model"]
    coef = lin.coef_ / scaler.scale_                      # fold standardisation into raw units
    intercept = float(lin.intercept_ - np.sum(lin.coef_ * scaler.mean_ / scaler.scale_))
    return {"family": "linear", "intercept": intercept, "coef": coef.tolist(),
            "center": X_train.mean(axis=0).tolist()}


def export_gbm(gbm: GradientBoostingRegressor) -> dict:
    init = float(np.ravel(gbm.init_.predict(np.zeros((1, gbm.n_features_in_))))[0])
    trees = []
    for est in gbm.estimators_[:, 0]:
        t = est.tree_
        trees.append({
            "feature": t.feature.tolist(),
            "threshold": [float(v) if f >= 0 else 0.0 for v, f in zip(t.threshold, t.feature)],
            "left": t.children_left.tolist(),
            "right": t.children_right.tolist(),
            "value": [float(v) for v in t.value[:, 0, 0]],
        })
    return {"family": "gbm", "init": init, "learning_rate": float(gbm.learning_rate), "trees": trees}


def export_model(estimator, X_train) -> dict:
    est = _unwrap(estimator)
    if isinstance(est, Pipeline):
        return export_linear(est, X_train)
    if isinstance(est, GradientBoostingRegressor):
        return export_gbm(est)
    raise TypeError(f"No web exporter for {type(est).__name__}; champion must be linear or gradient boosting.")


def build_payload(estimator, champion: dict, registry_versions: list[dict], ds: Dataset) -> dict:
    X_all = np.vstack([ds.X_train, ds.X_test])
    features = []
    for i, (name, label, unit) in enumerate(FEATURES):
        features.append({
            "name": name, "label": label, "unit": unit,
            "min": float(X_all[:, i].min()), "max": float(X_all[:, i].max()),
            "mean": float(ds.X_train[:, i].mean()),
        })
    preds = estimator.predict(ds.X_test)
    idx = np.linspace(0, len(ds.X_test) - 1, N_EXAMPLES).astype(int)
    examples = [{"features": ds.X_test[i].tolist(), "actual": float(ds.y_test[i]),
                 "predicted": float(preds[i])} for i in idx]
    return {
        "schema": 2,
        "champion": {k: champion[k] for k in
                     ("version", "name", "family", "description", "n_features_used", "cv_rmse", "cv_rmse_std", "cv_r2",
                      "holdout_rmse", "holdout_mae", "holdout_r2", "interval_halfwidth",
                      "holdout_interval_coverage")},
        "model": export_model(estimator, ds.X_train),
        "features": features,
        "target": {"description": "Quantitative measure of diabetes progression one year after baseline",
                   "min": float(np.min(np.r_[ds.y_train, ds.y_test])),
                   "max": float(np.max(np.r_[ds.y_train, ds.y_test]))},
        "examples": examples,
        "versions": [{k: v[k] for k in ("version", "name", "family", "n_features_used", "cv_rmse", "cv_rmse_std", "cv_r2",
                                        "holdout_rmse", "status", "reason")} for v in registry_versions],
        "data": {"n_train": int(len(ds.y_train)), "n_test": int(len(ds.y_test)), "fingerprint": ds.fingerprint},
    }
