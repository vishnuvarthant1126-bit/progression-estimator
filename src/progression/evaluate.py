"""Cross-validated and holdout evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

CV_SPLITS = 5
CV_REPEATS = 3
CV_SEED = 0
INTERVAL_COVERAGE = 0.90


def metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_validate(estimator, X, y) -> dict:
    """Repeated k-fold on the training split.

    Returns mean/std of RMSE, MAE and R², plus the out-of-fold absolute residuals
    (averaged over repeats) used to calibrate the prediction interval.
    """
    cv = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=CV_SEED)
    fold_scores = {"rmse": [], "mae": [], "r2": []}
    oof_abs_resid = np.zeros((CV_REPEATS, len(y)))
    for i, (tr, va) in enumerate(cv.split(X)):
        m = clone(estimator).fit(X[tr], y[tr])
        pred = m.predict(X[va])
        for k, v in metrics(y[va], pred).items():
            fold_scores[k].append(v)
        oof_abs_resid[i // CV_SPLITS, va] = np.abs(y[va] - pred)
    out = {}
    for k, vals in fold_scores.items():
        out[f"cv_{k}"] = float(np.mean(vals))
        out[f"cv_{k}_std"] = float(np.std(vals))
    resid = oof_abs_resid.mean(axis=0)
    out["interval_halfwidth"] = conformal_halfwidth(resid, INTERVAL_COVERAGE)
    return out


def conformal_halfwidth(abs_residuals, coverage: float) -> float:
    """Split-conformal quantile: prediction ± this covers `coverage` of new cases."""
    n = len(abs_residuals)
    q = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(abs_residuals, q, method="higher"))


def interval_coverage(y_true, y_pred, halfwidth: float) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= halfwidth))
