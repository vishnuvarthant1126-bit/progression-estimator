"""Per-feature contributions that sum exactly to the prediction.

* Linear models: contribution_i = coef_i * (x_i - mean_i), measured from the
  average training patient.
* Gradient boosting: path attribution (Saabas). Walking each tree, the change
  in node value at every split is credited to the split's feature.

In both cases  base_value + sum(contributions) == prediction,  which the test
suite checks. The web page runs the same arithmetic in JavaScript.
"""
from __future__ import annotations

import numpy as np


def linear_contributions(model: dict, x: np.ndarray) -> tuple[float, np.ndarray]:
    coef = np.asarray(model["coef"])
    center = np.asarray(model["center"])
    base = model["intercept"] + float(coef @ center)
    return base, coef * (np.asarray(x, dtype=float) - center)


def gbm_contributions(model: dict, x: np.ndarray, n_features: int) -> tuple[float, np.ndarray]:
    x32 = np.asarray(x, dtype=np.float32)
    lr = model["learning_rate"]
    base = model["init"]
    contrib = np.zeros(n_features)
    for t in model["trees"]:
        node = 0
        base += lr * t["value"][0]
        while t["left"][node] != -1:
            f = t["feature"][node]
            nxt = t["left"][node] if float(x32[f]) <= t["threshold"][node] else t["right"][node]
            contrib[f] += lr * (t["value"][nxt] - t["value"][node])
            node = nxt
    return base, contrib


def contributions(export: dict, x) -> tuple[float, np.ndarray]:
    m = export["model"]
    if m["family"] == "linear":
        return linear_contributions(m, x)
    if m["family"] == "gbm":
        return gbm_contributions(m, x, len(export["features"]))
    raise ValueError(f"unsupported family {m['family']}")


def predict(export: dict, x) -> float:
    base, c = contributions(export, x)
    return float(base + c.sum())
