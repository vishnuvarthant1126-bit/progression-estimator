"""Promotion gate: decides whether a challenger replaces the champion.

Rules (all on cross-validated training scores, so the holdout stays clean):
  * a challenger must beat the champion's CV RMSE by at least MIN_IMPROVEMENT
    to be promoted; a near-tie keeps the simpler, already-trusted champion;
  * within that tie band, a challenger that uses fewer input features is
    promoted (parsimony: equal accuracy, simpler model, easier to explain);
  * a challenger whose CV RMSE is more than REGRESSION_TOLERANCE worse, or
    whose CV R² drops by more than R2_DROP, is flagged as a regression.
"""
from __future__ import annotations

from dataclasses import dataclass

MIN_IMPROVEMENT = 0.25        # RMSE points
REGRESSION_TOLERANCE = 0.03   # 3 % relative RMSE increase
R2_DROP = 0.02


@dataclass(frozen=True)
class Decision:
    promote: bool
    status: str   # "champion" | "promoted" | "rejected" | "regression"
    reason: str


def decide(challenger: dict, champion: dict | None) -> Decision:
    if champion is None:
        return Decision(True, "promoted", "First model: becomes the baseline champion.")
    c_rmse, k_rmse = challenger["cv_rmse"], champion["cv_rmse"]
    rel = (c_rmse - k_rmse) / k_rmse
    if rel > REGRESSION_TOLERANCE or challenger["cv_r2"] < champion["cv_r2"] - R2_DROP:
        return Decision(False, "regression",
                        f"CV RMSE {c_rmse:.2f} vs champion {k_rmse:.2f} (+{rel:.1%}): blocked as a regression.")
    if k_rmse - c_rmse >= MIN_IMPROVEMENT:
        return Decision(True, "promoted",
                        f"CV RMSE {c_rmse:.2f} beats champion {k_rmse:.2f} by {k_rmse - c_rmse:.2f}.")
    tie = abs(k_rmse - c_rmse) < MIN_IMPROVEMENT
    if tie and challenger.get("n_features_used", 99) < champion.get("n_features_used", 99):
        return Decision(True, "promoted",
                        f"CV RMSE {c_rmse:.2f} ties champion {k_rmse:.2f} using "
                        f"{challenger['n_features_used']} of {champion['n_features_used']} features: simpler model wins.")
    return Decision(False, "rejected",
                    f"CV RMSE {c_rmse:.2f} vs champion {k_rmse:.2f}: gain under {MIN_IMPROVEMENT}, champion kept.")
