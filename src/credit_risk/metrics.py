from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

EPS = 1e-6


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Weighted mean absolute gap between confidence and observed frequency over equal-width bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def score_probabilities(
    y: np.ndarray, p: np.ndarray, threshold: float | None = None, ece_bins: int = 10
) -> dict[str, Any]:
    """Ranking and probability-quality metrics, plus precision/recall at `threshold` if given."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    out = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
        "ece": expected_calibration_error(y, p, ece_bins),
    }
    if threshold is not None:
        pred = (p >= threshold).astype(int)
        out["precision"] = float(precision_score(y, pred, zero_division=0))
        out["recall"] = float(recall_score(y, pred, zero_division=0))
    return out
