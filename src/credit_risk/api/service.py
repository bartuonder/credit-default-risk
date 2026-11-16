import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from credit_risk.explain import Explainer, top_contributions

log = logging.getLogger("credit_risk.service")


def risk_band(probability: float, bands: list[dict[str, Any]]) -> str:
    """Name of the first band whose upper bound exceeds the probability."""
    for band in bands:
        if probability < band["upper"]:
            return band["name"]
    return bands[-1]["name"]


class ModelService:
    """Loads the fitted model and its card once; scores and explains frames of applicants."""

    def __init__(self, model_path: Path, card_path: Path):
        self.card: dict[str, Any] = json.loads(Path(card_path).read_text())
        self.model = joblib.load(model_path)
        self.explainer = Explainer(self.model)
        self.threshold: float = self.card["threshold"]
        self.bands: list[dict[str, Any]] = self.card["risk_bands"]
        self.version: str = self.card["version"]
        self.explanation_unit: str = self.card.get("explanation_unit", "model margin")
        log.info("model loaded", extra={"model": self.card["model"], "version": self.version})

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(frame)[:, 1]

    def score(
        self, frame: pd.DataFrame, explain: bool = False, top_k: int = 5
    ) -> list[dict[str, Any]]:
        probabilities = self.predict(frame)
        explanations = top_contributions(self.explainer, frame, top_k) if explain else None
        results = []
        for i, p in enumerate(probabilities):
            flagged = bool(p >= self.threshold)
            results.append(
                {
                    "default_probability": float(p),
                    "flagged": flagged,
                    "decision": "flag as likely default" if flagged else "do not flag",
                    "threshold": self.threshold,
                    "risk_band": risk_band(float(p), self.bands),
                    "explanation": explanations[i] if explanations else None,
                }
            )
        return results

    def info(self) -> dict[str, Any]:
        keys = [
            "name", "version", "model", "hyperparameters", "calibration", "threshold",
            "cost_assumption", "risk_bands", "metrics_test", "metrics_test_ci95", "confusion_test",
            "expected_cost_per_client_test", "calibration_curve_test", "brier_decomposition_test",
            "cv_comparison", "n_train", "n_test", "feature_schema", "data", "libraries",
            "git_commit", "intended_use", "explanation_unit",
        ]  # fmt: skip
        return {k: self.card[k] for k in keys if k in self.card}
