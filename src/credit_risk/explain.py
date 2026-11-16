from typing import Any

import numpy as np
import pandas as pd
import shap

from credit_risk import schema

FEATURE_LABELS = {
    "LIMIT_BAL": "Credit limit",
    "SEX": "Sex",
    "EDUCATION": "Education",
    "MARRIAGE": "Marital status",
    "AGE": "Age",
    "n_delayed": "Months with a payment delay",
    "max_delay": "Longest payment delay",
    "mean_delay": "Average delay (months)",
    "status_trend": "Repayment status trend",
    "util_1": "Latest utilisation",
    "util_mean": "Average utilisation",
    "util_max": "Peak utilisation",
    "bill_trend": "Bill trend (share of limit)",
    "payratio_1": "Latest payment / bill",
    "payratio_mean": "Average payment / bill",
    "payratio_total": "Total payment / bill",
    **{f"PAY_{i}": f"Repayment status, month -{i - 1}" for i in range(1, 7)},
    **{f"BILL_AMT{i}": f"Bill amount, month -{i - 1}" for i in range(1, 7)},
    **{f"PAY_AMT{i}": f"Payment amount, month -{i - 1}" for i in range(1, 7)},
}
FEATURE_LABELS["PAY_1"] = "Repayment status, latest month"
FEATURE_LABELS["BILL_AMT1"] = "Bill amount, latest month"
FEATURE_LABELS["PAY_AMT1"] = "Payment amount, latest month"


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def pipeline_members(model) -> list:
    """Fitted pipelines behind a model: calibration ensemble members, or the pipeline itself."""
    if hasattr(model, "calibrated_classifiers_"):
        return [c.estimator for c in model.calibrated_classifiers_]
    return [model]


def _positive_class(values) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    return values[..., -1] if values.ndim == 3 else values


def _base_value(expected) -> float:
    return float(np.atleast_1d(expected)[-1])


class Explainer:
    """SHAP contributions per original feature, averaged over the members of the model."""

    def __init__(self, model, categorical: list[str] | None = None):
        self.members = pipeline_members(model)
        self.explainers = [shap.TreeExplainer(m[-1]) for m in self.members]
        self.categorical = categorical or list(schema.CATEGORY_VALUES)
        self.base_value = float(np.mean([_base_value(e.expected_value) for e in self.explainers]))

    def encoded(self, X: pd.DataFrame) -> list[pd.DataFrame]:
        return [m[:-1].transform(X) for m in self.members]

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """(n, n_features) contributions, one-hot columns summed back to their feature."""
        total = None
        for explainer, enc in zip(self.explainers, self.encoded(X), strict=True):
            values = pd.DataFrame(_positive_class(explainer.shap_values(enc)), columns=enc.columns)
            total = values if total is None else total + values
        mean = total / len(self.members)
        for c in self.categorical:
            cols = [k for k in mean.columns if k.startswith(f"{c}_")]
            if cols:
                mean[c] = mean[cols].sum(axis=1)
                mean = mean.drop(columns=cols)
        mean.index = X.index
        return mean

    def feature_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Raw and engineered feature values, in original units."""
        return self.members[0].named_steps["features"].transform(X)


def top_contributions(
    explainer: Explainer, X: pd.DataFrame, k: int = 5
) -> list[list[dict[str, Any]]]:
    """Per row, the k features with the largest absolute contribution."""
    contrib = explainer.contributions(X)
    values = explainer.feature_values(X)
    out = []
    for i in range(len(X)):
        row = contrib.iloc[i]
        top = row.abs().sort_values(ascending=False).index[:k]
        items = []
        for name in top:
            raw = values.iloc[i][name]
            if name in schema.CATEGORY_VALUES:
                shown: Any = schema.CATEGORY_VALUES[name].get(int(raw), int(raw))
            else:
                shown = float(raw) if not float(raw).is_integer() else int(raw)
            items.append(
                {
                    "feature": name,
                    "label": label(name),
                    "value": shown,
                    "contribution": float(row[name]),
                    "direction": "increases risk" if row[name] > 0 else "decreases risk",
                }
            )
        out.append(items)
    return out
