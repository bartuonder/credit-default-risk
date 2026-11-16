from typing import Any

import numpy as np
import pandas as pd

from credit_risk import schema

NOTE = (
    "Descriptive diagnostic only. Group differences mix real differences in default rates with "
    "model behaviour, group sizes are uneven, and rates have wide intervals for small groups. "
    "Error-rate parity, calibration within groups and equal approval rates cannot all hold when "
    "base rates differ. SEX and EDUCATION are model inputs in this benchmark; using them (or close "
    "proxies) in real lending can be restricted by law. Nothing here establishes that the model is fair."
)


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return [float(centre - half), float(centre + half)]


def _rate(k: int, n: int) -> dict[str, Any]:
    return {"value": (k / n if n else None), "ci95": wilson(k, n)}


def group_row(y: np.ndarray, flag: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum(flag & (y == 1)))
    fp = int(np.sum(flag & (y == 0)))
    fn = int(np.sum(~flag & (y == 1)))
    tn = int(np.sum(~flag & (y == 0)))
    n = len(y)
    return {
        "n": n,
        "default_rate": float(y.mean()) if n else None,
        "approval_rate": _rate(tn + fn, n),
        "false_positive_rate": _rate(fp, fp + tn),
        "false_negative_rate": _rate(fn, fn + tp),
        "precision": _rate(tp, tp + fp),
    }


def age_band(age: pd.Series, bands: list[list[int]]) -> pd.Series:
    labels = pd.Series(index=age.index, dtype=object)
    for lo, hi in bands:
        labels[(age >= lo) & (age <= hi)] = f"{lo}-{hi}" if hi < 150 else f"{lo}+"
    return labels.fillna("other")


def fairness_report(
    X: pd.DataFrame, y: pd.Series, p: np.ndarray, threshold: float, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Approval, FPR and FNR by SEX, age band and EDUCATION at the decision threshold."""
    flag = np.asarray(p) >= threshold
    yv = y.to_numpy()
    attributes = {
        "SEX": X[schema.SEX].map(schema.CATEGORY_VALUES[schema.SEX]),
        "AGE_BAND": age_band(X[schema.AGE], cfg["fairness"]["age_bands"]),
        "EDUCATION": X[schema.EDUCATION].map(schema.CATEGORY_VALUES[schema.EDUCATION]),
    }
    out: dict[str, Any] = {
        "threshold": threshold,
        "note": NOTE,
        "overall": group_row(yv, flag),
        "attributes": {},
    }
    for name, values in attributes.items():
        rows = {}
        for level in sorted(values.dropna().unique(), key=str):
            mask = (values == level).to_numpy()
            rows[str(level)] = group_row(yv[mask], flag[mask])
        approvals = [r["approval_rate"]["value"] for r in rows.values() if r["n"] >= 100]
        out["attributes"][name] = {
            "groups": rows,
            "approval_rate_range": [float(min(approvals)), float(max(approvals))],
        }
    return out
