from typing import Any

import numpy as np
import pandas as pd

from credit_risk import schema
from credit_risk.fairness import age_band
from credit_risk.features import engineer

EXAMPLE_COLUMNS = [
    schema.LIMIT,
    schema.AGE,
    "PAY_1",
    "PAY_2",
    "n_delayed",
    "util_mean",
    "payratio_mean",
]


def segments(X: pd.DataFrame) -> dict[str, pd.Series]:
    """Named partitions of the clients used to locate where the model fails."""
    return {
        "latest repayment status": pd.cut(
            X["PAY_1"],
            [-3, -1, 0, 1, 20],
            labels=["paid duly or no use", "revolving", "1 month late", "2+ months late"],
        ).astype(str),
        "months with a delay": pd.cut(
            engineer(X)["n_delayed"], [-1, 0, 2, 6], labels=["none", "1-2", "3-6"]
        ).astype(str),
        "credit limit": pd.qcut(
            X[schema.LIMIT], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
        ).astype(str),
        "age": age_band(X[schema.AGE], [[21, 29], [30, 39], [40, 49], [50, 59], [60, 200]]),
    }


def _segment_row(y, p, flag, cost_fn, cost_fp) -> dict[str, Any]:
    fn = int(np.sum(~flag & (y == 1)))
    fp = int(np.sum(flag & (y == 0)))
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    return {
        "n": len(y),
        "default_rate": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "flagged_share": float(flag.mean()),
        "false_negative_rate": fn / pos if pos else None,
        "false_positive_rate": fp / neg if neg else None,
        "cost_per_client": float((cost_fn * fn + cost_fp * fp) / len(y)),
    }


def error_analysis(
    X: pd.DataFrame, y: pd.Series, p: np.ndarray, threshold: float, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Error rates by segment and the most confident mistakes with their feature values."""
    d, e = cfg["decision"], cfg["errors"]
    p = np.asarray(p)
    flag = p >= threshold
    yv = y.to_numpy()
    seg_out = {}
    for name, labels in segments(X).items():
        seg_out[name] = {
            str(level): _segment_row(yv[m], p[m], flag[m], d["cost_fn"], d["cost_fp"])
            for level in sorted(labels.unique())
            for m in [(labels == level).to_numpy()]
        }
    frame = pd.concat([X, engineer(X)[["n_delayed", "util_mean", "payratio_mean"]]], axis=1)
    frame = frame[EXAMPLE_COLUMNS].assign(probability=p, default=yv)
    n = e["n_examples"]
    fn_rows = frame[(yv == 1) & ~flag].nsmallest(n, "probability")
    fp_rows = frame[(yv == 0) & flag].nlargest(n, "probability")
    defaults, goods = yv == 1, yv == 0
    return {
        "threshold": threshold,
        "segments": seg_out,
        "missed_defaults_share": float((~flag & defaults).sum() / defaults.sum()),
        "defaults_below_0.1": float((p[defaults] < 0.1).mean()),
        "non_defaults_above_confident_threshold": float(
            (p[goods] >= e["confident_threshold"]).mean()
        ),
        "confident_threshold": e["confident_threshold"],
        "false_negative_examples": _records(fn_rows),
        "false_positive_examples": _records(fp_rows),
        "missed_default_profile": {
            "pay_1_mean_missed": float(X.loc[(~flag) & defaults, "PAY_1"].mean()),
            "pay_1_mean_caught": float(X.loc[flag & defaults, "PAY_1"].mean()),
        },
    }


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for idx, row in frame.iterrows():
        out.append(
            {
                "id": int(idx),
                **{
                    k: (
                        round(float(v), 3)
                        if k in ("probability", "util_mean", "payratio_mean")
                        else int(v)
                    )
                    for k, v in row.items()
                },
            }
        )
    return out
