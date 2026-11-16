import numpy as np
import pytest

from credit_risk import schema
from credit_risk.errors import error_analysis
from credit_risk.fairness import age_band, fairness_report, group_row, wilson


def test_group_row_rates():
    y = np.array([1, 1, 0, 0, 0, 0])
    flag = np.array([True, False, True, False, False, False])
    row = group_row(y, flag)
    assert row["n"] == 6
    assert row["approval_rate"]["value"] == pytest.approx(4 / 6)
    assert row["false_negative_rate"]["value"] == pytest.approx(0.5)
    assert row["false_positive_rate"]["value"] == pytest.approx(0.25)


def test_wilson_interval_contains_estimate():
    lo, hi = wilson(30, 100)
    assert lo < 0.3 < hi
    assert wilson(0, 0) is None


def test_age_band_labels():
    import pandas as pd

    bands = age_band(pd.Series([22, 35, 61, 18]), [[21, 29], [30, 39], [60, 200]])
    assert list(bands) == ["21-29", "30-39", "60+", "other"]


def test_fairness_report_structure(cfg, frame):
    X, y = frame.drop(columns=schema.TARGET), frame[schema.TARGET]
    p = np.random.default_rng(0).uniform(size=len(y))
    report = fairness_report(X, y, p, 0.3, cfg)
    assert set(report["attributes"]) == {"SEX", "AGE_BAND", "EDUCATION"}
    assert sum(g["n"] for g in report["attributes"]["SEX"]["groups"].values()) == len(y)
    assert "fair" in report["note"]


def test_error_analysis_examples(cfg, frame):
    X, y = frame.drop(columns=schema.TARGET), frame[schema.TARGET]
    p = np.random.default_rng(1).uniform(size=len(y))
    result = error_analysis(X, y, p, 0.3, cfg)
    fn = result["false_negative_examples"]
    assert len(fn) <= cfg["errors"]["n_examples"]
    assert all(r["default"] == 1 and r["probability"] < 0.3 for r in fn)
    assert all(
        r["default"] == 0 and r["probability"] >= 0.3 for r in result["false_positive_examples"]
    )
    assert set(result["segments"]) >= {"latest repayment status", "credit limit"}
