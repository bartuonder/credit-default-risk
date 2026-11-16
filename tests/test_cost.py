import numpy as np
import pytest

from credit_risk.calibration import brier_decomposition, reliability_curve
from credit_risk.cost import (
    GRID,
    bayes_threshold,
    best_threshold,
    cost_curve,
    expected_cost,
    sensitivity,
)


def test_expected_cost_counts_errors():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1])
    # threshold 0.5: one FN (cost 5) and one FP (cost 1) over four clients
    assert expected_cost(y, p, 0.5, 5.0, 1.0) == pytest.approx(6 / 4)
    assert expected_cost(y, p, 0.0, 5.0, 1.0) == pytest.approx(2 / 4)
    assert expected_cost(y, p, 1.01, 5.0, 1.0) == pytest.approx(10 / 4)


def test_bayes_threshold():
    assert bayes_threshold(5, 1) == pytest.approx(1 / 6)
    assert bayes_threshold(1, 1) == pytest.approx(0.5)


def test_threshold_matches_bayes_for_calibrated_probabilities():
    rng = np.random.default_rng(0)
    p = rng.beta(1.2, 4, 40_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    t = best_threshold(y, p, 5.0, 1.0)
    assert abs(t - bayes_threshold(5, 1)) < 0.04


def test_higher_fn_cost_lowers_threshold():
    rng = np.random.default_rng(1)
    p = rng.beta(1.2, 4, 20_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    rows = sensitivity(y, p, [1, 5, 10], ref_threshold=0.17)
    thresholds = [r["threshold"] for r in rows]
    assert thresholds == sorted(thresholds, reverse=True)
    assert all(r["expected_cost"] <= r["cost_at_reference_threshold"] + 1e-12 for r in rows)


def test_cost_curve_averages_repeats():
    y = np.array([1, 0, 1, 0])
    oof = np.array([[0.9, 0.1, 0.8, 0.2], [0.9, 0.1, 0.8, 0.2]])
    single = cost_curve(y, oof[0], 5, 1)
    assert np.allclose(cost_curve(y, oof, 5, 1), single)
    assert single.shape == GRID.shape


def test_brier_decomposition_identity_for_binned_probabilities():
    rng = np.random.default_rng(2)
    levels = np.linspace(0.05, 0.95, 10)
    p = rng.choice(levels, 30_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    d = brier_decomposition(y, p)
    assert abs(d["within_bin_residual"]) < 1e-9
    assert d["reliability"] < 0.001


def test_reliability_curve_shape():
    rng = np.random.default_rng(3)
    p = rng.uniform(size=1000)
    y = (rng.uniform(size=1000) < p).astype(int)
    curve = reliability_curve(y, p, bins=10)
    assert len(curve["mean_predicted"]) == 10 and sum(curve["count"]) == 1000
    assert np.all(np.diff(curve["mean_predicted"]) >= 0)
