import numpy as np
import pytest

from credit_risk.metrics import expected_calibration_error, score_probabilities
from credit_risk.stats import corrected_resampled_ttest, wilcoxon_test


def test_perfect_and_random_scores():
    y = np.array([0, 0, 1, 1])
    perfect = score_probabilities(y, np.array([0.1, 0.2, 0.8, 0.9]), threshold=0.5)
    assert perfect["roc_auc"] == 1.0 and perfect["pr_auc"] == 1.0
    assert perfect["precision"] == 1.0 and perfect["recall"] == 1.0
    constant = score_probabilities(y, np.full(4, 0.5))
    assert constant["roc_auc"] == 0.5
    assert constant["brier"] == pytest.approx(0.25)


def test_ece_zero_for_calibrated_bins():
    p = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, p) == 0.0
    assert expected_calibration_error(1 - y, p) == pytest.approx(1.0)


def test_ttest_correction_widens_the_test():
    d = np.array([0.01, 0.012, 0.009, 0.011, 0.013] * 3)
    corrected = corrected_resampled_ttest(d, n_train=24_000 * 4 // 5, n_test=24_000 // 5)
    naive = corrected_resampled_ttest(d, n_train=10**9, n_test=1)
    assert corrected["p_value"] > naive["p_value"]
    assert corrected["mean_diff"] == pytest.approx(d.mean())


def test_ttest_and_wilcoxon_on_zero_differences():
    d = np.zeros(15)
    assert corrected_resampled_ttest(d, 100, 25)["p_value"] == 1.0
    assert wilcoxon_test(d)["p_value"] == 1.0


def test_wilcoxon_detects_consistent_shift():
    d = np.linspace(0.01, 0.02, 15)
    assert wilcoxon_test(d)["p_value"] < 0.01
