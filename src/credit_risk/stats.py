import numpy as np
from scipy import stats


def corrected_resampled_ttest(diffs: np.ndarray, n_train: int, n_test: int) -> dict[str, float]:
    """Nadeau-Bengio corrected t-test on paired fold-wise differences from repeated k-fold."""
    d = np.asarray(diffs, dtype=float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return {"mean_diff": float(d.mean()), "t": 0.0, "p_value": 1.0}
    t = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return {"mean_diff": float(d.mean()), "t": float(t), "p_value": float(p)}


def wilcoxon_test(diffs: np.ndarray) -> dict[str, float]:
    d = np.asarray(diffs, dtype=float)
    if np.allclose(d, 0):
        return {"statistic": 0.0, "p_value": 1.0}
    res = stats.wilcoxon(d)
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}
