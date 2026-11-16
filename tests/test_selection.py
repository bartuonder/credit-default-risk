import numpy as np
import pandas as pd

from credit_risk.selection import choose_model, paired_differences


def _folds(means, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for model, mean in means.items():
        for r in range(3):
            for f in range(5):
                rows.append(
                    {"model": model, "repeat": r, "fold": f, "pr_auc": mean + rng.normal(0, noise)}
                )
    return pd.DataFrame(rows)


def test_paired_differences_align_on_fold(cfg):
    folds = _folds({"a": 0.6, "b": 0.5})
    assert np.allclose(paired_differences(folds, "a", "b", "pr_auc"), 0.1)


def test_clear_winner_is_selected(cfg):
    folds = _folds({"random_forest": 0.60, "hist_gradient_boosting": 0.50, "lightgbm": 0.50}, 0.001)
    choice = choose_model(folds, cfg, 19_200, 4_800)
    assert choice["selected"] == "random_forest"


def test_tie_prefers_smaller_model(cfg):
    folds = _folds(
        {"random_forest": 0.5601, "hist_gradient_boosting": 0.56, "lightgbm": 0.5599}, 0.01
    )
    choice = choose_model(folds, cfg, 19_200, 4_800)
    assert len(choice["indistinguishable_from_top"]) == 3
    assert choice["selected"] == "lightgbm"
