import numpy as np

from credit_risk import schema
from credit_risk.cv import cross_validate, fold_scores_at_threshold, make_cv, summarise_folds
from credit_risk.models import make_pipeline


def test_cross_validate_shapes_and_coverage(cfg, frame):
    X, y = frame.drop(columns=schema.TARGET), frame[schema.TARGET]
    cv = make_cv(cfg, n_repeats=2)
    result = cross_validate(make_pipeline("logistic_regression", cfg), X, y, cv, n_jobs=1)
    assert result.oof.shape == (2, len(X))
    assert not np.isnan(result.oof).any()
    assert len(result.folds) == 10
    assert result.folds["roc_auc"].mean() > 0.6
    summary = summarise_folds(result.folds)
    assert set(summary["roc_auc"]) == {"mean", "std"}


def test_same_folds_for_every_model(cfg, frame):
    X, y = frame.drop(columns=schema.TARGET), frame[schema.TARGET]
    a = [te for _, te in make_cv(cfg).split(X, y)]
    b = [te for _, te in make_cv(cfg).split(X, y)]
    assert all(np.array_equal(i, j) for i, j in zip(a, b, strict=True))


def test_threshold_scores_per_fold(cfg, frame):
    X, y = frame.drop(columns=schema.TARGET), frame[schema.TARGET]
    cv = make_cv(cfg, n_repeats=1)
    result = cross_validate(make_pipeline("dummy", cfg), X, y, cv, n_jobs=1)
    low = fold_scores_at_threshold(result, y, cv, 0.0)
    assert (low["recall"] == 1.0).all()
