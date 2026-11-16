import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV

from credit_risk import schema
from credit_risk.explain import Explainer, top_contributions
from credit_risk.models import make_pipeline


@pytest.fixture(scope="module")
def fitted(cfg, frame):
    X, y = frame.drop(columns=schema.TARGET), frame[schema.TARGET]
    pipe = make_pipeline("lightgbm", cfg, {"n_estimators": 60}).fit(X, y)
    return pipe, X, y


def test_contributions_add_up_to_the_model_margin(fitted):
    pipe, X, _ = fitted
    explainer = Explainer(pipe)
    contrib = explainer.contributions(X.iloc[:25])
    margin = pipe[-1].predict(pipe[:-1].transform(X.iloc[:25]), raw_score=True)
    assert np.allclose(contrib.sum(axis=1) + explainer.base_value, margin, atol=1e-6)


def test_one_hot_columns_are_grouped_by_feature(fitted):
    pipe, X, _ = fitted
    contrib = Explainer(pipe).contributions(X.iloc[:5])
    assert {"SEX", "EDUCATION", "MARRIAGE"} <= set(contrib.columns)
    assert not any(c.startswith("SEX_") for c in contrib.columns)


def test_top_contributions_shape_and_order(fitted):
    pipe, X, _ = fitted
    result = top_contributions(Explainer(pipe), X.iloc[:3], k=4)
    assert len(result) == 3 and all(len(r) == 4 for r in result)
    for items in result:
        sizes = [abs(i["contribution"]) for i in items]
        assert sizes == sorted(sizes, reverse=True)
        assert {"feature", "label", "value", "contribution", "direction"} <= set(items[0])


def test_calibrated_ensemble_is_averaged_over_members(cfg, frame):
    X, y = frame.drop(columns=schema.TARGET), frame[schema.TARGET]
    base = make_pipeline("lightgbm", cfg, {"n_estimators": 40})
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(X, y)
    explainer = Explainer(model)
    assert len(explainer.members) == 3
    assert explainer.contributions(X.iloc[:4]).shape[0] == 4
