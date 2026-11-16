import pytest

from credit_risk.config import load_config
from tests.synthetic import make_synthetic_frame


@pytest.fixture(scope="module")
def frame():
    return make_synthetic_frame()


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, cfg, frame):
    """A small fitted model and card in a temp directory (synthetic data, tests only)."""
    import json

    import joblib

    from credit_risk import schema
    from credit_risk.calibration import wrap_calibration
    from credit_risk.final import feature_schema
    from credit_risk.models import make_pipeline

    X, y = frame.drop(columns=schema.TARGET), frame[schema.TARGET]
    pipe = make_pipeline("lightgbm", cfg, {"n_estimators": 40, "num_leaves": 7})
    model = wrap_calibration(pipe, "sigmoid", 3).fit(X, y)
    directory = tmp_path_factory.mktemp("artifacts")
    joblib.dump(model, directory / "model.joblib")
    card = {
        "name": "test-model",
        "version": "0.0.0-test",
        "model": "lightgbm",
        "calibration": "sigmoid",
        "threshold": 0.25,
        "cost_assumption": {"false_negative": 5.0, "false_positive": 1.0, "note": "test"},
        "risk_bands": cfg["decision"]["risk_bands"],
        "metrics_test": {"roc_auc": 0.7},
        "calibration_curve_test": {"mean_predicted": [0.1], "observed_rate": [0.1], "count": [10]},
        "feature_schema": feature_schema(),
        "explanation_unit": "log-odds",
    }
    (directory / "model_card.json").write_text(json.dumps(card))
    return directory
