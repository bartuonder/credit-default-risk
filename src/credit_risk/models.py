from typing import Any

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_risk.features import build_preprocessor, column_plan

MODEL_NAMES = [
    "dummy",
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "lightgbm",
]
CANDIDATES = ["random_forest", "hist_gradient_boosting", "lightgbm"]


def make_estimator(name: str, params: dict[str, Any], seed: int, n_jobs: int = 1):
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logistic_regression":
        return LogisticRegression(**params)
    if name == "random_forest":
        return RandomForestClassifier(**params, random_state=seed, n_jobs=n_jobs)
    if name == "hist_gradient_boosting":
        # Fixed number of iterations so folds are comparable; no internal early stopping.
        return HistGradientBoostingClassifier(**params, early_stopping=False, random_state=seed)
    if name == "lightgbm":
        return LGBMClassifier(**params, random_state=seed, n_jobs=n_jobs, verbose=-1)
    raise ValueError(f"unknown model: {name}")


def make_pipeline(
    name: str,
    cfg: dict[str, Any],
    params: dict[str, Any] | None = None,
    drop_groups: tuple[str, ...] = (),
    n_jobs: int = 1,
) -> Pipeline:
    """Preprocessing and estimator in one pipeline, so folds never share fitted state."""
    base = {} if name == "dummy" else dict(cfg["models"][name]["params"])
    base.update(params or {})
    base.pop("n_jobs", None)
    plan = column_plan(cfg, drop_groups)
    prep = build_preprocessor(plan, scale=name == "logistic_regression")
    estimator = make_estimator(name, base, cfg["seed"], n_jobs)
    return Pipeline([*prep.steps, ("model", estimator)])
