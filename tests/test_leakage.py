import numpy as np
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

from credit_risk import schema
from credit_risk.models import make_pipeline


def _xy(frame):
    return frame.drop(columns=schema.TARGET), frame[schema.TARGET]


def test_scaler_is_fitted_on_training_rows_only(cfg, frame):
    X, y = _xy(frame)
    pipe = make_pipeline("logistic_regression", cfg)
    train, test = X.index[:400], X.index[400:]
    fitted = clone(pipe).fit(X.loc[train], y.loc[train])
    scaler = fitted.named_steps["encode"].named_transformers_["numeric"]
    numeric_cols = list(scaler.feature_names_in_)
    built = fitted.named_steps["features"].transform(X)
    assert np.allclose(scaler.mean_, built.loc[train, numeric_cols].mean().to_numpy())
    assert not np.allclose(scaler.mean_, built.loc[test, numeric_cols].mean().to_numpy())


def test_held_out_rows_do_not_change_the_fit(cfg, frame):
    X, y = _xy(frame)
    train, test = X.index[:400], X.index[400:]
    corrupted = X.copy()
    corrupted.loc[test, schema.LIMIT] = 10**9
    corrupted.loc[test, "BILL_AMT1"] = 10**9
    a = clone(make_pipeline("logistic_regression", cfg)).fit(X.loc[train], y.loc[train])
    b = clone(make_pipeline("logistic_regression", cfg)).fit(corrupted.loc[train], y.loc[train])
    assert np.allclose(a.named_steps["model"].coef_, b.named_steps["model"].coef_)
    assert np.allclose(a.predict_proba(X.loc[test[:20]]), b.predict_proba(X.loc[test[:20]]))


def test_folds_are_disjoint_and_cover_all_rows(frame):
    X, y = _xy(frame)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
    seen = np.zeros(len(X), dtype=int)
    for i, (tr, te) in enumerate(cv.split(X, y)):
        assert set(tr).isdisjoint(te)
        seen[te] += 1
        if i % 5 == 4:
            assert (seen == (i // 5) + 1).all()
