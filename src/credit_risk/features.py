from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from credit_risk import schema

ENGINEERED_BLOCKS = {
    "utilisation": ["util_1", "util_mean", "util_max"],
    "payment_ratio": ["payratio_1", "payratio_mean", "payratio_total"],
    "delay_counts": ["n_delayed", "max_delay", "mean_delay"],
    "status_trend": ["status_trend"],
    "bill_trend": ["bill_trend"],
}
_TIME = np.arange(6, dtype=float) - 2.5  # centred month index, oldest to newest


def _slope(values: np.ndarray) -> np.ndarray:
    """Least-squares slope per row over six months; `values` is (n, 6), oldest first."""
    centred = values - values.mean(axis=1, keepdims=True)
    return centred @ _TIME / np.sum(_TIME**2)


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    """All engineered columns for a raw feature frame; row-wise, no fitted state."""
    limit = X[schema.LIMIT].to_numpy(float)
    bills = X[schema.BILL_AMT].to_numpy(float)  # (n, 6), newest first
    pays = X[schema.PAY_AMT].to_numpy(float)
    status = X[schema.PAY_STATUS].to_numpy(float)

    util = np.clip(bills / limit[:, None], -1.0, 3.0)
    # A payment in month k settles the bill of month k+1 (columns are newest first).
    owed = bills[:, 1:]
    ratio = np.where(owed > 0, pays[:, :5] / np.where(owed > 0, owed, 1.0), 1.0)
    ratio = np.clip(ratio, 0.0, 1.0)
    owed_total = np.clip(owed, 0, None).sum(axis=1)
    total = np.where(
        owed_total > 0, pays[:, :5].sum(axis=1) / np.where(owed_total > 0, owed_total, 1.0), 1.0
    )
    delayed = np.clip(status, 0, None)

    out = pd.DataFrame(
        {
            "util_1": util[:, 0],
            "util_mean": util.mean(axis=1),
            "util_max": util.max(axis=1),
            "payratio_1": ratio[:, 0],
            "payratio_mean": ratio.mean(axis=1),
            "payratio_total": np.clip(total, 0.0, 1.0),
            "n_delayed": (status > 0).sum(axis=1),
            "max_delay": status.max(axis=1),
            "mean_delay": delayed.mean(axis=1),
            "status_trend": _slope(status[:, ::-1]),
            "bill_trend": _slope((bills / limit[:, None])[:, ::-1]),
        },
        index=X.index,
    )
    return out


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Select raw columns and add engineered ones; stateless, so safe inside any fold."""

    def __init__(self, columns: tuple[str, ...], engineered: tuple[str, ...] = ()):
        self.columns = columns
        self.engineered = engineered

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X
        if self.engineered:
            frame = pd.concat([X, engineer(X)[self._engineered_columns()]], axis=1)
        return frame[list(self.columns)]

    def _engineered_columns(self) -> list[str]:
        return [c for block in self.engineered for c in ENGINEERED_BLOCKS[block]]


def column_plan(
    cfg: dict[str, Any],
    drop_groups: tuple[str, ...] = (),
    engineered: tuple[str, ...] | None = None,
) -> dict[str, list[str]]:
    """Final feature columns split by treatment, after removing ablated feature groups."""
    fcfg = cfg["features"]
    blocks = tuple(fcfg["engineered"] if engineered is None else engineered)
    allowed = set(schema.RAW_FEATURES) | {c for b in blocks for c in ENGINEERED_BLOCKS[b]}
    dropped = {c for g in drop_groups for c in fcfg["groups"][g]}
    columns = [c for c in _ordered_columns(blocks) if c in allowed and c not in dropped]
    categorical = [c for c in columns if c in fcfg["categorical"]]
    amounts = [c for c in columns if c in {schema.LIMIT, *schema.BILL_AMT, *schema.PAY_AMT}]
    numeric = [c for c in columns if c not in categorical and c not in amounts]
    return {
        "columns": columns,
        "categorical": categorical,
        "amounts": amounts,
        "numeric": numeric,
        "engineered": list(blocks),
    }


def _ordered_columns(blocks: tuple[str, ...]) -> list[str]:
    return [*schema.RAW_FEATURES, *[c for b in blocks for c in ENGINEERED_BLOCKS[b]]]


def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def build_preprocessor(plan: dict[str, list[str]], scale: bool) -> Pipeline:
    """Feature builder plus column-wise encoding; scaling and log amounts only for linear models."""
    onehot = OneHotEncoder(
        categories=[sorted(schema.CATEGORY_VALUES[c]) for c in plan["categorical"]],
        sparse_output=False,
        handle_unknown="ignore",
    )
    if scale:
        amounts = Pipeline(
            [
                ("log", FunctionTransformer(_signed_log, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
        numeric = StandardScaler()
    else:
        amounts, numeric = "passthrough", "passthrough"
    transformers = []
    if plan["categorical"]:
        transformers.append(("categorical", onehot, plan["categorical"]))
    if plan["amounts"]:
        transformers.append(("amounts", amounts, plan["amounts"]))
    if plan["numeric"]:
        transformers.append(("numeric", numeric, plan["numeric"]))
    encoder = ColumnTransformer(transformers, verbose_feature_names_out=False)
    encoder.set_output(transform="pandas")
    builder = FeatureBuilder(tuple(plan["columns"]), tuple(plan["engineered"]))
    return Pipeline([("features", builder), ("encode", encoder)])
