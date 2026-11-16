from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

from credit_risk.metrics import score_probabilities


@dataclass
class CVResult:
    folds: pd.DataFrame  # one row per (repeat, fold)
    oof: np.ndarray  # (n_repeats, n_samples) out-of-fold probabilities


def make_cv(cfg: dict[str, Any], n_repeats: int | None = None) -> RepeatedStratifiedKFold:
    """Evaluation folds, shared by every model so paired comparisons are valid."""
    c = cfg["cv"]
    return RepeatedStratifiedKFold(
        n_splits=c["n_splits"],
        n_repeats=n_repeats or c["n_repeats"],
        random_state=cfg["seed"] + c["seed_offset"],
    )


def _fit_predict(estimator, X, y, train, test) -> np.ndarray:
    fitted = clone(estimator).fit(X.iloc[train], y.iloc[train])
    return fitted.predict_proba(X.iloc[test])[:, 1]


def cross_validate(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: RepeatedStratifiedKFold,
    ece_bins: int = 10,
    n_jobs: int = 4,
) -> CVResult:
    """Fit a fresh clone per fold (preprocessing included) and score on the held-out fold."""
    splits = list(cv.split(X, y))
    preds = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict)(estimator, X, y, tr, te) for tr, te in splits
    )
    k = cv.get_n_splits() // cv.n_repeats
    oof = np.full((cv.n_repeats, len(X)), np.nan)
    rows = []
    for i, ((_, te), p) in enumerate(zip(splits, preds, strict=True)):
        oof[i // k, te] = p
        rows.append(
            {
                "repeat": i // k,
                "fold": i % k,
                **score_probabilities(y.iloc[te], p, ece_bins=ece_bins),
            }
        )
    return CVResult(pd.DataFrame(rows), oof)


def fold_scores_at_threshold(
    result: CVResult, y: pd.Series, cv: RepeatedStratifiedKFold, threshold: float
) -> pd.DataFrame:
    """Per-fold precision and recall at a decision threshold, from stored predictions."""
    from sklearn.metrics import precision_score, recall_score

    k = cv.get_n_splits() // cv.n_repeats
    rows = []
    for i, (_, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        pred = (result.oof[i // k, te] >= threshold).astype(int)
        rows.append(
            {
                "repeat": i // k,
                "fold": i % k,
                "precision": precision_score(y.iloc[te], pred, zero_division=0),
                "recall": recall_score(y.iloc[te], pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def summarise_folds(folds: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Mean and standard deviation over folds for every metric column."""
    metrics = [c for c in folds.columns if c not in ("repeat", "fold")]
    return {
        m: {"mean": float(folds[m].mean()), "std": float(folds[m].std(ddof=1))} for m in metrics
    }
