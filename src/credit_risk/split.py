import json
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk import schema
from credit_risk.config import path_for
from credit_risk.data import load_clean


def make_split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.Index, pd.Index]:
    """Stratified hold-out split; returns (train_ids, test_ids)."""
    train_ids, test_ids = train_test_split(
        df.index, test_size=test_size, stratify=df[schema.TARGET], random_state=seed
    )
    return pd.Index(sorted(train_ids)), pd.Index(sorted(test_ids))


def _split_path(cfg: dict[str, Any]):
    return path_for(cfg, "results") / "split.json"


def ensure_split(cfg: dict[str, Any]) -> dict[str, Any]:
    """Create the split once and persist the test ids; later calls reuse the stored ids."""
    path = _split_path(cfg)
    if path.exists():
        return json.loads(path.read_text())
    df = load_clean(cfg)
    train_ids, test_ids = make_split(df, cfg["split"]["test_size"], cfg["seed"])
    info = {
        "seed": cfg["seed"],
        "test_size": cfg["split"]["test_size"],
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "train_default_rate": float(df.loc[train_ids, schema.TARGET].mean()),
        "test_default_rate": float(df.loc[test_ids, schema.TARGET].mean()),
        "test_ids": [int(i) for i in test_ids],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info) + "\n")
    return info


def load_train(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Training features and target. Everything except the final evaluation uses only this."""
    info = ensure_split(cfg)
    df = load_clean(cfg)
    train = df.drop(index=info["test_ids"])
    return train.drop(columns=schema.TARGET), train[schema.TARGET]


def load_test(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Hold-out features and target; only the final evaluation may call this."""
    info = ensure_split(cfg)
    test = load_clean(cfg).loc[info["test_ids"]]
    return test.drop(columns=schema.TARGET), test[schema.TARGET]
