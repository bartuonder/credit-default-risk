"""Synthetic frames with the dataset schema; used only by tests, never for reported results."""

import numpy as np
import pandas as pd

from credit_risk import schema


def make_synthetic_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    limit = rng.choice([10_000, 50_000, 100_000, 200_000, 500_000], n)
    status = rng.choice([-2, -1, 0, 0, 0, 1, 2, 3], size=(n, 6))
    bills = (rng.uniform(-0.05, 1.1, size=(n, 6)) * limit[:, None]).round()
    pays = (rng.uniform(0, 0.4, size=(n, 6)) * np.abs(bills)).round()
    logit = -1.4 + 0.9 * (status[:, 0] > 0) + 0.3 * (status[:, 1] > 0) - limit / 600_000
    target = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(
        {
            schema.LIMIT: limit,
            schema.SEX: rng.integers(1, 3, n),
            schema.EDUCATION: rng.integers(1, 5, n),
            schema.MARRIAGE: rng.integers(1, 4, n),
            schema.AGE: rng.integers(21, 70, n),
            **{c: status[:, i] for i, c in enumerate(schema.PAY_STATUS)},
            **{c: bills[:, i].astype(int) for i, c in enumerate(schema.BILL_AMT)},
            **{c: pays[:, i].astype(int) for i, c in enumerate(schema.PAY_AMT)},
            schema.TARGET: target,
        }
    )
    df.index = pd.RangeIndex(1, n + 1, name="id")
    return df


def make_raw_like(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Frame shaped like the raw sheet after reading: original names and undocumented codes."""
    df = make_synthetic_frame(n, seed).reset_index()
    df = df.rename(columns={"id": "ID", "PAY_1": "PAY_0", "default": "default payment next month"})
    df.loc[0, "EDUCATION"] = 0
    df.loc[1, "EDUCATION"] = 5
    df.loc[2, "EDUCATION"] = 6
    df.loc[3, "MARRIAGE"] = 0
    return df
