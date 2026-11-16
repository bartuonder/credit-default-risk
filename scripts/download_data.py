"""Fallback loader used only when the raw Excel file is missing.

Tries `ucimlrepo` and then OpenML, and writes the cleaned interim CSV directly.
"""

import sys

import pandas as pd

from credit_risk import schema
from credit_risk.config import load_config, path_for
from credit_risk.data import clean, validate

UCI_ID = 350
OPENML_NAME = "default-of-credit-card-clients"


def from_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=UCI_ID)
    frame = ds.data.features.copy()
    frame["default payment next month"] = ds.data.targets.iloc[:, 0]
    frame.insert(0, "ID", range(1, len(frame) + 1))
    return frame.rename(columns={f"X{i}": c for i, c in enumerate(schema.RAW_FEATURES, 1)})


def from_openml() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    ds = fetch_openml(name=OPENML_NAME, version=1, as_frame=True)
    frame = ds.frame.copy()
    frame.insert(0, "ID", range(1, len(frame) + 1))
    return frame


def main() -> int:
    cfg = load_config()
    for loader in (from_ucimlrepo, from_openml):
        try:
            raw = loader()
            df = clean(raw, cfg)
            validate(df, cfg["data"]["target"])
        except Exception as exc:
            print(f"{loader.__name__} failed: {exc}", file=sys.stderr)
            continue
        out = path_for(cfg, "interim")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out)
        print(f"wrote {out}")
        return 0
    print("no data source available", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
