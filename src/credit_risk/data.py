import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk import schema
from credit_risk.config import load_config, parse_args, path_for


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw(path: Path, header_row: int = 1) -> pd.DataFrame:
    """Read the Excel file; row 0 holds descriptive labels (X1..X23), row 1 the real header."""
    return pd.read_excel(path, header=header_row, engine="calamine")


def clean(raw: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Rename columns, merge undocumented category codes and index by client id."""
    data_cfg = cfg["data"]
    df = raw.rename(columns=data_cfg["rename"]).copy()
    df = df.set_index(data_cfg["id_column"])
    df.index.name = "id"
    for column, mapping in data_cfg["recode"].items():
        df[column] = df[column].replace(mapping)
    return df[[*schema.RAW_FEATURES, data_cfg["target"]]].astype("int64")


def validate(df: pd.DataFrame, target: str = schema.TARGET) -> None:
    """Raise ValueError listing every violated expectation."""
    problems: list[str] = []
    missing = [c for c in [*schema.RAW_FEATURES, target] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df[schema.RAW_FEATURES + [target]].isna().any().any():
        problems.append("null values present")
    for column, allowed in schema.CATEGORY_VALUES.items():
        bad = set(df[column].unique()) - set(allowed)
        if bad:
            problems.append(f"{column}: unexpected codes {sorted(bad)}")
    if not df[target].isin([0, 1]).all():
        problems.append("target is not binary")
    if (df[schema.LIMIT] <= 0).any():
        problems.append("LIMIT_BAL must be positive")
    lo, hi = schema.AGE_RANGE
    if not df[schema.AGE].between(lo, hi).all():
        problems.append(f"AGE outside [{lo}, {hi}]")
    lo, hi = schema.PAY_STATUS_RANGE
    for column in schema.PAY_STATUS:
        if not df[column].between(lo, hi).all():
            problems.append(f"{column} outside [{lo}, {hi}]")
    if (df[schema.PAY_AMT] < 0).any().any():
        problems.append("negative payment amounts")
    if problems:
        raise ValueError("; ".join(problems))


def summarise(raw: pd.DataFrame, df: pd.DataFrame, raw_path: Path, cfg: dict[str, Any]) -> dict:
    """Facts about the raw file and the cleaned table, written to results/."""
    target = cfg["data"]["target"]
    renamed = raw.rename(columns=cfg["data"]["rename"])
    features = df.drop(columns=target)
    return {
        "raw_file": str(raw_path.name),
        "raw_sha256": file_sha256(raw_path),
        "n_rows": len(df),
        "n_columns_raw": raw.shape[1],
        "target_rate": float(df[target].mean()),
        "n_defaults": int(df[target].sum()),
        "duplicate_rows_excluding_id": int(features.duplicated().sum()),
        "raw_category_counts": {
            c: {int(k): int(v) for k, v in renamed[c].value_counts().sort_index().items()}
            for c in schema.CATEGORY_VALUES
        },
        "recoded_rows": {
            c: int(renamed[c].isin(list(m)).sum()) for c, m in cfg["data"]["recode"].items()
        },
        "pay_status_counts": {
            int(k): int(v)
            for k, v in pd.Series(df[schema.PAY_STATUS].to_numpy().ravel())
            .value_counts()
            .sort_index()
            .items()
        },
        "negative_bill_rows": int((df[schema.BILL_AMT] < 0).any(axis=1).sum()),
        "columns": list(df.columns),
    }


def load_clean(cfg: dict[str, Any]) -> pd.DataFrame:
    """Cleaned table from the interim CSV, building it from the raw file when absent."""
    interim = path_for(cfg, "interim")
    if not interim.exists():
        build(cfg)
    return pd.read_csv(interim, index_col="id")


def build(cfg: dict[str, Any]) -> pd.DataFrame:
    raw_path = path_for(cfg, "raw")
    raw = read_raw(raw_path, cfg["data"]["header_row"])
    df = clean(raw, cfg)
    validate(df, cfg["data"]["target"])
    interim = path_for(cfg, "interim")
    interim.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(interim)
    results = path_for(cfg, "results")
    results.mkdir(parents=True, exist_ok=True)
    (results / "data_summary.json").write_text(
        json.dumps(summarise(raw, df, raw_path, cfg), indent=2) + "\n"
    )
    return df


def main() -> None:
    args = parse_args("Convert the raw Excel file to a clean CSV.")
    df = build(load_config(args.config))
    print(f"wrote {len(df)} rows")


if __name__ == "__main__":
    main()
