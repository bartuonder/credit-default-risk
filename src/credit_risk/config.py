import argparse
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("CREDIT_RISK_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML configuration; relative paths are resolved against the project root."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = ROOT / path
    with open(path) as fh:
        return yaml.safe_load(fh)


def path_for(cfg: dict[str, Any], key: str) -> Path:
    """Absolute path of an entry in `paths`."""
    return ROOT / cfg["paths"][key]


def parse_args(description: str, **extra: Any) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None)
    for name, kwargs in extra.items():
        parser.add_argument(f"--{name}", **kwargs)
    return parser.parse_args()
