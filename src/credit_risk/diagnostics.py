import json
from typing import Any

import numpy as np

from credit_risk.calibration import load_calibration
from credit_risk.config import load_config, parse_args, path_for
from credit_risk.cost import load_threshold
from credit_risk.errors import error_analysis
from credit_risk.fairness import fairness_report
from credit_risk.split import load_train


def run_oof_diagnostics(cfg: dict[str, Any]) -> None:
    """Fairness and error analysis on training-set out-of-fold predictions (repeat-averaged)."""
    X, y = load_train(cfg)
    results = path_for(cfg, "results")
    oof = np.load(results / "cache" / "oof_calibration.npz")[load_calibration(cfg)["selected"]]
    p = oof.mean(axis=0)
    threshold = load_threshold(cfg)["threshold"]
    (results / "fairness_oof.json").write_text(
        json.dumps(fairness_report(X, y, p, threshold, cfg), indent=2) + "\n"
    )
    (results / "errors_oof.json").write_text(
        json.dumps(error_analysis(X, y, p, threshold, cfg), indent=2) + "\n"
    )


def main() -> None:
    args = parse_args("Fairness and error analysis on out-of-fold predictions.")
    run_oof_diagnostics(load_config(args.config))


if __name__ == "__main__":
    main()
