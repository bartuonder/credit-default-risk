import json
import platform
import subprocess
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn

from credit_risk import __version__, schema
from credit_risk.calibration import (
    brier_decomposition,
    load_calibration,
    reliability_curve,
    wrap_calibration,
)
from credit_risk.config import ROOT, load_config, parse_args, path_for
from credit_risk.cost import expected_cost, load_threshold
from credit_risk.errors import error_analysis
from credit_risk.fairness import fairness_report
from credit_risk.metrics import score_probabilities
from credit_risk.models import make_pipeline
from credit_risk.selection import load_selection
from credit_risk.split import load_test, load_train

BOOTSTRAP = 1000


def fit_final(cfg: dict[str, Any]) -> None:
    """Fit the chosen pipeline, with its calibration, on the full training set."""
    X, y = load_train(cfg)
    sel, cal = load_selection(cfg), load_calibration(cfg)
    pipe = make_pipeline(sel["selected"], cfg, sel["params"], n_jobs=4)
    model = wrap_calibration(pipe, cal["selected"], cfg["calibration"]["cv"]).fit(X, y)
    out = path_for(cfg, "artifacts")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib", compress=3)
    size = (out / "model.joblib").stat().st_size / 1e6
    print(f"[fit] wrote artifacts/model.joblib ({size:.1f} MB)")


def _bootstrap(y: np.ndarray, p: np.ndarray, threshold: float, cfg: dict[str, Any], seed: int):
    rng = np.random.default_rng(seed)
    d = cfg["decision"]
    keys = ["roc_auc", "pr_auc", "brier", "log_loss", "precision", "recall", "expected_cost"]
    draws = {k: [] for k in keys}
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        s = score_probabilities(y[idx], p[idx], threshold)
        s["expected_cost"] = expected_cost(y[idx], p[idx], threshold, d["cost_fn"], d["cost_fp"])
        for k in keys:
            draws[k].append(s[k])
    return {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()
    }


def feature_schema() -> list[dict[str, Any]]:
    rows = [{"name": schema.LIMIT, "type": "integer", "min": 1}]
    for name, values in schema.CATEGORY_VALUES.items():
        rows.append(
            {"name": name, "type": "category", "values": {str(k): v for k, v in values.items()}}
        )
    rows.append(
        {
            "name": schema.AGE,
            "type": "integer",
            "min": schema.AGE_RANGE[0],
            "max": schema.AGE_RANGE[1],
        }
    )
    lo, hi = schema.PAY_STATUS_RANGE
    rows += [{"name": c, "type": "integer", "min": lo, "max": hi} for c in schema.PAY_STATUS]
    rows += [{"name": c, "type": "integer"} for c in schema.BILL_AMT]
    rows += [{"name": c, "type": "integer", "min": 0} for c in schema.PAY_AMT]
    return rows


def _git_commit() -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return head
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_evaluation(cfg: dict[str, Any]) -> dict[str, Any]:
    """Score the hold-out set once and write final metrics, diagnostics and the model card."""
    model = joblib.load(path_for(cfg, "artifacts") / "model.joblib")
    sel, cal, thr = load_selection(cfg), load_calibration(cfg), load_threshold(cfg)
    X, y = load_test(cfg)
    p = model.predict_proba(X)[:, 1]
    yv = y.to_numpy()
    t, d = thr["threshold"], cfg["decision"]
    flag = p >= t
    metrics = score_probabilities(yv, p, t, cfg["metrics"]["ece_bins"])
    cost = expected_cost(yv, p, t, d["cost_fn"], d["cost_fp"])
    final = {
        "model": sel["selected"],
        "calibration": cal["selected"],
        "n_test": len(yv),
        "test_default_rate": float(yv.mean()),
        "threshold": t,
        "metrics": metrics,
        "confusion": {
            "tp": int(np.sum(flag & (yv == 1))),
            "fp": int(np.sum(flag & (yv == 0))),
            "fn": int(np.sum(~flag & (yv == 1))),
            "tn": int(np.sum(~flag & (yv == 0))),
        },
        "flagged_share": float(flag.mean()),
        "expected_cost_per_client": cost,
        "cost_flag_nobody": float(d["cost_fn"] * yv.mean()),
        "cost_flag_everybody": float(d["cost_fp"] * (1 - yv.mean())),
        "cost_fn": d["cost_fn"],
        "cost_fp": d["cost_fp"],
        "bootstrap_ci95": _bootstrap(yv, p, t, cfg, cfg["seed"]),
        "bootstrap_resamples": BOOTSTRAP,
        "reliability": reliability_curve(yv, p),
        "brier_decomposition": brier_decomposition(yv, p),
    }
    results = path_for(cfg, "results")
    (results / "final_metrics.json").write_text(json.dumps(final, indent=2) + "\n")
    (results / "fairness_test.json").write_text(
        json.dumps(fairness_report(X, y, p, t, cfg), indent=2) + "\n"
    )
    (results / "errors_test.json").write_text(
        json.dumps(error_analysis(X, y, p, t, cfg), indent=2) + "\n"
    )
    pd.DataFrame({"id": X.index, "probability": p, "default": yv}).to_csv(
        results / "cache" / "test_predictions.csv", index=False
    )
    write_model_card(cfg, final, sel, cal, thr)
    print(
        f"[evaluate] test roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f} "
        f"brier={metrics['brier']:.4f} cost/client={cost:.4f}"
    )
    return final


def write_model_card(cfg, final, sel, cal, thr) -> None:
    summary = json.loads((path_for(cfg, "results") / "data_summary.json").read_text())
    card = {
        "name": "credit-default-risk",
        "version": __version__,
        "model": sel["selected"],
        "hyperparameters": sel["params"],
        "calibration": cal["selected"],
        "threshold": thr["threshold"],
        "cost_assumption": {
            "false_negative": thr["cost_fn"],
            "false_positive": thr["cost_fp"],
            "note": thr["assumption"],
        },
        "risk_bands": cfg["decision"]["risk_bands"],
        "metrics_test": final["metrics"],
        "metrics_test_ci95": final["bootstrap_ci95"],
        "confusion_test": final["confusion"],
        "expected_cost_per_client_test": final["expected_cost_per_client"],
        "calibration_curve_test": final["reliability"],
        "brier_decomposition_test": final["brier_decomposition"],
        "n_train": json.loads((path_for(cfg, "results") / "split.json").read_text())["n_train"],
        "n_test": final["n_test"],
        "feature_schema": feature_schema(),
        "data": {"raw_sha256": summary["raw_sha256"], "n_rows": summary["n_rows"]},
        "libraries": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "shap": shap.__version__,
        },
        "git_commit": _git_commit(),
        "intended_use": "Educational demonstration on 2005 Taiwanese data; not for lending decisions.",
    }
    out = path_for(cfg, "artifacts") / "model_card.json"
    out.write_text(json.dumps(card, indent=2) + "\n")


def main() -> None:
    args = parse_args("Fit the final model or evaluate it on the hold-out set.")
    cfg = load_config(args.config)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
