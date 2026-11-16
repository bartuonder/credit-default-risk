import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from credit_risk.compare import save_comparison
from credit_risk.config import load_config, parse_args, path_for
from credit_risk.cv import cross_validate, make_cv, summarise_folds
from credit_risk.models import make_pipeline
from credit_risk.selection import compare_pair, load_selection
from credit_risk.split import load_train


def wrap_calibration(pipeline, method: str, cv: int):
    """Cross-validated calibration around the full pipeline; `raw` returns it unchanged."""
    if method == "raw":
        return pipeline
    return CalibratedClassifierCV(estimator=pipeline, method=method, cv=cv)


def reliability_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, list[float]]:
    """Equal-frequency reliability bins: mean predicted probability vs observed default rate."""
    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, bins)
    return {
        "mean_predicted": [float(p[g].mean()) for g in groups],
        "observed_rate": [float(y[g].mean()) for g in groups],
        "count": [int(len(g)) for g in groups],
    }


def brier_decomposition(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Murphy decomposition over equal-width bins: brier ~ reliability - resolution + uncertainty."""
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    base = y.mean()
    rel = res = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            w = m.mean()
            rel += w * (p[m].mean() - y[m].mean()) ** 2
            res += w * (y[m].mean() - base) ** 2
    unc = base * (1 - base)
    brier = float(np.mean((p - y) ** 2))
    return {
        "brier": brier,
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
        "within_bin_residual": float(brier - (rel - res + unc)),
    }


def run_calibration(cfg: dict[str, Any]) -> dict[str, Any]:
    X, y = load_train(cfg)
    sel = load_selection(cfg)
    cv = make_cv(cfg)
    methods = ["raw", *cfg["calibration"]["methods"]]
    results = {}
    for method in methods:
        pipe = make_pipeline(sel["selected"], cfg, sel["params"], n_jobs=1)
        est = wrap_calibration(pipe, method, cfg["calibration"]["cv"])
        results[method] = cross_validate(est, X, y, cv, cfg["metrics"]["ece_bins"])
        s = summarise_folds(results[method].folds)
        print(
            f"[calibrate] {method}: brier={s['brier']['mean']:.5f} log_loss={s['log_loss']['mean']:.5f}"
        )
    save_comparison(cfg, results, "calibration")

    folds = pd.concat([r.folds.assign(model=m) for m, r in results.items()], ignore_index=True)
    n = len(y)
    n_test = n // cfg["cv"]["n_splits"]
    alpha = cfg["selection"]["alpha"]
    best = min(methods, key=lambda m: folds.loc[folds["model"] == m, "log_loss"].mean())
    chosen = "raw"
    if best != "raw":
        test = compare_pair(folds, "raw", best, "log_loss", n - n_test, n_test)
        # a positive raw - best difference means the calibrated model has the lower loss
        if test["corrected_ttest"]["mean_diff"] > 0 and test["corrected_ttest"]["p_value"] < alpha:
            chosen = best
    yv = y.to_numpy()
    out = {
        "model": sel["selected"],
        "methods": {m: summarise_folds(r.folds) for m, r in results.items()},
        "paired_vs_raw": {
            m: {
                k: compare_pair(folds, "raw", m, k, n - n_test, n_test)
                for k in ("brier", "log_loss")
            }
            for m in methods
            if m != "raw"
        },
        "lowest_log_loss": best,
        "selected": chosen,
        "rule": (
            "lowest mean log loss; a calibration method replaces the raw model only if the "
            f"improvement is significant (corrected resampled t-test, alpha={alpha})"
        ),
        "reliability": {
            m: reliability_curve(np.tile(yv, results[m].oof.shape[0]), results[m].oof.ravel())
            for m in methods
        },
        "brier_decomposition": {m: _mean_decomposition(yv, results[m].oof) for m in methods},
    }
    (path_for(cfg, "results") / "calibration.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[calibrate] selected: {chosen}")
    return out


def _mean_decomposition(y: np.ndarray, oof: np.ndarray) -> dict[str, float]:
    parts = [brier_decomposition(y, oof[r]) for r in range(oof.shape[0])]
    return {k: float(np.mean([p[k] for p in parts])) for k in parts[0]}


def load_calibration(cfg: dict[str, Any]) -> dict[str, Any]:
    return json.loads((path_for(cfg, "results") / "calibration.json").read_text())


def main() -> None:
    args = parse_args("Compare probability calibration methods in cross-validation.")
    run_calibration(load_config(args.config))


if __name__ == "__main__":
    main()
