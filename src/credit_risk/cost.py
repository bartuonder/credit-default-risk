import json
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.calibration import load_calibration
from credit_risk.config import load_config, parse_args, path_for
from credit_risk.cv import CVResult, fold_scores_at_threshold, make_cv, summarise_folds
from credit_risk.split import load_train

GRID = np.round(np.arange(0.02, 0.9801, 0.005), 3)


def expected_cost(
    y: np.ndarray, p: np.ndarray, threshold: float, cost_fn: float, cost_fp: float
) -> float:
    """Mean cost per client when flagging p >= threshold as default."""
    flag = p >= threshold
    fn = np.sum((~flag) & (y == 1))
    fp = np.sum(flag & (y == 0))
    return float((cost_fn * fn + cost_fp * fp) / len(y))


def cost_curve(
    y: np.ndarray, oof: np.ndarray, cost_fn: float, cost_fp: float, grid=GRID
) -> np.ndarray:
    """Expected cost per threshold, averaged over the repeats of out-of-fold predictions."""
    oof = np.atleast_2d(oof)
    return np.array(
        [np.mean([expected_cost(y, p, t, cost_fn, cost_fp) for p in oof]) for t in grid]
    )


def best_threshold(
    y: np.ndarray, oof: np.ndarray, cost_fn: float, cost_fp: float, grid=GRID
) -> float:
    curve = cost_curve(y, oof, cost_fn, cost_fp, grid)
    return float(grid[int(np.argmin(curve))])


def bayes_threshold(cost_fn: float, cost_fp: float) -> float:
    """Optimal cut-off for perfectly calibrated probabilities."""
    return cost_fp / (cost_fp + cost_fn)


def sensitivity(y: np.ndarray, oof: np.ndarray, ratios, ref_threshold: float, cost_fp: float = 1.0):
    """For each FN:FP cost ratio, the optimal threshold and the regret of keeping `ref_threshold`."""
    prevalence = float(y.mean())
    rows = []
    for r in ratios:
        cost_fn = r * cost_fp
        t = best_threshold(y, oof, cost_fn, cost_fp)
        best = float(
            np.mean([expected_cost(y, p, t, cost_fn, cost_fp) for p in np.atleast_2d(oof)])
        )
        ref = float(
            np.mean(
                [expected_cost(y, p, ref_threshold, cost_fn, cost_fp) for p in np.atleast_2d(oof)]
            )
        )
        flagged = float(np.mean([(p >= t).mean() for p in np.atleast_2d(oof)]))
        rows.append(
            {
                "ratio": float(r),
                "threshold": t,
                "bayes_threshold": bayes_threshold(cost_fn, cost_fp),
                "expected_cost": best,
                "cost_at_reference_threshold": ref,
                "flagged_share": flagged,
                "cost_flag_nobody": float(cost_fn * prevalence),
                "cost_flag_everybody": float(cost_fp * (1 - prevalence)),
            }
        )
    return rows


def run_threshold(cfg: dict[str, Any]) -> dict[str, Any]:
    _, y = load_train(cfg)
    yv = y.to_numpy()
    cal = load_calibration(cfg)
    cache = np.load(path_for(cfg, "results") / "cache" / "oof_calibration.npz")
    oof = cache[cal["selected"]]
    d = cfg["decision"]
    curve = cost_curve(yv, oof, d["cost_fn"], d["cost_fp"])
    threshold = float(GRID[int(np.argmin(curve))])
    cv = make_cv(cfg)
    at_t = fold_scores_at_threshold(CVResult(pd.DataFrame(), oof), y, cv, threshold)
    out = {
        "calibration": cal["selected"],
        "cost_fn": d["cost_fn"],
        "cost_fp": d["cost_fp"],
        "assumption": "False negative = {:g} x false positive; an assumption, not a measured cost.".format(
            d["cost_fn"] / d["cost_fp"]
        ),
        "threshold": threshold,
        "bayes_threshold": bayes_threshold(d["cost_fn"], d["cost_fp"]),
        "expected_cost_at_threshold": float(curve.min()),
        "oof_cost_flag_nobody": float(d["cost_fn"] * yv.mean()),
        "oof_cost_flag_everybody": float(d["cost_fp"] * (1 - yv.mean())),
        "oof_precision_recall": summarise_folds(at_t),
        "flagged_share": float(np.mean([(p >= threshold).mean() for p in oof])),
        "curve": {"threshold": GRID.tolist(), "expected_cost": curve.tolist()},
        "curves_by_ratio": {
            str(r): cost_curve(yv, oof, r * d["cost_fp"], d["cost_fp"]).tolist()
            for r in (1, 2, 5, 10)
        },
        "sensitivity": sensitivity(yv, oof, d["ratios"], threshold, d["cost_fp"]),
    }
    (path_for(cfg, "results") / "threshold.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[threshold] {threshold} (bayes {out['bayes_threshold']:.3f})")
    return out


def load_threshold(cfg: dict[str, Any]) -> dict[str, Any]:
    return json.loads((path_for(cfg, "results") / "threshold.json").read_text())


def main() -> None:
    args = parse_args("Choose the decision threshold on out-of-fold predictions.")
    run_threshold(load_config(args.config))


if __name__ == "__main__":
    main()
