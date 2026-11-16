import itertools
import json
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.compare import run_comparison, save_comparison
from credit_risk.config import load_config, parse_args, path_for
from credit_risk.models import CANDIDATES
from credit_risk.stats import corrected_resampled_ttest, wilcoxon_test
from credit_risk.tuning import TUNED, best_params

SCORING_TO_COLUMN = {"average_precision": "pr_auc", "roc_auc": "roc_auc"}
TEST_METRICS = ["pr_auc", "roc_auc", "brier", "log_loss"]


def paired_differences(folds: pd.DataFrame, a: str, b: str, metric: str) -> np.ndarray:
    """Fold-wise differences a - b, aligned on (repeat, fold)."""
    wide = folds.pivot_table(index=["repeat", "fold"], columns="model", values=metric)
    return (wide[a] - wide[b]).to_numpy()


def compare_pair(folds, a, b, metric, n_train, n_test) -> dict[str, Any]:
    d = paired_differences(folds, a, b, metric)
    return {
        "a": a,
        "b": b,
        "metric": metric,
        "n_pairs": int(len(d)),
        "corrected_ttest": corrected_resampled_ttest(d, n_train, n_test),
        "wilcoxon": wilcoxon_test(d),
    }


def choose_model(
    folds: pd.DataFrame, cfg: dict[str, Any], n_train: int, n_test: int
) -> dict[str, Any]:
    """Best mean score; ties within the corrected t-test go to the smallest model by config order."""
    metric = SCORING_TO_COLUMN[cfg["metrics"]["selection"]]
    means = folds[folds["model"].isin(CANDIDATES)].groupby("model")[metric].mean()
    top = means.idxmax()
    alpha = cfg["selection"]["alpha"]
    indistinguishable = [top] + [
        m
        for m in CANDIDATES
        if m != top
        and compare_pair(folds, top, m, metric, n_train, n_test)["corrected_ttest"]["p_value"]
        >= alpha
    ]
    order = cfg["selection"]["tie_break_order"]
    selected = min(indistinguishable, key=order.index)
    return {
        "metric": metric,
        "mean_scores": {m: float(v) for m, v in means.items()},
        "top_by_mean": top,
        "indistinguishable_from_top": indistinguishable,
        "selected": selected,
        "rule": (
            f"highest mean {metric}; models whose paired difference to the top is not significant "
            f"(corrected resampled t-test, alpha={alpha}) are treated as ties and the first in "
            f"tie_break_order is chosen"
        ),
    }


def run_selection(cfg: dict[str, Any]) -> dict[str, Any]:
    params = best_params(cfg)
    results = run_comparison(cfg, TUNED, params, tag="tuned")
    save_comparison(cfg, results, "tuned")
    folds = pd.concat([r.folds.assign(model=n) for n, r in results.items()], ignore_index=True)
    n = results[TUNED[0]].oof.shape[1]
    n_test = n // cfg["cv"]["n_splits"]
    n_train = n - n_test
    choice = choose_model(folds, cfg, n_train, n_test)
    ranked = sorted(CANDIDATES, key=lambda m: -choice["mean_scores"][m])
    pairs = [(a, b) for a, b in itertools.combinations(ranked, 2)]
    pairs += [(ranked[0], "logistic_regression")]
    tests = [compare_pair(folds, a, b, m, n_train, n_test) for a, b in pairs for m in TEST_METRICS]
    out = path_for(cfg, "results")
    (out / "paired_tests.json").write_text(json.dumps(tests, indent=2) + "\n")
    choice["params"] = params[choice["selected"]]
    (out / "selection.json").write_text(json.dumps(choice, indent=2) + "\n")
    print(f"[select] {choice['selected']} (top by mean: {choice['top_by_mean']})")
    return choice


def load_selection(cfg: dict[str, Any]) -> dict[str, Any]:
    return json.loads((path_for(cfg, "results") / "selection.json").read_text())


def main() -> None:
    args = parse_args("Cross-validate tuned models, run paired tests and select one.")
    run_selection(load_config(args.config))


if __name__ == "__main__":
    main()
