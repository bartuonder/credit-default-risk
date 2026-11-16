import json
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.config import load_config, parse_args, path_for
from credit_risk.cv import cross_validate, make_cv, summarise_folds
from credit_risk.models import MODEL_NAMES, make_pipeline
from credit_risk.split import load_train


def run_comparison(
    cfg: dict[str, Any], names: list[str], params: dict | None = None, tag: str = "default"
):
    """Repeated-CV scores and out-of-fold predictions for each named model."""
    X, y = load_train(cfg)
    cv = make_cv(cfg)
    results = {}
    for name in names:
        pipe = make_pipeline(name, cfg, (params or {}).get(name))
        results[name] = cross_validate(pipe, X, y, cv, cfg["metrics"]["ece_bins"])
        s = summarise_folds(results[name].folds)
        print(
            f"[{tag}] {name}: roc_auc={s['roc_auc']['mean']:.4f} pr_auc={s['pr_auc']['mean']:.4f}"
        )
    return results


def save_comparison(cfg: dict[str, Any], results: dict, tag: str) -> None:
    out = path_for(cfg, "results")
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    folds = pd.concat([r.folds.assign(model=n) for n, r in results.items()], ignore_index=True)
    folds.to_csv(out / f"cv_folds_{tag}.csv", index=False)
    summary = {n: summarise_folds(r.folds) for n, r in results.items()}
    (out / f"cv_summary_{tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(cache / f"oof_{tag}.npz", **{n: r.oof for n, r in results.items()})


def main() -> None:
    args = parse_args("Cross-validate baselines and candidates with default parameters.")
    cfg = load_config(args.config)
    save_comparison(cfg, run_comparison(cfg, MODEL_NAMES), "default")


if __name__ == "__main__":
    main()
