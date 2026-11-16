import json
from typing import Any

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from credit_risk.config import load_config, parse_args, path_for
from credit_risk.models import make_pipeline
from credit_risk.split import load_train

TUNED = ["logistic_regression", "random_forest", "hist_gradient_boosting", "lightgbm"]


def tune_model(cfg: dict[str, Any], name: str, X, y) -> dict[str, Any]:
    """Random search on the training set with folds that differ from the evaluation folds."""
    space = cfg["models"][name]["search"]
    folds = StratifiedKFold(
        cfg["cv"]["tuning_splits"],
        shuffle=True,
        random_state=cfg["seed"] + cfg["cv"]["tuning_seed_offset"],
    )
    search = RandomizedSearchCV(
        make_pipeline(name, cfg),
        {f"model__{k}": v for k, v in space.items()},
        n_iter=min(cfg["tuning"]["n_iter"], int(np.prod([len(v) for v in space.values()]))),
        scoring=cfg["tuning"]["scoring"],
        cv=folds,
        random_state=cfg["seed"],
        n_jobs=4,
        refit=False,
    )
    search.fit(X, y)
    res = search.cv_results_
    trials = [
        {
            "params": {k.removeprefix("model__"): _plain(v) for k, v in p.items()},
            "mean_score": float(m),
            "std_score": float(s),
        }
        for p, m, s in zip(
            res["params"], res["mean_test_score"], res["std_test_score"], strict=True
        )
    ]
    best = {k.removeprefix("model__"): _plain(v) for k, v in search.best_params_.items()}
    print(f"[tune] {name}: best {cfg['tuning']['scoring']}={search.best_score_:.4f} {best}")
    return {
        "search_space": space,
        "best_params": best,
        "best_score": float(search.best_score_),
        "trials": sorted(trials, key=lambda t: -t["mean_score"]),
    }


def _plain(v):
    return v.item() if isinstance(v, np.generic) else v


def run_tuning(cfg: dict[str, Any]) -> dict[str, Any]:
    X, y = load_train(cfg)
    out = {
        "n_iter": cfg["tuning"]["n_iter"],
        "scoring": cfg["tuning"]["scoring"],
        "tuning_splits": cfg["cv"]["tuning_splits"],
        "seed": cfg["seed"],
        "models": {name: tune_model(cfg, name, X, y) for name in TUNED},
    }
    path = path_for(cfg, "results") / "tuning.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    return out


def load_tuning(cfg: dict[str, Any]) -> dict[str, Any]:
    return json.loads((path_for(cfg, "results") / "tuning.json").read_text())


def best_params(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {n: m["best_params"] for n, m in load_tuning(cfg)["models"].items()}


def main() -> None:
    args = parse_args("Random-search hyperparameters on the training set.")
    run_tuning(load_config(args.config))


if __name__ == "__main__":
    main()
