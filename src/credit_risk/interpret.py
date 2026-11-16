import json
from typing import Any

import joblib
import numpy as np

from credit_risk.config import load_config, parse_args, path_for
from credit_risk.explain import Explainer, label
from credit_risk.split import load_train


def run_interpretation(cfg: dict[str, Any]) -> dict[str, Any]:
    """Global SHAP on a training sample with the fitted model."""
    X, _ = load_train(cfg)
    model = joblib.load(path_for(cfg, "artifacts") / "model.joblib")
    n = min(cfg["interpretability"]["shap_sample"], len(X))
    sample = X.sample(n=n, random_state=cfg["seed"])
    explainer = Explainer(model)
    contrib = explainer.contributions(sample)
    values = explainer.feature_values(sample)[contrib.columns]
    importance = contrib.abs().mean().sort_values(ascending=False)
    results = path_for(cfg, "results")
    np.savez_compressed(
        results / "shap_values.npz",
        shap=contrib.to_numpy(dtype=np.float32),
        values=values.to_numpy(dtype=np.float32),
        features=np.array(list(contrib.columns), dtype="U40"),
        base_value=explainer.base_value,
    )
    summary = {
        "n_samples": n,
        "base_value": explainer.base_value,
        "units": "model margin (log-odds) for gradient-boosted trees",
        "importance": [
            {"feature": f, "label": label(f), "mean_abs_shap": float(v)}
            for f, v in importance.items()
        ],
    }
    (results / "shap_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args("Compute global SHAP values.")
    run_interpretation(load_config(args.config))


if __name__ == "__main__":
    main()
