import argparse
from collections.abc import Callable
from typing import Any

from credit_risk import (
    ablation,
    calibration,
    compare,
    cost,
    data,
    diagnostics,
    final,
    interpret,
    selection,
    split,
    tuning,
)
from credit_risk.config import load_config, path_for
from credit_risk.models import MODEL_NAMES


def _baselines(cfg: dict[str, Any]) -> None:
    compare.save_comparison(cfg, compare.run_comparison(cfg, MODEL_NAMES), "default")


TRAIN_STAGES: list[tuple[str, str, Callable[[dict[str, Any]], Any]]] = [
    ("split", "results/split.json", split.ensure_split),
    ("baselines", "results/cv_summary_default.json", _baselines),
    ("tuning", "results/tuning.json", tuning.run_tuning),
    ("selection", "results/selection.json", selection.run_selection),
    ("calibration", "results/calibration.json", calibration.run_calibration),
    ("threshold", "results/threshold.json", cost.run_threshold),
    ("ablation", "results/ablation.json", ablation.run_ablation),
    ("fit", "artifacts/model.joblib", final.fit_final),
    ("interpretation", "results/shap_summary.json", interpret.run_interpretation),
    ("diagnostics", "results/errors_oof.json", diagnostics.run_oof_diagnostics),
]
EVAL_STAGES = [("evaluation", "results/final_metrics.json", final.run_evaluation)]


def run(cfg: dict[str, Any], stages, force: bool) -> None:
    root = path_for(cfg, "results").parent
    (path_for(cfg, "results") / "cache").mkdir(parents=True, exist_ok=True)
    for name, output, fn in stages:
        if (root / output).exists() and not force:
            print(f"[skip] {name}: {output} exists")
            continue
        print(f"[run] {name}")
        fn(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the training or evaluation stages.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", choices=["train", "evaluate"], required=True)
    parser.add_argument("--force", action="store_true", help="rerun stages whose outputs exist")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not path_for(cfg, "interim").exists():
        data.build(cfg)
    run(cfg, TRAIN_STAGES if args.stage == "train" else EVAL_STAGES, args.force)


if __name__ == "__main__":
    main()
