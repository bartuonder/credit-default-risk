import json
from typing import Any

from credit_risk.config import load_config, parse_args, path_for
from credit_risk.cv import cross_validate, make_cv, summarise_folds
from credit_risk.models import make_pipeline
from credit_risk.selection import load_selection
from credit_risk.split import load_train

METRICS = ("pr_auc", "roc_auc")


def _delta(full, other) -> dict[str, dict[str, float]]:
    """Paired fold-wise change of `other` relative to `full`."""
    out = {}
    for m in METRICS:
        d = other.folds[m].to_numpy() - full.folds[m].to_numpy()
        out[m] = {"mean": float(d.mean()), "std": float(d.std(ddof=1))}
    return out


def run_ablation(cfg: dict[str, Any]) -> dict[str, Any]:
    """Drop one feature group at a time, and keep one group at a time, with shared folds."""
    X, y = load_train(cfg)
    sel = load_selection(cfg)
    groups = list(cfg["features"]["groups"])
    cv = make_cv(cfg, n_repeats=cfg["ablation"]["n_repeats"])

    def run(drop: tuple[str, ...]):
        pipe = make_pipeline(sel["selected"], cfg, sel["params"], drop_groups=drop)
        return cross_validate(pipe, X, y, cv, cfg["metrics"]["ece_bins"])

    full = run(())
    out: dict[str, Any] = {
        "model": sel["selected"],
        "n_repeats": cfg["ablation"]["n_repeats"],
        "full": summarise_folds(full.folds),
        "drop_group": {},
        "only_group": {},
    }
    for g in groups:
        dropped = run((g,))
        out["drop_group"][g] = {
            "scores": summarise_folds(dropped.folds),
            "delta": _delta(full, dropped),
        }
        kept = run(tuple(x for x in groups if x != g))
        out["only_group"][g] = {"scores": summarise_folds(kept.folds), "delta": _delta(full, kept)}
        print(
            f"[ablation] {g}: drop d_pr_auc={out['drop_group'][g]['delta']['pr_auc']['mean']:+.4f}"
        )
    (path_for(cfg, "results") / "ablation.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main() -> None:
    args = parse_args("Feature-group ablation.")
    run_ablation(load_config(args.config))


if __name__ == "__main__":
    main()
