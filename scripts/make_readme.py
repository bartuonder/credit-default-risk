"""Render README.md from docs/readme_template.md using files in results/ and artifacts/."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient  # noqa: E402

from credit_risk.api.main import create_app  # noqa: E402
from credit_risk.api.schemas import EXAMPLE_HIGH_RISK  # noqa: E402
from credit_risk.api.settings import Settings  # noqa: E402
from credit_risk.config import ROOT, load_config, path_for  # noqa: E402
from make_report import GROUP_LABELS, MODEL_LABELS, ms, table  # noqa: E402


def load(cfg, name, base="results"):
    return json.loads((path_for(cfg, base) / name).read_text())


def main() -> None:
    cfg = load_config()
    data, split = load(cfg, "data_summary.json"), load(cfg, "split.json")
    final, thr = load(cfg, "final_metrics.json"), load(cfg, "threshold.json")
    cal, abl = load(cfg, "calibration.json"), load(cfg, "ablation.json")
    tuned, default = load(cfg, "cv_summary_tuned.json"), load(cfg, "cv_summary_default.json")
    tests, sel = load(cfg, "paired_tests.json"), load(cfg, "selection.json")
    shap_summary, fair = load(cfg, "shap_summary.json"), load(cfg, "fairness_test.json")
    tuning = load(cfg, "tuning.json")
    m, ci = final["metrics"], final["bootstrap_ci95"]

    def line(k, d=3):
        return f"{m[k]:.{d}f} ({ci[k][0]:.{d}f} to {ci[k][1]:.{d}f})"

    results_table = table(
        ["Item", "Value"],
        [
            ["Model", f"{MODEL_LABELS[final['model']]} ({final['calibration']} probabilities)"],
            ["ROC-AUC", line("roc_auc")],
            ["PR-AUC", line("pr_auc")],
            ["Brier score", line("brier", 4)],
            ["Log loss", line("log_loss", 4)],
            ["Decision threshold", f"{final['threshold']:.3f}"],
            ["Precision / recall at threshold", f"{line('precision', 2)} / {line('recall', 2)}"],
            [
                "Expected cost per client",
                f"{final['expected_cost_per_client']:.3f} (flag nobody {final['cost_flag_nobody']:.3f}, "
                f"flag everybody {final['cost_flag_everybody']:.3f})",
            ],
        ],
    )

    def pair(a, b, metric):
        return next(t for t in tests if t["a"] == a and t["b"] == b and t["metric"] == metric)

    ranked = sorted(tuned, key=lambda k: -tuned[k]["pr_auc"]["mean"])
    best = sel["selected"]
    lr_gap = pair(best, "logistic_regression", "pr_auc")["corrected_ttest"]["mean_diff"]
    pvals = ", ".join(
        f"{t['corrected_ttest']['p_value']:.2f}"
        for t in tests
        if t["metric"] == "pr_auc" and "logistic_regression" not in (t["a"], t["b"])
    )
    alpha = cfg["selection"]["alpha"]
    assert all(float(v) >= alpha for v in pvals.split(", ")), (
        "template text assumes the tuned ensembles tie"
    )
    drop_rep = -abl["drop_group"]["repayment_status"]["delta"]["pr_auc"]["mean"]
    others = max(
        -abl["drop_group"][g]["delta"]["pr_auc"]["mean"]
        for g in abl["drop_group"]
        if g != "repayment_status"
    )
    top_feature = shap_summary["importance"][0]["label"].lower()
    raw, iso = cal["methods"]["raw"], cal["methods"]["isotonic"]
    findings = "\n".join(
        [
            f"- The selected {MODEL_LABELS[best]} model reaches ROC-AUC {m['roc_auc']:.3f} and PR-AUC {m['pr_auc']:.3f} on the hold-out set; "
            f"the cross-validated estimates ({tuned[best]['roc_auc']['mean']:.3f} and {tuned[best]['pr_auc']['mean']:.3f}) are consistent with it.",
            f"- Tree ensembles beat logistic regression by {lr_gap:.3f} PR-AUC in cross-validation, but the three tuned ensembles are not significantly different from each other, so the smaller one was chosen.",
            f"- Repayment status dominates: dropping that feature group lowers PR-AUC by {drop_rep:.3f}, against at most {others:.3f} for any other group, and the top SHAP feature is {top_feature}.",
            f"- Raw gradient-boosting probabilities are already well calibrated (cross-validated ECE {raw['ece']['mean']:.3f}; isotonic {iso['ece']['mean']:.3f}), so no calibrator was kept; test-set ECE is {m['ece']:.3f}.",
            f"- With a {thr['cost_fn'] / thr['cost_fp']:g}:1 cost ratio the threshold is {thr['threshold']:.2f}, flagging {final['flagged_share']:.0%} of test clients at "
            f"{final['expected_cost_per_client']:.3f} cost per client, versus {final['cost_flag_nobody']:.3f} when flagging nobody.",
        ]
    )

    comparison = table(
        ["Model", "ROC-AUC", "PR-AUC", "Brier", "Log loss"],
        [
            [
                MODEL_LABELS[n] + " (default)",
                *[ms(s[k]) for k in ("roc_auc", "pr_auc", "brier", "log_loss")],
            ]
            for n, s in default.items()
        ]
        + [
            [
                MODEL_LABELS[n] + " (tuned)",
                *[ms(tuned[n][k]) for k in ("roc_auc", "pr_auc", "brier", "log_loss")],
            ]
            for n in ranked + ["logistic_regression"]
            if n in tuned and n != "logistic_regression"
        ]
        + [
            [
                MODEL_LABELS["logistic_regression"] + " (tuned)",
                *[
                    ms(tuned["logistic_regression"][k])
                    for k in ("roc_auc", "pr_auc", "brier", "log_loss")
                ],
            ]
        ],
    )
    cmat = final["confusion"]
    test_table = table(
        ["Metric", "Value (95% CI)"],
        [
            ["ROC-AUC", line("roc_auc")],
            ["PR-AUC", line("pr_auc")],
            ["Brier score", line("brier", 4)],
            ["Log loss", line("log_loss", 4)],
            ["Precision at threshold", line("precision")],
            ["Recall at threshold", line("recall")],
            [
                "Confusion (TP / FP / FN / TN)",
                f"{cmat['tp']} / {cmat['fp']} / {cmat['fn']} / {cmat['tn']}",
            ],
        ],
    )
    calibration_table = table(
        ["Method", "Brier", "Log loss", "ECE"],
        [
            [k, ms(v["brier"], 5), ms(v["log_loss"], 5), ms(v["ece"], 4)]
            for k, v in cal["methods"].items()
        ],
    )
    sensitivity = table(
        ["FN:FP", "Optimal threshold", "Flagged", "Cost", "Cost at the chosen threshold"],
        [
            [
                f"{r['ratio']:g}",
                f"{r['threshold']:.3f}",
                f"{r['flagged_share']:.1%}",
                f"{r['expected_cost']:.3f}",
                f"{r['cost_at_reference_threshold']:.3f}",
            ]
            for r in thr["sensitivity"]
        ],
    )
    ablation = table(
        ["Group", "PR-AUC drop when removed", "ROC-AUC drop when removed", "PR-AUC alone"],
        [
            [
                GROUP_LABELS[g],
                f"{-v['delta']['pr_auc']['mean']:.4f}",
                f"{-v['delta']['roc_auc']['mean']:.4f}",
                f"{abl['only_group'][g]['scores']['pr_auc']['mean']:.3f}",
            ]
            for g, v in abl["drop_group"].items()
        ],
    )
    ranges = "; ".join(
        f"{attr.replace('_', ' ').lower()} {info['approval_rate_range'][0]:.0%} to {info['approval_rate_range'][1]:.0%}"
        for attr, info in fair["attributes"].items()
    )
    fairness_text = (
        f"Approval rates on the test set at threshold {fair['threshold']:.2f}, across groups with at least 100 clients: {ranges}. "
        f"{fair['note']}"
    )

    app = create_app(Settings())
    with TestClient(app) as client:
        body = {k: v for k, v in EXAMPLE_HIGH_RISK.items()}
        response = client.post("/predict?explain=true&top_k=3", json=body).json()
        max_batch = app.state.settings.max_batch_size
    for key in ("default_probability",):
        response[key] = round(response[key], 4)
    for item in response["explanation"]:
        item["contribution"] = round(item["contribution"], 3)
        item["value"] = (
            round(item["value"], 3) if isinstance(item["value"], float) else item["value"]
        )

    values = {
        "n_test": f"{split['n_test']:,}",
        "n_train": f"{split['n_train']:,}",
        "n_rows": f"{data['n_rows']:,}",
        "default_rate": f"{data['target_rate']:.1%}",
        "n_splits": cfg["cv"]["n_splits"],
        "n_repeats": cfg["cv"]["n_repeats"],
        "tuning_iter": tuning["n_iter"],
        "tuning_splits": tuning["tuning_splits"],
        "cost_ratio": f"{thr['cost_fn'] / thr['cost_fp']:g}",
        "threshold": f"{thr['threshold']:.3f}",
        "bayes": f"{thr['bayes_threshold']:.3f}",
        "results_table": results_table,
        "findings": findings,
        "comparison_table": comparison,
        "pvalues": pvals,
        "lr_gap": f"{lr_gap:.3f}",
        "test_table": test_table,
        "calibration_table": calibration_table,
        "sensitivity_table": sensitivity,
        "ablation_table": ablation,
        "fairness_text": fairness_text,
        "max_batch": max_batch,
        "example_request": json.dumps(body, separators=(",", ":")),
        "example_response": json.dumps(response, indent=2),
        "seed": cfg["seed"],
        "sha": data["raw_sha256"],
    }
    text = (ROOT / "docs" / "readme_template.md").read_text()
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", str(v))
    assert "{{" not in text, "unfilled placeholder"
    (ROOT / "README.md").write_text(text)
    print("wrote README.md")


if __name__ == "__main__":
    main()
