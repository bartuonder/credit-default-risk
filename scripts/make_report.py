"""Build reports/report.md and reports/figures/*.png from files in results/ only."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from credit_risk.config import load_config, path_for
from credit_risk.explain import label
from credit_risk.plotting import BLUE, GREEN, GREY, ORANGE, PALETTE, apply_style, save

MODEL_LABELS = {
    "dummy": "Majority class",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Hist. gradient boosting",
    "lightgbm": "LightGBM",
}
GROUP_LABELS = {
    "demographics": "Demographics",
    "credit_limit": "Credit limit",
    "repayment_status": "Repayment status",
    "bill_amounts": "Bill amounts",
    "payment_amounts": "Payment amounts",
}


def ms(d: dict, digits: int = 4) -> str:
    return f"{d['mean']:.{digits}f} ± {d['std']:.{digits}f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


class Report:
    def __init__(self, cfg):
        self.cfg = cfg
        self.res = path_for(cfg, "results")
        self.out = path_for(cfg, "reports")
        self.fig = self.out / "figures"
        self.parts: list[str] = []

    def load(self, name: str):
        return json.loads((self.res / name).read_text())

    def add(self, *chunks: str) -> None:
        self.parts.extend(chunks)

    def image(self, name: str, alt: str) -> str:
        return f"![{alt}](figures/{name})"

    # ---- sections ----

    def data(self):
        d, s = self.load("data_summary.json"), self.load("split.json")
        self.add(
            "## Data and split\n",
            table(
                ["Item", "Value"],
                [
                    ["Clients", f"{d['n_rows']:,}"],
                    ["Default rate", f"{d['target_rate']:.2%}"],
                    [
                        "Training rows",
                        f"{s['n_train']:,} (default rate {s['train_default_rate']:.2%})",
                    ],
                    [
                        "Hold-out test rows",
                        f"{s['n_test']:,} (default rate {s['test_default_rate']:.2%})",
                    ],
                    ["Raw file SHA-256", f"`{d['raw_sha256'][:16]}...`"],
                ],
            ),
        )

    def comparison(self):
        default, tuned = self.load("cv_summary_default.json"), self.load("cv_summary_tuned.json")
        tuning = self.load("tuning.json")
        cv = self.cfg["cv"]
        header = ["Model", "ROC-AUC", "PR-AUC", "Brier", "Log loss"]

        def row(name, s):
            return [
                MODEL_LABELS[name],
                *[ms(s[k]) for k in ("roc_auc", "pr_auc", "brier", "log_loss")],
            ]

        self.add(
            "## Model comparison\n",
            f"Repeated stratified {cv['n_splits']}-fold cross-validation with {cv['n_repeats']} repeats on the "
            "training set; mean ± standard deviation over folds. All models share the same folds.\n",
            "### Default parameters\n",
            table(header, [row(n, default[n]) for n in default]),
            "\n### Tuned parameters\n",
            f"Random search, {tuning['n_iter']} candidates per model at most, {tuning['tuning_splits']}-fold CV "
            f"scored by `{tuning['scoring']}`, on folds that differ from the evaluation folds.\n",
            table(header, [row(n, tuned[n]) for n in tuned]),
        )
        rows = []
        for name, m in tuning["models"].items():
            params = ", ".join(f"{k}={v}" for k, v in m["best_params"].items())
            rows.append([MODEL_LABELS[name], f"{m['best_score']:.4f}", f"`{params}`"])
        self.add(
            "\n### Best hyperparameters\n", table(["Model", "Search PR-AUC", "Parameters"], rows)
        )

        folds = pd.read_csv(self.res / "cv_folds_tuned.csv")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        order = list(tuned)
        for ax, metric, title in zip(
            axes, ["pr_auc", "roc_auc"], ["PR-AUC", "ROC-AUC"], strict=True
        ):
            data = [folds.loc[folds["model"] == m, metric] for m in order]
            ax.boxplot(
                data, tick_labels=[MODEL_LABELS[m].replace(" ", "\n", 1) for m in order], widths=0.5
            )
            ax.set_title(f"{title} per fold (tuned)")
        save(fig, self.fig / "cv_comparison.png")
        self.add("\n" + self.image("cv_comparison.png", "Fold-wise scores"))

        sel = self.load("selection.json")
        tests = self.load("paired_tests.json")
        rows = [
            [
                f"{MODEL_LABELS[t['a']]} vs {MODEL_LABELS[t['b']]}",
                t["metric"],
                f"{t['corrected_ttest']['mean_diff']:+.4f}",
                f"{t['corrected_ttest']['p_value']:.3f}",
                f"{t['wilcoxon']['p_value']:.4f}",
            ]
            for t in tests
            if t["metric"] in ("pr_auc", "roc_auc")
        ]
        self.add(
            "\n### Paired fold-wise comparison\n",
            "Differences are computed fold by fold (same splits). The corrected resampled t-test "
            "(Nadeau and Bengio) accounts for overlapping training sets; the Wilcoxon signed-rank test "
            "does not and is shown for reference.\n",
            table(
                ["Pair (A vs B)", "Metric", "Mean diff (A-B)", "Corrected t-test p", "Wilcoxon p"],
                rows,
            ),
            f"\nSelected model: **{MODEL_LABELS[sel['selected']]}**. Rule: {sel['rule']}.",
        )

    def calibration(self):
        c = self.load("calibration.json")
        rows = []
        for m, s in c["methods"].items():
            b = c["brier_decomposition"][m]
            rows.append(
                [
                    m,
                    ms(s["brier"], 5),
                    ms(s["log_loss"], 5),
                    ms(s["ece"], 5),
                    f"{b['reliability']:.5f}",
                    f"{b['resolution']:.5f}",
                    f"{b['uncertainty']:.5f}",
                ]
            )
        fig, axes = plt.subplots(
            1, len(c["reliability"]), figsize=(3.6 * len(c["reliability"]), 3.4), sharey=True
        )
        for ax, (m, r) in zip(axes, c["reliability"].items(), strict=True):
            ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
            ax.plot(r["mean_predicted"], r["observed_rate"], marker="o", color=BLUE)
            ax.set_title(m)
            ax.set_xlabel("mean predicted probability")
            ax.set_xlim(0, max(r["mean_predicted"]) * 1.1)
            ax.set_ylim(0, max(r["observed_rate"]) * 1.1)
        axes[0].set_ylabel("observed default rate")
        save(fig, self.fig / "calibration_reliability.png")
        pv = c["paired_vs_raw"]
        prow = [
            [
                m,
                f"{pv[m]['log_loss']['corrected_ttest']['mean_diff']:+.5f}",
                f"{pv[m]['log_loss']['corrected_ttest']['p_value']:.3f}",
            ]
            for m in pv
        ]
        self.add(
            "## Probability calibration\n",
            "Calibration is cross-validated (5 inner folds) and wrapped around the full pipeline, so it is "
            "re-fitted inside every outer fold. Equal-frequency reliability bins on pooled out-of-fold "
            "predictions; Brier decomposition uses 10 equal-width bins.\n",
            table(
                ["Method", "Brier", "Log loss", "ECE", "Reliability", "Resolution", "Uncertainty"],
                rows,
            ),
            "\n" + self.image("calibration_reliability.png", "Reliability diagrams"),
            "\nPaired log-loss difference, raw minus calibrated (positive favours calibration):\n",
            table(["Method", "Mean diff", "Corrected t-test p"], prow),
            f"\nSelected: **{c['selected']}**. Rule: {c['rule']}.",
        )

    def cost(self):
        t = self.load("threshold.json")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        ax = axes[0]
        grid = t["curve"]["threshold"]
        for (r, curve), color in zip(t["curves_by_ratio"].items(), PALETTE, strict=False):
            ax.plot(
                grid,
                curve,
                color=color,
                lw=2 if float(r) * self.cfg["decision"]["cost_fp"] == t["cost_fn"] else 1,
                label=f"FN = {r} x FP",
            )
        ax.axvline(t["threshold"], color=ORANGE, ls="--", lw=1)
        ax.set_xlabel("threshold")
        ax.set_ylabel("expected cost per client")
        ax.set_title("Threshold versus expected cost")
        ax.legend()
        ax = axes[1]
        s = t["sensitivity"]
        ax.plot(
            [r["ratio"] for r in s],
            [r["threshold"] for r in s],
            marker="o",
            color=BLUE,
            label="empirical optimum",
        )
        ax.plot(
            [r["ratio"] for r in s],
            [r["bayes_threshold"] for r in s],
            color=GREY,
            ls="--",
            label="1 / (1 + ratio)",
        )
        ax.set_xlabel("cost ratio (FN : FP)")
        ax.set_ylabel("optimal threshold")
        ax.set_title("Sensitivity to the cost ratio")
        ax.legend()
        save(fig, self.fig / "cost_curve.png")
        pr = t["oof_precision_recall"]
        rows = [
            [
                f"{r['ratio']:g}",
                f"{r['threshold']:.3f}",
                f"{r['bayes_threshold']:.3f}",
                f"{r['flagged_share']:.1%}",
                f"{r['expected_cost']:.4f}",
                f"{r['cost_at_reference_threshold']:.4f}",
                f"{r['cost_flag_nobody']:.4f}",
                f"{r['cost_flag_everybody']:.4f}",
            ]
            for r in t["sensitivity"]
        ]
        self.add(
            "## Cost-based decision rule\n",
            f"Assumption: {t['assumption']} Cost of a correct decision is zero. The threshold minimises the "
            "expected cost on out-of-fold training predictions (averaged over repeats).\n",
            table(
                ["Item", "Value"],
                [
                    ["Chosen threshold", f"{t['threshold']:.3f}"],
                    [
                        "Bayes threshold for calibrated probabilities (cost_fp / (cost_fp + cost_fn))",
                        f"{t['bayes_threshold']:.3f}",
                    ],
                    [
                        "Expected cost per client (out-of-fold)",
                        f"{t['expected_cost_at_threshold']:.4f}",
                    ],
                    [
                        "Cost if nobody is flagged / everybody is flagged",
                        f"{t['oof_cost_flag_nobody']:.4f} / {t['oof_cost_flag_everybody']:.4f}",
                    ],
                    ["Share of clients flagged", f"{t['flagged_share']:.1%}"],
                    ["Precision at threshold (out-of-fold)", ms(pr["precision"])],
                    ["Recall at threshold (out-of-fold)", ms(pr["recall"])],
                ],
            ),
            "\n" + self.image("cost_curve.png", "Threshold versus expected cost"),
            "\n### Sensitivity to the cost ratio\n",
            table(
                [
                    "FN:FP",
                    "Optimal threshold",
                    "Bayes threshold",
                    "Flagged",
                    "Cost",
                    "Cost at 5:1 threshold",
                    "Flag nobody",
                    "Flag everybody",
                ],
                rows,
            ),
        )

    def interpretability(self):
        shap_summary = self.load("shap_summary.json")
        data = np.load(self.res / "shap_values.npz", allow_pickle=False)
        values, feats, names = data["shap"], data["values"], [str(f) for f in data["features"]]
        pretty = [label(n) for n in names]
        shap.summary_plot(values, feats, feature_names=pretty, max_display=15, show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        save(fig, self.fig / "shap_summary.png")
        top = [i["feature"] for i in shap_summary["importance"][:4]]
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
        for ax, name in zip(axes, top, strict=True):
            idx = names.index(name)
            shap.dependence_plot(
                idx, values, feats, feature_names=pretty, interaction_index=None, ax=ax, show=False
            )
            ax.set_title(label(name))
        save(fig, self.fig / "shap_dependence.png")
        rows = [
            [i + 1, x["label"], f"{x['mean_abs_shap']:.4f}"]
            for i, x in enumerate(shap_summary["importance"][:10])
        ]
        self.add(
            "## Interpretability\n",
            f"SHAP values for the fitted model on {shap_summary['n_samples']:,} random training rows. Units: "
            f"{shap_summary['units']}. One-hot columns are summed back to their original feature.\n",
            table(["Rank", "Feature", "Mean abs SHAP"], rows),
            "\n" + self.image("shap_summary.png", "SHAP summary"),
            "\n" + self.image("shap_dependence.png", "SHAP dependence for the top features"),
        )

    def ablation(self):
        a = self.load("ablation.json")
        groups = list(a["drop_group"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
        for ax, key, title in zip(
            axes,
            ["drop_group", "only_group"],
            ["Drop one group", "Keep only one group"],
            strict=True,
        ):
            y = np.arange(len(groups))
            ax.barh(
                y - 0.2,
                [-a[key][g]["delta"]["pr_auc"]["mean"] for g in groups],
                height=0.4,
                color=BLUE,
                label="PR-AUC",
            )
            ax.barh(
                y + 0.2,
                [-a[key][g]["delta"]["roc_auc"]["mean"] for g in groups],
                height=0.4,
                color=ORANGE,
                label="ROC-AUC",
            )
            ax.set_yticks(y, [GROUP_LABELS[g] for g in groups])
            ax.set_title(title)
            ax.set_xlabel("drop in score versus the full model")
        axes[0].invert_yaxis()
        axes[0].legend()
        save(fig, self.fig / "ablation.png")
        rows = []
        for g in groups:
            d = a["drop_group"][g]["delta"]
            rows.append(
                [
                    GROUP_LABELS[g],
                    f"{-d['pr_auc']['mean']:.4f} ± {d['pr_auc']['std']:.4f}",
                    f"{-d['roc_auc']['mean']:.4f} ± {d['roc_auc']['std']:.4f}",
                    f"{a['only_group'][g]['scores']['pr_auc']['mean']:.4f}",
                    f"{a['only_group'][g]['scores']['roc_auc']['mean']:.4f}",
                ]
            )
        self.add(
            "## Feature-group ablation\n",
            f"Model: {MODEL_LABELS[a['model']]} (uncalibrated), repeated CV with {a['n_repeats']} repeats on the same "
            f"folds as the full model. Full model: PR-AUC {ms(a['full']['pr_auc'])}, ROC-AUC {ms(a['full']['roc_auc'])}. "
            "Engineered features belong to the group of the information they summarise.\n",
            table(
                [
                    "Group",
                    "Drop in PR-AUC (dropped)",
                    "Drop in ROC-AUC (dropped)",
                    "PR-AUC (only this group)",
                    "ROC-AUC (only this group)",
                ],
                rows,
            ),
            "\n" + self.image("ablation.png", "Feature-group ablation"),
        )

    def fairness(self):
        parts = ["## Fairness diagnostic\n"]
        f = self.load("fairness_test.json")
        parts.append(f"> {f['note']}\n")
        parts.append(
            f"Hold-out test set at threshold {f['threshold']:.3f}. Approval means not flagged. Intervals are 95% Wilson intervals.\n"
        )
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
        rows = []
        for ax, (metric, title) in zip(
            axes,
            [
                ("approval_rate", "Approval rate"),
                ("false_positive_rate", "False positive rate"),
                ("false_negative_rate", "False negative rate"),
            ],
            strict=True,
        ):
            ticks, vals, errs, colors = [], [], [], []
            for ci, (attr, info) in enumerate(f["attributes"].items()):
                for level, r in info["groups"].items():
                    m = r[metric]
                    if m["value"] is None:
                        continue
                    ticks.append(f"{attr.title().replace('_', ' ')}: {level}")
                    vals.append(m["value"])
                    errs.append([m["value"] - m["ci95"][0], m["ci95"][1] - m["value"]])
                    colors.append(PALETTE[ci])
            y = np.arange(len(ticks))
            ax.barh(
                y, vals, xerr=np.array(errs).T, color=colors, error_kw={"lw": 1, "ecolor": GREY}
            )
            ax.set_yticks(y, ticks, fontsize=8)
            ax.set_ylim(len(ticks) - 0.5, -0.5)
            ax.set_title(title)
        save(fig, self.fig / "fairness.png")
        for attr, info in f["attributes"].items():
            for level, r in info["groups"].items():
                rows.append(
                    [
                        attr.replace("_", " ").title(),
                        level,
                        r["n"],
                        f"{r['default_rate']:.1%}",
                        f"{r['approval_rate']['value']:.1%}",
                        self._rate(r["false_positive_rate"]),
                        self._rate(r["false_negative_rate"]),
                    ]
                )
        parts.append(
            table(["Attribute", "Group", "n", "Default rate", "Approval rate", "FPR", "FNR"], rows)
        )
        parts.append("\n" + self.image("fairness.png", "Error and approval rates by group"))
        self.add(*parts)

    @staticmethod
    def _rate(r: dict) -> str:
        return "n/a" if r["value"] is None else f"{r['value']:.1%}"

    def errors(self):
        e = self.load("errors_test.json")
        rows = []
        for seg, levels in e["segments"].items():
            for level, r in levels.items():
                rows.append(
                    [
                        seg,
                        level,
                        r["n"],
                        f"{r['default_rate']:.1%}",
                        f"{r['mean_predicted']:.1%}",
                        f"{r['flagged_share']:.1%}",
                        "n/a"
                        if r["false_negative_rate"] is None
                        else f"{r['false_negative_rate']:.1%}",
                        "n/a"
                        if r["false_positive_rate"] is None
                        else f"{r['false_positive_rate']:.1%}",
                        f"{r['cost_per_client']:.3f}",
                    ]
                )

        def examples(items):
            cols = [
                "id",
                "probability",
                "default",
                "PAY_1",
                "PAY_2",
                "n_delayed",
                "LIMIT_BAL",
                "AGE",
                "util_mean",
                "payratio_mean",
            ]
            return table(cols, [[i[c] for c in cols] for i in items])

        self.add(
            "## Error analysis (hold-out test set)\n",
            f"- Defaults not flagged: {e['missed_defaults_share']:.1%} of all defaults; "
            f"{e['defaults_below_0.1']:.1%} of defaults received a probability below 0.10.\n"
            f"- Non-defaults with probability at or above {e['confident_threshold']}: {e['non_defaults_above_confident_threshold']:.2%}.\n"
            f"- Mean PAY_1 among missed defaults {e['missed_default_profile']['pay_1_mean_missed']:.2f} versus "
            f"{e['missed_default_profile']['pay_1_mean_caught']:.2f} among caught defaults.\n",
            table(
                [
                    "Segment",
                    "Level",
                    "n",
                    "Default rate",
                    "Mean predicted",
                    "Flagged",
                    "FNR",
                    "FPR",
                    "Cost per client",
                ],
                rows,
            ),
            "\n### Most confident missed defaults (lowest probability)\n",
            examples(e["false_negative_examples"]),
            "\n### Most confident false alarms (highest probability, no default)\n",
            examples(e["false_positive_examples"]),
        )

    def final(self):
        m = self.load("final_metrics.json")
        ci = m["bootstrap_ci95"]
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        r = m["reliability"]
        ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
        ax.plot(r["mean_predicted"], r["observed_rate"], marker="o", color=GREEN)
        ax.set_xlim(0, max(r["mean_predicted"]) * 1.1)
        ax.set_ylim(0, max(r["observed_rate"]) * 1.1)
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed default rate")
        ax.set_title("Reliability on the test set")
        save(fig, self.fig / "test_reliability.png")

        def line(key, digits=4):
            return f"{m['metrics'][key]:.{digits}f} ({ci[key][0]:.{digits}f} to {ci[key][1]:.{digits}f})"

        c = m["confusion"]
        d = m["brier_decomposition"]
        self.add(
            "## Final evaluation on the hold-out test set\n",
            f"Model: {MODEL_LABELS[m['model']]}, calibration: {m['calibration']}, threshold {m['threshold']:.3f}. "
            f"Evaluated once on {m['n_test']:,} clients; intervals are 95% percentile bootstrap ({m['bootstrap_resamples']} resamples).\n",
            table(
                ["Metric", "Value (95% CI)"],
                [
                    ["ROC-AUC", line("roc_auc")],
                    ["PR-AUC", line("pr_auc")],
                    ["Brier score", line("brier", 5)],
                    ["Log loss", line("log_loss", 5)],
                    ["Expected calibration error", f"{m['metrics']['ece']:.5f}"],
                    ["Precision at threshold", line("precision")],
                    ["Recall at threshold", line("recall")],
                    [
                        "Expected cost per client",
                        f"{m['expected_cost_per_client']:.4f} ({ci['expected_cost'][0]:.4f} to {ci['expected_cost'][1]:.4f})",
                    ],
                    [
                        "Cost if nobody is flagged / everybody is flagged",
                        f"{m['cost_flag_nobody']:.4f} / {m['cost_flag_everybody']:.4f}",
                    ],
                    ["Share flagged", f"{m['flagged_share']:.1%}"],
                ],
            ),
            "\n"
            + table(
                ["", "Predicted default", "Predicted no default"],
                [["Actual default", c["tp"], c["fn"]], ["Actual no default", c["fp"], c["tn"]]],
            ),
            f"\nBrier decomposition: reliability {d['reliability']:.5f}, resolution {d['resolution']:.5f}, uncertainty {d['uncertainty']:.5f}.\n",
            self.image("test_reliability.png", "Test-set reliability"),
        )

    def build(self):
        apply_style()
        self.fig.mkdir(parents=True, exist_ok=True)
        self.add(
            "# Credit default risk: evaluation report\n",
            "Generated by `scripts/make_report.py` from the files in `results/`.\n",
        )
        for step in (
            self.data,
            self.comparison,
            self.calibration,
            self.cost,
            self.interpretability,
            self.ablation,
            self.fairness,
            self.errors,
            self.final,
        ):
            step()
        (self.out / "report.md").write_text("\n\n".join(self.parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the evaluation report.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    Report(load_config(args.config)).build()
    print(f"wrote {Path('reports/report.md')}")


if __name__ == "__main__":
    main()
