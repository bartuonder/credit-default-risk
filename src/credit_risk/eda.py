import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_risk import schema
from credit_risk.config import load_config, parse_args, path_for
from credit_risk.plotting import BLUE, ORANGE, apply_style, save
from credit_risk.split import load_train

AGE_BINS = [17, 29, 39, 49, 59, 100]
AGE_LABELS = ["<30", "30-39", "40-49", "50-59", "60+"]


def _rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = df.groupby(column)[schema.TARGET].agg(["mean", "size"])
    return grouped.rename(columns={"mean": "rate", "size": "n"})


def summary(df: pd.DataFrame) -> dict[str, Any]:
    """Numbers quoted in the EDA text."""
    bills = df[schema.BILL_AMT]
    over_limit = bills.gt(df[schema.LIMIT], axis=0).any(axis=1)
    pay1 = _rate_by(df, "PAY_1")
    corr = df.corr(numeric_only=True)[schema.TARGET].drop(schema.TARGET)
    return {
        "n_rows": len(df),
        "default_rate": float(df[schema.TARGET].mean()),
        "missing_values": int(df.isna().sum().sum()),
        "negative_bill_share": float(bills.lt(0).any(axis=1).mean()),
        "bill_over_limit_share": float(over_limit.mean()),
        "zero_payment_share": float(df[schema.PAY_AMT].eq(0).to_numpy().mean()),
        "pay_minus2_share": float(df[schema.PAY_STATUS].eq(-2).to_numpy().mean()),
        "pay_zero_share": float(df[schema.PAY_STATUS].eq(0).to_numpy().mean()),
        "limit_median": float(df[schema.LIMIT].median()),
        "age_range": [int(df[schema.AGE].min()), int(df[schema.AGE].max())],
        "default_rate_by_pay1": {
            int(k): {"rate": float(v.rate), "n": int(v.n)} for k, v in pay1.iterrows()
        },
        "default_rate_by_sex": {int(k): float(v) for k, v in _rate_by(df, "SEX").rate.items()},
        "default_rate_by_education": {
            int(k): float(v) for k, v in _rate_by(df, "EDUCATION").rate.items()
        },
        "default_rate_by_marriage": {
            int(k): float(v) for k, v in _rate_by(df, "MARRIAGE").rate.items()
        },
        "correlation_with_target": {k: float(v) for k, v in corr.sort_values().items()},
        "pay_status_cross_correlation_mean": float(
            df[schema.PAY_STATUS].corr().to_numpy()[np.triu_indices(6, 1)].mean()
        ),
    }


def plot_class_balance(df, out):
    counts = df[schema.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4, 3.2))
    bars = ax.bar(["no default", "default"], counts.to_numpy(), color=[BLUE, ORANGE])
    for bar, c in zip(bars, counts, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, c, f"{c:,}\n{c / len(df):.1%}", ha="center")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_title("Class balance (training set)")
    ax.grid(axis="x", visible=False)
    save(fig, out / "class_balance.png")


def plot_numeric(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, column, title in zip(
        axes,
        [schema.LIMIT, schema.AGE, "BILL_AMT1"],
        ["Credit limit", "Age", "Latest bill"],
        strict=True,
    ):
        values = df[column]
        if column == "BILL_AMT1":
            values = np.sign(values) * np.log10(1 + values.abs())
        bins = np.histogram_bin_edges(values, bins=40)
        for label, color, name in [(0, BLUE, "no default"), (1, ORANGE, "default")]:
            ax.hist(
                values[df[schema.TARGET] == label], bins=bins, alpha=0.6, color=color, label=name
            )
        ax.set_title(title + (" (signed log10)" if column == "BILL_AMT1" else ""))
    axes[0].legend()
    save(fig, out / "numeric_distributions.png")


def plot_categories(df, out):
    work = df.copy()
    work["AGE_BAND"] = pd.cut(work[schema.AGE], AGE_BINS, labels=AGE_LABELS)
    base = work[schema.TARGET].mean()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for ax, column in zip(axes, ["SEX", "EDUCATION", "MARRIAGE", "AGE_BAND"], strict=True):
        stats = _rate_by(work, column)
        labels = [
            schema.CATEGORY_VALUES.get(column, {}).get(k, k) if column != "AGE_BAND" else k
            for k in stats.index
        ]
        ax.bar([str(x) for x in labels], stats["rate"], color=BLUE)
        ax.axhline(base, color=ORANGE, ls="--", lw=1)
        ax.set_title(column.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("default rate")
    save(fig, out / "default_rate_by_category.png")


def plot_repayment(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    stats = _rate_by(df, "PAY_1")
    ax = axes[0]
    ax.bar(stats.index.astype(str), stats["rate"], color=BLUE)
    for x, (_, row) in enumerate(stats.iterrows()):
        ax.text(x, row.rate, f"n={int(row.n):,}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_ylim(0, stats["rate"].max() * 1.2)
    ax.set_xlabel("latest repayment status (PAY_1)")
    ax.set_ylabel("default rate")
    ax.set_title("Default rate by latest repayment status")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    months = list(range(6, 0, -1))
    for label, color, name in [(0, BLUE, "no default"), (1, ORANGE, "default")]:
        sub = df[df[schema.TARGET] == label]
        ax.plot(
            range(6), [sub[f"PAY_{m}"].mean() for m in months], marker="o", color=color, label=name
        )
    ax.set_xticks(range(6), ["Apr", "May", "Jun", "Jul", "Aug", "Sep"])
    ax.set_ylabel("mean repayment status")
    ax.set_title("Repayment status trajectory")
    ax.legend()
    save(fig, out / "repayment_status.png")


def plot_correlation(df, out):
    cols = [
        schema.LIMIT,
        schema.AGE,
        *schema.PAY_STATUS,
        *schema.BILL_AMT,
        *schema.PAY_AMT,
        schema.TARGET,
    ]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols)), cols, fontsize=7)
    ax.grid(visible=False)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Spearman correlation")
    ax.set_title("Correlation matrix")
    save(fig, out / "correlation.png")


def write_markdown(s: dict[str, Any], path) -> None:
    pay1 = s["default_rate_by_pay1"]
    lines = [
        "# Exploratory data analysis (training set)",
        "",
        "Generated by `credit_risk.eda` from the training split only.",
        "",
        f"- Rows: {s['n_rows']:,}; default rate {s['default_rate']:.2%}; missing values: {s['missing_values']}.",
        f"- Median credit limit: {s['limit_median']:,.0f}; age range {s['age_range'][0]} to {s['age_range'][1]}.",
        f"- Default rate for clients paying duly (PAY_1 = -1): {pay1[-1]['rate']:.1%}; "
        f"for a two-month delay (PAY_1 = 2): {pay1[2]['rate']:.1%}.",
        f"- Mean pairwise correlation between the six repayment-status columns: "
        f"{s['pay_status_cross_correlation_mean']:.2f}.",
        "",
        "## Odd values",
        "",
        f"- Clients with at least one negative bill amount: {s['negative_bill_share']:.2%}.",
        f"- Clients with a bill above the credit limit in any month: {s['bill_over_limit_share']:.2%}.",
        f"- Share of zero payments across all client-months: {s['zero_payment_share']:.1%}.",
        f"- Undocumented repayment codes: -2 in {s['pay_minus2_share']:.1%} and 0 in "
        f"{s['pay_zero_share']:.1%} of client-months.",
        "",
        "## Correlation with the target (Pearson)",
        "",
        "| Feature | Correlation |",
        "|---|---|",
    ]
    corr = s["correlation_with_target"]
    top = sorted(corr, key=lambda k: -abs(corr[k]))[:8]
    lines += [f"| {k} | {corr[k]:+.3f} |" for k in top]
    lines += [
        "",
        "## Figures",
        "",
        "![class balance](eda/class_balance.png)",
        "![distributions](eda/numeric_distributions.png)",
        "![categories](eda/default_rate_by_category.png)",
        "![repayment](eda/repayment_status.png)",
        "![correlation](eda/correlation.png)",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args("Run exploratory data analysis on the training split.")
    cfg = load_config(args.config)
    X, y = load_train(cfg)
    df = X.assign(**{schema.TARGET: y})
    reports = path_for(cfg, "reports")
    out = reports / "eda"
    apply_style()
    for fn in (plot_class_balance, plot_numeric, plot_categories, plot_repayment, plot_correlation):
        fn(df, out)
    s = summary(df)
    results = path_for(cfg, "results")
    (results / "eda_summary.json").write_text(json.dumps(s, indent=2) + "\n")
    write_markdown(s, reports / "eda_summary.md")


if __name__ == "__main__":
    main()
