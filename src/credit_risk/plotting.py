import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#2f6db5"
ORANGE = "#d9822b"
GREY = "#6b7280"
GREEN = "#3a8f5c"
PURPLE = "#7a5ab0"
PALETTE = [BLUE, ORANGE, GREEN, PURPLE, GREY, "#b5483c"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
