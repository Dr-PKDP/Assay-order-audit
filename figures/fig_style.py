"""
fig_style.py

Shared style module for the rebuilt figure set. Journal convention for
multi-panel medical/life-science figures: minimal in-figure text (no long
titles restating what the caption says), panel letters (A, B, C...) in the
top-left of each panel, consistent color coding across the whole figure set,
and axis labels that state units/quantities only -- interpretation goes in
the caption or the main text, not baked into the plot.

Color convention used throughout, consistent across all 5 figures:
    Stage-1 (mRNA)        -> #2b7a78 (teal)
    Stage-2 (methylation) -> #d1495b (red)
    Internal / training   -> #1b6ca8 (blue)
    External / test       -> #d9534f (red-orange)
    Reported / inflated    -> #999999 (grey) -- signals "not to be trusted at
                              face value" without needing a text warning
    Bound / theoretical    -> #1b6ca8 (blue)
    Honest / re-simulated  -> #2b7a78 (teal)
    Chance line            -> grey dotted, no legend entry needed if labelled
                              once in the caption
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

COLOR_STAGE1 = "#2b7a78"
COLOR_STAGE2 = "#d1495b"
COLOR_INTERNAL = "#1b6ca8"
COLOR_EXTERNAL = "#d9534f"
COLOR_REPORTED = "#999999"
COLOR_BOUND = "#4a5a9e"
COLOR_HONEST = "#2b7a78"
COLOR_CHANCE = "#aaaaaa"


def panel_label(ax, letter, x=-0.14, y=1.08, fontsize=None):
    """Bold panel letter (A, B, C...) at the top-left corner, journal style.
    Default size is set relative to the current axis label font size (not a fixed
    absolute value), so panel letters stay proportionate to a panel's own text
    regardless of how that panel is embedded or sized."""
    if fontsize is None:
        fontsize = plt.rcParams["axes.labelsize"] * 0.9
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fontsize, fontweight="bold",
           va="top", ha="left", color="black")


def no_title(ax):
    """Panels get axis labels only; the descriptive title lives in the caption."""
    ax.set_title("")
