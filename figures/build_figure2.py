"""
build_figure2.py

Figure 2: Robustness checks -- cross-platform harmonization and cross-series
generalization.
  A. FSQN vs self-standardization baseline, both external cohorts.
  B. Leave-one-series-out retrain vs pooled-training baseline, both directions.

DATA SOURCE NOTE: unlike Figure 1, these summary numbers were computed on the
user's own machine (score_fsqn_vs_baseline.py, leave_one_series_out.py) and
relayed via chat, not read from a results/ file in this repo -- the
underlying per-sample prediction arrays were never uploaded back, only the
printed accuracy/AUC/Brier/balanced-accuracy summaries. Those summary values
are hardcoded below, each traceable to the exact terminal output pasted in
conversation, and match the numbers already in the manuscript text
(Section 6.1, Limitations). If the underlying prediction CSVs are ever
uploaded, this script should be rewritten to read them directly instead.
"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import *

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
FIG.mkdir(exist_ok=True, parents=True)

# ---- Panel A data: FSQN vs self-standardization baseline ----------------
fsqn_data = {
    "GSE118553": {
        "baseline": dict(accuracy=0.5556, auc=0.5130, brier=0.3013),
        "fsqn": dict(accuracy=0.5714, auc=0.5065, brier=0.3084),
    },
    "GSE5281": {
        "baseline": dict(accuracy=0.6471, auc=0.7036, brier=0.2243),
        "fsqn": dict(accuracy=0.7059, auc=0.5731, brier=0.2550),
    },
}

# ---- Panel B data: leave-one-series-out vs pooled training --------------
loso_data = [
    ("train GSE44770\ntest GSE33000", dict(accuracy=0.9550, auc=0.9841, brier=0.0409)),
    ("train GSE33000\ntest GSE44770", dict(accuracy=0.9783, auc=0.9998, brier=0.0147)),
]
pooled_baseline = dict(accuracy=0.9656, auc=0.9930, brier=0.0271)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# ---------------------------------------------------------------- Panel A
ax_a = axes[0]
panel_label(ax_a, "A")
cohorts = list(fsqn_data.keys())
ax_a2 = ax_a.twinx()

positions = np.arange(len(cohorts)) * 1.4
bw = 0.28
for i, c in enumerate(cohorts):
    p0 = positions[i]
    ax_a.bar(p0 - 1.5*bw, fsqn_data[c]["baseline"]["accuracy"] * 100, width=bw, color=COLOR_REPORTED, alpha=0.9)
    ax_a.bar(p0 - 0.5*bw, fsqn_data[c]["fsqn"]["accuracy"] * 100, width=bw, color=COLOR_HONEST, alpha=0.9)
    ax_a2.bar(p0 + 0.5*bw, fsqn_data[c]["baseline"]["auc"], width=bw, color=COLOR_REPORTED, alpha=0.55, hatch="//")
    ax_a2.bar(p0 + 1.5*bw, fsqn_data[c]["fsqn"]["auc"], width=bw, color=COLOR_HONEST, alpha=0.55, hatch="//")

ax_a.axhline(50, color=COLOR_CHANCE, ls=":", lw=1)
ax_a.set_xticks(positions)
ax_a.set_xticklabels(cohorts, fontsize=10)
ax_a.set_ylabel("Accuracy (%)")
ax_a2.set_ylabel("AUC")
ax_a.set_ylim(0, 105)
ax_a2.set_ylim(0, 1.05)
from matplotlib.patches import Patch
ax_a.legend(handles=[
    Patch(facecolor=COLOR_REPORTED, label="Self-standardization"),
    Patch(facecolor=COLOR_HONEST, label="FSQN"),
    Patch(facecolor="white", edgecolor="black", label="Accuracy (solid)"),
    Patch(facecolor="white", edgecolor="black", hatch="//", label="AUC (hatched)"),
], fontsize=7.4, loc="upper center", ncol=2)

# ---------------------------------------------------------------- Panel B
ax_b = axes[1]
panel_label(ax_b, "B")
directions = [d[0] for d in loso_data]
x2 = np.arange(len(directions) + 1)  # +1 for pooled baseline reference
acc_vals = [pooled_baseline["accuracy"]] + [d[1]["accuracy"] for d in loso_data]
auc_vals = [pooled_baseline["auc"]] + [d[1]["auc"] for d in loso_data]
labels2 = ["Pooled\n(both series\nin training)"] + directions
colors2 = [COLOR_REPORTED, COLOR_HONEST, COLOR_HONEST]

w2 = 0.35
ax_b.bar(x2 - w2/2, [a * 100 for a in acc_vals], width=w2, color=colors2, alpha=0.9)
ax_b2 = ax_b.twinx()
ax_b2.bar(x2 + w2/2, auc_vals, width=w2, color=colors2, alpha=0.55, hatch="//")
ax_b.set_xticks(x2)
ax_b.set_xticklabels(labels2, fontsize=8.8)
ax_b.set_ylabel("Accuracy (%)")
ax_b2.set_ylabel("AUC")
ax_b.set_ylim(0, 105)
ax_b2.set_ylim(0, 1.05)
from matplotlib.patches import Patch
ax_b.legend(handles=[Patch(facecolor="white", edgecolor="black", label="Accuracy (solid)"),
                     Patch(facecolor="white", edgecolor="black", hatch="//", label="AUC (hatched)")],
           fontsize=7.7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)

fig.tight_layout()
fig.savefig(FIG / "figure2_robustness_checks.png", bbox_inches="tight")
fig.savefig(FIG / "figure2_robustness_checks.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved figure2_robustness_checks.png/.pdf")
