"""
build_figure4.py

Figure 4: Ablation study -- feature budget and ensemble size.
  A. Stage-1 accuracy vs top-K selected features.
  B. Stage-1 accuracy vs bootstrap ensemble size.
  C. Stage-2 accuracy vs top-K selected features.
  D. Stage-2 accuracy vs bootstrap ensemble size.

Replaces: fig9_ablation_study. Reads directly from
results/ablation_study_summary.json, unchanged by any of this session's
corrections (Section 5.7's own text already frames these as ordering-across-
configurations results from a single 5-fold pass, not point estimates).
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import *

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"

ablation = json.load(open(RES / "ablation_study_summary.json"))

fig, axes = plt.subplots(2, 2, figsize=(8, 6.6))

panels = [
    (axes[0, 0], "A", ablation["stage1"]["top_k"], "k", "Top-K features (Stage 1)", COLOR_STAGE1),
    (axes[0, 1], "B", ablation["stage1"]["ensemble_size"], "n_boot", "Bootstrap ensemble size (Stage 1)", COLOR_STAGE1),
    (axes[1, 0], "C", ablation["stage2"]["top_k"], "k", "Top-K features (Stage 2)", COLOR_STAGE2),
    (axes[1, 1], "D", ablation["stage2"]["ensemble_size"], "n_boot", "Bootstrap ensemble size (Stage 2)", COLOR_STAGE2),
]

for ax, letter, data, xkey, xlabel, color in panels:
    panel_label(ax, letter)
    xs = [d[xkey] for d in data]
    ys = [d["accuracy"] * 100 for d in data]
    ax.plot(xs, ys, marker="o", color=color, lw=1.8, markersize=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig(FIG / "figure4_ablation.png", bbox_inches="tight")
fig.savefig(FIG / "figure4_ablation.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved figure4_ablation.png/.pdf")
