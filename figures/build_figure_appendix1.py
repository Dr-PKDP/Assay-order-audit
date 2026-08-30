"""
build_figure_appendix1.py

Appendix Figure A1: the actual seed-level distributions summarized as mean/sd
in Table 7 and as "mean p=0.571, range 0.020-0.954" in Section 5.5 -- shown
directly rather than only as summary statistics.
  A. Distribution of k=1 honest-estimate accuracy across 200 partner-draw
     seeds, margin<0.6 subset (n=58).
  B. Distribution of paired DeLong p-values (baseline vs k=1) across 30
     seeds, same subset.
"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fig_style import *

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"

accs_subset = np.load(ROOT / "appendix_seed_accs_subset.npy")
delong_ps = np.load(ROOT / "appendix_delong_ps.npy")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

ax_a = axes[0]
panel_label(ax_a, "A")
ax_a.hist(accs_subset * 100, bins=20, color=COLOR_HONEST, edgecolor="white", alpha=0.9)
ax_a.axvline(accs_subset.mean() * 100, color=COLOR_REPORTED, ls="--", lw=1.5, label="k=1 seed mean")
ax_a.axvline(68.97, color=COLOR_INTERNAL, ls=":", lw=1.5, label="mRNA-only baseline")
ax_a.set_xlabel("k=1 accuracy, margin<0.6 subset (%)")
ax_a.set_ylabel("Count (of 200 seeds)")
ax_a.legend(fontsize=8.8, loc="upper left")

ax_b = axes[1]
panel_label(ax_b, "B")
ax_b.hist(delong_ps, bins=15, color=COLOR_BOUND, edgecolor="white", alpha=0.9)
ax_b.axvline(0.05, color=COLOR_EXTERNAL, ls="--", lw=1.5, label="α = 0.05")
ax_b.set_xlabel("Paired DeLong p-value (baseline vs k=1)")
ax_b.set_ylabel("Count (of 30 seeds)")
ax_b.legend(fontsize=8.8, loc="upper left")

fig.tight_layout()
fig.savefig(FIG / "figureA1_seed_distributions.png", bbox_inches="tight")
fig.savefig(FIG / "figureA1_seed_distributions.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved figureA1_seed_distributions.png/.pdf")
