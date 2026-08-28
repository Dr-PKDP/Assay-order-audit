"""
build_figure3.py

Figure 3: The standard fusion estimate is unattainable -- bound and honest
re-simulation.
  A. Accuracy by confidence-margin subset: mRNA-only, reported k=20 ceiling,
     attainability bound (95% CI), honest k=1 re-simulation (+/- 1 SD).
  B. Cost-accuracy frontier: reported (inflated) vs bound-based honest.
  C. Fusion meta-model design ablation (same-label vs random pairing).

Replaces: fig1_cost_accuracy_pareto, fig1b_honest_frontier,
fig6_low_confidence_fusion_benefit, fig10_fusion_metamodel_ablation, and the
fusion-related rows of fig7_forest_plot_cis (folded into Panel A's error
bars rather than a separate forest plot, to avoid a redundant panel showing
the same comparison twice).

Panel A/B bound and k=1 values come from bound_ci_and_honest_band.py,
power_and_equivalence.py and score_fsqn_vs_baseline.py output already
verified against Tables 6-8 in the manuscript. Panel C reads directly from
results/ablation_study_summary.json (unchanged raw data -- Section 5.3's own
text already frames these correctly as "what the convention produces").
"""
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import *

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"

ablation = json.load(open(RES / "ablation_study_summary.json"))

# ---- Panel A/B data: subset, n, pre-assay acc, reported k=20, bound point,
#      bound CI lo/hi, k=1 mean, k=1 sd -- all cross-checked against Tables 6-8
SUBSETS = [
    ("Margin<0.2", 24, 66.67, 95.83, 73.62, 56.37, 88.36, 64.71, 9.12),
    ("Margin<0.4", 41, 70.73, 95.12, 76.84, 64.72, 87.82, 69.59, 5.93),
    ("Margin<0.6", 58, 68.97, 93.10, 75.44, 64.48, 85.40, 68.25, 4.31),
    ("Whole cohort", 697, 96.56, 98.57, 97.28, 95.94, 98.33, 96.49, 0.36),
]

fig = plt.figure(figsize=(14, 3.6))
gs = fig.add_gridspec(1, 3, wspace=0.38)

# ---------------------------------------------------------------- Panel A
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, "A")
x = np.arange(len(SUBSETS))
w = 0.2

pre_vals = [s[2] for s in SUBSETS]
reported_vals = [s[3] for s in SUBSETS]
bound_vals = [s[4] for s in SUBSETS]
bound_err = np.array([[s[4] - s[5] for s in SUBSETS], [s[6] - s[4] for s in SUBSETS]])
k1_vals = [s[7] for s in SUBSETS]
k1_err = [s[8] for s in SUBSETS]

ax_a.bar(x - 1.5*w, pre_vals, width=w, color=COLOR_INTERNAL, alpha=0.9, label="mRNA-only")
ax_a.bar(x - 0.5*w, reported_vals, width=w, color=COLOR_REPORTED, alpha=0.9, label="Reported (k=20)")
ax_a.bar(x + 0.5*w, bound_vals, width=w, yerr=bound_err, capsize=3, color=COLOR_BOUND, alpha=0.9, label="Bound (95% CI)")
ax_a.bar(x + 1.5*w, k1_vals, width=w, yerr=k1_err, capsize=3, color=COLOR_HONEST, alpha=0.9, label="Honest k=1 (\u00b1 SD)")

ax_a.set_xticks(x)
ax_a.set_xticklabels([s[0] for s in SUBSETS], fontsize=8.8, rotation=15, ha="right")
ax_a.set_ylabel("Accuracy (%)")
ax_a.set_ylim(0, 108)
ax_a.legend(fontsize=7.7, loc="lower right")

# ---------------------------------------------------------------- Panel B
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, "B")

# reported frontier: taus 0.2/0.5/0.8/full from Table 7-equivalent already in text
reported_tau = [0, 0.02, 0.2, 0.5, 0.8, 1.0]
reported_cost = [1.0, 1.02, 1.05, 1.121, 1.20, 2.45]
reported_acc = [96.56, 96.56, 97.56, 98.42, 98.57, 98.57]

bound_cost = [1.0, 1.050, 1.085, 1.121, 1.200, 1.312, 2.450]
bound_acc = [96.56, 96.80, 96.92, 97.10, 97.16, 97.22, 97.28]

ax_b.plot(reported_cost, reported_acc, color=COLOR_REPORTED, lw=1.8, marker="o", markersize=3,
         label="Reported (k=20 simulation)")
ax_b.plot(bound_cost, bound_acc, color=COLOR_BOUND, lw=1.8, marker="o", markersize=3,
         label="Attainability bound")
ax_b.scatter([1.0], [96.56], color=COLOR_INTERNAL, zorder=5, s=55, marker="s", label="mRNA-only")
ax_b.set_xlabel("Mean assay cost per patient (mRNA=1.0)")
ax_b.set_ylabel("Overall accuracy (%)")
ax_b.legend(fontsize=8, loc="lower right")
ax_b.set_ylim(96, 99.2)

# ---------------------------------------------------------------- Panel C
ax_c = fig.add_subplot(gs[0, 2])
panel_label(ax_c, "C")
fmd = ablation["fusion_meta_model_design"]
names = ["mRNA-only\n(no fusion)", "Simple avg\n(ceiling)", "Learned\n(ceiling)",
        "Simple avg\n(floor)", "Learned\n(floor)"]
accs = [d["accuracy"] * 100 for d in fmd]
colors_c = [COLOR_INTERNAL if "no fusion" in d["name"] else
           (COLOR_REPORTED if "CEILING" in d["name"] else COLOR_HONEST) for d in fmd]
ax_c.bar(np.arange(len(fmd)), accs, color=colors_c)
ax_c.set_xticks(np.arange(len(fmd)))
ax_c.set_xticklabels(names, fontsize=7.4, rotation=20, ha="right")
ax_c.set_ylabel("Accuracy (%)")
ax_c.set_ylim(94, 100)

fig.savefig(FIG / "figure3_fusion_bound.png", bbox_inches="tight")
fig.savefig(FIG / "figure3_fusion_bound.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved figure3_fusion_bound.png/.pdf")
