"""
build_figure_mechanism.py

The cleanest single visual of the paper's mechanism: the standalone Stage-2
assay AUC is a constant 0.646, while the AUC of the synthetic averaged partner
feature built from that same assay climbs with the partner count k, tracking
the closed-form prediction. Nothing about the assay changed; only the number of
same-class partners averaged into the surrogate.

Data sources: results/k_sweep.csv (predicted effective AUC) and
results/variance_vs_k_empirical.csv (observed partner-feature AUC).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({"font.size": 11.5, "figure.dpi": 300, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})

STANDALONE = "#d1495b"   # Stage-2 methylation
THEORY     = "#1b6ca8"   # model-derived
OBSERVED   = "#2b7a78"   # empirical

emp = pd.read_csv("results/variance_vs_k_empirical.csv")
k = emp["k"].to_numpy()
auc_obs = emp["auc_obs"].to_numpy()
auc_thy = emp["auc_theory"].to_numpy()
standalone = 0.646

fig, ax = plt.subplots(figsize=(6.6, 3.3))
ax.axhline(standalone, color=STANDALONE, ls="--", lw=1.8,
           label="Standalone Stage-2 assay AUC (0.646, constant)")
ax.plot(k, auc_thy, color=THEORY, lw=1.8, marker="o", ms=4.5,
        label=r"Predicted: $\Phi(\sqrt{k}\,d/\sqrt{2})$")
ax.plot(k, auc_obs, color=OBSERVED, lw=0, marker="s", ms=6, mfc="none", mew=1.6,
        label="Observed synthetic partner AUC")

ax.set_xscale("log")
ax.set_xticks(k)
ax.set_xticklabels([str(v) for v in k])
ax.minorticks_off()
ax.set_xlabel("Partners averaged, $k$")
ax.set_ylabel("AUC")
ax.set_ylim(0.55, 1.02)
ax.legend(frameon=False, fontsize=9.5, loc="lower right")
ax.grid(axis="y", alpha=0.25, lw=0.6)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"results/figures/figure_mechanism_partner_auc.{ext}", bbox_inches="tight")
print("saved results/figures/figure_mechanism_partner_auc.png/.pdf")
