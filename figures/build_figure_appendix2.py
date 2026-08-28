"""
build_figure_appendix2.py

Appendix Figure A2: ROC curves and confusion matrices for all three
harmonization conditions (baseline self-standardization, FSQN, ComBat) on
both external cohorts -- the visual evidence behind Section 6.1's and
Figure 2's summary accuracy/AUC/DeLong numbers, including the striking
ComBat-on-GSE5281 result (AUC 0.39, significantly below baseline).

  A. ROC curves, GSE118553 (n=63), all three conditions.
  B. ROC curves, GSE5281 (n=34), all three conditions.
  C. Confusion matrices, all 6 cohort x condition combinations.

Every AUC/accuracy value here was cross-checked against the summary
statistics already reported in Section 6.1 before this figure was built --
see the conversation record for the exact verification (30/30 metrics
matched exactly across both files).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fig_style import *

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results"
FIG = ROOT / "results" / "figures"

g118 = pd.read_csv(DATA / "GSE118553_all_predictions.csv")
g5281 = pd.read_csv(DATA / "GSE5281_all_predictions.csv")

CONDITIONS = [("baseline_prob", "Baseline (self-std)", COLOR_REPORTED),
             ("fsqn_prob", "FSQN", COLOR_HONEST),
             ("combat_prob", "ComBat", COLOR_BOUND)]

fig = plt.figure(figsize=(11, 7.1))
gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.08, wspace=0.28)

# ---------------------------------------------------------------- Panel A: ROC GSE118553
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, "A")
for col, name, color in CONDITIONS:
    fpr, tpr, _ = roc_curve(g118["label"], g118[col])
    auc = roc_auc_score(g118["label"], g118[col])
    ax_a.plot(fpr, tpr, color=color, lw=1.8, label=f"{name} (AUC={auc:.3f})")
ax_a.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
ax_a.set_xlabel("False positive rate")
ax_a.set_ylabel("True positive rate")
ax_a.set_title("GSE118553 (n=63)", fontsize=11.2, fontweight="normal")
ax_a.legend(fontsize=8.8, loc="lower right")

# ---------------------------------------------------------------- Panel B: ROC GSE5281
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, "B")
for col, name, color in CONDITIONS:
    fpr, tpr, _ = roc_curve(g5281["label"], g5281[col])
    auc = roc_auc_score(g5281["label"], g5281[col])
    ax_b.plot(fpr, tpr, color=color, lw=1.8, label=f"{name} (AUC={auc:.3f})")
ax_b.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
ax_b.set_xlabel("False positive rate")
ax_b.set_ylabel("True positive rate")
ax_b.set_title("GSE5281 (n=34)", fontsize=11.2, fontweight="normal")
ax_b.legend(fontsize=8.8, loc="lower right")

# ---------------------------------------------------------------- Panel C: confusion matrix strip
cms = []
for df, cohort_name, n in [(g118, "GSE118553", 63), (g5281, "GSE5281", 34)]:
    for col, name, _ in CONDITIONS:
        pred = (df[col] > 0.5).astype(int)
        cm = confusion_matrix(df["label"], pred)
        cms.append((f"{cohort_name}\n{name}\n(n={n})", cm))

inset_gs = gs[1, :].subgridspec(1, 6, wspace=0.25)
for i, (title, cm) in enumerate(cms):
    axi = fig.add_subplot(inset_gs[0, i])
    axi.imshow(cm, cmap="Blues", vmin=0)
    for r in range(2):
        for c in range(2):
            axi.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=9.4,
                     color="white" if cm[r, c] > cm.max() / 2 else "black")
    axi.set_xticks([0, 1]); axi.set_xticklabels(["C", "AD"], fontsize=8.3)
    axi.set_xlabel(title, fontsize=8)
    if i == 0:
        axi.set_yticks([0, 1]); axi.set_yticklabels(["C", "AD"], fontsize=8.3)
        axi.set_ylabel("True", fontsize=8.8)
    else:
        axi.set_yticks([])

fig.text(0.055, 0.40, "C", fontsize=plt.rcParams["axes.labelsize"] * 0.9, fontweight="bold",
         va="top", ha="left", color="black")

fig.savefig(FIG / "figureA2_roc_confusion.png", bbox_inches="tight")
fig.savefig(FIG / "figureA2_roc_confusion.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved figureA2_roc_confusion.png/.pdf")

print("\nData check -- AUC from this script vs Section 6.1's reported values:")
for df, name in [(g118, "GSE118553"), (g5281, "GSE5281")]:
    for col, cname, _ in CONDITIONS:
        auc = roc_auc_score(df["label"], df[col])
        print(f"  {name:<10} {cname:<20} AUC={auc:.4f}")
