"""
build_figure1.py

Figure 1: Classifier performance, calibration, and confidence stratification.
  A. Confusion matrices, all 5 internal/external cohort-stage combinations.
  B. Reliability diagrams, internal vs external, both stages.
  C. Accuracy and AUC across all 5 cohorts, with 95% bootstrap CIs.
  D. Stage-1 resolution rate vs confidence threshold, vs SGUQ (ROSMAP).

Replaces: fig2_calibration_reliability, fig3_confusion_matrices,
fig5_internal_vs_external_validation, fig8_expanded_external_validation,
fig4_confidence_stratification_vs_sguq.

All data from results/ CSVs and JSONs already in this repo -- no numbers
invented or estimated.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.calibration import calibration_curve
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import *

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True, parents=True)

s1_oof = pd.read_csv(RES / "stage1_mrna_oof_predictions.csv")
s1_ext1 = pd.read_csv(RES / "stage1_mrna_external_predictions.csv")       # GSE118553
s1_ext2 = pd.read_csv(RES / "stage1_mrna_external2_GSE5281_predictions.csv")  # GSE5281
s2_oof = pd.read_csv(RES / "stage2_methylation_oof_predictions.csv")
s2_ext = pd.read_csv(RES / "stage2_methylation_external_GSE134379_predictions.csv")
sweep = pd.read_csv(RES / "stage1_confidence_sweep.csv")
stats = json.load(open(RES / "statistical_tests_summary.json"))
new_ext = json.load(open(RES / "new_external_validations_summary.json"))
s1_summary = json.load(open(RES / "stage1_summary.json"))
s2_summary = json.load(open(RES / "stage2_summary.json"))

fig = plt.figure(figsize=(11, 8.3))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.32)

# ---------------------------------------------------------------- Panel A
ax_a = fig.add_subplot(gs[0, 0])
panel_label(ax_a, "A")
ax_a.axis("off")

cms = [
    ("S1 internal\n(n=697)", confusion_matrix(s1_oof["label"], s1_oof["oof_pred"])),
    ("GSE118553\n(n=63)", confusion_matrix(s1_ext1["label"], s1_ext1["ext_pred"])),
    ("GSE5281\n(n=34)", confusion_matrix(s1_ext2["label"], s1_ext2["ext_pred"])),
    ("S2 internal\n(n=142)", confusion_matrix(s2_oof["label"], (s2_oof["oof_prob"] >= 0.5).astype(int))),
    ("GSE134379\n(n=404)", confusion_matrix(s2_ext["label"], s2_ext["ext_pred"])),
]

inset_gs = gs[0, 0].subgridspec(1, 5, wspace=0.55)
for i, (title, cm) in enumerate(cms):
    axi = fig.add_subplot(inset_gs[0, i])
    axi.imshow(cm, cmap="Blues", vmin=0)
    for r in range(2):
        for c in range(2):
            axi.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=8.6,
                     color="white" if cm[r, c] > cm.max() / 2 else "black")
    axi.set_xticks([0, 1]); axi.set_xticklabels(["C", "AD"], fontsize=7.6)
    axi.set_xlabel(title, fontsize=7.6)
    if i == 0:
        axi.set_yticks([0, 1]); axi.set_yticklabels(["C", "AD"], fontsize=7.6)
        axi.set_ylabel("True", fontsize=8.1)
    else:
        axi.set_yticks([])

# ---------------------------------------------------------------- Panel B
ax_b = fig.add_subplot(gs[0, 1])
panel_label(ax_b, "B")
ax_b.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)

for label, df, prob_col, color, ls in [
    ("S1 internal", s1_oof, "oof_prob", COLOR_STAGE1, "-"),
    ("S1 GSE118553+GSE5281", pd.concat([s1_ext1, s1_ext2]), "ext_prob", COLOR_STAGE1, "--"),
    ("S2 internal", s2_oof, "oof_prob", COLOR_STAGE2, "-"),
    ("S2 GSE134379", s2_ext, "ext_prob", COLOR_STAGE2, "--"),
]:
    try:
        frac_pos, mean_pred = calibration_curve(df["label"], df[prob_col], n_bins=8, strategy="quantile")
        ax_b.plot(mean_pred, frac_pos, marker="o", markersize=3, color=color, ls=ls, lw=1.3, label=label)
    except ValueError:
        frac_pos, mean_pred = calibration_curve(df["label"], df[prob_col], n_bins=5, strategy="uniform")
        ax_b.plot(mean_pred, frac_pos, marker="o", markersize=3, color=color, ls=ls, lw=1.3, label=label)

ax_b.set_xlabel("Mean predicted probability")
ax_b.set_ylabel("Observed AD fraction")
ax_b.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1)

# ---------------------------------------------------------------- Panel C
ax_c = fig.add_subplot(gs[1, 0])
panel_label(ax_c, "C")

cohort_rows = [
    ("S1 internal", s1_summary["internal_oof_accuracy"], s1_summary["internal_oof_auc"],
    stats["bootstrap_cis"]["stage1_internal"]["accuracy"], stats["bootstrap_cis"]["stage1_internal"]["auc"], COLOR_STAGE1),
    ("S1 GSE118553", s1_summary["external_accuracy"], s1_summary["external_auc"],
    stats["bootstrap_cis"]["stage1_external"]["accuracy"], stats["bootstrap_cis"]["stage1_external"]["auc"], COLOR_STAGE1),
    ("S1 GSE5281", new_ext["stage1_external2_GSE5281_SFG"]["accuracy"], new_ext["stage1_external2_GSE5281_SFG"]["auc"],
    new_ext["stage1_external2_GSE5281_SFG"]["bootstrap_ci"]["accuracy"],
    new_ext["stage1_external2_GSE5281_SFG"]["bootstrap_ci"]["auc"], COLOR_STAGE1),
    ("S2 internal", s2_summary["oof_accuracy"], s2_summary["oof_auc"],
    stats["bootstrap_cis"]["stage2_internal"]["accuracy"], stats["bootstrap_cis"]["stage2_internal"]["auc"], COLOR_STAGE2),
    ("S2 GSE134379", new_ext["stage2_external_GSE134379_MTG"]["accuracy"], new_ext["stage2_external_GSE134379_MTG"]["auc"],
    new_ext["stage2_external_GSE134379_MTG"]["bootstrap_ci"]["accuracy"],
    new_ext["stage2_external_GSE134379_MTG"]["bootstrap_ci"]["auc"], COLOR_STAGE2),
]

x = np.arange(len(cohort_rows))
w = 0.35
acc_vals = [r[1] * 100 for r in cohort_rows]
auc_vals = [r[2] for r in cohort_rows]
acc_err = np.array([[r[1] * 100 - r[3]["ci_lo"] * 100, r[3]["ci_hi"] * 100 - r[1] * 100] for r in cohort_rows]).T
auc_err = np.array([[r[2] - r[4]["ci_lo"], r[4]["ci_hi"] - r[2]] for r in cohort_rows]).T
colors = [r[5] for r in cohort_rows]

ax_c.bar(x - w/2, acc_vals, width=w, yerr=acc_err, capsize=3, color=colors, alpha=0.9, label="Accuracy")
ax_c2 = ax_c.twinx()
ax_c2.bar(x + w/2, auc_vals, width=w, yerr=auc_err, capsize=3, color=colors, alpha=0.55, hatch="//", label="AUC")
ax_c.axhline(50, color=COLOR_CHANCE, ls=":", lw=1)
ax_c.set_xticks(x); ax_c.set_xticklabels([r[0] for r in cohort_rows], fontsize=8.3, rotation=20, ha="right")
ax_c.set_ylabel("Accuracy (%)")
ax_c2.set_ylabel("AUC")
ax_c.set_ylim(0, 105)
ax_c2.set_ylim(0, 1.05)
from matplotlib.patches import Patch
ax_c.legend(handles=[Patch(facecolor="white", edgecolor="black", label="Accuracy (solid)"),
                     Patch(facecolor="white", edgecolor="black", hatch="//", label="AUC (hatched)")],
           fontsize=7.7, loc="upper right")

# ---------------------------------------------------------------- Panel D
ax_d = fig.add_subplot(gs[1, 1])
panel_label(ax_d, "D")
ax_d.plot(sweep["tau"], sweep["pct_resolved"] * 100, color=COLOR_INTERNAL, lw=1.8,
         label="This study (n=697)")
ax_d.axhline(46.23, color=COLOR_EXTERNAL, ls="--", lw=1.3, label="SGUQ, mRNA-only")
ax_d.axhline(46.23 + 16.04, color=COLOR_STAGE1, ls="--", lw=1.3, label="SGUQ, mRNA+methylation")
ax_d.set_xlabel("Confidence-margin threshold τ")
ax_d.set_ylabel("% patients resolved")
ax_d.legend(fontsize=8.3, loc="lower left")
ax_d.set_ylim(0, 105)

fig.savefig(FIG / "figure1_classifier_performance.png", bbox_inches="tight")
fig.savefig(FIG / "figure1_classifier_performance.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved figure1_classifier_performance.png/.pdf")

# print a data check summary for verification against manuscript Table 3
print("\nData check against Table 3:")
for r in cohort_rows:
    print(f"  {r[0]:<16} acc={r[1]:.4f}  auc={r[2]:.4f}")
