"""Confidence intervals and significance tests for the Next-Best-Assay recommender.

Adds, on top of the point estimates already computed:
  1. Bootstrap (patient-resampling) 95% CIs for every headline metric
     (accuracy, balanced accuracy, AUC, F1, Brier) for Stage-1 internal,
     Stage-1 external, Stage-2 internal, and the fusion ceiling/floor, both
     whole-cohort and on the low-confidence (tau<0.6) subset.
  2. DeLong's test for paired ROC-AUC comparisons (same 697 patients get
     mRNA-only vs. fusion-ceiling vs. fusion-floor predictions, so the
     standard paired DeLong covariance applies).
  3. McNemar's exact test for paired accuracy/discordant-prediction
     comparisons on the same patient sets.
  4. A two-proportion z-test for the (unpaired) internal-vs-external
     accuracy drop, since those are two different patient cohorts.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root
from scipy import stats
from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score, brier_score_loss

np.random.seed(2026)
RES = ROOT / "results"
N_BOOT = 2000

s1_oof = pd.read_csv(RES / "stage1_mrna_oof_predictions.csv")
s1_ext = pd.read_csv(RES / "stage1_mrna_external_predictions.csv")
s2_oof = pd.read_csv(RES / "stage2_methylation_oof_predictions.csv")
fused = pd.read_csv(RES / "fusion_predictions_all.csv")

# ============================================================
# 1. Bootstrap 95% CIs (patient-level resampling, percentile method)
# ============================================================
def bootstrap_ci(y, prob, pred=None, n_boot=N_BOOT, seed=0):
    """Returns dict of point estimate + (lo, hi) 95% CI for acc, bal_acc, auc, f1, brier."""
    rng = np.random.RandomState(seed)
    n = len(y)
    if pred is None:
        pred = (prob >= 0.5).astype(int)
    metrics = {"accuracy": [], "balanced_accuracy": [], "auc": [], "f1": [], "brier": []}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb, pb, prb = y[idx], pred[idx], prob[idx]
        if len(np.unique(yb)) < 2:
            continue  # skip degenerate resamples for AUC
        metrics["accuracy"].append(accuracy_score(yb, pb))
        metrics["balanced_accuracy"].append(balanced_accuracy_score(yb, pb))
        metrics["auc"].append(roc_auc_score(yb, prb))
        metrics["f1"].append(f1_score(yb, pb, zero_division=0))
        metrics["brier"].append(brier_score_loss(yb, prb))
    out = {}
    for k, v in metrics.items():
        v = np.array(v)
        point_fn = {"accuracy": accuracy_score, "balanced_accuracy": balanced_accuracy_score,
                    "auc": roc_auc_score, "f1": f1_score, "brier": brier_score_loss}[k]
        if k in ("auc", "brier"):
            point = point_fn(y, prob)
        elif k == "f1":
            point = point_fn(y, pred, zero_division=0)
        else:
            point = point_fn(y, pred)
        out[k] = {"point": float(point), "ci_lo": float(np.percentile(v, 2.5)), "ci_hi": float(np.percentile(v, 97.5)),
                   "n_boot_valid": int(len(v))}
    return out


results = {}

print(f"=== Bootstrapping 95 percent CIs (patient resampling, n_boot={N_BOOT}) ===")

results["stage1_internal"] = bootstrap_ci(s1_oof["label"].values, s1_oof["oof_prob"].values,
                                           s1_oof["oof_pred"].values, seed=1)
print("Stage-1 internal (n=697):", json.dumps(results["stage1_internal"], indent=2))

results["stage1_external"] = bootstrap_ci(s1_ext["label"].values, s1_ext["ext_prob"].values,
                                           s1_ext["ext_pred"].values, seed=2)
print("Stage-1 external (n=63):", json.dumps(results["stage1_external"], indent=2))

results["stage2_internal"] = bootstrap_ci(s2_oof["label"].values,
                                           s2_oof["oof_prob"].values,
                                           (s2_oof["oof_prob"].values >= 0.5).astype(int), seed=3)
print("Stage-2 internal (n=142):", json.dumps(results["stage2_internal"], indent=2))

y_all = fused["label"].values
mrna_prob = fused["mrna_only_prob"].values
mrna_pred = (mrna_prob >= 0.5).astype(int)
ceil_prob = fused["fusion_ceiling_prob"].values
ceil_pred = (ceil_prob >= 0.5).astype(int)
floor_prob = fused["fusion_floor_prob"].values
floor_pred = (floor_prob >= 0.5).astype(int)

results["mrna_only_whole"] = bootstrap_ci(y_all, mrna_prob, mrna_pred, seed=4)
results["fusion_ceiling_whole"] = bootstrap_ci(y_all, ceil_prob, ceil_pred, seed=5)
results["fusion_floor_whole"] = bootstrap_ci(y_all, floor_prob, floor_pred, seed=6)
print("mRNA-only whole cohort:", json.dumps(results["mrna_only_whole"], indent=2))
print("Fusion ceiling whole cohort:", json.dumps(results["fusion_ceiling_whole"], indent=2))
print("Fusion floor whole cohort:", json.dumps(results["fusion_floor_whole"], indent=2))

# Low-confidence subset (tau < 0.6, n=58) — the key subgroup for the recommender
low_mask = fused["confidence_margin"].values < 0.6
n_low = int(low_mask.sum())
results["low_conf_subset_n"] = n_low
results["mrna_only_low_conf"] = bootstrap_ci(y_all[low_mask], mrna_prob[low_mask], mrna_pred[low_mask], seed=7)
results["fusion_ceiling_low_conf"] = bootstrap_ci(y_all[low_mask], ceil_prob[low_mask], ceil_pred[low_mask], seed=8)
results["fusion_floor_low_conf"] = bootstrap_ci(y_all[low_mask], floor_prob[low_mask], floor_pred[low_mask], seed=9)
print(f"Low-confidence subset (n={n_low}) mRNA-only:", json.dumps(results["mrna_only_low_conf"], indent=2))
print(f"Low-confidence subset (n={n_low}) fusion ceiling:", json.dumps(results["fusion_ceiling_low_conf"], indent=2))
print(f"Low-confidence subset (n={n_low}) fusion floor:", json.dumps(results["fusion_floor_low_conf"], indent=2))

# ============================================================
# 2. DeLong's test for paired AUC comparisons
# ============================================================
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_paired_test(y, prob_a, prob_b):
    """Paired DeLong test for AUC_a == AUC_b on the SAME sample set y.
    Returns (auc_a, auc_b, z_stat, p_value)."""
    pos = y == 1
    neg = y == 0
    m, n = pos.sum(), neg.sum()

    def structural_components(prob):
        pos_scores, neg_scores = prob[pos], prob[neg]
        tx = _compute_midrank(pos_scores)
        ty = _compute_midrank(neg_scores)
        tz = _compute_midrank(prob)
        v01 = (tz[pos] - tx) / n
        v10 = 1.0 - (tz[neg] - ty) / m
        auc = tz[pos].sum() / (m * n) - (m + 1) / (2 * n)
        return auc, v01, v10

    auc_a, v01_a, v10_a = structural_components(prob_a)
    auc_b, v01_b, v10_b = structural_components(prob_b)

    s01 = np.cov(np.vstack([v01_a, v01_b]))
    s10 = np.cov(np.vstack([v10_a, v10_b]))
    sigma = s01 / m + s10 / n
    var = sigma[0, 0] + sigma[1, 1] - 2 * sigma[0, 1]
    if var <= 0:
        return auc_a, auc_b, np.nan, 1.0
    z = (auc_a - auc_b) / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return auc_a, auc_b, z, p


print("\n=== DeLong's paired AUC tests (same patients, whole cohort n=697) ===")
delong_results = {}
auc_a, auc_b, z, p = delong_paired_test(y_all, mrna_prob, ceil_prob)
delong_results["mrna_vs_ceiling_whole"] = {"auc_mrna": auc_a, "auc_ceiling": auc_b, "z": z, "p_value": p}
print(f"mRNA-only (AUC={auc_a:.4f}) vs Fusion-ceiling (AUC={auc_b:.4f}): z={z:.3f}, p={p:.4g}")

auc_a, auc_b, z, p = delong_paired_test(y_all, mrna_prob, floor_prob)
delong_results["mrna_vs_floor_whole"] = {"auc_mrna": auc_a, "auc_floor": auc_b, "z": z, "p_value": p}
print(f"mRNA-only (AUC={auc_a:.4f}) vs Fusion-floor (AUC={auc_b:.4f}): z={z:.3f}, p={p:.4g}  [expect n.s. -- null check]")

auc_a, auc_b, z, p = delong_paired_test(y_all[low_mask], mrna_prob[low_mask], ceil_prob[low_mask])
delong_results["mrna_vs_ceiling_low_conf"] = {"auc_mrna": auc_a, "auc_ceiling": auc_b, "z": z, "p_value": p, "n": n_low}
print(f"[Low-conf subset n={n_low}] mRNA-only (AUC={auc_a:.4f}) vs Fusion-ceiling (AUC={auc_b:.4f}): z={z:.3f}, p={p:.4g}")

auc_a, auc_b, z, p = delong_paired_test(y_all[low_mask], mrna_prob[low_mask], floor_prob[low_mask])
delong_results["mrna_vs_floor_low_conf"] = {"auc_mrna": auc_a, "auc_floor": auc_b, "z": z, "p_value": p, "n": n_low}
print(f"[Low-conf subset n={n_low}] mRNA-only (AUC={auc_a:.4f}) vs Fusion-floor (AUC={auc_b:.4f}): z={z:.3f}, p={p:.4g}  [expect n.s.]")

# ============================================================
# 3. McNemar's exact test for paired prediction comparisons
# ============================================================
def mcnemar_test(y, pred_a, pred_b):
    correct_a = (pred_a == y)
    correct_b = (pred_b == y)
    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    table = [[int(np.sum(correct_a & correct_b)), b], [c, int(np.sum(~correct_a & ~correct_b))]]
    res = mcnemar(table, exact=True)
    return {"b_a_right_b_wrong": b, "c_a_wrong_b_right": c, "statistic": float(res.statistic), "p_value": float(res.pvalue)}


print("\n=== McNemar's exact tests (paired predictions, same patients) ===")
mcnemar_results = {}
mcnemar_results["mrna_vs_ceiling_whole"] = mcnemar_test(y_all, mrna_pred, ceil_pred)
print("mRNA-only vs Fusion-ceiling (whole cohort):", mcnemar_results["mrna_vs_ceiling_whole"])

mcnemar_results["mrna_vs_floor_whole"] = mcnemar_test(y_all, mrna_pred, floor_pred)
print("mRNA-only vs Fusion-floor (whole cohort) [expect n.s.]:", mcnemar_results["mrna_vs_floor_whole"])

mcnemar_results["mrna_vs_ceiling_low_conf"] = mcnemar_test(y_all[low_mask], mrna_pred[low_mask], ceil_pred[low_mask])
print(f"mRNA-only vs Fusion-ceiling (low-conf subset n={n_low}):", mcnemar_results["mrna_vs_ceiling_low_conf"])

mcnemar_results["mrna_vs_floor_low_conf"] = mcnemar_test(y_all[low_mask], mrna_pred[low_mask], floor_pred[low_mask])
print(f"mRNA-only vs Fusion-floor (low-conf subset n={n_low}) [expect n.s.]:", mcnemar_results["mrna_vs_floor_low_conf"])

# ============================================================
# 4. Two-proportion z-test: internal vs external accuracy (unpaired cohorts)
# ============================================================
def two_proportion_ztest(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return {"p1": p1, "p2": p2, "z": z, "p_value": p}


n_int, n_ext = len(s1_oof), len(s1_ext)
k_int = int((s1_oof["oof_pred"] == s1_oof["label"]).sum())
k_ext = int((s1_ext["ext_pred"] == s1_ext["label"]).sum())
prop_test = two_proportion_ztest(k_int, n_int, k_ext, n_ext)
prop_test.update({"n_internal": n_int, "n_external": n_ext, "k_internal_correct": k_int, "k_external_correct": k_ext})
print("\n=== Two-proportion z-test: Stage-1 internal vs external accuracy (unpaired cohorts) ===")
print(json.dumps(prop_test, indent=2))

# ============================================================
# Save everything
# ============================================================
all_stats = {
    "bootstrap_cis": results,
    "delong_tests": delong_results,
    "mcnemar_tests": mcnemar_results,
    "internal_vs_external_proportion_test": prop_test,
    "n_bootstrap_resamples": N_BOOT,
}
with open(RES / "statistical_tests_summary.json", "w") as f:
    json.dump(all_stats, f, indent=2)
print("\nSaved statistical_tests_summary.json")
