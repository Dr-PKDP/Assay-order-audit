"""Simulated late-fusion analysis + cost-aware Pareto curve.

Because GSE33000/GSE44770 (mRNA) and GSE80970 (methylation) are unpaired
studies (no single patient has both measurements), any "what if this patient
also had a methylation result" estimate is necessarily a simulation, not a
validated per-patient result. To keep this honest, we report an explicit
sensitivity band:
  - CEILING: partners drawn from the SAME true diagnostic class (the
    convention used by AE-Trans/UnCOT-AD) -> optimistic upper bound on the
    value of a second assay.
  - FLOOR: partners drawn at random, ignoring diagnostic class -> what a
    label-uninformative "assay" would look like; if fusion improves accuracy
    even here, that signals leakage in the meta-fusion model rather than real
    signal.
The gap between ceiling and floor is reported as a sensitivity range in the
cost-accuracy Pareto curve, not as a fabricated point estimate.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

np.random.seed(7)
RES = ROOT / "results"

s1 = pd.read_csv(RES / "stage1_mrna_oof_predictions.csv")          # n=697
s2 = pd.read_csv(RES / "stage2_methylation_oof_predictions.csv")   # n=142

EPS = 1e-4
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

s1_logit = logit(s1["oof_prob"].values)
s2_logit = logit(s2["oof_prob"].values)
s1_label = s1["label"].values
s2_label = s2["label"].values

N_PARTNERS = 20  # partner draws averaged per mRNA sample, per regime

def build_fused_predictions(regime, meta_weights=None, fit_meta=False):
    """regime: 'same_label' or 'random'. Returns fused_prob array (n=697,) and,
    if fit_meta, also the fitted LogisticRegression meta-model (2 features:
    logit_mRNA, logit_methyl)."""
    rng = np.random.RandomState(123 if regime == "same_label" else 456)
    fused_logit_features = np.zeros((len(s1_logit), 2))  # [logit_mRNA, mean logit_methyl_partner]
    for i in range(len(s1_logit)):
        if regime == "same_label":
            pool_idx = np.where(s2_label == s1_label[i])[0]
        else:
            pool_idx = np.arange(len(s2_label))
        draw = rng.choice(pool_idx, size=min(N_PARTNERS, len(pool_idx)), replace=True)
        fused_logit_features[i, 0] = s1_logit[i]
        fused_logit_features[i, 1] = s2_logit[draw].mean()
    return fused_logit_features

# ---- Fit the fusion meta-model on SAME-LABEL synthetic pairs via 5-fold CV
#      (legitimate at training time: labels are known for all training rows) ----
same_label_features = build_fused_predictions("same_label")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_oof_prob_ceiling = np.zeros(len(s1_label))
for train_idx, test_idx in skf.split(same_label_features, s1_label):
    meta = LogisticRegression(max_iter=2000).fit(same_label_features[train_idx], s1_label[train_idx])
    meta_oof_prob_ceiling[test_idx] = meta.predict_proba(same_label_features[test_idx])[:, 1]

# Fit a separate meta-model on RANDOM-pairing features for the floor scenario,
# using the SAME meta architecture/procedure so the comparison is apples-to-apples
random_label_features = build_fused_predictions("random")
meta_oof_prob_floor = np.zeros(len(s1_label))
for train_idx, test_idx in skf.split(random_label_features, s1_label):
    meta = LogisticRegression(max_iter=2000).fit(random_label_features[train_idx], s1_label[train_idx])
    meta_oof_prob_floor[test_idx] = meta.predict_proba(random_label_features[test_idx])[:, 1]

mrna_only_prob = s1["oof_prob"].values
mrna_pred = (mrna_only_prob >= 0.5).astype(int)
ceiling_pred = (meta_oof_prob_ceiling >= 0.5).astype(int)
floor_pred = (meta_oof_prob_floor >= 0.5).astype(int)

print("=== Simulated fusion sensitivity band (whole cohort, n=697) ===")
print(f"mRNA-only:            acc={accuracy_score(s1_label, mrna_pred):.4f}  auc={roc_auc_score(s1_label, mrna_only_prob):.4f}")
print(f"Fusion CEILING (same-label partners): acc={accuracy_score(s1_label, ceiling_pred):.4f}  auc={roc_auc_score(s1_label, meta_oof_prob_ceiling):.4f}")
print(f"Fusion FLOOR (random partners):       acc={accuracy_score(s1_label, floor_pred):.4f}  auc={roc_auc_score(s1_label, meta_oof_prob_floor):.4f}")

# ---- Restrict to low-confidence mRNA subset (the patients the recommender
#      would actually flag for a second assay) ----
conf = s1["confidence_margin"].values
print("\n=== Fusion benefit restricted to low-confidence mRNA subsets ===")
low_conf_rows = []
for tau in [0.9, 0.8, 0.6, 0.4, 0.2]:
    mask = conf < tau
    if mask.sum() < 5:
        continue
    acc_mrna = accuracy_score(s1_label[mask], mrna_pred[mask])
    acc_ceil = accuracy_score(s1_label[mask], ceiling_pred[mask])
    acc_floor = accuracy_score(s1_label[mask], floor_pred[mask])
    low_conf_rows.append({"tau_below": tau, "n_patients": int(mask.sum()), "acc_mrna_only": acc_mrna,
                            "acc_fusion_ceiling": acc_ceil, "acc_fusion_floor": acc_floor})
    print(f"  confidence < {tau}: n={mask.sum():4d}  mRNA-only={acc_mrna:.4f}  fusion-ceiling={acc_ceil:.4f}  fusion-floor={acc_floor:.4f}")

pd.DataFrame(low_conf_rows).to_csv(RES / "fusion_low_confidence_benefit.csv", index=False)

fused_df = pd.DataFrame({
    "sample": s1["sample"], "label": s1_label, "confidence_margin": conf,
    "mrna_only_prob": mrna_only_prob, "fusion_ceiling_prob": meta_oof_prob_ceiling,
    "fusion_floor_prob": meta_oof_prob_floor,
})
fused_df.to_csv(RES / "fusion_predictions_all.csv", index=False)

# ============================================================
# COST-AWARE PARETO ANALYSIS
# ============================================================
# Relative assay costs (cited): mRNA-seq academic core pricing clusters
# ~$150-$300/sample (e.g. Indiana University Medical Genomics Core $250-300,
# UTHealth Houston CGC $159-209); Illumina MethylationEPIC array pricing
# clusters ~$300-$450/sample (Univ. of Iowa $412, Dartmouth $343, USC Norris
# $339-408, JHU GRCF $370-410). We take representative midpoints of
# $275 (mRNA) and $400 (methylation) => cost ratio (methylation/mRNA) = 1.45,
# and run a sensitivity sweep over the ratio (1.2-1.8) since real institutional
# pricing varies substantially.
MRNA_COST = 1.0
METH_COST_RATIOS = [1.2, 1.45, 1.8]  # sensitivity band around the literature midpoint

taus = np.round(np.arange(0.0, 1.001, 0.02), 2)
pareto_rows = []
for tau in taus:
    resolved_mask = conf >= tau
    pct_resolved = resolved_mask.mean()
    if resolved_mask.sum() > 0:
        acc_resolved_stage1 = accuracy_score(s1_label[resolved_mask], mrna_pred[resolved_mask])
    else:
        acc_resolved_stage1 = np.nan
    unresolved_mask = ~resolved_mask
    n_unresolved = unresolved_mask.sum()
    if n_unresolved > 0:
        acc_unresolved_ceiling = accuracy_score(s1_label[unresolved_mask], ceiling_pred[unresolved_mask])
        acc_unresolved_floor = accuracy_score(s1_label[unresolved_mask], floor_pred[unresolved_mask])
    else:
        acc_unresolved_ceiling = acc_unresolved_floor = np.nan

    overall_acc_ceiling = (resolved_mask.sum() * acc_resolved_stage1 + n_unresolved * acc_unresolved_ceiling) / len(s1_label) if resolved_mask.sum() > 0 else acc_unresolved_ceiling
    overall_acc_floor = (resolved_mask.sum() * acc_resolved_stage1 + n_unresolved * acc_unresolved_floor) / len(s1_label) if resolved_mask.sum() > 0 else acc_unresolved_floor

    for ratio in METH_COST_RATIOS:
        avg_cost = pct_resolved * MRNA_COST + (1 - pct_resolved) * (MRNA_COST + ratio)
        pareto_rows.append({
            "tau": tau, "meth_cost_ratio": ratio, "pct_resolved_stage1": pct_resolved,
            "avg_cost_per_patient": avg_cost,
            "overall_accuracy_ceiling": overall_acc_ceiling, "overall_accuracy_floor": overall_acc_floor,
        })

pareto_df = pd.DataFrame(pareto_rows)
pareto_df.to_csv(RES / "cost_accuracy_pareto.csv", index=False)

# Fixed baselines for comparison (tau=0 -> everyone gets mRNA only; tau=1 -> everyone gets full panel)
mrna_only_baseline_acc = accuracy_score(s1_label, mrna_pred)
full_panel_ceiling_acc = accuracy_score(s1_label, ceiling_pred)
full_panel_floor_acc = accuracy_score(s1_label, floor_pred)

baselines = {
    "mrna_only_fixed": {"avg_cost": MRNA_COST, "accuracy": mrna_only_baseline_acc},
    "full_panel_fixed_ceiling": {"avg_cost_ratio_1.45": MRNA_COST + 1.45, "accuracy": full_panel_ceiling_acc},
    "full_panel_fixed_floor": {"avg_cost_ratio_1.45": MRNA_COST + 1.45, "accuracy": full_panel_floor_acc},
}
print("\n=== Fixed baselines ===")
print(json.dumps(baselines, indent=2))

with open(RES / "fusion_and_pareto_summary.json", "w") as f:
    json.dump({
        "whole_cohort": {
            "mrna_only": {"acc": accuracy_score(s1_label, mrna_pred), "auc": roc_auc_score(s1_label, mrna_only_prob)},
            "fusion_ceiling": {"acc": accuracy_score(s1_label, ceiling_pred), "auc": roc_auc_score(s1_label, meta_oof_prob_ceiling)},
            "fusion_floor": {"acc": accuracy_score(s1_label, floor_pred), "auc": roc_auc_score(s1_label, meta_oof_prob_floor)},
        },
        "baselines": baselines,
        "n_partners_averaged": N_PARTNERS,
        "meth_cost_ratios_tested": METH_COST_RATIOS,
    }, f, indent=2)

print("\nSaved fusion_low_confidence_benefit.csv, fusion_predictions_all.csv, cost_accuracy_pareto.csv, fusion_and_pareto_summary.json")
