"""Ablation study for the Cost-Aware Adaptive Multi-Omics Panel Recommender.

Ablates, on the primary training cohorts (Stage-1 mRNA n=697; Stage-2
methylation n=142), using a single stratified 5-fold CV split per configuration
(not the repeated 5x5 protocol used for the headline numbers, for computational
tractability -- point estimates only, no CIs, since this is a design-sensitivity
sweep rather than a headline result):

  (a) Bootstrap ensemble size: n_boot in {1, 5, 10, 25}
  (b) Feature-selection size (top-K ANOVA F-test features): sweep per stage
  (c) Class-balancing on/off (class_weight='balanced' vs None)
  (d) Fusion meta-model design: simple logit-averaging vs the learned
      2-feature logistic meta-model, on both the ceiling and floor simulated-
      fusion regimes (whole cohort, n=697)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.utils import resample

np.random.seed(42)
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"


def fit_bootstrap_ensemble(X_train, y_train, n_boot, k, seed=0, balanced=True):
    ensemble = []
    rng = np.random.RandomState(seed)
    n = X_train.shape[0]
    cw = "balanced" if balanced else None
    for b in range(n_boot):
        if n_boot == 1:
            # no bootstrap resampling when ensemble size is 1 (single model on
            # the full training fold) -- isolates the ensembling effect itself
            Xb, yb = X_train, y_train
        else:
            idx = resample(np.arange(n), replace=True, n_samples=n, random_state=rng.randint(0, 1_000_000), stratify=y_train)
            Xb, yb = X_train[idx], y_train[idx]
        scaler = StandardScaler().fit(Xb)
        Xb_s = scaler.transform(Xb)
        k_use = min(k, Xb_s.shape[1])
        selector = SelectKBest(f_classif, k=k_use).fit(Xb_s, yb)
        Xb_sel = selector.transform(Xb_s)
        clf = LogisticRegression(C=1.0, class_weight=cw, max_iter=3000, solver="lbfgs")
        clf.fit(Xb_sel, yb)
        ensemble.append((scaler, selector, clf))
    return ensemble


def predict_ensemble(ensemble, X_test):
    preds = []
    for scaler, selector, clf in ensemble:
        Xt = selector.transform(scaler.transform(X_test))
        preds.append(clf.predict_proba(Xt)[:, 1])
    preds = np.array(preds)
    return preds.mean(axis=0)


def single_5fold_cv(X, y, n_boot, k, balanced=True, seed=42):
    """Single stratified 5-fold CV (not repeated); returns acc, bal_acc, auc."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_prob = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        ens = fit_bootstrap_ensemble(X[train_idx], y[train_idx], n_boot=n_boot, k=k, seed=seed, balanced=balanced)
        oof_prob[test_idx] = predict_ensemble(ens, X[test_idx])
    oof_pred = (oof_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, oof_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, oof_pred)),
        "auc": float(roc_auc_score(y, oof_prob)),
    }


print("Loading primary cohorts...")
gene = pd.read_csv(RAW / "gene_data.csv")
s1_feature_cols = [c for c in gene.columns if c not in ("sample", "label")]
X1 = gene[s1_feature_cols].values
y1 = gene["label"].values.astype(int)

meth = pd.read_csv(RAW / "methylation_data.csv")
s2_feature_cols = [c for c in meth.columns if c not in ("sample", "label")]
X2 = meth[s2_feature_cols].values
y2 = meth["label"].values.astype(int)

print(f"Stage-1 (mRNA): X={X1.shape}, y balance={np.bincount(y1)}")
print(f"Stage-2 (methylation): X={X2.shape}, y balance={np.bincount(y2)}")

ablation = {"stage1": {}, "stage2": {}}

# ============================================================
# (a) Ensemble size ablation
# ============================================================
print("\n=== (a) Bootstrap ensemble size ablation (single 5-fold CV) ===")
ensemble_sizes = [1, 5, 10, 25]
ablation["stage1"]["ensemble_size"] = []
for n_boot in ensemble_sizes:
    res = single_5fold_cv(X1, y1, n_boot=n_boot, k=300, balanced=True)
    res["n_boot"] = n_boot
    ablation["stage1"]["ensemble_size"].append(res)
    print(f"  Stage-1  n_boot={n_boot:3d}: acc={res['accuracy']:.4f}  bal_acc={res['balanced_accuracy']:.4f}  auc={res['auc']:.4f}")

ablation["stage2"]["ensemble_size"] = []
for n_boot in ensemble_sizes:
    res = single_5fold_cv(X2, y2, n_boot=n_boot, k=150, balanced=True)
    res["n_boot"] = n_boot
    ablation["stage2"]["ensemble_size"].append(res)
    print(f"  Stage-2  n_boot={n_boot:3d}: acc={res['accuracy']:.4f}  bal_acc={res['balanced_accuracy']:.4f}  auc={res['auc']:.4f}")

# ============================================================
# (b) Feature-selection size (top-K) ablation
# ============================================================
print("\n=== (b) Feature-selection size (top-K) ablation (single 5-fold CV, n_boot=25) ===")
k_grid_s1 = [50, 100, 150, 300, 500, 1000]
ablation["stage1"]["top_k"] = []
for k in k_grid_s1:
    res = single_5fold_cv(X1, y1, n_boot=25, k=k, balanced=True)
    res["k"] = k
    ablation["stage1"]["top_k"].append(res)
    print(f"  Stage-1  K={k:5d}: acc={res['accuracy']:.4f}  bal_acc={res['balanced_accuracy']:.4f}  auc={res['auc']:.4f}")

k_grid_s2 = [25, 50, 100, 150, 300, 500]
ablation["stage2"]["top_k"] = []
for k in k_grid_s2:
    res = single_5fold_cv(X2, y2, n_boot=25, k=k, balanced=True)
    res["k"] = k
    ablation["stage2"]["top_k"].append(res)
    print(f"  Stage-2  K={k:5d}: acc={res['accuracy']:.4f}  bal_acc={res['balanced_accuracy']:.4f}  auc={res['auc']:.4f}")

# ============================================================
# (c) Class-balancing on/off
# ============================================================
print("\n=== (c) Class-balancing on/off (single 5-fold CV, n_boot=25, baseline K) ===")
ablation["stage1"]["class_balancing"] = []
for balanced in [True, False]:
    res = single_5fold_cv(X1, y1, n_boot=25, k=300, balanced=balanced)
    res["class_weight_balanced"] = balanced
    ablation["stage1"]["class_balancing"].append(res)
    print(f"  Stage-1  balanced={balanced}: acc={res['accuracy']:.4f}  bal_acc={res['balanced_accuracy']:.4f}  auc={res['auc']:.4f}")

ablation["stage2"]["class_balancing"] = []
for balanced in [True, False]:
    res = single_5fold_cv(X2, y2, n_boot=25, k=150, balanced=balanced)
    res["class_weight_balanced"] = balanced
    ablation["stage2"]["class_balancing"].append(res)
    print(f"  Stage-2  balanced={balanced}: acc={res['accuracy']:.4f}  bal_acc={res['balanced_accuracy']:.4f}  auc={res['auc']:.4f}")

# ============================================================
# (d) Fusion meta-model design: simple averaging vs learned meta-model
# ============================================================
print("\n=== (d) Fusion meta-model design: simple logit-averaging vs learned meta-model ===")
s1_oof = pd.read_csv(RES / "stage1_mrna_oof_predictions.csv")
s2_oof = pd.read_csv(RES / "stage2_methylation_oof_predictions.csv")
fused = pd.read_csv(RES / "fusion_predictions_all.csv")  # contains the learned-meta-model ceiling/floor probs already

EPS = 1e-4
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

s1_logit = logit(s1_oof["oof_prob"].values)
s2_logit = logit(s2_oof["oof_prob"].values)
s1_label = s1_oof["label"].values
s2_label = s2_oof["label"].values

N_PARTNERS = 20
def build_partner_logit(regime, seed):
    rng = np.random.RandomState(seed)
    partner_logit = np.zeros(len(s1_logit))
    for i in range(len(s1_logit)):
        if regime == "same_label":
            pool_idx = np.where(s2_label == s1_label[i])[0]
        else:
            pool_idx = np.arange(len(s2_label))
        draw = rng.choice(pool_idx, size=min(N_PARTNERS, len(pool_idx)), replace=True)
        partner_logit[i] = s2_logit[draw].mean()
    return partner_logit

# Reuse the SAME partner draws/seeds as script 07 for an apples-to-apples comparison
partner_logit_ceiling = build_partner_logit("same_label", seed=123)
partner_logit_floor = build_partner_logit("random", seed=456)

simple_avg_ceiling_prob = sigmoid(0.5 * s1_logit + 0.5 * partner_logit_ceiling)
simple_avg_floor_prob = sigmoid(0.5 * s1_logit + 0.5 * partner_logit_floor)

learned_ceiling_prob = fused["fusion_ceiling_prob"].values
learned_floor_prob = fused["fusion_floor_prob"].values
mrna_only_prob = fused["mrna_only_prob"].values

def summarize(y, prob, name):
    pred = (prob >= 0.5).astype(int)
    return {"name": name, "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "auc": float(roc_auc_score(y, prob))}

fusion_ablation = [
    summarize(s1_label, mrna_only_prob, "mRNA-only (no fusion)"),
    summarize(s1_label, simple_avg_ceiling_prob, "Simple averaging - CEILING"),
    summarize(s1_label, learned_ceiling_prob, "Learned meta-model - CEILING"),
    summarize(s1_label, simple_avg_floor_prob, "Simple averaging - FLOOR"),
    summarize(s1_label, learned_floor_prob, "Learned meta-model - FLOOR"),
]
for row in fusion_ablation:
    print(f"  {row['name']:32s}: acc={row['accuracy']:.4f}  bal_acc={row['balanced_accuracy']:.4f}  auc={row['auc']:.4f}")

ablation["fusion_meta_model_design"] = fusion_ablation

with open(RES / "ablation_study_summary.json", "w") as f:
    json.dump(ablation, f, indent=2)
print("\nSaved ablation_study_summary.json")
