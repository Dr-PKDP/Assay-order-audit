"""Run the trained Stage-1 (mRNA) and Stage-2 (methylation) classifiers on the
two NEW independent external cohorts:
  - Stage-1 second external mRNA cohort: GSE5281, superior frontal gyrus (SFG),
    n=34 (23 AD, 11 control), Affymetrix GPL570.
  - Stage-2 first-ever external methylation cohort: GSE134379, middle temporal
    gyrus (Banner Sun Health / "Arizona-1"), n=404 (225 AD, 179 non-demented),
    Illumina 450K.

Both cohorts are entirely independent of the primary training/CV data and of
each other (different institutions/brain banks, different array batches).
Reuses the same bootstrap-ensemble protocol and refit-on-common-features
pattern as scripts 05/06, plus the bootstrap-CI method from script 09.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, brier_score_loss, balanced_accuracy_score, confusion_matrix
from sklearn.utils import resample

np.random.seed(42)
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"

N_BOOT_CI = 2000


def fit_bootstrap_ensemble(X_train, y_train, n_boot, k, seed=0):
    ensemble = []
    rng = np.random.RandomState(seed)
    n = X_train.shape[0]
    for b in range(n_boot):
        idx = resample(np.arange(n), replace=True, n_samples=n, random_state=rng.randint(0, 1_000_000), stratify=y_train)
        Xb, yb = X_train[idx], y_train[idx]
        scaler = StandardScaler().fit(Xb)
        Xb_s = scaler.transform(Xb)
        k_use = min(k, Xb_s.shape[1])
        selector = SelectKBest(f_classif, k=k_use).fit(Xb_s, yb)
        Xb_sel = selector.transform(Xb_s)
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, solver="lbfgs")
        clf.fit(Xb_sel, yb)
        ensemble.append((scaler, selector, clf))
    return ensemble


def bootstrap_ci(y, prob, pred=None, n_boot=N_BOOT_CI, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y)
    if pred is None:
        pred = (prob >= 0.5).astype(int)
    metrics = {"accuracy": [], "balanced_accuracy": [], "auc": [], "f1": [], "brier": []}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb, pb, prb = y[idx], pred[idx], prob[idx]
        if len(np.unique(yb)) < 2:
            continue
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


summary_all = {}

# ============================================================
# Stage-1: second external mRNA cohort (GSE5281, SFG)
# ============================================================
print("=" * 70)
print("STAGE-1 second external validation: GSE5281 (SFG, n=34)")
print("=" * 70)

gene = pd.read_csv(RAW / "gene_data.csv")
feature_cols = [c for c in gene.columns if c not in ("sample", "label")]
X_full = gene[feature_cols].values
y_full = gene["label"].values.astype(int)

ext2 = pd.read_csv(PROC / "external_validation2_GSE5281_SFG_AD_control.csv")
ext2_feature_cols = [c for c in ext2.columns if c not in ("sample", "label")]
common_features_s1b = [c for c in feature_cols if c in ext2_feature_cols]
print(f"Common features (training panel intersect GSE5281 SFG): {len(common_features_s1b)} / {len(feature_cols)}")

X_common_s1b = gene[common_features_s1b].values
final_ensemble_s1b = fit_bootstrap_ensemble(X_common_s1b, y_full, n_boot=25, k=300, seed=1998)

X_ext2_raw = ext2[common_features_s1b].values
X_ext2_scaled_self = StandardScaler().fit_transform(X_ext2_raw)

ext2_preds = []
for scaler, selector, clf in final_ensemble_s1b:
    Xt = selector.transform(X_ext2_scaled_self)
    ext2_preds.append(clf.predict_proba(Xt)[:, 1])
ext2_preds = np.array(ext2_preds)
ext2_prob = ext2_preds.mean(axis=0)
ext2_unc = ext2_preds.std(axis=0)
ext2_pred = (ext2_prob >= 0.5).astype(int)
ext2_y = ext2["label"].values.astype(int)

ext2_acc = accuracy_score(ext2_y, ext2_pred)
ext2_bal_acc = balanced_accuracy_score(ext2_y, ext2_pred)
ext2_auc = roc_auc_score(ext2_y, ext2_prob)
ext2_f1 = f1_score(ext2_y, ext2_pred)
ext2_brier = brier_score_loss(ext2_y, ext2_prob)
ext2_cm = confusion_matrix(ext2_y, ext2_pred)

print(f"Accuracy: {ext2_acc:.4f}  Balanced accuracy: {ext2_bal_acc:.4f}  AUC: {ext2_auc:.4f}  F1: {ext2_f1:.4f}  Brier: {ext2_brier:.4f}")
print("Confusion matrix [control, AD]:\n", ext2_cm)

ext2_df = pd.DataFrame({
    "sample": ext2["sample"].values, "label": ext2_y, "ext_prob": ext2_prob, "ext_uncertainty": ext2_unc,
    "ext_pred": ext2_pred, "confidence_margin": np.abs(ext2_prob - 0.5) * 2,
})
ext2_df.to_csv(RES / "stage1_mrna_external2_GSE5281_predictions.csv", index=False)

ci_s1_ext2 = bootstrap_ci(ext2_y, ext2_prob, ext2_pred, seed=101)
print("Bootstrap 95% CIs:", json.dumps(ci_s1_ext2, indent=2))

summary_all["stage1_external2_GSE5281_SFG"] = {
    "cohort_n": int(len(ext2_y)), "label_balance": {"AD": int((ext2_y == 1).sum()), "control": int((ext2_y == 0).sum())},
    "common_features": len(common_features_s1b),
    "accuracy": ext2_acc, "balanced_accuracy": ext2_bal_acc, "auc": ext2_auc, "f1": ext2_f1, "brier": ext2_brier,
    "confusion_matrix": ext2_cm.tolist(), "bootstrap_ci": ci_s1_ext2,
}

# ============================================================
# Stage-2: first external methylation cohort (GSE134379, MTG)
# ============================================================
print("\n" + "=" * 70)
print("STAGE-2 external validation: GSE134379 (MTG, n=404)")
print("=" * 70)

meth = pd.read_csv(RAW / "methylation_data.csv")
meth_feature_cols = [c for c in meth.columns if c not in ("sample", "label")]
X_meth_full = meth[meth_feature_cols].values
y_meth_full = meth["label"].values.astype(int)

ext_meth = pd.read_csv(PROC / "GSE134379_mtg_methylation_gene_level.csv")
ext_meth_feature_cols = [c for c in ext_meth.columns if c not in ("sample", "label")]
common_features_s2 = [c for c in meth_feature_cols if c in ext_meth_feature_cols]
print(f"Common features (training panel intersect GSE134379 MTG): {len(common_features_s2)} / {len(meth_feature_cols)}")

X_common_s2 = meth[common_features_s2].values
final_ensemble_s2 = fit_bootstrap_ensemble(X_common_s2, y_meth_full, n_boot=25, k=150, seed=2998)

X_ext_meth_raw = ext_meth[common_features_s2].values
X_ext_meth_scaled_self = StandardScaler().fit_transform(X_ext_meth_raw)

ext_meth_preds = []
for scaler, selector, clf in final_ensemble_s2:
    Xt = selector.transform(X_ext_meth_scaled_self)
    ext_meth_preds.append(clf.predict_proba(Xt)[:, 1])
ext_meth_preds = np.array(ext_meth_preds)
ext_meth_prob = ext_meth_preds.mean(axis=0)
ext_meth_unc = ext_meth_preds.std(axis=0)
ext_meth_pred = (ext_meth_prob >= 0.5).astype(int)
ext_meth_y = ext_meth["label"].values.astype(int)

ext_meth_acc = accuracy_score(ext_meth_y, ext_meth_pred)
ext_meth_bal_acc = balanced_accuracy_score(ext_meth_y, ext_meth_pred)
ext_meth_auc = roc_auc_score(ext_meth_y, ext_meth_prob)
ext_meth_f1 = f1_score(ext_meth_y, ext_meth_pred)
ext_meth_brier = brier_score_loss(ext_meth_y, ext_meth_prob)
ext_meth_cm = confusion_matrix(ext_meth_y, ext_meth_pred)

print(f"Accuracy: {ext_meth_acc:.4f}  Balanced accuracy: {ext_meth_bal_acc:.4f}  AUC: {ext_meth_auc:.4f}  F1: {ext_meth_f1:.4f}  Brier: {ext_meth_brier:.4f}")
print("Confusion matrix [control, AD]:\n", ext_meth_cm)

ext_meth_df = pd.DataFrame({
    "sample": ext_meth["sample"].values, "label": ext_meth_y, "ext_prob": ext_meth_prob, "ext_uncertainty": ext_meth_unc,
    "ext_pred": ext_meth_pred, "confidence_margin": np.abs(ext_meth_prob - 0.5) * 2,
})
ext_meth_df.to_csv(RES / "stage2_methylation_external_GSE134379_predictions.csv", index=False)

ci_s2_ext = bootstrap_ci(ext_meth_y, ext_meth_prob, ext_meth_pred, seed=102)
print("Bootstrap 95% CIs:", json.dumps(ci_s2_ext, indent=2))

summary_all["stage2_external_GSE134379_MTG"] = {
    "cohort_n": int(len(ext_meth_y)), "label_balance": {"AD": int((ext_meth_y == 1).sum()), "control": int((ext_meth_y == 0).sum())},
    "common_features": len(common_features_s2),
    "accuracy": ext_meth_acc, "balanced_accuracy": ext_meth_bal_acc, "auc": ext_meth_auc, "f1": ext_meth_f1, "brier": ext_meth_brier,
    "confusion_matrix": ext_meth_cm.tolist(), "bootstrap_ci": ci_s2_ext,
}

with open(RES / "new_external_validations_summary.json", "w") as f:
    json.dump(summary_all, f, indent=2)
print("\nSaved new_external_validations_summary.json")
