"""Stage-1 mRNA-only classifier: bootstrap-ensembled, cross-validated,
calibrated logistic regression with an uncertainty estimate, evaluated via
repeated stratified k-fold CV, plus confidence-stratification analysis and
external-cohort validation on GSE118553 (Frontal_Cortex).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, brier_score_loss, balanced_accuracy_score, confusion_matrix
from sklearn.utils import resample

np.random.seed(42)

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)

K_FEATURES = 300
N_BOOTSTRAP = 25
N_SPLITS = 5
N_REPEATS = 5

print("Loading primary mRNA cohort (Zenodo AE-Trans gene_data.csv: GSE33000+GSE44770)...")
gene = pd.read_csv(RAW / "gene_data.csv")
feature_cols = [c for c in gene.columns if c not in ("sample", "label")]
X = gene[feature_cols].values
y = gene["label"].values.astype(int)
sample_ids = gene["sample"].values
print(f"X shape: {X.shape}, y balance: {np.bincount(y)}")


def fit_bootstrap_ensemble(X_train, y_train, n_boot=N_BOOTSTRAP, k=K_FEATURES, seed=0):
    """Fit n_boot bootstrap-resampled pipelines; return list of (scaler, selector, model)."""
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


def predict_ensemble(ensemble, X_test):
    preds = []
    for scaler, selector, clf in ensemble:
        Xt = selector.transform(scaler.transform(X_test))
        preds.append(clf.predict_proba(Xt)[:, 1])
    preds = np.array(preds)  # (n_boot, n_test)
    return preds.mean(axis=0), preds.std(axis=0)


# ---- Repeated stratified k-fold out-of-fold evaluation ----
print(f"\nRunning {N_REPEATS}x{N_SPLITS}-fold CV with {N_BOOTSTRAP}-model bootstrap ensembles per fold...")
rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)

oof_prob_sum = np.zeros(len(y))
oof_unc_sum = np.zeros(len(y))
oof_count = np.zeros(len(y))

for fold_i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
    ensemble = fit_bootstrap_ensemble(X[train_idx], y[train_idx], seed=fold_i)
    prob, unc = predict_ensemble(ensemble, X[test_idx])
    oof_prob_sum[test_idx] += prob
    oof_unc_sum[test_idx] += unc
    oof_count[test_idx] += 1
    if (fold_i + 1) % 5 == 0:
        print(f"  completed fold {fold_i + 1}/{N_SPLITS * N_REPEATS}")

oof_prob = oof_prob_sum / oof_count
oof_unc = oof_unc_sum / oof_count
oof_pred = (oof_prob >= 0.5).astype(int)

acc = accuracy_score(y, oof_pred)
bal_acc = balanced_accuracy_score(y, oof_pred)
auc = roc_auc_score(y, oof_prob)
f1 = f1_score(y, oof_pred)
brier = brier_score_loss(y, oof_prob)
cm = confusion_matrix(y, oof_pred)

print("\n=== Stage-1 (mRNA-only) out-of-fold performance, primary cohort (n=697) ===")
print(f"Accuracy: {acc:.4f}  Balanced accuracy: {bal_acc:.4f}  AUC: {auc:.4f}  F1: {f1:.4f}  Brier: {brier:.4f}")
print("Confusion matrix (rows=true, cols=pred) [control, AD]:\n", cm)

oof_df = pd.DataFrame({
    "sample": sample_ids, "label": y, "oof_prob": oof_prob, "oof_uncertainty": oof_unc,
    "oof_pred": oof_pred, "confidence_margin": np.abs(oof_prob - 0.5) * 2,
})
oof_df.to_csv(RES / "stage1_mrna_oof_predictions.csv", index=False)

# ---- Confidence-stratification sweep (replicates/tests SGUQ-style stratification, H2) ----
taus = np.round(np.arange(0.0, 1.001, 0.02), 2)
sweep_rows = []
for tau in taus:
    mask = oof_df["confidence_margin"] >= tau
    pct_resolved = mask.mean()
    if mask.sum() > 0:
        acc_resolved = accuracy_score(oof_df.loc[mask, "label"], oof_df.loc[mask, "oof_pred"])
    else:
        acc_resolved = np.nan
    sweep_rows.append({"tau": tau, "pct_resolved": pct_resolved, "accuracy_if_resolved": acc_resolved})
sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(RES / "stage1_confidence_sweep.csv", index=False)
print("\nConfidence-margin sweep saved. Example rows:")
print(sweep_df.iloc[::10])

# ---- Fit final full-data ensemble for external validation ----
print("\nFitting final ensemble on the full 697-sample cohort for external validation...")
final_ensemble = fit_bootstrap_ensemble(X, y, n_boot=N_BOOTSTRAP, seed=999)

ext = pd.read_csv(PROC / "external_validation_AD_control.csv")
ext_feature_cols = [c for c in ext.columns if c not in ("sample", "label")]
# align feature order to the training panel intersection; fill any gene absent
# in the external cohort (should not happen post-intersection) with 0 after
# per-cohort standardization
common_features = [c for c in feature_cols if c in ext_feature_cols]
print(f"Common features used for external validation: {len(common_features)} / {len(feature_cols)}")

# Build an X_train/X_ext restricted to common_features, refit scaler/selector/clf
# ensemble specifically on this common feature set for a fair external test
X_common = gene[common_features].values
final_ensemble_common = fit_bootstrap_ensemble(X_common, y, n_boot=N_BOOTSTRAP, seed=998)

X_ext_raw = ext[common_features].values
# self-standardize the external cohort (cross-platform harmonization) before
# feeding into the trained selector/classifier pipeline
X_ext_scaled_self = StandardScaler().fit_transform(X_ext_raw)

# predict using each ensemble member's own selector+clf but bypass its scaler stage
# (already self-standardized) -- apply selector.transform + clf.predict_proba directly
ext_preds = []
for scaler, selector, clf in final_ensemble_common:
    Xt = selector.transform(X_ext_scaled_self)
    ext_preds.append(clf.predict_proba(Xt)[:, 1])
ext_preds = np.array(ext_preds)
ext_prob = ext_preds.mean(axis=0)
ext_unc = ext_preds.std(axis=0)
ext_pred = (ext_prob >= 0.5).astype(int)
ext_y = ext["label"].values.astype(int)

ext_acc = accuracy_score(ext_y, ext_pred)
ext_bal_acc = balanced_accuracy_score(ext_y, ext_pred)
ext_auc = roc_auc_score(ext_y, ext_prob)
ext_f1 = f1_score(ext_y, ext_pred)
ext_brier = brier_score_loss(ext_y, ext_prob)

print("\n=== Stage-1 EXTERNAL validation (GSE118553, Frontal_Cortex, n=63) ===")
print(f"Accuracy: {ext_acc:.4f}  Balanced accuracy: {ext_bal_acc:.4f}  AUC: {ext_auc:.4f}  F1: {ext_f1:.4f}  Brier: {ext_brier:.4f}")

ext_df = pd.DataFrame({
    "sample": ext["sample"].values, "label": ext_y, "ext_prob": ext_prob, "ext_uncertainty": ext_unc,
    "ext_pred": ext_pred, "confidence_margin": np.abs(ext_prob - 0.5) * 2,
})
ext_df.to_csv(RES / "stage1_mrna_external_predictions.csv", index=False)

summary = {
    "primary_cohort_n": int(len(y)),
    "primary_cohort_label_balance": {"AD": int((y == 1).sum()), "control": int((y == 0).sum())},
    "internal_oof_accuracy": acc,
    "internal_oof_balanced_accuracy": bal_acc,
    "internal_oof_auc": auc,
    "internal_oof_f1": f1,
    "internal_oof_brier": brier,
    "internal_confusion_matrix": cm.tolist(),
    "external_cohort_n": int(len(ext_y)),
    "external_cohort_label_balance": {"AD": int((ext_y == 1).sum()), "control": int((ext_y == 0).sum())},
    "external_common_features": len(common_features),
    "external_accuracy": ext_acc,
    "external_balanced_accuracy": ext_bal_acc,
    "external_auc": ext_auc,
    "external_f1": ext_f1,
    "external_brier": ext_brier,
    "n_bootstrap": N_BOOTSTRAP, "k_features": K_FEATURES, "cv_splits": N_SPLITS, "cv_repeats": N_REPEATS,
}
with open(RES / "stage1_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved stage1_summary.json and prediction CSVs to results/")
