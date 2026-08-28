"""Stage-2 methylation-only classifier: same bootstrap-ensembled, calibrated,
cross-validated protocol as Stage 1, run on the smaller (n=142) GSE80970
methylation cohort (gene-mapped features, Methy_<gene> columns)."""
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
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)

K_FEATURES = 150   # smaller k given n=142 samples, to reduce overfitting risk
N_BOOTSTRAP = 25
N_SPLITS = 5
N_REPEATS = 10  # more repeats to stabilize OOF estimates given the small n

print("Loading Stage-2 methylation cohort (Zenodo AE-Trans methylation_data.csv: GSE80970)...")
meth = pd.read_csv(RAW / "methylation_data.csv")
feature_cols = [c for c in meth.columns if c not in ("sample", "label")]
X = meth[feature_cols].values
y = meth["label"].values.astype(int)
sample_ids = meth["sample"].values
print(f"X shape: {X.shape}, y balance: {np.bincount(y)}")


def fit_bootstrap_ensemble(X_train, y_train, n_boot=N_BOOTSTRAP, k=K_FEATURES, seed=0):
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
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)


print(f"\nRunning {N_REPEATS}x{N_SPLITS}-fold CV with {N_BOOTSTRAP}-model bootstrap ensembles per fold...")
rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)

oof_prob_sum = np.zeros(len(y))
oof_count = np.zeros(len(y))

for fold_i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
    ensemble = fit_bootstrap_ensemble(X[train_idx], y[train_idx], seed=fold_i)
    prob, unc = predict_ensemble(ensemble, X[test_idx])
    oof_prob_sum[test_idx] += prob
    oof_count[test_idx] += 1
    if (fold_i + 1) % 10 == 0:
        print(f"  completed fold {fold_i + 1}/{N_SPLITS * N_REPEATS}", flush=True)

oof_prob = oof_prob_sum / oof_count
oof_pred = (oof_prob >= 0.5).astype(int)

acc = accuracy_score(y, oof_pred)
bal_acc = balanced_accuracy_score(y, oof_pred)
auc = roc_auc_score(y, oof_prob)
f1 = f1_score(y, oof_pred)
brier = brier_score_loss(y, oof_prob)
cm = confusion_matrix(y, oof_pred)

print("\n=== Stage-2 (methylation-only) out-of-fold performance (n=142) ===")
print(f"Accuracy: {acc:.4f}  Balanced accuracy: {bal_acc:.4f}  AUC: {auc:.4f}  F1: {f1:.4f}  Brier: {brier:.4f}")
print("Confusion matrix [control, AD]:\n", cm)

oof_df = pd.DataFrame({
    "sample": sample_ids, "label": y, "oof_prob": oof_prob, "oof_pred": oof_pred,
    "confidence_margin": np.abs(oof_prob - 0.5) * 2,
})
oof_df.to_csv(RES / "stage2_methylation_oof_predictions.csv", index=False)

summary = {
    "cohort_n": int(len(y)), "label_balance": {"AD": int((y == 1).sum()), "control": int((y == 0).sum())},
    "oof_accuracy": acc, "oof_balanced_accuracy": bal_acc, "oof_auc": auc, "oof_f1": f1, "oof_brier": brier,
    "confusion_matrix": cm.tolist(), "n_bootstrap": N_BOOTSTRAP, "k_features": K_FEATURES,
    "cv_splits": N_SPLITS, "cv_repeats": N_REPEATS,
}
with open(RES / "stage2_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved stage2_summary.json and OOF predictions.")
