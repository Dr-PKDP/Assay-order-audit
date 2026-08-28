"""
leave_one_series_out.py

True leave-one-series-out test for the Stage-1 primary cohort, which is two
accessions pooled together (GSE33000, n=467, 66.4% AD; GSE44770, n=230,
56.1% AD -- different label balance, different accession, same platform).

This needs no new data: gene_data.csv already carries GSM sample IDs, and
GSM142xxx = GSE33000 / GSM109xxx = GSE44770 was already confirmed against
the manuscript's reported per-series label counts.

This is a stronger test than the partial diagnostic already in the
manuscript's Limitations, which only checks whether a model trained on BOTH
series performs homogeneously on each -- that can look clean even if the
model partially uses series identity as a shortcut, since the shortcut was
available for both series during training. Training on one series and
scoring an entirely unseen series removes that possibility.

Reuses fit_bootstrap_ensemble/predict_ensemble VERBATIM from
scripts/05_stage1_mrna_model.py.

Usage:
    python leave_one_series_out.py --gene-data data/raw/gene_data.csv
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             brier_score_loss, balanced_accuracy_score)
from sklearn.utils import resample

np.random.seed(42)
K_FEATURES = 300
N_BOOTSTRAP = 25


# ---- verbatim from scripts/05_stage1_mrna_model.py ----------------------
def fit_bootstrap_ensemble(X_train, y_train, n_boot=N_BOOTSTRAP, k=K_FEATURES, seed=0):
    ensemble = []
    rng = np.random.RandomState(seed)
    n = X_train.shape[0]
    for b in range(n_boot):
        idx = resample(np.arange(n), replace=True, n_samples=n,
                       random_state=rng.randint(0, 1_000_000), stratify=y_train)
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


def paired_delong(y_true, s_a, s_b):
    """Same verified implementation used in score_fsqn_vs_baseline.py."""
    pos = y_true == 1
    neg = y_true == 0
    n1, n0 = pos.sum(), neg.sum()

    def midrank(x):
        J = np.argsort(x)
        Z = x[J]
        T = np.zeros(len(x))
        i = 0
        while i < len(x):
            j = i
            while j < len(x) - 1 and Z[j + 1] == Z[i]:
                j += 1
            T[i:j + 1] = 0.5 * (i + j) + 1
            i = j + 1
        T2 = np.empty(len(x))
        T2[J] = T
        return T2

    def fastDeLong(scores_pos, scores_neg):
        m, n = len(scores_pos), len(scores_neg)
        tx = midrank(scores_pos)
        ty = midrank(scores_neg)
        tz = midrank(np.concatenate([scores_pos, scores_neg]))
        v01 = (tz[:m] - tx) / n
        v10 = 1.0 - (tz[m:] - ty) / m
        auc = tz[:m].sum() / m / n - (m + 1.0) / (2.0 * n)
        return auc, v01, v10

    aucs, v01s, v10s = [], [], []
    for s in (s_a, s_b):
        auc, v01, v10 = fastDeLong(s[pos], s[neg])
        aucs.append(auc)
        v01s.append(v01)
        v10s.append(v10)
    v01s, v10s = np.array(v01s), np.array(v10s)
    cov = np.cov(v01s) / n1 + np.cov(v10s) / n0
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z)))
    return aucs[0], aucs[1], z, p


def evaluate(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    return dict(
        accuracy=accuracy_score(y_true, pred),
        balanced_accuracy=balanced_accuracy_score(y_true, pred),
        auc=roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else float("nan"),
        f1=f1_score(y_true, pred),
        brier=brier_score_loss(y_true, prob),
    )


def run_direction(gene, feature_cols, train_series_name, train_mask,
                  test_series_name, test_mask, seed):
    X_train = gene.loc[train_mask, feature_cols].values
    y_train = gene.loc[train_mask, "label"].values.astype(int)
    X_test = gene.loc[test_mask, feature_cols].values
    y_test = gene.loc[test_mask, "label"].values.astype(int)

    print(f"\n--- train on {train_series_name} (n={len(y_train)}), "
         f"test on {test_series_name} (n={len(y_test)}) ---")
    print(f"  train label balance: AD={int((y_train==1).sum())} "
         f"control={int((y_train==0).sum())}")
    print(f"  test label balance:  AD={int((y_test==1).sum())} "
         f"control={int((y_test==0).sum())}")

    ensemble = fit_bootstrap_ensemble(X_train, y_train, seed=seed)
    prob, unc = predict_ensemble(ensemble, X_test)
    metrics = evaluate(y_test, prob)
    print(f"  accuracy={metrics['accuracy']:.4f}  balanced_acc={metrics['balanced_accuracy']:.4f}"
         f"  AUC={metrics['auc']:.4f}  Brier={metrics['brier']:.4f}")
    return metrics, prob, y_test


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gene-data", required=True)
    args = p.parse_args()

    print("loading primary cohort ...")
    gene = pd.read_csv(args.gene_data)
    feature_cols = [c for c in gene.columns if c not in ("sample", "label")]
    is_gse33000 = gene["sample"].str.startswith("GSM142")
    is_gse44770 = ~is_gse33000
    print(f"  GSE33000: {is_gse33000.sum()} samples   GSE44770: {is_gse44770.sum()} samples")

    m1, prob1, y1 = run_direction(gene, feature_cols, "GSE44770", is_gse44770,
                                  "GSE33000", is_gse33000, seed=101)
    m2, prob2, y2 = run_direction(gene, feature_cols, "GSE33000", is_gse33000,
                                  "GSE44770", is_gse44770, seed=102)

    print(f"\n=== Summary ===")
    print(f"{'direction':<28}{'accuracy':>10}{'AUC':>10}{'Brier':>10}")
    print(f"{'train GSE44770->test GSE33000':<28}{m1['accuracy']:>10.4f}"
         f"{m1['auc']:>10.4f}{m1['brier']:>10.4f}")
    print(f"{'train GSE33000->test GSE44770':<28}{m2['accuracy']:>10.4f}"
         f"{m2['auc']:>10.4f}{m2['brier']:>10.4f}")
    print(f"\n(compare against the pooled-training internal accuracy of 0.9656, AUC 0.993")
    print(f" already reported -- a large drop here would indicate the pooled model")
    print(f" partly relies on series identity rather than pure biology)")
