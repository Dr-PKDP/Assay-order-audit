"""
score_fsqn_vs_baseline.py

Reuses fit_bootstrap_ensemble/predict_ensemble VERBATIM from
scripts/05_stage1_mrna_model.py, then scores each external cohort two ways:

  BASELINE (self-standardization, the manuscript's "weakest option available"):
      external cohort z-scored against its OWN mean/std, then fed straight to
      each ensemble member's selector + classifier, bypassing that member's
      own fitted StandardScaler. This is exactly what the original external-
      validation code in 05_stage1_mrna_model.py does.

  FSQN: external cohort already harmonized onto the TRAINING cohort's raw
      per-gene distribution (via fsqn_harmonization.py), so instead of self-
      standardizing, each ensemble member's OWN fitted scaler is applied
      (transform, not fit_transform) -- the scaler now sees data on a
      comparable scale to what it was fit on, which is the whole point of
      doing FSQN first.

IMPORTANT CAVEAT: this study's original external-cohort processing used the
old NCBI .annot.gz probe-to-gene format for both GPL570 and GPL10558. That
exact file is no longer available (see the conversation history -- Thermo
Fisher and Illumina both discontinued the direct-download pages), so this
pipeline uses BioMart (GPL570) and Bioconductor's illuminaHumanv4.db
(GPL10558) instead. The resulting gene sets and probe coverage differ
somewhat from what produced the manuscript's already-published external
accuracy figures, so BASELINE here will not exactly reproduce those
published numbers. What IS a fair, apples-to-apples comparison is BASELINE
vs FSQN computed under this one consistent pipeline -- that is what this
script is actually for.

Usage:
    python score_fsqn_vs_baseline.py \
        --gene-data data/raw/gene_data.csv \
        --external-baseline data/processed/GSE118553_gene_level.csv \
        --external-fsqn data/processed/GSE118553_gene_level_fsqn.csv \
        --external-labels data/processed/GSE118553_frontal_ADcontrol_probes_labels.csv \
        --cohort-name GSE118553
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


def paired_delong(y_true, s_a, s_b):
    """Paired DeLong test comparing AUC of s_a vs s_b on the same samples.
    Verified implementation, reused as-is from k1_significance_test.py."""
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
    v01s = np.array(v01s)
    v10s = np.array(v10s)
    s01 = np.cov(v01s)
    s10 = np.cov(v10s)
    cov = s01 / n1 + s10 / n0
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z)))
    return aucs[0], aucs[1], z, p


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


def score_baseline(ensemble, X_ext_raw):
    """Self-standardization: bypass each member's own scaler."""
    X_ext_scaled_self = StandardScaler().fit_transform(X_ext_raw)
    preds = []
    for scaler, selector, clf in ensemble:
        Xt = selector.transform(X_ext_scaled_self)
        preds.append(clf.predict_proba(Xt)[:, 1])
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)


def score_fsqn(ensemble, X_ext_fsqn):
    """FSQN: apply each member's own fitted scaler, since the data is
    already harmonized onto the reference cohort's raw scale."""
    preds = []
    for scaler, selector, clf in ensemble:
        Xt = selector.transform(scaler.transform(X_ext_fsqn))
        preds.append(clf.predict_proba(Xt)[:, 1])
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)


def evaluate(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    return dict(
        accuracy=accuracy_score(y_true, pred),
        balanced_accuracy=balanced_accuracy_score(y_true, pred),
        auc=roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else float("nan"),
        f1=f1_score(y_true, pred),
        brier=brier_score_loss(y_true, prob),
    )


def gating_check(y_true, prob):
    """Does the confidence margin retain any association with correctness?
    (the direct test the manuscript's Limitations flags as missing)"""
    pred = (prob >= 0.5).astype(int)
    correct = (pred == y_true).astype(int)
    margin = 2 * np.abs(prob - 0.5)
    if len(np.unique(correct)) < 2:
        return float("nan")
    return roc_auc_score(correct, margin)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gene-data", required=True, help="primary training cohort CSV "
                                                       "(gene_data.csv, samples x genes)")
    p.add_argument("--external-baseline", required=True,
                  help="un-harmonized gene-level CSV (genes x samples)")
    p.add_argument("--external-fsqn", required=True,
                  help="FSQN-harmonized gene-level CSV (genes x samples)")
    p.add_argument("--external-labels", required=True,
                  help="labels CSV with 'sample','label' columns")
    p.add_argument("--external-combat", default=None,
                  help="optional ComBat-harmonized gene-level CSV (genes x samples), "
                       "for a three-way baseline/FSQN/ComBat comparison")
    p.add_argument("--cohort-name", default="external cohort")
    p.add_argument("--save-predictions", default=None,
                  help="optional path to save per-sample predictions (sample, label, "
                       "baseline_prob, fsqn_prob, combat_prob) as a CSV -- needed for "
                       "ROC curves or confusion matrices beyond the summary metrics "
                       "already printed above")
    args = p.parse_args()

    print(f"=== {args.cohort_name} ===\n")

    print("loading primary training cohort ...")
    gene = pd.read_csv(args.gene_data)
    feature_cols = [c for c in gene.columns if c not in ("sample", "label")]
    y_train = gene["label"].values.astype(int)
    print(f"  {gene.shape[0]} samples x {len(feature_cols)} genes")

    print("loading external cohort (baseline, un-harmonized) ...")
    ext_base = pd.read_csv(args.external_baseline, index_col=0).T  # -> samples x genes
    print("loading external cohort (FSQN-harmonized) ...")
    ext_fsqn = pd.read_csv(args.external_fsqn, index_col=0).T
    labels = pd.read_csv(args.external_labels).set_index("sample")["label"]

    common_samples = [s for s in ext_base.index if s in labels.index]
    ext_base = ext_base.loc[common_samples]
    ext_fsqn = ext_fsqn.loc[[s for s in common_samples if s in ext_fsqn.index]]
    y_ext = labels.loc[ext_base.index].values.astype(int)

    common_features = [c for c in feature_cols if c in ext_base.columns]
    common_features_fsqn = [c for c in common_features if c in ext_fsqn.columns]
    print(f"  common features (baseline): {len(common_features)} / {len(feature_cols)}")
    print(f"  common features (fsqn, after FSQN's own gene drop): {len(common_features_fsqn)}")

    # use the smaller common set so both paths score the exact same feature space,
    # which is required for a fair baseline-vs-FSQN comparison
    use_features = [c for c in common_features if c in common_features_fsqn]
    print(f"  final shared feature set used for both scorings: {len(use_features)}\n")

    X_train_common = gene[use_features].values
    X_ext_base = ext_base[use_features].values
    X_ext_fsqn = ext_fsqn.loc[ext_base.index, use_features].values

    print(f"fitting {N_BOOTSTRAP}-model bootstrap ensemble on {len(use_features)} "
         f"common genes (seed=998, matching the original script) ...")
    ensemble = fit_bootstrap_ensemble(X_train_common, y_train, seed=998)

    base_prob, base_unc = score_baseline(ensemble, X_ext_base)
    fsqn_prob, fsqn_unc = score_fsqn(ensemble, X_ext_fsqn)

    base_metrics = evaluate(y_ext, base_prob)
    fsqn_metrics = evaluate(y_ext, fsqn_prob)
    base_gating = gating_check(y_ext, base_prob)
    fsqn_gating = gating_check(y_ext, fsqn_prob)

    methods = [("baseline (self-std)", base_metrics, base_gating, base_prob)]
    methods.append(("FSQN", fsqn_metrics, fsqn_gating, fsqn_prob))

    combat_prob = None
    if args.external_combat:
        print("loading external cohort (ComBat-harmonized) ...")
        ext_combat = pd.read_csv(args.external_combat, index_col=0).T
        common_features_combat = [c for c in use_features if c in ext_combat.columns]
        if len(common_features_combat) < len(use_features):
            print(f"  note: ComBat file covers {len(common_features_combat)} / "
                 f"{len(use_features)} of the shared feature set -- scoring on the "
                 f"intersection only for this method")
        X_ext_combat = ext_combat.loc[ext_base.index, common_features_combat].values
        # ComBat, like FSQN, outputs data on the reference's raw scale, so it uses
        # the same scoring path (each ensemble member's own fitted scaler applied).
        # Re-fit the ensemble on the combat-specific feature subset only if it
        # differs from use_features; otherwise reuse the already-fit ensemble.
        if len(common_features_combat) == len(use_features):
            combat_prob, combat_unc = score_fsqn(ensemble, X_ext_combat)
        else:
            X_train_combat = gene[common_features_combat].values
            ensemble_combat = fit_bootstrap_ensemble(X_train_combat, y_train, seed=998)
            combat_prob, combat_unc = score_fsqn(ensemble_combat, X_ext_combat)
        combat_metrics = evaluate(y_ext, combat_prob)
        combat_gating = gating_check(y_ext, combat_prob)
        methods.append(("ComBat", combat_metrics, combat_gating, combat_prob))

    header = f"{'metric':<20}" + "".join(f"{name:>20}" for name, *_ in methods)
    print(f"\n{header}")
    for k in base_metrics:
        row = f"{k:<20}" + "".join(f"{m[k]:>20.4f}" for _, m, _, _ in methods)
        print(row)
    print(f"\n{'gating AUC (margin predicts correctness)':<41}" +
         "".join(f"{g:>20.4f}" for _, _, g, _ in methods))
    print("  (0.5 = no association; this is the direct test of whether confidence-gated")
    print("   ordering would work on this cohort, per Section 6.1/Limitations)")

    auc_a, auc_b, z, dl_p = paired_delong(y_ext.astype(float), base_prob, fsqn_prob)
    print(f"\nPaired DeLong test, baseline AUC vs FSQN AUC (same {len(y_ext)} samples):")
    print(f"  baseline AUC={auc_a:.4f}  FSQN AUC={auc_b:.4f}  z={z:.3f}  p={dl_p:.4f}")
    if dl_p > 0.05:
        print("  not significant -- the AUC difference is within what sampling noise")
        print("  at this sample size would produce; treat baseline and FSQN as")
        print("  statistically indistinguishable on this cohort, not 'FSQN is worse/better'.")
    else:
        print("  significant at alpha=0.05.")

    if combat_prob is not None:
        auc_a2, auc_c, z2, dl_p2 = paired_delong(y_ext.astype(float), base_prob, combat_prob)
        print(f"\nPaired DeLong test, baseline AUC vs ComBat AUC (same {len(y_ext)} samples):")
        print(f"  baseline AUC={auc_a2:.4f}  ComBat AUC={auc_c:.4f}  z={z2:.3f}  p={dl_p2:.4f}")
        if dl_p2 > 0.05:
            print("  not significant -- same interpretation as above, for ComBat vs baseline.")
        else:
            print("  significant at alpha=0.05.")

    if args.save_predictions:
        pred_df = pd.DataFrame({
            "sample": ext_base.index,
            "label": y_ext,
            "baseline_prob": base_prob,
            "fsqn_prob": fsqn_prob,
        })
        if combat_prob is not None:
            pred_df["combat_prob"] = combat_prob
        import os
        out_dir = os.path.dirname(args.save_predictions)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pred_df.to_csv(args.save_predictions, index=False)
        print(f"\nPer-sample predictions saved to {args.save_predictions} "
             f"({len(pred_df)} samples, {pred_df.shape[1]-2} conditions)")
