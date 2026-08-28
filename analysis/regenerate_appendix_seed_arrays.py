"""
regenerate_appendix_seed_arrays.py

Regenerates appendix_seed_accs_subset.npy and appendix_delong_ps.npy, which
back Figure A1, using the learned 2-feature meta-model (fuse_probs, copied
verbatim from seed_aware_inference.py) instead of the simple unweighted
logit averaging these arrays previously came from (via k1_honest_estimate.py
/ k1_significance_test.py). See k1_learned_model_all_thresholds.py for the
full explanation of why the learned-model version is the internally
consistent one.

Panel A: per-seed accuracy, margin<0.6 subset, all 200 partner-draw seeds.
Panel B: paired DeLong p-value (baseline vs k=1), margin<0.6 subset, the
first 30 partner-draw seeds -- matching the original figure's own seed
count for this specific panel (a pre-existing design choice, unrelated to
the method correction).

Usage:
    python regenerate_appendix_seed_arrays.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import norm


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fuse_probs(l1, y1, pool, k, seed):
    """Verbatim from seed_aware_inference.py."""
    rng = np.random.RandomState(seed)
    partner = np.array([rng.choice(pool[int(yi)], size=k, replace=True).mean()
                        for yi in y1])
    X = np.column_stack([l1, partner])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    prob = np.zeros(len(y1))
    for tr, te in skf.split(X, y1):
        clf = LogisticRegression(max_iter=1000).fit(X[tr], y1[tr])
        prob[te] = clf.predict_proba(X[te])[:, 1]
    return prob


def paired_delong(y_true, s_a, s_b):
    """Verbatim from k1_significance_test.py / leave_one_series_out.py."""
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


if __name__ == "__main__":
    s1 = pd.read_csv("results/stage1_mrna_oof_predictions.csv")
    s2 = pd.read_csv("results/stage2_methylation_oof_predictions.csv")
    y1 = s1["label"].to_numpy().astype(int)
    y2 = s2["label"].to_numpy().astype(int)
    p1 = s1["oof_prob"].to_numpy()
    l1 = logit(p1)
    l2 = logit(s2["oof_prob"].to_numpy())
    margin = s1["confidence_margin"].to_numpy()
    pool = {1: l2[y2 == 1], 0: l2[y2 == 0]}
    mask = margin < 0.6

    print("Panel A: 200-seed accuracy distribution, margin<0.6 (learned meta-model)")
    accs_subset = np.zeros(200)
    for seed in range(200):
        prob = fuse_probs(l1, y1, pool, 1, seed)
        pred = (prob > 0.5).astype(int)
        accs_subset[seed] = accuracy_score(y1[mask], pred[mask])
        if (seed + 1) % 50 == 0:
            print(f"  seed {seed+1}/200")
    print(f"  mean={accs_subset.mean()*100:.2f}%  sd={accs_subset.std()*100:.2f}")
    np.save("appendix_seed_accs_subset.npy", accs_subset)

    print("\nPanel B: 30-seed paired DeLong p-values, margin<0.6 (learned meta-model)")
    delong_ps = np.zeros(30)
    for seed in range(30):
        prob = fuse_probs(l1, y1, pool, 1, seed)
        _, _, _, p = paired_delong(y1[mask].astype(float), p1[mask], prob[mask])
        delong_ps[seed] = p
    print(f"  mean p={delong_ps.mean():.3f}  min={delong_ps.min():.3f}  max={delong_ps.max():.3f}")
    np.save("appendix_delong_ps.npy", delong_ps)

    print("\nSaved appendix_seed_accs_subset.npy and appendix_delong_ps.npy")
