"""
k_sweep_uncertainty.py

Extends k_sweep.py with:
  - 95% percentile intervals across partner-draw seeds for accuracy at each k
  - explicit comparison of the k-sweep's own k=20 seed-mean against the
    single-seed headline figure reported in Table 8, so the two are not
    read as inconsistent

Usage:
    python k_sweep_uncertainty.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

K_VALUES = [1, 2, 3, 5, 10, 20, 50]
N_SEEDS = 30
MARGIN_THRESHOLD = 0.6
HEADLINE_K20 = 0.9310  # single fixed-seed value reported as the original convention's result


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fuse(l1, y1, pool, k, seed):
    rng = np.random.RandomState(seed)
    partner_avg = np.array([
        rng.choice(pool[int(yi)], size=k, replace=True).mean() for yi in y1
    ])
    X = np.column_stack([l1, partner_avg])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.zeros(len(y1), dtype=int)
    for tr, te in skf.split(X, y1):
        clf = LogisticRegression(max_iter=1000).fit(X[tr], y1[tr])
        pred[te] = clf.predict(X[te])
    return pred


if __name__ == "__main__":
    s1 = pd.read_csv("results/stage1_mrna_oof_predictions.csv")
    s2 = pd.read_csv("results/stage2_methylation_oof_predictions.csv")
    y1 = s1["label"].to_numpy().astype(int)
    y2 = s2["label"].to_numpy().astype(int)
    l1 = logit(s1["oof_prob"].to_numpy())
    l2 = logit(s2["oof_prob"].to_numpy())
    margin = s1["confidence_margin"].to_numpy()
    mask_low = margin < MARGIN_THRESHOLD
    pool = {1: l2[y2 == 1], 0: l2[y2 == 0]}

    print("k-sweep with seed-level uncertainty")
    print("=" * 78)
    print(f"{'k':>4}{'mean acc':>11}{'95% CI lo':>12}{'95% CI hi':>12}{'n_seeds':>9}")
    print("-" * 78)

    rows = []
    all_k_all_seed_accs = []  # for the trend test: (log k, acc) pairs
    for k in K_VALUES:
        accs = []
        for seed in range(N_SEEDS):
            pred = fuse(l1, y1, pool, k, seed)
            a = accuracy_score(y1[mask_low], pred[mask_low])
            accs.append(a)
            all_k_all_seed_accs.append((np.log(k), a))
        accs = np.array(accs)
        lo, hi = np.percentile(accs, [2.5, 97.5])
        print(f"{k:>4}{accs.mean()*100:>10.2f}%{lo*100:>11.2f}%{hi*100:>11.2f}%{N_SEEDS:>9}")
        rows.append({"k": k, "mean_acc_pct": round(accs.mean()*100, 2),
                     "ci_lo_pct": round(lo*100, 2), "ci_hi_pct": round(hi*100, 2),
                     "n_seeds": N_SEEDS})

    logk, acc = zip(*all_k_all_seed_accs)

    k20_seed_mean = rows[K_VALUES.index(20)]["mean_acc_pct"] / 100
    pct_rank = (np.array([a for lk, a in all_k_all_seed_accs if np.isclose(np.exp(lk), 20)]) < HEADLINE_K20).mean() * 100
    print(f"\nk=20 seed-mean (this sweep): {k20_seed_mean*100:.2f}%")
    print(f"Table 8 headline k=20 value (single fixed seed): {HEADLINE_K20*100:.2f}%")
    print(f"Headline value sits at the {pct_rank:.0f}th percentile of the 30-seed k=20 distribution here")

    pd.DataFrame(rows).to_csv("results/k_sweep_uncertainty.csv", index=False)
    print("\nWrote results/k_sweep_uncertainty.csv")
