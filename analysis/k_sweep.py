"""
k_sweep.py

Empirical partner-count sweep. Runs the same-label fusion construction at
k = 1, 2, 3, 5, 10, 20, 50 and reports observed fusion accuracy and AUC
against the sqrt(k) prediction from the pairing-arithmetic identity.

The identity predicts that averaging k independent class-conditional partner
logits scales the synthetic feature's separation by sqrt(k), so its effective
AUC is Phi(sqrt(k) * d / sqrt(2)). This script asks whether the *observed*
fusion trajectory tracks that predicted trajectory, which the effective-AUC
table alone cannot establish.

Each k is averaged over N_SEEDS independent partner draws so the reported
point is not a single arbitrary draw.

Usage:
    python k_sweep.py
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

K_VALUES = [1, 2, 3, 5, 10, 20, 50]
N_SEEDS = 30
MARGIN_THRESHOLD = 0.6


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def auc_to_d(auc):
    return norm.ppf(auc) * np.sqrt(2.0)


def d_to_auc(d):
    return norm.cdf(d / np.sqrt(2.0))


def fuse(l1, y1, pool, k, seed):
    rng = np.random.RandomState(seed)
    partner_avg = np.array([
        rng.choice(pool[int(yi)], size=k, replace=True).mean() for yi in y1
    ])
    X = np.column_stack([l1, partner_avg])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.zeros(len(y1), dtype=int)
    prob = np.zeros(len(y1), dtype=float)
    for tr, te in skf.split(X, y1):
        clf = LogisticRegression(max_iter=1000).fit(X[tr], y1[tr])
        pred[te] = clf.predict(X[te])
        prob[te] = clf.predict_proba(X[te])[:, 1]
    return pred, prob


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

    auc2 = roc_auc_score(y2, s2["oof_prob"].to_numpy())
    d2 = auc_to_d(auc2)

    base_whole = accuracy_score(y1, s1["oof_pred"].to_numpy())
    base_low = accuracy_score(y1[mask_low], s1["oof_pred"].to_numpy()[mask_low])

    print(f"Stage-2 standalone AUC = {auc2:.4f}  (d = {d2:.4f})")
    print(f"mRNA-only baseline: whole = {base_whole*100:.2f}%, "
          f"margin<{MARGIN_THRESHOLD} = {base_low*100:.2f}%")
    print()
    print("Empirical k-sweep vs sqrt(k) prediction "
          f"(mean over {N_SEEDS} partner-draw seeds)")
    print("=" * 88)
    print(f"{'k':>4}{'pred eff AUC':>14}{'obs fusion AUC':>16}"
          f"{'acc whole':>12}{'acc low-conf':>14}{'sd low-conf':>13}")
    print("-" * 88)

    rows = []
    for k in K_VALUES:
        pred_auc = d_to_auc(np.sqrt(k) * d2)
        accs_w, accs_l, aucs = [], [], []
        for seed in range(N_SEEDS):
            pred, prob = fuse(l1, y1, pool, k, seed)
            accs_w.append(accuracy_score(y1, pred))
            accs_l.append(accuracy_score(y1[mask_low], pred[mask_low]))
            aucs.append(roc_auc_score(y1, prob))
        aw, al, au = np.mean(accs_w), np.mean(accs_l), np.mean(aucs)
        sl = np.std(accs_l)
        print(f"{k:>4}{pred_auc:>14.4f}{au:>16.4f}"
              f"{aw*100:>11.2f}{al*100:>13.2f}{sl*100:>12.2f}")
        rows.append({"k": k,
                     "predicted_effective_auc": round(pred_auc, 4),
                     "observed_fusion_auc_mean": round(au, 4),
                     "acc_whole_mean_pct": round(aw * 100, 2),
                     "acc_lowconf_mean_pct": round(al * 100, 2),
                     "acc_lowconf_sd_pct": round(sl * 100, 2),
                     "n_seeds": N_SEEDS})

    pd.DataFrame(rows).to_csv("results/k_sweep.csv", index=False)
    print("\nWrote results/k_sweep.csv")
