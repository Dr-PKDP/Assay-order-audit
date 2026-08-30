"""
k1_learned_model_all_thresholds.py

Regenerates Table 10 (per-threshold k=1 summary) and Table A11 (MDE/TOST)
using the SAME learned 2-feature logistic-regression meta-model as
k_sweep.py and seed_aware_inference.py -- the method used for every other
value of k in the partner-count sweep (Table 9).

Background: an earlier version of Table 10 and Table A11 was built from
k1_honest_estimate.py, which uses simple unweighted logit averaging instead
of the learned meta-model. That is a genuinely different combination rule
from the one used at every other k in Table 9 and in Section 5.5's headline
bootstrap (seed_aware_inference.py), with no stated reason for the switch.
Cross-checking against Table 9's own k=1 row (68.22%, 30 seeds, learned
model) confirmed the learned-model numbers are the internally consistent
ones: they agree with Table 9 to within normal seed-count noise (0.03pp),
while the simple-averaging numbers differed by 20x that (0.65-0.68pp),
indicating a genuine method difference rather than noise. This script
produces the corrected, consistent numbers.

fuse_probs() is copied verbatim from seed_aware_inference.py to guarantee
identical methodology to what already backs Table 9 and Section 5.5's
prose, not a reimplementation.

Usage:
    python k1_learned_model_all_thresholds.py
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

N_SEEDS = 200


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


def mcnemar_mde(n, p_disc, power=0.80, alpha=0.05):
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    if n * p_disc < 5:
        return np.nan
    se = np.sqrt(p_disc / n)
    return (z_alpha + z_beta) * se


def tost_paired(y_true, pred_a, pred_b, margin_pp):
    a_correct = (pred_a == y_true).astype(int)
    b_correct = (pred_b == y_true).astype(int)
    diff = a_correct.mean() - b_correct.mean()
    n = len(y_true)
    n_disc = int(((a_correct == 1) & (b_correct == 0)).sum()) + \
        int(((a_correct == 0) & (b_correct == 1)).sum())
    se = np.sqrt(n_disc) / n if n_disc > 0 else 1e-6
    m = margin_pp / 100.0
    p_lower = 1 - norm.cdf((diff - (-m)) / se)
    p_upper = 1 - norm.cdf((m - diff) / se)
    return diff, se, max(p_lower, p_upper)


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
    N = len(y1)
    base_pred = (p1 > 0.5).astype(int)

    print("Computing k=1 learned-meta-model probabilities, 200 seeds...")
    all_probs = np.zeros((N_SEEDS, N))
    for s in range(N_SEEDS):
        all_probs[s] = fuse_probs(l1, y1, pool, 1, s)
    pred_matrix = (all_probs > 0.5).astype(int)

    subsets6 = {"0.2": margin < 0.2, "0.4": margin < 0.4, "0.6": margin < 0.6,
                "0.8": margin < 0.8, "0.9": margin < 0.9, "whole": np.ones(N, dtype=bool)}

    print("\n=== Table 10: per-threshold k=1 summary ===")
    print(f"{'tau':>6}{'n':>6}{'mRNA-only':>11}{'k1 mean':>10}{'k1 sd':>8}{'overall':>10}")
    table10_rows = []
    for name, mask in subsets6.items():
        n_sub = mask.sum()
        pre = accuracy_score(y1[mask], base_pred[mask])
        seed_accs = np.array([accuracy_score(y1[mask], pred_matrix[j, mask]) for j in range(N_SEEDS)])
        k1_mean, k1_sd = seed_accs.mean(), seed_accs.std()
        if name != "whole":
            tau = float(name)
            below = margin < tau
            overall_per_seed = []
            for j in range(N_SEEDS):
                combined_pred = base_pred.copy()
                combined_pred[below] = pred_matrix[j, below]
                overall_per_seed.append(accuracy_score(y1, combined_pred))
            overall = np.mean(overall_per_seed)
        else:
            overall = k1_mean
        print(f"{name:>6}{n_sub:>6}{pre*100:>10.2f}%{k1_mean*100:>9.2f}%{k1_sd*100:>7.2f}{overall*100:>9.2f}%")
        table10_rows.append({"tau": name, "n": int(n_sub), "mrna_only_pct": round(pre*100, 2),
                              "k1_mean_pct": round(k1_mean*100, 2), "k1_sd_pct": round(k1_sd*100, 2),
                              "overall_pct": round(overall*100, 2)})
    pd.DataFrame(table10_rows).to_csv("results/table10_learned_model.csv", index=False)

    subsets4 = {"whole cohort": np.ones(N, dtype=bool), "margin < 0.6": margin < 0.6,
                "margin < 0.4": margin < 0.4, "margin < 0.2": margin < 0.2}

    print("\n=== Table A11: MDE / TOST ===")
    disc_rates, mdes = {}, {}
    for name, mask in subsets4.items():
        bs, cs = [], []
        for j in range(N_SEEDS):
            fused_pred = pred_matrix[j]
            yb, pb, fb = y1[mask], base_pred[mask], fused_pred[mask]
            bs.append(int(((pb == yb) & (fb != yb)).sum()))
            cs.append(int(((pb != yb) & (fb == yb)).sum()))
        n_sub = mask.sum()
        disc_rates[name] = (np.mean(bs) + np.mean(cs)) / n_sub
        mdes[name] = mcnemar_mde(n_sub, disc_rates[name])

    a11_rows = []
    for name, mask in subsets4.items():
        n_sub = mask.sum()
        yb = y1[mask]
        result = {"subset": name, "n": int(n_sub), "mde_pp": round(mdes[name]*100, 2)}
        for margin_pp in [2.0, 5.0]:
            diffs, ps = [], []
            for j in range(N_SEEDS):
                fused_pred = pred_matrix[j][mask]
                diff, se, p_tost = tost_paired(yb, base_pred[mask], fused_pred, margin_pp)
                diffs.append(diff); ps.append(p_tost)
            mean_p = np.mean(ps)
            print(f"{name:<14} margin=±{margin_pp}pp  p={mean_p:.3f}  "
                  f"{'equivalent' if mean_p < 0.05 else 'not established'}")
            result[f"tost_{int(margin_pp)}pp_diff_pp"] = round(np.mean(diffs)*100, 2)
            result[f"tost_{int(margin_pp)}pp_p"] = round(mean_p, 3)
            result[f"tost_{int(margin_pp)}pp_established"] = bool(mean_p < 0.05)
        a11_rows.append(result)
    pd.DataFrame(a11_rows).to_csv("results/table_a11_learned_model.csv", index=False)
    print("\nWrote results/table10_learned_model.csv and results/table_a11_learned_model.csv")
