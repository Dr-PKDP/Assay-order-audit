"""
bayes_error_corrected_bound.py

Recomputes the attainability bound using a directly estimated mean posterior
Bayes error, ehat = mean(min(p_i, 1 - p_i)), in place of (1 - observed
accuracy).

Rationale: observed accuracy of a thresholded classifier is not the same
quantity as Bayes accuracy under the model's own posterior. The two coincide
only if the model emits the true posterior and thresholds it Bayes-optimally.
Using (1 - accuracy) as "mean pre-assay Bayes error" therefore conflates two
different objects. This script reports both, so the sensitivity of the
bound comparison to that choice is explicit rather than assumed.

Both bootstrap resamples propagate uncertainty in the Stage-1 subset and the
Stage-2 AUC jointly.

Usage:
    python bayes_error_corrected_bound.py
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score, roc_auc_score

SEED = 42
N_BOOT = 2000
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(301)
GH_W = GH_W / GH_W.sum()


def auc_to_d(auc):
    return norm.ppf(auc) * np.sqrt(2.0)


def post_error(p, d):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    lo = np.log(p / (1 - p))
    out = 0.0
    for mu, w in ((d, p), (0.0, 1 - p)):
        q = 1 / (1 + np.exp(-(lo + d * (GH_X + mu) - d ** 2 / 2)))
        out += w * np.sum(GH_W * np.minimum(q, 1 - q))
    return out


def bound_from_error(pre_error, d):
    """Attainability bound given a mean pre-assay error and separation d."""
    pre_accuracy = 1.0 - pre_error
    w = min(max(pre_error / 0.5, 0.0), 1.0)
    v = 0.5 - post_error(0.5, d)
    return pre_accuracy + w * v


def bayes_error(p):
    return np.minimum(p, 1.0 - p).mean()


if __name__ == "__main__":
    s1 = pd.read_csv("results/stage1_mrna_oof_predictions.csv")
    s2 = pd.read_csv("results/stage2_methylation_oof_predictions.csv")
    y1, p1 = s1["label"].to_numpy().astype(int), s1["oof_prob"].to_numpy()
    y2, p2 = s2["label"].to_numpy().astype(int), s2["oof_prob"].to_numpy()
    margin = s1["confidence_margin"].to_numpy()
    N = len(y1)

    SUBSETS = {"whole cohort": np.ones(N, dtype=bool),
               "margin < 0.9": margin < 0.9, "margin < 0.8": margin < 0.8,
               "margin < 0.6": margin < 0.6, "margin < 0.4": margin < 0.4,
               "margin < 0.2": margin < 0.2}
    REPORTED = {"whole cohort": 0.9857, "margin < 0.9": 0.9467,
                "margin < 0.8": 0.9375, "margin < 0.6": 0.9310,
                "margin < 0.4": 0.9512, "margin < 0.2": 0.9583}

    rng = np.random.default_rng(SEED)
    idx2_all = np.arange(len(y2))
    rows = []

    print("Attainability bound: observed-error vs posterior-Bayes-error input")
    print("=" * 96)
    print(f"{'Subset':<15}{'n':>5}{'1-acc':>9}{'ehat':>9}"
          f"{'bound(acc)':>12}{'bound(ehat)':>13}{'CI-hi(ehat)':>13}"
          f"{'reported':>10}{'verdict':>18}")
    print("-" * 96)

    for name, mask in SUBSETS.items():
        idx_sub = np.where(mask)[0]
        n_sub = len(idx_sub)

        obs_err = 1.0 - accuracy_score(y1[idx_sub], p1[idx_sub] > 0.5)
        e_hat = bayes_error(p1[idx_sub])
        auc2 = roc_auc_score(y2, p2)
        d2 = auc_to_d(auc2)

        b_obs = bound_from_error(obs_err, d2)
        b_bayes = bound_from_error(e_hat, d2)

        boot = np.zeros(N_BOOT)
        for b in range(N_BOOT):
            bi = rng.choice(idx_sub, size=n_sub, replace=True)
            e_b = bayes_error(p1[bi])
            b2 = rng.choice(idx2_all, size=len(idx2_all), replace=True)
            auc_b = roc_auc_score(y2[b2], p2[b2]) if len(np.unique(y2[b2])) > 1 else 0.5
            boot[b] = bound_from_error(e_b, auc_to_d(auc_b))
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        rep = REPORTED[name]
        verdict = "exceeds CI upper" if rep > ci_hi else (
            "exceeds point est." if rep > b_bayes else "within bound")

        print(f"{name:<15}{n_sub:>5}{obs_err*100:>8.2f}{e_hat*100:>9.2f}"
              f"{b_obs*100:>11.2f}{b_bayes*100:>12.2f}{ci_hi*100:>12.2f}"
              f"{rep*100:>10.2f}{verdict:>18}")

        rows.append({"subset": name, "n": n_sub,
                     "observed_error_pp": round(obs_err * 100, 2),
                     "bayes_error_pp": round(e_hat * 100, 2),
                     "bound_from_observed_acc": round(b_obs * 100, 2),
                     "bound_from_bayes_error": round(b_bayes * 100, 2),
                     "bound_bayes_ci_lo": round(ci_lo * 100, 2),
                     "bound_bayes_ci_hi": round(ci_hi * 100, 2),
                     "reported": round(rep * 100, 2),
                     "verdict": verdict})

    pd.DataFrame(rows).to_csv("results/bayes_error_corrected_bound_table11.csv", index=False)
    print("\nWrote results/bayes_error_corrected_bound.csv")
