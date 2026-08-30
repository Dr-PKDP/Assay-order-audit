"""
recalibrated_bound.py

Addresses the objection that assumption A1 (calibrated Stage-1 posteriors) is
not satisfied empirically -- the Stage-1 internal calibration slope is 1.241
with a 95% CI of [1.033, 1.652], which excludes 1. Because
ehat = mean min(p, 1-p) is the posterior Bayes error only when p is a genuine
posterior probability, the bound's empirical application depends on how far
from calibrated the Stage-1 probabilities actually are.

Method: cross-fitted (nested) recalibration. The 697 Stage-1 out-of-fold
probabilities are split into 5 stratified folds; for each fold a calibrator
(Platt scaling, i.e. logistic regression on the logit, or isotonic regression)
is fit on the other four folds and applied to the held-out fold. This yields
genuinely held-out calibrated probabilities with no sample calibrated by a
model that saw it. ehat and the attainability bound are then recomputed from
those probabilities.

Confidence-margin subsets are held at their ORIGINAL definitions so that the
recomputed bound is compared against the same patients (and therefore the same
reported accuracies) as in Table 8. Subset membership under recalibrated
margins is reported separately as a diagnostic.

Direction of the effect is predictable from the bound's monotonicity:
B(ehat) = 1 - ehat*(1 - 2V) is strictly decreasing in ehat, so if
recalibration sharpens the probabilities (slope toward 1 from above), ehat
falls and the bound becomes MORE permissive. The question this script answers
is whether the reported same-label accuracies still exceed it.

Usage:
    python recalibrated_bound.py
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

N_SPLITS = 5
SEED = 42
N_BOOT = 2000
GH_NODES = 200


def auc_to_d(auc):
    return np.sqrt(2) * norm.ppf(auc)


def v_half(d):
    """Value of information at p=1/2 under the equal-variance binormal model,
    by Gauss-Hermite quadrature (same construction as
    bayes_error_corrected_bound.py)."""
    x, w = np.polynomial.hermite.hermgauss(GH_NODES)
    # s ~ 0.5*N(d,1) + 0.5*N(0,1); posterior at p=1/2 is f1/(f1+f0)
    # E_s[min(p', 1-p')] = 0.5*E_{s~N(d,1)}[.] + 0.5*E_{s~N(0,1)}[.]
    total = 0.0
    for mean in (d, 0.0):
        s = mean + np.sqrt(2) * x
        f1 = norm.pdf(s, loc=d, scale=1)
        f0 = norm.pdf(s, loc=0.0, scale=1)
        post = f1 / (f1 + f0)
        total += 0.5 * np.sum(w * np.minimum(post, 1 - post)) / np.sqrt(np.pi)
    return 0.5 - total


def bound_from_ehat(ehat, v):
    return 1.0 - ehat * (1.0 - 2.0 * v)


def calib_slope_intercept(y, p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    lg = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lg, y)
    slope = float(lr.coef_[0][0])
    # calibration-in-the-large: intercept with slope fixed at 1
    off = np.log(p / (1 - p))
    lr0 = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    X0 = np.ones((len(y), 1))
    # fit intercept-only on the offset-corrected outcome via simple search
    from scipy.optimize import minimize_scalar

    def nll(a):
        z = off + a
        q = 1 / (1 + np.exp(-z))
        q = np.clip(q, eps, 1 - eps)
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))

    a = minimize_scalar(nll, bounds=(-5, 5), method="bounded").x
    return slope, float(a)


def crossfit_calibrate(y, p, method, seed=SEED):
    """Cross-fitted calibration: no sample is calibrated by a model that saw it."""
    out = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    eps = 1e-6
    for tr, te in skf.split(p.reshape(-1, 1), y):
        if method == "platt":
            lg_tr = np.log(np.clip(p[tr], eps, 1 - eps) / (1 - np.clip(p[tr], eps, 1 - eps)))
            lg_te = np.log(np.clip(p[te], eps, 1 - eps) / (1 - np.clip(p[te], eps, 1 - eps)))
            m = LogisticRegression(solver="lbfgs", max_iter=1000).fit(lg_tr.reshape(-1, 1), y[tr])
            out[te] = m.predict_proba(lg_te.reshape(-1, 1))[:, 1]
        elif method == "isotonic":
            m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            m.fit(p[tr], y[tr])
            out[te] = m.predict(p[te])
        else:
            raise ValueError(method)
    return out


if __name__ == "__main__":
    s1 = pd.read_csv("results/stage1_mrna_oof_predictions.csv")
    s2 = pd.read_csv("results/stage2_methylation_oof_predictions.csv")
    y = s1["label"].to_numpy().astype(int)
    p_raw = s1["oof_prob"].to_numpy()
    margin = s1["confidence_margin"].to_numpy()

    auc2 = roc_auc_score(s2["label"].to_numpy().astype(int), s2["oof_prob"].to_numpy())
    d2 = auc_to_d(auc2)
    v = v_half(d2)
    print(f"Stage-2 AUC (full precision): {auc2:.6f}")
    print(f"d = {d2:.6f}   V(1/2; d) = {v:.6f}")
    print()

    variants = {"raw (as reported)": p_raw}
    for m in ("platt", "isotonic"):
        variants[f"cross-fitted {m}"] = crossfit_calibrate(y, p_raw, m)

    print("=== Calibration of Stage-1 probabilities, before and after ===")
    print(f"{'variant':<24}{'slope':>9}{'intercept':>11}{'AUC':>9}{'acc':>9}")
    for name, p in variants.items():
        sl, ic = calib_slope_intercept(y, p)
        acc = ((p >= 0.5).astype(int) == y).mean()
        print(f"{name:<24}{sl:>9.3f}{ic:>11.3f}{roc_auc_score(y, p):>9.4f}{acc*100:>8.2f}%")
    print()

    SUBSETS = {
        "Margin < 0.2": margin < 0.2,
        "Margin < 0.4": margin < 0.4,
        "Margin < 0.6": margin < 0.6,
        "Whole cohort": np.ones(len(y), dtype=bool),
    }
    REPORTED = {"Margin < 0.2": 95.83, "Margin < 0.4": 95.12,
                "Margin < 0.6": 93.10, "Whole cohort": 98.57}

    print("=== ehat and attainability bound, original subsets, per calibration variant ===")
    rows = []
    for name, p in variants.items():
        ehat_col = np.minimum(p, 1 - p)
        print(f"\n-- {name} --")
        print(f"{'subset':<15}{'n':>5}{'ehat':>9}{'bound':>9}{'CI upper':>10}{'reported':>10}{'verdict':>12}")
        for sname, mask in SUBSETS.items():
            n = int(mask.sum())
            ehat = float(ehat_col[mask].mean())
            b = bound_from_ehat(ehat, v)
            # bootstrap CI: resample patients within subset and Stage-2 AUC independently
            rng = np.random.RandomState(SEED)
            idx_pool = np.where(mask)[0]
            y2 = s2["label"].to_numpy().astype(int)
            p2 = s2["oof_prob"].to_numpy()
            boots = np.zeros(N_BOOT)
            for b_i in range(N_BOOT):
                bi = rng.choice(idx_pool, size=n, replace=True)
                e_b = float(ehat_col[bi].mean())
                j = rng.choice(len(y2), size=len(y2), replace=True)
                while len(np.unique(y2[j])) < 2:
                    j = rng.choice(len(y2), size=len(y2), replace=True)
                d_b = auc_to_d(roc_auc_score(y2[j], p2[j]))
                boots[b_i] = bound_from_ehat(e_b, v_half(d_b))
            ci_hi = float(np.percentile(boots, 97.5))
            rep = REPORTED[sname]
            verdict = "exceeds" if rep > ci_hi * 100 else "WITHIN"
            print(f"{sname:<15}{n:>5}{ehat*100:>8.2f}%{b*100:>8.2f}%{ci_hi*100:>9.2f}%{rep:>9.2f}%{verdict:>12}")
            rows.append({"variant": name, "subset": sname, "n": n,
                         "ehat_pct": round(ehat * 100, 2),
                         "bound_pct": round(b * 100, 2),
                         "bound_ci_upper_pct": round(ci_hi * 100, 2),
                         "reported_pct": rep,
                         "exceeds_ci_upper": bool(rep > ci_hi * 100)})

    pd.DataFrame(rows).to_csv("results/recalibrated_bound.csv", index=False)
    print("\nWrote results/recalibrated_bound.csv")

    # diagnostic: how much does subset membership shift under recalibrated margins?
    print("\n=== Subset membership shift under recalibrated margins (diagnostic) ===")
    for name, p in variants.items():
        if name == "raw (as reported)":
            continue
        m_new = np.abs(2 * p - 1)
        for tau in (0.2, 0.4, 0.6):
            old_n = int((margin < tau).sum())
            new_n = int((m_new < tau).sum())
            overlap = int(((margin < tau) & (m_new < tau)).sum())
            print(f"{name:<24} tau<{tau}: original n={old_n:>3}, recalibrated n={new_n:>3}, overlap={overlap:>3}")
