"""
sqrtk_simulation_and_controls.py

Three additions requested by review, all separating the pairing-arithmetic
MECHANISM from the AD pipeline that illustrates it.

PART A -- Controlled simulation of the sqrt(k) identity.
   Synthetic scores S|Y=0 ~ N(0,1), S|Y=1 ~ N(d,1) for a grid of standalone
   AUCs. For each partner count k, same-class partners are drawn and averaged
   exactly as the convention does, and the empirical within-class variance,
   standardized separation and AUC of the averaged partner feature are compared
   against the theoretical sigma^2/k, sqrt(k)*d and Phi(sqrt(k)*d/sqrt(2)).
   Run both with an infinite population and with a finite same-class pool
   matching the real Stage-2 cohort (74 AD / 68 control), which is where the
   identity is expected to break down.

PART B -- Empirical variance-vs-k check on the real pipeline.
   The same three quantities measured on the actual Stage-2 out-of-fold logit
   pool, testing the mechanism directly rather than only observing that
   accuracy rises with k.

PART C -- Opposite-label and label-permuted pairing controls.
   Random pairing only asks "does an unrelated partner help?". Drawing the
   partner from the OPPOSITE class asks the sharper question: is the effect
   driven by outcome conditioning specifically? If same-label pairing helps and
   opposite-label pairing systematically hurts, with random pairing in between,
   the conditioning mechanism is unambiguous.

Usage:
    python sqrtk_simulation_and_controls.py
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

K_VALUES = [1, 2, 3, 5, 10, 20, 50, 100]
AUC_GRID = [0.55, 0.60, 0.646, 0.70, 0.80, 0.90]
N_SIM = 4000
N_REP = 200
SEED = 42


def auc_to_d(auc):
    return np.sqrt(2) * norm.ppf(auc)


def theo_auc(k, d):
    return norm.cdf(np.sqrt(k) * d / np.sqrt(2))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def simulate(d, k, n, rng, pool_sizes=None):
    """Return averaged same-class partner feature and labels.
    pool_sizes=None -> infinite population; else (n_pos, n_neg) finite pool."""
    y = rng.binomial(1, 0.5, size=n)
    if pool_sizes is None:
        part = np.where(
            y[:, None] == 1,
            rng.normal(d, 1.0, size=(n, k)),
            rng.normal(0.0, 1.0, size=(n, k)),
        ).mean(axis=1)
    else:
        n_pos, n_neg = pool_sizes
        pool_pos = rng.normal(d, 1.0, size=n_pos)
        pool_neg = rng.normal(0.0, 1.0, size=n_neg)
        part = np.empty(n)
        for i in range(n):
            pool = pool_pos if y[i] == 1 else pool_neg
            part[i] = rng.choice(pool, size=k, replace=True).mean()
    return y, part


def measure(y, feat):
    v1 = feat[y == 1].var(ddof=1)
    v0 = feat[y == 0].var(ddof=1)
    pooled = np.sqrt((v1 + v0) / 2)
    sep = (feat[y == 1].mean() - feat[y == 0].mean()) / pooled if pooled > 0 else np.nan
    return pooled ** 2, sep, roc_auc_score(y, feat)


if __name__ == "__main__":
    rng = np.random.RandomState(SEED)

    # ---------------- PART A ----------------
    print("=" * 78)
    print("PART A -- controlled simulation of the sqrt(k) identity")
    print("=" * 78)
    rowsA = []
    for auc0 in AUC_GRID:
        d = auc_to_d(auc0)
        print(f"\nstandalone AUC = {auc0:.3f}  (d = {d:.4f})   [infinite population]")
        print(f"{'k':>4}{'var obs':>10}{'var thy':>10}{'sep obs':>10}{'sep thy':>10}"
              f"{'AUC obs':>10}{'AUC thy':>10}{'abs err':>9}")
        for k in K_VALUES:
            y, feat = simulate(d, k, N_SIM, rng)
            var_o, sep_o, auc_o = measure(y, feat)
            var_t, sep_t, auc_t = 1.0 / k, np.sqrt(k) * d, theo_auc(k, d)
            print(f"{k:>4}{var_o:>10.4f}{var_t:>10.4f}{sep_o:>10.4f}{sep_t:>10.4f}"
                  f"{auc_o:>10.4f}{auc_t:>10.4f}{abs(auc_o - auc_t):>9.4f}")
            rowsA.append({"pool": "infinite", "standalone_auc": auc0, "k": k,
                          "var_obs": round(var_o, 5), "var_theory": round(var_t, 5),
                          "sep_obs": round(sep_o, 4), "sep_theory": round(sep_t, 4),
                          "auc_obs": round(auc_o, 4), "auc_theory": round(auc_t, 4),
                          "abs_err": round(abs(auc_o - auc_t), 4)})

    print("\n" + "-" * 78)
    print("finite same-class pool matching the real Stage-2 cohort (74 AD / 68 control)")
    print("-" * 78)
    for auc0 in [0.646]:
        d = auc_to_d(auc0)
        print(f"\nstandalone AUC = {auc0:.3f}  (d = {d:.4f})   [finite pool 74/68]")
        print(f"{'k':>4}{'AUC obs':>10}{'AUC thy':>10}{'abs err':>9}   note")
        for k in K_VALUES:
            aucs = []
            for _ in range(60):
                y, feat = simulate(d, k, N_SIM, rng, pool_sizes=(74, 68))
                aucs.append(roc_auc_score(y, feat))
            auc_o = float(np.mean(aucs))
            auc_t = theo_auc(k, d)
            note = "" if abs(auc_o - auc_t) < 0.02 else "<-- departs from identity"
            print(f"{k:>4}{auc_o:>10.4f}{auc_t:>10.4f}{abs(auc_o - auc_t):>9.4f}   {note}")
            rowsA.append({"pool": "finite_74_68", "standalone_auc": auc0, "k": k,
                          "var_obs": np.nan, "var_theory": np.nan,
                          "sep_obs": np.nan, "sep_theory": np.nan,
                          "auc_obs": round(auc_o, 4), "auc_theory": round(auc_t, 4),
                          "abs_err": round(abs(auc_o - auc_t), 4)})
    pd.DataFrame(rowsA).to_csv("results/sqrtk_simulation.csv", index=False)

    # ---------------- PART B ----------------
    print("\n" + "=" * 78)
    print("PART B -- empirical variance-vs-k on the real Stage-2 pool")
    print("=" * 78)
    s1 = pd.read_csv("results/stage1_mrna_oof_predictions.csv")
    s2 = pd.read_csv("results/stage2_methylation_oof_predictions.csv")
    y1 = s1["label"].to_numpy().astype(int)
    y2 = s2["label"].to_numpy().astype(int)
    l2 = logit(s2["oof_prob"].to_numpy())
    pool = {1: l2[y2 == 1], 0: l2[y2 == 0]}
    d_real = auc_to_d(roc_auc_score(y2, s2["oof_prob"].to_numpy()))
    print(f"real Stage-2 d = {d_real:.4f}, pool sizes: {len(pool[1])} AD / {len(pool[0])} control")
    print(f"{'k':>4}{'var obs':>10}{'var k=1/k':>11}{'sep obs':>10}{'sep thy':>10}"
          f"{'AUC obs':>10}{'AUC thy':>10}")
    rowsB = []
    var1 = None
    for k in K_VALUES:
        vs, ss, aus = [], [], []
        for rep in range(40):
            r = np.random.RandomState(1000 + rep)
            part = np.array([r.choice(pool[int(yy)], size=k, replace=True).mean() for yy in y1])
            v, s, a = measure(y1, part)
            vs.append(v); ss.append(s); aus.append(a)
        var_o, sep_o, auc_o = np.mean(vs), np.mean(ss), np.mean(aus)
        if k == 1:
            var1 = var_o
        print(f"{k:>4}{var_o:>10.4f}{var1/k:>11.4f}{sep_o:>10.4f}"
              f"{np.sqrt(k)*d_real:>10.4f}{auc_o:>10.4f}{theo_auc(k, d_real):>10.4f}")
        rowsB.append({"k": k, "var_obs": round(float(var_o), 5),
                      "var_scaled_from_k1": round(float(var1 / k), 5),
                      "sep_obs": round(float(sep_o), 4),
                      "sep_theory": round(float(np.sqrt(k) * d_real), 4),
                      "auc_obs": round(float(auc_o), 4),
                      "auc_theory": round(float(theo_auc(k, d_real)), 4)})
    pd.DataFrame(rowsB).to_csv("results/variance_vs_k_empirical.csv", index=False)

    # ---------------- PART C ----------------
    print("\n" + "=" * 78)
    print("PART C -- opposite-label and label-permuted pairing controls (k = 20)")
    print("=" * 78)
    p1 = s1["oof_prob"].to_numpy()
    lg1 = logit(p1)
    margin = s1["confidence_margin"].to_numpy()
    base_pred = (p1 > 0.5).astype(int)
    K = 20

    def run_condition(kind, seed):
        r = np.random.RandomState(seed)
        if kind == "same":
            part = np.array([r.choice(pool[int(yy)], size=K, replace=True).mean() for yy in y1])
        elif kind == "opposite":
            part = np.array([r.choice(pool[1 - int(yy)], size=K, replace=True).mean() for yy in y1])
        elif kind == "permuted":
            yp = r.permutation(y1)
            part = np.array([r.choice(pool[int(yy)], size=K, replace=True).mean() for yy in yp])
        elif kind == "random":
            part = np.array([r.choice(l2, size=K, replace=True).mean() for _ in y1])
        else:
            raise ValueError(kind)
        X = np.column_stack([lg1, part])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        prob = np.zeros(len(y1))
        for tr, te in skf.split(X, y1):
            clf = LogisticRegression(max_iter=1000).fit(X[tr], y1[tr])
            prob[te] = clf.predict_proba(X[te])[:, 1]
        return (prob > 0.5).astype(int)

    sub = margin < 0.6
    base_low = accuracy_score(y1[sub], base_pred[sub]) * 100
    base_all = accuracy_score(y1, base_pred) * 100
    print(f"mRNA-only baseline: whole cohort {base_all:.2f}%, margin<0.6 {base_low:.2f}%")
    print(f"\n{'condition':<20}{'whole cohort':>16}{'margin<0.6':>16}{'delta low-conf':>17}")
    rowsC = []
    for kind in ("same", "random", "permuted", "opposite"):
        aw, al = [], []
        for rep in range(30):
            pr = run_condition(kind, 2000 + rep)
            aw.append(accuracy_score(y1, pr) * 100)
            al.append(accuracy_score(y1[sub], pr[sub]) * 100)
        mw, sw = np.mean(aw), np.std(aw)
        ml, sl = np.mean(al), np.std(al)
        print(f"{kind:<20}{mw:>10.2f}% ({sw:>4.2f}){ml:>10.2f}% ({sl:>4.2f})"
              f"{ml - base_low:>+15.2f} pp")
        rowsC.append({"condition": kind, "k": K, "n_seeds": 30,
                      "acc_whole_mean_pct": round(float(mw), 2),
                      "acc_whole_sd_pct": round(float(sw), 2),
                      "acc_lowconf_mean_pct": round(float(ml), 2),
                      "acc_lowconf_sd_pct": round(float(sl), 2),
                      "delta_lowconf_vs_mrna_pp": round(float(ml - base_low), 2)})
    pd.DataFrame(rowsC).to_csv("results/pairing_controls.csv", index=False)
    print("\nWrote results/sqrtk_simulation.csv, results/variance_vs_k_empirical.csv, "
          "results/pairing_controls.csv")
