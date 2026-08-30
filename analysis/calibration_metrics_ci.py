"""
calibration_metrics_ci.py

Adds bootstrap 95% CIs to the calibration slope, intercept, ECE, and ICI
already computed by calibration_metrics.py, using the same patient-level
resampling as the rest of the paper (2,000 resamples). Particularly
important for the small external cohorts (n=34, n=63), where point estimates
alone understate the uncertainty.

Usage:
    python calibration_metrics_ci.py
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import statsmodels.api as sm

N_BOOT = 2000
SEED = 42


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y, p):
    lp = logit(p)
    slope = intercept = np.nan
    try:
        m = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
        slope = m.params[1]
    except Exception:
        pass
    try:
        m0 = sm.Logit(y, np.ones((len(y), 1)), offset=lp).fit(disp=0)
        intercept = m0.params[0]
    except Exception:
        pass
    return slope, intercept


def ece(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    total, n = 0.0, len(p)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return total


def ici(y, p):
    try:
        iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        return float(np.mean(np.abs(iso.predict(p) - p)))
    except Exception:
        return np.nan


COHORTS = [
    ("Stage-1 internal (OOF)", "results/stage1_mrna_oof_predictions.csv", "oof_prob"),
    ("Stage-1 GSE118553", "results/stage1_mrna_external_predictions.csv", None),
    ("Stage-1 GSE5281", "results/stage1_mrna_external2_GSE5281_predictions.csv", None),
    ("Stage-2 internal (OOF)", "results/stage2_methylation_oof_predictions.csv", "oof_prob"),
    ("Stage-2 GSE134379", "results/stage2_methylation_external_GSE134379_predictions.csv", None),
]

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    print("Calibration metrics with bootstrap 95% CIs")
    print("=" * 100)
    rows = []
    for name, path, probcol in COHORTS:
        df = pd.read_csv(path)
        if probcol is None:
            probcol = [c for c in df.columns
                       if "prob" in c.lower() and "uncert" not in c.lower()][0]
        y = df["label"].to_numpy().astype(int)
        p = df[probcol].to_numpy()
        n = len(y)

        boot_slope, boot_int, boot_ece, boot_ici = [], [], [], []
        for _ in range(N_BOOT):
            bi = rng.integers(0, n, n)
            yb, pb = y[bi], p[bi]
            if len(np.unique(yb)) < 2:
                continue
            s, ic = calibration_slope_intercept(yb, pb)
            boot_slope.append(s)
            boot_int.append(ic)
            boot_ece.append(ece(yb, pb))
            boot_ici.append(ici(yb, pb))

        def ci(arr):
            arr = np.array([a for a in arr if np.isfinite(a)])
            if len(arr) < 50:
                return (np.nan, np.nan)
            return tuple(np.percentile(arr, [2.5, 97.5]))

        s_lo, s_hi = ci(boot_slope)
        i_lo, i_hi = ci(boot_int)
        e_lo, e_hi = ci(boot_ece)
        c_lo, c_hi = ci(boot_ici)

        print(f"{name} (n={n})")
        print(f"  slope CI      [{s_lo:.3f}, {s_hi:.3f}]")
        print(f"  intercept CI  [{i_lo:.3f}, {i_hi:.3f}]")
        print(f"  ECE CI        [{e_lo:.3f}, {e_hi:.3f}]")
        print(f"  ICI CI        [{c_lo:.3f}, {c_hi:.3f}]")
        rows.append({"cohort": name, "n": n,
                     "slope_ci_lo": round(s_lo, 3), "slope_ci_hi": round(s_hi, 3),
                     "intercept_ci_lo": round(i_lo, 3), "intercept_ci_hi": round(i_hi, 3),
                     "ece_ci_lo": round(e_lo, 3), "ece_ci_hi": round(e_hi, 3),
                     "ici_ci_lo": round(c_lo, 3), "ici_ci_hi": round(c_hi, 3)})

    pd.DataFrame(rows).to_csv("results/calibration_metrics_ci.csv", index=False)
    print("\nWrote results/calibration_metrics_ci.csv")
