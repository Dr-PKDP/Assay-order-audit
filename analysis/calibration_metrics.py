"""
calibration_metrics.py

Reports calibration slope, calibration intercept (calibration-in-the-large),
expected calibration error (ECE) under two binning schemes, and the
integrated calibration index (ICI), for every cohort-stage combination.

Rationale: Brier score is a composite of calibration and discrimination and
cannot on its own establish that calibration specifically degraded, or that
it degraded faster than ranking. Calibration slope and intercept separate
those effects: slope < 1 indicates overconfident (too extreme) predictions,
and a non-zero intercept indicates systematic over- or under-prediction of
prevalence.

Convention follows the standard weak-calibration framework: logistic
recalibration of the outcome on the linear predictor, where slope = 1 and
intercept = 0 denote perfect calibration.

Usage:
    python calibration_metrics.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y, p):
    """Slope from regressing y on the linear predictor; intercept with slope
    fixed at 1 (calibration-in-the-large)."""
    lp = logit(p)
    slope = np.nan
    try:
        m = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
        slope = m.params[1]
    except Exception:
        pass
    intercept = np.nan
    try:
        m0 = sm.Logit(y, np.ones((len(y), 1)), offset=lp).fit(disp=0)
        intercept = m0.params[0]
    except Exception:
        pass
    return slope, intercept


def ece(y, p, n_bins=10, strategy="uniform"):
    if strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:
        edges = np.percentile(p, np.linspace(0, 100, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    total, n = 0.0, len(p)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return total


def ici(y, p):
    """Integrated calibration index: mean |calibrated - predicted|, using an
    isotonic calibration curve."""
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
    print("Calibration metrics by cohort")
    print("=" * 108)
    print(f"{'Cohort':<26}{'n':>5}{'prev':>7}{'AUC':>8}{'Brier':>8}"
          f"{'slope':>9}{'intercept':>11}{'ECE-10':>9}{'ECE-q':>8}{'ICI':>8}")
    print("-" * 108)

    rows = []
    for name, path, probcol in COHORTS:
        df = pd.read_csv(path)
        if probcol is None:
            cand = [c for c in df.columns
                    if "prob" in c.lower() and "uncert" not in c.lower()]
            probcol = cand[0]
        y = df["label"].to_numpy().astype(int)
        p = df[probcol].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        auc = roc_auc_score(y, p)
        br = brier_score_loss(y, p)
        slope, intercept = calibration_slope_intercept(y, p)
        e10 = ece(y, p, 10, "uniform")
        eq = ece(y, p, 10, "quantile")
        i_ci = ici(y, p)
        print(f"{name:<26}{len(y):>5}{y.mean():>7.3f}{auc:>8.3f}{br:>8.3f}"
              f"{slope:>9.3f}{intercept:>11.3f}{e10:>9.3f}{eq:>8.3f}{i_ci:>8.3f}")
        rows.append({"cohort": name, "n": len(y), "prevalence": round(y.mean(), 3),
                     "auc": round(auc, 3), "brier": round(br, 3),
                     "calibration_slope": round(slope, 3),
                     "calibration_intercept": round(intercept, 3),
                     "ece_10_uniform": round(e10, 3), "ece_10_quantile": round(eq, 3),
                     "ici": round(i_ci, 3)})

    pd.DataFrame(rows).to_csv("results/calibration_metrics.csv", index=False)
    print("\nSlope 1.0 and intercept 0.0 denote perfect calibration.")
    print("Slope < 1 indicates predictions that are too extreme (overconfident).")
    print("\nWrote results/calibration_metrics.csv")
