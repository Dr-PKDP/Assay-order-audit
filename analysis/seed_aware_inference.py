"""
seed_aware_inference.py

Replaces the earlier practice of averaging DeLong p-values across
partner-draw seeds, which is not a valid inferential procedure, with two
defensible alternatives:

  (A) Seed-aggregated estimand. Average the k=1 fusion probabilities across
      seeds into one prediction per patient, then run a single paired test
      against the mRNA-only baseline. This defines one primary estimand
      rather than pooling many dependent tests.

  (B) Bootstrap targeting the mean effect over the seed distribution.
      Resample patients only; within each resampled patient set, average
      the accuracy difference across all 200 stored partner-draw seeds
      before taking the percentile interval across bootstrap replicates.
      This targets E_S[Delta_S] (the seed-averaged effect) with uncertainty
      from patient sampling alone, distinct from a procedure that draws one
      seed per replicate, which would instead estimate the wider
      distribution of a single arbitrary partner draw's effect.

The seed-level distribution of effect sizes is also reported descriptively,
which is the appropriate use of per-seed results.

Usage:
    python seed_aware_inference.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

N_SEEDS = 200
N_BOOT = 2000
MARGIN_THRESHOLD = 0.6
BOOT_SEED = 42


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fuse_probs(l1, y1, pool, k, seed):
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


if __name__ == "__main__":
    s1 = pd.read_csv("results/stage1_mrna_oof_predictions.csv")
    s2 = pd.read_csv("results/stage2_methylation_oof_predictions.csv")
    y1 = s1["label"].to_numpy().astype(int)
    y2 = s2["label"].to_numpy().astype(int)
    p1 = s1["oof_prob"].to_numpy()
    l1 = logit(p1)
    l2 = logit(s2["oof_prob"].to_numpy())
    margin = s1["confidence_margin"].to_numpy()
    mask = margin < MARGIN_THRESHOLD
    pool = {1: l2[y2 == 1], 0: l2[y2 == 0]}

    print(f"Seed-aware inference for k=1 fusion vs mRNA-only baseline")
    print(f"Subset: margin < {MARGIN_THRESHOLD}  (n = {mask.sum()})")
    print("=" * 78)

    all_probs = np.zeros((N_SEEDS, len(y1)))
    for s in range(N_SEEDS):
        all_probs[s] = fuse_probs(l1, y1, pool, 1, s)

    # (A) seed-aggregated primary estimand
    p_agg = all_probs.mean(axis=0)
    base_acc = accuracy_score(y1[mask], (p1[mask] > 0.5).astype(int))
    agg_acc = accuracy_score(y1[mask], (p_agg[mask] > 0.5).astype(int))
    base_auc = roc_auc_score(y1[mask], p1[mask])
    agg_auc = roc_auc_score(y1[mask], p_agg[mask])

    print("(A) Seed-aggregated estimand (one prediction per patient)")
    print(f"    mRNA-only   accuracy = {base_acc*100:6.2f}%   AUC = {base_auc:.4f}")
    print(f"    k=1 fusion  accuracy = {agg_acc*100:6.2f}%   AUC = {agg_auc:.4f}")
    print(f"    difference           = {(agg_acc-base_acc)*100:+6.2f} pp   "
          f"AUC {agg_auc-base_auc:+.4f}")

    # (B) Hierarchical bootstrap targeting the mean effect over the seed distribution.
    # For each bootstrap replicate: resample patients, then average the accuracy
    # difference across ALL 200 stored partner-draw seeds within that resampled set
    # (not one seed per replicate), so the percentile interval reflects uncertainty
    # in the seed-averaged effect due to patient sampling, not seed-to-seed noise.
    rng = np.random.default_rng(BOOT_SEED)
    idx = np.where(mask)[0]
    diffs = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        bi = rng.choice(idx, size=len(idx), replace=True)
        y_bi = y1[bi]
        base_acc_bi = ((p1[bi] > 0.5).astype(int) == y_bi).mean()
        preds_all_seeds = (all_probs[:, bi] > 0.5).astype(int)
        seed_accs = (preds_all_seeds == y_bi[None, :]).mean(axis=1)
        diffs[b] = seed_accs.mean() - base_acc_bi
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print()
    print("(B) Bootstrap targeting the mean effect over the empirical seed distribution")
    print(f"    mean accuracy difference = {diffs.mean()*100:+.2f} pp")
    print(f"    95% CI = [{lo*100:+.2f}, {hi*100:+.2f}] pp")
    print(f"    CI includes zero: {bool(lo <= 0 <= hi)}")

    # seed-level effect distribution, reported descriptively only
    seed_diffs = np.array([
        accuracy_score(y1[mask], (all_probs[s][mask] > 0.5).astype(int)) - base_acc
        for s in range(N_SEEDS)])
    print()
    print(f"(C) Seed-level effect distribution (descriptive, {N_SEEDS} seeds)")
    print(f"    mean = {seed_diffs.mean()*100:+.2f} pp, "
          f"sd = {seed_diffs.std()*100:.2f} pp")
    print(f"    range = [{seed_diffs.min()*100:+.2f}, {seed_diffs.max()*100:+.2f}] pp")
    print(f"    fraction of seeds favouring fusion = "
          f"{(seed_diffs > 0).mean()*100:.1f}%")

    # Monte Carlo precision of the 200-seed mean itself, answering whether 200
    # seeds is enough to approximate the full random-pairing distribution's mean.
    mc_se = seed_diffs.std() / np.sqrt(N_SEEDS)
    mc_halfwidth = 1.96 * mc_se
    patient_ci_halfwidth = (hi - lo) / 2
    print()
    print("(D) Monte Carlo precision of the 200-seed mean")
    print(f"    MC standard error of the mean = {mc_se*100:.3f} pp")
    print(f"    95% MC-error half-width = +/-{mc_halfwidth*100:.2f} pp")
    print(f"    patient-sampling CI half-width = +/-{patient_ci_halfwidth*100:.2f} pp")
    print(f"    MC error as fraction of patient-sampling CI: "
          f"{mc_halfwidth/patient_ci_halfwidth*100:.1f}%")

    pd.DataFrame([{
        "subset": f"margin<{MARGIN_THRESHOLD}", "n": int(mask.sum()),
        "baseline_acc_pct": round(base_acc * 100, 2),
        "seed_aggregated_acc_pct": round(agg_acc * 100, 2),
        "seed_aggregated_diff_pp": round((agg_acc - base_acc) * 100, 2),
        "baseline_auc": round(base_auc, 4),
        "seed_aggregated_auc": round(agg_auc, 4),
        "hier_boot_mean_diff_pp": round(diffs.mean() * 100, 2),
        "hier_boot_ci_lo_pp": round(lo * 100, 2),
        "hier_boot_ci_hi_pp": round(hi * 100, 2),
        "seed_diff_sd_pp": round(seed_diffs.std() * 100, 2),
        "mc_se_of_mean_pp": round(mc_se * 100, 3),
        "mc_95_halfwidth_pp": round(mc_halfwidth * 100, 2),
        "n_seeds": N_SEEDS,
    }]).to_csv("results/seed_aware_inference.csv", index=False)
    print("\nWrote results/seed_aware_inference.csv")
