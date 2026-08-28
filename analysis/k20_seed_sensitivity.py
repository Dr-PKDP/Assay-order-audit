"""
k20_seed_sensitivity.py

THE canonical script for the k=20 same-label partner-draw seed-sensitivity
check reported in Section 5.4 of the manuscript. The standard convention
(Section 4.2) draws 20 same-label methylation partners per mRNA sample using
one fixed random seed and reports the resulting fusion accuracy as if it
were a single, deterministic number. This script asks the question the
convention skips: how much does that number move if a different, equally
arbitrary seed had been drawn instead?

Method: for each of N_SEEDS partner-draw seeds, draw 20 same-label Stage-2
partners per Stage-1 sample (with replacement, matching the standard
convention in Section 4.2), average their logits, fit a two-feature logistic
meta-model (mRNA logit, averaged partner logit) by 5-fold stratified CV, and
record out-of-fold accuracy on both the whole cohort and the margin<0.6
subset. Both the partner draw and the CV fold assignment are reseeded
together per seed, so each of the N_SEEDS rows is a fully independent
realization of "what if the convention's one arbitrary seed had come out
differently."

Usage:
    python k20_seed_sensitivity.py \
        --stage1 results/stage1_mrna_oof_predictions.csv \
        --stage2 results/stage2_methylation_oof_predictions.csv \
        --out results/k20_seed_sensitivity.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

N_SEEDS = 30
K_PARTNERS = 20
MARGIN_THRESHOLD = 0.6


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fuse_one_seed(l1, y1, pool, seed):
    """k=20 same-label fusion, learned meta-model, 5-fold CV, one partner-draw seed."""
    rng = np.random.RandomState(seed)
    partner_avg = np.array([
        rng.choice(pool[int(yi)], size=K_PARTNERS, replace=True).mean()
        for yi in y1
    ])
    X = np.column_stack([l1, partner_avg])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.zeros(len(y1), dtype=int)
    for tr, te in skf.split(X, y1):
        clf = LogisticRegression(max_iter=1000).fit(X[tr], y1[tr])
        pred[te] = clf.predict(X[te])
    return pred


def run(stage1_csv, stage2_csv, out_csv):
    s1 = pd.read_csv(stage1_csv)
    s2 = pd.read_csv(stage2_csv)
    y1 = s1["label"].to_numpy().astype(int)
    y2 = s2["label"].to_numpy().astype(int)
    l1 = logit(s1["oof_prob"].to_numpy())
    l2 = logit(s2["oof_prob"].to_numpy())
    margin = s1["confidence_margin"].to_numpy()
    mask_low = margin < MARGIN_THRESHOLD
    pool = {1: l2[y2 == 1], 0: l2[y2 == 0]}

    rows = []
    for seed in range(N_SEEDS):
        pred = fuse_one_seed(l1, y1, pool, seed)
        acc_whole = accuracy_score(y1, pred)
        acc_low = accuracy_score(y1[mask_low], pred[mask_low])
        rows.append({"seed": seed, "acc_whole": acc_whole, "acc_low_conf": acc_low})

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    reported = 0.9310  # Table 5's k=20 same-label figure for margin<0.6, one fixed seed
    pctile = (df["acc_low_conf"] < reported).mean() * 100

    print(f"k=20 same-label fusion, seed-sensitivity over {N_SEEDS} partner-draw seeds")
    print("=" * 78)
    print(f"Whole cohort:      mean={df['acc_whole'].mean()*100:.2f}%  "
          f"sd={df['acc_whole'].std()*100:.2f}  "
          f"range=[{df['acc_whole'].min()*100:.1f}, {df['acc_whole'].max()*100:.1f}]")
    print(f"Margin<{MARGIN_THRESHOLD} subset: mean={df['acc_low_conf'].mean()*100:.2f}%  "
          f"sd={df['acc_low_conf'].std()*100:.2f}  "
          f"range=[{df['acc_low_conf'].min()*100:.1f}, {df['acc_low_conf'].max()*100:.1f}]")
    print(f"\nReported figure ({reported*100:.2f}%) sits at the "
          f"{pctile:.1f}th percentile of this distribution.")
    print(f"\nWrote {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage1", default="results/stage1_mrna_oof_predictions.csv")
    p.add_argument("--stage2", default="results/stage2_methylation_oof_predictions.csv")
    p.add_argument("--out", default="results/k20_seed_sensitivity.csv")
    args = p.parse_args()
    run(args.stage1, args.stage2, args.out)
