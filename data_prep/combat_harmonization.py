"""
combat_harmonization.py

Empirical-Bayes batch correction (ComBat, Johnson et al. 2007 -- already
reference [8] in the manuscript) as an alternative to FSQN for harmonizing
an external cohort against the primary training panel. Unlike FSQN's
per-gene distribution mapping, ComBat models a mean (location) and variance
(scale) batch effect per gene, with empirical-Bayes shrinkage that borrows
strength across genes -- a different theoretical approach, and the field's
most standard one, so a null result here closes the harmonization question
far more convincingly than FSQN alone.

NO DIAGNOSIS COVARIATE IS USED. An earlier version of this script passed
each cohort's true diagnosis label as a covariate to "preserve" disease-
correlated signal during batch correction -- a standard, legitimate ComBat
usage in other contexts (e.g. batch-correcting a full cohort before
differential expression analysis, where nothing downstream re-predicts the
covariate from held-out samples). It is NOT legitimate here: this harmonized
output is later scored by a classifier trained to predict that same label,
and ComBat's covariate-preservation mechanism uses each sample's OWN
covariate value to shape its OWN adjusted expression -- meaning each
external sample's true label was being baked directly into the features
used to predict it. This was caught empirically, not just reasoned about:
a synthetic external cohort built with zero true signal (labels independent
of expression by construction) still scored well above chance (AUC ~0.63 at
2,000 genes/300 selected, and presumably higher still at this study's actual
~13,000-gene scale) after the label-covariate version of this transform --
proof of leakage, since no real classifier can beat chance on data with no
signal. Removing the covariate and re-running the same adversarial test
confirmed the leak closes (see the test suite this script was developed
against). This version uses batch identity only -- no covariates -- exactly
matching FSQN's use of no label information at all, for a fair comparison
between the two methods.

Uses neuroCombat rather than the `combat` PyPI package: the latter's
ref_batch handling has a real indexing bug in the installed version (throws
IndexError regardless of whether covariates are supplied), verified by
testing before switching packages. neuroCombat needed one compatibility
patch (`np.int` -> `int`, a numpy deprecation from 1.20 the package
predates) -- a safe, well-documented, zero-behavior-change fix.

The training cohort is passed as `ref_batch` so its own distribution is
preserved (matching FSQN's target-onto-reference direction).

Usage:
    python combat_harmonization.py \
        --reference data/raw/gene_data.csv \
        --target data/processed/GSE118553_gene_level.csv \
        --out data/processed/GSE118553_gene_level_combat.csv
"""
import argparse
import os
import numpy as np
np.int = int  # neuroCombat compatibility shim -- see module docstring

import pandas as pd
from neuroCombat import neuroCombat


def load_matrix(path, transpose=False):
    df = pd.read_csv(path, index_col=0)
    if transpose:
        df = df.T
    return df


def combat_transform(reference_df, target_df):
    """
    reference_df, target_df: genes (rows) x samples (columns).

    Returns target_df's samples only, harmonized onto reference_df's
    distribution via batch identity alone -- no diagnosis covariate, and
    therefore no risk of leaking target labels into the harmonized
    features (see module docstring). Genes outside the reference are
    dropped, matching fsqn_harmonization.py's convention exactly, for a
    fair comparison between the two methods.
    """
    common_genes = reference_df.index.intersection(target_df.index)
    if len(common_genes) == 0:
        raise ValueError("no overlapping genes between reference and target -- "
                         "check both are indexed by the same gene identifier")

    ref = reference_df.loc[common_genes]
    tgt = target_df.loc[common_genes]

    dat = np.hstack([ref.to_numpy(dtype=float), tgt.to_numpy(dtype=float)])
    batch = ["reference"] * ref.shape[1] + ["target"] * tgt.shape[1]
    covars = pd.DataFrame({"batch": batch})

    result = neuroCombat(dat=dat, covars=covars, batch_col="batch", ref_batch="reference")
    adjusted = result["data"]

    adj_target = adjusted[:, ref.shape[1]:]
    return pd.DataFrame(adj_target, index=common_genes, columns=tgt.columns)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True, help="training cohort gene-level CSV "
                                                       "(gene_data.csv: samples x genes, "
                                                       "with sample and label columns)")
    p.add_argument("--reference-label-col", default="label",
                  help="column name to drop from the reference CSV before harmonizing "
                       "(the label itself is not used, just excluded from the gene matrix)")
    p.add_argument("--target", required=True, help="external cohort gene-level CSV "
                                                    "(genes x samples)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    raw_ref = pd.read_csv(args.reference)
    sample_col = raw_ref.columns[0]
    reference = raw_ref.drop(columns=[args.reference_label_col]).set_index(sample_col).T

    target = load_matrix(args.target)

    print(f"reference: {reference.shape[0]} genes x {reference.shape[1]} samples")
    print(f"target: {target.shape[0]} genes x {target.shape[1]} samples")

    result = combat_transform(reference, target)
    result.to_csv(args.out)
    print(f"ComBat applied to {result.shape[0]} common genes, {result.shape[1]} target samples")
    print(f"written to {args.out}")

