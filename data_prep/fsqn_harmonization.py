"""
fsqn_harmonization.py

Feature-specific quantile normalization (FSQN, Franks et al. 2018) and a
simpler rank-based transform, for harmonizing an external cross-platform
cohort (GSE118553 on GPL10558, or GSE5281 on GPL570) against the primary
training cohort's gene expression distribution -- in place of the per-sample
z-scoring the manuscript's Section 6.1 flags as the weakest available option.

FSQN, per gene independently: map each value in the target (external) cohort
to the reference (training) cohort's distribution for that same gene, via
rank -> quantile -> inverse-CDF. This differs from standard quantile
normalization, which forces every sample to one shared distribution across
ALL genes; FSQN preserves each gene's own reference distribution shape.

Expected input orientation: genes as rows, samples as columns (matching
collapse_probes_to_genes.py's output). If your reference gene panel
(gene_data.csv) is samples x genes instead, transpose it first -- see the
__main__ block for a --transpose flag.

Usage:
    python fsqn_harmonization.py \
        --reference data/raw/gene_data.csv --reference-transpose \
        --target data/processed/GSE118553_gene_level.csv \
        --method fsqn \
        --out data/processed/GSE118553_gene_level_fsqn.csv
"""
import argparse
import numpy as np
import pandas as pd


def fsqn_transform(reference_df, target_df):
    """
    reference_df, target_df: genes (rows) x samples (columns), same gene index
    space (already intersected before calling, or this intersects for you).
    Returns target_df with each gene's values remapped onto the reference
    cohort's distribution for that gene.

    Genes present in target but absent from reference are dropped from the
    output entirely, not passed through unharmonized -- the point of this
    step is producing a feature set the existing trained classifier can
    score, and it has no weights for genes outside its training panel
    regardless of whether they're harmonized. Genes present in both but with
    fewer than 2 valid reference values are kept with their original,
    untransformed target values (there's no distribution to map onto).
    """
    common_genes = reference_df.index.intersection(target_df.index)
    if len(common_genes) == 0:
        raise ValueError("no overlapping genes between reference and target -- "
                         "check that both are indexed by the same gene identifier "
                         "(symbol vs Ensembl ID mismatches are the usual cause)")

    ref = reference_df.loc[common_genes]
    tgt = target_df.loc[common_genes]
    out = tgt.copy().astype(float)

    for gene in common_genes:
        ref_vals = np.sort(ref.loc[gene].to_numpy(dtype=float))
        ref_vals = ref_vals[~np.isnan(ref_vals)]
        if len(ref_vals) < 2:
            continue  # can't build a distribution from <2 reference points

        tgt_vals = tgt.loc[gene].to_numpy(dtype=float)
        valid = ~np.isnan(tgt_vals)
        if valid.sum() == 0:
            continue

        # rank -> quantile in [0, 1] among the target cohort's own values
        ranks = pd.Series(tgt_vals[valid]).rank(method="average").to_numpy()
        n = valid.sum()
        quantiles = (ranks - 1) / (n - 1) if n > 1 else np.array([0.5] * n)

        # map those quantiles onto the reference distribution
        ref_quantile_positions = np.linspace(0, 1, len(ref_vals))
        mapped = np.interp(quantiles, ref_quantile_positions, ref_vals)

        row = np.array(out.loc[gene], dtype=float, copy=True)
        row[valid] = mapped
        out.loc[gene] = row

    return out


def rank_transform(target_df):
    """
    Simpler baseline: replace each sample's gene values with their
    within-sample percentile rank. Platform-scale-invariant by construction,
    since only the relative ordering within a sample matters -- no reference
    cohort needed, which is both its main advantage and its main limitation
    (it can't correct a systematic per-gene shift the way FSQN can).
    """
    return target_df.rank(axis=0, pct=True)


def load_matrix(path, transpose=False):
    df = pd.read_csv(path, index_col=0)
    if transpose:
        df = df.T
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True, help="training cohort gene-level CSV")
    p.add_argument("--reference-transpose", action="store_true",
                  help="pass this if the reference CSV is samples x genes "
                       "(rows=samples) instead of genes x samples")
    p.add_argument("--target", required=True, help="external cohort gene-level CSV "
                                                    "(genes x samples, e.g. from "
                                                    "collapse_probes_to_genes.py)")
    p.add_argument("--method", choices=["fsqn", "rank"], default="fsqn")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import os
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    target = load_matrix(args.target)
    print(f"target: {target.shape[0]} genes x {target.shape[1]} samples")

    if args.method == "fsqn":
        reference = load_matrix(args.reference, transpose=args.reference_transpose)
        print(f"reference: {reference.shape[0]} genes x {reference.shape[1]} samples")
        result = fsqn_transform(reference, target)
        print(f"FSQN applied to {len(reference.index.intersection(target.index))} "
             f"common genes")
    else:
        result = rank_transform(target)
        print("rank transform applied (no reference cohort used)")

    result.to_csv(args.out)
    print(f"written to {args.out}")
