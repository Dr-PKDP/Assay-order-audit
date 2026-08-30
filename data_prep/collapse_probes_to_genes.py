"""
collapse_probes_to_genes.py

Maps a series matrix's probe-level expression to gene-level features using
an annotation file (GPL570_annotation.tsv or GPL10558_annotation.tsv), then
aggregates multi-probe genes by mean -- matching the convention already used
elsewhere in this pipeline for GSE118553.

AMBIGUOUS PROBES (one probe mapping to multiple genes, e.g. the GAGE family
on GPL10558): this is a real methodological choice, not a technical detail.
Two modes are supported:

    "drop"        -- discard ambiguous probes entirely (default; the safer
                     choice for new analysis, avoids injecting one probe's
                     signal into multiple unrelated gene-level features)
    "first"       -- keep the first gene listed for each ambiguous probe
                     (matches how the existing pipeline processed GSE5281's
                     GPL570 annotation -- use this if you need results
                     consistent with the manuscript's already-reported
                     GSE5281 numbers)

The annotation files are read positionally (column 0 = probe ID, column 1 =
gene symbol) rather than by exact header text, since the BioMart and
Bioconductor scripts that produced them don't necessarily use identical
column headers.

Usage:
    python collapse_probes_to_genes.py \
        --matrix data/raw/GSE5281_series_matrix.txt.gz \
        --annotation data/raw/GPL570_annotation.tsv \
        --mode drop \
        --out data/processed/GSE5281_gene_level.csv
"""
import argparse
import os
import sys
import pandas as pd

from read_series_matrix import read_series_matrix


def load_expression(matrix_path):
    """Accepts either a raw gzipped GEO series matrix (.txt.gz) or a plain
    CSV already filtered/subsetted by something like
    filter_gse118553_samples.py (samples as columns, probes as rows,
    produced via DataFrame.to_csv() with the probe ID as the index)."""
    if matrix_path.endswith(".gz"):
        expr, _ = read_series_matrix(matrix_path)
        return expr
    else:
        return pd.read_csv(matrix_path, index_col=0)


def load_annotation(path, mode):
    ann = pd.read_csv(path, sep="\t")
    probe_col, symbol_col = ann.columns[0], ann.columns[1]
    ann = ann[[probe_col, symbol_col]].copy()
    ann.columns = ["probe", "gene"]

    n_raw = len(ann)
    ann = ann.dropna(subset=["gene"])
    ann = ann[ann["gene"].astype(str).str.strip() != ""]
    n_no_symbol = n_raw - len(ann)

    dupe_counts = ann["probe"].value_counts()
    ambiguous_probes = set(dupe_counts[dupe_counts > 1].index)
    n_ambiguous = len(ambiguous_probes)

    if mode == "drop":
        ann = ann[~ann["probe"].isin(ambiguous_probes)]
    elif mode == "first":
        ann = ann.drop_duplicates(subset=["probe"], keep="first")
    else:
        raise ValueError(f"unknown mode: {mode!r} (use 'drop' or 'first')")

    mapping = dict(zip(ann["probe"], ann["gene"]))
    print(f"  annotation: {n_raw} rows -> {n_no_symbol} dropped (no gene symbol), "
         f"{n_ambiguous} ambiguous probes ({mode}), {len(mapping)} usable probe->gene pairs")
    return mapping


def collapse(matrix_path, annotation_path, mode, out_path):
    print(f"reading expression matrix: {matrix_path}")
    expr = load_expression(matrix_path)
    print(f"  {expr.shape[0]} probes x {expr.shape[1]} samples")

    print(f"reading annotation: {annotation_path}")
    mapping = load_annotation(annotation_path, mode)

    matched = expr.index.isin(mapping.keys())
    print(f"  {matched.sum()} / {len(expr)} probes in the expression matrix "
         f"found in the annotation ({matched.sum()/len(expr):.1%})")

    expr_matched = expr.loc[matched].copy()
    expr_matched["__gene__"] = [mapping[p] for p in expr_matched.index]

    gene_level = expr_matched.groupby("__gene__").mean(numeric_only=True)
    gene_level.index.name = "gene"

    gene_level.to_csv(out_path)
    print(f"  collapsed to {gene_level.shape[0]} genes x {gene_level.shape[1]} samples")
    print(f"  written to {out_path}")
    return gene_level


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", required=True, help="gzipped series matrix file")
    p.add_argument("--annotation", required=True, help="annotation TSV (probe, gene, ...)")
    p.add_argument("--mode", choices=["drop", "first"], default="drop",
                  help="how to handle probes mapping to multiple genes (default: drop)")
    p.add_argument("--out", required=True, help="output CSV path for the gene-level matrix")
    args = p.parse_args()

    import os
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    collapse(args.matrix, args.annotation, args.mode, args.out)
