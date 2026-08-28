"""
check_annotation_files.py

Quick sanity check on the two annotation files you just produced, before
they go into the pipeline. Confirms row counts, unique probe coverage, and
shows what the 1:many mappings AnnotationDbi warned about actually look like.

Usage:
    python check_annotation_files.py
"""
import pandas as pd

for path, expected_unique in [
    ("data/raw/GPL570_annotation.tsv", 54675),   # nominal probe count on this chip
    ("data/raw/GPL10558_annotation.tsv", 47323),  # nominal probe count on this chip
]:
    print(f"=== {path} ===")
    try:
        df = pd.read_csv(path, sep="\t")
    except FileNotFoundError:
        print("  not found -- skipping\n")
        continue

    probe_col = df.columns[0]
    print(f"  total rows: {len(df)}")
    print(f"  unique probes: {df[probe_col].nunique()}  (chip nominally has ~{expected_unique})")

    dupe_counts = df[probe_col].value_counts()
    multi = dupe_counts[dupe_counts > 1]
    print(f"  probes with >1 row (the '1:many' warning): {len(multi)} "
         f"({len(multi) / df[probe_col].nunique():.1%} of unique probes)")

    if len(multi) > 0:
        example_probe = multi.index[0]
        print(f"  example -- probe {example_probe!r} maps to:")
        print(df[df[probe_col] == example_probe].to_string(index=False))

    n_missing_symbol = df.iloc[:, 1].isna().sum() if df.shape[1] > 1 else None
    if n_missing_symbol is not None:
        print(f"  rows with no gene symbol at all: {n_missing_symbol} "
             f"({n_missing_symbol/len(df):.1%})")
    print()
