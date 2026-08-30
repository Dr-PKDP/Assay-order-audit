"""
filter_gse118553_samples.py

GSE118553 profiles four brain regions and three diagnostic categories (401
samples total). This study's design uses only frontal-cortex samples (96
total), split into the accuracy-test set (AD vs control, 63 samples: 40 AD +
23 control) and a held-aside asymptomatic-AD set (33 samples, not used for
accuracy claims). This filters the raw series matrix down to those two sets
before probe-to-gene collapse, so collapse_probes_to_genes.py operates on
the right sample set instead of all 401.

Field names confirmed from this series' actual metadata (via
inspect_sample_metadata.py):
    tissue:        'Sample_source_name_ch1' (or the matching
                   'Sample_characteristics_ch1 [line 4]' -- both carry the
                   same value, source_name_ch1 is used here as the simpler
                   single-line field)
    disease state: 'Sample_characteristics_ch1 [line 5]', values
                   'disease state: AD' / 'disease state: control' /
                   'disease state: AsymAD'

If you run this against a different series where the inspector found
different field names, edit TISSUE_KEY / DIAGNOSIS_KEY below to match.

Usage:
    python filter_gse118553_samples.py \
        --matrix data/raw/GSE118553_series_matrix.txt.gz \
        --out-test data/processed/GSE118553_frontal_ADcontrol_probes.csv \
        --out-asymad data/processed/GSE118553_frontal_asymAD_probes.csv
"""
import argparse
import os
import pandas as pd
from read_series_matrix import read_series_matrix

TISSUE_KEY = "Sample_source_name_ch1"
TISSUE_VALUE = "Frontal_Cortex"
DIAGNOSIS_KEY = "Sample_characteristics_ch1 [line 5]"
DIAGNOSIS_PREFIX = "disease state: "


def filter_samples(matrix_path, out_test_path, out_asymad_path):
    print(f"reading {matrix_path} ...")
    expr, meta = read_series_matrix(matrix_path)
    print(f"  {expr.shape[0]} probes x {expr.shape[1]} samples (all regions/diagnoses)")

    if TISSUE_KEY not in meta:
        raise KeyError(f"'{TISSUE_KEY}' not found in metadata. Available keys:\n  "
                       + "\n  ".join(meta.keys()) +
                       "\nRe-run inspect_sample_metadata.py and update TISSUE_KEY.")
    if DIAGNOSIS_KEY not in meta:
        raise KeyError(f"'{DIAGNOSIS_KEY}' not found in metadata. Available keys:\n  "
                       + "\n  ".join(meta.keys()) +
                       "\nRe-run inspect_sample_metadata.py and update DIAGNOSIS_KEY.")

    tissue = meta[TISSUE_KEY]
    diagnosis_raw = meta[DIAGNOSIS_KEY]
    diagnosis = [d.replace(DIAGNOSIS_PREFIX, "").strip() for d in diagnosis_raw]

    if len(tissue) != expr.shape[1] or len(diagnosis) != expr.shape[1]:
        raise ValueError(
            f"metadata length mismatch: {len(tissue)} tissue / {len(diagnosis)} "
            f"diagnosis values vs {expr.shape[1]} expression columns -- the "
            f"series matrix format may differ from what this script assumes."
        )

    frontal_mask = [t == TISSUE_VALUE for t in tissue]
    n_frontal = sum(frontal_mask)
    print(f"  {n_frontal} samples are {TISSUE_VALUE}")

    diag_counts = {}
    for keep, d in zip(frontal_mask, diagnosis):
        if keep:
            diag_counts[d] = diag_counts.get(d, 0) + 1
    print(f"  diagnosis breakdown within {TISSUE_VALUE}: {diag_counts}")

    test_cols = [c for c, keep, d in zip(expr.columns, frontal_mask, diagnosis)
                if keep and d in ("AD", "control")]
    asymad_cols = [c for c, keep, d in zip(expr.columns, frontal_mask, diagnosis)
                  if keep and d == "AsymAD"]
    test_labels = {c: (1 if d == "AD" else 0)
                   for c, keep, d in zip(expr.columns, frontal_mask, diagnosis)
                   if keep and d in ("AD", "control")}

    test_df = expr[test_cols]
    asymad_df = expr[asymad_cols]

    print(f"  AD+control test set: {test_df.shape[1]} samples")
    print(f"  AsymAD held-aside set: {asymad_df.shape[1]} samples")

    for path, df in [(out_test_path, test_df), (out_asymad_path, asymad_df)]:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(path)
        print(f"  written: {path}")

    labels_path = out_test_path.rsplit(".", 1)[0] + "_labels.csv"
    pd.DataFrame({"sample": list(test_labels.keys()),
                 "label": list(test_labels.values())}).to_csv(labels_path, index=False)
    print(f"  written: {labels_path}  (1=AD, 0=control, matches sample order not required "
         f"-- join on 'sample')")

    return test_df, asymad_df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", required=True)
    p.add_argument("--out-test", required=True,
                  help="output CSV for the AD-vs-control accuracy test set")
    p.add_argument("--out-asymad", required=True,
                  help="output CSV for the held-aside asymptomatic-AD set")
    args = p.parse_args()
    filter_samples(args.matrix, args.out_test, args.out_asymad)
