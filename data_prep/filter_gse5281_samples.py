"""
filter_gse5281_samples.py

GSE5281 profiles six brain regions across 161 samples, submitted in two
batches (2006 and 2007) that use inconsistently-cased metadata field values
("Organ Region: X" vs "organ region: X") and trailing non-breaking-space
characters on values. This study's design uses only superior frontal gyrus
(34 samples: 23 AD, 11 control) -- confirmed by 29 + 5 = 34 across the two
casing variants, matching the manuscript's reported n.

Field names confirmed from this series' actual metadata (via
inspect_sample_metadata.py):
    region:    'Sample_characteristics_ch1 [line 5]', values like
               'Organ Region: Superior Frontal Gyrus\xa0' or
               'organ region: Superior Frontal Gyrus\xa0' (case varies by
               submission batch; trailing \xa0 present)
    diagnosis: 'Sample_characteristics_ch1 [line 9]', values like
               "Disease State: Alzheimer's Disease\xa0" / 'Disease State: normal\xa0'
               or the lowercase-prefixed equivalents

Both fields are parsed case-insensitively on the "key: " prefix and stripped
of \xa0 and ordinary whitespace before comparison, so the two submission
batches merge correctly into one filter instead of only matching half the
samples.

Usage:
    python filter_gse5281_samples.py \
        --matrix data/raw/GSE5281_series_matrix.txt.gz \
        --out data/processed/GSE5281_sfg_probes.csv
"""
import argparse
import os
import re
import pandas as pd
from read_series_matrix import read_series_matrix

REGION_KEY = "Sample_characteristics_ch1 [line 5]"
REGION_VALUE = "superior frontal gyrus"
DIAGNOSIS_KEY = "Sample_characteristics_ch1 [line 9]"
DIAGNOSIS_MAP = {
    "alzheimer's disease": "AD",
    "normal": "control",
}


def clean_value(raw, prefix_pattern):
    """Strip a 'Key: ' prefix (case-insensitive) plus \xa0/whitespace."""
    v = re.sub(prefix_pattern, "", raw, flags=re.IGNORECASE)
    return v.replace("\xa0", "").strip()


def filter_samples(matrix_path, out_path):
    print(f"reading {matrix_path} ...")
    expr, meta = read_series_matrix(matrix_path)
    print(f"  {expr.shape[0]} probes x {expr.shape[1]} samples (all regions)")

    for key in (REGION_KEY, DIAGNOSIS_KEY):
        if key not in meta:
            raise KeyError(f"'{key}' not found in metadata. Available keys:\n  "
                           + "\n  ".join(meta.keys()) +
                           "\nRe-run inspect_sample_metadata.py and update the "
                           "REGION_KEY/DIAGNOSIS_KEY constants at the top of this file.")

    region_raw = meta[REGION_KEY]
    diagnosis_raw = meta[DIAGNOSIS_KEY]
    if len(region_raw) != expr.shape[1] or len(diagnosis_raw) != expr.shape[1]:
        raise ValueError(
            f"metadata length mismatch: {len(region_raw)} region / "
            f"{len(diagnosis_raw)} diagnosis values vs {expr.shape[1]} "
            f"expression columns."
        )

    region = [clean_value(r, r"^organ region:\s*") for r in region_raw]
    diagnosis_clean = [clean_value(d, r"^disease state:\s*") for d in diagnosis_raw]
    diagnosis = [DIAGNOSIS_MAP.get(d.lower(), d) for d in diagnosis_clean]

    region_mask = [r.lower() == REGION_VALUE for r in region]
    n_region = sum(region_mask)
    print(f"  {n_region} samples are Superior Frontal Gyrus (both batches combined)")

    diag_counts = {}
    for keep, d in zip(region_mask, diagnosis):
        if keep:
            diag_counts[d] = diag_counts.get(d, 0) + 1
    print(f"  diagnosis breakdown within Superior Frontal Gyrus: {diag_counts}")
    print(f"  (manuscript reports n=34: 23 AD, 11 control -- compare against the above)")

    keep_cols = [c for c, keep, d in zip(expr.columns, region_mask, diagnosis)
                if keep and d in ("AD", "control")]
    keep_labels = {c: (1 if d == "AD" else 0)
                   for c, keep, d in zip(expr.columns, region_mask, diagnosis)
                   if keep and d in ("AD", "control")}
    out_df = expr[keep_cols]
    print(f"  final AD+control set: {out_df.shape[1]} samples")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(out_path)
    print(f"  written: {out_path}")

    labels_path = out_path.rsplit(".", 1)[0] + "_labels.csv"
    pd.DataFrame({"sample": list(keep_labels.keys()),
                 "label": list(keep_labels.values())}).to_csv(labels_path, index=False)
    print(f"  written: {labels_path}  (1=AD, 0=control, join on 'sample')")

    return out_df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    filter_samples(args.matrix, args.out)
