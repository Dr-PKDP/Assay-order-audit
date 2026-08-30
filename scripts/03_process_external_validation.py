"""Process GSE118553 (Illumina HT-12 v4, GPL10558) as an independent external
validation cohort for the mRNA-only Stage-1 classifier.

Steps:
1. Parse series matrix -> expression matrix (probes x samples) + sample metadata
   (disease state, tissue region).
2. Parse GPL10558 annotation -> probe ID -> gene symbol map.
3. Collapse probes to genes (mean of probes per gene).
4. Subset samples to a single tissue region for homogeneity.
5. Intersect features with the 16193-gene panel used in the Zenodo AE-Trans
   gene_data.csv, save an aligned external-validation matrix + labels.
"""
import gzip
import re
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

def parse_series_matrix(gse):
    path = RAW / f"{gse}_series_matrix.txt.gz"
    sample_ids = None
    meta_rows = {}  # tag -> list of raw strings per sample
    data_rows = []
    probe_ids = []
    in_table = False
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [v.strip('"') for v in line.split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [v.strip('"') for v in line.split("\t")[1:]]
                # key by the characteristic name (before ':')
                if vals:
                    key = vals[0].split(":")[0].strip() if ":" in vals[0] else f"char_{len(meta_rows)}"
                    meta_rows[key] = vals
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            elif line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            elif in_table:
                if line.startswith('"ID_REF"'):
                    continue
                parts = line.split("\t")
                probe_ids.append(parts[0].strip('"'))
                data_rows.append([float(x) if x not in ("", "null", "NA") else np.nan for x in parts[1:]])
    expr = pd.DataFrame(data_rows, index=probe_ids, columns=sample_ids)
    meta = pd.DataFrame({k: [v.split(":", 1)[1].strip() if ":" in v else v for v in vals] for k, vals in meta_rows.items()}, index=sample_ids)
    return expr, meta

print("Parsing GSE118553 series matrix (this can take a minute for a 401-sample x 47323-probe matrix)...")
expr, meta = parse_series_matrix("GSE118553")
print("Expression matrix shape:", expr.shape)
print("Metadata columns:", meta.columns.tolist())
print(meta.head())
print("\nTissue value counts:\n", meta["tissue"].value_counts() if "tissue" in meta.columns else "no tissue col")
print("\nDisease state value counts:\n", meta["disease state"].value_counts() if "disease state" in meta.columns else "no disease state col")

meta.to_csv(OUT / "GSE118553_sample_metadata_full.csv")
expr.to_parquet(OUT / "GSE118553_raw_expression.parquet")
print("\nSaved raw parsed matrix + metadata.")
