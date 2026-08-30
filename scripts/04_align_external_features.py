"""Map GSE118553 probes -> gene symbols, subset to Frontal_Cortex (closest
region to the prefrontal-cortex training data), aggregate probe->gene by mean,
and intersect with the 16193-gene panel used in the primary Zenodo cohort."""
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

# --- 1. Probe -> gene symbol map from GPL10558 annotation ---
probe2gene = {}
with gzip.open(RAW / "GPL10558.annot.gz", "rt", errors="replace") as f:
    header = None
    in_table = False
    for line in f:
        line = line.rstrip("\n")
        if line.startswith("!platform_table_begin"):
            in_table = True
            continue
        if line.startswith("!platform_table_end"):
            break
        if not in_table:
            continue
        parts = line.split("\t")
        if header is None:
            header = parts
            continue
        pid = parts[0]
        sym = parts[2] if len(parts) > 2 else ""
        if sym:
            probe2gene[pid] = sym

print(f"Probe->gene mappings loaded: {len(probe2gene)}")

# --- 2. Load expression + metadata ---
expr = pd.read_parquet(PROC / "GSE118553_raw_expression.parquet")
meta = pd.read_csv(PROC / "GSE118553_sample_metadata_full.csv", index_col=0)
print("Expr shape:", expr.shape, "Meta shape:", meta.shape)

# --- 3. Subset to Frontal_Cortex, and to AD/control only (drop AsymAD for the
#         primary binary validation set; save AsymAD subset separately for a
#         bonus ambiguous-case analysis) ---
fc_mask = meta["tissue"] == "Frontal_Cortex"
fc_meta = meta[fc_mask]
print("\nFrontal_Cortex disease-state counts:\n", fc_meta["disease state"].value_counts())

primary_mask = fc_meta["disease state"].isin(["AD", "control"])
primary_meta = fc_meta[primary_mask]
asymad_meta = fc_meta[fc_meta["disease state"] == "AsymAD"]
print(f"\nPrimary (AD/control) Frontal_Cortex samples: {len(primary_meta)}")
print(f"AsymAD Frontal_Cortex samples (bonus set): {len(asymad_meta)}")

expr_fc = expr[fc_meta.index]

# --- 4. Collapse probes -> gene symbol (mean across probes mapping to the
#         same gene), restrict to probes with a known symbol ---
gene_series = pd.Series({pid: probe2gene.get(pid, "") for pid in expr_fc.index})
mapped = gene_series[gene_series != ""]
expr_mapped = expr_fc.loc[mapped.index]
expr_mapped["__gene__"] = mapped.values
gene_expr = expr_mapped.groupby("__gene__").mean()
print("\nGene-level expression matrix (Frontal_Cortex, all disease states):", gene_expr.shape)

# --- 5. Intersect with the 16193-gene panel from the primary cohort ---
primary_genes = pd.read_csv(PROC / "primary_gene_feature_list.csv", header=None)[0].tolist()
common_genes = sorted(set(primary_genes) & set(gene_expr.index))
print(f"\nGenes in primary panel: {len(primary_genes)}")
print(f"Genes overlapping with GSE118553 (Frontal_Cortex): {len(common_genes)}")

external_primary = gene_expr.loc[common_genes, primary_meta.index].T
external_primary["label"] = (primary_meta["disease state"] == "AD").astype(int).values
external_primary.insert(0, "sample", external_primary.index)
external_primary.to_csv(PROC / "external_validation_AD_control.csv", index=False)

external_asymad = gene_expr.loc[common_genes, asymad_meta.index].T
external_asymad.insert(0, "sample", external_asymad.index)
external_asymad.to_csv(PROC / "external_validation_AsymAD_bonus.csv", index=False)

print(f"\nSaved external_validation_AD_control.csv: {external_primary.shape}")
print(f"Saved external_validation_AsymAD_bonus.csv: {external_asymad.shape}")
print("\nLabel balance in external AD/control set:", external_primary['label'].value_counts().to_dict())
