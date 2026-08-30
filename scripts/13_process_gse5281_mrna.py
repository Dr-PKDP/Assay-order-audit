"""Process GSE5281 (Affymetrix HG-U133 Plus 2 / GPL570, laser-capture-microdissected
neurons from 6 brain regions) as a second, independent external validation cohort
for the Stage-1 mRNA classifier, distinct from GSE118553.

Region choice: superior frontal gyrus (SFG) -- the closest match among GSE5281's
six regions to the prefrontal/frontal cortex tissue used for primary training
(GSE33000/GSE44770) and for the first external validation cohort (GSE118553,
frontal cortex).
"""
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def parse_series_matrix(gse):
    path = RAW / f"{gse}_series_matrix.txt.gz"
    sample_ids = None
    titles = None
    meta_rows = {}
    data_rows = []
    probe_ids = []
    in_table = False
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [v.strip('"') for v in line.split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                titles = [v.strip('"') for v in line.split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [v.strip('"') for v in line.split("\t")[1:]]
                if vals:
                    key = vals[0].split(":")[0].strip() if ":" in vals[0] else f"char_{len(meta_rows)}"
                    meta_rows.setdefault(key, []).append(vals)
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
    # some characteristic keys repeat across multiple !Sample_characteristics_ch1
    # lines (different rows use different keys per key length); flatten by taking
    # the first occurrence list matching len(sample_ids)
    meta = {}
    for k, list_of_vals in meta_rows.items():
        for vals in list_of_vals:
            if len(vals) == len(sample_ids):
                meta[k] = [v.split(":", 1)[1].strip() if ":" in v else v for v in vals]
                break
    meta_df = pd.DataFrame(meta, index=sample_ids)
    meta_df["title"] = titles
    return expr, meta_df


print("Parsing GSE5281 series matrix (161 samples x ~54675 probes)...")
expr, meta = parse_series_matrix("GSE5281")
print("Expression matrix shape:", expr.shape)
print("Metadata columns:", meta.columns.tolist())
print(meta["Disease State"].value_counts() if "Disease State" in meta.columns else "no Disease State col")
print(meta["Organ Region"].value_counts() if "Organ Region" in meta.columns else "no Organ Region col")

meta.to_csv(PROC / "GSE5281_sample_metadata_full.csv")
expr.to_parquet(PROC / "GSE5281_raw_expression.parquet")

# --- Subset to SFG (superior frontal gyrus) region, AD vs normal ---
region_col = "Organ Region"
disease_col = "Disease State"
sfg_mask = meta[region_col].str.contains("Superior Frontal", case=False, na=False)
sfg_meta = meta[sfg_mask]
print(f"\nSFG samples: {len(sfg_meta)}")
print(sfg_meta[disease_col].value_counts())

label_map = {"normal": 0, "Alzheimer's Disease": 1}
sfg_meta = sfg_meta[sfg_meta[disease_col].isin(label_map.keys())].copy()
sfg_meta["label"] = sfg_meta[disease_col].map(label_map)
print(f"\nFinal SFG AD/control samples: {len(sfg_meta)} -> {sfg_meta['label'].value_counts().to_dict()}")

expr_sfg = expr[sfg_meta.index]

# --- Probe -> gene symbol map from GPL570 annotation (NCBI .annot format) ---
probe2gene = {}
with gzip.open(RAW / "GPL570.annot.gz", "rt", errors="replace") as f:
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

print(f"\nProbe->gene mappings loaded (GPL570): {len(probe2gene)}")

gene_series = pd.Series({pid: probe2gene.get(pid, "") for pid in expr_sfg.index})
mapped = gene_series[gene_series != ""]
expr_mapped = expr_sfg.loc[mapped.index].copy()
expr_mapped["__gene__"] = mapped.values
# some probes map to multi-gene strings like "GENE1///GENE2"; take first
expr_mapped["__gene__"] = expr_mapped["__gene__"].apply(lambda x: x.split("///")[0].strip())
gene_expr = expr_mapped.groupby("__gene__").mean(numeric_only=True)
print("\nGene-level SFG expression matrix:", gene_expr.shape)

# --- Intersect with primary 16193-gene panel ---
primary_genes = pd.read_csv(PROC / "primary_gene_feature_list.csv", header=None)[0].tolist()
common_genes = sorted(set(primary_genes) & set(gene_expr.index))
print(f"\nGenes in primary panel: {len(primary_genes)}")
print(f"Genes overlapping with GSE5281 SFG: {len(common_genes)}")

external = gene_expr.loc[common_genes, sfg_meta.index].T
external["label"] = sfg_meta["label"].values
external.insert(0, "sample", external.index)
out_path = PROC / "external_validation2_GSE5281_SFG_AD_control.csv"
external.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}, shape={external.shape}")
print("Label balance:", external["label"].value_counts().to_dict())
