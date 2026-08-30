"""Process GSE134379 (Banner Sun Health, Illumina 450K, middle temporal gyrus)
as a new independent Stage-2 methylation external-validation cohort, distinct
from the GSE80970 (Mount Sinai) training/CV cohort.

Steps:
1. Build a cg-probe -> gene-symbol map from the GPL13534 manifest.
2. Restrict to probes whose gene is present in the existing 16,193-gene
   Stage-2 feature panel (methylation_data.csv "Methy_<gene>" columns).
3. Stream the large per-region-per-diagnosis processed beta-value files
   (GSE134379_processedSamples_mtgAD.txt.gz / mtgND.txt.gz) with awk-style
   filtering so the huge raw files never enter memory whole.
4. Collapse probe-level beta values to gene-level (mean of probes per gene),
   producing a sample x gene matrix aligned to the existing Methy_<gene>
   naming convention, with an AD/ND diagnosis label.
"""
import gzip
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(exist_ok=True, parents=True)

# --- 1. Target gene panel (existing Stage-2 methylation feature set) ---
meth_cols = pd.read_csv(RAW / "methylation_data.csv", nrows=0).columns.tolist()
target_genes = set(c[len("Methy_"):] for c in meth_cols if c.startswith("Methy_"))
print(f"Target gene panel size (existing Stage-2 features): {len(target_genes)}")

# --- 2. Build probe -> gene map from GPL13534 manifest, restricted to target genes ---
print("Parsing GPL13534 manifest for probe -> gene mapping...")
manifest = pd.read_csv(
    RAW / "GPL13534_manifest.csv.gz",
    skiprows=7,  # skip Illumina header block down to the [Assay] column header row
    usecols=["Name", "UCSC_RefGene_Name"],
    dtype=str,
    low_memory=False,
)
manifest = manifest.dropna(subset=["Name"])


def first_gene(x):
    if pd.isna(x) or x == "":
        return None
    return str(x).split(";")[0]


manifest["gene"] = manifest["UCSC_RefGene_Name"].apply(first_gene)
probe2gene = dict(zip(manifest["Name"], manifest["gene"]))
# Restrict to probes whose gene is in our existing target panel
probe2gene = {p: g for p, g in probe2gene.items() if g in target_genes}
print(f"Probes mapping to target-panel genes: {len(probe2gene)}")

probe_set_file = RAW / "gse134379_target_probes.txt"
with open(probe_set_file, "w") as f:
    for p in probe2gene:
        f.write(p + "\n")

# --- 3. Stream-filter the huge beta files with awk: keep only rows whose
#         probe id is in our target probe set, and keep only Beta columns
#         (drop the interleaved Detection P Value columns). ---
awk_script = r'''
BEGIN{FS="\t"; OFS="\t"}
NR==FNR{keep[$1]=1; next}
FNR==1{
  # header row for the data file: build reduced header (probe placeholder + beta col names)
  printf "probe";
  for(i=1;i<=NF;i+=2){printf "\t%s", $i}
  print "";
  next
}
($1 in keep){
  printf "%s", $1;
  for(i=2;i<=NF;i+=2){printf "\t%s", $i}
  print ""
}
'''
awk_script_path = RAW / "filter_beta.awk"
awk_script_path.write_text(awk_script)

for tag in ["mtgAD", "mtgND"]:
    src = RAW / f"GSE134379_processedSamples_{tag}.txt.gz"
    dst = RAW / f"GSE134379_{tag}_filtered.tsv"
    print(f"Streaming filter for {tag} ...")
    cmd = f"zcat {src} | awk -f {awk_script_path} {probe_set_file} -"
    with open(dst, "w") as out:
        subprocess.run(["bash", "-c", cmd], stdout=out, check=True)
    print(f"  -> wrote {dst}")

print("Done filtering. Loading filtered matrices...")

# --- 4. Load filtered matrices, map probe->gene, average per gene, build
#         final sample x gene matrices with diagnosis labels. ---
def load_and_collapse(tag, label):
    df = pd.read_csv(RAW / f"GSE134379_{tag}_filtered.tsv", sep="\t")
    df = df.rename(columns={df.columns[0]: "probe"})
    df["gene"] = df["probe"].map(probe2gene)
    df = df.dropna(subset=["gene"])
    sample_cols = [c for c in df.columns if c not in ("probe", "gene")]
    gene_level = df.groupby("gene")[sample_cols].mean()
    gene_level = gene_level.T  # samples x genes
    gene_level.columns = [f"Methy_{g}" for g in gene_level.columns]
    gene_level.insert(0, "label", label)
    gene_level.insert(0, "sample", [f"{tag}_{i}" for i in range(len(gene_level))])
    return gene_level


ad_df = load_and_collapse("mtgAD", 1)
nd_df = load_and_collapse("mtgND", 0)
print(f"AD samples: {ad_df.shape}, ND samples: {nd_df.shape}")

combined = pd.concat([ad_df, nd_df], axis=0, ignore_index=True)
print(f"Combined GSE134379 MTG methylation matrix: {combined.shape}")

out_path = PROC / "GSE134379_mtg_methylation_gene_level.csv"
combined.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
