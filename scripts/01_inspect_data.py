"""Inspect downloaded GEO series matrix files: extract sample metadata,
platform, and dimensions without loading full expression matrices into memory
unnecessarily. Prints a summary per dataset."""
import gzip
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root

RAW = ROOT / "data" / "raw"

def inspect(gse):
    path = RAW / f"{gse}_series_matrix.txt.gz"
    meta = {}
    n_samples = None
    n_probes = 0
    header_lines = []
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("!Series_title"):
                meta["title"] = line.strip().split("\t", 1)[-1]
            elif line.startswith("!Series_summary") and "summary" not in meta:
                meta["summary"] = line.strip().split("\t", 1)[-1][:300]
            elif line.startswith("!Sample_geo_accession"):
                vals = line.strip().split("\t")[1:]
                n_samples = len(vals)
            elif line.startswith("!Sample_characteristics_ch1") or line.startswith("!Sample_description"):
                header_lines.append(line.strip())
            elif line.startswith("!Platform_title"):
                meta["platform"] = line.strip().split("\t", 1)[-1]
            elif line.startswith("!series_matrix_table_begin"):
                # count remaining data rows quickly (probes)
                next(f)  # header row (sample ids)
                for row in f:
                    if row.startswith("!series_matrix_table_end"):
                        break
                    n_probes += 1
                break
    meta["n_samples"] = n_samples
    meta["n_probes"] = n_probes
    print(f"\n=== {gse} ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print(f"  characteristic rows found: {len(header_lines)}")
    for hl in header_lines[:6]:
        vals = hl.split("\t")
        tag = vals[0]
        sample_vals = vals[1:6]
        print(f"    {tag[:40]:40s} -> {sample_vals}")

for gse in ["GSE33000", "GSE44770", "GSE80970", "GSE118553", "GSE29378"]:
    inspect(gse)
