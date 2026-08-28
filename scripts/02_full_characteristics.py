"""Dump ALL characteristic rows (unique value counts) for the mRNA series
GSE33000 and GSE44770 to locate the diagnosis (AD/HD/control) label."""
import gzip
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repo root
from collections import Counter

RAW = ROOT / "data" / "raw"

def full_dump(gse):
    path = RAW / f"{gse}_series_matrix.txt.gz"
    print(f"\n=== {gse} : all characteristic/description rows ===")
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_characteristics_ch1") or line.startswith("!Sample_description") or line.startswith("!Sample_title"):
                vals = line.strip().split("\t")
                tag = vals[0]
                data = [v.strip('"') for v in vals[1:]]
                c = Counter([d.split(":")[0].strip() if ":" in d else d for d in data])
                print(f"  {tag}: unique-key-counts={dict(c)}")
                # show a few full unique values
                uniq = Counter(data)
                sample_uniques = list(uniq.items())[:8]
                print(f"    sample unique values: {sample_uniques}")
            if line.startswith("!series_matrix_table_begin"):
                break

for gse in ["GSE33000", "GSE44770"]:
    full_dump(gse)
