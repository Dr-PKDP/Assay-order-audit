"""
inspect_sample_metadata.py

GSE118553 profiles four brain regions and three diagnostic categories, but
this study only needs frontal-cortex, AD-vs-control samples (63 total).
Rather than guess which metadata field/value encodes region and diagnosis
(this varies a lot between GEO submissions), this prints every metadata key
and its distinct values so the right filter can be written precisely instead
of guessed.

Usage:
    python inspect_sample_metadata.py data/raw/GSE118553_series_matrix.txt.gz
"""
import sys
import gzip
from collections import defaultdict


def inspect(path):
    meta = defaultdict(list)
    line_counts = defaultdict(int)
    sample_ids = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            key = parts[0].lstrip("!")
            values = [v.strip('"') for v in parts[1:]]
            if key == "Sample_geo_accession":
                sample_ids = values
            # GEO repeats the same key once per characteristic line (e.g. region
            # on one Sample_characteristics_ch1 line, diagnosis on the next) --
            # keep these separate rather than merging, or region/diagnosis values
            # become indistinguishable in the printout.
            line_counts[key] += 1
            labeled_key = f"{key} [line {line_counts[key]}]" if line_counts[key] > 1 \
                else key
            # re-check: if this is the first time we've seen this key more than
            # once, retroactively relabel the first occurrence too
            if line_counts[key] == 2 and key in meta:
                meta[f"{key} [line 1]"] = meta.pop(key)
            meta[labeled_key if line_counts[key] > 1 else key] = values

    print(f"{len(sample_ids)} samples found\n")
    print("Metadata keys and up to 15 distinct values each:")
    print("=" * 70)
    for key, values in meta.items():
        if key in ("Sample_geo_accession",):
            continue
        distinct = sorted(set(values))
        if len(distinct) <= 1:
            continue  # constant across all samples, not useful for filtering
        print(f"\n{key}  ({len(distinct)} distinct values)")
        for v in distinct[:15]:
            count = values.count(v)
            print(f"    {v!r}  (n={count})")
        if len(distinct) > 15:
            print(f"    ... and {len(distinct) - 15} more")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_sample_metadata.py <series_matrix.txt.gz>")
        sys.exit(1)
    inspect(sys.argv[1])
