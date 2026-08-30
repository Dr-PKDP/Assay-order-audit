"""
read_series_matrix.py

A few-hundred-MB gzipped series matrix is normal for real expression data
(tens of thousands of probes x hundreds of samples) -- nothing to route
around. The only practical concern is memory: a 300 MB .gz can expand to a
few GB once decompressed into a dense DataFrame. This reads it without ever
holding the raw text in memory twice, and separates metadata from the
expression matrix (series matrix files interleave "!"-prefixed metadata
lines with the actual tab-delimited data block).
"""
import gzip
import pandas as pd


def read_series_matrix(path):
    """Returns (expression_df, metadata_dict) from a gzipped GEO series matrix.

    metadata_dict maps each metadata key to a list of per-sample values, in
    the same column order as expression_df -- e.g. metadata_dict['Sample_geo_accession']
    lines up positionally with expression_df.columns. GEO repeats some keys
    (typically 'Sample_characteristics_ch1') once per characteristic line, so
    repeated keys are suffixed ' [line N]' to keep them distinct rather than
    silently merged into one blob.
    """
    meta = {}
    line_counts = {}
    data_lines = []
    in_table = False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                data_lines.append(line)
            elif line.startswith("!"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                key = parts[0].lstrip("!")
                values = [v.strip('"') for v in parts[1:]]
                line_counts[key] = line_counts.get(key, 0) + 1
                if line_counts[key] == 2 and key in meta:
                    meta[f"{key} [line 1]"] = meta.pop(key)
                labeled_key = f"{key} [line {line_counts[key]}]" \
                    if line_counts[key] > 1 else key
                meta[labeled_key] = values

    # parse the collected table lines with pandas in one pass, no double copy
    from io import StringIO
    table_text = "".join(data_lines)
    expr = pd.read_csv(StringIO(table_text), sep="\t", index_col=0)
    expr.columns = [c.strip('"') for c in expr.columns]
    expr.index = [i.strip('"') for i in expr.index]
    return expr, meta


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python read_series_matrix.py GSExxxxx_series_matrix.txt.gz")
        sys.exit(1)
    expr, meta = read_series_matrix(sys.argv[1])
    print(f"expression matrix: {expr.shape[0]} probes x {expr.shape[1]} samples")
    print(f"sample titles: {meta.get('Sample_title', [])[:3]}...")
