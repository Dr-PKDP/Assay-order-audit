"""
get_platform_annotation_biomart.py

Live query to Ensembl BioMart for probe -> gene mappings, in place of the
huge (42-68 GB) GEO platform family files and the now-broken manufacturer
support pages. No file download at all -- this queries Ensembl's servers
directly for exactly the probes present in your data.

Confirmed working BioMart attributes for these two platforms:
    GPL570   (Affymetrix HG-U133 Plus 2) -> "affy_hg_u133_plus_2"
    GPL10558 (Illumina HumanHT-12 V4)    -> "illumina_humanht_12_v4"

Known limitation: BioMart's probe-ID coverage is not 100% -- historically
reported at roughly 75-90% depending on platform and Ensembl release, since
some probes don't map cleanly to a current Ensembl gene model. Cross-check
against the Bioconductor annotation packages (hgu133plus2.db /
illuminaHumanv4.db, see get_platform_annotation.R) if you need the residual
unmapped probes; that route uses manufacturer-curated mappings rather than
Ensembl's own re-alignment and tends to have fuller coverage.

Usage:
    python get_platform_annotation_biomart.py
"""
from pybiomart import Server
import os

server = Server(host="http://www.ensembl.org")
dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"]


def fetch_annotation(probe_attribute, out_path):
    print(f"querying BioMart for {probe_attribute} ...")
    result = dataset.query(
        attributes=[probe_attribute, "external_gene_name", "ensembl_gene_id"],
    )
    result = result.dropna(subset=[result.columns[0]])
    result = result.drop_duplicates(subset=[result.columns[0]])
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.to_csv(out_path, sep="\t", index=False)
    print(f"  {len(result)} probe-to-gene rows written to {out_path}")
    return result


if __name__ == "__main__":
    fetch_annotation("affy_hg_u133_plus_2", "data/raw/GPL570_annotation.tsv")
    fetch_annotation("illumina_humanht_12_v4", "data/raw/GPL10558_annotation.tsv")
