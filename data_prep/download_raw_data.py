"""
download_raw_data.py

Python-native replacement for download_raw_data.sh -- that one needed bash
(Git Bash / WSL), which isn't available in a plain Windows Command Prompt.
This uses only urllib, which is part of the Python standard library, so it
needs no extra install.

Fetches the six GEO series matrices, the GSE134379 supplementary files, and
the Zenodo gene panel into data/raw/, under the exact filenames the rest of
this pipeline expects.

Confidence note: same as before -- the GSE folder-path pattern (drop the
accession's last 3 digits for the containing folder) is one I verified
against several worked examples and NCBI's own documented convention, but I
could not test these specific URLs myself (my environment is blocked from
reaching ncbi.nlm.nih.gov and zenodo.org). If any single file 404s, the
GEO accession page's "Series Matrix File(s)" / "Supplementary file" link is
the reliable fallback for that one file -- see FALLBACK_PAGES at the bottom.

Usage:
    python download_raw_data.py
"""
import os
import urllib.request
import urllib.error

OUT_DIR = "data/raw"

FILES = [
    ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE33nnn/GSE33000/matrix/GSE33000_series_matrix.txt.gz",
     "GSE33000_series_matrix.txt.gz"),
    ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE44nnn/GSE44770/matrix/GSE44770_series_matrix.txt.gz",
     "GSE44770_series_matrix.txt.gz"),
    ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE80nnn/GSE80970/matrix/GSE80970_series_matrix.txt.gz",
     "GSE80970_series_matrix.txt.gz"),
    ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE118nnn/GSE118553/matrix/GSE118553_series_matrix.txt.gz",
     "GSE118553_series_matrix.txt.gz"),
    ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE5nnn/GSE5281/matrix/GSE5281_series_matrix.txt.gz",
     "GSE5281_series_matrix.txt.gz"),
    ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE134nnn/GSE134379/suppl/GSE134379_processedSamples_mtgAD.txt.gz",
     "GSE134379_processedSamples_mtgAD.txt.gz"),
    ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE134nnn/GSE134379/suppl/GSE134379_processedSamples_mtgND.txt.gz",
     "GSE134379_processedSamples_mtgND.txt.gz"),
    ("https://zenodo.org/records/13933763/files/gene_data.csv",
     "gene_data.csv"),
    ("https://zenodo.org/records/13933763/files/methylation_data.csv",
     "methylation_data.csv"),
]

FALLBACK_PAGES = {
    "GSE33000": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE33000",
    "GSE44770": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE44770",
    "GSE80970": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE80970",
    "GSE118553": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE118553",
    "GSE5281": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE5281",
    "GSE134379": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE134379",
    "gene_data.csv": "https://doi.org/10.5281/zenodo.13933763",
    "methylation_data.csv": "https://doi.org/10.5281/zenodo.13933763",
}


def fallback_key(filename):
    for key in FALLBACK_PAGES:
        if key.split("_")[0] in filename or key == filename:
            return key
    return None


def fetch(url, out_path):
    if os.path.exists(out_path):
        print(f"  already have {out_path}, skipping")
        return True
    print(f"  fetching {out_path} ...")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://zenodo.org/",
        })
        with urllib.request.urlopen(req, timeout=120) as resp, open(out_path, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"    done ({size_mb:.1f} MB)")
        return True
    except urllib.error.HTTPError as e:
        print(f"    FAILED: HTTP {e.code} -- {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"    FAILED: {e.reason}")
        return False


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    failures = []
    for url, filename in FILES:
        out_path = os.path.join(OUT_DIR, filename)
        ok = fetch(url, out_path)
        if not ok:
            failures.append(filename)

    print()
    if failures:
        print(f"{len(failures)} file(s) failed to auto-download:")
        for fn in failures:
            key = fallback_key(fn)
            page = FALLBACK_PAGES.get(key, "(see data/README.md)")
            print(f"  {fn}")
            print(f"    -> get it manually from: {page}")
    else:
        print("All files downloaded successfully.")
