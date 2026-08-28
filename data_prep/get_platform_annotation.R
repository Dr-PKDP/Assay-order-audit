# get_platform_annotation.R
#
# GPL10558 (Illumina HumanHT-12 V4) annotation via Bioconductor. Ensembl
# BioMart no longer carries a mapping for this specific array version (only
# older HumanRef-8 v3 / HumanWG-6 v3 remain there), so this is the route for
# it. GPL570 is skipped here since the BioMart script already produced that
# one -- see get_platform_annotation_biomart.py.
#
# ---- personal library setup (fixes the "not writable" install error) ----
# R's default install location under Program Files needs admin rights, which
# most Windows accounts don't have day to day. This gives R a folder you
# already own instead, which is the standard fix, not just a workaround.
user_lib <- file.path(
    Sys.getenv("USERPROFILE"), "Documents", "R", "win-library",
    paste(R.version$major, strsplit(R.version$minor, "\\.")[[1]][1], sep = ".")
)
if (!dir.exists(user_lib)) {
    dir.create(user_lib, recursive = TRUE)
    cat("created personal R library at:", user_lib, "\n")
}
.libPaths(c(user_lib, .libPaths()))

# ---- install (only runs the parts not already installed) ----------------
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", lib = user_lib, repos = "https://cloud.r-project.org")
BiocManager::install("illuminaHumanv4.db", update = FALSE, ask = FALSE, lib = user_lib)

library(illuminaHumanv4.db)
library(AnnotationDbi)

# ---- GPL10558: probe ID -> gene symbol -----------------------------------
gpl10558_map <- AnnotationDbi::select(
    illuminaHumanv4.db,
    keys = keys(illuminaHumanv4.db, keytype = "PROBEID"),
    columns = c("SYMBOL", "GENENAME"),
    keytype = "PROBEID"
)

out_dir <- "data/raw"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
write.table(gpl10558_map, file.path(out_dir, "GPL10558_annotation.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("GPL10558:", nrow(gpl10558_map), "probe-to-gene rows written to",
    file.path(out_dir, "GPL10558_annotation.tsv"), "\n")
