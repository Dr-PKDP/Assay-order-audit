# data_prep/

Scripts for acquiring and harmonizing the raw data needed for the audit's
cross-platform checks (FSQN, external re-validation). Not needed to
reproduce the original `scripts/01-17` pipeline, which uses the pre-curated
`data/raw/gene_data.csv` panel directly.

## Order

1. **`download_raw_data.py`** -- fetches the six GEO series matrices and the
   Zenodo gene panel into `data/raw/`. Pure standard-library Python, no
   dependencies. If the Zenodo files 403 (bot-protection), download them
   manually from the record page -- see the script's own output for the URL.

2. **Platform annotation** (probe -> gene mapping for GPL570 and GPL10558):
   - **`get_platform_annotation_biomart.py`** -- try this first. Live query
     to Ensembl BioMart, no download. Confirmed working for GPL570; Ensembl
     does not carry a mapping for GPL10558 specifically (only older
     HumanRef-8 v3 / HumanWG-6 v3), so it will not produce a GPL10558
     result -- that's expected, not a bug.
   - **`get_platform_annotation.R`** -- fallback for GPL10558 via
     Bioconductor's `illuminaHumanv4.db`. Needs R. Sets up a personal
     library path to avoid the common Windows "not writable" install error.
   - **`check_annotation_files.py`** -- sanity-check the two annotation
     files once downloaded (coverage, ambiguous-probe rate).

3. **`inspect_sample_metadata.py`** -- run against each raw series matrix to
   find the exact field names for tissue/region and diagnosis before
   filtering. GSE118553 and GSE5281 use different metadata conventions
   (different submitters); don't assume the same field names for both.

4. **Sample filtering** -- subset each external cohort to the samples the
   study design actually uses (frontal cortex / superior frontal gyrus,
   AD-vs-control only):
   - `filter_gse118553_samples.py` -- filters to Frontal_Cortex, splits
     AD/control (63 samples) from the held-aside AsymAD set (33 samples);
     writes a companion `..._labels.csv`.
   - `filter_gse5281_samples.py` -- filters to Superior Frontal Gyrus (34
     samples), handling GSE5281's two-batch mixed-case metadata (region
     field "Sample_characteristics_ch1 [line 5]", values differently
     capitalized across the 2006/2007 submission batches); also writes a
     `..._labels.csv`.

5. **`collapse_probes_to_genes.py`** -- probe-level expression -> gene-level,
   using the annotation from step 2. Accepts either a raw `.gz` series
   matrix or the plain CSV output of step 4's filtering. Default
   `--mode drop` discards ambiguous multi-gene probes (the defensible
   choice for new analysis); `--mode first` matches how the *original*
   pipeline handled GSE5281's GPL570 annotation, if you need numbers
   consistent with that specific prior run instead.

6. **`fsqn_harmonization.py`** -- feature-specific quantile normalization,
   mapping each external cohort's per-gene distribution onto the primary
   training panel's distribution for that gene, using `gene_data.csv` as
   the reference. Genes absent from the reference are dropped (the trained
   classifier has no weights for them regardless of harmonization).

7. **`combat_harmonization.py`** -- empirical-Bayes batch correction
   (Johnson et al. 2007, already reference [8] in the manuscript) as an
   alternative to FSQN. Uses `neuroCombat` rather than the `combat` PyPI
   package -- the latter's `ref_batch` option has a real indexing bug in
   the installed version, reproduced and confirmed before switching.
   `neuroCombat` needs one compatibility shim (`np.int` -> `int`, baked
   into the script) for a numpy deprecation the package predates; this is
   a safe, zero-behavior-change fix, not a workaround for a logic bug.
   Uses batch identity only, **no diagnosis covariate** -- an earlier
   version passed each cohort's true label as a covariate to "preserve"
   disease-correlated signal, which turned out to leak the target
   cohort's own labels into its own harmonized features (caught via an
   adversarial test: a synthetic external cohort with zero true signal
   still scored well above chance after that version's transform --
   impossible without leakage; a paired comparison across 10 seeds
   confirmed it systematically, p<0.0001). The current version passes
   the same adversarial test at chance level (p=0.12 vs 0.5, correctly
   non-significant). Run it the same way as FSQN.

## Comparing all three (baseline / FSQN / ComBat)

`analysis/score_fsqn_vs_baseline.py` accepts an optional `--external-combat`
argument for a three-way comparison in one run -- see `analysis/` for the
full command.

## What this does NOT reproduce exactly

The *original* `scripts/03-04` and `scripts/13` used the old NCBI
`.annot.gz` probe-annotation format, which Thermo Fisher and Illumina have
since discontinued as a direct download. This folder's annotation sources
(BioMart, Bioconductor) are different, so re-running `collapse_probes_to_genes.py`
here will not bit-for-bit reproduce the exact gene counts the original
pipeline got -- expect close but not identical coverage. This is disclosed
in `analysis/score_fsqn_vs_baseline.py`'s docstring and in the manuscript's
Section 6.1.
