# Same-Label Partner Averaging Inflates Unpaired Multi-Omics Fusion Estimates

Reproducible code and results for:

> **Same-Label Partner Averaging Inflates Unpaired Multi-Omics Fusion Estimates: A Pairing-Arithmetic Audit Applied to Alzheimer's Disease**

## Overview

Reports of multi-omics fusion for disease classification are proliferating faster than the tools to check them. A specific convention has become standard for estimating what a second molecular layer adds when no cohort has both layers measured on the same patients: pair each sample from one modality with samples from the other modality that share its diagnostic label, and average several such partners together. This repository provides three checks that test whether that convention is trustworthy — a pairing-arithmetic inflation identity, a combination-rule-free attainability bound, and a single-partner re-simulation with a significance test — and applies all three to an open-data Alzheimer's disease case study built entirely on public GEO/Zenodo data (no controlled-access repository, no data use agreement required).

**The standard convention fails all three checks on this pipeline.** The reported same-label fusion accuracy (93.10% on the hardest confidence subset, *k* = 20 averaged partners) sits above the attainability bound's own 95% confidence interval (71.60%, upper limit 77.00%), and the inflation tracks the pairing-arithmetic identity's sqrt(k) prediction closely across *k* = 1-50. A single-partner (*k* = 1) re-simulation returns an estimate close to the mRNA-only baseline (68.25% vs. 68.97%, 95% bootstrap CI -6.45 to +5.39 percentage points) rather than reproducing the large *k* = 20 gain. Neither feature-specific quantile normalization (FSQN) nor ComBat rescues cross-platform external validation, and a true leave-one-series-out retrain confirms the internal classifier is not relying on a series-identity shortcut.

None of this is a claim that multi-omics fusion cannot help diagnosis. It is a claim that, at the assay strength and sample size available here, the standard method for claiming it does help does not survive its own audit, and the same audit is available at no additional data cost to any study making a comparable claim.

## Repository Structure

```
.
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT (code); see data/README.md for data licenses
├── scripts/                    # Numbered data-processing and model-training pipeline
├── data_prep/                  # Raw data acquisition, annotation, filtering, FSQN/ComBat harmonization
├── analysis/                   # The three-check audit: bound + CI, k=1 re-simulation,
│                                #   k=20 seed sensitivity, significance/multiple-comparison testing,
│                                #   LOSO retrain, FSQN/ComBat-vs-baseline scoring
├── figures/                    # Scripts that build the manuscript's 6 figures
├── data/
│   └── README.md               # How to download/regenerate raw and processed data (not tracked in git)
└── results/                    # JSON/CSV outputs from every analysis step (tracked in git)
    └── figures/                # Current figures (PNG + PDF)
```

`data/raw/` and `data/processed/` are **not** tracked in this repository (see `.gitignore`) because the combined size exceeds typical code-hosting limits. `data/README.md` gives exact instructions to regenerate them from the public sources cited in the manuscript's Data Availability statement.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Reproducing the Pipeline

Run in this order. Steps 1-15 (`scripts/`) build the underlying classifiers and process every cohort; the `analysis/` scripts then run the audit on top of their output.

### 1. Data processing and classifiers (`scripts/`)

| Step | Script | Purpose |
|---|---|---|
| 1 | `01_inspect_data.py` | Inspect raw GEO series matrices |
| 2 | `02_full_characteristics.py` | Locate the AD/control label in GSE33000/GSE44770 metadata |
| 3 | `03_process_external_validation.py` | Process GSE118553 as external validation cohort #1 |
| 4 | `04_align_external_features.py` | Map GSE118553 probes to genes, align to the primary panel |
| 5 | `05_stage1_mrna_model.py` | Train/evaluate the Stage-1 bootstrap-ensembled mRNA classifier |
| 6 | `06_stage2_methylation_model.py` | Train/evaluate the Stage-2 methylation classifier |
| 7 | `07_fusion_and_cost_pareto.py` | Same-label / random-pairing fusion simulation (Table 7's raw output) |
| 9 | `09_statistical_analysis.py` | Bootstrap CIs, DeLong, McNemar tests across cohorts and subsets |
| 12 | `12_process_gse134379_methylation.py` | Process GSE134379 as Stage-2 external validation |
| 13 | `13_process_gse5281_mrna.py` | Process GSE5281 as Stage-1 external validation cohort #2 |
| 14 | `14_new_external_validations.py` | Score trained models on GSE5281 and GSE134379 |
| 15 | `15_ablation_study.py` | Ensemble size / top-K / class balancing / fusion meta-model ablation |

(Step numbering follows the order scripts were originally written in; a few intermediate steps were superseded during development and removed rather than renumbered — see the `analysis/` section below for their replacements.)

### 2. Data preparation for the audit (`data_prep/`)

Needed only for the cross-platform harmonization checks (FSQN, ComBat, external-cohort re-scoring). See `data_prep/README.md` for the exact command sequence: `download_raw_data.py` then `get_platform_annotation_biomart.py` (plus `get_platform_annotation.R` as a fallback for GPL10558), then `inspect_sample_metadata.py`, then `filter_gse118553_samples.py` / `filter_gse5281_samples.py`, then `collapse_probes_to_genes.py`, then `fsqn_harmonization.py` / `combat_harmonization.py`.

### 3. The audit itself (`analysis/`)

Run in this order; later scripts depend on earlier ones' output:

1. `bayes_error_corrected_bound.py`: the attainability bound with bootstrap 95% CI, using posterior Bayes error (Table 8)
2. `bayes_error_corrected_bound_table11.py`: the informal, observed-accuracy version of the bound (Table A10)
3. `k_sweep.py`: empirical partner-count sweep, *k* = 1-50, against the sqrt(k) prediction (Table 9)
4. `k20_seed_sensitivity.py`: canonical *k* = 20 partner-draw seed-sensitivity check (Section 5.4); sweeps 30 seeds of the standard convention's same-label fusion, quantifying how much the headline ceiling figure moves under an equally arbitrary alternative seed. Fully deterministic, verified bit-for-bit identical across repeated runs.
5. `seed_aware_inference.py`: the seed-averaged *k* = 1 bootstrap CI reported in the manuscript's main text (Section 5.5)
6. `k1_learned_model_all_thresholds.py`: canonical source for Table 10 and Table A11, the *k* = 1 single-partner construction and its MDE/TOST equivalence testing, at every confidence-margin threshold, using the same learned two-feature meta-model as `k_sweep.py` and `seed_aware_inference.py`. (An earlier version of this analysis used simple unweighted logit averaging instead of the learned meta-model, inconsistent with every other value of *k* in this pipeline; this script is the corrected, consistent version and should be used for anything *k* = 1 related.)
7. `regenerate_appendix_seed_arrays.py`: the per-seed accuracy and DeLong *p*-value distributions behind Appendix Figure A1, using the same corrected learned-meta-model method as (6)
8. `multiple_comparison_correction_v2.py`: Holm-Bonferroni and Benjamini-Hochberg correction across the full family of significance tests (Table A9)
9. `calibration_metrics.py` / `calibration_metrics_ci.py`: calibration slope, intercept, ECE, ICI and their bootstrap CIs across all five cohort/stage combinations (Table 6, Table A3)
10. `leave_one_series_out.py`: true leave-one-series-out retrain (Section 7, Limitations)
11. `score_fsqn_vs_baseline.py`: FSQN/ComBat vs. self-standardization scoring (Section 6.1). Pass `--external-combat <path>` for a three-way baseline/FSQN/ComBat comparison in one run, and `--save-predictions <path>` to write per-sample probabilities for all conditions (needed by `figures/build_figure_appendix2.py`'s ROC curves and confusion matrices)

### 4. Figures (`figures/`)

```bash
cd figures
for f in build_figure*.py; do python "$f"; done
```

Each writes to `results/figures/`. The four main-text figures and two appendix figures read directly from `results/` CSVs and JSONs produced by the steps above.

All scripts use `numpy` seed 42 (or, where the *k* = 1 / *k* = 20 seed distributions are involved, an explicit seed count set in the script) so re-running against the same input data reproduces the manuscript's point estimates.

## Results Summary

| Result | Headline metric |
|---|---|
| Stage-1 (mRNA) internal | 96.56% accuracy, 0.993 AUC (*n* = 697) |
| Stage-1 external #1 (GSE118553) | 57.14% accuracy, 0.484 AUC (*n* = 63) |
| Stage-1 external #2 (GSE5281) | 55.88% accuracy, 0.415 AUC (*n* = 34) |
| Stage-2 (methylation) internal | 63.38% accuracy, 0.646 AUC (*n* = 142) |
| Stage-2 external (GSE134379) | 56.68% accuracy, 0.617 AUC (*n* = 404) |
| Same-label fusion estimate (margin < 0.6, *k* = 20) | 93.10%, exceeds the attainability bound's 95% CI upper tail (77.00%) |
| Single-partner *k* = 1 re-simulation (margin < 0.6) | 68.25% (sd 4.31 across 200 seeds), close to the 68.97% mRNA-only baseline; seed-averaged difference -0.64 pp, 95% CI [-6.45, +5.39] |
| FSQN / ComBat vs. self-standardization | no significant difference on either external cohort under FSQN; ComBat significantly worse on GSE5281 (delta AUC = -0.312) |
| Leave-one-series-out retrain | both directions generalize well (AUC 0.984 and 0.9998), no series-identity shortcut |

Full statistical detail is in `results/`.

## Data Sources

All data are public and require no DUA. See the manuscript's Data Availability statement for the complete list of GEO accessions and the Zenodo archive used, and `data/README.md` for download instructions.

## Citation

If you use this code or build on this study, please cite the manuscript (see `CITATION.cff`) and the original data sources listed in the Data Availability statement.

## License

Code in this repository is released under the [MIT License](./LICENSE). The underlying datasets retain their original licenses (public GEO records; CC-BY-4.0 for the Zenodo-hosted feature panel); no data files with restrictive licenses are redistributed here.
