"""Multiple-comparison correction for the significance-test family reported in
Table A9 of the manuscript.

Eight formal hypothesis tests (four DeLong paired-AUC tests, one McNemar exact
test, and three DeLong tests from the harmonization comparison) form the
de-duplicated correction family. Two additional tests (same-label k=20 vs.
mRNA-only, and random pairing vs. mRNA-only, both at margin<0.9) are
numerically identical to their whole-cohort counterparts because the
margin<0.9 subset contains every patient whose prediction changed under
either construction, so the discordant-pair sets coincide exactly; these are
reported for completeness but excluded from the correction basis.

A two-proportion z-test comparing internal vs. external Stage-1 accuracy was
previously included as a ninth test but has been removed from the family: it
is a descriptive comparison, not central to the paper's confirmatory claims,
and its removal is reflected in this corrected 8-test family.
"""
from statsmodels.stats.multitest import multipletests

TESTS = [
    ("1: Same-label k=20 vs mRNA-only (whole cohort, paired DeLong)", 0.0005),
    ("2: Same-label k=20 vs mRNA-only (margin<0.6, paired DeLong)", 0.0002),
    ("3: Random pairing vs mRNA-only (whole cohort, paired DeLong)", 0.010),
    ("4: Random pairing vs mRNA-only (whole cohort, McNemar exact)", 1.0),
    ("5: FSQN vs baseline (GSE118553, paired DeLong)", 0.71),
    ("6: FSQN vs baseline (GSE5281, paired DeLong)", 0.12),
    ("7: ComBat vs baseline (GSE118553, paired DeLong)", 0.26),
    ("8: ComBat vs baseline (GSE5281, paired DeLong)", 0.004),
]

if __name__ == "__main__":
    names = [t[0] for t in TESTS]
    pvals = [t[1] for t in TESTS]

    reject_h, padj_h, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    reject_b, padj_b, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    print("Corrected 8-test family (two-proportion z-test removed)")
    print("=" * 90)
    print(f"{'Test':<62}{'raw p':>10}{'Holm':>10}{'BH':>10}")
    print("-" * 90)
    for n, p, ph, pb in zip(names, pvals, padj_h, padj_b):
        print(f"{n:<62}{p:>10.4f}{ph:>10.4f}{pb:>10.4f}")

    import pandas as pd
    pd.DataFrame({
        "test": names, "raw_p": pvals,
        "holm_adj_p": padj_h, "holm_reject": reject_h,
        "bh_adj_p": padj_b, "bh_reject": reject_b,
    }).to_csv("results/multiple_comparison_correction_v2.csv", index=False)
    print("\nWrote results/multiple_comparison_correction_v2.csv")
