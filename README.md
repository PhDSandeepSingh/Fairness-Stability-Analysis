# Fairness-Stability Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

A reproducible experimental benchmark auditing how reliably tabular bias-mitigation methods survive train/test split noise across repeated stratified partitions, using paired Nadeau–Bengio overlap-corrected inference, Holm–Bonferroni multiplicity control, 80%-power minimum detectable effects (MDEs), and matched-threshold reference baselines.

## Key Experimental Findings

| Benchmark Dimension | Primary Result ($n_{\text{te}}/n_{\text{tr}} = 0.30/0.525$, $R = 20$ splits) |
| :--- | :--- |
| **Total Audited Comparisons** | 88 paired comparisons (44 DPD + 44 EOD across 4 datasets, 3 learners, 5 mitigation variants) |
| **Demographic Parity Difference (DPD)** | **23 / 44** robust gains · 21 indistinguishable from noise · 0 harms |
| **Equalized Odds Difference (EOD)** | **7 / 44** robust gains · 34 indistinguishable from noise · **3 robust harms** |
| **Overlap-Correction Impact** | Naive paired *t*-test flags **66** effects vs. **33** after overlap correction (**33 / 66** naive hits unsupported) |
| **Single-Split Reversal Rate ($F$)** | Median **0.00** (DPD) and **0.20** (EOD); **52.3%** of EOD comparisons reverse sign on $\ge 20\%$ of splits |
| **Matched-Threshold Reference** | Apparent balanced-accuracy gain of `ThresholdOptimizer` (**+0.025** vs. 0.5 threshold) turns negative (**−0.031**) once the baseline threshold is also tuned on the post-processing split |
| **Power & MDE (80% Power)** | Critical standardised effect **3.02** (50% power); 80%-power MDE **3.70** paired-SD units (exceeding full baseline gaps on German Credit) |

## Benchmarks & Experimental Matrix

- **Datasets (UCI Machine Learning Repository):** Adult (`sex`), Bank Marketing (`age 25–60`, `duration` removed), Default of Credit Card Clients (`sex`), and Statlog German Credit (`age >= 25`). Exact duplicates removed before splitting; SHA-256 checksums recorded in `results/data_sha256.json`.
- **Base Learners:** Logistic Regression (`LR`), Random Forest (`RF`), and Histogram Gradient Boosting (`HGB`).
- **Mitigation Methods:** Reweighing (`RW`), `ThresholdOptimizer` for DP and EO (`TO-DP`, `TO-EO`), and `ExponentiatedGradient` for DP and EO (`EG-DP`, `EG-EO`, LR base).
- **Split Protocol:** $R = 20$ stratified splits (30% test, 52.5% fit, 17.5% post-processing calibration).

## Repository Structure

```text
.
├── Paper1_Fairness_Audit.ipynb              # Primary audit notebook (0.30/0.525 fit-split ratio)
├── make_notebook.py                         # Programmatic notebook generator
├── run_round2_p1.py                         # Metric-unit MDE & sensitivity calculation script
├── run_chunk.py                             # Resumable split-execution helper
├── gen_tables.py                            # Programmatic summary & table generator from raw_runs.csv
├── data/                                    # Raw UCI benchmark files
└── results/
    ├── raw_runs.csv                         # All per-split paired learner/mitigation runs
    ├── table_paired_deltas.csv              # Nadeau–Bengio & descriptive Wilson CIs (0.30/0.525)
    ├── table_paired_deltas_ratio07.csv      # Outer-split ratio sensitivity (0.30/0.70)
    ├── table_sensitivity_tuned_threshold.csv # Matched-threshold baseline comparison
    ├── table_mde_units.csv                  # Dataset-specific 80%-power MDEs in raw DPD/EOD units
    ├── mde_summary.json                     # Power & MDE summary metadata
    ├── data_sha256.json                     # Input dataset integrity hashes
    ├── environment.json                     # Software & package version snapshot
    ├── fig_paired_change.png/.pdf           # Paired DPD/EOD change boxplots across splits
    └── fig_sign_flip.png/.pdf               # Directional sign-flip rate distribution
```

## Quick Start & Reproducibility

```bash
# 1. Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib fairlearn==0.14.0 xlrd nbformat

# 2. Recompute metric-unit MDEs and summary outputs from saved runs
python run_round2_p1.py

# 3. Regenerate all summary tables and metrics from results/raw_runs.csv
python gen_tables.py

# 4. (Optional) Re-execute the full 20-split audit notebook
jupyter nbconvert --to notebook --execute Paper1_Fairness_Audit.ipynb
```

## Topics / Tags

`algorithmic-fairness` · `bias-mitigation` · `reproducibility` · `statistical-testing` · `fairlearn` · `tabular-ml` · `responsible-ai` · `nadeau-bengio`

## License

This codebase and experimental analysis are released under the **MIT License**. Underlying UCI benchmark datasets remain subject to their original repository terms.
