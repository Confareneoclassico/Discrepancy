# Changelog

## 2.0.0 (revision for Environmental Modelling & Software)

Version 2.0 replaces the monolithic `discrepancy.py` of version 1.x by a
library (`src/discrepancy.py`) and separate experiment and reporting scripts.
It corrects errors found while revising the paper. All results of the
revised paper were recomputed with this version.

### Corrections

- **PCE and Shapley indices.** Version 1.x computed total-order indices and
  Shapley effects from the *monomial* coefficients of the polynomial returned
  by `chaospy.fit_regression` (`approx.coefficients`), not from the
  coefficients of the orthonormal basis. Sums of squared monomial
  coefficients are not variance components, so the indices were wrong in
  general (for y = x1 + 3(x2 - 0.5)^2 version 1.x returned T = (0.05, 0.95)
  instead of (0.625, 0.375)). Version 2.0 fits an orthonormal Legendre
  (or Hermite) basis by least squares and selects the degree by the corrected
  leave-one-out error; it no longer depends on chaospy.
- **Benchmark functions.** The function labelled "Bratley 1988" in 1.x was
  the Bratley, Fox & Niederreiter (1992) function; the function labelled
  "Bratley 1992" (with weights c_j = 2j - 1) is not a published test
  function; the Oakley-O'Hagan coefficients a2, a3 and the matrix M were
  wrong (M had been replaced by a small diagonal matrix). Version 2.0 follows
  the definitions of the R package `sensobol` and of the original articles.
- **Becker metafunction.** Version 1.x used a simplified random-function
  generator. Version 2.0 ports `sensobol::metafunction` (13 univariate
  functions, mixture coefficients, pairwise and three-way interactions).
- **Sensitivity-of-sensitivity study.** In 1.x the sampling-design factor was
  decoded as floor(1 + u), which is always 1, so the design never varied and
  its index was zero by construction. Version 2.0 decodes the factor
  correctly and caches paired samples for the three designs.
- **Multiverse reference.** The reference total-order indices were Jansen
  estimates with base sample size 2^9; they are now computed with 2^16.
- **HYMOD.** Columns x4 and x5 of `hydro.csv` are Rs (range 0-0.1) and Rq
  (range 0.1-0.99); 1.x labelled them the other way round. The reference
  indices hard-coded in 1.x, which came from a HYMOD simulation with a
  different setup, are
  inconsistent with the sample (the first-order
  index of alpha estimated from the sample, about 0.12, exceeds the
  hard-coded total-order index, 0.073); `hymod_reference.py` now computes the reference
  from the sample through two independent surrogates.
- **Rank-to-column mapping.** The column of a rank r is ceil((r - 1) s / N);
  version 1.x and the first 2.0 drafts computed it in floating point, which
  misplaces the ranks that fall exactly on a block boundary for some N (for
  instance N = 600 with s = 25). It is now computed in integer arithmetic.
  None of the sample sizes of the benchmarks, the convergence study, HYMOD,
  the Becker ensemble or the null distribution is affected with s =
  ceil(sqrt(N)); the sensitivity analysis of the method, which varies the
  grid exponent, was rerun.
- **Savage-score correlation.** Signed (1.x took the absolute value in some
  tables), with ties averaged.

- **HYMOD reference in `experiments.py`.** The intermediate file
  `hymod_metrics.csv` is now computed against the corrected reference (the
  tables and figures always used `hymod_refs.json`).
- **Screening errors and the theory.** New script `screening_check.py`
  (Supplementary Material B, Section B.7 and Table S15). It estimates the
  limits of the adjusted ersatz for all benchmark inputs at large N and
  checks them against the bounds of SM A, Proposition 5.4 (written to
  `results/screening_limits.csv`); compares the flag rates of the Becker
  inputs with T_i = 0 with the distribution under independence at their
  sample sizes; relates the type I error at theta = 0.05 in the convergence
  study to N/s^2; checks the ranking guarantee on the pairs of benchmark
  inputs with well-separated delta; and reports the lowering of the
  statistic by the maximin LHS and Sobol' designs. It writes the body of
  Table S15 to `tables/tab_screening_theory.tex`.
- **Script fixes (final check of all entry points).** `experiments.py`
  without arguments printed a traceback instead of the usage message, and
  `experiments.py sos_first --quick` took `--quick` for the cache directory;
  both are fixed, and `sos_first` accepts `--quick`. `sos` and `sos_first` run before `becker` (which stores the samples they use) now stop with a message saying so instead of a traceback. `delta_reference.py`
  run on a subset of the functions replaced `results/delta_refs.json` and
  deleted the other references; it now updates the file. The bootstrap of
  `sos_indices` no longer prints warnings when a resample of the type II
  error of PAWN has zero variance on the small nested designs of Figure S2
  (the interval is NaN and is not shown); `jansen_total` and `saltelli_first` return NaN without a warning when the output is constant on A. Unused imports removed. The
  docstring of `design_matrix` describes the scrambling of
  `scipy.stats.qmc.Sobol` correctly (linear matrix scramble and digital
  shift, not Owen's nested scrambling). `make_tables.py` writes the Becker
  table exactly as printed in the paper ("Maximin LHS"). `theory_checks.py`
  computes the optimiser of Remark 7.3 in 40-digit arithmetic and prints six
  digits. `reference_indices.py --seeds` recomputes the Jansen references
  with 14 other scrambling seeds; its docstring claimed agreement within
  5e-4 for any seed, but the largest deviation is 7.1e-4 (Oakley-O'Hagan);
  Section 2.7 of the paper now states agreement within 1e-3.
- **R implementation (folder `R/`, provenance only).** Checked against the
  Python package on random samples (including inputs equal to 0, tied
  outputs and N = 2500), the original R functions differed in four ways,
  now corrected: an input value equal to 0 was dropped from the grid
  instead of being assigned to the first row; the column of the rank grid
  was computed in floating point (wrong cells for N = 2500, for instance);
  a constant output made the min-max scaling of the S-ersatz divide by
  zero; and `savage_scores_fun` used the average ranks of tied values as
  indices, which gave wrong Savage scores whenever two values were equal.
  The corrected functions agree exactly with `src/discrepancy.py` on all
  test cases. `code_discrepancy_reloaded.R` sourced its function files from
  a folder `functions` that is not in the repository; it now sources them
  from `R/`, loads `logitnorm` (needed for phi = 7 and 8) and runs
  `mclapply` on at least one core (one on Windows); `update_matrix` is
  defined once, in `discrepancy_fun.R`. `discrepancy.R`, the exploratory
  script of the first version, used absolute paths of the original machine
  and wrote its HYMOD design over `hydro.csv`; it now reads the HYMOD
  outputs from `data/`, writes to `R/output/`, and labels xi and zeta of the
  Multiverse table in the order of the columns (they were swapped). It still
  needs `AE_df.rds`, the reference indices of the first version, which is
  not part of the repository.

### Additions

- Simple random sampling and maximin Latin hypercube designs alongside
  scrambled Sobol' sequences; replicated benchmarks (100 replicates).
- Reference values of Borgonovo's delta (`delta_reference.py`) and agreement
  metrics against both references.
- Rank-transformed statistic without fill (`ersatz_unfilled`), random
  tie-breaking option, and the asymptotic quantities used in Supplementary
  Material A.
- `null_distribution.py`: null distribution of the adjusted ersatz with
  Monte Carlo standard errors (Supplementary Material A, Table 2); Figure 5b
  is drawn from its output.
- `theory_checks.py`: reproduces every constant, the computer-assisted steps
  and the counterexamples of Supplementary Material A, and the Gaussian-copula
  values of its Table 3 (quadrature in rotated coordinates, checked by Monte
  Carlo). It verifies the zero condition of Proposition 5.4(iii) on
  [lambda_c, 3/4) by a monotonicity argument and monotone bracketing on
  161,897 cells (formerly a grid check), and prints the optimiser of the
  linear programme of Remark 7.3.
- `reference_indices.py`: recomputes the reference total-order indices of
  `refs.json` (analytical, or Jansen estimates with base size 2^16-2^17).
- `src/results/`: the raw results used in the paper.
- `make_figures.py`: figures are about 10 inches wide, so that their text
  prints at 7-9 pt at the text width of the manuscript. Axes are not
  truncated: the convergence figure shows rho over [-1, 1] with common axes in
  each column, and the four panels of the Becker figure share one vertical
  scale.
- Sensitivity analysis of the method: first-order indices in addition to
  total-order indices, and a base sample of 2^12 instead of 2^8 (the new
  design contains the old one as its first 256 rows). The design adds the
  matrix B to [A; AB_1..AB_5] (28,672 evaluations; results/sos_outputs.csv
  and results/sos_outputs_B.csv); `saltelli_first` in `discrepancy.py`
  implements the estimator of Saltelli et al. (2010) on the centred output;
  `python experiments.py sos_first <dir of sos_cache.npy>` adds the B
  evaluations to an existing run of the A and AB blocks. `make_tables.py`
  writes Table S14.
- Convergence of the sensitivity analysis of the method: `make_figures.py`
  computes the indices on the nested designs of base size 2^6 to 2^12 (the
  first n rows of every block), plots them (Figure S2) and writes
  results/sos_convergence.csv; `sos_convergence_stats` in `experiments.py`
  computes the ranking, screening and index statistics of Sarrazin et al.
  (2016) (results/sos_convergence_stats.csv).

### Documentation and packaging

- The paper is now titled "An Adjusted Ersatz Discrepancy for Global
  Sensitivity Analysis: Theory and Benchmarks" (submitted as "Improving
  Discrepancy Measures for Global Sensitivity Analysis"); README.md,
  CITATION.cff and the docstring of `discrepancy.py` use the new title.
- Installation: Python 3.9 or later is required and is now stated in
  `requirements.txt` and README.md, with instructions for creating the
  environment with a recent interpreter (pip under Python 3.7 cannot find
  numpy>=1.24). `requirements-lock.txt` gives the exact versions used for the
  results (Python 3.12 or later). Python 3.9 with the minimum versions
  reproduces the tables and the theory checks exactly.
- CITATION.cff lists the current affiliations and the ORCID iDs of all
  authors.
- `data/README.md` gives the HYMOD simulation period (28 July 1952 to
  30 September 1962).

