# Original R implementation (provenance)

These scripts are the R implementation of the S-ersatz and of its first
adjustment, as used by Puy, Roy & Saltelli (2024) and in the first version of
this project. They are kept for provenance and cross-checking. The results of
the paper are computed with the Python code in `../src`, which follows the
same grid conventions (`ceiling` cell indices, ranks with
`ties.method = "first"`).

Version 2.0.0 corrects four errors in the functions (inputs equal to 0,
floating-point rank columns, constant outputs, Savage scores with ties) and
the paths of the two scripts, as listed in `../CHANGELOG.md`. With these corrections,
`s_ersatz`, `s_ersatz_adj`, `jansen_fun` and `savage_scores_fun` give the
same values as the Python package on random test samples.

- `discrepancy_fun.R`, `jansen_fun.R`, `random_distributions_fun.R`,
  `savage_scores_fun.R`, `triggers_fun.R`: functions.
- `code_discrepancy_reloaded.R`: Becker-ensemble study of the first version
  (needs the R packages sensobol, data.table, tidyverse, Rfast, qrng,
  logitnorm and scales). Run from the repository root or from `R/`.
- `discrepancy.R`: exploratory script of the first version (benchmark
  functions, HYMOD, convergence plots). Run from `R/`. It writes to
  `R/output/` and needs `AE_df.rds`, the reference indices of the first
  version, which is not part of the repository.
