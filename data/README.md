# Data

## `hydro.csv` and `hydro2.csv`

These files contain the input sample and corresponding model output for the HYMOD
conceptual rainfall--runoff model, used in the manuscript's real-world case study
(`§ Real-world HYMOD`).

### Confirmed directly from the files (verified by inspection, June 2026)

| Field | Value |
|---|---|
| Number of rows | 50,000 — matches the manuscript's stated $N=50{,}000$ |
| `hydro.csv` columns | unnamed index, `x1`, `x2`, `x3`, `x4`, `x5` (input design matrix only, no output) |
| `hydro2.csv` columns | unnamed index, `x1`, `x2`, `x3`, `x4`, `x5`, `NSE` (same design matrix **with** the output column) |
| Output metric | `NSE` — Nash--Sutcliffe efficiency, consistent with the manuscript's stated output measure |
| Input ranges (approx., from spot-checking rows) | `x1` ≈ tens to hundreds (consistent with Cmax ∈ [1, 500]); `x2` ≈ 0--2 (consistent with bexp ∈ [0.1, 2.0]); `x3`, `x4`, `x5` ≈ 0--1 (consistent with alpha, Rq ∈ [0.1, 0.99] and Rs ∈ [0, 0.1], though `x3` exceeding 0.99 in places — see Discrepancy below) |
| File format | CSV, comma-separated, double-quoted header in `hydro.csv`; `hydro2.csv` has duplicated `\r\r\n` line endings (likely produced by a Windows-authored script saved/re-read on a Unix system, or vice versa) |

### Discrepancy to resolve before publishing this repository

Spot-checking row 2 of both files shows `x3` differs between `hydro.csv`
(`0.30697665810585`) and `hydro2.csv` (`0.309328666`) for what should be the same
sample — the values agree to only 2 significant figures, not the full precision
expected of the same design matrix written twice. **This needs author verification**:
either (a) `hydro2.csv` was regenerated from a slightly different run/seed than
`hydro.csv` and the two are not meant to be the same design (in which case the
manuscript text and any code assuming row-correspondence between the two files should
be double-checked), or (b) one file underwent a lossy rounding step the other did not.

### Still PLACEHOLDER — needs author input

| Field | Status |
|---|---|
| Mapping of `x1`...`x5` to named parameters (Cmax, bexp, alpha, Rq, Rs) | Inferred from value ranges above, not confirmed against the original generation script |
| Sampling method (Sobol QRN, LHS, other) | Not determinable from the static CSV |
| Random seed | Not determinable from the static CSV |
| Generation script | Not included in this repository |
| Reference HYMOD implementation used | Manuscript cites Vrugt et al. (2003) for the Leaf River configuration; confirm exact code/version used to generate these specific 50,000 runs |

**Action needed before publishing this repository:** resolve the `x3` discrepancy
above, confirm the parameter-to-column mapping, and ideally include the generation
script itself (Python or R) so the data is regenerable rather than only distributable
as a static 50,000-row CSV pair.

<!-- ## File sizes

`hydro.csv` (~4.9 MB) and `hydro2.csv` (~4.0 MB) are within typical GitHub file-size
limits (100 MB hard limit, soft warning above ~50 MB) and do not require Git LFS. -->
