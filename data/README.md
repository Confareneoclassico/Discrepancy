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
