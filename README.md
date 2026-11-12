# An Adjusted Ersatz Discrepancy for Global Sensitivity Analysis: Theory and Benchmarks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.NNNNNNN.svg)](https://doi.org/10.5281/zenodo.NNNNNNN)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<!-- ACTION: replace XXXXXXX above (twice) and in CITATION.cff with the Zenodo DOI of release 2.0.0. -->

Code, data and supplementary material for

> Lo Piano, S., A. Lachi, R. Sheikholeslami, A. Puy, P. T. Roy,
> A. Saltelli, **"An Adjusted Ersatz Discrepancy for Global Sensitivity
> Analysis: Theory and Benchmarks"**, *Environmental Modelling and Software*
> (revised version, 2026; submitted as "Improving Discrepancy Measures for
> Global Sensitivity Analysis").

The paper revisits the S-ersatz of
[Puy, Roy & Saltelli (2024, *Technometrics*)](https://doi.org/10.1080/00401706.2024.2304341),
a grid-occupancy statistic computed from the scatterplot of an input against
the output, and proposes the **adjusted ersatz**: the output is rank-transformed
before gridding and isolated empty cells are filled with a Moore-neighbourhood
rule. Supplementary Material A proves that, under simple random sampling, the
rank-transformed grid without fill is a histogram estimator of Borgonovo's
delta that converges to a different f-divergence, and that the adjusted
ersatz converges to a functional that lies between delta - 0.141 and
delta + 0.048 for every dependence structure.
The adjusted ersatz is compared at equal sample cost with a degree-adaptive
polynomial chaos expansion (PCE), PCE-based Shapley effects and the
given-data PAWN index, against total-order Sobol' indices and against delta.

Version 2.0 corrects several errors of version 1.x; see [CHANGELOG.md](CHANGELOG.md).

## Repository structure

```
.
├── src/
│   ├── discrepancy.py        # library: designs, ersatz statistics, PCE/Shapley,
│   │                         # PAWN, Jansen, Savage scores, test functions,
│   │                         # Becker metafunction
│   ├── experiments.py        # benchmarks, convergence, HYMOD, Becker, SoS
│   ├── reference_indices.py  # reproduces the reference total-order indices (refs.json)
│   ├── delta_reference.py    # large-sample delta references
│   ├── hymod_reference.py    # surrogate-based references for HYMOD
│   ├── null_distribution.py  # null distribution of the adjusted ersatz (SM A, Table 2)
│   ├── theory_checks.py      # numerical and computer-assisted steps of SM A
│   ├── screening_check.py    # screening errors against the theory (SM B, Section B.7)
│   ├── summaries.py          # agreement metrics against T_i and delta_i
│   ├── make_tables.py        # LaTeX tables of the paper and SM B
│   ├── make_figures.py       # figures of the paper and SM B
│   ├── refs.json             # reference total-order indices
│   └── results/              # raw results used in the paper (CSV/JSON)
├── data/
│   ├── hydro.csv             # HYMOD inputs (Cmax, bexp, alpha, Rs, Rq)
│   ├── hydro2.csv            # HYMOD inputs as run + NSE output
│   └── README.md
├── R/                        # original R implementation (provenance; fixes in CHANGELOG)
├── supplementary/            # Supplementary Materials A and B (PDF and LaTeX)
├── examples/minimal_example.py
├── requirements.txt          # minimum versions (Python 3.9 or later)
├── requirements-lock.txt     # exact versions used for the results
├── CHANGELOG.md
├── CITATION.cff
└── LICENSE                   # MIT
```

## Installation

The code requires **Python 3.9 or later**. Check the interpreter that will
create the environment first:

```bash
python3 --version          # Windows: py --list
```

With an older interpreter (for example Python 3.7), pip finds no NumPy 1.24
or later and stops with `Could not find a version that satisfies the
requirement numpy>=1.24`. Create the environment with a recent interpreter
instead, for example Python 3.12:

```bash
git clone https://github.com/Confareneoclassico/Discrepancy.git
cd Discrepancy
python3.12 -m venv venv              # Windows: py -3.12 -m venv venv
source venv/bin/activate             # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

or, with conda:

```bash
conda create -n discrepancy python=3.12
conda activate discrepancy
pip install -r requirements.txt
```

`requirements.txt` gives minimum versions. The results of the paper were
computed with Python 3.13 and the exact versions in `requirements-lock.txt`
(`pip install -r requirements-lock.txt` reproduces that environment on
Python 3.12 or later). With Python 3.9 and the minimum versions, the LaTeX
tables and the output of `theory_checks.py` are identical to those of the
locked environment, and a quick benchmark run (`--quick`) agrees with it to
within 1e-15.

## Reproducing the results

Run from `src/`; every random stream is seeded. The folder `src/results/`
already contains the raw results used in the paper, so the last line alone
regenerates all tables and figures.

```bash
cd src
python experiments.py benchmarks     # N = 512, 100 replicates, 3 designs  (~2 min)
python experiments.py convergence    # N = 2^4..2^13, 20 replicates        (~15 min)
python experiments.py hymod          # HYMOD sample, blocks and subsets    (~5 min)
python experiments.py becker         # 512 Becker configurations, 3 designs (~10 min)
python experiments.py sos            # sensitivity of the sensitivity analysis: S_i and T_i (~1.5 h; needs becker)
python reference_indices.py          # checks refs.json (~1 min; --seeds: 14 seeds, ~5 min)
python delta_reference.py            # delta references (~5 min)
python hymod_reference.py            # HYMOD references (~3 min)
python null_distribution.py          # null distribution, SM A Table 2 (~8 min)
python theory_checks.py              # verifications of SM A (~3 min)
python screening_check.py            # screening errors and the theory, SM B Table S15 (~3 min)
python summaries.py && python make_tables.py && python make_figures.py
```

Times are for two cores. Raw results are written to `src/results/`, tables to
`src/tables/` and figures to `src/figs/`.

## Minimal example

```python
import sys; sys.path.insert(0, "src")
import discrepancy as D

U = D.design_matrix(512, 8, design="srs", seed=1)    # inputs on [0, 1]
Y = D.sobol_g(U)
print(D.ersatz_all(U, Y))                  # adjusted ersatz, one value per input
T, Sh, fit = D.pce_indices(U, Y)           # PCE total-order indices and Shapley effects
print(T, fit["p"], fit["loo"])
print(D.pawn_given_data(U, Y))             # given-data PAWN (max KS)
```

To screen inputs with the adjusted ersatz on its own scale, compare each value
with an upper quantile of its distribution under independence, which under
simple random sampling depends only on N (Supplementary Material A, Table 2).

## Data

`data/hydro.csv` and `data/hydro2.csv` contain 50,000 runs of the HYMOD model
for the Leaf River catchment on an Owen-scrambled Sobol' design mapped to
uniform distributions; the output is the Nash-Sutcliffe efficiency. See
`data/README.md`.

## Citing

Until the article is published, please cite the archived software release:

```bibtex
@software{lopiano2026discrepancy,
  author    = {Lo Piano, Samuele and Lachi, Alessio and Sheikholeslami, Razi and
               Puy, Arnald and Roy, Pamphile T. and Saltelli, Andrea},
  title     = {An Adjusted Ersatz Discrepancy for Global Sensitivity Analysis:
               Theory and Benchmarks},
  year      = {2026},
  version   = {2.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.NNNNNNN},
  url       = {https://github.com/Confareneoclassico/Discrepancy}
}
```

## License

MIT, see [LICENSE](LICENSE).

## Contact

a.lachi@unilink.it (corresponding author); samuele.lo.piano@pg.edu.pl
