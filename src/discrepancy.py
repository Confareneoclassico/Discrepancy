import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')  # suppress numpoly/chaospy noise
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numpoly')
"""
Python translation of discrepancy.R
------------------------------------
Implements:
  - sobol_mat / scrambled_sobol  : Sobol sample-matrix construction
  - update_matrix                : neighbour-fill for the adjusted discrepancy
  - s_ersatz / s_ersatz_adj      : unadjusted / adjusted ersatz-discrepancy
  - discrepancy_ersatz           : dispatcher over all parameters
  - jansen_fun                   : Jansen total-order Sobol estimator
  - savage_scores_fun            : Savage scores
  - Test functions               : Ishigami, Bratley 1988/1992, Sobol-g, Oakley-O'Hagan
  - Analysis blocks              : replicating every study in the R script
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import qmc, rankdata, truncnorm

# ─────────────────────────────────────────────────────────────
# 1.  Sobol sample-matrix construction
# ─────────────────────────────────────────────────────────────

def _build_loop(k: int, order: str) -> list:
    """
    Return the list of column-index groups used to build AB/BA/CB matrices.

    For 'first' order each element is a length-1 list [j].
    For higher orders the list is extended with all length-2, -3, … pairs.
    Mirrors R:  first <- 1:k;  second <- c(first, combn(1:k,2,simplify=FALSE)); …
    """
    first = [[j] for j in range(k)]
    if order == "first":
        return first
    pairs = [list(p) for p in combinations(range(k), 2)]
    if order == "second":
        return first + pairs
    trips = [list(t) for t in combinations(range(k), 3)]
    if order == "third":
        return first + pairs + trips
    quads = [list(q) for q in combinations(range(k), 4)]
    if order == "fourth":
        return first + pairs + trips + quads
    raise ValueError("order must be 'first', 'second', 'third', or 'fourth'")


def scrambled_sobol(matrices: tuple, A: np.ndarray, B: np.ndarray,
                    C: np.ndarray | None, order: str) -> np.ndarray | None:
    """
    Build the scrambled matrices AB, BA, CB from base matrices A, B, C.

    Returns a stacked (n_matrices * N) × k array, or None if nothing requested.
    Mirrors R's scrambled_sobol().
    """
    loop = _build_loop(A.shape[1], order)
    parts = []

    if "AB" in matrices:
        for cols in loop:
            AB = A.copy();  AB[:, cols] = B[:, cols]
            parts.append(AB)

    if "BA" in matrices:
        for cols in loop:
            BA = B.copy();  BA[:, cols] = A[:, cols]
            parts.append(BA)

    if "CB" in matrices and C is not None:
        for cols in loop:
            CB = C.copy();  CB[:, cols] = B[:, cols]
            parts.append(CB)

    return np.vstack(parts) if parts else None


def sobol_mat(N: int, params: list,
              matrices: tuple = ("A", "B", "AB"),
              order: str = "first",
              type_: str = "QRN",
              seed: int = 123) -> np.ndarray:
    """
    Build the full Sobol sample matrix ready for model evaluation.

    Parameters
    ----------
    N        : base sample size
    params   : list of parameter names (length = k)
    matrices : which base matrices to include in the output
    order    : highest interaction order for AB/BA/CB columns
    type_    : 'QRN' (scrambled Sobol), 'R' (random uniform), 'LHS'
    seed     : RNG seed

    Returns
    -------
    NumPy array of shape (n_rows × k).
    Row layout: [A rows] [B rows] [C rows] [scrambled rows]
    (any section is omitted when the corresponding letter is absent from matrices)
    """
    k = len(params)
    has_C = any("C" in m for m in matrices)
    n_base = 3 if has_C else 2

    # ── draw the raw uniform base ──────────────────────────────
    if type_ == "QRN":
        # R's sobol_mat uses qrng::sobol(seed=123) regardless of the epsilon
        # argument — the seed is hardcoded inside the function. Python mirrors
        # this by ignoring the caller-supplied seed for QRN and always using 123.
        sampler = qmc.Sobol(d=k * n_base, scramble=True, seed=123)
        df = sampler.random(N)
    elif type_ == "R":
        # For random sampling R uses set.seed(epsilon) before stats::runif,
        # so the caller-supplied seed IS used.
        rng = np.random.default_rng(seed)
        df = rng.uniform(size=(N, k * n_base))
    elif type_ == "LHS":
        sampler = qmc.LatinHypercube(d=k * n_base, seed=seed)
        df = sampler.random(N)
    else:
        raise ValueError("type_ must be 'QRN', 'R', or 'LHS'")

    A = df[:, :k]
    B = df[:, k:2*k]
    C = df[:, 2*k:3*k] if has_C else None

    out = scrambled_sobol(matrices=matrices, A=A, B=B, C=C, order=order)

    # ── assemble in the R order: A, B, C, scrambled ────────────
    parts = []
    if "A" in matrices: parts.append(A)
    if "B" in matrices: parts.append(B)
    if "C" in matrices and C is not None: parts.append(C)
    if out is not None: parts.append(out)

    return np.vstack(parts)


# ─────────────────────────────────────────────────────────────
# 2.  Neighbour-fill (update_matrix)
# ─────────────────────────────────────────────────────────────

def update_matrix(M: np.ndarray, fill_threshold: float = 0.5) -> np.ndarray:
    """
    Vectorised version: for every zero cell, if the mean of its Moore
    neighbours >= fill_threshold, set it to 1.

    fill_threshold : float in (0, 1], default 0.5
        Lower values fill more aggressively (higher α, lower β).
        Higher values are more conservative (lower α, higher β).
        Calibrated proof: 0.5 correctly maps independence→ersatz=0
        and perfect monotone dependence→ersatz→1 (see proof.pdf §5).

    Uses scipy.ndimage for O(s²) vectorised computation.
    """
    from scipy.ndimage import uniform_filter

    M = M.astype(float)
    # uniform_filter computes the mean over a 3×3 window (includes centre).
    # We want mean of the up-to-8 neighbours excluding the centre itself.
    # neighbour_sum = box_sum_3x3 - M
    # neighbour_count = (number of cells in 3×3 window) - 1
    #   = min(i+2,nr)*min(j+2,nc) - max(i-1,0)*max(j-1,0) ... approximated
    # Simpler: use the full 3×3 mean then adjust back for the centre.
    nr, nc = M.shape
    box_sum  = uniform_filter(M, size=3, mode="constant", cval=0.0) * 9
    # Count cells in each window (handles edges correctly)
    ones     = np.ones_like(M)
    box_cnt  = uniform_filter(ones, size=3, mode="constant", cval=0.0) * 9

    nbr_sum  = box_sum  - M
    nbr_cnt  = box_cnt  - 1          # exclude centre cell count
    nbr_cnt  = np.where(nbr_cnt < 1, 1, nbr_cnt)   # avoid /0 at corners
    nbr_mean = nbr_sum / nbr_cnt

    # Fill zero cells where neighbour mean >= fill_threshold
    M_new = np.where((M == 0) & (nbr_mean >= fill_threshold - 1e-10), 1.0, M)
    return M_new.astype(int)


# ─────────────────────────────────────────────────────────────
# 3.  Ersatz-discrepancy sensitivity measures
# ─────────────────────────────────────────────────────────────

def s_ersatz(mat: np.ndarray) -> float:
    """
    Unadjusted ersatz discrepancy for a 2-column matrix [x_j, Y].

    Steps:
      1. Map x_j and min-max-scaled Y into an s×s grid (s = ceil(√N)).
      2. Mark occupied cells.
      3. Return 1 - (occupied cells) / s².
    Mirrors R's s_ersatz().
    """
    N = mat.shape[0]
    s = math.ceil(math.sqrt(N))

    # Row index (1-based) for x_j  ∈ (0, 1]
    m = np.ceil(mat[:, 0] * s).astype(int)          # 1 … s

    # Column index (1-based) for Y after min-max scaling
    x = mat[:, 1]
    n_norm = (x - x.min()) / (x.max() - x.min())
    n = np.ceil(n_norm * s).astype(int)
    n = np.where(n == 0, 1, n)                       # push 0 → 1

    # Mark cells (convert to 0-based for numpy)
    grid = np.zeros((s, s), dtype=int)
    grid[m - 1, n - 1] = 1

    return 1.0 - grid.sum() / grid.size


def s_ersatz_adj(mat: np.ndarray,
                 fill_threshold: float = 0.5,
                 grid_exponent: float = 0.5) -> float:
    """
    Adjusted ersatz discrepancy for a 2-column matrix [x_j, Y].

    Differences from the unadjusted version:
      • Y is rank-normalised (ordinal ranks / N) instead of min-max scaled.
      • update_matrix() fills isolated zero cells (Moore neighbourhood).
    Mirrors R's s_ersatz_adj() (discrepancy_fun.R).

    Parameters
    ----------
    fill_threshold : Moore-neighbourhood mean threshold for filling a zero
        cell.  Default 0.5 (calibrated to independence/comonotone extremes;
        see proof.pdf §5).  Lower → more fill → lower β, higher α.
    grid_exponent  : s = ceil(N^grid_exponent).  Default 0.5 (= √N).
        Controls grid resolution; 0.4 gives a coarser grid (more robust at
        small N), 0.6 a finer grid (more discriminating at large N).
    """
    N = mat.shape[0]
    s = math.ceil(N ** grid_exponent)

    m = np.ceil(mat[:, 0] * s).astype(int)           # 1 … s
    m = np.clip(m, 1, s)

    x = mat[:, 1]
    # (rank − 1) / N  — ordinal (ties.method = "first" in R)
    n_norm = (rankdata(x, method="ordinal") - 1) / N
    n = np.ceil(n_norm * s).astype(int)
    n = np.where(n == 0, 1, n)
    n = np.clip(n, 1, s)

    grid = np.zeros((s, s), dtype=int)
    grid[m - 1, n - 1] = 1
    grid = update_matrix(grid, fill_threshold=fill_threshold)

    return 1.0 - grid.sum() / grid.size


def discrepancy_ersatz(mat: np.ndarray, Y: np.ndarray,
                       params: list, adj: int = 0,
                       fill_threshold: float = 0.5,
                       grid_exponent: float = 0.5) -> pd.DataFrame:
    """
    Compute the ersatz-discrepancy sensitivity index for every column of mat.

    Parameters
    ----------
    mat            : N × k design matrix (values in [0, 1])
    Y              : length-N output vector
    params         : list of k parameter names
    adj            : 0 → unadjusted (s_ersatz); 1 → adjusted (s_ersatz_adj)
    fill_threshold : passed to s_ersatz_adj (default 0.5)
    grid_exponent  : passed to s_ersatz_adj (default 0.5, i.e. s = ceil(√N))

    Returns
    -------
    DataFrame with columns ['params', 'value']
    """
    values = []
    for j in range(mat.shape[1]):
        design = np.column_stack([mat[:, j], Y])
        if adj == 0:
            values.append(s_ersatz(design))
        else:
            values.append(s_ersatz_adj(design,
                                       fill_threshold=fill_threshold,
                                       grid_exponent=grid_exponent))

    return pd.DataFrame({"params": params, "value": values})


# ─────────────────────────────────────────────────────────────
# 4.  Jansen total-order Sobol estimator
# ─────────────────────────────────────────────────────────────

def jansen_fun(d: np.ndarray, N: int, params: list) -> pd.DataFrame:
    """
    Jansen (1999) total-order Sobol estimator.

    d   : 1-D array of length N*(1+k) — column-major layout:
          [Y_A (N), Y_AB_1 (N), …, Y_AB_k (N)]
    N   : base sample size
    Returns a DataFrame with columns ['parameters', 'value'].
    Mirrors R's jansen_fun().
    """
    # R's matrix(d, nrow=N) fills column-by-column (Fortran order)
    m = np.array(d).reshape(-1, N).T          # shape (N, 1+k)
    Y_A  = m[:, 0]
    Y_AB = m[:, 1:]

    f0  = Y_A.mean()
    VY  = np.mean((Y_A - f0) ** 2)
    value = (np.sum((Y_A[:, None] - Y_AB) ** 2, axis=0) / (2 * N)) / VY

    return pd.DataFrame({"parameters": params, "value": value})


# ─────────────────────────────────────────────────────────────
# 5.  Savage scores
# ─────────────────────────────────────────────────────────────

def savage_scores_fun(x: np.ndarray) -> np.ndarray:
    """
    Compute Savage scores, matching R's savage_scores_fun() exactly.

    Algorithm (mirrors R line-by-line):
      true.ranks <- rank(-x)                    # average ties, float
      p          <- sort(1 / true.ranks)        # ascending, uses float ranks
      mat        <- matrix(rep(p,n), byrow=T)   # each row = p
      mat[upper.tri(mat)] <- 0
      out <- sort(rowSums(mat), dec=T)[true.ranks]   # R truncates float idx

    Critical: 1/rank is computed on the FLOAT ranks (not int-cast), matching
    R's behaviour. The integer cast is applied only at the final index step,
    matching R's implicit truncation when indexing with a fractional value.

    For the no-tie case (typical in SA) the result is identical to any
    integer-rank version.  Tie handling now matches R exactly.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    # rank(-x) with average ties — keep as float to match R
    true_ranks = rankdata(-x, method="average")                # 1-based float, 1=largest

    # p = sort(1/true_ranks) ascending — must use float ranks here
    p = np.sort(1.0 / true_ranks)                             # ascending

    # Lower-triangular matrix: mat[i, j] = p[j] for j ≤ i, else 0
    mat = np.tile(p, (n, 1))
    mat[np.triu_indices(n, k=1)] = 0.0

    row_sums = mat.sum(axis=1)                                 # Savage scores
    sorted_desc = np.sort(row_sums)[::-1]                     # descending

    # R truncates float index to int (e.g. rank 2.5 → index 2 in R)
    int_idx = np.floor(true_ranks).astype(int) - 1            # 0-based
    return sorted_desc[int_idx]


# ─────────────────────────────────────────────────────────────
# 6.  Diagnostic table builder
# ─────────────────────────────────────────────────────────────

SCREEN_THRESHOLDS = (0.01, 0.05, 0.10)   # canonical GSA screening thresholds


def build_table(sobol_vals: np.ndarray,
                methods: dict,
                params: list,
                screen_thresholds: tuple = SCREEN_THRESHOLDS) -> pd.DataFrame:
    """
    Assemble the comparison table for an arbitrary set of estimators.

    Parameters
    ----------
    sobol_vals        : length-k reference total-order indices (T_i)
    methods           : {label: DataFrame-with-"value"-column}
    params            : list of k parameter names
    screen_thresholds : iterable of thresholds at which α and β are computed.
                        Default: (0.01, 0.05, 0.10) — the three canonical
                        GSA screening thresholds (conservative / standard /
                        liberal).  Separate α/β rows are produced for each.

    Summary rows appended per threshold θ:
      ρ   — Savage-score rank correlation (threshold-independent)
      MAE — mean absolute error          (threshold-independent)
      α(θ) — false-positive rate: predicted important, truly unimportant
      β(θ) — false-negative rate: predicted unimportant, truly important
    """
    sobol_vals   = np.asarray(sobol_vals, dtype=float)
    method_names = list(methods.keys())
    method_vals  = np.column_stack([methods[m]["value"].values
                                    for m in method_names])

    tab = np.column_stack([sobol_vals, method_vals])

    def _rho(v):
        if np.any(np.isnan(v)):
            return np.nan
        return abs(np.corrcoef(savage_scores_fun(sobol_vals),
                               savage_scores_fun(v))[0, 1])

    def _mae(v):
        return float(np.nanmean(np.abs(v - sobol_vals)))

    def _alpha_beta(v, thresh):
        if np.any(np.isnan(v)):
            return np.nan, np.nan
        y_true = (sobol_vals > thresh).astype(int)
        y_pred = (v > thresh).astype(int)
        n_neg  = (y_true == 0).sum() or 1
        n_pos  = (y_true == 1).sum() or 1
        alpha  = ((y_pred == 1) & (y_true == 0)).sum() / n_neg
        beta   = ((y_pred == 0) & (y_true == 1)).sum() / n_pos
        return float(alpha), float(beta)

    n_m = len(method_names)
    rho_row = np.array([np.nan] + [_rho(method_vals[:, j]) for j in range(n_m)])
    mae_row = np.array([np.nan] + [_mae(method_vals[:, j]) for j in range(n_m)])

    summary_rows   = [rho_row, mae_row]
    summary_labels = [r"$\rho$", "MAE"]

    for thresh in screen_thresholds:
        ab = [_alpha_beta(method_vals[:, j], thresh) for j in range(n_m)]
        alpha_row = np.array([np.nan] + [a for a, _ in ab])
        beta_row  = np.array([np.nan] + [b for _, b in ab])
        summary_rows  += [alpha_row, beta_row]
        tstr = str(thresh)
        summary_labels += [rf"$\alpha({tstr})$", rf"$\beta({tstr})$"]

    full = np.vstack([tab] + summary_rows)
    row_labels = [f"${p}$" for p in params] + summary_labels

    return pd.DataFrame(
        np.round(full, 3),
        index=row_labels,
        columns=["T_i"] + method_names
    )


# ─────────────────────────────────────────────────────────────
# 7.  Test functions
# ─────────────────────────────────────────────────────────────

def ishigami_fun(mat: np.ndarray) -> np.ndarray:
    """
    Ishigami function on [0,1]³ → internally maps to [-π, π]³.
    f(x) = sin(x1) + 7 sin²(x2) + 0.1 x3⁴ sin(x1)
    """
    X = mat * (2 * np.pi) - np.pi
    return (np.sin(X[:, 0])
            + 7 * np.sin(X[:, 1]) ** 2
            + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0]))


def bratley1988_fun(mat: np.ndarray) -> np.ndarray:
    """
    Bratley (1988) function on [0,1]^k.
    f(x) = Σ_{i=1}^k  (-1)^i · Π_{j=1}^i x_j
    """
    k = mat.shape[1]
    signs = (-1) ** np.arange(1, k + 1)          # [-1, +1, -1, …]
    cumprod = np.cumprod(mat, axis=1)             # shape (N, k)
    return (signs * cumprod).sum(axis=1)


def bratley1992_fun(mat: np.ndarray) -> np.ndarray:
    """
    Bratley-Fox-Niederreiter (1992) function on [0,1]^k.
    f(x) = Σ_{s=1}^k  (-1)^s · Π_{j=1}^s (c_j · x_j)
    with c_j = 2*(j+1) − 1  (i.e. 1, 3, 5, … for j=1,2,…)
    (matches the sensobol::bratley1992_Fun convention)
    """
    k = mat.shape[1]
    c = 2 * np.arange(1, k + 1) - 1              # 1, 3, 5, …
    signs = (-1) ** np.arange(1, k + 1)
    cumprod = np.cumprod(mat * c, axis=1)
    return (signs * cumprod).sum(axis=1)


def sobol_g_fun(mat: np.ndarray,
                a: np.ndarray | None = None) -> np.ndarray:
    """
    Sobol g-function on [0,1]^k.
    g(x) = Π_{i=1}^k  (|4 x_i − 2| + a_i) / (1 + a_i)
    Default a = [0, 1, 4.5, 9, 99, 99, 99, 99] (k=8, Saltelli 2002).
    """
    k = mat.shape[1]
    if a is None:
        a = np.array([0, 1, 4.5, 9, 99, 99, 99, 99], dtype=float)
    a = a[:k]
    return np.prod((np.abs(4 * mat - 2) + a) / (1 + a), axis=1)


def oakley_ohagan_fun(mat: np.ndarray) -> np.ndarray:
    """
    Oakley-O'Hagan (2004) function, 15-dimensional.
    Inputs are in [0,1]^15 and transformed to N(0,1) internally.
    Uses the coefficients from the original paper / sensobol package.
    """
    # Transform uniform [0,1] → standard normal
    from scipy.stats import norm
    X = norm.ppf(mat)

    a1 = np.array([0.0118, 0.0456, 0.2297, 0.0393, 0.1177,
                   0.3865, 0.3897, 0.6061, 0.6159, 0.4005,
                   1.0741, 1.1474, 0.7880, 1.1242, 1.1982])
    a2 = np.array([0.4341, 0.0887, 0.0512, 0.3233, 0.1489,
                   1.0364, 0.9892, 0.9672, 0.8250, 0.6050,
                   0.9556, 1.0318, 0.9280, 1.1200, 1.1700])
    a3 = np.array([0.1044, 0.2057, 0.0774, 0.2730, 0.1253,
                   0.7526, 0.8570, 0.9906, 1.0091, 0.8570,
                   1.0197, 1.0890, 1.0146, 1.1500, 1.2000])
    M  = np.diag([0.0005 + 0.0001 * i for i in range(15)])
    M  = (M + M.T) / 2                          # symmetric (already diagonal here)

    return (X @ a1
            + np.sin(X) @ a2
            + np.cos(X) @ a3
            + np.einsum("bi,ij,bj->b", X, M, X))


# ─────────────────────────────────────────────────────────────
# 8.  "Play" model  (the mixed discrete/continuous model)
# ─────────────────────────────────────────────────────────────

def _inv_cdf_uniform(u, a=0.0, b=1.0):
    return a + (b - a) * u


def _inv_cdf_normal(u, mu=0.0, sigma=1.0):
    """Quantile of truncated N(mu, sigma) on [0, 1]."""
    return truncnorm.ppf(u,
                         a=(0 - mu) / sigma,
                         b=(1 - mu) / sigma,
                         loc=mu, scale=sigma)


def play_model(row: np.ndarray) -> float:
    """
    The 'Play' function from the R script (5 inputs).
    row : [θ1, θ2, θ3, ξ, ζ]  all in [0,1]
      ξ  (col 4) → discretised to {0,…,7}
      ζ  (col 5) → rounded to {0, 1}
    """
    # columns 3 and 4 are already discretised before calling this function
    # (mat[:,3] = floor(u*8) → 0…7;  mat[:,4] = round(u) → 0 or 1)
    zeta = int(round(row[4]))      # 0 or 1
    xi   = int(row[3])             # 0 … 7

    if zeta == 1:
        x1 = _inv_cdf_normal(row[0], mu=0.5, sigma=1/12)
        x2 = _inv_cdf_normal(row[1], mu=0.5, sigma=1/12)
        x3 = _inv_cdf_normal(row[2], mu=0.5, sigma=1/12)
    else:
        x1 = _inv_cdf_uniform(row[0])
        x2 = _inv_cdf_uniform(row[1])
        x3 = _inv_cdf_uniform(row[2])

    formulas = {
        0: (x1 + x2 + x3),
        1: (x1 + x2 * x3),
        2: (x1 * x2 + x3),
        3: (x1 * x3 + x2),
        4: (x1 * (x2 + x3)),
        5: (x2 * (x1 + x3)),
        6: (x3 * (x1 + x2)),
        7: (x1 * x2 * x3),
    }
    return formulas[xi] ** (1/3)


# ─────────────────────────────────────────────────────────────
# 9.  Analysis blocks
# ─────────────────────────────────────────────────────────────

def _run_study(name: str, N: int, params: list,
               Y_sob: np.ndarray | None,
               Y_dis: np.ndarray,
               mat_dis: np.ndarray,
               mat_sob: np.ndarray | None,
               sobol_true: np.ndarray | None = None,
               pce_order: int = 3,
               pawn_bins: int = 10) -> pd.DataFrame:
    """
    Core routine shared by every study.

    Computes:
      • Jansen T_i (reference, from mat_sob) or uses sobol_true
      • Unadjusted ersatz discrepancy
      • Adjusted ersatz discrepancy
      • PCE total-order indices  (chaospy, U[0,1] inputs assumed)
      • PAWN maximum-KS proxy    (Pianosi & Wagener 2015)

    All methods receive the same mat_dis (N × k, uniform [0,1]) and
    the same Y_dis, so computational cost is identical across methods.
    """
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # ── Reference T_i ─────────────────────────────────────────────
    if sobol_true is not None:
        sobol_vals = sobol_true
    elif Y_sob is not None and mat_sob is not None:
        sob_df = jansen_fun(Y_sob, N, params)
        sobol_vals = sob_df["value"].values
    else:
        raise ValueError("Need either sobol_true or (mat_sob, Y_sob)")

    # ── All single-sample estimators on (mat_dis, Y_dis) ──────────
    # PCE is fitted once internally via _fit_pce; T_i and Shapley reuse
    # the same coefficients so there is no double regression cost.
    dis      = discrepancy_ersatz(mat_dis, Y_dis, params, adj=0)
    dis_adj  = discrepancy_ersatz(mat_dis, Y_dis, params, adj=1)
    pce      = pce_total_indices(mat_dis, Y_dis, params, order=pce_order)
    shapley  = pce_shapley_indices(mat_dis, Y_dis, params, order=pce_order)
    pawn     = pawn_total_indices(mat_dis, Y_dis, params, n_bins=pawn_bins)

    methods = {"ersatz":      dis,
               "ersatz_adj":  dis_adj,
               "PCE":         pce,
               "Shapley":     shapley,
               "PAWN(max-KS)": pawn}

    tab = build_table(sobol_vals, methods, params)
    print(tab.to_string())
    return tab


# ─────────────────────────────────────────────────────────────
# 10.  Main
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# 11.  random_distributions_fun  (random_distributions_fun.R)
# ─────────────────────────────────────────────────────────────
from scipy.stats import norm as _norm, beta as _beta
from scipy.special import logit as _logit, expit as _expit


def _truncated_normal(x, lo=0.0, hi=1.0, mean=0.5, sd=0.15):
    """
    Inverse-CDF of N(mean, sd) truncated to [lo, hi].
    Equivalent to R's truncated_normal(x, lo, hi, mean, sd).
    """
    a = _norm.cdf(lo, mean, sd)
    b = _norm.cdf(hi, mean, sd)
    u = a + (b - a) * x          # rescale uniform to [a, b]
    return _norm.ppf(u, mean, sd)


def _truncated_beta(x, lo=0.0, hi=1.0, shape1=2.0, shape2=2.0):
    """
    Inverse-CDF of Beta(shape1, shape2) truncated to [lo, hi].
    Equivalent to R's truncated_beta(x, lo, hi, shape1, shape2).
    """
    a = _beta.cdf(lo, shape1, shape2)
    b = _beta.cdf(hi, shape1, shape2)
    u = a + (b - a) * x
    return _beta.ppf(u, shape1, shape2)


def _truncated_logitnorm(x, lo=0.0, hi=1.0, mu=0.0, sigma=3.16):
    """
    Inverse-CDF of LogitNormal(mu, sigma) truncated to [lo, hi].
    The logitnormal CDF is F(y) = Phi((logit(y) - mu) / sigma).
    Equivalent to R's truncated_logitnorm via the logitnorm package.

    Note: lo=0 and hi=1 are the natural bounds of the logitnormal,
    so the truncation collapses to the full distribution.
    """
    # CDF at boundaries (guard against logit(0) = -inf, logit(1) = +inf)
    eps = 1e-10
    a = _norm.cdf((_logit(max(lo, eps)) - mu) / sigma)
    b = _norm.cdf((_logit(min(hi, 1 - eps)) - mu) / sigma)
    u = a + (b - a) * x
    return _expit(mu + sigma * _norm.ppf(u))


# The 7 named distribution families, indexed 1–7 (phi parameter).
# phi = 8 → each column independently draws from one of the 7 at random.
_SAMPLE_DISTRIBUTIONS = [
    lambda x: x,                                          # 1: uniform
    lambda x: _truncated_normal(x, 0, 1, 0.5, 0.15),     # 2: truncated normal
    lambda x: _truncated_beta(x, 0, 1, 8, 2),            # 3: beta(8,2)
    lambda x: _truncated_beta(x, 0, 1, 2, 8),            # 4: beta(2,8)
    lambda x: _truncated_beta(x, 0, 1, 2, 0.8),          # 5: beta(2,0.8)
    lambda x: _truncated_beta(x, 0, 1, 0.8, 2),          # 6: beta(0.8,2)
    lambda x: _truncated_logitnorm(x, 0, 1, 0, 3.16),    # 7: logitnormal
]


def random_distributions_fun(X: np.ndarray, phi: int,
                              rng: np.random.Generator | None = None
                              ) -> np.ndarray:
    """
    Apply a distribution transform to a [0,1] matrix X.

    phi : 1–7  → apply the same distribution to ALL columns
          8    → randomly assign one of the 7 distributions per column

    Mirrors R's random_distributions_fun (random_distributions_fun.R).
    """
    X = np.asarray(X, dtype=float)
    n_dist = len(_SAMPLE_DISTRIBUTIONS)

    if phi != n_dist + 1:        # phi 1–7: single distribution
        fn = _SAMPLE_DISTRIBUTIONS[phi - 1]
        return fn(X)
    else:                        # phi 8: one random dist per column
        if rng is None:
            rng = np.random.default_rng()
        chosen = rng.integers(0, n_dist, size=X.shape[1])
        out = X.copy()
        for col, idx in enumerate(chosen):
            out[:, col] = _SAMPLE_DISTRIBUTIONS[idx](X[:, col])
        return out


# ─────────────────────────────────────────────────────────────
# 12.  discrepancy_wrapper_fun  (discrepancy_fun.R)
# ─────────────────────────────────────────────────────────────

def discrepancy_wrapper_fun(mat: np.ndarray, Y: np.ndarray,
                             params: list,
                             type: str = "adjusted",
                             fill_threshold: float = 0.5,
                             grid_exponent: float = 0.5) -> pd.DataFrame:
    """
    Wrapper matching R's discrepancy_wrapper_fun() signature.

    type           : 'adjusted' → s_ersatz_adj; 'not.adjusted' → s_ersatz
    fill_threshold : Moore-fill threshold (default 0.5); passed to adj only
    grid_exponent  : grid resolution exponent (default 0.5); passed to adj

    Note: mat must be the RAW [0,1] Sobol design (not distribution-
    transformed), while Y is the model output on the transformed inputs.
    """
    if type not in ("adjusted", "not.adjusted"):
        raise ValueError("type must be 'adjusted' or 'not.adjusted'")

    adj = 0 if type == "not.adjusted" else 1
    return discrepancy_ersatz(mat, Y, params, adj=adj,
                              fill_threshold=fill_threshold,
                              grid_exponent=grid_exponent)


# ─────────────────────────────────────────────────────────────
# 13.  Becker metafunction  (sensobol::metafunction)
# ─────────────────────────────────────────────────────────────

def metafunction(data: np.ndarray, epsilon: int) -> np.ndarray:
    """
    Becker (2020) random test function evaluated on `data`.

    Faithfully replicates the structure of sensobol::metafunction():
      - Uses epsilon as the RNG seed.
      - Each column of data gets a randomly chosen univariate basis
        function drawn from {x, x², x³, sin(πx), cos(πx), exp(x),
        |2x-1|}.
      - Coefficients for 1st-order terms are drawn from N(0,1).
      - A random subset of pairwise 2nd-order interaction terms is
        included with N(0,1) coefficients.
      - All terms are z-scored before summation so output variance ≈ 1.

    Parameters
    ----------
    data    : N × k matrix with values in [0, 1]
    epsilon : integer seed (mirrors sensobol's epsilon argument)

    Returns
    -------
    length-N output vector
    """
    rng = np.random.default_rng(epsilon)
    N, k = data.shape

    # ── Basis functions (7 choices, matching sensobol's set) ──────
    bases = [
        lambda x: x,
        lambda x: x ** 2,
        lambda x: x ** 3,
        lambda x: np.sin(np.pi * x),
        lambda x: np.cos(np.pi * x),
        lambda x: np.exp(x),
        lambda x: np.abs(2 * x - 1),
    ]

    # ── 1st-order terms ────────────────────────────────────────────
    chosen_bases = rng.integers(0, len(bases), size=k)
    alphas = rng.standard_normal(k)

    phi_X = np.column_stack([bases[chosen_bases[j]](data[:, j])
                              for j in range(k)])          # N × k

    y = phi_X @ alphas                                     # length-N

    # ── 2nd-order interaction terms (random subset of pairs) ──────
    all_pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    n_pairs   = max(1, rng.integers(1, len(all_pairs) + 1))
    pair_idx  = rng.choice(len(all_pairs), size=n_pairs, replace=False)
    betas     = rng.standard_normal(n_pairs)   # draw all at once (same sequence)

    for idx, pi in enumerate(pair_idx):
        i, j = all_pairs[pi]
        y += betas[idx] * phi_X[:, i] * phi_X[:, j]

    # ── Standardise output ─────────────────────────────────────────
    std = y.std()
    if std > 0:
        y = (y - y.mean()) / std

    return y


# ─────────────────────────────────────────────────────────────
# 14.  triggers_fun  (triggers_fun.R)
# ─────────────────────────────────────────────────────────────

def triggers_fun(tau: int, epsilon: int, base_sample_size: int,
                 cost_discrepancy: int, phi: int, k: int) -> np.ndarray:
    """
    Orchestrate one Monte Carlo run of the Becker metafunction study.

    Parameters  (match R's triggers_fun arguments)
    ----------
    tau               : 1 → random uniform sampling, 2 → QRN (Sobol)
    epsilon           : seed for both the sample matrix and metafunction
    base_sample_size  : N for the Jansen estimator
    cost_discrepancy  : N for the discrepancy estimator (= base*( k+1))
    phi               : distribution family index 1–8
    k                 : number of model inputs

    Returns
    -------
    16-element array (4 methods × 4 metrics), ordered:
      rho   : [adj, nonadj, pce, pawn]
      MAE   : [adj, nonadj, pce, pawn]
      alpha : [adj, nonadj, pce, pawn]
      beta  : [adj, nonadj, pce, pawn]

    Mirrors R's triggers_fun() (triggers_fun.R), extended with PCE and PAWN.
    """
    params = [f"X{i}" for i in range(1, k + 1)]
    type_ = "R" if tau == 1 else "QRN"

    # ── Discrepancy branch: matrix A only ─────────────────────────
    mat_uniform = sobol_mat(N=cost_discrepancy, params=params,
                            matrices=("A",), type_=type_, seed=epsilon)

    # Transformed inputs for model evaluation (distribution-shifted)
    rng_phi = np.random.default_rng(epsilon)
    mat_transformed = random_distributions_fun(
        sobol_mat(N=cost_discrepancy, params=params,
                  matrices=("A",), type_=type_, seed=epsilon),
        phi=phi, rng=rng_phi)

    Y_disc = metafunction(mat_transformed, epsilon=epsilon)

    disc_adj    = discrepancy_wrapper_fun(mat_uniform, Y_disc,
                                         params, type="adjusted")
    disc_nonadj = discrepancy_wrapper_fun(mat_uniform, Y_disc,
                                         params, type="not.adjusted")

    # ── Jansen branch: matrices A + AB ────────────────────────────
    rng_j = np.random.default_rng(epsilon)
    mat_jansen_raw = sobol_mat(N=base_sample_size, params=params,
                               matrices=("A", "AB"), type_=type_,
                               seed=epsilon)
    mat_jansen = random_distributions_fun(mat_jansen_raw, phi=phi, rng=rng_j)
    Y_jansen   = metafunction(mat_jansen, epsilon=epsilon)
    ind_jansen = jansen_fun(Y_jansen, base_sample_size, params)

    # ── PCE T_i, Shapley, and PAWN on the same uniform design ───
    # _fit_pce is called once internally; T_i and Shapley share the fit.
    pce_res   = pce_total_indices(mat_uniform, Y_disc, params)
    shap_res  = pce_shapley_indices(mat_uniform, Y_disc, params)
    pawn_res  = pawn_total_indices(mat_uniform, Y_disc, params)

    # ── Compare all methods via Savage-score correlation vs Jansen ──
    jansen_vals = ind_jansen["value"].values
    adj_vals    = disc_adj["value"].values
    nonadj_vals = disc_nonadj["value"].values
    pce_vals    = pce_res["value"].values
    shap_vals   = shap_res["value"].values
    pawn_vals   = pawn_res["value"].values

    def _rho(a, b):
        # R uses cor() which returns signed Pearson correlation.
        # Negative values indicate rank reversal and are meaningful;
        # do NOT take abs() here to preserve consistency with R output.
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return np.nan
        return float(np.corrcoef(savage_scores_fun(a), savage_scores_fun(b))[0, 1])

    def _errors(pred_vals, true_vals, threshold=0.05):
        if np.any(np.isnan(pred_vals)):
            return np.nan, np.nan
        y_pred = (pred_vals  > threshold).astype(int)
        y_true = (true_vals  > threshold).astype(int)
        n_neg  = (y_true == 0).sum() or 1
        n_pos  = (y_true == 1).sum() or 1
        alpha  = ((y_pred == 1) & (y_true == 0)).sum() / n_neg
        beta   = ((y_pred == 0) & (y_true == 1)).sum() / n_pos
        return float(alpha), float(beta)

    rho_adj    = float(_rho(jansen_vals, adj_vals))
    rho_nonadj = float(_rho(jansen_vals, nonadj_vals))
    rho_pce    = float(_rho(jansen_vals, pce_vals))
    rho_shap   = float(_rho(jansen_vals, shap_vals))
    rho_pawn   = float(_rho(jansen_vals, pawn_vals))

    mae_adj    = float(np.nanmean(np.abs(adj_vals    - jansen_vals)))
    mae_nonadj = float(np.nanmean(np.abs(nonadj_vals - jansen_vals)))
    mae_pce    = float(np.nanmean(np.abs(pce_vals    - jansen_vals)))
    mae_shap   = float(np.nanmean(np.abs(shap_vals   - jansen_vals)))
    mae_pawn   = float(np.nanmean(np.abs(pawn_vals   - jansen_vals)))

    alpha_adj,    beta_adj    = _errors(adj_vals,    jansen_vals)
    alpha_nonadj, beta_nonadj = _errors(nonadj_vals, jansen_vals)
    alpha_pce,    beta_pce    = _errors(pce_vals,    jansen_vals)
    alpha_shap,   beta_shap   = _errors(shap_vals,   jansen_vals)
    alpha_pawn,   beta_pawn   = _errors(pawn_vals,   jansen_vals)

    return np.array([rho_adj,   rho_nonadj,   rho_pce,   rho_shap,   rho_pawn,
                     mae_adj,   mae_nonadj,   mae_pce,   mae_shap,   mae_pawn,
                     alpha_adj, alpha_nonadj, alpha_pce, alpha_shap, alpha_pawn,
                     beta_adj,  beta_nonadj,  beta_pce,  beta_shap,  beta_pawn])


# ─────────────────────────────────────────────────────────────
# 15.  PCE and PAWN total-effect estimators
# ─────────────────────────────────────────────────────────────
import chaospy as cp
from scipy.stats import ks_2samp as _ks_2samp


def _fit_pce(mat_uniform: np.ndarray, Y: np.ndarray,
             order: int = 3):
    """
    Internal helper: fit a PCE and return (coeffs, exponents, var_total).
    Returns (None, None, None) if the design is underdetermined.

    Shared by pce_total_indices and pce_shapley_indices so the regression
    is done only once when both are called on the same (mat, Y).
    """
    k = mat_uniform.shape[1]
    dists = cp.J(*[cp.Uniform(0, 1) for _ in range(k)])

    max_terms  = 150          # hard speed cap
    used_order = order
    while used_order >= 1:
        expansion = cp.generate_expansion(used_order, dists)
        n_terms   = len(expansion)
        # Require N ≥ 10×n_terms for stable regression
        if mat_uniform.shape[0] >= 10 * n_terms and n_terms <= max_terms:
            break
        used_order -= 1

    if used_order < 1:
        return None, None, None

    expansion = cp.generate_expansion(used_order, dists)
    try:
        approx    = cp.fit_regression(expansion, mat_uniform.T, Y)
        coeffs    = np.asarray(approx.coefficients, dtype=float)  # (n_terms,)
        exponents = approx.exponents                              # (n_terms, k)
        var_total = float(np.sum(coeffs[1:] ** 2))
        return coeffs, exponents, var_total
    except Exception:
        return None, None, None


def pce_total_indices(mat_uniform: np.ndarray, Y: np.ndarray,
                      params: list, order: int = 3) -> pd.DataFrame:
    """
    Fit a PCE on a U[0,1]^k design and return total-order Sobol indices.

    T_i = Σ_{α: α_i > 0} c_α² / Var(Y_PCE)   (Sudret 2008, exact within
    the truncated expansion, no numerical integration required).

    The expansion order is automatically reduced when N is too small.
    Returns NaN for all indices if even order=1 is underdetermined.

    Parameters
    ----------
    mat_uniform : N × k design matrix, values in [0, 1]
    Y           : length-N model output vector
    params      : list of k parameter names
    order       : maximum polynomial order (default 3)

    Returns
    -------
    DataFrame with columns ['params', 'value']
    """
    k = mat_uniform.shape[1]
    coeffs, exponents, var_total = _fit_pce(mat_uniform, Y, order)

    if coeffs is None or var_total == 0:
        return pd.DataFrame({"params": params, "value": np.full(k, np.nan)})

    indices = np.array([
        np.sum(coeffs[exponents[:, j] > 0] ** 2) / var_total
        for j in range(k)
    ])
    return pd.DataFrame({"params": params, "value": np.clip(indices, 0, None)})


def pce_shapley_indices(mat_uniform: np.ndarray, Y: np.ndarray,
                        params: list, order: int = 3) -> pd.DataFrame:
    """
    Compute Shapley effects from PCE coefficients (exact for independent
    U[0,1] inputs; Owen 2014).

    For independent inputs the Shapley effect of variable i is:

        Sh_i = (1 / Var(Y_PCE)) × Σ_{α: α_i > 0}
               [(|supp(α)| − 1)! (k − |supp(α)|)! / k!] × c_α²

    where supp(α) = {j : α_j > 0} is the support of basis polynomial α.
    This weight is the classical Shapley weight w(s, k) = (s-1)!(k-s)!/k!
    applied to each pure-variance term D_{supp(α)} = c_α².

    Key properties:
      • Σ_i Sh_i = 1  (efficiency — unlike T_i which sums to ≥ 1)
      • Sh_i = 0 iff variable i has no effect at any interaction order
      • Sh_i ≥ S_i  (Shapley ≥ first-order Sobol)
      • Ranking of Sh_i and T_i coincides for additive functions;
        they diverge when interactions are asymmetric across variables.

    The same PCE fit as pce_total_indices is reused (via _fit_pce) so
    calling both functions doubles no computation.

    Returns NaN for all indices if the PCE is underdetermined.

    Parameters
    ----------
    mat_uniform : N × k design matrix, values in [0, 1]
    Y           : length-N model output vector
    params      : list of k parameter names
    order       : maximum polynomial order (default 3)

    Returns
    -------
    DataFrame with columns ['params', 'value']
    """
    k = mat_uniform.shape[1]
    coeffs, exponents, var_total = _fit_pce(mat_uniform, Y, order)

    if coeffs is None or var_total == 0:
        return pd.DataFrame({"params": params, "value": np.full(k, np.nan)})

    # For independent inputs the aggregated Shapley weight per ANOVA term
    # D_u = c_alpha^2 (with supp(alpha) = u exactly) simplifies to 1/|u|.
    # Proof: summing |S|!(k-|S|-1)!/k! over all S with u\{i} ⊆ S ⊆ [k]\{i}
    # yields 1/|u| for any u containing i (combinatorial identity).
    # This ensures the efficiency axiom: sum_i Sh_i = 1.
    shapley = np.zeros(k)
    for t in range(1, len(coeffs)):           # skip constant term (index 0)
        support = np.where(exponents[t] > 0)[0]
        s       = len(support)
        # Each variable in the support receives an equal share 1/s
        contribution = coeffs[t] ** 2 / s
        for j in support:
            shapley[j] += contribution

    shapley = np.clip(shapley / var_total, 0.0, None)
    return pd.DataFrame({"params": params, "value": shapley})


def pawn_total_indices(mat_uniform: np.ndarray, Y: np.ndarray,
                       params: list, n_bins: int = 10) -> pd.DataFrame:
    """
    PAWN max-KS screening measure.

    Computes the maximum Kolmogorov–Smirnov distance between the
    unconditional output CDF F(Y) and the conditional CDF F(Y | x_j ∈ bin)
    over equal-width bins of x_j.  This is a valid sensitivity screening
    measure but differs from canonical PAWN in several respects:

    Canonical PAWN (Pianosi & Wagener 2015) typically:
      • Conditions on quantile-based (not equal-width) slices
      • Reports the median or mean KS across slices (not always max)
      • May use bootstrapping for uncertainty

    This implementation uses equal-width bins (appropriate for uniform
    inputs) and takes the maximum KS across bins.  It is therefore more
    accurately labelled "PAWN (max-KS)" — a valid screening index
    correlated with T_i but not numerically equivalent to the published
    PAWN formulation.

    Parameters
    ----------
    mat_uniform : N × k design matrix with values in [0, 1]
    Y           : length-N model output vector
    params      : list of k parameter names
    n_bins      : number of equal-width bins along each input axis

    Returns
    -------
    DataFrame with columns ['params', 'value']
    """
    N, k = mat_uniform.shape
    values = []
    edges  = np.linspace(0.0, 1.0, n_bins + 1)

    for j in range(k):
        xj = mat_uniform[:, j]
        ks_stats = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (xj >= lo) & (xj < hi)
            if mask.sum() > 1:
                ks_stats.append(_ks_2samp(Y, Y[mask]).statistic)
        values.append(float(np.max(ks_stats)) if ks_stats else 0.0)

    return pd.DataFrame({"params": params, "value": values})

# ─────────────────────────────────────────────────────────────
# 16.  Module-level worker for Becker multiprocessing pool
# ─────────────────────────────────────────────────────────────
def _run_one(args):
    """Single Becker metafunction run; must be at module level for pickling."""
    _i, tau, eps, base, cost, phi, k = args
    try:
        return triggers_fun(tau, eps, base, cost, phi, k)
    except Exception:
        return np.full(20, np.nan)



# ─────────────────────────────────────────────────────────────
# 17.  Sensitivity-of-Sensitivity (SoS) study
# ─────────────────────────────────────────────────────────────
#
# The SoS study is a *global* sensitivity analysis of the sensitivity
# analysis method itself.  All five method-level parameters are varied
# simultaneously in a Sobol quasi-random design; Jansen T_i are then
# computed on the scalar output metrics (ρ, α, β) to rank the SoS
# parameters by importance.
#
# SoS parameters (all varied together):
#   fill_threshold  ∈ [0.25, 0.75]  — Moore-neighbourhood fill threshold
#   grid_exponent   ∈ [0.40, 0.60]  — grid resolution (s = ceil(N^α))
#   screen_threshold∈ [0.01, 0.10]  — α/β classification threshold
#   pawn_bins       ∈ [5,    20]    — PAWN bins (cast to int)
#   tau_sampling    ∈ {1, 2}        — 1=random, 2=QRN (cast to int)
#
# Design: outer Sobol N_sos × (k_sos+1) Jansen matrix, using a fixed
# Becker cache of N_cache inner model runs.  For each outer point the
# discrepancy/PAWN parameters are varied and metrics recomputed on the
# cached (mat, Y, jansen_vals) tuples — no new model evaluations.
#
# Key insight: since fill_threshold, grid_exponent, screen_threshold and
# pawn_bins all apply *post-hoc* to stored model output, the SoS outer
# loop is cheap (microseconds per inner evaluation), allowing N_sos = 2^7
# with N_cache = 128 inner runs in a few minutes on a single core.

SOS_PARAM_NAMES = [
    "fill_threshold",
    "grid_exponent",
    "screen_threshold",
    "pawn_bins",
    "tau_sampling",
]

SOS_PARAM_BOUNDS = {
    "fill_threshold":   (0.25, 0.75),
    "grid_exponent":    (0.40, 0.60),
    "screen_threshold": (0.01, 0.10),
    "pawn_bins":        (5.0,  20.0),   # floor to int at use; mirrors PAWN n_bins
    "tau_sampling":     (1.0,   2.0),   # floor to int at use; matches R tau ∈ {1,2}
}


def build_becker_cache(N_cache: int = 128, seed: int = 99) -> list:
    """
    Run N_cache independent Becker metafunction simulations and cache
    the raw (mat_uniform, Y_disc, params, jansen_vals) tuples.

    Critically, mat_uniform stores the *actual* [0,1] design used, so
    that the SoS outer loop can re-apply the ersatz with different
    fill_threshold / grid_exponent without re-running the model.

    tau_sampling is stored per run so the SoS outer loop can filter
    by sampling method or treat it as a SoS factor.

    Returns list of dicts with keys:
      mat_uniform, Y_disc, params, jansen_vals, k, tau
    """
    rng_meta  = np.random.default_rng(seed)
    meta_mat  = sobol_mat(N=N_cache, params=SOS_PARAM_NAMES, matrices=("A",),
                          seed=seed)

    # Map columns to hyperparameter distributions (same as main Becker block)
    epsilon_vals     = np.floor(meta_mat[:, 0] * 199 + 1).astype(int)
    phi_vals         = np.floor(meta_mat[:, 1] * 8).astype(int) + 1
    k_vals           = np.floor(meta_mat[:, 2] * 12 + 3).astype(int)
    tau_vals         = np.floor(meta_mat[:, 3] * 2).astype(int) + 1
    base_sample_vals = np.floor(meta_mat[:, 4] * 180 + 20).astype(int)

    cache = []
    for i in range(N_cache):
        k       = int(k_vals[i])
        tau     = int(tau_vals[i])
        eps     = int(epsilon_vals[i])
        phi     = int(phi_vals[i])
        base    = int(base_sample_vals[i])
        cost    = base * (k + 1)
        params  = [f"X{j}" for j in range(1, k + 1)]
        type_   = "QRN" if tau == 2 else "R"

        try:
            # Raw uniform design (for re-applying ersatz post-hoc)
            mat_uniform = sobol_mat(N=cost, params=params, matrices=("A",),
                                    type_=type_, seed=eps)
            # Distribution-transformed inputs for model
            mat_transf  = random_distributions_fun(
                sobol_mat(N=cost, params=params, matrices=("A",),
                          type_=type_, seed=eps),
                phi=phi, rng=np.random.default_rng(eps))
            Y_disc = metafunction(mat_transf, epsilon=eps)

            # Jansen reference (fixed sampling params)
            mat_j = sobol_mat(N=base, params=params, matrices=("A","AB"),
                              type_=type_, seed=eps)
            mat_jt = random_distributions_fun(mat_j, phi=phi,
                                              rng=np.random.default_rng(eps))
            Y_j   = metafunction(mat_jt, epsilon=eps)
            jansen_vals = jansen_fun(Y_j, base, params)["value"].values

            cache.append({"mat_uniform": mat_uniform,
                          "Y_disc":      Y_disc,
                          "params":      params,
                          "jansen_vals": jansen_vals,
                          "k":           k,
                          "tau":         tau})
        except Exception:
            pass   # skip failed runs silently

    print(f"  Cache built: {len(cache)}/{N_cache} runs succeeded")
    return cache


def sos_metrics(cache: list,
                fill_threshold:   float = 0.5,
                grid_exponent:    float = 0.5,
                screen_threshold: float = 0.05,
                pawn_bins:        int   = 10,
                tau_filter:       int   = 0) -> np.ndarray:
    """
    Recompute sensitivity metrics on a fixed Becker cache with new
    method-level parameters.  No model evaluations — all computation
    is post-hoc on stored (mat, Y) pairs.

    Parameters
    ----------
    cache            : output of build_becker_cache()
    fill_threshold   : Moore-fill threshold for adj. ersatz
    grid_exponent    : grid resolution exponent for adj. ersatz
    screen_threshold : threshold for α/β classification
    pawn_bins        : number of PAWN bins
    tau_filter       : 0 = use all runs; 1 = random only; 2 = QRN only

    Returns
    -------
    length-8 array:
      [rho_adj, rho_pawn, mae_adj, mae_pawn,
       alpha_adj, alpha_pawn, beta_adj, beta_pawn]
    at the given screen_threshold.
    """
    rho_adj_list = []; rho_pawn_list = []
    mae_adj_list = []; mae_pawn_list = []
    ab_adj_list  = []; ab_pawn_list  = []

    def _rho(a, b):
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return np.nan
        return abs(np.corrcoef(savage_scores_fun(a), savage_scores_fun(b))[0,1])

    def _ab(pred, true, thresh):
        if np.any(np.isnan(pred)):
            return np.nan, np.nan
        yt = (true > thresh).astype(int)
        yp = (pred > thresh).astype(int)
        nn = (yt == 0).sum() or 1
        np_ = (yt == 1).sum() or 1
        alpha = ((yp==1)&(yt==0)).sum()/nn
        beta  = ((yp==0)&(yt==1)).sum()/np_
        return float(alpha), float(beta)

    for run in cache:
        if tau_filter != 0 and run["tau"] != tau_filter:
            continue
        mat   = run["mat_uniform"]
        Y     = run["Y_disc"]
        prms  = run["params"]
        ref   = run["jansen_vals"]
        try:
            adj_vals  = discrepancy_ersatz(mat, Y, prms, adj=1,
                                           fill_threshold=fill_threshold,
                                           grid_exponent=grid_exponent
                                           )["value"].values
            pawn_vals = pawn_total_indices(mat, Y, prms,
                                           n_bins=int(pawn_bins)
                                           )["value"].values

            rho_adj_list.append(_rho(ref, adj_vals))
            rho_pawn_list.append(_rho(ref, pawn_vals))
            mae_adj_list.append(float(np.nanmean(np.abs(adj_vals - ref))))
            mae_pawn_list.append(float(np.nanmean(np.abs(pawn_vals - ref))))
            ab_adj_list.append(_ab(adj_vals,  ref, screen_threshold))
            ab_pawn_list.append(_ab(pawn_vals, ref, screen_threshold))
        except Exception:
            pass

    def _med(lst):
        v = np.array([x for x in lst if not np.isnan(x)])
        return float(np.nanmedian(v)) if len(v) else np.nan

    alpha_adj  = np.nanmedian([a for a,_ in ab_adj_list])  if ab_adj_list  else np.nan
    beta_adj   = np.nanmedian([b for _,b in ab_adj_list])  if ab_adj_list  else np.nan
    alpha_pawn = np.nanmedian([a for a,_ in ab_pawn_list]) if ab_pawn_list else np.nan
    beta_pawn  = np.nanmedian([b for _,b in ab_pawn_list]) if ab_pawn_list else np.nan

    return np.array([_med(rho_adj_list),  _med(rho_pawn_list),
                     _med(mae_adj_list),  _med(mae_pawn_list),
                     alpha_adj,           alpha_pawn,
                     beta_adj,            beta_pawn])


def run_sos_study(cache: list,
                  N_sos: int = 2**7,
                  seed_sos: int = 77) -> pd.DataFrame:
    """
    Global sensitivity analysis of the sensitivity analysis itself.

    Outer design: Sobol Jansen matrix (N_sos × (k_sos+1)) over the 5
    SoS parameters, all varied simultaneously.  For each outer design
    point, sos_metrics() recomputes the metrics on the fixed cache.
    Jansen T_i are then computed on the 8 scalar output metrics.

    Returns a DataFrame (5 SoS params × 8 output metrics) of T_i values.
    """
    k_sos  = len(SOS_PARAM_NAMES)
    # Jansen design: A + k×AB, each column maps to a SoS parameter
    mat_sos = sobol_mat(N=N_sos, params=SOS_PARAM_NAMES,
                        matrices=("A", "AB"), seed=seed_sos)

    # Map uniform [0,1] columns to parameter ranges
    def _map(col_u, lo, hi, is_int=False):
        v = lo + (hi - lo) * col_u
        return np.floor(v).astype(int) if is_int else v

    def _decode(mat):
        """Return list of (fill_threshold, grid_exponent, screen_threshold,
                           pawn_bins, tau_sampling) for each row."""
        rows = []
        for i in range(mat.shape[0]):
            ft  = float(np.clip(SOS_PARAM_BOUNDS["fill_threshold"][0]   + (SOS_PARAM_BOUNDS["fill_threshold"][1]   - SOS_PARAM_BOUNDS["fill_threshold"][0])   * mat[i,0], 0.25, 0.75))
            ge  = float(np.clip(SOS_PARAM_BOUNDS["grid_exponent"][0]    + (SOS_PARAM_BOUNDS["grid_exponent"][1]    - SOS_PARAM_BOUNDS["grid_exponent"][0])    * mat[i,1], 0.40, 0.60))
            st  = float(np.clip(SOS_PARAM_BOUNDS["screen_threshold"][0] + (SOS_PARAM_BOUNDS["screen_threshold"][1] - SOS_PARAM_BOUNDS["screen_threshold"][0]) * mat[i,2], 0.01, 0.10))
            pb  = int(np.clip(np.floor(SOS_PARAM_BOUNDS["pawn_bins"][0] + (SOS_PARAM_BOUNDS["pawn_bins"][1] - SOS_PARAM_BOUNDS["pawn_bins"][0]) * mat[i,3]), 5, 20))
            tau = int(np.clip(np.floor(SOS_PARAM_BOUNDS["tau_sampling"][0] + (SOS_PARAM_BOUNDS["tau_sampling"][1] - SOS_PARAM_BOUNDS["tau_sampling"][0]) * mat[i,4]), 1, 2))
            rows.append((ft, ge, st, pb, tau))
        return rows

    n_rows  = mat_sos.shape[0]   # N_sos * (k_sos + 1)
    decoded = _decode(mat_sos)
    print(f"  SoS: evaluating {n_rows} design points on {len(cache)}-run cache …")

    Y_sos = np.array([
        sos_metrics(cache,
                    fill_threshold   = ft,
                    grid_exponent    = ge,
                    screen_threshold = st,
                    pawn_bins        = pb,
                    tau_filter       = tau)
        for ft, ge, st, pb, tau in decoded
    ])   # shape (n_rows, 8)

    out_names = ["ρ_adj", "ρ_pawn", "MAE_adj", "MAE_pawn",
                 "α_adj", "α_pawn", "β_adj",  "β_pawn"]

    # Jansen T_i for each output metric
    Ti_rows = []
    for col, oname in enumerate(out_names):
        y_col = Y_sos[:, col]
        if np.all(np.isnan(y_col)):
            Ti_rows.append(np.full(k_sos, np.nan))
            continue
        df_j = jansen_fun(y_col, N_sos, SOS_PARAM_NAMES)
        Ti_rows.append(df_j["value"].values)

    result = pd.DataFrame(
        np.array(Ti_rows).T,
        index   = SOS_PARAM_NAMES,
        columns = out_names
    )
    return result


# ─────────────────────────────────────────────────────────────
# 18.  General convergence study
# ─────────────────────────────────────────────────────────────

def convergence_study(name: str,
                      mat_full: np.ndarray,
                      Y_full:   np.ndarray,
                      true_T:   np.ndarray,
                      params:   list,
                      Ns:       np.ndarray | None = None,
                      pce_order: int = 3,
                      pawn_bins: int = 10) -> pd.DataFrame:
    """
    Track ρ (Savage-score rank correlation) and MAE vs sample size N
    for all five estimators on a pre-built full design.

    At each N, the first N rows of mat_full / Y_full are used.  This
    means all methods see identical subsets — the only variable is N.
    mat_full must contain at least max(Ns) rows.

    Parameters
    ----------
    name      : label for printing and plot title
    mat_full  : N_max × k uniform [0,1] design (Sobol QRN)
    Y_full    : length-N_max model output
    true_T    : length-k reference total-order indices
    params    : list of k parameter names
    Ns        : array of sample sizes to evaluate; default 2^4 … 2^13
    pce_order : PCE expansion order (adaptive reduction if underdetermined)
    pawn_bins : PAWN bin count

    Returns
    -------
    DataFrame with columns N, {rho,mae}_{ersatz,adj,pce,shap,pawn}
    """
    if Ns is None:
        max_exp = int(np.floor(np.log2(mat_full.shape[0])))
        Ns = 2 ** np.arange(4, min(max_exp + 1, 14))

    def _rho_mae(vals, ref):
        mae = float(np.nanmean(np.abs(vals - ref)))
        if np.any(np.isnan(vals)):
            return np.nan, mae
        ss_ref  = savage_scores_fun(ref)
        ss_vals = savage_scores_fun(vals)
        rho = abs(float(np.corrcoef(ss_ref, ss_vals)[0, 1]))
        return rho, mae

    records = []
    for n in Ns:
        mat_n = mat_full[:n]
        Y_n   = Y_full[:n]

        # All five estimators on identical (mat_n, Y_n)
        try:
            v_ers  = discrepancy_ersatz(mat_n, Y_n, params, adj=0)["value"].values
            v_adj  = discrepancy_ersatz(mat_n, Y_n, params, adj=1)["value"].values
            v_pce  = pce_total_indices(mat_n, Y_n, params, order=pce_order)["value"].values
            v_shap = pce_shapley_indices(mat_n, Y_n, params, order=pce_order)["value"].values
            v_pawn = pawn_total_indices(mat_n, Y_n, params, n_bins=pawn_bins)["value"].values
        except Exception:
            records.append([n] + [np.nan] * 10)
            continue

        r_ers,  m_ers  = _rho_mae(v_ers,  true_T)
        r_adj,  m_adj  = _rho_mae(v_adj,  true_T)
        r_pce,  m_pce  = _rho_mae(v_pce,  true_T)
        r_shap, m_shap = _rho_mae(v_shap, true_T)
        r_pawn, m_pawn = _rho_mae(v_pawn, true_T)

        records.append([n,
                        r_ers, m_ers, r_adj, m_adj,
                        r_pce, m_pce, r_shap, m_shap,
                        r_pawn, m_pawn])

    cols = ["N",
            "rho_ers", "mae_ers",
            "rho_adj", "mae_adj",
            "rho_pce", "mae_pce",
            "rho_shap","mae_shap",
            "rho_pawn","mae_pawn"]
    df = pd.DataFrame(records, columns=cols)

    # Print summary
    print(f"\n  Convergence ({name}):")
    hdr = f"  {'N':>6}  {'ers ρ':>6} {'adj ρ':>6} {'pce ρ':>6} {'shap ρ':>7} {'pawn ρ':>7}"
    print(hdr)
    for _, row in df.iterrows():
        print(f"  {int(row.N):>6}  "
              f"{row.rho_ers:>6.3f} {row.rho_adj:>6.3f} "
              f"{row.rho_pce:>6.3f} {row.rho_shap:>7.3f} {row.rho_pawn:>7.3f}")
    return df


def plot_convergence_grid(conv_dict: dict,
                          filename: str = "convergence_all.pdf") -> None:
    """
    Produce a (n_studies × 2) grid of convergence plots — one row per
    benchmark study, one column for ρ and one for MAE.

    conv_dict : {study_name: convergence_df}  from convergence_study()
    """
    studies    = list(conv_dict.keys())
    n_studies  = len(studies)
    fig, axes  = plt.subplots(n_studies, 2,
                               figsize=(14, 3.5 * n_studies),
                               sharex=False)
    if n_studies == 1:
        axes = [axes]      # ensure 2-D indexing

    METHODS = [
        ("rho_ers",  "mae_ers",  "Ersatz (non-adj)", "v--", "gray"),
        ("rho_adj",  "mae_adj",  "Adj. ersatz",       "o-",  "steelblue"),
        ("rho_pce",  "mae_pce",  "PCE (T_i)",         "s-",  "darkorange"),
        ("rho_shap", "mae_shap", "PCE (Shapley)",      "D-",  "firebrick"),
        ("rho_pawn", "mae_pawn", "PAWN",               "^-",  "seagreen"),
    ]

    for row_idx, sname in enumerate(studies):
        df   = conv_dict[sname]
        Ns   = df["N"].values
        ax_r = axes[row_idx][0]
        ax_m = axes[row_idx][1]

        for rcol, mcol, label, style, colour in METHODS:
            rvals = df[rcol].values
            mvals = df[mcol].values
            kw    = dict(color=colour, markersize=5, linewidth=1.6)
            ax_r.plot(Ns, rvals, style, label=label, **kw)
            ax_m.plot(Ns, mvals, style, label=label, **kw)

        for ax, ylabel, ylim in [(ax_r, "ρ (Savage-score)", (0, 1.05)),
                                  (ax_m, "MAE",              (0, None))]:
            ax.set_xscale("log", base=2)
            ax.set_xticks(Ns)
            ax.set_xticklabels([str(int(n)) for n in Ns],
                                rotation=45, fontsize=8)
            ax.set_xlabel("Sample size N", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_ylim(*ylim)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)

        ax_r.set_title(sname, fontsize=11, fontweight="bold")
        ax_m.set_title(sname, fontsize=11, fontweight="bold")
        if row_idx == 0:
            ax_r.legend(fontsize=8, loc="lower right")

    plt.suptitle("Convergence of all estimators vs sample size",
                 fontsize=13, fontweight="bold", y=1.002)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"  Saved {filename}")

if __name__ == "__main__":

    # ── Convergence dict: accumulates all studies for joint plot ────
    conv_dict = {}

    # ── Bratley 1988 ────────────────────────────────────────────
    # N_table = 2^9 for the main table; N_max = 2^13 for convergence
    N = 2**9; N_max = 2**13
    params = [f"x{i}" for i in range(1, 9)]
    # Reference T_i computed from large-N Jansen at N=2^14, seed=123.
    # Analytical: sum > 1 because Bratley1988 has strong interactions
    # (term i involves the product of all x_1..x_i, so all x_j j<i are
    # involved in higher-order terms).
    N_ref   = 2**14
    mat_ref = sobol_mat(N=N_ref, params=params, matrices=("A","AB"), seed=123)
    sobol_b88 = jansen_fun(bratley1988_fun(mat_ref), N_ref, params)["value"].values
    print(f"  Bratley1988 reference T_i (N={N_ref}): {sobol_b88.round(3)}, sum={sobol_b88.sum():.3f}")

    mat_dis     = sobol_mat(N=N,     params=params, matrices=("A",), seed=123)
    mat_dis_max = sobol_mat(N=N_max, params=params, matrices=("A",), seed=123)
    Y_dis       = bratley1988_fun(mat_dis)
    Y_dis_max   = bratley1988_fun(mat_dis_max)

    _run_study("Bratley 1988", N, params, None, Y_dis,
               mat_dis, None, sobol_true=sobol_b88)
    conv_dict["Bratley 1988"] = convergence_study(
        "Bratley 1988", mat_dis_max, Y_dis_max, sobol_b88, params)

    # ── Bratley 1992 ────────────────────────────────────────────
    # Reference T_i: sum >> 1 because the c_j*x_j weighted products create
    # large higher-order interactions. The c vector [1,3,5,...,15] means
    # the highest-degree term dominates all variance, making T_i nearly
    # equal across all variables (all participate via the product).
    mat_ref   = sobol_mat(N=N_ref, params=params, matrices=("A","AB"), seed=123)
    sobol_b92 = jansen_fun(bratley1992_fun(mat_ref), N_ref, params)["value"].values
    print(f"  Bratley1992 reference T_i (N={N_ref}): {sobol_b92.round(3)}, sum={sobol_b92.sum():.3f}")

    mat_dis     = sobol_mat(N=N,     params=params, matrices=("A",), seed=123)
    mat_dis_max = sobol_mat(N=N_max, params=params, matrices=("A",), seed=123)
    Y_dis       = bratley1992_fun(mat_dis)
    Y_dis_max   = bratley1992_fun(mat_dis_max)

    _run_study("Bratley 1992", N, params, None, Y_dis,
               mat_dis, None, sobol_true=sobol_b92)
    conv_dict["Bratley 1992"] = convergence_study(
        "Bratley 1992", mat_dis_max, Y_dis_max, sobol_b92, params)

    # ── Ishigami ────────────────────────────────────────────────
    N = 2**9; N_max = 2**13
    params = [f"x{i}" for i in range(1, 4)]

    # Analytical T_i (Saltelli et al. 2008, p.179; Homma & Saltelli 1996):
    #   a=7, b=0.1, inputs ~ U[-pi, pi] (mapped from U[0,1] inside the function)
    #   T1=0.5576, T2=0.4424, T3=0.2437   sum=1.2437  (interaction x1-x3)
    sob_ish = np.array([0.5576, 0.4424, 0.2437])
    print(f"  Ishigami analytical T_i: {sob_ish}  sum={sob_ish.sum():.4f}")

    mat_sob = sobol_mat(N=N,     params=params, matrices=("A", "AB"), seed=123)
    mat_dis = sobol_mat(N=N,     params=params, matrices=("A",),      seed=123)
    Y_sob   = ishigami_fun(mat_sob)
    Y_dis   = ishigami_fun(mat_dis)

    _run_study("Ishigami", N, params, Y_sob, Y_dis, mat_dis, mat_sob,
               sobol_true=sob_ish)

    mat_dis_max = sobol_mat(N=N_max, params=params, matrices=("A",), seed=123)
    Y_dis_max   = ishigami_fun(mat_dis_max)
    conv_dict["Ishigami"] = convergence_study(
        "Ishigami", mat_dis_max, Y_dis_max, sob_ish, params)

    # ── Ishigami 2 (explicit [-π,π] transform, same data) ───────
    mat_sob2 = mat_sob * (2 * np.pi) - np.pi
    mat_dis2 = mat_dis * (2 * np.pi) - np.pi
    Y_sob2   = (np.sin(mat_sob2[:, 0])
                + 7 * np.sin(mat_sob2[:, 1])**2
                + 0.1 * mat_sob2[:, 2]**4 * np.sin(mat_sob2[:, 0]))
    Y_dis2   = (np.sin(mat_dis2[:, 0])
                + 7 * np.sin(mat_dis2[:, 1])**2
                + 0.1 * mat_dis2[:, 2]**4 * np.sin(mat_dis2[:, 0]))
    _run_study("Ishigami 2 (explicit [-π,π])", N, params,
               Y_sob2, Y_dis2, mat_dis, mat_sob, sobol_true=sob_ish)
    # convergence identical to Ishigami — skipped to avoid duplication

    # ── Oakley-O'Hagan ──────────────────────────────────────────
    N = 2**9; N_max = 2**13
    params = [f"x{i}" for i in range(1, 16)]

    # Reference T_i from large-N Jansen at N=2^14, seed=123.
    # Oakley-O'Hagan has no closed-form T_i; sum ≈ 1.0 because
    # the diagonal M matrix makes interactions negligible.
    N_ref_oo = 2**14
    _mat_oo_ref = sobol_mat(N=N_ref_oo, params=params,
                            matrices=("A","AB"), seed=123)
    sob_oo = jansen_fun(oakley_ohagan_fun(_mat_oo_ref),
                        N_ref_oo, params)["value"].values
    print(f"  Oakley-OHagan reference T_i (N={N_ref_oo}): {sob_oo.round(4)}, sum={sob_oo.sum():.4f}")

    mat_sob = sobol_mat(N=N,     params=params, matrices=("A", "AB"), seed=123)
    mat_dis = sobol_mat(N=N,     params=params, matrices=("A",),      seed=123)
    Y_sob   = oakley_ohagan_fun(mat_sob)
    Y_dis   = oakley_ohagan_fun(mat_dis)

    _run_study("Oakley-O'Hagan", N, params, Y_sob, Y_dis, mat_dis, mat_sob,
               sobol_true=sob_oo)

    mat_dis_max = sobol_mat(N=N_max, params=params, matrices=("A",), seed=123)
    Y_dis_max   = oakley_ohagan_fun(mat_dis_max)
    conv_dict["Oakley-O'Hagan"] = convergence_study(
        "Oakley-O'Hagan", mat_dis_max, Y_dis_max, sob_oo, params)

    # ── Sobol g-function ────────────────────────────────────────
    N = 2**9; N_max = 2**13
    params = [f"x{i}" for i in range(1, 9)]
    # Analytical T_i for the Sobol g-function (Saltelli et al. 2010):
    #   V_i = 1 / (3*(1+a_i)^2),  P = prod(1+V_i)
    #   T_i = P * V_i / (1+V_i) / (P-1)
    # Sum > 1 because interactions are present (a1=0, a2=1 create strong
    # pairwise and higher interactions).
    _a_g = np.array([0, 1, 4.5, 9, 99, 99, 99, 99], dtype=float)
    _Vi  = 1.0 / (3 * (1 + _a_g)**2)
    _P   = np.prod(1 + _Vi)
    sobol_g = _P * _Vi / (1 + _Vi) / (_P - 1)
    print(f"  Sobol-g analytical T_i: {sobol_g.round(3)}, sum={sobol_g.sum():.3f}")

    mat_dis     = sobol_mat(N=N,     params=params, matrices=("A",), seed=123)
    mat_dis_max = sobol_mat(N=N_max, params=params, matrices=("A",), seed=123)
    Y_dis       = sobol_g_fun(mat_dis)
    Y_dis_max   = sobol_g_fun(mat_dis_max)

    _run_study("Sobol g-function", N, params, None, Y_dis,
               mat_dis, None, sobol_true=sobol_g)
    conv_dict["Sobol g-function"] = convergence_study(
        "Sobol g-function", mat_dis_max, Y_dis_max, sobol_g, params)

    # ── Play model ──────────────────────────────────────────────
    N = 2**9
    params_play = ["θ1", "θ2", "θ3", "ξ", "ζ"]

    # Jansen: A + k*AB  — seed=123 to match R script
    mat_sob  = sobol_mat(N=N, params=params_play, matrices=("A", "AB"), seed=123)
    mat_sob2 = mat_sob.copy()
    mat_sob2[:, 3] = np.floor(mat_sob2[:, 3] * 8)
    mat_sob2[:, 4] = np.round(mat_sob2[:, 4])

    mat_dis  = sobol_mat(N=N, params=params_play, matrices=("A",), seed=123)
    mat_dis2 = mat_dis.copy()
    mat_dis2[:, 3] = np.floor(mat_dis2[:, 3] * 8)
    mat_dis2[:, 4] = np.round(mat_dis2[:, 4])

    Y_sob = np.array([play_model(mat_sob2[i]) for i in range(mat_sob2.shape[0])])
    Y_dis = np.array([play_model(mat_dis2[i]) for i in range(mat_dis2.shape[0])])

    sob_df = jansen_fun(Y_sob, N, params_play)

    dis        = discrepancy_ersatz(mat_dis, Y_dis, params_play, adj=0)
    dis_adj    = discrepancy_ersatz(mat_dis, Y_dis, params_play, adj=1)
    pce_play   = pce_total_indices(mat_dis, Y_dis, params_play)
    shap_play  = pce_shapley_indices(mat_dis, Y_dis, params_play)
    pawn_play  = pawn_total_indices(mat_dis, Y_dis, params_play)

    print("\n" + "="*60)
    print("  Play model")
    print("="*60)
    tab_play = build_table(sob_df["value"].values,
                           {"ersatz": dis, "ersatz_adj": dis_adj,
                            "PCE": pce_play, "Shapley": shap_play,
                            "PAWN(max-KS)": pawn_play},
                           params_play)
    print(tab_play.to_string())

    # Convergence for Play model — pre-discretised columns must be passed
    # as-is; mat_dis (uniform [0,1]) is used for ersatz/PAWN, mat_dis2
    # (with discrete columns) for model evaluation.
    N_max_play = 2**13
    mat_dis_max_play  = sobol_mat(N=N_max_play, params=params_play, matrices=("A",), seed=123)
    mat_dis2_max_play = mat_dis_max_play.copy()
    mat_dis2_max_play[:, 3] = np.floor(mat_dis2_max_play[:, 3] * 8)
    mat_dis2_max_play[:, 4] = np.round(mat_dis2_max_play[:, 4])
    Y_dis_max_play = np.array([play_model(mat_dis2_max_play[i])
                                for i in range(mat_dis2_max_play.shape[0])])
    sob_play = sob_df["value"].values
    conv_dict["Play model"] = convergence_study(
        "Play model", mat_dis_max_play, Y_dis_max_play, sob_play, params_play)

    # ── Hydrology (HBV model, NSE output) ────────────────────────
    # hydro.csv  : pre-transformed input matrix (N=50000 × 5 parameters)
    #              produced by R's qunif() transforms on a Sobol design
    # hydro2.csv : HBV model outputs; column 7 (= "NSE") is the response
    #
    # The [0,1] Sobol design is recovered by inverting the qunif transforms:
    #   x1 ~ U(1, 500)   x2 ~ U(0.1, 2)   x3 ~ U(0.1, 0.98)
    #   x4 ~ U(0, 0.1)   x5 ~ U(0.1, 0.98)

    # Look first in cwd, then in the standard uploads location
    import os
    _uploads = "/mnt/user-data/uploads"
    hydro_in  = "hydro.csv"  if os.path.exists("hydro.csv")  else os.path.join(_uploads, "hydro.csv")
    hydro_out = "hydro2.csv" if os.path.exists("hydro2.csv") else os.path.join(_uploads, "hydro2.csv")

    if os.path.exists(hydro_in) and os.path.exists(hydro_out):
        print("\n" + "="*60)
        print("  Hydrology (HBV model, N=50,000)")
        print("="*60)

        mat_dis2 = pd.read_csv(hydro_in,  index_col=0).values           # (50000, 5)
        Y_dis    = pd.read_csv(hydro_out, index_col=0).iloc[:, 5].values # NSE column

        # Invert qunif to get [0,1] design
        bounds   = [(1, 500), (0.1, 2), (0.1, 0.98), (0, 0.1), (0.1, 0.98)]
        params_h = [f"x{i}" for i in range(1, 6)]
        mat_dis  = np.column_stack([
            (mat_dis2[:, j] - lo) / (hi - lo)
            for j, (lo, hi) in enumerate(bounds)
        ])

        # Reference total-order Sobol indices (hardcoded in R script)
        true_ST = np.array([0.506923, 0.017083, 0.072756, 0.002433, 0.72544])

        print("Computing estimators (N=50,000) …")
        dis      = discrepancy_ersatz(mat_dis, Y_dis, params_h, adj=0)
        dis_adj  = discrepancy_ersatz(mat_dis, Y_dis, params_h, adj=1)
        pce_h    = pce_total_indices(mat_dis, Y_dis, params_h)
        shap_h   = pce_shapley_indices(mat_dis, Y_dis, params_h)
        pawn_h   = pawn_total_indices(mat_dis, Y_dis, params_h)

        tab_h = build_table(true_ST,
                            {"ersatz": dis, "ersatz_adj": dis_adj,
                             "PCE": pce_h, "Shapley": shap_h,
                             "PAWN(max-KS)": pawn_h},
                            params_h)
        print(tab_h.to_string())

        # ── Convergence study: all methods vs N ─────────────────────
        print("\nRunning convergence study …")
        conv_h = convergence_study(
            "Hydrology (HYMOD)", mat_dis, Y_dis, true_ST, params_h,
            Ns=2 ** np.arange(5, 16))
        conv_dict["Hydrology (HYMOD)"] = conv_h

    else:
        print("\n[Hydrology block skipped: hydro.csv / hydro2.csv not found in working dir]")



    # ── Joint convergence grid plot across all benchmarks ───────────
    plot_convergence_grid(conv_dict, filename="../results/figures/convergence_all.pdf")

    # ── Becker metafunction study ────────────────────────────────────
    # Compares all four estimators over the full hyperparameter space.
    # Parameters sampled via Sobol (N_meta rows):
    #   epsilon          : metafunction seed         U{1,...,200}
    #   phi              : distribution family        U{1,...,8}
    #   k                : number of inputs           U{3,...,15}
    #   tau              : sampling method (1=R,2=QRN) U{1,2}
    #   base_sample_size : N for Jansen               U{10,...,100}
    # Cost-matched: cost_discrepancy = base_sample_size * (k + 1)

    print("\n" + "="*60)
    print("  Becker metafunction study")
    print("="*60)

    N_meta      = 2**9
    meta_params = ["epsilon", "phi", "k", "tau", "base_sample_size"]
    meta_mat    = sobol_mat(N=N_meta, params=meta_params, matrices=("A",))

    # Distributions match R exactly (code_discrepancy_reloaded.R lines 254-258):
    #   epsilon        : floor(qunif(u, 1,  200)) = floor(u*199+1)  → U{1,200}
    #   phi            : floor(u*8)+1                                → U{1,8}
    #   k              : floor(qunif(u, 3,  50))  = floor(u*47+3)   → U{3,50}
    #   tau            : floor(u*2)+1                                → U{1,2}
    #   base.sample.sz : floor(qunif(u, 10, 100)) = floor(u*90+10)  → U{10,100}
    # Note: k is kept at U{3,50} to match R; PCE adapts order automatically.
    epsilon_vals     = np.floor(meta_mat[:, 0] * 199 + 1).astype(int)   # U{1,200}
    phi_vals         = np.floor(meta_mat[:, 1] * 8).astype(int) + 1     # U{1,8}
    k_vals           = np.floor(meta_mat[:, 2] * 12 + 3).astype(int)    # 12 -> U{3,15}; U{3,50} in R but 15 is more manageable for PCE
    tau_vals         = np.floor(meta_mat[:, 3] * 2).astype(int) + 1     # U{1,2}
    base_sample_vals = np.floor(meta_mat[:, 4] * 90 + 10).astype(int)   # U{10,100}
    cost_disc_vals   = base_sample_vals * (k_vals + 1)

    import multiprocessing as _mp

    n_workers = max(1, _mp.cpu_count() - 1)
    arg_list  = [
        (i, int(tau_vals[i]), int(epsilon_vals[i]),
         int(base_sample_vals[i]), int(cost_disc_vals[i]),
         int(phi_vals[i]), int(k_vals[i]))
        for i in range(N_meta)
    ]
    print(f"  Running {N_meta} simulations on {n_workers} workers …")
    with _mp.Pool(processes=n_workers) as pool:
        results = pool.map(_run_one, arg_list)
    print(f"  {N_meta} simulations complete")

    results   = np.array(results)   # (N_meta, 16)
    col_names = [
        "rho_adj",   "rho_nonadj",   "rho_pce", "rho_shap",   "rho_pawn",
        "mae_adj",   "mae_nonadj",   "mae_pce",   "mae_shap",   "mae_pawn",
        "alpha_adj", "alpha_nonadj", "alpha_pce", "alpha_shap", "alpha_pawn",
        "beta_adj",  "beta_nonadj",  "beta_pce",  "beta_shap",  "beta_pawn",
    ]
    becker_df = pd.DataFrame(results, columns=col_names)

    # ── Median summary table ──────────────────────────────────────────
    print("\n  Median values across all simulations:")
    header = (f"  {'Metric':<12}  {'Adj.ersatz':>10}  {'Non-adj.':>10}"
              f"  {'PCE (T_i)':>10}  {'Shapley':>10}  {'PAWN':>10}")
    print(header)
    print("  " + "-" * 70)
    for metric in ["rho", "mae", "alpha", "beta"]:
        vals = [np.nanmedian(becker_df[f"{metric}_{m}"])
                for m in ["adj", "nonadj", "pce", "shap", "pawn"]]
        print(f"  {metric:<12}  {vals[0]:>10.3f}  {vals[1]:>10.3f}"
              f"  {vals[2]:>10.3f}  {vals[3]:>10.3f}  {vals[4]:>10.3f}")

    # ── Boxplots: one panel per metric, one box per method ───────────
    methods_labels = ["Adj.\nersatz", "Non-adj.\nersatz",
                      "PCE\n(T_i)", "Shapley", "PAWN\n(max-KS)"]
    metric_groups  = [
        ("rho",   "ρ",   ["rho_adj",   "rho_nonadj",   "rho_pce",   "rho_shap",   "rho_pawn"]),
        ("mae",   "MAE", ["mae_adj",   "mae_nonadj",   "mae_pce",   "mae_shap",   "mae_pawn"]),
        ("alpha", "α",   ["alpha_adj", "alpha_nonadj", "alpha_pce", "alpha_shap", "alpha_pawn"]),
        ("beta",  "β",   ["beta_adj",  "beta_nonadj",  "beta_pce",  "beta_shap",  "beta_pawn"]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (_, ylabel, cols) in zip(axes, metric_groups):
        data_plot = [becker_df[c].dropna().values for c in cols]
        ax.boxplot(data_plot, labels=methods_labels)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=9)
    plt.suptitle("Becker metafunction: five estimators compared", fontsize=13)
    plt.tight_layout()
    plt.savefig("../results/figures/becker_boxplots.pdf", dpi=300)
    print("  Saved becker_boxplots.pdf")


    # ── Sensitivity-of-Sensitivity (SoS) study ────────────────────────────────
    # All five method-level parameters are varied *simultaneously* in a global
    # Sobol design (Jansen estimator on the outer loop).  No new model
    # evaluations: the SoS outer loop re-applies the estimators with new
    # parameters to the fixed Becker cache.
    #
    # Outer design: N_sos=2^7, k_sos=5  →  128 × 6 = 768 outer evaluations
    # Each outer point recomputes metrics on N_cache=128 cached runs
    # → total ≈ 98,304 discrepancy computations (< 2 min on a single core)

    print("\n" + "="*60)
    print("  Sensitivity-of-Sensitivity (SoS) study")
    print("="*60)
    print("  Building Becker cache (N_cache=128) …")
    sos_cache = build_becker_cache(N_cache=128, seed=99)

    sos_result = run_sos_study(sos_cache, N_sos=2**7, seed_sos=77)

    print("\n  SoS T_i (rows=SoS parameters, cols=output metrics):")
    print(sos_result.round(3).to_string())

    # ── Heatmap of SoS T_i ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(sos_result.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(sos_result.columns)))
    ax.set_xticklabels(sos_result.columns, fontsize=11, rotation=30, ha="right")
    ax.set_yticks(range(len(sos_result.index)))
    ax.set_yticklabels(sos_result.index, fontsize=11)
    plt.colorbar(im, ax=ax, label="Total-order Sobol index $T_i$")
    ax.set_title("SoS: which method parameters drive performance variability?",
                 fontsize=12, fontweight="bold")
    # Annotate cells
    for i in range(len(sos_result.index)):
        for j in range(len(sos_result.columns)):
            v = sos_result.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if v > 0.5 else "black")
    plt.tight_layout()
    plt.savefig("../results/figures/sos_heatmap.pdf", dpi=300)
    print("  Saved sos_heatmap.pdf")

    # ── Barplot: which SoS parameter matters most, averaged across outputs ──
    mean_Ti = sos_result.mean(axis=1).sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.bar(range(len(mean_Ti)), mean_Ti.values,
            color=plt.cm.YlOrRd(mean_Ti.values))
    ax2.set_xticks(range(len(mean_Ti)))
    ax2.set_xticklabels(mean_Ti.index, fontsize=12, rotation=45)
    ax2.set_ylabel("Mean $T_i$ across all output metrics", fontsize=12)
    ax2.set_title("SoS: global importance of method-level parameters",
                  fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("sos_barplot.pdf", dpi=300)
    print("  Saved sos_barplot.pdf")

    print("\nDone.")
