"""
discrepancy.py -- Adjusted ersatz discrepancy and comparator estimators
=======================================================================

Library module accompanying

    Lo Piano, S., Lachi, A., Sheikholeslami, R., Puy, A., Roy, P. T.,
    Saltelli, A., "An Adjusted Ersatz Discrepancy for Global Sensitivity
    Analysis: Theory and Benchmarks" (Environmental Modelling & Software,
    revised version; submitted as "Improving Discrepancy Measures for Global
    Sensitivity Analysis").

Version 2.0 (revision).  Changes with respect to version 1.x are listed
in CHANGELOG.md; the most important ones are

  * PCE indices are now computed from the coefficients of an
    *orthonormal* polynomial basis (version 1.x read the monomial
    coefficients of the fitted polynomial, which is not a valid Sobol'
    decomposition);
  * the benchmark functions follow their published definitions (as
    implemented in the R package sensobol);
  * the Becker (2020) metafunction is a faithful port of sensobol's
    `metafunction`;
  * the sampling-design factor of the sensitivity-of-sensitivity study
    is decoded correctly (in 1.x it was constant).

Everything is vectorised NumPy; the only non-standard dependency is
SciPy.  Inputs of every estimator are given on the unit hypercube
(probability-integral-transformed inputs, u = F_i(x_i)).
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import qmc, norm, beta as beta_dist, rankdata, ks_2samp
from scipy.spatial.distance import pdist
from scipy.special import expit

# =====================================================================
# 1.  Sampling designs on [0,1]^d
# =====================================================================

DESIGNS = ("sobol", "srs", "lhs")


def design_matrix(N: int, d: int, design: str = "sobol",
                  seed: Optional[int] = None,
                  maximin_candidates: int = 20) -> np.ndarray:
    """Return an N x d design on [0,1)^d.

    design
        'sobol' : scrambled Sobol' sequence (scipy.stats.qmc.Sobol,
                  scramble=True: linear matrix scramble followed by a
                  digital shift); the first N points of the sequence.
        'srs'   : simple random sampling (i.i.d. uniform points).
        'lhs'   : maximin Latin hypercube -- the best (largest minimum
                  inter-point distance) of `maximin_candidates` random
                  Latin hypercubes.
    """
    if design == "sobol":
        eng = qmc.Sobol(d=d, scramble=True, seed=seed)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # N not a power of two
            return eng.random(N)
    if design == "srs":
        return np.random.default_rng(seed).random((N, d))
    if design == "lhs":
        rng = np.random.default_rng(seed)
        best, best_score = None, -np.inf
        for _ in range(max(1, maximin_candidates)):
            cand = qmc.LatinHypercube(d=d, seed=rng).random(N)
            score = pdist(cand).min() if N > 1 else 0.0
            if score > best_score:
                best, best_score = cand, score
        return best
    raise ValueError(f"unknown design '{design}'")


def jansen_matrices(N: int, k: int, design: str = "sobol",
                    seed: Optional[int] = None) -> np.ndarray:
    """Stacked [A; AB_1; ...; AB_k] matrix ((k+1)N x k) for Jansen's
    total-order estimator; A and B are the two halves of a 2k-column
    design."""
    D = design_matrix(N, 2 * k, design, seed)
    A, B = D[:, :k], D[:, k:]
    blocks = [A]
    for i in range(k):
        AB = A.copy()
        AB[:, i] = B[:, i]
        blocks.append(AB)
    return np.vstack(blocks)


def jansen_total(Y: np.ndarray, N: int, k: int) -> np.ndarray:
    """Jansen (1999) estimator of the total-order indices from the
    output of `jansen_matrices` (Y has length (k+1)N, block layout)."""
    Y = np.asarray(Y, float).reshape(k + 1, N)
    YA = Y[0]
    V = YA.var()
    if V == 0:                      # constant output on A: indices undefined
        return np.full(k, np.nan)
    return np.array([0.5 * np.mean((YA - Y[i + 1]) ** 2) / V
                     for i in range(k)])


def saltelli_first(Y: np.ndarray, YB: np.ndarray, N: int, k: int,
                   centre: bool = True) -> np.ndarray:
    """Saltelli et al. (2010) estimator of the first-order indices,
    S_i = mean((f(B) - c) [f(AB_i) - f(A)]) / V, from the output of
    `jansen_matrices` (block layout) and the output YB at the matrix B of
    the same 2k-column design.

    With centre=True, c is the mean of f(A) and f(B); since
    E[f(AB_i) - f(A)] = 0, centring leaves the expectation unchanged but
    removes the term of the variance that grows with the squared output
    mean (c = 0 gives the original estimator). V is the variance of f(A),
    as in `jansen_total`, so that S_i and T_i share their denominator.
    An input that does not affect the output has f(AB_i) = f(A) and
    S_i = T_i = 0 exactly."""
    Y = np.asarray(Y, float).reshape(k + 1, N)
    YA, YB = Y[0], np.asarray(YB, float)
    V = YA.var()
    if V == 0:                      # constant output on A: indices undefined
        return np.full(k, np.nan)
    c = 0.5 * (YA.mean() + YB.mean()) if centre else 0.0
    return np.array([np.mean((YB - c) * (Y[i + 1] - YA)) / V
                     for i in range(k)])


# =====================================================================
# 2.  Ersatz discrepancy measures
# =====================================================================

def _row_index(x: np.ndarray, s: int) -> np.ndarray:
    """0-based row (input-axis) cell index of points x in [0,1]."""
    return np.clip(np.ceil(x * s).astype(int), 1, s) - 1


def _rank_column_index(y: np.ndarray, s: int, ties: str = "ordinal",
                       rng: Optional[np.random.Generator] = None
                       ) -> np.ndarray:
    """0-based column (output-axis) cell index after the rank transform
    v = (rank - 1)/N.  Ties are broken by order of appearance
    ('ordinal', R's ties.method = "first") or at random ('random')."""
    N = len(y)
    if ties == "ordinal":
        r = rankdata(y, method="ordinal")
    elif ties == "random":
        rng = np.random.default_rng() if rng is None else rng
        perm = rng.permutation(N)
        r = np.empty(N, int)
        r[perm] = rankdata(y[perm], method="ordinal")
    else:
        raise ValueError("ties must be 'ordinal' or 'random'")
    # column b = max(1, ceil((r - 1) s / N)), in exact integer arithmetic
    # (floating-point ceil misplaces ranks that fall exactly on a block
    # boundary for some N, e.g. N = 600, s = 25)
    n = -((-(np.asarray(r, dtype=np.int64) - 1) * s) // N)
    n[n == 0] = 1
    return np.clip(n, 1, s) - 1


def occupancy_grid(x: np.ndarray, y: np.ndarray, s: int,
                   rank_y: bool = True, ties: str = "ordinal",
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Binary s x s occupancy matrix of the pairs (x_j, y_j)."""
    m = _row_index(np.asarray(x, float), s)
    y = np.asarray(y, float)
    if rank_y:
        n = _rank_column_index(y, s, ties, rng)
    else:                                   # min-max scaling (S-ersatz)
        rngy = y.max() - y.min()
        yn = (y - y.min()) / rngy if rngy > 0 else np.zeros_like(y)
        n = np.ceil(yn * s).astype(int)
        n[n == 0] = 1
        n = np.clip(n, 1, s) - 1
    G = np.zeros((s, s), dtype=np.int8)
    G[m, n] = 1
    return G


_NEIGH_COUNT_CACHE: Dict[int, np.ndarray] = {}


def _neighbour_counts(s: int) -> np.ndarray:
    """Number of in-grid Moore neighbours of every cell (8/5/3)."""
    if s not in _NEIGH_COUNT_CACHE:
        c = np.full((s, s), 8, dtype=int)
        c[0, :] = c[-1, :] = c[:, 0] = c[:, -1] = 5
        c[0, 0] = c[0, -1] = c[-1, 0] = c[-1, -1] = 3
        if s == 1:
            c[:] = 0
        _NEIGH_COUNT_CACHE[s] = c
    return _NEIGH_COUNT_CACHE[s]


def moore_fill(G: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Non-cascading Moore-neighbourhood fill: an empty cell is set to 1
    if the fraction of its in-grid Moore neighbours that are occupied
    in the ORIGINAL grid G is >= threshold."""
    s = G.shape[0]
    P = np.pad(G.astype(int), 1)
    nb = (P[:-2, :-2] + P[:-2, 1:-1] + P[:-2, 2:] +
          P[1:-1, :-2] + P[1:-1, 2:] +
          P[2:, :-2] + P[2:, 1:-1] + P[2:, 2:])
    cnt = _neighbour_counts(s)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(cnt > 0, nb / np.maximum(cnt, 1), 0.0)
    fill = (G == 0) & (frac >= threshold - 1e-12)
    out = G.copy()
    out[fill] = 1
    return out


def grid_size(N: int, grid_exponent: float = 0.5) -> int:
    """s = ceil(N ** alpha_g)."""
    return max(1, int(math.ceil(N ** grid_exponent - 1e-12)))


def ersatz_input(x: np.ndarray, y: np.ndarray, *, adjusted: bool = True,
                 fill: bool = True, fill_threshold: float = 0.5,
                 grid_exponent: float = 0.5, ties: str = "ordinal",
                 rng: Optional[np.random.Generator] = None) -> float:
    """Ersatz statistic of one input.

    adjusted=False           : S-ersatz of Puy et al. (2024): min-max
                               scaled output, no fill, s = ceil(sqrt N).
    adjusted=True, fill=True : adjusted ersatz (this paper).
    adjusted=True, fill=False: rank-transformed grid without fill (the
                               'unfilled' statistic of the theory).
    Returns the fraction of empty cells."""
    N = len(y)
    if not adjusted:
        s = grid_size(N, 0.5)
        G = occupancy_grid(x, y, s, rank_y=False)
        return 1.0 - G.mean()
    s = grid_size(N, grid_exponent)
    G = occupancy_grid(x, y, s, rank_y=True, ties=ties, rng=rng)
    if fill:
        G = moore_fill(G, fill_threshold)
    return 1.0 - G.mean()


def ersatz_all(X: np.ndarray, Y: np.ndarray, **kw) -> np.ndarray:
    """Ersatz statistic for every column of X."""
    return np.array([ersatz_input(X[:, i], Y, **kw)
                     for i in range(X.shape[1])])


# =====================================================================
# 3.  Polynomial chaos expansion (orthonormal basis, OLS, LOO degree)
# =====================================================================

def _orthonormal_1d(t: np.ndarray, p: int, family: str) -> np.ndarray:
    """N x (p+1) table of orthonormal polynomials of degree 0..p.
    family 'legendre': t ~ U(0,1);  'hermite': t ~ N(0,1)."""
    N = len(t)
    out = np.empty((N, p + 1))
    out[:, 0] = 1.0
    if family == "legendre":
        x = 2.0 * t - 1.0
        Pm1, P0 = np.ones(N), x
        if p >= 1:
            out[:, 1] = math.sqrt(3.0) * x
        for n in range(1, p):
            P1 = ((2 * n + 1) * x * P0 - n * Pm1) / (n + 1)
            out[:, n + 1] = math.sqrt(2 * (n + 1) + 1) * P1
            Pm1, P0 = P0, P1
    elif family == "hermite":
        Hm1, H0 = np.ones(N), t.copy()
        if p >= 1:
            out[:, 1] = t
        for n in range(1, p):
            H1 = t * H0 - n * Hm1
            out[:, n + 1] = H1 / math.sqrt(math.factorial(n + 1))
            Hm1, H0 = H0, H1
    else:
        raise ValueError(family)
    return out


def total_degree_indices(k: int, p: int) -> np.ndarray:
    """All multi-indices alpha in N^k with |alpha| <= p (row 0 = 0)."""
    out = [np.zeros(k, dtype=int)]
    for deg in range(1, p + 1):
        # compositions of deg into k non-negative parts
        for bars in combinations(range(deg + k - 1), k - 1):
            prev, a = -1, []
            for b in bars:
                a.append(b - prev - 1)
                prev = b
            a.append(deg + k - 1 - prev - 1)
            out.append(np.array(a, dtype=int))
    return np.array(out)


def _pce_design(U: np.ndarray, alphas: np.ndarray,
                family: str) -> np.ndarray:
    N, k = U.shape
    p = int(alphas.max()) if alphas.size else 0
    if family == "hermite":
        T = norm.ppf(np.clip(U, 1e-12, 1 - 1e-12))
    else:
        T = U
    tabs = [_orthonormal_1d(T[:, j], p, family) for j in range(k)]
    Psi = np.ones((N, len(alphas)))
    for j in range(k):
        aj = alphas[:, j]
        if np.any(aj > 0):
            Psi *= tabs[j][:, aj]
    return Psi


def pce_fit(U: np.ndarray, Y: np.ndarray, family: str = "legendre",
            pmax: int = 10, min_ratio: float = 2.0,
            max_terms: int = 1000) -> Optional[dict]:
    """Degree-adaptive full total-degree PCE by ordinary least squares.

    For p = 1, 2, ... the full basis of total degree p is fitted while
    the number of terms P satisfies N >= min_ratio * P and P <= max_terms;
    the degree with the smallest corrected leave-one-out error
    (Blatman & Sudret 2010) is retained.  Returns None if not even p=1
    is admissible."""
    N, k = U.shape
    Y = np.asarray(Y, float)
    vy = Y.var()
    best = None
    for p in range(1, pmax + 1):
        P = math.comb(k + p, p)
        if N < min_ratio * P or P > max_terms:
            break
        alphas = total_degree_indices(k, p)
        Psi = _pce_design(U, alphas, family)
        Q, R = np.linalg.qr(Psi, mode="reduced")
        if np.min(np.abs(np.diag(R))) < 1e-10 * np.max(np.abs(np.diag(R))):
            continue
        coef = np.linalg.solve(R, Q.T @ Y)
        h = np.sum(Q ** 2, axis=1)
        resid = Y - Psi @ coef
        if np.any(h > 1 - 1e-8) or vy == 0:
            continue
        err = np.mean((resid / (1 - h)) ** 2) / vy
        Rinv = np.linalg.solve(R, np.eye(P))
        trC = N * np.sum(Rinv ** 2)                  # tr((Psi'Psi/N)^-1)
        err *= (N / (N - P)) * (1 + trC / N)
        if best is None or err < best["loo"]:
            best = dict(p=p, alphas=alphas, coef=coef, loo=err)
    return best


def pce_indices(U: np.ndarray, Y: np.ndarray, family: str = "legendre",
                **kw) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    """(T_i, Shapley_i, fit) from the PCE; NaNs if the fit fails."""
    k = U.shape[1]
    fit = pce_fit(U, Y, family=family, **kw)
    if fit is None:
        return np.full(k, np.nan), np.full(k, np.nan), None
    a, c = fit["alphas"][1:], fit["coef"][1:]
    var = np.sum(c ** 2)
    if var <= 0:
        return np.zeros(k), np.zeros(k), fit
    c2 = c ** 2
    supp = (a > 0)
    card = supp.sum(axis=1)
    T = np.array([c2[supp[:, i]].sum() for i in range(k)]) / var
    Sh = np.array([(c2[supp[:, i]] / card[supp[:, i]]).sum()
                   for i in range(k)]) / var
    return T, Sh, fit


# =====================================================================
# 4.  PAWN, given-data version (Pianosi & Wagener 2018), max-KS
# =====================================================================

def pawn_given_data(U: np.ndarray, Y: np.ndarray, n_bins: int = 10,
                    stat: str = "max") -> np.ndarray:
    """KS distance between the unconditional output sample and the
    output sub-samples conditional on n_bins equal-width (= equiprobable,
    inputs being uniform) intervals of each input; aggregated with
    `stat` ('max' or 'median') over the intervals."""
    N, k = U.shape
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = np.empty(k)
    for i in range(k):
        b = np.clip(np.searchsorted(edges, U[:, i], side="right") - 1,
                    0, n_bins - 1)
        ks = [ks_2samp(Y, Y[b == j]).statistic
              for j in range(n_bins) if np.sum(b == j) > 1]
        if not ks:
            out[i] = np.nan
        else:
            out[i] = np.max(ks) if stat == "max" else np.median(ks)
    return out


# =====================================================================
# 5.  Savage scores and agreement metrics
# =====================================================================

def savage_scores(x: np.ndarray) -> np.ndarray:
    """Savage scores (Iman & Conover 1987): the input of rank r (1 = most
    important) receives sum_{j=r}^{k} 1/j; tied inputs receive the
    average of their scores."""
    x = np.asarray(x, float)
    k = len(x)
    base = np.array([np.sum(1.0 / np.arange(r, k + 1))
                     for r in range(1, k + 1)])
    order = np.argsort(-x, kind="mergesort")
    sc = np.empty(k)
    sc[order] = base
    # average over ties
    for v in np.unique(x):
        idx = np.where(x == v)[0]
        if len(idx) > 1:
            sc[idx] = sc[idx].mean()
    return sc


def savage_rho(est: np.ndarray, ref: np.ndarray) -> float:
    """Signed Pearson correlation of Savage scores; NaN if undefined."""
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    if np.any(~np.isfinite(est)) or np.any(~np.isfinite(ref)):
        return np.nan
    a, b = savage_scores(est), savage_scores(ref)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def screening_errors(est: np.ndarray, ref: np.ndarray, theta: float
                     ) -> Tuple[float, float]:
    """(alpha, beta) at threshold theta: alpha = share of truly
    non-influential inputs (ref <= theta) flagged as influential
    (est > theta); beta = share of truly influential inputs missed.
    NaN when the corresponding class is empty."""
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    if np.any(~np.isfinite(est)):
        return np.nan, np.nan
    neg, pos = ref <= theta, ref > theta
    a = np.mean(est[neg] > theta) if neg.any() else np.nan
    b = np.mean(est[pos] <= theta) if pos.any() else np.nan
    return float(a), float(b)


def summary_metrics(est: np.ndarray, ref: np.ndarray,
                    thetas: Sequence[float] = (0.01, 0.05, 0.10)) -> dict:
    out = {"rho": savage_rho(est, ref),
           "mae": float(np.mean(np.abs(np.asarray(est) - ref)))
           if np.all(np.isfinite(est)) else np.nan}
    for t in thetas:
        a, b = screening_errors(est, ref, t)
        out[f"alpha_{t:.2f}"] = a
        out[f"beta_{t:.2f}"] = b
    return out


# =====================================================================
# 6.  Benchmark functions (definitions as in R package sensobol)
# =====================================================================

def bratley1988(U: np.ndarray) -> np.ndarray:
    """Bratley & Fox (1988): prod_i |4 x_i - 2| (sensobol::bratley1988_Fun)."""
    return np.prod(np.abs(4.0 * U - 2.0), axis=1)


def bratley1992(U: np.ndarray) -> np.ndarray:
    """Bratley, Fox & Niederreiter (1992): sum_i (-1)^i prod_{j<=i} x_j
    (sensobol::bratley1992_Fun)."""
    k = U.shape[1]
    return (np.cumprod(U, axis=1) * (-1.0) ** np.arange(1, k + 1)).sum(1)


def ishigami(U: np.ndarray, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    """Ishigami & Homma (1990), inputs U(-pi, pi), a = 7, b = 0.1."""
    X = U * 2.0 * np.pi - np.pi
    return (np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2
            + b * X[:, 2] ** 4 * np.sin(X[:, 0]))


SOBOL_G_A = np.array([0, 1, 4.5, 9, 99, 99, 99, 99], float)


def sobol_g(U: np.ndarray, a: np.ndarray = SOBOL_G_A) -> np.ndarray:
    """Sobol' G function (sensobol::sobol_Fun)."""
    return np.prod((np.abs(4.0 * U - 2.0) + a) / (1.0 + a), axis=1)


_OO_A1 = np.array([0.0118, 0.0456, 0.2297, 0.0393, 0.1177, 0.3865, 0.3897,
                   0.6061, 0.6159, 0.4005, 1.0741, 1.1474, 0.7880, 1.1242,
                   1.1982])
_OO_A2 = np.array([0.4341, 0.0887, 0.0512, 0.3233, 0.1489, 1.0360, 0.9892,
                   0.9672, 0.8977, 0.8083, 1.8426, 2.4712, 2.3946, 2.0045,
                   2.2621])
_OO_A3 = np.array([0.1044, 0.2057, 0.0774, 0.2730, 0.1253, 0.7526, 0.8570,
                   1.0331, 0.8388, 0.7970, 2.2145, 2.0382, 2.4004, 2.0541,
                   1.9845])
_OO_M = np.array([
    -0.022482886, -0.18501666, 0.13418263, 0.36867264, 0.17172785, 0.13651143, -0.44034404,
    -0.081422854, 0.71321025, -0.44361072, 0.50383394, -0.024101458, -0.045939684, 0.21666181,
    0.055887417, 0.2565963, 0.053792287, 0.25800381, 0.23795905, -0.59125756, -0.081627077,
    -0.28749073, 0.41581639, 0.49752241, 0.083893165, -0.11056683, 0.033222351, -0.13979497,
    -0.031020556, -0.22318721, -0.055999811, 0.19542252, 0.095529005, -0.2862653, -0.14441303,
    0.22369356, 0.14527412, 0.28998481, 0.2310501, -0.31929879, -0.29039128, -0.20956898, 0.43139047,
    0.024429152, 0.044904409, 0.66448103, 0.43069872, 0.29924645, -0.16202441, -0.31479544,
    -0.39026802, 0.17679822, 0.057952663, 0.17230342, 0.13466011, -0.3527524, 0.25146896, -0.018810529,
    0.36482392, -0.32504618, -0.121278, 0.12463327, 0.10656519, 0.046562296, -0.21678617, 0.19492172,
    -0.065521126, 0.024404669, -0.09682886, 0.19366196, 0.33354757, 0.31295994, -0.083615456,
    -0.25342082, 0.37325717, -0.2837623, -0.32820154, -0.10496068, -0.22073452, -0.13708154,
    -0.14426375, -0.11503319, 0.22424151, -0.030395022, -0.51505615, 0.017254978, 0.038957118,
    0.36069184, 0.30902452, 0.050030193, -0.077875893, 0.003745656, 0.88685604, -0.26590028,
    -0.079325357, -0.042734919, -0.18653782, -0.35604718, -0.17497421, 0.088699956, 0.40025886,
    -0.055979693, 0.13724479, 0.21485613, -0.011265799, -0.09229473, 0.59209563, 0.031338285,
    -0.033080861, -0.24308858, -0.099798547, 0.034460195, 0.095119813, -0.3380162, 0.0063860024,
    -0.61207299, 0.081325416, 0.88683114, 0.14254905, 0.14776204, -0.13189434, 0.52878496, 0.12652391,
    0.045113625, 0.58373514, 0.37291503, 0.11395325, -0.29479222, -0.57014085, 0.46291592, -0.094050179,
    0.13959097, -0.38607402, -0.4489706, -0.14602419, 0.058107658, -0.32289338, 0.093139162,
    0.072427234, -0.56919401, 0.52554237, 0.23656926, -0.011782016, 0.071820601, 0.078277291,
    -0.13355752, 0.22722721, 0.14369455, -0.45198935, -0.55574794, 0.66145875, 0.34633299, 0.14098019,
    0.51882591, -0.28019898, -0.1603226, -0.068413337, -0.20428242, 0.069672173, 0.23112577,
    -0.044368579, -0.16455425, 0.21620977, 0.0042702105, -0.087399014, 0.31599556, -0.027551859,
    0.13434254, 0.13497371, 0.05400568, -0.17374789, 0.17525393, 0.060258929, -0.17914162, -0.31056619,
    -0.25358691, 0.025847535, -0.43006001, -0.62266361, -0.033996882, -0.29038151, 0.03410127,
    0.034903413, -0.12121764, 0.026030714, -0.33546274, -0.41424111, 0.05324838, -0.27099455,
    -0.026251302, 0.41024137, 0.26636349, 0.15582891, -0.18666254, 0.019895831, -0.24388652,
    -0.44098852, 0.012618825, 0.24945112, 0.071101888, 0.24623792, 0.17484502, 0.0085286769,
    0.2514707, -0.14659862, -0.08462515, 0.36931333, -0.29955293, 0.1104436, -0.75690139, 0.041494323,
    -0.25980564, 0.46402128, -0.36112127, -0.94980789, -0.16504063, 0.0030943325, 0.052792942,
    0.22523648, 0.38390366, 0.45562427, -0.18631744, 0.0082333995, 0.16670803, 0.16045688]
).reshape(15, 15)


def oakley_ohagan(U: np.ndarray) -> np.ndarray:
    """Oakley & O'Hagan (2004) 15-input function, inputs N(0,1)
    (sensobol::oakley_Fun; coefficients from the original article)."""
    Z = norm.ppf(np.clip(U, 1e-15, 1 - 1e-15))
    return (Z @ _OO_A1 + np.sin(Z) @ _OO_A2 + np.cos(Z) @ _OO_A3
            + np.einsum("ni,ij,nj->n", Z, _OO_M, Z))


def _trunc_normal_ppf(u, mu, sd, lo=0.0, hi=1.0):
    a, b = norm.cdf(lo, mu, sd), norm.cdf(hi, mu, sd)
    return norm.ppf(a + (b - a) * u, mu, sd)


def multiverse(U: np.ndarray) -> np.ndarray:
    """Mixed-input 'Multiverse' model (Saltelli et al. 2025), as coded in
    the authors' R script: columns (theta1, theta2, theta3, xi, zeta);
    xi = floor(8u) selects the model form, zeta = round(u) switches the
    three continuous inputs from U(0,1) to N(1/2, 1/12) truncated to
    [0,1]."""
    xi = np.floor(U[:, 3] * 8).astype(int)
    zeta = np.round(U[:, 4]).astype(int)
    T = U[:, :3].copy()
    tn = _trunc_normal_ppf(U[:, :3], 0.5, 1.0 / 12.0)
    T[zeta == 1] = tn[zeta == 1]
    x1, x2, x3 = T[:, 0], T[:, 1], T[:, 2]
    forms = np.stack([x1 + x2 + x3, x1 + x2 * x3, x1 * x2 + x3,
                      x1 * x3 + x2, x1 * (x2 + x3), x2 * (x1 + x3),
                      x3 * (x1 + x2), x1 * x2 * x3], axis=1)
    return np.cbrt(forms[np.arange(len(U)), xi])


# =====================================================================
# 7.  Becker (2020) metafunction -- port of sensobol::metafunction
# =====================================================================

def _piecewise_large(x):
    return (-1.0) ** np.floor(4 * x) * (0.125 - np.mod(x, 0.25)) + 0.125


def _piecewise_small(x):
    return ((-1.0) ** np.floor(32 * x) * (0.03125 - 2 * np.mod(x, 0.03125))
            + 0.03125) / 2


BECKER_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Linear": lambda x: x,
    "Quadratic": lambda x: x ** 2,
    "Cubic": lambda x: x ** 3,
    "Exponential": lambda x: np.exp(x) / (np.e - 1),
    "Periodic": lambda x: np.sin(2 * np.pi * x) / 2,
    "Discontinuous": lambda x: (x > 0.5).astype(float),
    "Non.monotonic": lambda x: 4 * (x - 0.5) ** 2,
    "Inverse": lambda x: (10 - 1 / 1.1) ** -1 * (x + 0.1) ** -1,
    "No.effect": lambda x: 0.0 * x,
    "Trigonometric": lambda x: np.cos(x),
    "Piecewise.large": _piecewise_large,
    "Piecewise.small": _piecewise_small,
    "Oscillation": lambda x: x ** 2 - 0.2 * np.cos(7 * np.pi * x),
}


class BeckerMetafunction:
    """One random draw of the Becker (2020) metafunction, following the
    structure of sensobol::metafunction (k_2 = 0.5, k_3 = 0.2):

      * each input receives one of 13 univariate functions at random;
      * coefficients are drawn from the mixture 0.7 N(0, 0.5) + 0.3 N(0, 5);
      * ceil(k k_2) pairwise and ceil(k k_3) three-way interactions
        (one if k = 3) are drawn without replacement.

    Interaction terms are evaluated row-wise for every k (sensobol
    collapses a single sampled pair/triplet to a constant, a quirk not
    reproduced here).  R's random-number stream cannot be reproduced, so
    a given epsilon defines a different (but identically distributed)
    function than in R."""

    def __init__(self, k: int, epsilon: int, k_2: float = 0.5,
                 k_3: float = 0.2):
        rng = np.random.default_rng(epsilon)
        names = list(BECKER_FUNCTIONS)
        self.k = k
        self.funcs = [names[j] for j in rng.integers(0, len(names), k)]
        comp = rng.choice(2, size=200, p=[0.7, 0.3])
        z = np.tile(rng.standard_normal(100), 2)
        coefs = z * np.sqrt(np.array([0.5, 5.0]))[comp]
        self.c1 = rng.choice(coefs, size=k, replace=False)
        pairs = list(combinations(range(k), 2))
        n2 = min(len(pairs), int(math.ceil(k * k_2)))
        self.pairs = [pairs[j] for j in rng.choice(len(pairs), n2,
                                                   replace=False)]
        self.c2 = rng.choice(coefs, size=n2, replace=True)
        trips = list(combinations(range(k), 3))
        n3 = 1 if len(trips) == 1 else min(len(trips),
                                            int(math.ceil(k * k_3)))
        self.trips = [trips[j] for j in rng.choice(len(trips), n3,
                                                   replace=False)]
        self.c3 = rng.choice(coefs, size=n3, replace=True)

    def __call__(self, Xt: np.ndarray) -> np.ndarray:
        F = np.column_stack([BECKER_FUNCTIONS[f](Xt[:, i])
                             for i, f in enumerate(self.funcs)])
        y = F @ self.c1
        for c, (a, b) in zip(self.c2, self.pairs):
            y = y + c * F[:, a] * F[:, b]
        for c, (a, b, d) in zip(self.c3, self.trips):
            y = y + c * F[:, a] * F[:, b] * F[:, d]
        return y


def _trunc_beta_ppf(u, a, b):
    return beta_dist.ppf(u, a, b)            # truncation to [0,1] is void


def _logitnorm_ppf(u, mu=0.0, sigma=3.16):
    return expit(mu + sigma * norm.ppf(np.clip(u, 1e-15, 1 - 1e-15)))


INPUT_DISTRIBUTIONS: List[Callable[[np.ndarray], np.ndarray]] = [
    lambda u: u,                                         # 1 uniform
    lambda u: _trunc_normal_ppf(u, 0.5, 0.15),           # 2 normal
    lambda u: _trunc_beta_ppf(u, 8, 2),                  # 3 beta(8,2)
    lambda u: _trunc_beta_ppf(u, 2, 8),                  # 4 beta(2,8)
    lambda u: _trunc_beta_ppf(u, 2, 0.8),                # 5 beta(2,0.8)
    lambda u: _trunc_beta_ppf(u, 0.8, 2),                # 6 beta(0.8,2)
    lambda u: _logitnorm_ppf(u, 0.0, 3.16),              # 7 logit-normal
]


def input_distribution_map(k: int, phi: int, seed: int) -> List[int]:
    """Distribution index (0..6) of every input: phi in 1..7 gives the
    same family to all inputs; phi = 8 draws one family per input."""
    if 1 <= phi <= 7:
        return [phi - 1] * k
    rng = np.random.default_rng(10_000 + seed)
    return list(rng.integers(0, 7, k))


def transform_inputs(U: np.ndarray, dist_idx: Sequence[int]) -> np.ndarray:
    return np.column_stack([INPUT_DISTRIBUTIONS[d](U[:, i])
                            for i, d in enumerate(dist_idx)])


# =====================================================================
# 8.  One-call evaluation of all given-data estimators
# =====================================================================

def all_estimators(U: np.ndarray, Y: np.ndarray, *, pce_family="legendre",
                   fill_threshold=0.5, grid_exponent=0.5, pawn_bins=10,
                   ties="ordinal") -> Dict[str, np.ndarray]:
    """All estimators on one given-data sample (U uniform inputs)."""
    T_pce, Sh, fit = pce_indices(U, Y, family=pce_family)
    return {
        "ersatz": ersatz_all(U, Y, adjusted=False),
        "ersatz_unfilled": ersatz_all(U, Y, adjusted=True, fill=False,
                                      grid_exponent=grid_exponent,
                                      ties=ties),
        "ersatz_adj": ersatz_all(U, Y, adjusted=True, fill=True,
                                 fill_threshold=fill_threshold,
                                 grid_exponent=grid_exponent, ties=ties),
        "pce": T_pce,
        "shapley": Sh,
        "pawn": pawn_given_data(U, Y, n_bins=pawn_bins),
        "_pce_degree": np.array([fit["p"] if fit else np.nan]),
    }
