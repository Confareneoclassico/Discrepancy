"""
experiments.py -- numerical experiments of the revised manuscript.

Usage:  python experiments.py {benchmarks|convergence|hymod|becker|sos|sos_first} [--quick]

All raw results are written to ./results/*.csv (long format) so that
tables and figures can be regenerated without re-running the models.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import discrepancy as D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
THETAS = (0.01, 0.05, 0.10)
METHODS = ("ersatz", "ersatz_unfilled", "ersatz_adj", "pce", "shapley",
           "pawn")
NPROC = max(1, (os.cpu_count() or 2))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "refs.json")) as fh:
    REFS = {k: np.array(v) for k, v in json.load(fh).items()}
# exact symmetries: the eight inputs of the Bratley-Fox product function and
# the three continuous inputs of the Multiverse function have equal indices
REFS["bratley1988"] = np.full(8, REFS["bratley1988"].mean())
REFS["multiverse"][:3] = REFS["multiverse"][:3].mean()

FUNCTIONS = {
    #  name          (callable,          k,  PCE family)
    "bratley1988": (D.bratley1988, 8, "legendre"),
    "bratley1992": (D.bratley1992, 8, "legendre"),
    "ishigami": (D.ishigami, 3, "legendre"),
    "oakley": (D.oakley_ohagan, 15, "hermite"),
    "sobol_g": (D.sobol_g, 8, "legendre"),
    "multiverse": (D.multiverse, 5, "legendre"),
}
FID = {n: i for i, n in enumerate(FUNCTIONS)}


def _records(tag: dict, est: dict, ref: np.ndarray):
    """Long-format rows: per-input values and summary metrics."""
    vals, mets = [], []
    for m in METHODS:
        v = est[m]
        for i, x in enumerate(v):
            vals.append({**tag, "method": m, "input": i, "value": x,
                         "ref": ref[i]})
        mets.append({**tag, "method": m,
                     **D.summary_metrics(v, ref, THETAS)})
    return vals, mets


# ---------------------------------------------------------------------
# 1. Benchmarks at N = 512, replicated, three designs
# ---------------------------------------------------------------------

def _bench_job(args):
    fname, design, rep, N = args
    f, k, fam = FUNCTIONS[fname]
    seed = 100_000 * (FID[fname] + 1) + 1_000 * D.DESIGNS.index(design) + rep
    U = D.design_matrix(N, k, design, seed=seed)
    Y = f(U)
    est = D.all_estimators(U, Y, pce_family=fam)
    tag = dict(function=fname, design=design, rep=rep, N=N,
               pce_degree=float(est["_pce_degree"][0]))
    return _records(tag, est, REFS[fname])


def run_benchmarks(R=100, N=512):
    jobs = [(f, d, r, N) for f in FUNCTIONS for d in D.DESIGNS
            for r in range(R)]
    t0 = time.time()
    with Pool(NPROC) as pool:
        res = pool.map(_bench_job, jobs, chunksize=4)
    vals = [v for a, _ in res for v in a]
    mets = [m for _, b in res for m in b]
    pd.DataFrame(vals).to_csv(f"{OUT}/bench_values.csv", index=False)
    pd.DataFrame(mets).to_csv(f"{OUT}/bench_metrics.csv", index=False)
    print(f"benchmarks done in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------
# 2. Convergence with N (nested designs), replicated
# ---------------------------------------------------------------------

NS_CONV = [2 ** e for e in range(4, 14)]


def _conv_job(args):
    fname, design, rep = args
    f, k, fam = FUNCTIONS[fname]
    seed = 500_000 + 100_000 * FID[fname] + 1_000 * D.DESIGNS.index(design) + rep
    Umax = D.design_matrix(NS_CONV[-1], k, design, seed=seed)
    Ymax = f(Umax)
    rows = []
    for N in NS_CONV:
        est = D.all_estimators(Umax[:N], Ymax[:N], pce_family=fam)
        for m in METHODS:
            rows.append(dict(function=fname, design=design, rep=rep, N=N,
                             method=m,
                             **D.summary_metrics(est[m], REFS[fname], THETAS)))
    return rows


def run_convergence(R=20, functions=None):
    functions = functions or list(FUNCTIONS)
    jobs = [(f, d, r) for f in functions for d in ("sobol", "srs")
            for r in range(R)]
    t0 = time.time()
    with Pool(NPROC) as pool:
        res = pool.map(_conv_job, jobs, chunksize=1)
    new = pd.DataFrame([x for r in res for x in r])
    path = f"{OUT}/convergence.csv"
    if os.path.exists(path) and len(functions) < len(FUNCTIONS):
        old = pd.read_csv(path)
        new = pd.concat([old[~old.function.isin(functions)], new])
    new.to_csv(path, index=False)
    print(f"convergence done in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------
# 3. HYMOD (given data)
# ---------------------------------------------------------------------

HYMOD_NAMES = ["Cmax", "bexp", "alpha", "Rs", "Rq"]       # x1..x5
# Reference total-order indices of the HYMOD sample, computed by
# hymod_reference.py (mean of the PCE and gradient-boosting surrogates).
# Version 1.x used [0.507, 0.017, 0.073, 0.002, 0.725], which is inconsistent
# with the sample (the first-order index of alpha, about 0.12, exceeds 0.073).
HYMOD_REF = np.array([0.2406, 0.0248, 0.4026, 0.0039, 0.7040])
HYMOD_BOUNDS = [(1, 500), (0.1, 2), (0.1, 0.98), (0, 0.1), (0.1, 0.98)]


def load_hymod(path_in="hydro.csv", path_out="hydro2.csv"):
    Xr = pd.read_csv(path_in, index_col=0).values
    Y = pd.read_csv(path_out, index_col=0)["NSE"].values
    U = np.column_stack([(Xr[:, j] - lo) / (hi - lo)
                         for j, (lo, hi) in enumerate(HYMOD_BOUNDS)])
    return np.clip(U, 0, 1), Y


_HY = {}


def _hymod_init(data_dir):
    U, Y = load_hymod(os.path.join(data_dir, "hydro.csv"),
                      os.path.join(data_dir, "hydro2.csv"))
    _HY["U"], _HY["Y"] = U, Y


def _hymod_job(args):
    kind, N, rep = args
    U, Y = _HY["U"], _HY["Y"]
    if kind == "sobol_block":
        idx = np.arange(rep * N, (rep + 1) * N)
    elif kind == "sobol_prefix":
        idx = np.arange(N)
    else:
        idx = np.random.default_rng(7_000 + rep).choice(len(Y), N,
                                                       replace=False)
    est = D.all_estimators(U[idx], Y[idx])
    tag = dict(function="hymod", design=kind, rep=rep, N=N,
               pce_degree=float(est["_pce_degree"][0]))
    return _records(tag, est, HYMOD_REF)


def run_hymod(data_dir="."):
    U, Y = load_hymod(os.path.join(data_dir, "hydro.csv"),
                      os.path.join(data_dir, "hydro2.csv"))
    n = len(Y)
    jobs = [("sobol_prefix", n, 0)]
    for N in [2 ** e for e in range(5, 16)]:
        R = 20 if N <= 4096 else 5
        jobs += [("sobol_block", N, r) for r in range(min(R, n // N))]
        jobs += [("srs_subset", N, r) for r in range(R)]
    jobs += [("sobol_block", 512, r) for r in range(20, n // 512)]
    jobs += [("srs_subset", 512, r) for r in range(20, 100)]
    jobs.sort(key=lambda j: -j[1])               # big jobs first
    t0 = time.time()
    with Pool(NPROC, initializer=_hymod_init, initargs=(data_dir,)) as pool:
        res = pool.map(_hymod_job, jobs, chunksize=1)
    pd.DataFrame([v for a, _ in res for v in a]).to_csv(
        f"{OUT}/hymod_values.csv", index=False)
    pd.DataFrame([m for _, b in res for m in b]).to_csv(
        f"{OUT}/hymod_metrics.csv", index=False)
    print(f"hymod done in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------
# 4. Becker metafunction ensemble (paired designs, accurate reference)
# ---------------------------------------------------------------------

def becker_configs(n_conf=512, seed=2026):
    """(epsilon, phi, k, Ns) for n_conf configurations, from a scrambled
    Sobol' design: epsilon ~ U{1..200}, phi ~ U{1..8}, k ~ U{3..15},
    Ns ~ U{10..100}; the given-data cost is C = Ns (k + 1)."""
    M = D.design_matrix(n_conf, 4, "sobol", seed=seed)
    eps = np.floor(M[:, 0] * 200).astype(int) + 1
    phi = np.floor(M[:, 1] * 8).astype(int) + 1
    k = np.floor(M[:, 2] * 13).astype(int) + 3
    Ns = np.floor(M[:, 3] * 91).astype(int) + 10
    return [dict(conf=i, epsilon=int(eps[i]), phi=int(phi[i]),
                 k=int(k[i]), Ns=int(Ns[i])) for i in range(n_conf)]


def _becker_sample(c, design):
    f = D.BeckerMetafunction(c["k"], c["epsilon"])
    dist = D.input_distribution_map(c["k"], c["phi"], c["epsilon"])
    C = c["Ns"] * (c["k"] + 1)
    seed = 3_000_000 + 10 * c["conf"] + D.DESIGNS.index(design)
    U = D.design_matrix(C, c["k"], design, seed=seed)
    return U, f(D.transform_inputs(U, dist)), f, dist


def becker_reference(c, N_ref=2 ** 13):
    f = D.BeckerMetafunction(c["k"], c["epsilon"])
    dist = D.input_distribution_map(c["k"], c["phi"], c["epsilon"])
    X = D.jansen_matrices(N_ref, c["k"], "sobol", seed=9_000 + c["conf"])
    Y = f(D.transform_inputs(X, dist))
    if np.var(Y[:N_ref]) == 0:
        return np.zeros(c["k"])
    return np.clip(D.jansen_total(Y, N_ref, c["k"]), 0, None)


def _becker_job(c):
    ref = becker_reference(c)
    out_v, out_m, cache = [], [], {}
    for design in D.DESIGNS:
        U, Y, _, _ = _becker_sample(c, design)
        if np.var(Y) == 0:
            continue
        est = D.all_estimators(U, Y)
        tag = dict(function="becker", design=design, rep=c["conf"],
                   N=len(Y), k=c["k"], phi=c["phi"], epsilon=c["epsilon"],
                   Ns=c["Ns"], pce_degree=float(est["_pce_degree"][0]))
        v, m = _records(tag, est, ref)
        out_v += v
        out_m += m
        if c["conf"] < 128:
            cache[design] = (U, Y)
    return out_v, out_m, (c, ref, cache)


def run_becker(n_conf=512):
    confs = becker_configs(n_conf)
    t0 = time.time()
    with Pool(NPROC) as pool:
        res = pool.map(_becker_job, confs, chunksize=2)
    pd.DataFrame([v for a, _, _ in res for v in a]).to_csv(
        f"{OUT}/becker_values.csv", index=False)
    pd.DataFrame([m for _, b, _ in res for m in b]).to_csv(
        f"{OUT}/becker_metrics.csv", index=False)
    cache = [x for _, _, x in res if x[0]["conf"] < 128]
    np.save(f"{OUT}/sos_cache.npy", np.array(cache, dtype=object),
            allow_pickle=True)
    print(f"becker done in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------
# 5. Sensitivity of the sensitivity analysis (fixed cache)
# ---------------------------------------------------------------------

SOS_FACTORS = ["fill_threshold", "grid_exponent", "screen_threshold",
               "pawn_bins", "design"]
SOS_OUT = ["rho_adj", "mae_adj", "alpha_adj", "beta_adj",
           "rho_pawn", "mae_pawn", "alpha_pawn", "beta_pawn"]


def _sos_decode(u):
    return dict(fill_threshold=0.25 + 0.5 * u[0],
                grid_exponent=0.40 + 0.20 * u[1],
                screen_threshold=0.01 + 0.09 * u[2],
                pawn_bins=int(min(20, 5 + math.floor(16 * u[3]))),
                design=D.DESIGNS[int(min(2, math.floor(3 * u[4])))])


_SOS_CACHE = None
_SOS_PAWN = None


def _sos_init(cache_path):
    global _SOS_CACHE, _SOS_PAWN
    _SOS_CACHE = list(np.load(cache_path, allow_pickle=True))
    _SOS_PAWN = np.load(cache_path.replace("sos_cache", "sos_pawn"),
                        allow_pickle=True).item()


def _sos_eval(u):
    p = _sos_decode(u)
    r_a, m_a, a_a, b_a, r_p, m_p, a_p, b_p = ([] for _ in range(8))
    for j, (c, ref, cache) in enumerate(_SOS_CACHE):
        if p["design"] not in cache:
            continue
        U, Y = cache[p["design"]]
        adj = D.ersatz_all(U, Y, adjusted=True, fill=True,
                           fill_threshold=p["fill_threshold"],
                           grid_exponent=p["grid_exponent"])
        pw = _SOS_PAWN[(j, p["design"], p["pawn_bins"])]
        th = p["screen_threshold"]
        r_a.append(D.savage_rho(adj, ref))
        r_p.append(D.savage_rho(pw, ref))
        m_a.append(np.mean(np.abs(adj - ref)))
        m_p.append(np.mean(np.abs(pw - ref)))
        a, b = D.screening_errors(adj, ref, th)
        a_a.append(a); b_a.append(b)
        a, b = D.screening_errors(pw, ref, th)
        a_p.append(a); b_p.append(b)
    return [np.nanmean(x) for x in (r_a, m_a, a_a, b_a, r_p, m_p, a_p, b_p)]


def _sos_points(N_sos, seed=77):
    """Points of the SoS design: the stacked [A; AB_1; ...; AB_k] matrix of
    `jansen_matrices` and the matrix B of the same 2k-column design."""
    kf = len(SOS_FACTORS)
    X = D.jansen_matrices(N_sos, kf, "sobol", seed=seed)
    B = D.design_matrix(N_sos, 2 * kf, "sobol", seed)[:, kf:]
    return X, B


def _sos_evaluate(points, cache_path):
    with Pool(NPROC, initializer=_sos_init, initargs=(cache_path,)) as pool:
        return np.array(pool.map(_sos_eval, list(points), chunksize=8))


def sos_indices(Ys, YB, N_sos, n_boot=500, seed=1):
    """First-order (Saltelli et al. 2010, centred output) and total-order
    (Jansen 1999) indices of the SoS factors, with 95% percentile bootstrap
    intervals from resampling the rows of the design; the same resamples
    give the interval of the interaction share T_i - S_i.
    Ys: outputs at [A; AB_1; ...; AB_k]; YB: outputs at B."""
    kf = len(SOS_FACTORS)
    rows = []
    rng = np.random.default_rng(seed)
    for j, name in enumerate(SOS_OUT):
        y, yb = Ys[:, j], YB[:, j]
        T = D.jansen_total(y, N_sos, kf)
        S = D.saltelli_first(y, yb, N_sos, kf)
        Yb = y.reshape(kf + 1, N_sos)
        bT, bS = [], []
        # On small sub-designs a resample of an output that is almost always
        # zero (the type II error of PAWN) can have zero variance; its indices
        # are then undefined and the interval is NaN, without warnings.
        with np.errstate(invalid="ignore", divide="ignore"):
            for _ in range(n_boot):
                idx = rng.integers(0, N_sos, N_sos)
                bT.append(D.jansen_total(Yb[:, idx].ravel(), N_sos, kf))
                bS.append(D.saltelli_first(Yb[:, idx].ravel(), yb[idx],
                                           N_sos, kf))
            bT, bS = np.array(bT), np.array(bS)
            lo, hi = np.percentile(bT, [2.5, 97.5], axis=0)
            slo, shi = np.percentile(bS, [2.5, 97.5], axis=0)
            ilo, ihi = np.percentile(bT - bS, [2.5, 97.5], axis=0)
        for i, fac in enumerate(SOS_FACTORS):
            rows.append(dict(output=name, factor=fac, S=S[i], S_lo=slo[i],
                             S_hi=shi[i], T=T[i], lo=lo[i], hi=hi[i],
                             TmS=T[i] - S[i], TmS_lo=ilo[i], TmS_hi=ihi[i]))
    return pd.DataFrame(rows)


def sos_convergence_stats(Ys, YB, Ns, n_boot=500, seed=1):
    """Convergence statistics of Sarrazin et al. (2016) on the nested
    sub-designs of base size n in Ns (the first n rows of every block):
    ranking (95% quantile, over pairs of bootstrap resamples, of the rank
    distance weighted by the squared indices; converged below 1), screening
    (widest 95% interval of S_i or T_i among the parameters with T_i < 0.05;
    converged below 0.05) and values (widest interval among all parameters).
    Only the parameters that enter an output are included, and the type II
    error of PAWN, whose variance is negligible, is skipped. p_top is the
    share of resamples in which the parameter with the largest T_i keeps a
    larger T_i than the second."""
    kf, N = len(SOS_FACTORS), len(YB)
    Yall = Ys.reshape(kf + 1, N, -1)

    def rank_stat(B):
        Rk = (-B).argsort(axis=1).argsort(axis=1)
        j, m = np.triu_indices(len(B), 1)
        w = np.maximum(B[j], B[m]) ** 2
        return np.quantile((np.abs(Rk[j] - Rk[m]) * w).sum(1) / w.sum(1),
                           0.95)

    rows = []
    for n in Ns:
        rng = np.random.default_rng(seed)
        for o, name in enumerate(SOS_OUT):
            if name == "beta_pawn":
                continue
            Yb, yb = Yall[:, :n, o], YB[:n, o]
            act = [i for i in range(kf)
                   if np.any(Yall[i + 1, :, o] != Yall[0, :, o])]
            T = D.jansen_total(Yb.ravel(), n, kf)[act]
            bT, bS = [], []
            for _ in range(n_boot):
                idx = rng.integers(0, n, n)
                bT.append(D.jansen_total(Yb[:, idx].ravel(), n, kf)[act])
                bS.append(D.saltelli_first(Yb[:, idx].ravel(), yb[idx], n,
                                           kf)[act])
            bT, bS = np.array(bT), np.array(bS)
            wT = np.ptp(np.percentile(bT, [2.5, 97.5], axis=0), axis=0)
            wS = np.ptp(np.percentile(bS, [2.5, 97.5], axis=0), axis=0)
            low = T < 0.05
            top = np.argsort(-T)[:2]
            rows.append(dict(
                N=n, output=name, ranking_T=rank_stat(bT),
                ranking_S=rank_stat(bS),
                screening=np.r_[wT[low], wS[low]].max() if low.any()
                else np.nan,
                values=max(wT.max(), wS.max()),
                p_top=np.mean(bT[:, top[0]] > bT[:, top[1]])))
    return pd.DataFrame(rows)


def _sos_pawn_cache(cache_path):
    cache = list(np.load(cache_path, allow_pickle=True))
    pawn = {}
    for j, (c, ref, cc) in enumerate(cache):
        for design, (U, Y) in cc.items():
            for nb in range(5, 21):
                pawn[(j, design, nb)] = D.pawn_given_data(U, Y, n_bins=nb)
    np.save(cache_path.replace("sos_cache", "sos_pawn"), pawn,
            allow_pickle=True)


def _sos_require_cache(cache_path):
    if not os.path.exists(cache_path):
        raise SystemExit(f"{cache_path} not found: run 'python experiments.py "
                         "becker' first; it stores the samples on which the "
                         "sensitivity analysis of the method is evaluated.")


def run_sos(N_sos=2 ** 12, cache_dir=OUT):
    """Sensitivity analysis of the method: (k+2)N_sos evaluations."""
    cache_path = f"{cache_dir}/sos_cache.npy"
    _sos_require_cache(cache_path)
    _sos_pawn_cache(cache_path)
    X, B = _sos_points(N_sos)
    t0 = time.time()
    Ys = _sos_evaluate(X, cache_path)
    YB = _sos_evaluate(B, cache_path)
    pd.DataFrame(Ys, columns=SOS_OUT).to_csv(f"{OUT}/sos_outputs.csv",
                                             index=False)
    pd.DataFrame(YB, columns=SOS_OUT).to_csv(f"{OUT}/sos_outputs_B.csv",
                                             index=False)
    sos_indices(Ys, YB, N_sos).to_csv(f"{OUT}/sos_indices.csv", index=False)
    print(f"sos done in {time.time()-t0:.0f}s")


def run_sos_first_order(N_sos=2 ** 12, cache_dir=OUT):
    """Adds the N_sos evaluations at B to an existing run of the
    [A; AB_i] blocks (sos_outputs.csv) and recomputes all indices."""
    cache_path = f"{cache_dir}/sos_cache.npy"
    _sos_require_cache(cache_path)
    if not os.path.exists(cache_path.replace("sos_cache", "sos_pawn")):
        _sos_pawn_cache(cache_path)
    Ys = pd.read_csv(f"{OUT}/sos_outputs.csv")[SOS_OUT].values
    X, B = _sos_points(N_sos)
    assert len(Ys) == len(X), "sos_outputs.csv does not match the design"
    t0 = time.time()
    YB = _sos_evaluate(B, cache_path)
    pd.DataFrame(YB, columns=SOS_OUT).to_csv(f"{OUT}/sos_outputs_B.csv",
                                             index=False)
    sos_indices(Ys, YB, N_sos).to_csv(f"{OUT}/sos_indices.csv", index=False)
    print(f"sos (first order) done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    what = sys.argv[1]
    quick = "--quick" in sys.argv
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    if what == "benchmarks":
        run_benchmarks(R=4 if quick else 100)
    elif what == "convergence":
        fns = args or None
        run_convergence(R=2 if quick else 20, functions=fns)
    elif what == "hymod":
        default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "data")
        run_hymod(args[0] if args else default)
    elif what == "becker":
        run_becker(n_conf=16 if quick else 512)
    elif what == "sos":
        run_sos(N_sos=2 ** 4 if quick else 2 ** 12)
    elif what == "sos_first":
        # optional argument: directory holding sos_cache.npy
        run_sos_first_order(N_sos=2 ** 4 if quick else 2 ** 12,
                            cache_dir=args[0] if args else OUT)
    else:
        raise SystemExit(__doc__)
