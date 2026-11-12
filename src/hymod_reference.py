"""
hymod_reference.py -- reference indices for the HYMOD sample.

No independent Jansen design is available for the HYMOD configuration of
the paper, so the reference total-order indices are computed from the
50,000-run given-data sample through two surrogates of different nature:

  * a degree-adaptive PCE (discrepancy.pce_fit; degree 7, corrected
    leave-one-out error about 0.1% of the output variance);
  * a gradient-boosting regression (scikit-learn), test R^2 about 0.998.

The Jansen estimator (base size 2^15, scrambled Sobol' design) is applied
to each surrogate; the two sets of indices agree to within 0.002 and their
average is used.  The delta reference is the plug-in histogram estimator
(64 x 64 equiprobable cells) on 2^23 points of the PCE surrogate.

Writes results/hymod_refs.json.
"""
import json
import os

import numpy as np

import discrepancy as D
from experiments import load_hymod

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def pce_surrogate(U, Y):
    fit = D.pce_fit(U, Y)

    def f(X, chunk=2 ** 16):
        out = np.empty(len(X))
        for a in range(0, len(X), chunk):
            out[a:a + chunk] = D._pce_design(X[a:a + chunk], fit["alphas"],
                                             "legendre") @ fit["coef"]
        return out
    return f, fit


def gbm_surrogate(U, Y):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    Utr, Ute, Ytr, Yte = train_test_split(U, Y, test_size=0.2,
                                          random_state=0)
    kw = dict(max_iter=3000, learning_rate=0.05, max_leaf_nodes=63,
              early_stopping=True, random_state=0)
    r2 = 1 - np.mean((HistGradientBoostingRegressor(**kw).fit(Utr, Ytr)
                      .predict(Ute) - Yte) ** 2) / Yte.var()
    model = HistGradientBoostingRegressor(**kw).fit(U, Y)
    return model.predict, r2


def jansen_on(f, k, N=2 ** 15, seed=5):
    X = D.jansen_matrices(N, k, "sobol", seed=seed)
    return D.jansen_total(f(X), N, k)


def delta_on(f, k, S=64, n_chunks=8, chunk=2 ** 20, seed=3):
    rng = np.random.default_rng(seed)
    pilot = f(rng.random((2 ** 20, k)))
    edges = np.quantile(pilot, np.linspace(0, 1, S + 1)[1:-1])
    counts = np.zeros((k, S, S))
    for _ in range(n_chunks):
        X = rng.random((chunk, k))
        col = np.searchsorted(edges, f(X), side="right")
        row = np.minimum((X * S).astype(int), S - 1)
        for i in range(k):
            counts[i] += np.bincount(row[:, i] * S + col,
                                     minlength=S * S).reshape(S, S)
    out = []
    for i in range(k):
        P = counts[i] / counts[i].sum()
        out.append(0.5 * np.abs(P - np.outer(P.sum(1), P.sum(0))).sum())
    return np.array(out)


if __name__ == "__main__":
    U, Y = load_hymod(os.path.join(DATA, "hydro.csv"),
                      os.path.join(DATA, "hydro2.csv"))
    f_pce, fit = pce_surrogate(U, Y)
    f_gbm, r2 = gbm_surrogate(U, Y)
    T_pce, T_gbm = jansen_on(f_pce, 5), jansen_on(f_gbm, 5)
    delta = delta_on(f_pce, 5)
    print("PCE degree", fit["p"], "corrected LOO", round(fit["loo"], 5))
    print("GBM test R2", round(r2, 5))
    print("T (PCE)", T_pce.round(4), " T (GBM)", T_gbm.round(4))
    print("delta", delta.round(4))
    os.makedirs("results", exist_ok=True)
    json.dump({"hymod_T_surrogate": ((T_pce + T_gbm) / 2).tolist(),
               "hymod_T_pce": T_pce.tolist(), "hymod_T_gbm": T_gbm.tolist(),
               "hymod_delta": delta.tolist()},
              open("results/hymod_refs.json", "w"), indent=1)
