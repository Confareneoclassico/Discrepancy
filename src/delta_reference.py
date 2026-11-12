"""
delta_reference.py -- large-sample reference values of Borgonovo's delta.

delta_i = (1/2) * int int |c_i(u,v) - 1| du dv  (copula form), estimated by
the plug-in estimator on an s x s grid with s = 64 equiprobable bins per
axis and N = 2^24 points (about 4,100 points per cell), accumulated in
chunks.  Column (output) bins are empirical quantiles of a pilot sample
of 2^22 points; both margins are estimated from the counts.  The
residual positive bias of the plug-in estimator is of order
0.4/sqrt(N/s^2) ~ 0.006 for an independent input.
"""
import json
import sys

import numpy as np

from experiments import FUNCTIONS

S = 64
N_TOTAL = 2 ** 24
CHUNK = 2 ** 20


def delta_reference(fname, seed=31):
    f, k, _ = FUNCTIONS[fname]
    rng = np.random.default_rng(seed)
    pilot = f(rng.random((2 ** 22, k)))
    edges = np.quantile(pilot, np.linspace(0, 1, S + 1)[1:-1])
    counts = np.zeros((k, S, S))
    for _ in range(N_TOTAL // CHUNK):
        U = rng.random((CHUNK, k))
        Y = f(U)
        col = np.searchsorted(edges, Y, side="right")
        row = np.minimum((U * S).astype(int), S - 1)
        for i in range(k):
            counts[i] += np.bincount(row[:, i] * S + col,
                                     minlength=S * S).reshape(S, S)
    out = []
    for i in range(k):
        P = counts[i] / counts[i].sum()
        Q = np.outer(P.sum(1), P.sum(0))
        out.append(0.5 * np.abs(P - Q).sum())
    return np.array(out)


if __name__ == "__main__":
    names = sys.argv[1:] or list(FUNCTIONS)
    res = {}
    for n in names:
        res[n] = delta_reference(n).tolist()
        print(n, np.round(res[n], 4), flush=True)
    # update the stored references: a run on a subset of the functions must
    # not delete the references of the others
    path = "results/delta_refs.json"
    try:
        with open(path) as fh:
            stored = json.load(fh)
    except FileNotFoundError:
        stored = {}
    stored.update(res)
    with open(path, "w") as fh:
        json.dump(stored, fh, indent=1)
