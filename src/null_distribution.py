"""
null_distribution.py -- distribution of the adjusted ersatz of an input that
is independent of the output (Supplementary Material A, Section 9, Table 2;
main text, Figure 5b).

For each N and each design, X_1 is the tested input and Y is the sum of five
other inputs.  Under simple random sampling (SRS) the null distribution
depends only on N and s, whatever the law of Y; under scrambled Sobol'
sequences it depends on the model, and the sum of five inputs is a
representative case.  Monte Carlo standard errors of the mean, of the
quantiles and of the exceedance probabilities are computed by the bootstrap.

Writes results/null_floor.csv.
"""
import os
import time

import numpy as np
import pandas as pd

import discrepancy as D

NS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
THETAS = (0.01, 0.05, 0.10)


def replicates(N, design):
    if design == "lhs":                      # maximin LHS is the slow one
        return 2000 if N <= 1024 else (500 if N <= 2048 else 300)
    return 10000


def simulate(N, design, R, seed0=97_000):
    out = np.empty(R)
    for r in range(R):
        U = D.design_matrix(N, 6, design, seed=seed0 * N + r)
        out[r] = D.ersatz_input(U[:, 0], U[:, 1:].sum(1))
    return out


def summarise(v, rng, B=200):
    stats = lambda x: np.concatenate([[x.mean(), x.std(ddof=1)],
                                      np.quantile(x, [0.05, 0.95, 0.99]),
                                      [(x > t).mean() for t in THETAS]])
    est = stats(v)
    boot = np.array([stats(rng.choice(v, len(v))) for _ in range(B)])
    return est, boot.std(axis=0, ddof=1)


if __name__ == "__main__":
    rng = np.random.default_rng(2027)
    names = (["mean", "sd", "q05", "q95", "q99"]
             + [f"p_gt_{t:.2f}" for t in THETAS])
    rows = []
    t0 = time.time()
    for N in NS:
        s = int(np.ceil(np.sqrt(N)))
        for design in ("srs", "lhs", "sobol"):
            R = replicates(N, design)
            est, se = summarise(simulate(N, design, R), rng)
            row = dict(N=N, s=s, lambda_N=N / s ** 2, design=design, R=R)
            row.update(dict(zip(names, est)))
            row.update({f"se_{k}": x for k, x in zip(names, se)})
            rows.append(row)
            print(f"N={N:5d} {design:5s} R={R:5d} mean={est[0]:.4f} "
                  f"sd={est[1]:.4f} q95={est[3]:.4f} q99={est[4]:.4f} "
                  f"P>.01={est[5]:.4f} P>.05={est[6]:.4f} P>.10={est[7]:.4f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/null_floor.csv", index=False)
