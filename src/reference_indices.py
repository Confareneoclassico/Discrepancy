"""
reference_indices.py -- recompute the reference total-order indices stored
in refs.json and report the differences.

Analytical values: Sobol' G function, Bratley-Fox (1988) product function,
Ishigami function.  Jansen estimates on scrambled Sobol' designs: Bratley et
al. (1992), base size 2^17; Oakley-O'Hagan and Multiverse, base size 2^16.
The stored estimates come from one scrambling of the design; the
recomputation uses another (seed 1).  With --seeds, the Jansen estimates are
recomputed with 14 seeds (seed 1 among them): they differ from the stored
values by at most 7.1e-4, within the 1e-3 stated in the main text
(Section 2.7).

Writes results/refs_recomputed.json; refs.json itself is not modified.
"""
import json
import os
import sys

import numpy as np

import discrepancy as D


def analytic():
    a = D.SOBOL_G_A
    V = 1 / (3 * (1 + a) ** 2)
    P = np.prod(1 + V)
    g = P * V / (1 + V) / (P - 1)
    Vb = np.full(8, 1 / 3)                          # |4x - 2| has variance 1/3
    Pb = np.prod(1 + Vb)
    bf = Pb * Vb / (1 + Vb) / (Pb - 1)
    A, B = 7.0, 0.1                                  # Ishigami
    V1 = 0.5 * (1 + B * np.pi ** 4 / 5) ** 2
    V2 = A ** 2 / 8
    V13 = B ** 2 * np.pi ** 8 * (1 / 18 - 1 / 50)
    Vt = V1 + V2 + V13
    ish = np.array([(V1 + V13) / Vt, V2 / Vt, V13 / Vt])
    return {"sobol_g": g, "bratley1988": bf, "ishigami": ish}


def jansen(f, k, base, seed=1):
    X = D.jansen_matrices(base, k, "sobol", seed=seed)
    return D.jansen_total(f(X), base, k)


if __name__ == "__main__":
    out = analytic()
    out["bratley1992"] = jansen(D.bratley1992, 8, 2 ** 17)
    out["oakley"] = jansen(D.oakley_ohagan, 15, 2 ** 16)
    out["multiverse"] = jansen(D.multiverse, 5, 2 ** 16)
    stored = {k: np.array(v) for k, v in json.load(open("refs.json")).items()}
    for k, v in out.items():
        print(f"{k:12s} max |recomputed - stored| = "
              f"{np.max(np.abs(v - stored[k])):.1e}")
    os.makedirs("results", exist_ok=True)
    json.dump({k: v.tolist() for k, v in out.items()},
              open("results/refs_recomputed.json", "w"), indent=1)
    if "--seeds" in sys.argv:
        seeds = list(range(8)) + [42, 123, 2024, 2025, 2026, 12345]
        for name, f, k, base in [("bratley1992", D.bratley1992, 8, 2 ** 17),
                                 ("oakley", D.oakley_ohagan, 15, 2 ** 16),
                                 ("multiverse", D.multiverse, 5, 2 ** 16)]:
            dev = [np.max(np.abs(jansen(f, k, base, seed=sd) - stored[name]))
                   for sd in seeds]
            print(f"{name:12s} {len(seeds)} seeds: max |recomputed - stored|"
                  f" = {max(dev):.1e}")
