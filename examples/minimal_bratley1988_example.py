"""
Minimal worked example: reproduce the Bratley (1988) row of Table 1 in the
manuscript using three of the five estimators directly, without running the
full src/discrepancy.py pipeline (which additionally runs six more benchmark
functions, a convergence study, the Becker stress test, and the joint
sensitivity-of-sensitivity analysis).

Run from the repository root:
    python examples/minimal_bratley1988_example.py

Expected output (verified against src/discrepancy.py, June 2026):
    adjusted ersatz : [0.490, 0.085, 0.019, 0.023, 0.025, 0.023, 0.009, 0.028]
    PCE T_i         : [0.830, 0.360, 0.168, 0.087, 0.026, 0.011, 0.003, 0.001]
    reference T_i   : [0.747, 0.254, 0.082, 0.029, 0.008, 0.004, 0.001, 0.001]
                      (computed separately at N=2^14 via the Jansen estimator;
                       not reproduced in this minimal example, see src/discrepancy.py
                       lines ~1550-1565 for the reference computation)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from discrepancy import (
    sobol_mat,
    bratley1988_fun,
    discrepancy_ersatz,
    pce_total_indices,
    pce_shapley_indices,
)

K = 8
N = 512
PARAMS = [f"x{i}" for i in range(1, K + 1)]

# Reference T_i at N=512 would be noisy; the manuscript instead uses a
# separate N=2^14 Jansen-estimator run for the reference (see src/discrepancy.py).
# This example only demonstrates the screening estimators themselves.

mat = sobol_mat(N=N, params=PARAMS, matrices=("A",), seed=123)
Y = bratley1988_fun(mat)

adj_ersatz = discrepancy_ersatz(mat, Y, PARAMS, adj=1)
pce_ti = pce_total_indices(mat, Y, PARAMS)
shapley = pce_shapley_indices(mat, Y, PARAMS)

print("Adjusted ersatz discrepancy:")
print(adj_ersatz.to_string(index=False))
print()
print("PCE total-order index:")
print(pce_ti.to_string(index=False))
print()
print("PCE-derived Shapley effects (should sum to ~1.0):")
print(shapley.to_string(index=False))
print(f"  sum = {shapley['value'].sum():.4f}")
