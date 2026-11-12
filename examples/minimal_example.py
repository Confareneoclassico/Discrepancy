"""Minimal example: adjusted ersatz, PCE and PAWN on the Sobol' G function."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
import discrepancy as D  # noqa: E402

U = D.design_matrix(512, 8, design="srs", seed=1)     # inputs on [0, 1]
Y = D.sobol_g(U)

adj = D.ersatz_all(U, Y)                               # adjusted ersatz
T, Sh, fit = D.pce_indices(U, Y)                       # PCE total-order, Shapley
pawn = D.pawn_given_data(U, Y)                         # given-data PAWN

a = D.SOBOL_G_A
V = 1 / (3 * (1 + a) ** 2)
P = np.prod(1 + V)
T_exact = P * V / (1 + V) / (P - 1)

print("input  T_exact  adj_ersatz  PCE_T  PAWN")
for i in range(8):
    print(f"x{i+1}    {T_exact[i]:.3f}    {adj[i]:.3f}       {T[i]:.3f}  {pawn[i]:.3f}")
print(f"PCE degree {fit['p']}, corrected leave-one-out error {fit['loo']:.4f}")
print("Savage-score correlation with T:",
      {"adj": round(D.savage_rho(adj, T_exact), 3),
       "PCE": round(D.savage_rho(T, T_exact), 3),
       "PAWN": round(D.savage_rho(pawn, T_exact), 3)})
