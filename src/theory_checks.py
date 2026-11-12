"""
theory_checks.py -- numerical verifications for Supplementary Material A.

1. Constants and polynomial identities of Proposition 5.4 (exact or 40-digit
   arithmetic): phi(p), kappa_1, the roots of q, phi(p_0), L(e^{-3/4}).
2. Computer-assisted parts of Proposition 5.4: (iv) monotone bracketing of
   h_lam(t) <= 1 - t + kappa_lam on [0, 0.98] x [1/2, 1] and the bound on
   |h_lam'| on [0.98, 1] x [1/2, 1]; (iii) the tangent-line inequality on
   [lambda_c, 3/4): a monotonicity argument on [lambda_c, 0.55] x [0, 0.07]
   and monotone bracketing on the rest of [0, t0) x [lambda_c, 3/4).
3. Numerical thresholds: lambda_c (tangent-line inequality), the largest
   lambda with eps_lam = 0, eps_1, eps at lambda = 512/529.
4. Counterexamples of Proposition 5.4(iii) and Remark 7.3 (circulant and
   checkerboard copulas).
5. Gaussian copulas (Table 3 and Remarks 5.5, 7.3): Phi_1, Psi_1 and delta
   by quadrature in rotated normal-score coordinates, checked by Monte Carlo.
6. Remark 7.3: sharp ranking margin (linear programme) and its optimiser,
   and a strict reversal simulated at N = 2^14; the failure range of Proposition 5.4(iii);
   Section 8: Sobol' projections and the plain Latin hypercube null.

Run time: a few minutes.
"""
import numpy as np
import sympy as sp
from mpmath import mp, mpf, exp as mexp, log as mlog
from numpy.polynomial.legendre import leggauss
from scipy import integrate
from scipy.optimize import brentq
from scipy.special import ndtr

mp.dps = 40


# ---------------------------------------------------------------------
# 1. Exact constants
# ---------------------------------------------------------------------
p = sp.symbols("p")
phi_sym = sp.expand(sum(sp.binomial(8, l) * (1 - p) ** l * p ** (8 - l)
                        for l in range(0, 4)) * p)
# p * P[Bin(8, 1-p) <= 3] = p * sum_{l<=3} C(8,l) (1-p)^l p^(8-l)
assert sp.expand(phi_sym - p ** 6 * (56 - 140 * p + 120 * p ** 2
                                     - 35 * p ** 3)) == 0
dphi_sym = sp.diff(phi_sym, p)
assert sp.expand(dphi_sym - p ** 5 * (336 - 980 * p + 960 * p ** 2
                                      - 315 * p ** 3)) == 0
q_sym = sp.expand(sp.diff(p * dphi_sym, p))
assert sp.expand(q_sym - p ** 5 * (2016 - 6860 * p + 7680 * p ** 2
                                   - 2835 * p ** 3)) == 0
r3 = 336 - 980 * p + 960 * p ** 2 - 315 * p ** 3
print("discriminant of r3'", sp.discriminant(sp.diff(r3, p), p), " r3(1) =",
      r3.subs(p, 1))
cubic = sp.Poly(2016 - 6860 * p + 7680 * p ** 2 - 2835 * p ** 3, p)
roots = [sp.N(r, 30) for r in cubic.real_roots()]
print("real roots of the cubic factor of q:", [f"{float(r):.7f}" for r in roots])
p0 = roots[0]
phi_f = sp.lambdify(p, phi_sym, "mpmath")
dphi_f = sp.lambdify(p, dphi_sym, "mpmath")
L = lambda x: phi_f(x) - x * mlog(x) * dphi_f(x)
print("kappa_1 = phi(e^-1) =", mp.nstr(phi_f(mexp(-1)), 12))
print("phi(p0) =", mp.nstr(phi_f(mpf(str(p0))), 12),
      " L(e^{-3/4}) =", mp.nstr(L(mexp(mpf(-3) / 4)), 12))
lam_star = mp.findroot(lambda l: L(mexp(-l)) - phi_f(mpf(str(p0))), 0.74)
print("proof of Prop 5.4(ii) works for lambda >=", mp.nstr(lam_star, 8))

# ---------------------------------------------------------------------
# 2. Computer-assisted bracketing for Proposition 5.4(iv)
# ---------------------------------------------------------------------
phi = lambda x: x ** 6 * (56 - 140 * x + 120 * x ** 2 - 35 * x ** 3)
dphi = lambda x: x ** 5 * (336 - 980 * x + 960 * x ** 2 - 315 * x ** 3)
h = lambda t, lam: phi(np.exp(-lam * t))
dh = lambda t, lam: -lam * np.exp(-lam * t) * dphi(np.exp(-lam * t))

ta = np.arange(0, 0.98, 5e-5)
tb = ta + 5e-5
la = np.arange(0.5, 1.0, 1e-3)
lb = la + 1e-3
worst = -np.inf
for l1, l2 in zip(la, lb):
    up = phi(np.exp(-l1 * ta))                     # max of h on the cell
    lo = 1 - tb + phi(np.exp(-l2))                 # min of 1 - t + kappa
    worst = max(worst, np.max(up - lo))
print(f"bracketing on [0,0.98]x[1/2,1]: max(upper - lower) = {worst:.5f}"
      " (must be < 0)")
psi = lambda x: x * dphi(x)        # increasing on (0, p0)
ta2 = np.arange(0.98, 1.0, 5e-5)
bound = max(np.max(l2 * psi(np.exp(-l1 * ta2))) for l1, l2 in zip(la, lb))
assert np.exp(-0.5 * 0.98) < float(p0)
print(f"|h'| <= {bound:.4f} on [0.98,1]x[1/2,1] (must be < 1)")

# ---------------------------------------------------------------------
# 2b. Computer-assisted part of Proposition 5.4(iii): for lambda in
#     [lambda_c, 3/4), D(t) = h_lam(t) - l(t) >= 0 with
#     l(t) = kappa_lam + lam psi(e^{-lam}) (1 - t). On [t0, inf),
#     t0 = -log(p0)/lam < 1, h_lam is strictly convex (exact). On
#     [lambda_c, 0.55] x [0, 0.07], e^{-lam t} lies in [p1, 1], where
#     psi <= psi(1) = 1, so D' = lam [psi(e^{-lam}) - psi(e^{-lam t})]
#     >= lam [psi(e^{-0.55}) - 1] > 0 and D(t) >= D(0) = 1 - L(e^{-lam}) >= 0.
#     Elsewhere: monotone bracketing with h_lam(t) >= phi(e^{-l2 tb}) and
#     l(t) <= phi(e^{-l1}) + l2 psi(e^{-l1}) (1 - ta) (phi increasing;
#     psi increasing on (0, p0) and e^{-l1} <= e^{-lambda_c} < p0).
# ---------------------------------------------------------------------
lam_c = mp.findroot(lambda l: L(mexp(-l)) - 1, 0.48)
assert mexp(-lam_c) < mpf(str(p0))
assert 0.55 * 0.07 < -mlog(mpf(str(roots[1])))            # e^{-lam t} >= p1
psi_055 = mexp(-mpf("0.55")) * dphi_f(mexp(-mpf("0.55")))
assert psi_055 > 1
t0_hi = lambda lam: float(-mlog(mpf(str(p0))) / lam) + 1e-9   # t0, rounded up


def tangent_bracket(t_lo, t_hi, l_lo, l_hi, size=1e-3):
    ta = np.arange(t_lo, t_hi, size)
    la = np.arange(l_lo, l_hi, size)
    TA, LA = np.meshgrid(ta, la)
    TB, LB = np.minimum(TA + size, t_hi), np.minimum(LA + size, l_hi)
    low = (phi(np.exp(-LB * TB)) - phi(np.exp(-LA))
           - LB * psi(np.exp(-LA)) * (1 - TA))
    return low.size, low.min()


nA, mA = tangent_bracket(0.07, t0_hi(float(lam_c)), float(lam_c), 0.55)
nB, mB = tangent_bracket(0.0, t0_hi(0.55), 0.55, 0.75)
print(f"Prop 5.4(iii) on [lambda_c, 3/4): psi(e^-0.55) = {float(psi_055):.4f} > 1;"
      f" bracketing on {nA + nB} cells of size 1e-3: min lower bound"
      f" {min(mA, mB):.4f} (must be > 0)")

# ---------------------------------------------------------------------
# 3. Numerical thresholds
# ---------------------------------------------------------------------
tt = np.concatenate([np.linspace(0, 3, 300001), np.linspace(3, 60, 200001)])
gap = lambda lam: np.min(h(tt, lam) - h(1, lam) - dh(1, lam) * (tt - 1))
lam_c = brentq(lambda l: gap(l) + 1e-14, 0.45, 0.6)
print(f"tangent-line inequality holds numerically for lambda >= {lam_c:.4f}")
t01 = np.linspace(0, 1, 200001)
eps = lambda lam: np.max((1 - t01) - h(t01, lam))
print(f"eps_1 = {eps(1):.4f} at t = {t01[np.argmax((1 - t01) - h(t01, 1))]:.4f};"
      f" eps(512/529) = {eps(512 / 529):.4f}; kappa(512/529) ="
      f" {phi(np.exp(-512 / 529)):.4f}")
print(f"eps_lam = 0 for lambda <= {brentq(lambda l: eps(l) - 1e-12, .75, .85):.4f}")

# ---------------------------------------------------------------------
# 4. Counterexamples (circulant copulas c(u,v) = q((u+v) mod 1))
# ---------------------------------------------------------------------
lam = 0.2268                                     # N = 1600, alpha_g = 0.6
print(f"band copula, T=2, lambda={lam}: Psi = {0.5 + h(2, lam) / 2:.4f} <"
      f" kappa = {h(1, lam):.4f}; delta = 0.5")
for c0, c1 in [(0.0, 1.62), (0.586, 10.0)]:
    w = (c1 - 1) / (c1 - c0)
    print(f"two-valued density {{{c0}, {c1}}}: delta = {w * (1 - c0):.4f},"
          f" Psi_1 = {w * h(c0, 1) + (1 - w) * h(c1, 1):.4f}")
M = np.array([[0.1, 0.3, 1.8, 1.8], [0.3, 0.1, 1.8, 1.8],
              [1.8, 1.8, 0.2, 0.2], [1.8, 1.8, 0.2, 0.2]])
M2 = M.copy()
M2[0] = M2[1] = (M[0] + M[1]) / 2
for name, X in (("checkerboard", M), ("after merging rows 1-2", M2)):
    print(f"{name}: delta = {np.mean(np.maximum(1 - X, 0)):.4f},"
          f" Psi_1 = {np.mean(h(X, 1)):.4f}, Phi_1 = {np.mean(np.exp(-X)):.4f}")


# ---------------------------------------------------------------------
# 5. Gaussian copulas
# ---------------------------------------------------------------------
def npdf(z):
    return np.exp(-z * z / 2) / np.sqrt(2 * np.pi)


def functional(H, rho, n=600, S=11.0):
    """E[H(c_rho(Z1,Z2))] for independent standard normal (Z1,Z2), with
    c_rho = A exp(a s^2 - b d^2) in rotated coordinates s, d."""
    if rho == 0:
        return float(H(1.0))
    A0 = 1 / np.sqrt(1 - rho ** 2)
    a = rho / (2 * (1 + rho))
    w = np.sqrt((1 - rho) / rho)                  # d = w x, c = A e^{-x^2/2}
    xs, wx = leggauss(n)
    ss, ws = leggauss(n)
    ss, ws = ss * S, ws * S
    tot = 0.0
    for s_, wsi in zip(ss, ws):
        logA = np.log(A0) + a * s_ * s_
        X = np.sqrt(max(2 * (logA + 40), 1.0))   # beyond: c < e^{-40}
        x, wxx = xs * X, wx * X
        c = np.exp(logA - x * x / 2)
        tot += wsi * npdf(s_) * (1 + w * np.sum(wxx * (H(c) - 1) * npdf(x * w)))
    return tot


def delta_gauss(rho):
    """delta = E_X TV(N(rho X, 1 - rho^2), N(0,1))."""
    if rho == 0:
        return 0.0
    sg = np.sqrt(1 - rho ** 2)

    def tv(x):
        mu = rho * x
        A = 0.5 - 0.5 / sg ** 2
        B = mu / sg ** 2
        C = -mu ** 2 / (2 * sg ** 2) - np.log(sg)
        disc = B ** 2 - 4 * A * C
        if disc <= 0:
            return 0.0
        y1, y2 = sorted([(-B + np.sqrt(disc)) / (2 * A),
                         (-B - np.sqrt(disc)) / (2 * A)])
        return (ndtr((y2 - mu) / sg) - ndtr((y1 - mu) / sg)
                - (ndtr(y2) - ndtr(y1)))
    return integrate.quad(lambda x: tv(x) * npdf(x), -12, 12, limit=400,
                          epsabs=1e-13)[0]


def monte_carlo(rho, n=10 ** 7, seed=0):
    rng = np.random.default_rng(seed)
    z1, z2 = rng.standard_normal((2, n))
    c = np.exp(-(rho ** 2 * (z1 ** 2 + z2 ** 2) - 2 * rho * z1 * z2)
               / (2 * (1 - rho ** 2))) / np.sqrt(1 - rho ** 2)
    return (np.exp(-c).mean(), phi(np.exp(-c)).mean(),
            np.maximum(1 - c, 0).mean())


Phi1 = lambda c: np.exp(-c)
Psi1 = lambda c: phi(np.exp(-c))
print("\nTable 3:  rho  Phi_1  Psi_1  delta   (Monte Carlo, 1e7 draws)")
for rho in (0.0, 0.3, 0.6, 0.9, 0.99):
    mc = monte_carlo(rho) if rho > 0 else (np.exp(-1), phi(np.exp(-1)), 0)
    print(f"  {rho:.2f}  {functional(Phi1, rho):.4f}  {functional(Psi1, rho):.4f}"
          f"  {delta_gauss(rho):.4f}   ({mc[0]:.4f} {mc[1]:.4f} {mc[2]:.4f})")
rhos = np.concatenate([np.linspace(0, 0.95, 191), np.linspace(0.951, 0.999, 49)])
P = np.array([functional(Psi1, r) for r in rhos])
Dl = np.array([delta_gauss(r) for r in rhos])
print(f"rho scan ({len(rhos)} values in [0, 0.999]): max|Psi_1 - delta| ="
      f" {np.max(np.abs(P - Dl)):.5f} at rho = {rhos[np.argmax(np.abs(P - Dl))]:.3f};"
      f" max(delta - Psi_1) = {np.max(Dl - P):.4f} at rho ="
      f" {rhos[np.argmax(Dl - P)]:.3f}; Psi_1 increasing: {bool(np.all(np.diff(P) > 0))}")


# ---------------------------------------------------------------------
# 6. Ranking by Psi_1 against delta (Remark 7.3): sharp margin by linear
#    programming over two laws of the density values, and a strict
#    reversal simulated at N = 2^14
# ---------------------------------------------------------------------
def sharp_margin(lam):
    from scipy.optimize import linprog
    t = np.concatenate([np.linspace(0, 3, 6001), np.linspace(3.01, 30, 800)])
    hh, dd, K = phi(np.exp(-lam * t)), np.maximum(1 - t, 0), len(t)
    res = linprog(np.concatenate([dd, -dd]),
                  A_ub=np.vstack([np.concatenate([-hh, hh]),
                                  np.concatenate([t, np.zeros(K)]),
                                  np.concatenate([np.zeros(K), t])]),
                  b_ub=[0, 1, 1],
                  A_eq=np.vstack([np.r_[np.ones(K), np.zeros(K)],
                                  np.r_[np.zeros(K), np.ones(K)]]),
                  b_eq=[1, 1], bounds=(0, None), method="highs")
    return -res.fun


def sample_Ca(a2, N, rng):
    """C_a of Proposition 5.2: density 0 on [0,a]^2, 1 elsewhere, and
    singular mass a^2 on the diagonal of [0,a]^2."""
    a = np.sqrt(a2)
    U, V = np.empty(N), np.empty(N)
    sing = rng.random(N) < a2
    w = rng.random(sing.sum()) * a
    U[sing], V[sing] = w, w
    k, uu, vv = int((~sing).sum()), [], []
    while sum(len(x) for x in uu) < k:
        x, y = rng.random(2 * k + 10), rng.random(2 * k + 10)
        keep = ~((x < a) & (y < a))
        uu.append(x[keep]); vv.append(y[keep])
    U[~sing], V[~sing] = np.concatenate(uu)[:k], np.concatenate(vv)[:k]
    return U, V


def sample_mixture(m, N, rng):
    """(1 - m) Pi + m M, M the copula of (U, U)."""
    U, V = rng.random(N), rng.random(N)
    sing = rng.random(N) < m
    V[sing] = U[sing]
    return U, V


if __name__ == "__main__":
    import discrepancy as D
    for lam in (1.0, 512 / 529):
        print(f"sharp ranking margin at lambda = {lam:.4f}: {sharp_margin(lam):.4f}")
    # optimiser at lambda = 1: C_a (density values 0 and 1) against the
    # mixture w Pi + (1 - w) M; at the optimum h_1'(w) = -(1 - kappa_1)
    # (40-digit arithmetic, mp.dps = 40)
    k1m = phi_f(mexp(-1))
    hm = lambda t: phi_f(mexp(-t))
    dhm = lambda t: -mexp(-t) * dphi_f(mexp(-t))
    w = mp.findroot(lambda x: -dhm(x) - (1 - k1m), 0.6)
    a2 = (hm(w) - k1m) / (1 - k1m)
    print(f"optimiser at lambda = 1: C_a with a^2 = {mp.nstr(a2, 6)} against"
          f" w Pi + (1 - w) M, w = {mp.nstr(w, 6)}; delta gap"
          f" {mp.nstr(1 - w - a2, 6)}, Psi_1 = {mp.nstr(hm(w), 6)} for both")
    print(f"C_a, a^2 = 0.22: delta = 0.22, Psi_1 = {0.22 + 0.78 * h(1, 1):.4f};"
          f" 0.61 Pi + 0.39 M: delta = 0.39, Psi_1 = {h(0.61, 1):.4f}")
    rng = np.random.default_rng(2026)
    a = [D.ersatz_input(*sample_Ca(0.22, 2 ** 14, rng)) for _ in range(20)]
    b = [D.ersatz_input(*sample_mixture(0.39, 2 ** 14, rng)) for _ in range(20)]
    print(f"N = 2^14, 20 replicates: adjusted ersatz {np.mean(a):.4f} (C_a) and"
          f" {np.mean(b):.4f} (mixture)")
    # lambda_c is the root of L(e^{-lambda}) = 1; failure ranges of Prop 5.4(iii)
    lc = mp.findroot(lambda l: L(mexp(-l)) - 1, 0.48)
    print("lambda_c =", mp.nstr(lc, 8), " -log p0 =", mp.nstr(-mlog(mpf(str(p0))), 6),
          " -log p1 =", mp.nstr(-mlog(mpf(str(roots[1]))), 6))
    # Section 8: projections of scrambled Sobol' sequences, plain LHS null
    for (i, l) in [(0, 1), (2, 4)]:
        un, ad = [], []
        for r in range(200):
            U = D.design_matrix(512, 6, "sobol", seed=31_000 + r)
            un.append(D.ersatz_input(U[:, i], U[:, l], fill=False))
            ad.append(D.ersatz_input(U[:, i], U[:, l]))
        print(f"Sobol' coordinates ({i + 1},{l + 1}), N = 512: unfilled"
              f" {np.mean(un):.4f}, adjusted {np.mean(ad):.5f}")
    for N, R in [(512, 10000), (4096, 4000)]:
        v = np.array([D.ersatz_input((rng.permutation(N) + rng.random(N)) / N,
                                     rng.random(N)) for _ in range(R)])
        print(f"plain LHS null, N = {N}: mean {v.mean():.4f},"
              f" P(> 0.05) = {np.mean(v > 0.05):.3f}")
