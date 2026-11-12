"""screening_check.py -- screening errors of the adjusted ersatz against the
theory of Supplementary Material A (Supplementary Material B, Section B.7).

(a) Limits. For every input of the six benchmark functions the limit
    Psi_lambda(C_i) is estimated under SRS at lambda = 512/529 (N = 61952,
    s = 253; the value of lambda at N = 512) and at lambda = 1 (N = 65536,
    s = 256), with 10 replicates, and compared with the bounds
    [delta_i - eps_lambda, delta_i + kappa_lambda] of SM A, Proposition 5.4.
(b) Inputs independent of the output. The inputs of the Becker ensemble with
    T_i = 0 are independent of the output, so under SRS their statistic has
    the distribution under independence at their sample size (SM A,
    Theorem 4.1, and Proposition 3(a) of the paper). Their flag rates are
    compared with that distribution, simulated at each sample size.
(c) Oscillation with lambda_N. Type I error at theta = 0.05 in the
    convergence study (SRS) against the probability that an input independent
    of the output exceeds 0.05 (results/null_floor.csv).
(d) Ranking guarantee. Pairs of benchmark inputs whose delta values differ by
    more than eps + kappa at lambda = 512/529: share of the replicates at
    N = 512 in which the adjusted ersatz ranks them as delta does.
(e) Design. Mean of the statistic at N = 512 under the Sobol' and LHS designs
    minus its mean under SRS.

Run from src/; it reads the stored results (no rerun of the experiments is
needed), writes results/screening_limits.csv and tables/tab_screening_theory.tex
(SM B, Table S15), and takes about 3 minutes on two cores.
"""
import json
import os

import numpy as np
import pandas as pd

import discrepancy as D
from experiments import FUNCTIONS

THETAS = (0.01, 0.05, 0.10)
LAMS = {"512/529": 512 / 529, "1": 1.0}
pd.set_option("display.width", 200)


def phi(p):
    return p ** 6 * (56 - 140 * p + 120 * p ** 2 - 35 * p ** 3)


def kappa_eps(lam):
    """kappa_lambda = h_lambda(1) and eps_lambda = sup_t (1 - t - h_lambda(t))."""
    t = np.linspace(0.0, 1.0, 1_000_001)
    h = phi(np.exp(-lam * t))
    return phi(np.exp(-lam)), max(0.0, np.max(1 - t - h))


def limits(dref, tref):
    rows = []
    for lam_name, N, s in [("512/529", 61952, 253), ("1", 65536, 256)]:
        ag = np.log(s - 1e-9) / np.log(N)          # forces s cells per axis
        kap, eps = kappa_eps(N / s ** 2)
        for fname, (f, k, _) in FUNCTIONS.items():
            vals = []
            for r in range(10):
                rng = np.random.default_rng(1000 + r)
                X = rng.random((N, k))
                y = f(X)
                vals.append([D.ersatz_input(X[:, i], y, grid_exponent=ag)
                             for i in range(k)])
            vals = np.array(vals)
            for i in range(k):
                rows.append(dict(lam=lam_name, function=fname, input=i + 1,
                                 T=tref[fname][i], delta=dref[fname][i],
                                 psi=vals[:, i].mean(),
                                 se=vals[:, i].std(ddof=1) / np.sqrt(10),
                                 lo=dref[fname][i] - eps,
                                 hi=dref[fname][i] + kap))
    return pd.DataFrame(rows)


def main():
    dref = json.load(open("results/delta_refs.json"))
    tref = json.load(open("refs.json"))

    print("(a) limits of the adjusted ersatz and the bounds of SM A, Prop. 5.4")
    lim = limits(dref, tref)
    lim["inside"] = (lim.psi >= lim.lo) & (lim.psi <= lim.hi)
    for lam, d in lim.groupby("lam"):
        kap, eps = kappa_eps(LAMS[lam])
        dev = d.psi - d.delta
        print(f"  lambda = {lam:7s}: kappa = {kap:.4f}, eps = {eps:.4f}; "
              f"{d.inside.sum()}/{len(d)} limits inside the bounds; "
              f"Psi - delta from {dev.min():.3f} to {dev.max():.3f} "
              f"(largest standard error {d.se.max():.4f})")
    lim.to_csv("results/screening_limits.csv", index=False)

    print("\n(b) Becker inputs with T_i = 0: share above theta, against the "
          "distribution under independence (SRS) at the same N")
    b = pd.read_csv("results/becker_values.csv")
    a = b[(b.method == "ersatz_adj") & (b.ref == 0)]
    rng = np.random.default_rng(7)
    null = {}
    for N in sorted(a.N.unique()):
        v = np.array([D.ersatz_input(rng.random(N), rng.random(N))
                      for _ in range(3000)])
        null[N] = [np.mean(v > t) for t in THETAS] + [v.mean()]
    tab_a = []
    for des in ("srs", "lhs", "sobol"):
        d = a[a.design == des]
        pred = np.array([null[N] for N in d.N]).mean(0)
        obs = [np.mean(d.value > t) for t in THETAS]
        se = [np.sqrt(o * (1 - o) / len(d)) for o in obs]
        print(f"  {des:5s} ({len(d)} inputs): observed "
              + ", ".join(f"{o:.3f} ({s:.3f})" for o, s in zip(obs, se))
              + f", mean {d.value.mean():.4f};  independence (SRS): "
              + ", ".join(f"{p:.3f}" for p in pred[:3])
              + f", mean {pred[3]:.4f}")
        tab_a.append(obs + [d.value.mean()])
    tab_a.append(list(pred))       # same N in the three designs

    print("\n(c) type I error at theta = 0.05 under SRS, convergence study, "
          "against P(statistic > 0.05) for an independent input")
    c = pd.read_csv("results/convergence.csv")
    c = c[(c.method == "ersatz_adj") & (c.design == "srs")]
    nf = pd.read_csv("results/null_floor.csv")
    nf = nf[nf.design == "srs"].set_index("N")
    tab = c.groupby(["N", "function"])["alpha_0.05"].mean().unstack()
    tab = tab.dropna(axis=1, how="all")
    tab.insert(0, "lambda_N", [N / int(np.ceil(np.sqrt(N) - 1e-12)) ** 2
                               for N in tab.index])
    tab["independent"] = [nf.loc[N, "p_gt_0.05"] if N in nf.index else np.nan
                          for N in tab.index]
    print(tab.round(3).to_string())
    write_table(tab_a, tab)

    v = pd.read_csv("results/bench_values.csv")
    v = v[v.method == "ersatz_adj"]
    kap, eps = kappa_eps(512 / 529)
    print(f"\n(d) pairs with delta_i - delta_j > eps + kappa = {eps + kap:.4f}"
          " (lambda = 512/529): share of replicates ranked as by delta")
    for des in ("srs", "lhs", "sobol"):
        agree = tot = pairs = 0
        for fn, d in v[v.design == des].groupby("function"):
            dl = dref[fn]
            piv = d.pivot(index="rep", columns="input", values="value")
            for i in range(len(dl)):
                for j in range(len(dl)):
                    if dl[i] - dl[j] > eps + kap:
                        pairs += 1
                        agree += int((piv[i] > piv[j]).sum())
                        tot += len(piv)
        print(f"  {des:5s}: {pairs} pairs, {agree}/{tot} = {agree / tot:.4f}")

    print("\n(e) mean of the statistic at N = 512 minus its mean under SRS "
          "(min, median, max over the inputs of each function)")
    m = v.groupby(["function", "input", "design"]).value.mean().unstack()
    out = pd.DataFrame({"sobol": m.sobol - m.srs, "lhs": m.lhs - m.srs})
    print(out.groupby("function").agg(["min", "median", "max"]).round(3)
          .to_string())


def write_table(tab_a, tab):
    """LaTeX body of SM B, Table S15."""
    lab = ["SRS", "Maximin LHS", "Scrambled Sobol'",
           "Independent input, SRS, same $N$"]
    lines = [r"\begin{tabular}{@{}lcccc@{}}\toprule",
             r"\multicolumn{5}{@{}l}{(a) Becker inputs with $T_i=0$: share "
             r"above $\theta$, and mean} \\",
             r"Design & $\theta=0.01$ & $\theta=0.05$ & $\theta=0.10$ & Mean"
             r" \\ \midrule"]
    for name, row in zip(lab, tab_a):
        if name.startswith("Independent"):
            lines.append(r"\addlinespace[2pt]")
        lines.append(f"{name} & " + " & ".join(f"{x:.3f}" for x in row[:3])
                     + f" & {row[3]:.4f} \\\\")
    lines += [r"\bottomrule\end{tabular}", "", r"\vspace{8pt}", "",
              r"\begin{tabular}{@{}rccccc@{}}\toprule",
              r"\multicolumn{6}{@{}l}{(b) Type~I error at $\theta=0.05$, SRS,"
              r" convergence study} \\",
              r"$N$ & $N/s^2$ & Bratley et al. (1992) & Oakley--O'Hagan & "
              r"Sobol' $G$ & Independent input \\ \midrule"]
    for N, r in tab.iterrows():
        if N < 32:
            continue
        ind = "--" if np.isnan(r["independent"]) else f"{r['independent']:.2f}"
        lines.append(f"{N} & {r['lambda_N']:.3f} & {r['bratley1992']:.2f} & "
                     f"{r['oakley']:.2f} & {r['sobol_g']:.2f} & {ind} \\\\")
    lines.append(r"\bottomrule\end{tabular}")
    os.makedirs("tables", exist_ok=True)
    with open("tables/tab_screening_theory.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
