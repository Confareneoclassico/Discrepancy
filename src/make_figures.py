"""make_figures.py -- all figures of the revised manuscript (PDF)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import binom

import discrepancy as D
from summaries import REF_T, REF_D, R

FIG = "figs"
os.makedirs(FIG, exist_ok=True)
# Figures are about 10 inches wide and are printed at the text width of the
# manuscript (5.4 inches), so the 13-16 pt fonts below print at 7-9 pt.
plt.rcParams.update({"font.size": 15, "axes.titlesize": 16,
                     "axes.labelsize": 16, "xtick.labelsize": 13,
                     "ytick.labelsize": 13, "legend.fontsize": 13,
                     "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})
# reference categorical palette, fixed order (validated: adjacent pairs)
COL = {"ersatz_adj": "#2a78d6", "pce": "#eb6834", "pawn": "#1baf7a",
       "shapley": "#eda100", "ersatz": "#e87ba4"}
MARK = {"ersatz_adj": "o", "pce": "s", "pawn": "^", "shapley": "D",
        "ersatz": "v"}
LS = {"ersatz_adj": "-", "pce": "--", "pawn": "-.", "shapley": ":",
      "ersatz": (0, (5, 1, 1, 1))}
LAB = {"ersatz_adj": "Adjusted ersatz", "pce": "PCE", "pawn": "PAWN",
       "shapley": "Shapley (PCE)", "ersatz": "S-ersatz"}
ORDER = ["ersatz_adj", "pce", "pawn", "shapley", "ersatz"]
INK, INK2 = "#0b0b0b", "#52514e"


# ---------------------------------------------------------------------
# Figures 1-4: the construction of the statistic on the Sobol' G function
# ---------------------------------------------------------------------

def construction_figures(N=512, seed=11):
    U = D.design_matrix(N, 8, "srs", seed=seed)
    Y = D.sobol_g(U)
    V = (np.argsort(np.argsort(Y, kind="stable"), kind="stable")) / N
    s = D.grid_size(N)
    T = REF_T["sobol_g"]

    def grid8(fn, name, w=10, h=5.6, sub=None):
        fig, axes = plt.subplots(2, 4, figsize=(w, h), sharey=True)
        for i, ax in enumerate(axes.ravel()):
            fn(ax, i)
            title = rf"$x_{{{i+1}}}$  ($T_{{{i+1}}}={T[i]:.3f}$)"
            if sub is not None:
                title += "\n" + sub(i)
            ax.set_title(title, fontsize=15)
            if i % 4 == 0:
                ax.set_ylabel(r"$Y$" if name == "fig1" else r"$(R_j-1)/N$")
            if i >= 4:
                ax.set_xlabel(r"$x_i$")
        fig.tight_layout()
        fig.savefig(f"{FIG}/{name}.pdf")
        plt.close(fig)

    nb = 16
    edges = np.linspace(0, 1, nb + 1)
    mids = 0.5 * (edges[1:] + edges[:-1])

    def f1(ax, i):
        x = U[:, i]
        b = np.minimum((x * nb).astype(int), nb - 1)
        mu = np.array([Y[b == j].mean() for j in range(nb)])
        q1 = np.array([np.quantile(Y[b == j], 0.1) for j in range(nb)])
        q9 = np.array([np.quantile(Y[b == j], 0.9) for j in range(nb)])
        ax.fill_between(mids, q1, q9, color="#86b6ef", alpha=0.45, lw=0)
        ax.scatter(x, Y, s=4, color=INK2, alpha=0.55, lw=0)
        ax.plot(mids, mu, "o-", color="#eb6834", ms=4, lw=1.6,
                mec="white", mew=0.6)
        Sh = np.var(mu) / np.var(Y)
        ax.text(0.03, 0.97, rf"$\hat S_{{{i+1}}}\approx{Sh:.2f}$",
                transform=ax.transAxes, va="top", fontsize=13, color=INK)
    grid8(f1, "fig1")

    def f2(ax, i):
        ax.scatter(U[:, i], V, s=4, color=INK2, alpha=0.6, lw=0)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    grid8(f2, "fig2")

    def draw_grid(ax, G, F=None):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        for a in range(s + 1):
            ax.axhline(a / s, color="#d9d8d3", lw=0.5, zorder=0)
            ax.axvline(a / s, color="#d9d8d3", lw=0.5, zorder=0)
        ia, ib = np.nonzero(G)
        ax.scatter((ia + 0.5) / s, (ib + 0.5) / s, s=9, color=INK,
                   marker="s", lw=0)
        if F is not None:
            ja, jb = np.nonzero(F & (G == 0))
            ax.scatter((ja + 0.5) / s, (jb + 0.5) / s, s=16, color="#e34948",
                       marker="x", lw=1.1)

    grids = [D.occupancy_grid(U[:, i], Y, s) for i in range(8)]

    def f3(ax, i):
        draw_grid(ax, grids[i])
    grid8(f3, "fig3", h=6.2,
          sub=lambda i: rf"$\hat S^{{\circ}}_{{{i+1}}}={1-grids[i].mean():.3f}$")

    fills = [D.moore_fill(G) for G in grids]

    def f4(ax, i):
        draw_grid(ax, grids[i], fills[i])
    grid8(f4, "fig4", h=6.2,
          sub=lambda i: rf"$\hat S^{{\mathrm{{adj}}}}_{{{i+1}}}={1-fills[i].mean():.3f}$")


# ---------------------------------------------------------------------
# Figure 5: theory -- limit functions and the independence floor
# ---------------------------------------------------------------------

def h_lambda(t, lam=1.0):
    p = np.exp(-lam * t)
    return p * binom.cdf(3, 8, 1 - p)


def null_floor(Ns=(32, 64, 128, 256, 512, 1024, 2048, 4096), R=300):
    rows = []
    for N in Ns:
        for design in ("srs", "lhs", "sobol"):
            vals = []
            for r in range(R if N <= 1024 else 120):
                U = D.design_matrix(N, 6, design, seed=97_000 * N + r)
                Y = U[:, 1:].sum(1)
                vals.append(D.ersatz_input(U[:, 0], Y))
            v = np.array(vals)
            rows.append(dict(N=N, design=design, mean=v.mean(), sd=v.std(),
                             q05=np.quantile(v, 0.05),
                             q95=np.quantile(v, 0.95),
                             q99=np.quantile(v, 0.99)))
    return pd.DataFrame(rows)


def theory_figure():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    t = np.linspace(0, 2.5, 1000)
    ax = axes[0]
    ax.plot(t, np.clip(1 - t, 0, None), color=INK, lw=2.2,
            label=r"$(1-t)_+$  (Borgonovo's $\delta$)")
    ax.plot(t, np.exp(-t), color="#e87ba4", lw=2.2, ls="--",
            label=r"$e^{-t}$  (rank grid, no fill)")
    ax.plot(t, h_lambda(t), color="#2a78d6", lw=2.6,
            label=r"$h_1(t)$  (adjusted ersatz)")
    ax.axvline(1, color="#c3c2b7", lw=1, zorder=0)
    ax.set_xlabel(r"local copula density $t=c(u,v)$")
    ax.set_ylabel("contribution to the limit")
    ax.set_title("(a) Integrands of the limits")
    ax.legend(frameon=False, loc="upper right")
    ax = axes[1]
    path = f"{R}/null_floor.csv"            # written by null_distribution.py
    if os.path.exists(path):
        nf = pd.read_csv(path)
    else:
        nf = null_floor()
        nf.to_csv(path, index=False)
    lab = {"srs": "SRS", "lhs": "maximin LHS", "sobol": "Sobol'"}
    col = {"srs": "#2a78d6", "lhs": "#eb6834", "sobol": "#1baf7a"}
    mk = {"srs": "o", "lhs": "s", "sobol": "^"}
    for d in ("srs", "lhs", "sobol"):
        g = nf[nf.design == d]
        ax.fill_between(g.N, g.q05, g.q95, color=col[d], alpha=0.15, lw=0)
        ax.plot(g.N, g["mean"], marker=mk[d], color=col[d], lw=2,
                label=f"{lab[d]} (mean, 5\u201395%)")
    for th in (0.01, 0.05, 0.10):
        ax.axhline(th, color=INK2, lw=1, ls=":")
        ax.text(4700, th + 0.002, rf"$\theta={th:.2f}$", va="bottom", fontsize=13,
                color=INK2)
    ax.axhline(h_lambda(1.0), color=INK, lw=1.2, ls="--")
    ax.text(4700, h_lambda(1.0) - 0.002, r"$\kappa_1$", va="top",
            fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"sample size $N$")
    ax.set_ylabel(r"$S^{\mathrm{adj}}_i$ under independence")
    ax.set_title("(b) Value under independence")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(28, 12000)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig5_theory.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 6: convergence (median and inter-quartile band over replicates)
# ---------------------------------------------------------------------

def convergence_figure(design="sobol", name="fig6_convergence"):
    """Small multiples: one row per test case, rho_T (left) and MAE_T
    (right); common axes in each column, so rows can be compared."""
    c = pd.read_csv(f"{R}/convergence.csv")
    c = c[c.design == design]
    h = pd.read_csv(f"{R}/hymod_metrics_TD.csv")
    hd = {"sobol": "sobol_block", "srs": "srs_subset"}[design]
    h = h[h.design == hd]
    cases = [("bratley1988", "Bratley\u2013Fox\n(1988)"),
             ("bratley1992", "Bratley et al.\n(1992)"),
             ("ishigami", "Ishigami"), ("oakley", "Oakley\u2013\nO'Hagan"),
             ("sobol_g", "Sobol' G"), ("multiverse", "Multiverse"),
             ("hymod", "HYMOD")]
    fig, axes = plt.subplots(len(cases), 2, figsize=(9.8, 11.6), sharex=True)
    for r, (fn, title) in enumerate(cases):
        d = h if fn == "hymod" else c[c.function == fn]
        for j, met in enumerate(("rho", "mae")):
            ax = axes[r, j]
            ax.set_xscale("log", base=2)
            if met == "rho" and fn == "bratley1988":
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.tick_params(left=False, labelleft=False, bottom=False,
                               labelbottom=False)
                ax.text(0.5, 0.5, "$\\rho_T$ undefined: all eight $T_i$\n"
                        "are equal ($0.278$)", transform=ax.transAxes,
                        ha="center", va="center", fontsize=13, color=INK2)
            else:
                for m in ORDER:
                    g = d[d.method == m].groupby("N")[met]
                    med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
                    if med.isna().all():
                        continue
                    ax.fill_between(med.index, q1, q3, color=COL[m],
                                    alpha=0.13, lw=0)
                    ax.plot(med.index, med.values, ls=LS[m], marker=MARK[m],
                            color=COL[m], lw=1.6, ms=4.5, label=LAB[m])
                ax.axvline(512, color="#c3c2b7", lw=1, zorder=0)
                if met == "rho":
                    # full range of a correlation, identical on every row
                    ax.set_ylim(-1.05, 1.05)
                    ax.set_yticks([-1.0, 0.0, 1.0])
                    ax.axhline(0.0, color="#c9c8c4", lw=0.8, zorder=0)
                else:
                    ax.set_ylim(0.0, 0.7)
                    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
            if r == len(cases) - 1:
                ax.set_xlabel(r"sample size $N$")
        axes[r, 0].set_ylabel(title, rotation=0, ha="right", va="center",
                              labelpad=12)
    axes[0, 0].set_title(r"$\rho_T$")
    axes[0, 1].set_title(r"MAE$_T$")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.6)
    fig.savefig(f"{FIG}/{name}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 7: estimates against the two references (what each targets)
# ---------------------------------------------------------------------

def target_figure():
    v = pd.read_csv(f"{R}/bench_values.csv")
    v = v[v.design == "sobol"]
    hv = pd.read_csv(f"{R}/hymod_values.csv")
    hv = hv[(hv.design == "sobol_block") & (hv.N == 512)]
    hv["function"] = "hymod"
    v = pd.concat([v, hv])
    med = v.groupby(["function", "method", "input"]).value.median()\
        .reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.0), sharex=True,
                             sharey=True)
    for ax, m in zip(axes, ["ersatz_adj", "pce", "pawn"]):
        d = med[med.method == m]
        T = np.concatenate([REF_T[f][d[d.function == f].input.values]
                            for f in d.function.unique()])
        De = np.concatenate([REF_D[f][d[d.function == f].input.values]
                             for f in d.function.unique()])
        val = np.concatenate([d[d.function == f].value.values
                              for f in d.function.unique()])
        ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1.2, zorder=0)
        ax.scatter(T, val, s=26, marker="o", facecolor="white",
                   edgecolor=COL[m], lw=1.2, label=r"vs $T_i$")
        ax.scatter(De, val, s=26, marker="^", color=COL[m], lw=0,
                   label=r"vs $\delta_i$")
        mt, md = np.mean(np.abs(val - T)), np.mean(np.abs(val - De))
        ax.text(0.03, 0.97, f"MAE vs $T$: {mt:.3f}\nMAE vs $\\delta$: {md:.3f}",
                transform=ax.transAxes, va="top", fontsize=13)
        ax.set_title(LAB[m])
        ax.legend(frameon=False, loc="lower right")
    axes[0].set_ylabel("estimate (median, $N=512$)")
    fig.supxlabel(r"reference value ($T_i$ or $\delta_i$)", fontsize=16)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig7_targets.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 8: Becker ensemble, three designs
# ---------------------------------------------------------------------

def becker_figure():
    m = pd.read_csv(f"{R}/becker_metrics_T.csv")
    dcol = {"sobol": "#2a78d6", "lhs": "#eb6834", "srs": "#1baf7a"}
    dlab = {"sobol": "Sobol'", "lhs": "maximin LHS", "srs": "SRS"}
    mets = [("rho", r"(a) $\rho_T$"), ("mae", r"(b) MAE$_T$"),
            ("alpha_0.05", r"(c) $\alpha(0.05)$"),
            ("beta_0.05", r"(d) $\beta(0.05)$")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
    axes = axes.ravel()
    for ax, (met, lab) in zip(axes, mets):
        for k, des in enumerate(("sobol", "lhs", "srs")):
            data = [m[(m.method == mm) & (m.design == des)][met].dropna()
                    for mm in ORDER]
            pos = np.arange(len(ORDER)) * 4 + k
            bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True,
                            showfliers=False, medianprops=dict(color=INK))
            for b in bp["boxes"]:
                b.set_facecolor(dcol[des]); b.set_alpha(0.75)
            if met.startswith(("alpha", "beta")):
                ax.scatter(pos, [x.mean() for x in data], marker="D", s=16,
                           color=INK, zorder=3)
        ax.set_xticks(np.arange(len(ORDER)) * 4 + 1)
        ax.set_xticklabels(["Adj.\nersatz", "PCE", "PAWN", "Shapley",
                            "S-ersatz"])
        ax.set_title(lab)
        # same vertical scale on all four panels, no truncation
        ax.set_ylim(-0.1, 1.05)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.axhline(0.0, color="#c9c8c4", lw=0.8, zorder=0)
    axes[0].legend([Patch(color=dcol[d], alpha=0.75) for d in dcol],
                   [dlab[d] for d in dcol], frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig8_becker.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 9: sensitivity of the sensitivity analysis (applicable cells only)
# ---------------------------------------------------------------------

def sos_figure():
    """First-order and total-order indices of the algorithmic parameters,
    one row of panels per method and one panel per metric. Combinations
    in which a parameter does not enter a metric are hatched (n/a)."""
    s = pd.read_csv(f"{R}/sos_indices.csv").set_index(["output", "factor"])
    facs = ["grid_exponent", "fill_threshold", "pawn_bins",
            "screen_threshold", "design"]
    flab = {"grid_exponent": r"grid exponent $\alpha_g$",
            "fill_threshold": r"fill threshold $\theta_f$",
            "pawn_bins": "PAWN intervals $n$",
            "screen_threshold": r"screening threshold $\theta$",
            "design": "sampling design"}
    rows = [("Adjusted ersatz", "adj", {"pawn_bins"}),
            ("PAWN", "pawn", {"grid_exponent", "fill_threshold"})]
    mets = [("rho", r"$\rho_T$"), ("mae", r"MAE$_T$"),
            ("alpha", r"$\alpha$"), ("beta", r"$\beta$")]
    # ordinal pair of one hue (validated): first order dark, total light
    cS, cT = "#1c5cab", "#86b6ef"
    xmin, xmax = -0.1, 1.15
    y = np.arange(len(facs))[::-1].astype(float)
    fig = plt.figure(figsize=(10, 6.8))
    subs = fig.subfigures(2, 1, hspace=0.0)
    for sub, (title, key, na) in zip(subs, rows):
        axes = sub.subplots(1, 4, sharey=True)
        sub.subplots_adjust(left=0.25, right=0.99, top=0.78, bottom=0.12,
                            wspace=0.10)
        sub.suptitle(title, fontsize=16, x=0.01, y=0.98, ha="left",
                     fontweight="bold")
        for ax, (met, mlab) in zip(axes, mets):
            out = f"{met}_{key}"
            ax.axvline(0, color="#c9c8c4", lw=0.8, zorder=0)
            ax.axvline(1, color="#c9c8c4", lw=0.8, ls=":", zorder=0)
            for yi, f in zip(y, facs):
                inactive = (f in na or (f == "screen_threshold"
                                        and met in ("rho", "mae")))
                novar = out == "beta_pawn" and not inactive
                if inactive or novar:
                    ax.add_patch(plt.Rectangle(
                        (xmin, yi - 0.42), xmax - xmin, 0.84, fc="#ecebe7",
                        ec="white", hatch="//", lw=0, zorder=0))
                    ax.text(0.5 * (xmin + xmax), yi,
                            "no var." if novar else "n/a",
                            ha="center", va="center", color=INK2,
                            fontsize=13)
                    continue
                r = s.loc[(out, f)]
                for dy, v, lo, hi, col in (
                        (0.19, r.S, r.S_lo, r.S_hi, cS),
                        (-0.19, r["T"], r.lo, r.hi, cT)):
                    ax.barh(yi + dy, v, height=0.36, color=col, lw=0,
                            zorder=2)
                    ax.plot([max(lo, xmin), min(hi, xmax)], [yi + dy] * 2,
                            color=INK, lw=1.0, zorder=3, solid_capstyle="butt")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(-0.6, len(facs) - 0.4)
            ax.set_xticks([0, 0.5, 1])
            ax.set_title(mlab)
            ax.tick_params(axis="y", length=0)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels([flab[f] for f in facs])
    handles = [Patch(color=cS), Patch(color=cT)]
    subs[0].legend(handles, [r"first order $S_i$", r"total order $T_i$"],
                   loc="upper right", ncol=2, frameon=False,
                   bbox_to_anchor=(0.995, 1.01))
    fig.savefig(f"{FIG}/fig9_sos.pdf", bbox_inches="tight")
    plt.close(fig)


def sos_convergence_figure(n_min=2 ** 6):
    """Convergence of the indices of the sensitivity analysis of the
    method: S_i (dashed) and T_i (solid) with 95% bootstrap bands on the
    nested sub-designs of base size 2^6, 2^7, ..., N (the first n rows of
    every block of a scrambled Sobol' design form the design of size n)."""
    import experiments as E
    Ys = pd.read_csv(f"{R}/sos_outputs.csv")[E.SOS_OUT].values
    YB = pd.read_csv(f"{R}/sos_outputs_B.csv")[E.SOS_OUT].values
    k, N = len(E.SOS_FACTORS), len(YB)
    blocks = Ys.reshape(k + 1, N, len(E.SOS_OUT))
    Ns = [2 ** j for j in range(int(np.log2(n_min)), int(np.log2(N)) + 1)]
    rows = []
    for n in Ns:
        idx = E.sos_indices(blocks[:, :n].reshape(-1, len(E.SOS_OUT)), YB[:n], n)
        idx["N"] = n
        rows.append(idx)
    c = pd.concat(rows)
    c.to_csv(f"{R}/sos_convergence.csv", index=False)
    # convergence statistics of Sarrazin et al. (2016), SM B, Section B.6
    E.sos_convergence_stats(Ys, YB, Ns).to_csv(
        f"{R}/sos_convergence_stats.csv", index=False)
    fcol = {"grid_exponent": "#2a78d6", "fill_threshold": "#eb6834",
            "screen_threshold": "#1baf7a", "pawn_bins": "#6250d6",
            "design": "#e87ba4"}
    fmk = {"grid_exponent": "o", "fill_threshold": "s",
           "screen_threshold": "^", "pawn_bins": "D", "design": "v"}
    flab = {"grid_exponent": r"grid exponent $\alpha_g$",
            "fill_threshold": r"fill threshold $\theta_f$",
            "screen_threshold": r"screening threshold $\theta$",
            "pawn_bins": "PAWN intervals $n$", "design": "sampling design"}
    mets = [("rho", r"$\rho_T$"), ("mae", r"MAE$_T$"), ("alpha", r"$\alpha$"),
            ("beta", r"$\beta$")]
    fig = plt.figure(figsize=(10, 6.6))
    subs = fig.subfigures(2, 1, hspace=0.0)
    for sub, (title, key) in zip(subs, [("Adjusted ersatz", "adj"),
                                        ("PAWN", "pawn")]):
        axes = sub.subplots(1, 4, sharey=True)
        sub.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.17,
                            wspace=0.10)
        sub.suptitle(title, fontsize=16, x=0.01, y=0.98, ha="left",
                     fontweight="bold")
        for ax, (met, mlab) in zip(axes, mets):
            out = f"{met}_{key}"
            if out == "beta_pawn":
                ax.axis("off")
                continue
            d = c[c.output == out]
            last = d[d.N == N].set_index("factor")
            for f in E.SOS_FACTORS:
                if last.loc[f, "T"] < 0.03:          # inactive or negligible
                    continue
                g = d[d.factor == f].sort_values("N")
                for col, lo, hi, ls in (("T", "lo", "hi", "-"),
                                        ("S", "S_lo", "S_hi", "--")):
                    ax.fill_between(g.N, g[lo].clip(-0.1, 1.15),
                                    g[hi].clip(-0.1, 1.15), color=fcol[f],
                                    alpha=0.12, lw=0)
                    ax.plot(g.N, g[col], ls=ls, color=fcol[f], lw=1.6,
                            marker=fmk[f], ms=4.5,
                            mfc=fcol[f] if col == "T" else "white")
            ax.set_xscale("log", base=2)
            ax.set_xticks([2 ** 6, 2 ** 8, 2 ** 10, 2 ** 12][:1 + (len(Ns) - 1) // 2])
            ax.set_ylim(-0.1, 1.15)
            ax.axhline(0, color="#c9c8c4", lw=0.8, zorder=0)
            ax.set_title(mlab)
            ax.set_xlabel("base size $N$")
    # legend in the free panel
    from matplotlib.lines import Line2D
    lax = subs[1].axes[-1]
    hs = [Line2D([], [], color=fcol[f], marker=fmk[f], lw=1.6, ms=5)
          for f in fcol]
    hs += [Line2D([], [], color=INK2, lw=1.6, ls="-"),
           Line2D([], [], color=INK2, lw=1.6, ls="--", marker="o", mfc="white")]
    lax.legend(hs, [flab[f] for f in fcol] + [r"total order $T_i$",
                                               r"first order $S_i$"],
               loc="center", frameon=False, fontsize=12)
    fig.savefig(f"{FIG}/figS_sos_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    return c


if __name__ == "__main__":
    construction_figures()
    theory_figure()
    convergence_figure("sobol", "fig6_convergence")
    convergence_figure("srs", "figS_convergence_srs")
    target_figure()
    becker_figure()
    sos_figure()
    sos_convergence_figure()
    print("figures written to", FIG)
