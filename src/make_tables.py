"""make_tables.py -- LaTeX tables for the revised manuscript and SM."""
import numpy as np
import pandas as pd

from summaries import (METHODS, FNAME, REF_T, REF_D, R, TAB)

bm = pd.read_csv(f"{R}/bench_metrics_TD.csv")
hm = pd.read_csv(f"{R}/hymod_metrics_TD.csv")
km = pd.read_csv(f"{R}/becker_metrics_T.csv")
FUNCS = ["bratley1988", "bratley1992", "ishigami", "oakley", "sobol_g",
         "multiverse"]
DESIGN_LABEL = {"sobol": "Sobol'", "srs": "SRS", "lhs": "maximin LHS",
                "sobol_block": "Sobol'", "srs_subset": "SRS"}


def fmt(x, nd):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def block(df, metric, nd, best="max"):
    """one row: medians per method, best value in bold."""
    vals = [df[df.method == m][metric].median() for m in METHODS]
    arr = np.array([np.nan if v is None else v for v in vals], float)
    if np.all(np.isnan(arr)):
        return ["--"] * len(METHODS)
    target = np.nanmax(arr) if best == "max" else np.nanmin(arr)
    out = []
    for v in arr:
        s = fmt(v, nd)
        if not np.isnan(v) and abs(v - target) < 0.5 * 10 ** (-nd):
            s = r"\textbf{" + s + "}"
        out.append(s)
    return out


def subset(name, design):
    if name in FUNCS:
        return bm[(bm.function == name) & (bm.design == design)]
    if name == "hymod":
        d = {"sobol": "sobol_block", "srs": "srs_subset"}.get(design)
        return hm[(hm.design == d) & (hm.N == 512)] if d else None
    if name == "becker":
        return km[km.design == design]


def agreement_table(design="sobol", fname="tab_agreement"):
    head = (r"\begin{tabular}{@{}llccccc@{}}\toprule" "\n"
            r"Test case & Metric & S-ersatz & Adj.\ ersatz & PCE & Shapley "
            r"& PAWN \\ \midrule" "\n")
    lines = []
    for name in FUNCS + ["hymod", "becker"]:
        df = subset(name, design)
        if df is None or len(df) == 0:
            continue
        rows = [(r"$\rho_T$", "rho", 2, "max"), (r"MAE$_T$", "mae", 3, "min")]
        if name != "becker":
            rows += [(r"$\rho_\delta$", "rho_delta", 2, "max"),
                     (r"MAE$_\delta$", "mae_delta", 3, "min")]
        for j, (lab, met, nd, b) in enumerate(rows):
            first = FNAME[name] if j == 0 else ""
            lines.append(f"{first} & {lab} & " + " & ".join(block(df, met, nd, b))
                         + r" \\")
        lines.append(r"\addlinespace[2pt]")
    tex = head + "\n".join(lines[:-1]) + "\n" + r"\bottomrule\end{tabular}"
    open(f"{TAB}/{fname}_{design}.tex", "w").write(tex)
    return tex


def screening_table(design="sobol", fname="tab_screening"):
    head = (r"\begin{tabular}{@{}llccccc@{}}\toprule" "\n"
            r"Test case & $\theta$ & S-ersatz & Adj.\ ersatz & PCE & Shapley "
            r"& PAWN \\ \midrule" "\n")
    lines = []
    for name in FUNCS + ["hymod", "becker"]:
        df = subset(name, design)
        if df is None or len(df) == 0:
            continue
        for j, t in enumerate((0.01, 0.05, 0.10)):
            cells = []
            for m in METHODS:
                d = df[df.method == m]
                agg = np.nanmean if name == "becker" else np.nanmedian
                a = agg(d[f"alpha_{t:.2f}"]) if d[f"alpha_{t:.2f}"].notna().any() else np.nan
                b = agg(d[f"beta_{t:.2f}"]) if d[f"beta_{t:.2f}"].notna().any() else np.nan
                cells.append(f"{fmt(a, 2)}/{fmt(b, 2)}")
            first = FNAME[name] if j == 0 else ""
            lines.append(f"{first} & {t:.2f} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[2pt]")
    tex = head + "\n".join(lines[:-1]) + "\n" + r"\bottomrule\end{tabular}"
    open(f"{TAB}/{fname}_{design}.tex", "w").write(tex)
    return tex


def per_input_table(name, design="sobol"):
    """SM: per-input medians (and IQR) of every estimator, with T and delta."""
    v = pd.read_csv(f"{R}/bench_values.csv") if name in FUNCS else None
    if name == "hymod":
        v = pd.read_csv(f"{R}/hymod_values.csv")
        v = v[v.design == "sobol_prefix"]
    else:
        v = v[(v.function == name) & (v.design == design)]
    k = v.input.max() + 1
    names = ([f"$x_{{{i+1}}}$" for i in range(k)] if name != "multiverse" else
             [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\xi$", r"$\zeta$"])
    if name == "hymod":
        names = ["Cmax", "bexp", "alpha", "Rs", "Rq"]
    head = (r"\begin{tabular}{@{}lccccccc@{}}\toprule" "\n"
            r"Input & $T_i$ & $\delta_i$ & S-ersatz & Adj.\ ersatz & PCE & "
            r"Shapley & PAWN \\ \midrule" "\n")
    lines = []
    for i in range(k):
        cells = [fmt(REF_T[name][i], 3), fmt(REF_D[name][i], 3)]
        for m in METHODS:
            x = v[(v.method == m) & (v.input == i)].value
            if name == "hymod":
                cells.append(fmt(x.iloc[0], 3))
            else:
                q1, q3 = x.quantile(0.25), x.quantile(0.75)
                cells.append(f"{x.median():.3f}" + r"\,{\scriptsize[" +
                             f"{q1:.2f},{q3:.2f}" + "]}")
        lines.append(names[i] + " & " + " & ".join(cells) + r" \\")
    tex = head + "\n".join(lines) + "\n" + r"\bottomrule\end{tabular}"
    open(f"{TAB}/sm_inputs_{name}.tex", "w").write(tex)
    return tex


def hymod_full_table():
    d = hm[hm.design == "sobol_prefix"]
    head = (r"\begin{tabular}{@{}lccccc@{}}\toprule" "\n"
            r"Metric & S-ersatz & Adj.\ ersatz & PCE & Shapley & PAWN \\ "
            r"\midrule" "\n")
    rows = [(r"$\rho_T$", "rho", 2, "max"), (r"MAE$_T$", "mae", 3, "min"),
            (r"$\rho_\delta$", "rho_delta", 2, "max"),
            (r"MAE$_\delta$", "mae_delta", 3, "min")]
    lines = []
    for lab, met, nd, b in rows:
        lines.append(lab + " & " + " & ".join(block(d, met, nd, b)) + r" \\")
    for t in (0.01, 0.05, 0.10):
        cells = []
        for m in METHODS:
            x = d[d.method == m]
            cells.append(f"{fmt(x[f'alpha_{t:.2f}'].iloc[0], 2)}/"
                         f"{fmt(x[f'beta_{t:.2f}'].iloc[0], 2)}")
        lines.append(rf"$\alpha/\beta$ ({t:.2f}) & " + " & ".join(cells) + r" \\")
    tex = head + "\n".join(lines) + "\n" + r"\bottomrule\end{tabular}"
    open(f"{TAB}/tab_hymod_full.tex", "w").write(tex)
    return tex


def becker_by_design():
    head = (r"\begin{tabular}{@{}llccccc@{}}\toprule" "\n"
            r"Design & Metric & S-ersatz & Adj.\ ersatz & PCE & Shapley & "
            r"PAWN \\ \midrule" "\n")
    lines = []
    for des in ("sobol", "lhs", "srs"):
        d = km[km.design == des]
        for j, (lab, met, nd, b) in enumerate([(r"$\rho_T$", "rho", 2, "max"),
                                               (r"MAE$_T$", "mae", 3, "min")]):
            lab0 = DESIGN_LABEL[des]
            first = lab0[0].upper() + lab0[1:] if j == 0 else ""
            lines.append(f"{first} & {lab} & " +
                         " & ".join(block(d, met, nd, b)) + r" \\")
        for t in (0.05,):
            cells = []
            for m in METHODS:
                x = d[d.method == m]
                cells.append(f"{x[f'alpha_{t:.2f}'].mean():.2f}/"
                             f"{x[f'beta_{t:.2f}'].mean():.2f}")
            lines.append(r" & $\bar\alpha/\bar\beta$ (0.05) & " +
                         " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[2pt]")
    tex = head + "\n".join(lines[:-1]) + "\n" + r"\bottomrule\end{tabular}"
    open(f"{TAB}/tab_becker.tex", "w").write(tex)
    return tex


def sos_table():
    """Table S14: first-order and total-order indices of the algorithmic
    parameters with 95% bootstrap intervals (results/sos_indices.csv)."""
    s = pd.read_csv(f"{R}/sos_indices.csv").set_index(["output", "factor"])
    facs = [("grid_exponent", r"Grid exponent $\alpha_g$"),
            ("fill_threshold", r"Fill threshold $\theta_f$"),
            ("pawn_bins", r"PAWN intervals $n$"),
            ("screen_threshold", r"Screening threshold $\theta$"),
            ("design", "Sampling design")]
    mets = ["rho", "mae", "alpha", "beta"]
    blocks = [("Adjusted ersatz", "adj", {"pawn_bins"}),
              ("PAWN", "pawn", {"grid_exponent", "fill_threshold"})]

    def cell(out, f, idx, inactive):
        if inactive:
            return "n/a"
        if out == "beta_pawn":
            return "--"
        r = s.loc[(out, f)]
        v, lo, hi = ((r.S, r.S_lo, r.S_hi) if idx == "S"
                     else (r["T"], r.lo, r.hi))
        return f"{num(v)} {{\\scriptsize[{num(lo)}, {num(hi)}]}}"

    def num(x):
        t = f"{x:.2f}"
        t = "0.00" if t == "-0.00" else t
        return "$-$" + t[1:] if t.startswith("-") else t

    lines = [r"\begin{tabular}{@{}llcccc@{}}\toprule",
             r"Parameter & Index & $\rho_T$ & MAE$_T$ & $\alpha$ & $\beta$\\"]
    for b, (title, key, na) in enumerate(blocks):
        lines.append(r"\midrule" if b == 0 else r"\midrule")
        lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{{title}}}\\"
                     r"\addlinespace[2pt]")
        for f, flab in facs:
            ina = [f in na or (f == "screen_threshold" and m in ("rho", "mae"))
                   for m in mets]
            if all(ina):
                lines.append(f"{flab} & & n/a & n/a & n/a & n/a\\\\")
                continue
            for idx in ("S", "T"):
                cells = [cell(f"{m}_{key}", f, idx, a)
                         for m, a in zip(mets, ina)]
                head = flab if idx == "S" else ""
                lines.append(f"{head} & ${idx}_i$ & " + " & ".join(cells)
                             + "\\\\")
    lines.append(r"\bottomrule\end{tabular}")
    with open(f"{TAB}/sm_sos.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    for des in ("sobol", "srs", "lhs"):
        agreement_table(des)
        screening_table(des)
    for n in FUNCS + ["hymod"]:
        per_input_table(n)
    hymod_full_table()
    becker_by_design()
    sos_table()
    print(open(f"{TAB}/tab_agreement_sobol.tex").read())
    print(open(f"{TAB}/tab_screening_sobol.tex").read())
    print(open(f"{TAB}/tab_becker.tex").read())
    print(open(f"{TAB}/tab_hymod_full.tex").read())
