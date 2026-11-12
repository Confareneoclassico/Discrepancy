"""
summaries.py -- recompute agreement metrics from the saved per-input
values against both references (T_i and delta_i) and write the summary
tables used in the manuscript and supplementary material.
"""
import json
import os

import numpy as np
import pandas as pd

import discrepancy as D

R = "results"
TAB = "tables"
os.makedirs(TAB, exist_ok=True)
THETAS = (0.01, 0.05, 0.10)
METHODS = ["ersatz", "ersatz_adj", "pce", "shapley", "pawn"]
LABEL = {"ersatz": "S-ersatz", "ersatz_unfilled": "rank, unfilled",
         "ersatz_adj": "adj. ersatz", "pce": "PCE", "shapley": "Shapley",
         "pawn": "PAWN"}
FNAME = {"bratley1988": "Bratley--Fox (1988)",
         "bratley1992": "Bratley et al. (1992)",
         "ishigami": "Ishigami", "oakley": "Oakley--O'Hagan",
         "sobol_g": "Sobol' $G$", "multiverse": "Multiverse",
         "hymod": "HYMOD", "becker": "Becker ensemble"}

REF_T = {k: np.array(v) for k, v in json.load(open("refs.json")).items()}
REF_D = {k: np.array(v) for k, v in
         json.load(open(f"{R}/delta_refs.json")).items()}
H = json.load(open(f"{R}/hymod_refs.json"))
REF_T["hymod"] = np.array(H["hymod_T_surrogate"])
REF_D["hymod"] = np.array(H["hymod_delta"])
# exact symmetries of two test functions: equal reference values
for ref in (REF_T, REF_D):
    ref["bratley1988"] = np.full(8, ref["bratley1988"].mean())
    ref["multiverse"] = ref["multiverse"].copy()
    ref["multiverse"][:3] = ref["multiverse"][:3].mean()


def metrics_from_values(v: pd.DataFrame, keys, use_delta=True):
    rows = []
    for key, g in v.groupby(keys + ["method"]):
        g = g.sort_values("input")
        est = g.value.values
        fn = g.function.iloc[0]
        refT = g.ref.values if fn == "becker" else REF_T[fn]
        out = dict(zip(keys + ["method"], key))
        out.update(D.summary_metrics(est, refT, THETAS))
        if use_delta and fn in REF_D:
            out["rho_delta"] = D.savage_rho(est, REF_D[fn])
            out["mae_delta"] = float(np.mean(np.abs(est - REF_D[fn])))
        rows.append(out)
    return pd.DataFrame(rows)


def iqr(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return np.nan, np.nan
    return x.quantile(0.25), x.quantile(0.75)


if __name__ == "__main__":
    bv = pd.read_csv(f"{R}/bench_values.csv")
    bm = metrics_from_values(bv, ["function", "design", "rep"])
    hv = pd.read_csv(f"{R}/hymod_values.csv")
    hv["function"] = "hymod"
    hm = metrics_from_values(hv, ["function", "design", "N", "rep"])
    kv = pd.read_csv(f"{R}/becker_values.csv")
    km = metrics_from_values(kv, ["function", "design", "rep"],
                             use_delta=False)
    bm.to_csv(f"{R}/bench_metrics_TD.csv", index=False)
    hm.to_csv(f"{R}/hymod_metrics_TD.csv", index=False)
    km.to_csv(f"{R}/becker_metrics_T.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["rho", "mae", "rho_delta", "mae_delta"]
    print(bm.groupby(["function", "design", "method"])[cols]
          .median().round(3).to_string())
    print(hm[(hm.N == 512) | (hm.design == "sobol_prefix")]
          .groupby(["design", "method"])[cols].median().round(3).to_string())
