"""Gap-mechanism regression rerun on the tuned
grid, with the circularity checks carried over.
The gap values differ from script 14, which used
fixed regularisation, so this supersedes it.
Run in venv (analysis).
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC = f.PROC
GRID = os.path.join(PROC, "grid_tuned_std.csv")
OLD = os.path.join(PROC, "grid_quantile.csv")

FAM = {"early": ["early:plain",
                 "early:blockstd"],
       "inter": ["inter:" + v
                 for v in f.INTER_VARIANTS],
       "late": ["late:mean", "late:wsrc"]}


def build(g):
    rows = []
    keys = g[["direction",
              "outcome"]].drop_duplicates()
    for _, k in keys.iterrows():
        d, oc = k["direction"], k["outcome"]
        s = g[(g["direction"] == d)
              & (g["outcome"] == oc)]
        e = s[s["arch"] == "uni:ehr_only"]
        c = s[s["arch"] == "uni:ctpa_only"]
        if not len(e) or not len(c):
            continue
        ehr = float(e["auc"].iloc[0])
        ctpa = float(c["auc"].iloc[0])
        rec = {"direction": d, "outcome": oc,
               "ehr": ehr, "ctpa": ctpa,
               "gap": ehr - ctpa,
               "ratio": ctpa / ehr,
               "ev": int(s["ev"].iloc[0]),
               "cross": int(d in ("I2M", "M2I"))}
        for fam, names in FAM.items():
            sub = s[s["arch"].isin(names)]
            if len(sub):
                rec[fam + "_best"] = float(
                    sub["gain"].max())
        sub = s[~s["arch"].isin(
            ["uni:ehr_only", "uni:ctpa_only"])]
        rec["any_best"] = float(sub["gain"].max())
        rec["best_arch"] = sub.loc[
            sub["gain"].idxmax(), "arch"]
        rows.append(rec)
    t = pd.DataFrame(rows)
    t["late_auc"] = t["ehr"] + t["late_best"]
    return t.sort_values("gap")


def fit(x, y, lab, out):
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 4:
        print("  %-20s too few" % lab)
        return
    sl, ic, r, p, se = stats.linregress(x, y)
    rho, prho = stats.spearmanr(x, y)
    print("  %-20s slope %+.4f (se %.4f)"
          "  r=%+.3f  R2=%.3f  p=%.4f"
          "  rho=%+.3f  n=%d"
          % (lab, sl, se, r, r ** 2, p,
             rho, len(x)))
    out.append({"set": lab, "slope": sl,
                "se": se, "intercept": ic,
                "r": r, "r2": r ** 2, "p": p,
                "spearman": rho, "n": len(x)})


def partial(x, y, z, lab, out):
    m = ~(np.isnan(x) | np.isnan(y)
          | np.isnan(z))
    x, y, z = x[m], y[m], z[m]
    n = len(x)
    if n < 5:
        print("  %-20s too few" % lab)
        return
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    r, _ = stats.pearsonr(rx, ry)
    df = n - 3
    tt = r * np.sqrt(df / max(1e-12,
                              1.0 - r ** 2))
    p = 2 * (1 - stats.t.cdf(abs(tt), df))
    print("  %-20s partial r=%+.3f  p=%.4f"
          "  n=%d df=%d" % (lab, r, p, n, df))
    out.append({"set": lab, "slope": np.nan,
                "se": np.nan,
                "intercept": np.nan, "r": r,
                "r2": r ** 2, "p": p,
                "spearman": np.nan, "n": n})


t = build(pd.read_csv(GRID))
t.to_csv(os.path.join(PROC,
                      "gap_table_tuned.csv"),
         index=False)

print("GAP vs GAIN (tuned C), sorted by gap")
cols = ["direction", "outcome", "ehr", "ctpa",
        "gap", "ev", "late_best", "early_best",
        "inter_best", "any_best", "best_arch"]
print(t[cols].round(4).to_string(index=False))

res = []
print("")
print("REGRESSION OF GAIN ON GAP (12 cells)")
for lab in ["late_best", "early_best",
            "inter_best", "any_best"]:
    fit(t["gap"].values, t[lab].values,
        lab, res)

print("")
print("EXCLUDING cv_first")
t2 = t[t["outcome"] != "cv_first"]
for lab in ["late_best", "early_best",
            "any_best"]:
    fit(t2["gap"].values, t2[lab].values,
        lab + "_nocv", res)

print("")
print("CROSS-DATASET CELLS ONLY")
t3 = t[t["cross"] == 1]
for lab in ["late_best", "any_best"]:
    fit(t3["gap"].values, t3[lab].values,
        lab + "_cross", res)

print("")
print("CIRCULARITY CHECKS")
fit(t["ehr"].values, t["late_best"].values,
    "gain_vs_ehr", res)
fit(t["ctpa"].values, t["late_best"].values,
    "gain_vs_ctpa", res)
partial(t["gap"].values, t["late_best"].values,
        t["ehr"].values, "partial_gap_ehr", res)
fit(t["gap"].values, t["late_auc"].values,
    "fusedauc_vs_gap", res)
fit(t["ctpa"].values, t["late_auc"].values,
    "fusedauc_vs_ctpa", res)
fit(t["ratio"].values, t["late_best"].values,
    "gain_vs_ratio", res)

print("")
print("FITTED LINE, PREDICTED GAIN BY GAP")
o = [x for x in res if x["set"] == "late_best"]
if o:
    o = o[0]
    for gp in (0.02, 0.04, 0.06, 0.08, 0.10,
               0.12, 0.16, 0.20):
        print("  gap %.2f -> %+.4f"
              % (gp, o["intercept"]
                 + o["slope"] * gp))
    if o["slope"] < 0:
        print("  crosses zero at gap %.3f"
              % (-o["intercept"] / o["slope"]))

if os.path.exists(OLD):
    p = build(pd.read_csv(OLD))
    m = t[["direction", "outcome", "gap",
           "late_best"]].merge(
        p[["direction", "outcome", "gap",
           "late_best"]],
        on=["direction", "outcome"],
        suffixes=("_tuned", "_fixed"))
    print("")
    print("TUNED VS FIXED C")
    print(m.round(4).to_string(index=False))
    print("  gap correlation: %.3f"
          % float(np.corrcoef(
              m["gap_tuned"],
              m["gap_fixed"])[0, 1]))

pd.DataFrame(res).to_csv(
    os.path.join(PROC,
                 "gap_regression_tuned.csv"),
    index=False)
print("")
print("saved gap_table_tuned.csv,"
      " gap_regression_tuned.csv")