"""Gap-mechanism test rerun against the quantile
grid, with circularity checks. Gain contains the
EHR AUC with a minus sign and gap contains it with
a plus sign, so part of the fit is arithmetic
rather than mechanism. The checks separate them.
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
GRID = os.path.join(PROC, "grid_quantile.csv")
OLD = os.path.join(PROC, "grid_summary.csv")

grid = pd.read_csv(GRID)

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
                rec[fam + "_mean"] = float(
                    sub["gain"].mean())
        sub = s[~s["arch"].isin(
            ["uni:ehr_only", "uni:ctpa_only"])]
        rec["any_best"] = float(sub["gain"].max())
        rec["best_auc"] = float(sub["auc"].max())
        rec["best_arch"] = sub.loc[
            sub["gain"].idxmax(), "arch"]
        rows.append(rec)
    t = pd.DataFrame(rows)
    t["late_auc"] = t["ehr"] + t["late_best"]
    return t.sort_values("gap")


t = build(grid)
t.to_csv(os.path.join(PROC,
                      "gap_table_quantile.csv"),
         index=False)

print("GAP vs FUSION GAIN (quantile scaling),"
      " sorted by gap")
cols = ["direction", "outcome", "ehr", "ctpa",
        "gap", "ev", "late_best", "early_best",
        "inter_best", "any_best", "best_arch"]
print(t[cols].round(4).to_string(index=False))


def fit(x, y, lab, out):
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 4:
        print("  %-20s too few points" % lab)
        return
    sl, ic, r, p, se = stats.linregress(x, y)
    rho, prho = stats.spearmanr(x, y)
    print("  %-20s slope %+.4f (se %.4f)"
          "  r=%+.3f  R2=%.3f  p=%.4f"
          "  rho=%+.3f (p=%.4f)  n=%d"
          % (lab, sl, se, r, r ** 2, p,
             rho, prho, len(x)))
    out.append({"set": lab, "slope": sl,
                "se": se, "intercept": ic,
                "r": r, "r2": r ** 2, "p": p,
                "spearman": rho, "sp_p": prho,
                "n": len(x)})


def partial(x, y, z, lab, out):
    """Correlation between x and y holding z
    fixed, computed by residualising both on z."""
    m = ~(np.isnan(x) | np.isnan(y)
          | np.isnan(z))
    x, y, z = x[m], y[m], z[m]
    n = len(x)
    if n < 5:
        print("  %-20s too few points" % lab)
        return
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    r, _ = stats.pearsonr(rx, ry)
    df = n - 3
    tt = r * np.sqrt(df / max(1e-12,
                              1.0 - r ** 2))
    p = 2 * (1 - stats.t.cdf(abs(tt), df))
    rho, prho = stats.spearmanr(rx, ry)
    print("  %-20s partial r=%+.3f  p=%.4f"
          "  rho=%+.3f (p=%.4f)  n=%d df=%d"
          % (lab, r, p, rho, prho, n, df))
    out.append({"set": lab, "slope": np.nan,
                "se": np.nan,
                "intercept": np.nan, "r": r,
                "r2": r ** 2, "p": p,
                "spearman": rho, "sp_p": prho,
                "n": n})


res = []
print("")
print("REGRESSION OF GAIN ON GAP (all 12 cells)")
for lab in ["late_best", "early_best",
            "inter_best", "any_best"]:
    fit(t["gap"].values, t[lab].values,
        lab, res)

print("")
print("EXCLUDING cv_first (CTPA near chance,"
      " high leverage)")
t2 = t[t["outcome"] != "cv_first"]
for lab in ["late_best", "early_best",
            "any_best"]:
    fit(t2["gap"].values, t2[lab].values,
        lab + "_nocv", res)

print("")
print("CROSS-DATASET CELLS ONLY (I2M, M2I)")
t3 = t[t["cross"] == 1]
for lab in ["late_best", "any_best"]:
    fit(t3["gap"].values, t3[lab].values,
        lab + "_cross", res)

print("")
print("=" * 62)
print("CIRCULARITY CHECKS")
print("=" * 62)
print("gain = fusion - ehr; gap = ehr - ctpa.")
print("Both contain ehr, so part of the fit is"
      " arithmetic.")

print("")
print("CHECK 1: gain vs CTPA AUC alone")
print("  mechanism predicts POSITIVE;"
      " a pure ceiling effect predicts little")
fit(t["ctpa"].values, t["late_best"].values,
    "gain_vs_ctpa", res)
fit(t2["ctpa"].values, t2["late_best"].values,
    "gain_vs_ctpa_nocv", res)

print("")
print("CHECK 2: gain vs EHR AUC alone")
print("  a pure ceiling effect predicts a strong"
      " NEGATIVE here")
fit(t["ehr"].values, t["late_best"].values,
    "gain_vs_ehr", res)

print("")
print("CHECK 3: partial correlation,"
      " gain vs gap holding EHR fixed")
print("  survives if the gap matters beyond the"
      " shared ehr term")
partial(t["gap"].values, t["late_best"].values,
        t["ehr"].values, "partial_gap_ehr", res)
partial(t2["gap"].values, t2["late_best"].values,
        t2["ehr"].values,
        "partial_gap_ehr_nocv", res)

print("")
print("CHECK 4: partial correlation,"
      " gain vs CTPA holding EHR fixed")
partial(t["ctpa"].values, t["late_best"].values,
        t["ehr"].values, "partial_ctpa_ehr", res)

print("")
print("CHECK 5: fused AUC vs gap")
print("  fused AUC does not subtract ehr, so this"
      " is free of the shared term")
fit(t["gap"].values, t["late_auc"].values,
    "fusedauc_vs_gap", res)
fit(t["ctpa"].values, t["late_auc"].values,
    "fusedauc_vs_ctpa", res)

print("")
print("CHECK 6: ratio instead of difference")
print("  ctpa/ehr is a scale-free gap measure")
fit(t["ratio"].values, t["late_best"].values,
    "gain_vs_ratio", res)
fit(t2["ratio"].values, t2["late_best"].values,
    "gain_vs_ratio_nocv", res)

print("")
print("PREDICTED GAIN AT SELECTED GAPS")
for tag in ("late_best", "late_best_nocv"):
    o = [x for x in res if x["set"] == tag]
    if not o:
        continue
    o = o[0]
    print("  %s (slope %+.4f)" % (tag, o["slope"]))
    for gp in (0.04, 0.08, 0.12, 0.16, 0.20):
        print("    gap %.2f -> %+.4f"
              % (gp, o["intercept"]
                 + o["slope"] * gp))
    if o["slope"] < 0:
        z = -o["intercept"] / o["slope"]
        print("    gain crosses zero at"
              " gap %.3f" % z)

print("")
print("OUT-OF-SAMPLE PREDICTION FOR THE"
      " TRANSFORMER CTPA MODALITY")
o = [x for x in res if x["set"] == "late_best"]
if o:
    o = o[0]
    row = t[(t["direction"] == "I2M")
            & (t["outcome"] == "death_30d")]
    if len(row):
        ehr = float(row["ehr"].iloc[0])
        cur = float(row["ctpa"].iloc[0])
        obs = float(row["late_best"].iloc[0])
        print("  I2M death_30d: ehr %.4f,"
              " ctpa %.4f, observed gain %+.4f"
              % (ehr, cur, obs))
        for nc in (0.75, 0.78, 0.80, 0.82):
            gp = ehr - nc
            pr = o["intercept"] + o["slope"] * gp
            print("    if ctpa rises to %.2f:"
                  " gap %.3f -> predicted"
                  " gain %+.4f" % (nc, gp, pr))

if os.path.exists(OLD):
    o2 = build(pd.read_csv(OLD))
    m = t[["direction", "outcome", "gap",
           "late_best"]].merge(
        o2[["direction", "outcome", "gap",
            "late_best"]],
        on=["direction", "outcome"],
        suffixes=("_q", "_std"))
    print("")
    print("QUANTILE VS STANDARDISED GRID")
    print(m.round(4).to_string(index=False))
    print("  gap correlation between grids:"
          " %.3f"
          % float(np.corrcoef(m["gap_q"],
                              m["gap_std"])[0, 1]))
    fit(m["gap_std"].values,
        m["late_best_std"].values,
        "late_best_std_grid", res)

pd.DataFrame(res).to_csv(
    os.path.join(PROC,
                 "gap_regression_quantile.csv"),
    index=False)
print("")
print("saved gap_table_quantile.csv,"
      " gap_regression_quantile.csv")