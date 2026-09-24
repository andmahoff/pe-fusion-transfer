"""Formal test of the gap mechanism. Regresses
fusion gain on the unimodal gap across all twelve
direction-outcome cells, turning the qualitative
observation into a fitted relationship.
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
uni = pd.read_csv(os.path.join(PROC,
                               "unimodal.csv"))
grid = pd.read_csv(os.path.join(
    PROC, "grid_summary.csv"))

u = uni[uni["learner"] == "lr"]
u = u.pivot_table(index=["direction", "outcome"],
                  columns="modality",
                  values="auc").reset_index()
u["gap"] = u["ehr"] - u["ctpa"]
u["ratio"] = u["ctpa"] / u["ehr"]

FAM = {"early": ["early:plain",
                 "early:blockstd",
                 "early:pca"],
       "inter": ["inter:" + v
                 for v in f.INTER_VARIANTS],
       "late": ["late:mean", "late:wsrc"]}

rows = []
for _, r in u.iterrows():
    d, oc = r["direction"], r["outcome"]
    g = grid[(grid["direction"] == d)
             & (grid["outcome"] == oc)]
    rec = {"direction": d, "outcome": oc,
           "ehr": r["ehr"], "ctpa": r["ctpa"],
           "gap": r["gap"], "ratio": r["ratio"],
           "ev": int(g["ev"].iloc[0])}
    for fam, names in FAM.items():
        sub = g[g["arch"].isin(names)]
        if len(sub):
            rec[fam + "_best"] = sub["gain"].max()
            rec[fam + "_mean"] = sub["gain"].mean()
    sub = g[g["arch"] != "uni:ehr_only"]
    sub = sub[sub["arch"] != "uni:ctpa_only"]
    rec["any_best"] = sub["gain"].max()
    rec["best_arch"] = sub.loc[
        sub["gain"].idxmax(), "arch"]
    rows.append(rec)

t = pd.DataFrame(rows).sort_values("gap")
t.to_csv(os.path.join(PROC, "gap_table.csv"),
         index=False)

print("GAP vs FUSION GAIN, sorted by gap")
cols = ["direction", "outcome", "ehr", "ctpa",
        "gap", "ev", "late_best", "early_best",
        "inter_best", "any_best", "best_arch"]
print(t[cols].round(4).to_string(index=False))


def fit(x, y, lab):
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 4:
        print("  %-14s too few points" % lab)
        return None
    sl, ic, r, p, se = stats.linregress(x, y)
    rho, prho = stats.spearmanr(x, y)
    print("  %-14s slope %+.4f  r=%+.3f"
          "  R2=%.3f  p=%.4f  rho=%+.3f"
          " (p=%.4f)"
          % (lab, sl, r, r ** 2, p, rho, prho))
    return {"target": lab, "slope": sl,
            "intercept": ic, "r": r,
            "r2": r ** 2, "p": p,
            "spearman": rho, "sp_p": prho,
            "n": len(x)}


print("")
print("REGRESSION OF GAIN ON GAP (n=12 cells)")
res = []
for lab in ["late_best", "early_best",
            "inter_best", "any_best"]:
    o = fit(t["gap"].values, t[lab].values, lab)
    if o:
        res.append(o)

print("")
print("EXCLUDING cv_first (76-288 events,"
       " CTPA near chance)")
t2 = t[t["outcome"] != "cv_first"]
for lab in ["late_best", "early_best",
            "any_best"]:
    o = fit(t2["gap"].values, t2[lab].values,
            lab + "_nocv")
    if o:
        res.append(o)

print("")
print("PREDICTED GAIN AT SELECTED GAPS"
      " (late_best fit)")
o = [x for x in res
     if x["target"] == "late_best"]
if o:
    o = o[0]
    for gp in (0.05, 0.10, 0.15, 0.20):
        print("  gap %.2f -> %+.4f"
              % (gp, o["intercept"]
                 + o["slope"] * gp))

pd.DataFrame(res).to_csv(
    os.path.join(PROC, "gap_regression.csv"),
    index=False)
print("")
print("saved gap_table.csv, gap_regression.csv")