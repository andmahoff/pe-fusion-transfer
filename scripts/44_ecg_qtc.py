"""QTc correction for the derived ECG measurements.

Bazett divides QT by the square root of RR, which
over-corrects at fast rates. This cohort has a
median heart rate of 88 bpm, so Bazett inflates a
correct QT of 404 ms to 492 ms.

Computes four correction formulae, compares their
plausibility and their dependence on heart rate,
and rewrites the admission-level file with the
best one as the primary.
Run in venv_ecg only.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats as st

PROC = os.path.join("data", "processed")
REC = os.path.join(
    PROC, "ecg_derived_v2_record.csv")
DEST = os.path.join(PROC, "ecg_derived_v2.csv")
OUT = os.path.join("data", "raw", "ecg")

d = pd.read_csv(REC)
print("records:", len(d))

ok = d["qt_ms"].notna() & d["rr_mean"].notna()
print("with QT and RR:", int(ok.sum()))

rr = d["rr_mean"] / 1000.0
qt = d["qt_ms"]
hr = 60.0 / rr

d["qtc_bazett"] = qt / np.sqrt(rr)
d["qtc_fridericia"] = qt / (rr ** (1 / 3.0))
d["qtc_framingham"] = qt + 154.0 * (1.0 - rr)
d["qtc_hodges"] = qt + 1.75 * (hr - 60.0)

FORM = ["qtc_bazett", "qtc_fridericia",
        "qtc_framingham", "qtc_hodges"]

print("")
print("=" * 58)
print("COMPARISON OF CORRECTION FORMULAE")
print("=" * 58)
print("  %-18s %7s %7s %7s %8s"
      % ("formula", "median", "IQR lo",
         "IQR hi", "in range"))
for c in FORM:
    v = d.loc[ok, c].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if not len(v):
        continue
    pc = 100 * float(
        ((v >= 350) & (v <= 500)).mean())
    print("  %-18s %7.1f %7.1f %7.1f %7.1f%%"
          % (c, v.median(), v.quantile(0.25),
             v.quantile(0.75), pc))

print("")
print("DEPENDENCE ON HEART RATE")
print("  a good correction is INDEPENDENT of"
      " rate, so |r| near zero")
print("  %-18s %8s %8s" % ("formula", "r",
                           "slope /bpm"))
best, bestr = None, 9.9
for c in FORM + ["qt_ms"]:
    m = ok & d[c].notna() & np.isfinite(d[c])
    if m.sum() < 100:
        continue
    x = hr[m].values
    y = d.loc[m, c].values
    r, _ = st.pearsonr(x, y)
    sl = st.linregress(x, y).slope
    star = ""
    if c != "qt_ms" and abs(r) < bestr:
        bestr, best = abs(r), c
    print("  %-18s %+8.3f %8.3f%s"
          % (c, r, sl, star))

print("")
print("  least rate-dependent: %s (|r| %.3f)"
      % (best, bestr))

print("")
print("PLAUSIBILITY BY HEART-RATE BAND")
band = pd.cut(hr, [0, 60, 80, 100, 120, 300],
              labels=["<60", "60-80", "80-100",
                      "100-120", ">120"])
for c in FORM:
    v = d[c].replace([np.inf, -np.inf], np.nan)
    g = ((v >= 350) & (v <= 500)).groupby(
        band, observed=False).mean() * 100
    print("  %-18s %s"
          % (c, "  ".join(
              "%s %.0f%%" % (k, x)
              for k, x in g.items()
              if not np.isnan(x))))

BOUND = (300, 650)
for c in FORM:
    v = d[c]
    d[c] = v.where((v >= BOUND[0])
                   & (v <= BOUND[1]))

d["qtc_primary"] = d[best]
print("")
print("primary QTc set to:", best)
print("  coverage: %.1f%%"
      % (100 * float(
          d["qtc_primary"].notna().mean())))

d.to_csv(REC, index=False)
print("rewrote", REC, d.shape)

idx = pd.read_csv(os.path.join(
    OUT, "ecg_record_index.csv"))
m = idx[["subject_id", "hadm_id",
         "rec"]].drop_duplicates()
j = m.merge(d, on="rec", how="inner")
num = [c for c in j.columns
       if c not in ("rec", "subject_id",
                    "hadm_id")]
g = j.groupby(["subject_id", "hadm_id"])
agg = g[num].agg(["mean", "max", "min"])
agg.columns = ["%s_%s" % (a, b)
               for a, b in agg.columns]
agg = agg.reset_index()
agg["n_ecg"] = g.size().values
agg.to_csv(DEST, index=False)
print("rewrote", DEST, agg.shape)

print("")
print("FINAL RECORD-LEVEL SUMMARY")
for c in ["hr_mean", "rmssd", "pnn50",
          "pr_ms", "qrs_ms", "qt_ms",
          "qtc_primary", "qrs_axis"]:
    if c not in d.columns:
        continue
    v = d[c].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if len(v):
        print("  %-14s n=%4d  median %7.1f"
              "  IQR %.1f to %.1f"
              % (c, len(v), v.median(),
                 v.quantile(0.25),
                 v.quantile(0.75)))