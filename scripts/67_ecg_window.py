"""Apply the dissertation's ECG matching window.

Section 3.4: "ECGs were linked within 12 hours
prior to 48 hours post-admission to capture both
emergency department tracings and early ward
monitoring (3,507 admissions)."

The record index carries every linked ECG with no
window applied, giving 3,512 admissions over 6,004
recordings. This filters to -12h to +48h relative
to admission and checks whether the count lands
on 3,507.

Runs the exact baseline (C = 1.0, balanced,
grouped 5-fold, seed 42) on the windowed cohort
and compares against both the unwindowed version
and the published figures.
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ecg_window.csv
  data\\processed\\ecg_record_index_windowed.csv
  data\\processed\\p_ecg_windowed.csv
  results\\ecg_window_log.txt
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
RAW = os.path.join("data", "raw", "ecg")
SEED, NFOLD = 42, 5
WIN_LO, WIN_HI = -12.0, 48.0
OUTS = ["composite_30d", "death_30d",
        "death_30d_inhosp", "cv_first"]
PUB = {"composite_30d": (0.7207, 0.3065, 547),
       "death_30d": (0.7229, 0.2422, 398),
       "death_30d_inhosp": (0.7063, 0.1537,
                            242),
       "cv_first": (0.6849, 0.1272, 171)}
DEST = os.path.join(PROC, "ecg_window.csv")


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def fit_exact(X, y, grp, seed=SEED):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(X, y, grp):
        sc = StandardScaler()
        a = sc.fit_transform(X[tr])
        b = sc.transform(X[te])
        m = LogisticRegression(
            C=1.0, max_iter=5000,
            class_weight="balanced")
        m.fit(a, y[tr])
        p[te] = m.predict_proba(b)[:, 1]
    return p


rec = pd.read_csv(os.path.join(
    PROC, "exp0_logits_record.csv"))
idx = pd.read_csv(os.path.join(
    RAW, "ecg_record_index.csv"))
idx["rec"] = idx["rec"].astype(str).apply(
    lambda s: s if s.startswith("files/")
    else "files/" + s.replace("\\", "/")
    .strip("/"))
LG = [c for c in rec.columns
      if c.startswith("scp_")]

adm = pd.read_csv(os.path.join(
    FIG, "index_admission_times.csv"))
adm["admittime"] = pd.to_datetime(
    adm["admittime"], errors="coerce")
print("admission times:", adm.shape)
print("columns:", list(adm.columns))

R = idx.merge(rec, on="rec", how="inner")
R["ecg_t"] = pd.to_datetime(
    R["ecg_charttime"], errors="coerce")
R = R.merge(adm[["hadm_id", "admittime"]],
            on="hadm_id", how="left")
R["h_rel"] = (
    (R["ecg_t"] - R["admittime"])
    .dt.total_seconds() / 3600.0)

print("")
print("BEFORE THE WINDOW")
print("  recordings:", len(R),
      " (dissertation used 6,004)")
print("  admissions:",
      R["hadm_id"].nunique(),
      " (dissertation used 3,507)")
print("  with a usable timestamp:",
      int(R["h_rel"].notna().sum()))
print("")
print("  hours relative to admission:")
q = R["h_rel"].dropna()
for p_ in (0.01, 0.05, 0.25, 0.5, 0.75,
           0.95, 0.99):
    print("    %4.0f%%  %+9.1f h"
          % (100 * p_, q.quantile(p_)))

W = R[(R["h_rel"] >= WIN_LO)
      & (R["h_rel"] <= WIN_HI)].copy()
print("")
print("AFTER THE %+.0fh to %+.0fh WINDOW"
      % (WIN_LO, WIN_HI))
print("  recordings:", len(W))
print("  admissions:", W["hadm_id"].nunique(),
      " (target 3,507)")
lost = (R["hadm_id"].nunique()
        - W["hadm_id"].nunique())
print("  admissions dropped:", lost,
      flush=True)

W[["subject_id", "hadm_id", "rec",
   "ecg_charttime", "h_rel"]].to_csv(
    os.path.join(
        PROC,
        "ecg_record_index_windowed.csv"),
    index=False)

AW = W.groupby(["subject_id", "hadm_id"],
               as_index=False)[LG].mean()
AU = R.groupby(["subject_id", "hadm_id"],
               as_index=False)[LG].mean()
lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
DW = AW.merge(lab, on=["subject_id",
                       "hadm_id"],
              how="inner")
DU = AU.merge(lab, on=["subject_id",
                       "hadm_id"],
              how="inner")

rows, preds = [], []
for oc in OUTS:
    if oc not in DW.columns:
        continue
    t_auc, t_ap, t_ev = PUB[oc]
    print("")
    print("=" * 70)
    print("%s   published %.4f  (%d events)"
          % (oc, t_auc, t_ev), flush=True)

    out = {}
    for tag, dd in (("unwindowed", DU),
                    ("windowed", DW)):
        d = dd
        if oc == "cv_first":
            d = dd[dd["death_first"] == 0]
        y = pd.to_numeric(
            d[oc], errors="coerce").fillna(
            0).astype(int).values
        grp = d["subject_id"].values
        X = d[LG].values.astype(float)
        p = fit_exact(X, y, grp)
        au = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        out[tag] = (au, ap, len(y),
                    int(y.sum()))
        print("  %-11s n=%4d ev=%3d"
              "  AUROC %.4f (%+.4f)"
              "  AP %.4f (%+.4f)"
              % (tag, len(y), int(y.sum()),
                 au, au - t_auc, ap,
                 ap - t_ap), flush=True)
        rows.append({
            "outcome": oc, "cohort": tag,
            "n": len(y), "ev": int(y.sum()),
            "target_ev": t_ev, "auc": au,
            "published": t_auc,
            "diff": au - t_auc, "ap": ap,
            "published_ap": t_ap})
        if tag == "windowed":
            o = d[["subject_id",
                   "hadm_id"]].copy()
            o["p_ecg"] = p
            o["outcome"] = oc
            preds.append(o)

    bw = abs(out["windowed"][0] - t_auc)
    bu = abs(out["unwindowed"][0] - t_auc)
    print("  -> the window %s the match"
          " (%.4f vs %.4f from target)"
          % ("IMPROVES" if bw < bu
             else "does not improve",
             bw, bu))

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)
if preds:
    pd.concat(preds).to_csv(
        os.path.join(PROC,
                     "p_ecg_windowed.csv"),
        index=False)

print("")
print("=" * 70)
print("SUMMARY")
print(r.pivot_table(index="outcome",
                    columns="cohort",
                    values="auc")
      .round(4).to_string())
print("")
print("EVENT COUNTS vs PUBLISHED")
print(r.pivot_table(index="outcome",
                    columns="cohort",
                    values="ev")
      .join(r.groupby("outcome")[
          "target_ev"].first())
      .to_string())
print("")
print("ABSOLUTE DIFFERENCE FROM PUBLISHED")
g = r.copy()
g["absdiff"] = g["diff"].abs()
print(g.pivot_table(index="outcome",
                    columns="cohort",
                    values="absdiff")
      .round(4).to_string())
m = g.groupby("cohort")["absdiff"].mean()
print("")
print("  mean absolute difference:")
print(m.round(4).to_string())
best = m.idxmin()
print("")
print("  closer to the published model:",
      best)
print("")
print("saved", DEST, r.shape)
print("saved data/processed/"
      "p_ecg_windowed.csv")