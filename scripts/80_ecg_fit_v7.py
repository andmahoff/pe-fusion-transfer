"""Tests whether any of the measurement work
improves the model.

Compares feature sets on identical folds, each
swept over C, since every previous run showed the
operating point matters more than the feature set
and wider blocks want heavier shrinkage.

FEATURE SETS
  logits        71 SCP statement logits
  L+v4          the v4 measurements
  L+v5          v5: spatial QRS-T, P-wave family,
                voltages, corrected Shopp flags
  L+v7          v7: v5 plus the slope-based QRS,
                consensus QT/QTc and the Tp-e
                quality flag
  L+shopp       the six flags only
  L+danw        the weighted Daniel score as a
                single column. It correlates only
                0.508 with the unweighted count,
                so it is a genuinely different
                feature; the unweighted version
                added nothing.
  L+danw+shopp  both, since the score weights the
                graded T component at up to 12 of
                21 points while the flags do not

TP-E TEST
  L+v7_tpe40    Tp-e kept at the strict 40 ms
                consensus tolerance, 7% coverage
  L+v7_tpe60    Tp-e kept at a relaxed 60 ms
                tolerance, roughly 75% coverage
  Answers whether a noisy Tp-e beats no Tp-e.

REGULARISATION
  C swept over 13 values from 1e-6 to 1, and the
  grid-edge flag reports any cell still at a
  boundary.

Five seeds, class weighting balanced.
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ecg_fit_v7.csv
  results\\ecg_fit_v7_log.txt
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
RAW = os.path.join("data", "raw", "ecg")
SEEDS = [42, 7, 13, 1, 2]
NFOLD = 5
WIN_LO, WIN_HI = -12.0, 48.0
CGRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4,
         3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
         1e-1, 3e-1, 1.0]
OUTS = ["cv_first", "death_30d_inhosp",
        "composite_30d", "death_30d"]
DEST = os.path.join(PROC, "ecg_fit_v7.csv")
PUB = {"composite_30d": 0.7207,
       "death_30d": 0.7229,
       "death_30d_inhosp": 0.7063,
       "cv_first": 0.6849}
DROP = ("rec", "t_flag", "vm_noise",
        "vm_base", "vm_rpeak", "net_i",
        "net_avf", "apen")
SHOPP = ("shopp_hr100", "shopp_s1q3t3",
         "shopp_rbbb", "shopp_twi_v14",
         "shopp_ste_avr", "shopp_afib")
TPE = ("tpe_ms", "tpe_qt")


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def run(X, y, grp, seed, C):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(X, y, grp):
        im = SimpleImputer(strategy="median")
        a = im.fit_transform(X[tr])
        b = im.transform(X[te])
        sc = StandardScaler()
        a, b = sc.fit_transform(a), sc.transform(b)
        m = LogisticRegression(
            C=C, max_iter=8000,
            class_weight="balanced")
        m.fit(a, y[tr])
        p[te] = m.predict_proba(b)[:, 1]
    return roc_auc_score(y, p)


def sweep(X, y, grp):
    best = None
    for C in CGRID:
        v = np.array([run(X, y, grp, s, C)
                      for s in SEEDS])
        if best is None or v.mean() > best[1]:
            best = (C, v.mean(),
                    v.std(ddof=1))
    return best


# ---------- windowed logits ----------
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

R = idx.merge(rec, on="rec", how="inner")
R["t"] = pd.to_datetime(
    R["ecg_charttime"], errors="coerce")
R = R.merge(adm[["hadm_id", "admittime"]],
            on="hadm_id", how="left")
R["h"] = ((R["t"] - R["admittime"])
          .dt.total_seconds() / 3600.0)
R = R[(R["h"] >= WIN_LO)
      & (R["h"] <= WIN_HI)].copy()
A = R.groupby(["subject_id", "hadm_id"],
              as_index=False)[LG].mean()
keep = set(R["rec"])
print("windowed admissions:", len(A))


def load(fn, tag, extra_drop=()):
    p = os.path.join(PROC, fn)
    if not os.path.exists(p):
        print("  missing:", fn)
        return None, []
    m = pd.read_csv(p)
    m["rec"] = m["rec"].astype(str).apply(
        lambda s: s if s.startswith("files/")
        else "files/" + s.replace("\\", "/")
        .strip("/"))
    m = m[m["rec"].isin(keep)]
    cols = [c for c in m.columns
            if c not in DROP
            and c not in extra_drop
            and pd.api.types
            .is_numeric_dtype(m[c])]
    j = idx[["subject_id", "hadm_id",
             "rec"]].merge(
        m[["rec"] + cols], on="rec",
        how="inner")
    ag = j.groupby(["subject_id", "hadm_id"],
                   as_index=False)[cols].agg(
        ["mean", "max"])
    ag.columns = ["%s_%s" % (a, b) if b else a
                  for a, b in ag.columns]
    ag = ag.reset_index()
    kc = [c for c in ag.columns
          if c not in ("subject_id",
                       "hadm_id")]
    cov = ag[kc].notna().mean()
    kc = [c for c in kc if cov[c] >= 0.40]
    ag = ag.rename(columns={
        c: c + "__" + tag for c in kc})
    kc = [c + "__" + tag for c in kc]
    return ag[["subject_id", "hadm_id"]
              + kc], kc


V4, C4 = load("ecg_derived_v4_record.csv",
              "v4")
V5, C5 = load("ecg_derived_v5_record.csv",
              "v5")
# v7 without Tp-e, and two Tp-e variants
V7, C7 = load("ecg_derived_v7_record.csv",
              "v7", extra_drop=TPE)
V7T, C7T = load("ecg_derived_v7_record.csv",
                "tpe")
TPE_C = [c for c in C7T
         if c.startswith(TPE)]
print("v4 %d  v5 %d  v7 %d  tpe %d"
      % (len(C4), len(C5), len(C7),
         len(TPE_C)), flush=True)

lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
D = A.merge(lab, on=["subject_id", "hadm_id"],
            how="inner")
for fr in (V4, V5, V7, V7T):
    if fr is not None:
        D = D.merge(
            fr.drop(columns=["subject_id"]),
            on="hadm_id", how="left")
SH = [c for c in C7
      if c.startswith(tuple(
          s + "_" for s in SHOPP))]
DW = [c for c in C7
      if c.startswith("daniel_weighted")]
print("shopp %d  daniel_weighted %d"
      % (len(SH), len(DW)))
print("cohort:", len(D), flush=True)

res = []
t0 = time.time()

for oc in OUTS:
    d = D
    if oc == "cv_first":
        d = D[D["death_first"] == 0]
    y = pd.to_numeric(
        d[oc], errors="coerce").fillna(
        0).astype(int).values
    grp = d["subject_id"].values

    def blk(cols):
        return (d[cols].values.astype(float)
                if cols else None)
    XL = d[LG].values.astype(float)
    SETS = [("logits", XL)]
    for nm, cs in (("L+v4", C4), ("L+v5", C5),
                   ("L+v7", C7),
                   ("L+shopp", SH),
                   ("L+danw", DW)):
        if cs:
            SETS.append(
                (nm, np.column_stack(
                    [XL, blk(cs)])))
    if DW and SH:
        SETS.append(
            ("L+danw+shopp",
             np.column_stack(
                 [XL, blk(DW), blk(SH)])))
    if TPE_C:
        SETS.append(
            ("L+v7+tpe",
             np.column_stack(
                 [XL, blk(C7), blk(TPE_C)])))

    print("")
    print("=" * 78)
    print("%s  n=%d ev=%d   published %.4f"
          % (oc, len(y), int(y.sum()),
             PUB[oc]), flush=True)
    print("  %-14s %5s %8s %8s %8s %s"
          % ("feature set", "nfeat", "peak C",
             "AUROC", "SD", "vs logits"))

    base = None
    for nm, X in SETS:
        C, m, s = sweep(X, y, grp)
        if nm == "logits":
            base = m
        edge = ("  EDGE"
                if C in (CGRID[0], CGRID[-1])
                else "")
        print("  %-14s %5d %8.0e %8.4f"
              " %8.4f  %+.4f%s"
              % (nm, X.shape[1], C, m, s,
                 m - base, edge), flush=True)
        res.append({
            "outcome": oc, "featset": nm,
            "nfeat": X.shape[1], "C": C,
            "auc": m, "sd": s,
            "vs_logits": m - base,
            "published": PUB[oc],
            "vs_published": m - PUB[oc],
            "at_edge": int(
                C in (CGRID[0], CGRID[-1]))})

    s_ = [x for x in res
          if x["outcome"] == oc]
    b = max(s_, key=lambda x: x["auc"])
    print("")
    print("  best: %s  %.4f  (%+.4f over"
          " logits, %+.4f over published)"
          % (b["featset"], b["auc"],
             b["vs_logits"],
             b["vs_published"]), flush=True)

r = pd.DataFrame(res)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 78)
print("PEAK AUROC BY FEATURE SET")
print(r.pivot_table(index="featset",
                    columns="outcome",
                    values="auc")
      .round(4).to_string())
print("")
print("GAIN OVER LOGITS ALONE")
print(r.pivot_table(index="featset",
                    columns="outcome",
                    values="vs_logits")
      .round(4).to_string())
print("")
print("C AT THE PEAK")
print(r.pivot_table(index="featset",
                    columns="outcome",
                    values="C").to_string())
print("")
print("  columns vs optimal C, by feature set:")
w = r.groupby("featset").agg(
    nfeat=("nfeat", "first"),
    medC=("C", "median"))
print(w.sort_values("nfeat").round(6)
      .to_string())
ne = int(r["at_edge"].sum())
print("")
print("  cells at a grid edge: %d of %d"
      % (ne, len(r)))

print("")
print("SEED SD")
print(r.pivot_table(index="featset",
                    columns="outcome",
                    values="sd")
      .round(4).to_string())

print("")
print("DOES A NOISY Tp-e BEAT NO Tp-e?")
for oc in OUTS:
    s = r[r["outcome"] == oc]
    a = s[s["featset"] == "L+v7"]
    b = s[s["featset"] == "L+v7+tpe"]
    if len(a) and len(b):
        print("  %-18s without %.4f"
              "   with %.4f   %+.4f"
              % (oc, a["auc"].iloc[0],
                 b["auc"].iloc[0],
                 b["auc"].iloc[0]
                 - a["auc"].iloc[0]))

print("")
print("WEIGHTED DANIEL SCORE")
for oc in OUTS:
    s = r[r["outcome"] == oc]
    for nm in ("L+danw", "L+shopp",
               "L+danw+shopp"):
        q = s[s["featset"] == nm]
        if len(q):
            print("  %-18s %-14s %+.4f"
                  % (oc, nm,
                     q["vs_logits"].iloc[0]))

print("")
print("AVERAGE RANK ACROSS OUTCOMES")
rk = r.pivot_table(index="featset",
                   columns="outcome",
                   values="auc").rank(
    ascending=False).mean(axis=1)
print(rk.sort_values().round(2).to_string())

print("")
print("BEST PER OUTCOME")
for oc, s in r.groupby("outcome"):
    b = s.loc[s["auc"].idxmax()]
    print("  %-18s %-14s %.4f at C=%.0e"
          "  (published %.4f, %+.4f)"
          % (oc, b["featset"], b["auc"],
             b["C"], b["published"],
             b["vs_published"]))
print("")
print("saved", DEST, r.shape)