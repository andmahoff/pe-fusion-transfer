"""Tests whether the measurement work improves
the ECG model.

Replaces script 80. Five corrections:

  1 AGGREGATION. Script 80 re-summarised from
    the record file with plain mean and max, so
    the signed-worst and circular-mean fixes,
    which only existed in the admission-level
    files, never reached the model. Both are now
    applied here, after the -12h to +48h window.
      - for the ten measurements where abnormal
        means more negative (S in I, Q in III,
        st_min, the chest-lead T amplitudes),
        the worst recording is the minimum, not
        the maximum
      - axes are aggregated by circular mean;
        arithmetic mean puts -179 and +179, two
        degrees apart, at zero

  2 v8 FEATURES. Points at ecg_derived_v8, which
    carries the corrected Tp-e formula, the
    RR-scaled TP baseline, the conventional
    0.05 mV aVR threshold and the Daniel score
    coded from the published table.

  3 DANIEL SCORE tested three ways: as the
    continuous 0-21 score, at its two published
    cutoffs (>=10 for severe pulmonary
    hypertension, >=3 for a complicated
    in-hospital course), and alongside the Shopp
    flags.

  4 Tp-e tested explicitly, with the coverage
    filter lowered to 20% so a genuinely noisy
    Tp-e can be compared against no Tp-e rather
    than silently dropped as in script 80.

  5 REGULARISATION re-optimised per feature set
    over 13 values from 1e-6 to 1, since every
    previous run showed wider blocks want
    heavier shrinkage. Grid-edge cells flagged.

Five seeds, class weighting balanced.
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ecg_fit_v8.csv
  results\\ecg_fit_v8_log.txt
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
DEST = os.path.join(PROC, "ecg_fit_v8.csv")
PUB = {"composite_30d": 0.7207,
       "death_30d": 0.7229,
       "death_30d_inhosp": 0.7063,
       "cv_first": 0.6849}

DROP = ("rec", "t_flag", "vm_noise",
        "vm_base", "vm_rpeak", "net_i",
        "net_avf", "apen", "tp_win_ms")
# FIX 1: abnormal means more negative
NEG_BAD = ("s_wave_i", "q_wave_iii", "st_min",
           "t_v1", "t_v2", "t_v3", "t_v4",
           "t_v5", "t_v6", "t_wave_iii")
ANG = ("qrs_axis", "t_axis", "p_axis")
SHOPP = ("shopp_hr100", "shopp_s1q3t3",
         "shopp_rbbb", "shopp_twi_v14",
         "shopp_ste_avr", "shopp_afib")
DANC = ("daniel_score", "daniel_ge10",
        "daniel_ge3", "daniel_twi")
TPE = ("tpe_ms", "tpe_qt")
MINCOV = 0.20            # FIX 4


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
print("windowed admissions:", len(A),
      " recordings:", len(R), flush=True)


def aggregate(j, cols, tag):
    """FIX 1: signed-aware worst value, and
    circular means for the axes. Applied here
    because the window must be applied before
    summarising, which the admission-level
    files do not do."""
    g = j.groupby(["subject_id", "hadm_id"])
    pos = [c for c in cols
           if not c.startswith(NEG_BAD)
           and c not in ANG]
    neg = [c for c in cols
           if c.startswith(NEG_BAD)]
    parts = [g[cols].mean()
             .add_suffix("_mean")]
    if pos:
        parts.append(g[pos].max()
                     .add_suffix("_worst"))
    if neg:
        parts.append(g[neg].min()
                     .add_suffix("_worst"))
    ag = pd.concat(parts, axis=1).reset_index()
    for c in ANG:
        if c not in j.columns:
            continue
        rad = np.radians(j[c])
        tmp = j[["subject_id",
                 "hadm_id"]].copy()
        tmp["s"] = np.sin(rad)
        tmp["c"] = np.cos(rad)
        gg = tmp.groupby(["subject_id",
                          "hadm_id"]).mean()
        cm = np.degrees(np.arctan2(
            gg["s"], gg["c"])).rename(
            c + "_circ").reset_index()
        ag = ag.merge(cm,
                      on=["subject_id",
                          "hadm_id"],
                      how="left")
        ag = ag.drop(columns=[c + "_mean",
                              c + "_worst"],
                     errors="ignore")
    kc = [c for c in ag.columns
          if c not in ("subject_id",
                       "hadm_id")]
    cov = ag[kc].notna().mean()
    kc = [c for c in kc if cov[c] >= MINCOV]
    ag = ag.rename(columns={
        c: c + "__" + tag for c in kc})
    return (ag[["subject_id", "hadm_id"]
               + [c + "__" + tag for c in kc]],
            [c + "__" + tag for c in kc])


def load(fn, tag, extra_drop=()):
    p = os.path.join(PROC, fn)
    if not os.path.exists(p):
        print("  MISSING:", fn)
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
    return aggregate(j, cols, tag)


V4, C4 = load("ecg_derived_v4_record.csv",
              "v4")
V7, C7 = load("ecg_derived_v7_record.csv",
              "v7")
V8, C8 = load("ecg_derived_v8_record.csv",
              "v8", extra_drop=TPE)
V8T, C8T = load("ecg_derived_v8_record.csv",
                "tp")
TP_C = [c for c in C8T
        if c.startswith(TPE)]
print("v4 %d  v7 %d  v8 %d  tpe %d"
      % (len(C4), len(C7), len(C8),
         len(TP_C)), flush=True)

lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
D = A.merge(lab, on=["subject_id", "hadm_id"],
            how="inner")
for fr in (V4, V7, V8, V8T):
    if fr is not None:
        D = D.merge(
            fr.drop(columns=["subject_id"]),
            on="hadm_id", how="left")

SH = [c for c in C8 if c.startswith(
    tuple(s + "_" for s in SHOPP))]
DS = [c for c in C8
      if c.startswith("daniel_score")]
DCUT = [c for c in C8
        if c.startswith(("daniel_ge10",
                         "daniel_ge3"))]
DALL = [c for c in C8 if c.startswith(DANC)]
print("shopp %d  daniel_score %d"
      "  cutoffs %d  all daniel %d"
      % (len(SH), len(DS), len(DCUT),
         len(DALL)))
print("cohort:", len(D), flush=True)
print("")
print("AGGREGATION CHECK")
nb = [c for c in C8
      if any(c.startswith(n) for n in NEG_BAD)
      and "_worst" in c]
na = [c for c in C8 if "_circ" in c]
print("  signed-worst columns (minimum"
      " taken): %d" % len(nb))
print("  circular-mean axis columns: %d"
      % len(na))
print("  script 80 had neither", flush=True)

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
    for nm, cs in (("L+v4", C4),
                   ("L+v7", C7),
                   ("L+v8", C8),
                   ("L+shopp", SH),
                   ("L+danscore", DS),
                   ("L+dancut", DCUT),
                   ("L+danall", DALL)):
        if cs:
            SETS.append((nm, np.column_stack(
                [XL, blk(cs)])))
    if DALL and SH:
        SETS.append(("L+dan+shopp",
                     np.column_stack(
                         [XL, blk(DALL),
                          blk(SH)])))
    if TP_C:
        SETS.append(("L+v8+tpe",
                     np.column_stack(
                         [XL, blk(C8),
                          blk(TP_C)])))

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
print("GAIN OVER THE PUBLISHED FIGURE")
print(r.pivot_table(index="featset",
                    columns="outcome",
                    values="vs_published")
      .round(4).to_string())

print("")
print("C AT THE PEAK")
print(r.pivot_table(index="featset",
                    columns="outcome",
                    values="C").to_string())
w = r.groupby("featset").agg(
    nfeat=("nfeat", "first"),
    medC=("C", "median"))
print("")
print("  columns vs optimal C:")
print(w.sort_values("nfeat").round(6)
      .to_string())
ne = int(r["at_edge"].sum())
print("  cells at a grid edge: %d of %d"
      % (ne, len(r)))

print("")
print("v4 -> v7 -> v8 PROGRESSION")
p = r[r["featset"].isin(
    ["logits", "L+v4", "L+v7", "L+v8"])]
print(p.pivot_table(index="featset",
                    columns="outcome",
                    values="auc")
      .round(4).to_string())

print("")
print("DOES A NOISY Tp-e BEAT NO Tp-e?")
any_tpe = False
for oc in OUTS:
    s = r[r["outcome"] == oc]
    a = s[s["featset"] == "L+v8"]
    b = s[s["featset"] == "L+v8+tpe"]
    if len(a) and len(b):
        any_tpe = True
        print("  %-18s without %.4f"
              "   with %.4f   %+.4f"
              % (oc, a["auc"].iloc[0],
                 b["auc"].iloc[0],
                 b["auc"].iloc[0]
                 - a["auc"].iloc[0]))
if not any_tpe:
    print("  Tp-e below the %.0f%% coverage"
          " floor, so it could not be tested"
          % (100 * MINCOV))

print("")
print("DANIEL SCORE, THREE FORMS")
for oc in OUTS:
    s = r[r["outcome"] == oc]
    line = "  %-18s" % oc
    for nm in ("L+danscore", "L+dancut",
               "L+danall", "L+shopp",
               "L+dan+shopp"):
        q = s[s["featset"] == nm]
        if len(q):
            line += "  %s %+.4f" % (
                nm.replace("L+", ""),
                q["vs_logits"].iloc[0])
    print(line)

print("")
print("SEED SD")
print(r.pivot_table(index="featset",
                    columns="outcome",
                    values="sd")
      .round(4).to_string())

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