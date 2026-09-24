"""Tests whether the v4 features add to the ECG
modality.

Compares six feature sets on identical folds,
each swept over C, since the earlier work showed
the operating point matters more than the feature
set:

  logits          71 SCP statement logits
  logits+v3       plus the v3 measurements
                  (already worth +0.0088 on
                  cv_first)
  logits+v4       plus the v4 measurements
  logits+v3+v4    both measurement sets
  logits+daniel   plus the Daniel score as one
                  column, the validated composite
  logits+dancomp  plus the six Daniel components
                  separately

pc_ratio and apen are dropped: at 10-15 beats
their medians (1.01 and 0.04) sit far outside the
expected ranges, so they are ultra-short
artefacts rather than measurements.

Five seeds, class weighting balanced throughout.
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ecg_v4_test.csv
  results\\ecg_v4_test_log.txt
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
CGRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3,
         3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
OUTS = ["cv_first", "death_30d_inhosp",
        "composite_30d", "death_30d"]
DEST = os.path.join(PROC, "ecg_v4_test.csv")
PUB = {"composite_30d": 0.7207,
       "death_30d": 0.7229,
       "death_30d_inhosp": 0.7063,
       "cv_first": 0.6849}

DROP3 = ("rec", "t_flag", "vm_noise",
         "vm_base", "vm_rpeak", "net_i",
         "net_avf")
# ultra-short artefacts, see the medians
BAD4 = ("pc_ratio", "apen")
DAN = ["dan_hr100", "dan_s1q3t3",
       "dan_rbbb", "dan_twi_v14",
       "dan_ste_avr", "dan_afib"]


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
    best, curve = None, []
    for C in CGRID:
        v = np.array([run(X, y, grp, s, C)
                      for s in SEEDS])
        curve.append((C, v.mean()))
        if best is None or v.mean() > best[1]:
            best = (C, v.mean(),
                    v.std(ddof=1))
    return best, curve


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


def load_meas(path, drop):
    m = pd.read_csv(path)
    m["rec"] = m["rec"].astype(str).apply(
        lambda s: s if s.startswith("files/")
        else "files/" + s.replace("\\", "/")
        .strip("/"))
    m = m[m["rec"].isin(keep)]
    cols = [c for c in m.columns
            if c not in drop
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
    keepc = [c for c in ag.columns
             if c not in ("subject_id",
                          "hadm_id")]
    cov = ag[keepc].notna().mean()
    keepc = [c for c in keepc
             if cov[c] >= 0.50]
    return ag, keepc


V3, C3 = load_meas(
    os.path.join(
        PROC, "ecg_derived_v3_record.csv"),
    DROP3)
V4, C4 = load_meas(
    os.path.join(
        PROC, "ecg_derived_v4_record.csv"),
    DROP3 + BAD4)
C3 = [c + "_v3" for c in C3]
V3.columns = [c if c in ("subject_id",
                         "hadm_id")
              else c + "_v3"
              for c in V3.columns]
DANC = [c for c in C4
        if any(c.startswith(k) for k in DAN)]
DSC = [c for c in C4
       if c.startswith("daniel_score")]
print("v3 columns %d   v4 columns %d"
      % (len(C3), len(C4)))
print("  Daniel components %d  score %d"
      % (len(DANC), len(DSC)), flush=True)

lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
D = A.merge(lab, on=["subject_id", "hadm_id"],
            how="inner")
D = D.merge(V3.drop(columns=["subject_id"]),
            on="hadm_id", how="left")
D = D.merge(V4.drop(columns=["subject_id"]),
            on="hadm_id", how="left")
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
    XL = d[LG].values.astype(float)
    X3 = d[C3].values.astype(float)
    X4 = d[C4].values.astype(float)
    XD = (d[DSC].values.astype(float)
          if DSC else None)
    XC = d[DANC].values.astype(float)

    SETS = [
        ("logits", XL),
        ("logits+v3", np.column_stack(
            [XL, X3])),
        ("logits+v4", np.column_stack(
            [XL, X4])),
        ("logits+v3+v4", np.column_stack(
            [XL, X3, X4])),
        ("logits+dancomp", np.column_stack(
            [XL, XC])),
    ]
    if XD is not None:
        SETS.append(("logits+daniel",
                     np.column_stack([XL, XD])))

    print("")
    print("=" * 76)
    print("%s  n=%d ev=%d   published %.4f"
          % (oc, len(y), int(y.sum()),
             PUB[oc]), flush=True)
    print("  %-16s %6s %8s %8s %9s"
          % ("feature set", "nfeat", "peak C",
             "AUROC", "SD"))

    base = None
    for nm, X in SETS:
        (C, m, s), _ = sweep(X, y, grp)
        if nm == "logits":
            base = m
        edge = ("  (grid edge)"
                if C in (CGRID[0], CGRID[-1])
                else "")
        print("  %-16s %6d %8.0e %8.4f"
              " %9.4f  %+.4f%s"
              % (nm, X.shape[1], C, m, s,
                 m - base, edge), flush=True)
        res.append({
            "outcome": oc, "featset": nm,
            "nfeat": X.shape[1], "C": C,
            "auc": m, "sd": s,
            "vs_logits": m - base,
            "published": PUB[oc],
            "vs_published": m - PUB[oc]})

    s_ = [x for x in res
          if x["outcome"] == oc]
    b = max(s_, key=lambda x: x["auc"])
    print("")
    print("  best: %s  %.4f"
          "  (%+.4f over logits,"
          " %+.4f over published)"
          % (b["featset"], b["auc"],
             b["vs_logits"],
             b["vs_published"]), flush=True)

r = pd.DataFrame(res)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 76)
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
                    values="C")
      .to_string())
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
    print("  %-18s %-16s %.4f at C=%.0e"
          "  (published %.4f, %+.4f)"
          % (oc, b["featset"], b["auc"],
             b["C"], b["published"],
             b["vs_published"]))
print("")
print("saved", DEST, r.shape)