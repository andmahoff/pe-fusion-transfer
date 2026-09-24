"""Combine the Daniel components with the v4
morphology, and re-optimise regularisation for
the wider feature sets.

Script 75 found the two help different outcomes:
v4 gains on cv_first and composite, the Daniel
components on both mortality endpoints. This
combines them.

Two regularisation issues addressed:
  1 the C grid is extended to 1e-7, since v4
    peaked at 1e-4, only two steps above the old
    floor
  2 BLOCK SCALING divides each block by the
    square root of its width, so 130 v4 columns
    cannot dominate 71 logits and 12 Daniel
    components by count alone

v4core is the four continuous features that
validated cleanly against their literature
ranges: QRS-T angle, Tp-e, Tp-e/QT and R/S in V1.
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ecg_combine_v4.csv
  results\\ecg_combine_v4_log.txt
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
CGRID = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5,
         3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
         1e-2, 3e-2, 1e-1, 3e-1, 1.0]
OUTS = ["cv_first", "death_30d_inhosp",
        "composite_30d", "death_30d"]
DEST = os.path.join(
    PROC, "ecg_combine_v4.csv")
PUB = {"composite_30d": 0.7207,
       "death_30d": 0.7229,
       "death_30d_inhosp": 0.7063,
       "cv_first": 0.6849}

DROP = ("rec", "t_flag", "vm_noise",
        "vm_base", "vm_rpeak", "net_i",
        "net_avf", "pc_ratio", "apen")
DAN = ("dan_hr100", "dan_s1q3t3",
       "dan_rbbb", "dan_twi_v14",
       "dan_ste_avr", "dan_afib")
CORE = ("qrst_angle", "tpe_ms", "tpe_qt",
        "rs_ratio_v1")


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def run(X, y, grp, seed, C, blocks=None):
    """blocks: list of column counts. When
    given, each block is divided by the square
    root of its width after standardisation."""
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
        if blocks:
            i = 0
            for n in blocks:
                k = np.sqrt(n)
                a[:, i:i + n] /= k
                b[:, i:i + n] /= k
                i += n
        m = LogisticRegression(
            C=C, max_iter=8000,
            class_weight="balanced")
        m.fit(a, y[tr])
        p[te] = m.predict_proba(b)[:, 1]
    return roc_auc_score(y, p)


def sweep(X, y, grp, blocks=None):
    best = None
    for C in CGRID:
        v = np.array([run(X, y, grp, s, C,
                          blocks)
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

m4 = pd.read_csv(os.path.join(
    PROC, "ecg_derived_v4_record.csv"))
m4["rec"] = m4["rec"].astype(str).apply(
    lambda s: s if s.startswith("files/")
    else "files/" + s.replace("\\", "/")
    .strip("/"))
m4 = m4[m4["rec"].isin(keep)]
cols = [c for c in m4.columns
        if c not in DROP
        and pd.api.types.is_numeric_dtype(
            m4[c])]
j = idx[["subject_id", "hadm_id",
         "rec"]].merge(
    m4[["rec"] + cols], on="rec",
    how="inner")
ag = j.groupby(["subject_id", "hadm_id"],
               as_index=False)[cols].agg(
    ["mean", "max"])
ag.columns = ["%s_%s" % (a, b) if b else a
              for a, b in ag.columns]
ag = ag.reset_index()
ALL4 = [c for c in ag.columns
        if c not in ("subject_id",
                     "hadm_id")]
cov = ag[ALL4].notna().mean()
ALL4 = [c for c in ALL4 if cov[c] >= 0.50]
DANC = [c for c in ALL4
        if c.startswith(DAN)]
CORC = [c for c in ALL4
        if c.startswith(CORE)]
MORPH = [c for c in ALL4
         if c not in DANC
         and not c.startswith("daniel_")]
print("v4 total %d   Daniel %d   core %d"
      "   morphology %d"
      % (len(ALL4), len(DANC), len(CORC),
         len(MORPH)), flush=True)

lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
D = A.merge(lab, on=["subject_id", "hadm_id"],
            how="inner").merge(
    ag.drop(columns=["subject_id"]),
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
    XD = d[DANC].values.astype(float)
    XC = d[CORC].values.astype(float)
    XM = d[MORPH].values.astype(float)

    SETS = [
        ("logits", XL, None),
        ("L+dan", np.column_stack([XL, XD]),
         [len(LG), len(DANC)]),
        ("L+morph", np.column_stack([XL, XM]),
         [len(LG), len(MORPH)]),
        ("L+dan+core",
         np.column_stack([XL, XD, XC]),
         [len(LG), len(DANC), len(CORC)]),
        ("L+dan+morph",
         np.column_stack([XL, XD, XM]),
         [len(LG), len(DANC), len(MORPH)]),
    ]

    print("")
    print("=" * 78)
    print("%s  n=%d ev=%d   published %.4f"
          % (oc, len(y), int(y.sum()),
             PUB[oc]), flush=True)
    print("  %-14s %5s  %-24s  %s"
          % ("feature set", "nfeat",
             "unscaled", "block-scaled"))

    base = None
    for nm, X, blocks in SETS:
        cu, mu, su = sweep(X, y, grp)
        if blocks and len(blocks) > 1:
            cs_, ms, ss = sweep(X, y, grp,
                                blocks)
        else:
            cs_, ms, ss = cu, mu, su
        if nm == "logits":
            base = mu
        bm = max(mu, ms)
        eu = ("*" if cu == CGRID[0] else " ")
        print("  %-14s %5d  %.4f (SD %.4f)"
              " C=%.0e%s   %.4f C=%.0e"
              "   %+.4f"
              % (nm, X.shape[1], mu, su, cu,
                 eu, ms, cs_, bm - base),
              flush=True)
        res.append({
            "outcome": oc, "featset": nm,
            "nfeat": X.shape[1],
            "auc_unscaled": mu, "C_unscaled": cu,
            "sd_unscaled": su,
            "auc_scaled": ms, "C_scaled": cs_,
            "auc_best": bm,
            "vs_logits": bm - base,
            "published": PUB[oc],
            "vs_published": bm - PUB[oc],
            "at_grid_floor": int(
                cu == CGRID[0])})

    s_ = [x for x in res
          if x["outcome"] == oc]
    b = max(s_, key=lambda x: x["auc_best"])
    print("")
    print("  best: %s  %.4f"
          "  (%+.4f over logits,"
          " %+.4f over published)"
          % (b["featset"], b["auc_best"],
             b["vs_logits"],
             b["vs_published"]), flush=True)

r = pd.DataFrame(res)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 78)
print("BEST AUROC BY FEATURE SET")
print(r.pivot_table(index="featset",
                    columns="outcome",
                    values="auc_best")
      .round(4).to_string())
print("")
print("GAIN OVER LOGITS ALONE")
print(r.pivot_table(index="featset",
                    columns="outcome",
                    values="vs_logits")
      .round(4).to_string())
print("")
print("DOES BLOCK SCALING HELP?")
r["scale_gain"] = (r["auc_scaled"]
                   - r["auc_unscaled"])
print(r.pivot_table(index="featset",
                    columns="outcome",
                    values="scale_gain")
      .round(4).to_string())
print("")
print("  mean effect of block scaling: %+.4f"
      % r["scale_gain"].mean())
print("")
print("C AT THE PEAK, UNSCALED")
print(r.pivot_table(index="featset",
                    columns="outcome",
                    values="C_unscaled")
      .to_string())
nf = int(r["at_grid_floor"].sum())
print("")
print("  cells peaking at the 1e-7 floor: %d"
      " of %d" % (nf, len(r)))
if nf:
    print("  -> the grid is still too narrow"
          " for those cells")
print("")
print("AVERAGE RANK ACROSS OUTCOMES")
rk = r.pivot_table(index="featset",
                   columns="outcome",
                   values="auc_best").rank(
    ascending=False).mean(axis=1)
print(rk.sort_values().round(2).to_string())
print("")
print("BEST PER OUTCOME")
for oc, s in r.groupby("outcome"):
    b = s.loc[s["auc_best"].idxmax()]
    sc = ("block-scaled"
          if b["auc_scaled"] > b["auc_unscaled"]
          else "unscaled")
    print("  %-18s %-14s %.4f  %s"
          "  (published %.4f, %+.4f)"
          % (oc, b["featset"], b["auc_best"],
             sc, b["published"],
             b["vs_published"]))
print("")
print("saved", DEST, r.shape)