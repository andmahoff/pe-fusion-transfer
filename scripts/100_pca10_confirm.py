"""Ten-seed check of PCA to 10 components on
EHR+CTPA for in-hospital death.

In script 99 (three seeds), pca10 reached 0.8985
on death_30d_inhosp with EHR+CTPA against
late_wsrc at 0.8775, a margin of +0.0209 on 83
events, and lost on the other three outcomes.

This script tests the margin with
  ten seeds rather than three
  a paired bootstrap against late_wsrc
  the full k sweep from 2 to 40, since an effect
    confined to k=10 would suggest noise
  the same comparison on the other three
    outcomes
  a permutation check with shuffled labels,
    which should give about 0.5

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\pca10_confirm.csv
  results\\pca10_confirm_log.txt
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
SEEDS = [42, 7, 13, 1, 2, 3, 5, 8, 21, 99]
NFOLD = 5
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
KS = [2, 5, 8, 10, 12, 15, 20, 30, 40]
CT_LO, CT_HI = -48.0, 24.0
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
TARGET = "death_30d_inhosp"
DEST = os.path.join(PROC,
                    "pca10_confirm.csv")
# script 99, three seeds
S99 = {"death_30d_inhosp":
       {"pca10": 0.8985, "late": 0.8775},
       "death_30d":
       {"pca10": 0.8812, "late": 0.8831},
       "composite_30d":
       {"pca10": 0.8282, "late": 0.8437},
       "cv_first":
       {"pca10": 0.7552, "late": 0.8037}}


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def clean_fit(A):
    A = np.asarray(A, dtype=float).copy()
    med = np.nanmedian(A, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(A)
    if bad.any():
        ii = np.where(bad)
        A[ii] = np.take(med, ii[1])
    return A, med


def clean_apply(A, med):
    A = np.asarray(A, dtype=float).copy()
    bad = ~np.isfinite(A)
    if bad.any():
        ii = np.where(bad)
        A[ii] = np.take(med, ii[1])
    return A


def fit_pred(Xtr, ytr, Xte, gtr=None):
    im = SimpleImputer(strategy="median")
    a = im.fit_transform(Xtr)
    b = im.transform(Xte)
    sc = StandardScaler()
    a, b = sc.fit_transform(a), sc.transform(b)
    best, bc = -1.0, 1.0
    try:
        kk = min(3, max(2, int(ytr.sum()) // 10))
        icv = StratifiedGroupKFold(
            n_splits=kk, shuffle=True,
            random_state=42)
        g = (gtr if gtr is not None
             else np.arange(len(ytr)))
        for c in CS:
            q = np.zeros(len(ytr))
            for t2, v2 in icv.split(a, ytr, g):
                m = LogisticRegression(
                    C=c, max_iter=3000)
                m.fit(a[t2], ytr[t2])
                q[v2] = m.predict_proba(
                    a[v2])[:, 1]
            s = roc_auc_score(ytr, q)
            if s > best:
                best, bc = s, c
    except Exception:
        bc = 1.0
    m = LogisticRegression(C=bc, max_iter=3000)
    m.fit(a, ytr)
    return m.predict_proba(b)[:, 1]


def oof_pca(Xa, Xb, y, grp, k, seed):
    """PCA fitted inside each training fold."""
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        Etr, md = clean_fit(
            np.column_stack([Xa[tr], Xb[tr]]))
        Ete = clean_apply(
            np.column_stack([Xa[te], Xb[te]]),
            md)
        sc = StandardScaler()
        pc = PCA(n_components=k,
                 random_state=42)
        ztr = pc.fit_transform(
            sc.fit_transform(Etr))
        zte = pc.transform(sc.transform(Ete))
        p[te] = fit_pred(ztr, y[tr], zte,
                         grp[tr])
    return p


def oof_late(Xa, Xb, y, grp, seed):
    pa = np.zeros(len(y))
    pb = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(Xa, y, grp):
        pa[te] = fit_pred(Xa[tr], y[tr],
                          Xa[te], grp[tr])
        pb[te] = fit_pred(Xb[tr], y[tr],
                          Xb[te], grp[tr])
    pa, pb = _rank(pa), _rank(pb)
    bs, bw = -1.0, 0.5
    for w in np.arange(0, 1.001, 0.05):
        q = roc_auc_score(y, w * pb
                          + (1 - w) * pa)
        if q > bs:
            bs, bw = q, w
    return bw * pb + (1 - bw) * pa


def oof_early(Xa, Xb, y, grp, seed):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        Etr, md = clean_fit(
            np.column_stack([Xa[tr], Xb[tr]]))
        Ete = clean_apply(
            np.column_stack([Xa[te], Xb[te]]),
            md)
        p[te] = fit_pred(Etr, y[tr], Ete,
                         grp[tr])
    return p


# ---------------- cohort --------------------
mim_ehr = f.load_mimic_ehr()
EH = [c for c in f.EHR_COLS
      if c in mim_ehr.columns]
mm = f.load_mimic_ctpa46()
cidx = pd.read_csv(os.path.join(
    FIG, "ctpa_notes_index.csv"))
w = cidx[(cidx["h_before"] >= CT_LO)
         & (cidx["h_before"] <= CT_HI)]
hs = set(w["idx_hadm"].dropna().astype(int))
mm = mm[mm["hadm_id"].isin(hs)]
v2 = pd.read_csv(os.path.join(
    FIG, "mimic_imp_features_v2.csv"),
    usecols=["hadm_id"])
mm = mm[mm["hadm_id"].isin(set(v2["hadm_id"]))]
CT = [c for c in f.CTPA46 if c in mm.columns]
lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))

D = mm[["subject_id", "hadm_id"] + CT].merge(
    mim_ehr[["hadm_id"] + EH], on="hadm_id",
    how="inner").merge(
    lab, on=["subject_id", "hadm_id"],
    how="inner")
print("cohort:", len(D))
print("  EHR %d   CTPA %d   total %d features"
      % (len(EH), len(CT), len(EH) + len(CT)))
print("  script 99 used the three-modality"
      " cohort, so n may differ slightly",
      flush=True)

rows = []
t0 = time.time()

for oc in OUTS:
    if oc not in D.columns:
        continue
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    if y.sum() < 25:
        continue
    Xa = d[EH].values.astype(float)
    Xb = d[CT].values.astype(float)
    mark = "  (the cell in question)" \
        if oc == TARGET else ""

    print("")
    print("=" * 74)
    print("%s   n=%d ev=%d%s"
          % (oc, len(y), int(y.sum()), mark),
          flush=True)

    # references
    la = [roc_auc_score(
        y, oof_late(Xa, Xb, y, grp, s))
        for s in SEEDS]
    la = np.array(la)
    ea = [roc_auc_score(
        y, oof_early(Xa, Xb, y, grp, s))
        for s in SEEDS]
    ea = np.array(ea)
    print("  %-10s %8s %8s %8s %8s"
          % ("model", "mean", "SD", "min",
             "max"))
    print("  %-10s %8.4f %8.4f %8.4f %8.4f"
          % ("late_wsrc", la.mean(),
             la.std(ddof=1), la.min(),
             la.max()), flush=True)
    print("  %-10s %8.4f %8.4f %8.4f %8.4f"
          % ("early", ea.mean(),
             ea.std(ddof=1), ea.min(),
             ea.max()), flush=True)
    for nm, v in (("late_wsrc", la),
                  ("early", ea)):
        rows.append({
            "outcome": oc, "method": nm,
            "k": np.nan, "mean": v.mean(),
            "sd": v.std(ddof=1),
            "lo": v.min(), "hi": v.max(),
            "n_seeds": len(v)})

    print("")
    print("  PCA SWEEP  (a genuine effect"
          " should not sit at exactly k=10)")
    print("  %-10s %8s %8s %8s %8s %s"
          % ("k", "mean", "SD", "min", "max",
             "vs late"))
    best = None
    for k in KS:
        if k >= Xa.shape[1] + Xb.shape[1]:
            continue
        aa = np.array([
            roc_auc_score(
                y, oof_pca(Xa, Xb, y, grp,
                           k, s))
            for s in SEEDS])
        star = " *" if k == 10 else ""
        print("  %-10d %8.4f %8.4f %8.4f"
              " %8.4f  %+.4f%s"
              % (k, aa.mean(),
                 aa.std(ddof=1), aa.min(),
                 aa.max(),
                 aa.mean() - la.mean(), star),
              flush=True)
        rows.append({
            "outcome": oc,
            "method": "pca%d" % k, "k": k,
            "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "lo": aa.min(), "hi": aa.max(),
            "n_seeds": len(aa)})
        if best is None or aa.mean() > best[1]:
            best = (k, aa.mean())

    # paired bootstrap on the first seed
    p_pca = oof_pca(Xa, Xb, y, grp, 10,
                    SEEDS[0])
    p_late = oof_late(Xa, Xb, y, grp,
                      SEEDS[0])
    g_, lo_, hi_, _ = f.boot_diff(
        y, _rank(p_pca), _rank(p_late), grp)
    sig = "*" if (lo_ > 0 or hi_ < 0) else ""
    print("")
    print("  PAIRED BOOTSTRAP, seed 42")
    print("    pca10 vs late_wsrc  %+.4f"
          " [%+.4f,%+.4f] %s"
          % (g_, lo_, hi_, sig), flush=True)
    rows.append({
        "outcome": oc,
        "method": "pca10_vs_late",
        "k": 10, "mean": g_, "sd": np.nan,
        "lo": lo_, "hi": hi_,
        "n_seeds": 1,
        "sig": int(lo_ > 0 or hi_ < 0)})

    if oc in S99:
        print("")
        print("  AGAINST SCRIPT 99 (3 seeds)")
        s9 = S99[oc]
        p10 = [x for x in rows
               if x["outcome"] == oc
               and x["method"] == "pca10"]
        if p10:
            print("    pca10      3-seed %.4f"
                  "   10-seed %.4f   %+.4f"
                  % (s9["pca10"],
                     p10[0]["mean"],
                     p10[0]["mean"]
                     - s9["pca10"]))
        print("    late_wsrc  3-seed %.4f"
              "   10-seed %.4f   %+.4f"
              % (s9["late"], la.mean(),
                 la.mean() - s9["late"]))

# ---- permutation, a pipeline sanity check ----
print("")
print("=" * 74)
print("PERMUTATION CHECK  (labels shuffled)")
print("  a correct pipeline returns ~0.50;"
      " anything higher means leakage")
d, y = f.labels(D, TARGET)
grp = d["subject_id"].values
Xa = d[EH].values.astype(float)
Xb = d[CT].values.astype(float)
pv = []
for i in range(5):
    rng = np.random.default_rng(500 + i)
    ysh = y.copy()
    rng.shuffle(ysh)
    pv.append(roc_auc_score(
        ysh, oof_pca(Xa, Xb, ysh, grp, 10,
                     42)))
pv = np.array(pv)
print("  pca10 on shuffled labels:"
      " %.4f (SD %.4f)  over 5 draws"
      % (pv.mean(), pv.std(ddof=1)),
      flush=True)
rows.append({"outcome": "PERMUTED",
             "method": "pca10", "k": 10,
             "mean": pv.mean(),
             "sd": pv.std(ddof=1),
             "lo": pv.min(), "hi": pv.max(),
             "n_seeds": len(pv)})

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("TEN-SEED MEANS")
q = r[r["outcome"] != "PERMUTED"]
q = q[~q["method"].str.contains("_vs_")]
print(q.pivot_table(index="method",
                    columns="outcome",
                    values="mean")
      .round(4).to_string())

print("")
print("TEN-SEED STANDARD DEVIATIONS")
print(q.pivot_table(index="method",
                    columns="outcome",
                    values="sd")
      .round(4).to_string())

print("")
print("=" * 74)
print("VERDICT ON THE %s CELL" % TARGET)
s = q[q["outcome"] == TARGET]
p10 = s[s["method"] == "pca10"]
lw = s[s["method"] == "late_wsrc"]
if len(p10) and len(lw):
    m = float(p10["mean"].iloc[0])
    sd = float(p10["sd"].iloc[0])
    lm = float(lw["mean"].iloc[0])
    lsd = float(lw["sd"].iloc[0])
    diff = m - lm
    print("  script 99, 3 seeds:  +0.0209")
    print("  here, 10 seeds:      %+.4f"
          % diff)
    print("  pca10 %.4f (SD %.4f, range"
          " %.4f-%.4f)"
          % (m, sd, float(p10["lo"].iloc[0]),
             float(p10["hi"].iloc[0])))
    print("  late  %.4f (SD %.4f, range"
          " %.4f-%.4f)"
          % (lm, lsd, float(lw["lo"].iloc[0]),
             float(lw["hi"].iloc[0])))
    print("")
    if abs(diff) < 2 * max(sd, lsd):
        print("  -> the margin is smaller than"
              " twice the seed SD, so it is")
        print("     consistent with noise")
    elif diff > 0:
        print("  -> the margin survives ten"
              " seeds and exceeds twice the")
        print("     seed SD, so it is worth"
              " taking seriously")
    else:
        print("  -> the margin reverses on ten"
              " seeds, so the script 99")
        print("     result was a favourable"
              " three-seed draw")

print("")
print("IS k=10 SPECIAL?")
s = q[(q["outcome"] == TARGET)
      & q["method"].str.startswith("pca")]
if len(s):
    print(s[["method", "k", "mean", "sd"]]
          .sort_values("k").round(4)
          .to_string(index=False))
    b = s.loc[s["mean"].idxmax()]
    print("")
    print("  best k here: %d at %.4f"
          % (int(b["k"]), b["mean"]))
    print("  a real effect should be smooth"
          " across neighbouring k; a spike at")
    print("  exactly 10 with dips either side"
          " is the signature of noise")

print("")
print("THE SAME METHOD ON THE OTHER OUTCOMES")
print("  script 99 gave pca10 -0.0054,"
      " +0.0167 and -0.0476 elsewhere on this")
print("  pairing, so there was no consistent"
      " pattern to begin with")
for oc in OUTS:
    s = q[q["outcome"] == oc]
    a = s[s["method"] == "pca10"]
    b = s[s["method"] == "late_wsrc"]
    if len(a) and len(b):
        print("  %-18s pca10 %.4f   late %.4f"
              "   %+.4f"
              % (oc, a["mean"].iloc[0],
                 b["mean"].iloc[0],
                 a["mean"].iloc[0]
                 - b["mean"].iloc[0]))
print("")
print("saved", DEST, r.shape)