"""Learner sweep on the raw and PCA-compressed
EHR+CTPA representations.

L2 logistic regression had been the only head.
PCA components are orthogonal and ordered by
variance, so the collinearity L2 handles is
absent, and uniform shrinkage treats the first
component like the eighth.

LEARNERS
  l2        the incumbent
  plain     unregularised, the collinearity
            control
  ridge_var ridge with shrinkage proportional to
            each component's explained variance
  gb        histogram gradient boosting
  catboost  ordered boosting
  rf        random forest
  tabpfn    a pretrained tabular foundation
            model, which fits nothing at
            prediction time

All four outcomes are run, not only the cell
where PCA helped, since that cell was selected
from sixteen comparisons.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\learner_sweep.csv
  results\\learner_sweep_log.txt
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings(
    "ignore", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore", category=UserWarning)
sys.path.insert(0, "scripts")
import fusion_lib as f


def keep_awake(on=True):
    try:
        import ctypes
        flag = (0x80000000 | 0x00000001
                if on else 0x80000000)
        ctypes.windll.kernel32 \
            .SetThreadExecutionState(flag)
        return True
    except Exception:
        return False


import atexit
if keep_awake(True):
    print("sleep suppressed for this process")
atexit.register(keep_awake, False)

HAVE_CAT = False
try:
    from catboost import CatBoostClassifier
    HAVE_CAT = True
except Exception as exc:
    print("catboost unavailable:",
          repr(exc)[:60])

HAVE_PFN = False
try:
    from tabpfn import TabPFNClassifier
    HAVE_PFN = True
except Exception as exc:
    print("tabpfn unavailable:",
          repr(exc)[:60])

PROC, FIG = f.PROC, f.FIG
SEEDS = [42, 7, 13, 1, 2]
NFOLD = 5
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
PCA_K = 8
CT_LO, CT_HI = -48.0, 24.0
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(PROC,
                    "learner_sweep.csv")
# script 104, ten seeds, L2 head
PREV = {"death_30d_inhosp":
        {"late": 0.8823, "pca8": 0.9014}}


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


def prep_fold(Xtr, Xte):
    Xtr, md = clean_fit(Xtr)
    Xte = clean_apply(Xte, md)
    sc = StandardScaler()
    return sc.fit_transform(Xtr), \
        sc.transform(Xte)


# ---------------- the learners --------------
def learn_l2(Xt, yt, Xe, gt):
    """C selected by inner CV, as in the
    other scripts."""
    best, bc = -1.0, 1.0
    try:
        kk = min(3, max(2, int(yt.sum()) // 10))
        icv = StratifiedGroupKFold(
            n_splits=kk, shuffle=True,
            random_state=42)
        g = (gt if gt is not None
             else np.arange(len(yt)))
        for c in CS:
            q = np.zeros(len(yt))
            for t2, v2 in icv.split(Xt, yt, g):
                m = LogisticRegression(
                    C=c, max_iter=3000)
                m.fit(Xt[t2], yt[t2])
                q[v2] = m.predict_proba(
                    Xt[v2])[:, 1]
            s = roc_auc_score(yt, q)
            if s > best:
                best, bc = s, c
    except Exception:
        bc = 1.0
    m = LogisticRegression(C=bc,
                           max_iter=3000)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def learn_plain(Xt, yt, Xe, gt):
    """Effectively unregularised. On orthogonal
    components there is no collinearity for L2
    to handle, so if this matches l2 the
    shrinkage is doing nothing."""
    m = LogisticRegression(C=1e6,
                           max_iter=5000)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def learn_ridge_var(Xt, yt, Xe, gt,
                    evr=None):
    """Ridge with PER-COMPONENT shrinkage.

    Uniform L2 penalises the first component as
    hard as the eighth, when their contributions
    differ by an order of magnitude. Scaling
    each column by the square root of its
    explained variance makes the effective
    penalty inversely proportional to that
    variance, so the ordering is respected."""
    if evr is None:
        return learn_l2(Xt, yt, Xe, gt)
    w = np.sqrt(np.clip(evr, 1e-9, None))
    w = w / w.max()
    return learn_l2(Xt * w, yt, Xe * w, gt)


def learn_gb(Xt, yt, Xe, gt):
    m = HistGradientBoostingClassifier(
        max_depth=3, max_iter=150,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def learn_cat(Xt, yt, Xe, gt):
    """Ordered boosting, designed to resist the
    small-sample overfitting that sinks ordinary
    trees."""
    m = CatBoostClassifier(
        iterations=300, depth=3,
        learning_rate=0.05, l2_leaf_reg=6.0,
        random_seed=42, verbose=0,
        allow_writing_files=False)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def learn_rf(Xt, yt, Xe, gt):
    m = RandomForestClassifier(
        n_estimators=500, max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=42, n_jobs=-1)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def learn_pfn(Xt, yt, Xe, gt):
    """Pretrained on millions of synthetic
    tabular datasets. Fits NOTHING: the training
    rows are context for a single forward pass,
    so the overfit gap that sank 90 trained
    configurations cannot arise."""
    m = TabPFNClassifier(device="cpu")
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


LEARNERS = [("l2", learn_l2),
            ("plain", learn_plain),
            ("ridge_var", learn_ridge_var),
            ("gb", learn_gb),
            ("rf", learn_rf)]
if HAVE_CAT:
    LEARNERS.append(("catboost", learn_cat))
if HAVE_PFN:
    LEARNERS.append(("tabpfn", learn_pfn))


def oof(X, y, grp, learner, seed, use_pca):
    """Grouped OOF. When use_pca, the PCA is
    fitted inside each training fold and the
    explained-variance ratio is passed through
    for ridge_var."""
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        Xt, Xe = prep_fold(X[tr], X[te])
        evr = None
        if use_pca:
            kk = int(min(PCA_K,
                         Xt.shape[1] - 1,
                         len(tr) - 1))
            pc = PCA(n_components=kk,
                     random_state=42)
            Xt2 = pc.fit_transform(Xt)
            Xe2 = pc.transform(Xe)
            evr = pc.explained_variance_ratio_
            Xt, Xe = Xt2, Xe2
        if learner is learn_ridge_var:
            p[te] = learner(Xt, y[tr], Xe,
                            grp[tr], evr)
        else:
            p[te] = learner(Xt, y[tr], Xe,
                            grp[tr])
    return p


def oof_late(Xa, Xb, y, grp, seed):
    pa = np.zeros(len(y))
    pb = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        At, Ae = prep_fold(Xa[tr], Xa[te])
        Bt, Be = prep_fold(Xb[tr], Xb[te])
        pa[te] = learn_l2(At, y[tr], Ae,
                          grp[tr])
        pb[te] = learn_l2(Bt, y[tr], Be,
                          grp[tr])
    pa, pb = _rank(pa), _rank(pb)
    bs, bv = -1.0, pa
    for w in np.arange(0, 1.001, 0.05):
        v = w * pb + (1 - w) * pa
        a = roc_auc_score(y, v)
        if a > bs:
            bs, bv = a, v
    return bv


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
print("")
print("cohort:", len(D))
print("  EHR %d   CTPA %d   total %d"
      % (len(EH), len(CT), len(EH) + len(CT)))
print("  learners:",
      ", ".join(n for n, _ in LEARNERS),
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
    X = np.column_stack([Xa, Xb])
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())),
          flush=True)

    la = np.array([roc_auc_score(
        y, oof_late(Xa, Xb, y, grp, s))
        for s in SEEDS])
    ref = la.mean()
    print("  late_wsrc (L2 head)  %.4f"
          " (SD %.4f)"
          % (ref, la.std(ddof=1)), flush=True)
    rows.append({
        "outcome": oc, "learner": "late_wsrc",
        "repr": "late", "mean": ref,
        "sd": la.std(ddof=1),
        "vs_late": 0.0})

    for use_pca, rname in ((False, "raw74"),
                           (True,
                            "pca%d" % PCA_K)):
        print("")
        print("  %-10s  %-10s %8s %8s %9s"
              % (rname, "learner", "AUC",
                 "SD", "vs late"))
        for lname, fn in LEARNERS:
            try:
                aa = np.array([
                    roc_auc_score(
                        y, oof(X, y, grp, fn,
                               s, use_pca))
                    for s in SEEDS])
            except Exception as exc:
                print("    %-22s FAILED %s"
                      % (lname,
                         repr(exc)[:45]))
                continue
            print("    %-10s %10s %8.4f"
                  " %8.4f %+9.4f"
                  % ("", lname, aa.mean(),
                     aa.std(ddof=1),
                     aa.mean() - ref),
                  flush=True)
            rows.append({
                "outcome": oc,
                "learner": lname,
                "repr": rname,
                "mean": aa.mean(),
                "sd": aa.std(ddof=1),
                "vs_late": aa.mean() - ref})

    pd.DataFrame(rows).to_csv(DEST,
                              index=False)
    print("")
    print("  %s done in %.1f min  (saved)"
          % (oc, (time.time() - tb) / 60),
          flush=True)

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("AUROC BY LEARNER, RAW 74 FEATURES")
s = r[r["repr"] == "raw74"]
if len(s):
    print(s.pivot_table(index="learner",
                        columns="outcome",
                        values="mean")
          .round(4).to_string())

print("")
print("AUROC BY LEARNER, PCA %d COMPONENTS"
      % PCA_K)
s = r[r["repr"] == "pca%d" % PCA_K]
if len(s):
    print(s.pivot_table(index="learner",
                        columns="outcome",
                        values="mean")
          .round(4).to_string())

print("")
print("GAIN OVER late:wsrc, PCA COMPONENTS")
if len(s):
    print(s.pivot_table(index="learner",
                        columns="outcome",
                        values="vs_late")
          .round(4).to_string())

print("")
print("AVERAGE RANK ACROSS OUTCOMES")
for rp in ("raw74", "pca%d" % PCA_K):
    z = r[r["repr"] == rp]
    if not len(z):
        continue
    rk = z.pivot_table(index="learner",
                       columns="outcome",
                       values="mean").rank(
        ascending=False).mean(axis=1)
    print("")
    print("  " + rp)
    print(rk.sort_values().round(2)
          .to_string())

print("")
print("=" * 74)
print("DOES COMPRESSION HELP EACH LEARNER?")
print("  PCA minus raw, per learner. A learner")
print("  that already handles 74 correlated")
print("  features should gain nothing.")
a = r[r["repr"] == "raw74"][
    ["outcome", "learner", "mean"]]
b = r[r["repr"] == "pca%d" % PCA_K][
    ["outcome", "learner", "mean"]]
if len(a) and len(b):
    m = a.merge(b, on=["outcome", "learner"],
                suffixes=("_raw", "_pca"))
    m["d"] = m["mean_pca"] - m["mean_raw"]
    print(m.pivot_table(index="learner",
                        columns="outcome",
                        values="d")
          .round(4).to_string())
    print("")
    print("  mean effect of compression, by"
          " learner:")
    print(m.groupby("learner")["d"].mean()
          .sort_values(ascending=False)
          .round(4).to_string())

print("")
print("DOES L2 SHRINKAGE MATTER ON"
      " ORTHOGONAL COMPONENTS?")
print("  plain is effectively unregularised."
      " If it matches l2 on PCA columns,")
print("  the shrinkage is doing nothing there.")
z = r[r["repr"] == "pca%d" % PCA_K]
for oc in OUTS:
    q = z[z["outcome"] == oc]
    l2 = q[q["learner"] == "l2"]
    pl = q[q["learner"] == "plain"]
    rv = q[q["learner"] == "ridge_var"]
    if len(l2) and len(pl):
        line = ("  %-18s l2 %.4f   plain %.4f"
                "  (%+.4f)"
                % (oc, l2["mean"].iloc[0],
                   pl["mean"].iloc[0],
                   pl["mean"].iloc[0]
                   - l2["mean"].iloc[0]))
        if len(rv):
            line += ("   ridge_var %.4f"
                     " (%+.4f)"
                     % (rv["mean"].iloc[0],
                        rv["mean"].iloc[0]
                        - l2["mean"].iloc[0]))
        print(line)

print("")
print("BEST LEARNER PER OUTCOME AND"
      " REPRESENTATION")
for oc in OUTS:
    q = r[(r["outcome"] == oc)
          & (r["learner"] != "late_wsrc")]
    if not len(q):
        continue
    b_ = q.loc[q["mean"].idxmax()]
    lt = r[(r["outcome"] == oc)
           & (r["learner"] == "late_wsrc")]
    print("  %-18s %-10s on %-6s %.4f"
          "   late %.4f   %+.4f"
          % (oc, b_["learner"], b_["repr"],
             b_["mean"],
             lt["mean"].iloc[0]
             if len(lt) else np.nan,
             b_["vs_late"]))

print("")
print("AGAINST SCRIPT 104")
for oc, v in PREV.items():
    q = r[(r["outcome"] == oc)
          & (r["repr"] == "pca%d" % PCA_K)
          & (r["learner"] == "l2")]
    if len(q):
        print("  %s   l2 on pca8 here %.4f"
              "   script 104 %.4f   %+.4f"
              % (oc, q["mean"].iloc[0],
                 v["pca8"],
                 q["mean"].iloc[0]
                 - v["pca8"]))
        print("  (script 104 used ten seeds,"
              " this uses five)")
print("")
print("saved", DEST, r.shape)
keep_awake(False)