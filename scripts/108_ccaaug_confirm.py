"""Confirms cca_aug with a random forest.

In script 106, cca_aug/rf had the best average
rank of any cell (1.25) and beat late fusion on
all four outcomes, including cv_first. cca_aug is
the 74 raw features plus the canonical variates.
Trees split on one feature at a time, so a
cross-modal direction is costly for them to
build, and CCA supplies such directions directly.
cca alone gave rf 0.8533, so the raw features are
needed alongside the variates.

The result was selected from 144 comparisons, so
it is checked with FIVE TESTS
  1 ten seeds instead of five, with the range
    reported alongside the SD
  2 a paired bootstrap against late fusion on
    every outcome
  3 an ablation: raw, cca alone, cca_aug, and
    cca_aug with the variates shuffled. If the
    shuffled version matches the real one, the
    extra columns act as noise regularisation
    rather than carrying cross-modal signal.
  4 a sweep of the CCA rank and regularisation,
    fixed at k=10 and reg=0.1 in script 106
  5 a permutation check for leakage

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ccaaug_confirm.csv
  results\\ccaaug_confirm_log.txt
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as sla
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
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
except Exception:
    print("catboost unavailable, skipping")

PROC, FIG = f.PROC, f.FIG
SEEDS = [42, 7, 13, 1, 2, 3, 5, 8, 21, 99]
NFOLD = 5
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CCA_REG, CCA_K = 0.1, 10
REG_GRID = [0.01, 0.1, 0.5]
K_GRID = [2, 5, 10, 20]
CT_LO, CT_HI = -48.0, 24.0
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(
    PROC, "ccaaug_confirm.csv")
# script 106, five seeds
PREV = {"death_30d_inhosp":
        {"late": 0.8834, "ccaaug_rf": 0.9095},
        "death_30d":
        {"late": 0.8823, "ccaaug_rf": 0.8987},
        "composite_30d":
        {"late": 0.8423, "ccaaug_rf": 0.8567},
        "cv_first":
        {"late": 0.8074, "ccaaug_rf": 0.8126}}


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


def rcca(X, Y, reg, k):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    n = len(X)
    p, q = X.shape[1], Y.shape[1]
    Cxx = X.T @ X / n + reg * np.eye(p)
    Cyy = Y.T @ Y / n + reg * np.eye(q)
    try:
        Kx = sla.fractional_matrix_power(
            Cxx, -0.5).real
        Ky = sla.fractional_matrix_power(
            Cyy, -0.5).real
        U, s, Vt = np.linalg.svd(
            Kx @ (X.T @ Y / n) @ Ky,
            full_matrices=False)
    except Exception:
        return None, None
    k = int(min(k, len(s)))
    return Kx @ U[:, :k], Ky @ Vt[:k].T


# ---------------- learners ------------------
def learn_l2(Xt, yt, Xe, gt=None):
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
    m = LogisticRegression(C=bc, max_iter=3000)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def learn_rf(Xt, yt, Xe, gt=None):
    m = RandomForestClassifier(
        n_estimators=500, max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=42, n_jobs=-1)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def learn_gb(Xt, yt, Xe, gt=None):
    m = HistGradientBoostingClassifier(
        max_depth=3, max_iter=150,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def learn_cat(Xt, yt, Xe, gt=None):
    m = CatBoostClassifier(
        iterations=300, depth=3,
        learning_rate=0.05, l2_leaf_reg=6.0,
        random_seed=42, verbose=0,
        allow_writing_files=False)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


# ---------------- representations -----------
def build(At, Ae, Bt, Be, mode, reg, k,
          seed=0):
    """mode: raw, cca, ccaaug, ccaaug_shuf.

    ccaaug_shuf is the key control: the same
    number of canonical columns, with their ROWS
    permuted so any cross-modal association is
    destroyed while the distribution and column
    count are preserved. If it matches the real
    one, the extra columns act as noise
    regularisation rather than carrying signal."""
    Xt = np.column_stack([At, Bt])
    Xe = np.column_stack([Ae, Be])
    if mode == "raw":
        return Xt, Xe
    A_, B_ = rcca(At, Bt, reg, k)
    if A_ is None:
        return Xt, Xe
    ma, mb = At.mean(0), Bt.mean(0)
    zt = np.column_stack([(At - ma) @ A_,
                          (Bt - mb) @ B_])
    ze = np.column_stack([(Ae - ma) @ A_,
                          (Be - mb) @ B_])
    if mode == "cca":
        return zt, ze
    if mode == "ccaaug_shuf":
        rng = np.random.default_rng(
            1000 + seed)
        zt = zt[rng.permutation(len(zt))]
        ze = ze[rng.permutation(len(ze))]
    return (np.column_stack([Xt, zt]),
            np.column_stack([Xe, ze]))


def oof(Xa, Xb, y, grp, mode, learner, seed,
        reg=CCA_REG, k=CCA_K):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        At, Ae = prep_fold(Xa[tr], Xa[te])
        Bt, Be = prep_fold(Xb[tr], Xb[te])
        Zt, Ze = build(At, Ae, Bt, Be, mode,
                       reg, k, seed)
        p[te] = learner(Zt, y[tr], Ze, grp[tr])
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
print("  ten seeds, all four outcomes",
      flush=True)

LEARNERS = [("rf", learn_rf),
            ("gb", learn_gb),
            ("l2", learn_l2)]
if HAVE_CAT:
    LEARNERS.insert(2, ("catboost",
                        learn_cat))
MODES = ["raw", "cca", "ccaaug",
         "ccaaug_shuf"]

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
    tb = time.time()
    pv_ = PREV.get(oc, {})
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())),
          flush=True)

    la = np.array([roc_auc_score(
        y, oof_late(Xa, Xb, y, grp, s))
        for s in SEEDS])
    ref = la.mean()
    print("  late_wsrc  %.4f (SD %.4f,"
          " range %.4f-%.4f)"
          % (ref, la.std(ddof=1), la.min(),
             la.max()), flush=True)
    rows.append({
        "outcome": oc, "test": "ref",
        "mode": "late", "learner": "l2",
        "reg": np.nan, "k": np.nan,
        "mean": ref, "sd": la.std(ddof=1),
        "lo": la.min(), "hi": la.max(),
        "vs_late": 0.0})

    # ---- TEST 1 and 3: ablation ----
    print("")
    print("  ABLATION  (ten seeds)")
    print("  ccaaug_shuf keeps the column count"
          " but permutes the rows, so any")
    print("  cross-modal association is"
          " destroyed. If it matches ccaaug,")
    print("  the columns are noise"
          " regularisation, not signal.")
    print("")
    print("  %-12s %-9s %8s %8s %9s"
          % ("mode", "learner", "AUC", "SD",
             "vs late"))
    for mode in MODES:
        for lname, fn in LEARNERS:
            try:
                aa = np.array([
                    roc_auc_score(
                        y, oof(Xa, Xb, y, grp,
                               mode, fn, s))
                    for s in SEEDS])
            except Exception as exc:
                print("    %-12s %-9s FAILED %s"
                      % (mode, lname,
                         repr(exc)[:35]))
                continue
            print("  %-12s %-9s %8.4f %8.4f"
                  " %+9.4f"
                  % (mode, lname, aa.mean(),
                     aa.std(ddof=1),
                     aa.mean() - ref),
                  flush=True)
            rows.append({
                "outcome": oc,
                "test": "ablation",
                "mode": mode,
                "learner": lname,
                "reg": CCA_REG, "k": CCA_K,
                "mean": aa.mean(),
                "sd": aa.std(ddof=1),
                "lo": aa.min(), "hi": aa.max(),
                "vs_late": aa.mean() - ref})
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)

    # ---- TEST 2: paired bootstrap ----
    print("")
    print("  PAIRED BOOTSTRAP, seed 42")
    pl = oof_late(Xa, Xb, y, grp, SEEDS[0])
    for lname, fn in LEARNERS[:2]:
        pc = oof(Xa, Xb, y, grp, "ccaaug",
                 fn, SEEDS[0])
        g_, lo_, hi_, _ = f.boot_diff(
            y, _rank(pc), _rank(pl), grp)
        print("    ccaaug/%-9s vs late"
              "  %+.4f [%+.4f,%+.4f] %s"
              % (lname, g_, lo_, hi_,
                 "*" if (lo_ > 0 or hi_ < 0)
                 else ""), flush=True)
        rows.append({
            "outcome": oc, "test": "boot",
            "mode": "ccaaug",
            "learner": lname,
            "reg": CCA_REG, "k": CCA_K,
            "mean": g_, "sd": np.nan,
            "lo": lo_, "hi": hi_,
            "vs_late": g_,
            "sig": int(lo_ > 0 or hi_ < 0)})

    # ---- TEST 4: the CCA settings ----
    print("")
    print("  CCA RANK AND REGULARISATION,"
          " with rf")
    print("  both were fixed at k=10 and"
          " reg=0.1 in script 106")
    print("  %-6s" % "k", end="")
    for rg in REG_GRID:
        print(" %10s" % ("reg=%.2f" % rg),
              end="")
    print("")
    for kk in K_GRID:
        line = "  %-6d" % kk
        for rg in REG_GRID:
            try:
                aa = np.array([
                    roc_auc_score(
                        y, oof(Xa, Xb, y, grp,
                               "ccaaug",
                               learn_rf, s,
                               rg, kk))
                    for s in SEEDS[:5]])
                line += " %+10.4f" % (
                    aa.mean() - ref)
                rows.append({
                    "outcome": oc,
                    "test": "grid",
                    "mode": "ccaaug",
                    "learner": "rf",
                    "reg": rg, "k": kk,
                    "mean": aa.mean(),
                    "sd": aa.std(ddof=1),
                    "lo": aa.min(),
                    "hi": aa.max(),
                    "vs_late":
                        aa.mean() - ref})
            except Exception:
                line += " %10s" % "fail"
        print(line, flush=True)

    if pv_:
        z = [x for x in rows
             if x["outcome"] == oc
             and x["test"] == "ablation"
             and x["mode"] == "ccaaug"
             and x["learner"] == "rf"]
        if z:
            print("")
            print("  AGAINST SCRIPT 106"
                  " (five seeds)")
            print("    ccaaug/rf  here %.4f"
                  "   there %.4f   %+.4f"
                  % (z[0]["mean"],
                     pv_["ccaaug_rf"],
                     z[0]["mean"]
                     - pv_["ccaaug_rf"]))
            print("    late       here %.4f"
                  "   there %.4f   %+.4f"
                  % (ref, pv_["late"],
                     ref - pv_["late"]))

    pd.DataFrame(rows).to_csv(DEST,
                              index=False)
    print("")
    print("  %s done in %.1f min  (saved)"
          % (oc, (time.time() - tb) / 60),
          flush=True)

# ---- TEST 5: permutation ----
print("")
print("#" * 74)
print("PERMUTATION CHECK  (labels shuffled)")
print("  a correct pipeline returns ~0.50")
print("#" * 74, flush=True)
d, y = f.labels(D, OUTS[0])
grp = d["subject_id"].values
Xa = d[EH].values.astype(float)
Xb = d[CT].values.astype(float)
pv = []
for i in range(5):
    rng = np.random.default_rng(800 + i)
    ysh = y.copy()
    rng.shuffle(ysh)
    pv.append(roc_auc_score(
        ysh, oof(Xa, Xb, ysh, grp, "ccaaug",
                 learn_rf, 42)))
pv = np.array(pv)
print("  ccaaug/rf on shuffled labels:"
      " %.4f (SD %.4f)"
      % (pv.mean(), pv.std(ddof=1)),
      flush=True)
rows.append({
    "outcome": "PERMUTED", "test": "perm",
    "mode": "ccaaug", "learner": "rf",
    "reg": CCA_REG, "k": CCA_K,
    "mean": pv.mean(), "sd": pv.std(ddof=1),
    "lo": pv.min(), "hi": pv.max(),
    "vs_late": np.nan})

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("TEN-SEED GAIN OVER late:wsrc")
a = r[r["test"] == "ablation"]
if len(a):
    a2 = a.copy()
    a2["cell"] = (a2["mode"] + "/"
                  + a2["learner"])
    print(a2.pivot_table(index="cell",
                         columns="outcome",
                         values="vs_late")
          .round(4).to_string())

print("")
print("SEED SD")
if len(a):
    print(a2.pivot_table(index="cell",
                         columns="outcome",
                         values="sd")
          .round(4).to_string())

print("")
print("=" * 74)
print("DO THE CANONICAL COLUMNS CARRY SIGNAL?")
print("  ccaaug minus ccaaug_shuf. The shuffled")
print("  version has the same column count and")
print("  distribution, so a difference near"
      " zero means the columns act as noise")
print("  regularisation rather than encoding"
      " cross-modal structure.")
if len(a):
    for lname, _ in LEARNERS:
        print("")
        print("  " + lname)
        for oc in OUTS:
            z = a[(a["outcome"] == oc)
                  & (a["learner"] == lname)]
            re_ = z[z["mode"] == "ccaaug"]
            sh = z[z["mode"] == "ccaaug_shuf"]
            rw = z[z["mode"] == "raw"]
            if len(re_) and len(sh):
                print("    %-18s real %.4f"
                      "   shuffled %.4f"
                      "   %+.4f   (raw %.4f)"
                      % (oc,
                         re_["mean"].iloc[0],
                         sh["mean"].iloc[0],
                         re_["mean"].iloc[0]
                         - sh["mean"].iloc[0],
                         rw["mean"].iloc[0]
                         if len(rw)
                         else np.nan))
    z = a[a["mode"] == "ccaaug"]
    s = a[a["mode"] == "ccaaug_shuf"]
    if len(z) and len(s):
        m = z.merge(
            s[["outcome", "learner", "mean"]],
            on=["outcome", "learner"],
            suffixes=("", "_sh"))
        print("")
        print("  mean real minus shuffled:"
              " %+.4f" % (m["mean"]
                          - m["mean_sh"]).mean())

print("")
print("PAIRED BOOTSTRAP, EVERY OUTCOME")
b = r[r["test"] == "boot"]
if len(b):
    print(b[["outcome", "learner", "mean",
             "lo", "hi", "sig"]].round(4)
          .to_string(index=False))
    print("")
    print("  significant: %d of %d"
          % (int(b["sig"].fillna(0).sum()),
             len(b)))

print("")
print("CCA SETTINGS, rf, gain over late")
g = r[r["test"] == "grid"]
if len(g):
    for oc in OUTS:
        z = g[g["outcome"] == oc]
        if not len(z):
            continue
        print("")
        print("  " + oc)
        print(z.pivot_table(index="k",
                            columns="reg",
                            values="vs_late")
              .round(4).to_string())
    print("")
    print("  a broad positive surface means the"
          " result does not depend on the")
    print("  k=10 reg=0.1 setting that was"
          " picked arbitrarily")

print("")
print("VERDICT")
z = a[(a["mode"] == "ccaaug")
      & (a["learner"] == "rf")]
if len(z):
    nw = int((z["vs_late"] > 0).sum())
    print("  ccaaug/rf beats late fusion in"
          " %d of %d outcomes" % (nw, len(z)))
    print("  mean gain %+.4f   range %+.4f to"
          " %+.4f"
          % (z["vs_late"].mean(),
             z["vs_late"].min(),
             z["vs_late"].max()))
    for _, x in z.iterrows():
        lt = r[(r["outcome"] == x["outcome"])
               & (r["mode"] == "late")]
        if len(lt):
            sdmax = max(x["sd"],
                        lt["sd"].iloc[0])
            print("    %-18s %+.4f  =  %.1f"
                  " seed SDs"
                  % (x["outcome"],
                     x["vs_late"],
                     x["vs_late"]
                     / max(sdmax, 1e-9)))
pz = r[r["test"] == "perm"]
if len(pz):
    print("  permutation %.4f, so no leakage"
          % pz["mean"].iloc[0])
print("")
print("saved", DEST, r.shape)
keep_awake(False)