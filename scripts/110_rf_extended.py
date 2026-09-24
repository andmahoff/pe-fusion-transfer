"""Extended forest tuning: lower mtry and k, Extra
Trees, rotation forest, tree count and sampling.

In script 109 the gain fell steadily as mtry rose,
with 10%, the lowest value tested, best; k = 5,
also the lowest tested, beat 20 and 30. Lower
values of both are added here, including k = 1,
a single canonical direction per view.

ALSO TESTED
  Extra Trees      a single random split point per
                   feature and no bootstrap
  rotation forest  a fresh PCA rotation of random
                   feature subsets for each tree,
                   a practical form of oblique
                   splitting
  tree count and   both fixed in script 109
  sampling

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\rf_extended.csv
  results\\rf_extended_log.txt
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as sla
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss

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

PROC, FIG = f.PROC, f.FIG
SEEDS = [42, 7, 13, 1, 2]
SEEDS10 = [42, 7, 13, 1, 2, 3, 5, 8, 21, 99]
NFOLD = 5
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CCA_REG = 0.1
# absolute counts, since the column total moves
# with k
MTRY = [1, 2, 3, 4, 6, 8]
K_GRID = [1, 2, 3, 5, 10]
LEAF = [5, 10, 25]
NTREE_GRID = [250, 500, 1000, 2000]
MAXS = [None, 0.8, 0.6, 0.4]
DEPTH = 12
CT_LO, CT_HI = -48.0, 24.0
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(PROC,
                    "rf_extended.csv")
# script 109, ten seeds, tuned forest
BEST109 = {"death_30d_inhosp":
           {"late": 0.8823, "rf": 0.9147},
           "death_30d":
           {"late": 0.8822, "rf": 0.9024},
           "composite_30d":
           {"late": 0.8449, "rf": 0.8596},
           "cv_first":
           {"late": 0.8064, "rf": 0.8208}}


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


def build(At, Ae, Bt, Be, k):
    Xt = np.column_stack([At, Bt])
    Xe = np.column_stack([Ae, Be])
    if k < 1:
        return Xt, Xe
    A_, B_ = rcca(At, Bt, CCA_REG, k)
    if A_ is None:
        return Xt, Xe
    ma, mb = At.mean(0), Bt.mean(0)
    return (np.column_stack([
        Xt, (At - ma) @ A_, (Bt - mb) @ B_]),
        np.column_stack([
            Xe, (Ae - ma) @ A_,
            (Be - mb) @ B_]))


# ---------- rotation forest -----------------
class RotationForest:
    """Each tree is fitted on a PCA ROTATION of
    the features, built from a random partition
    into subsets and a bootstrap of the rows.

    This is the practical form of oblique
    splitting: axis-aligned partitions "limit
    the model's ability to capture dependencies
    between dimensions", and oblique forests
    split on "linear combinations of the
    covariates". It generalises cca_aug from
    twenty fixed directions to a fresh rotation
    per tree."""

    def __init__(self, n_estimators=300,
                 n_subsets=3, leaf=5,
                 depth=12, seed=42):
        self.n = n_estimators
        self.ns = n_subsets
        self.leaf = leaf
        self.depth = depth
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        self.rot_, self.tr_ = [], []
        for b in range(self.n):
            cols = rng.permutation(p)
            size = max(2, p // self.ns)
            R = np.zeros((p, p))
            for i in range(0, p, size):
                idx = cols[i:i + size]
                if len(idx) < 2:
                    for j in idx:
                        R[j, j] = 1.0
                    continue
                rows = rng.choice(
                    n, max(10, int(0.75 * n)),
                    replace=True)
                sub = X[np.ix_(rows, idx)]
                try:
                    pc = PCA(
                        n_components=len(idx),
                        random_state=int(
                            rng.integers(1e6)))
                    pc.fit(sub)
                    C = pc.components_.T
                except Exception:
                    C = np.eye(len(idx))
                for a_, j in enumerate(idx):
                    R[j, idx] = C[a_]
            Z = X @ R
            rows = rng.choice(n, n,
                              replace=True)
            t = DecisionTreeClassifier(
                max_depth=self.depth,
                min_samples_leaf=self.leaf,
                class_weight="balanced",
                random_state=int(
                    rng.integers(1e6)))
            t.fit(Z[rows], y[rows])
            self.rot_.append(R)
            self.tr_.append(t)
        return self

    def predict_proba(self, X):
        out = np.zeros(len(X))
        for R, t in zip(self.rot_, self.tr_):
            out += t.predict_proba(
                X @ R)[:, 1]
        out /= len(self.tr_)
        return np.column_stack([1 - out, out])


def make_model(kind, mtry, leaf, ntree,
               maxs, ncol, seed=42):
    mf = int(min(max(1, mtry), ncol))
    if kind == "extra":
        # a single random split point per
        # feature, and no bootstrap
        return ExtraTreesClassifier(
            n_estimators=ntree,
            max_features=mf, max_depth=DEPTH,
            min_samples_leaf=leaf,
            bootstrap=False,
            class_weight=
            "balanced_subsample",
            random_state=seed, n_jobs=-1)
    if kind == "rot":
        return RotationForest(
            n_estimators=min(ntree, 300),
            leaf=leaf, depth=DEPTH, seed=seed)
    return RandomForestClassifier(
        n_estimators=ntree, max_features=mf,
        max_depth=DEPTH,
        min_samples_leaf=leaf,
        max_samples=maxs,
        class_weight="balanced_subsample",
        random_state=seed, n_jobs=-1)


def oof(Xa, Xb, y, grp, seed, kind="rf",
        k=5, mtry=8, leaf=5, ntree=500,
        maxs=None):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        At, Ae = prep_fold(Xa[tr], Xa[te])
        Bt, Be = prep_fold(Xb[tr], Xb[te])
        Zt, Ze = build(At, Ae, Bt, Be, k)
        m = make_model(kind, mtry, leaf,
                       ntree, maxs,
                       Zt.shape[1])
        m.fit(Zt, y[tr])
        p[te] = m.predict_proba(Ze)[:, 1]
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
        for X_, Xt_, tag in ((At, Ae, "a"),
                             (Bt, Be, "b")):
            best, bc = -1.0, 1.0
            for c in CS:
                try:
                    mm = LogisticRegression(
                        C=c, max_iter=3000)
                    mm.fit(X_, y[tr])
                    s = roc_auc_score(
                        y[tr],
                        mm.predict_proba(
                            X_)[:, 1])
                    if s > best:
                        best, bc = s, c
                except Exception:
                    continue
            mm = LogisticRegression(
                C=bc, max_iter=3000)
            mm.fit(X_, y[tr])
            q = mm.predict_proba(Xt_)[:, 1]
            if tag == "a":
                pa[te] = q
            else:
                pb[te] = q
    pa, pb = _rank(pa), _rank(pb)
    bs, bv = -1.0, pa
    for w in np.arange(0, 1.001, 0.05):
        v = w * pb + (1 - w) * pa
        a = roc_auc_score(y, v)
        if a > bs:
            bs, bv = a, v
    return bv


def platt_oof(p, y, grp, seed=42):
    """Script 109 found the forest
    under-confident, slopes 1.33 to 1.67, and Platt
    corrected them to about 0.97 while leaving
    AUROC unchanged."""
    q = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(q / (1 - q)).reshape(-1, 1)
    out = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(x, y, grp):
        try:
            m = LogisticRegression(
                C=1e6, max_iter=2000)
            m.fit(x[tr], y[tr])
            out[te] = m.predict_proba(
                x[te])[:, 1]
        except Exception:
            out[te] = p[te]
    return out


def calib(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    try:
        b = brier_score_loss(y, p)
        x = np.log(p / (1 - p)).reshape(-1, 1)
        m = LogisticRegression(C=1e6,
                               max_iter=2000)
        m.fit(x, y)
        return b, float(m.coef_[0][0])
    except Exception:
        return np.nan, np.nan


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
print("  EHR %d   CTPA %d   raw %d"
      % (len(EH), len(CT), len(EH) + len(CT)))
print("  mtry grid:", MTRY,
      " (script 109's 10%% was about 8)")
print("  k grid:", K_GRID,
      " (5 was the lowest tested before)",
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
    b9 = BEST109.get(oc, {})
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())))
    if b9:
        print("  script 109 tuned %.4f   late"
              " %.4f" % (b9["rf"], b9["late"]),
              flush=True)

    la = np.array([roc_auc_score(
        y, oof_late(Xa, Xb, y, grp, s))
        for s in SEEDS])
    ref = la.mean()
    rows.append({
        "outcome": oc, "test": "ref",
        "kind": "late", "mtry": np.nan,
        "k": np.nan, "leaf": np.nan,
        "ntree": np.nan, "maxs": np.nan,
        "mean": ref, "sd": la.std(ddof=1),
        "vs_late": 0.0})

    # ---- TEST 1: lower mtry and k ----
    print("")
    print("  LOWER mtry x k, gain over late")
    print("  %-6s" % "mtry", end="")
    for kk in K_GRID:
        print(" %9s" % ("k=%d" % kk), end="")
    print("")
    best = None
    for mt in MTRY:
        line = "  %-6d" % mt
        for kk in K_GRID:
            try:
                aa = np.array([
                    roc_auc_score(
                        y, oof(Xa, Xb, y, grp,
                               s, "rf", kk,
                               mt, 5, 500))
                    for s in SEEDS])
            except Exception:
                line += " %9s" % "fail"
                continue
            line += " %+9.4f" % (aa.mean()
                                 - ref)
            rows.append({
                "outcome": oc,
                "test": "mtry_k", "kind": "rf",
                "mtry": mt, "k": kk, "leaf": 5,
                "ntree": 500, "maxs": np.nan,
                "mean": aa.mean(),
                "sd": aa.std(ddof=1),
                "vs_late": aa.mean() - ref})
            if best is None or \
                    aa.mean() > best[0]:
                best = (aa.mean(), mt, kk)
        print(line, flush=True)
    bmt = best[1] if best else 4
    bk = best[2] if best else 5
    if best:
        edge = ("  (at the grid edge)"
                if bmt == MTRY[0]
                or bk == K_GRID[0] else "")
        print("  best mtry=%d k=%d  %.4f%s"
              % (bmt, bk, best[0], edge),
              flush=True)
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)

    # ---- TEST 2: algorithm family ----
    print("")
    print("  ALGORITHM, at mtry=%d k=%d"
          % (bmt, bk))
    print("  extra: a single random split point"
          " per feature, no bootstrap")
    print("  rot:   a fresh PCA rotation per"
          " tree, the practical oblique form")
    print("  %-8s %-6s %8s %8s %9s"
          % ("kind", "leaf", "AUC", "SD",
             "vs late"))
    bestk = None
    for kind in ("rf", "extra", "rot"):
        for lf in LEAF:
            try:
                aa = np.array([
                    roc_auc_score(
                        y, oof(Xa, Xb, y, grp,
                               s, kind, bk,
                               bmt, lf, 500))
                    for s in SEEDS])
            except Exception as exc:
                print("    %-8s %-6d FAILED %s"
                      % (kind, lf,
                         repr(exc)[:35]))
                continue
            print("  %-8s %-6d %8.4f %8.4f"
                  " %+9.4f"
                  % (kind, lf, aa.mean(),
                     aa.std(ddof=1),
                     aa.mean() - ref),
                  flush=True)
            rows.append({
                "outcome": oc,
                "test": "algo", "kind": kind,
                "mtry": bmt, "k": bk,
                "leaf": lf, "ntree": 500,
                "maxs": np.nan,
                "mean": aa.mean(),
                "sd": aa.std(ddof=1),
                "vs_late": aa.mean() - ref})
            if bestk is None or \
                    aa.mean() > bestk[0]:
                bestk = (aa.mean(), kind, lf)
    bkind = bestk[1] if bestk else "rf"
    blf = bestk[2] if bestk else 5
    print("  best %s leaf=%d  %.4f"
          % (bkind, blf,
             bestk[0] if bestk else np.nan),
          flush=True)
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)

    # ---- TEST 3: tree count ----
    print("")
    print("  TREE COUNT, at %s" % bkind)
    for nt in NTREE_GRID:
        try:
            aa = np.array([
                roc_auc_score(
                    y, oof(Xa, Xb, y, grp, s,
                           bkind, bk, bmt,
                           blf, nt))
                for s in SEEDS])
        except Exception:
            continue
        print("    %-6d  %.4f (SD %.4f)"
              "  %+.4f"
              % (nt, aa.mean(),
                 aa.std(ddof=1),
                 aa.mean() - ref), flush=True)
        rows.append({
            "outcome": oc, "test": "ntree",
            "kind": bkind, "mtry": bmt,
            "k": bk, "leaf": blf,
            "ntree": nt, "maxs": np.nan,
            "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_late": aa.mean() - ref})

    # ---- TEST 4: sampling ----
    if bkind == "rf":
        print("")
        print("  SAMPLING FRACTION")
        for ms in MAXS:
            try:
                aa = np.array([
                    roc_auc_score(
                        y, oof(Xa, Xb, y, grp,
                               s, "rf", bk,
                               bmt, blf, 500,
                               ms))
                    for s in SEEDS])
            except Exception:
                continue
            print("    %-6s  %.4f (SD %.4f)"
                  "  %+.4f"
                  % (str(ms), aa.mean(),
                     aa.std(ddof=1),
                     aa.mean() - ref),
                  flush=True)
            rows.append({
                "outcome": oc,
                "test": "maxsamples",
                "kind": "rf", "mtry": bmt,
                "k": bk, "leaf": blf,
                "ntree": 500,
                "maxs": (-1 if ms is None
                         else ms),
                "mean": aa.mean(),
                "sd": aa.std(ddof=1),
                "vs_late": aa.mean() - ref})

    # ---- TEST 5: final, ten seeds ----
    print("")
    print("  FINAL, TEN SEEDS: %s mtry=%d"
          " k=%d leaf=%d"
          % (bkind, bmt, bk, blf))
    aa, br, sl = [], [], []
    bc_, bs_ = [], []
    for s in SEEDS10:
        p = oof(Xa, Xb, y, grp, s, bkind, bk,
                bmt, blf, 1000)
        aa.append(roc_auc_score(y, p))
        b_, s2 = calib(p, y)
        br.append(b_)
        sl.append(s2)
        pc = platt_oof(p, y, grp, s)
        ok = np.isfinite(pc)
        bc_.append(roc_auc_score(y[ok],
                                 pc[ok]))
        b2, s3 = calib(pc[ok], y[ok])
        bs_.append((b2, s3))
    aa = np.array(aa)
    print("    AUROC %.4f (SD %.4f, range"
          " %.4f-%.4f)   %+.4f vs late"
          % (aa.mean(), aa.std(ddof=1),
             aa.min(), aa.max(),
             aa.mean() - ref))
    print("    raw    Brier %.5f  slope %.3f"
          % (np.nanmean(br), np.nanmean(sl)))
    print("    platt  Brier %.5f  slope %.3f"
          "   AUROC %.4f"
          % (np.nanmean([x[0] for x in bs_]),
             np.nanmean([x[1] for x in bs_]),
             float(np.mean(bc_))), flush=True)
    if b9:
        print("    script 109 gave %.4f, so"
              " %+.4f from this round"
              % (b9["rf"], aa.mean()
                 - b9["rf"]))
    rows.append({
        "outcome": oc, "test": "final",
        "kind": bkind, "mtry": bmt, "k": bk,
        "leaf": blf, "ntree": 1000,
        "maxs": np.nan, "mean": aa.mean(),
        "sd": aa.std(ddof=1), "lo": aa.min(),
        "hi": aa.max(),
        "brier": np.nanmean(br),
        "slope": np.nanmean(sl),
        "brier_platt": np.nanmean(
            [x[0] for x in bs_]),
        "slope_platt": np.nanmean(
            [x[1] for x in bs_]),
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
print("LOWER mtry x k, gain over late")
m = r[r["test"] == "mtry_k"]
for oc in OUTS:
    z = m[m["outcome"] == oc]
    if not len(z):
        continue
    print("")
    print("  " + oc)
    print(z.pivot_table(index="mtry",
                        columns="k",
                        values="vs_late")
          .round(4).to_string())

print("")
print("IS THERE AN INTERIOR OPTIMUM NOW?")
print("  script 109 was monotone decreasing in"
      " mtry with 10% (about 8) the lowest")
print("  value tested, so the optimum was never"
      " found")
if len(m):
    z = m.groupby("mtry")["vs_late"].mean()
    print("")
    print("  mean gain by mtry")
    print(z.round(4).to_string())
    bm = z.idxmax()
    print("  best mtry %d%s"
          % (bm, "  (still at the grid edge)"
             if bm == MTRY[0] else ""))
    zk = m.groupby("k")["vs_late"].mean()
    print("")
    print("  mean gain by k")
    print(zk.round(4).to_string())
    bkk = zk.idxmax()
    print("  best k %d%s"
          % (bkk, "  (still at the grid edge)"
             if bkk == K_GRID[0] else ""))
    if bkk == 1:
        print("  k=1 winning would mean a"
              " single canonical direction per")
        print("  view carries the cross-modal"
              " signal, which is a cleaner")
        print("  and more interpretable result"
              " than ten")

print("")
print("ALGORITHM FAMILY")
a = r[r["test"] == "algo"]
if len(a):
    print(a.pivot_table(index="kind",
                        columns="outcome",
                        values="vs_late")
          .round(4).to_string())
    print("")
    z = a.groupby("kind")["mean"].mean()
    print("  mean AUROC by family")
    print(z.round(4).to_string())
    print("")
    print("  extra trees often gives no gain in general use;")
    print("  this data has favoured more randomisation,")
    print("  so both were tested")

print("")
print("TREE COUNT")
t = r[r["test"] == "ntree"]
if len(t):
    print(t.pivot_table(index="ntree",
                        columns="outcome",
                        values="vs_late")
          .round(4).to_string())
    print("")
    print("  usually improves then plateaus;"
          " a flat surface means 500 was fine")

print("")
print("SAMPLING FRACTION")
s = r[r["test"] == "maxsamples"]
if len(s):
    print(s.pivot_table(index="maxs",
                        columns="outcome",
                        values="vs_late")
          .round(4).to_string())
    print("  -1 means no subsampling")

print("")
print("=" * 74)
print("FINAL vs SCRIPT 109")
fz = r[r["test"] == "final"]
print("  %-18s %9s %9s %9s %8s"
      % ("outcome", "script109", "here",
         "gain", "kind"))
for oc in OUTS:
    z = fz[fz["outcome"] == oc]
    b9 = BEST109.get(oc, {})
    if len(z) and b9:
        print("  %-18s %9.4f %9.4f %+9.4f"
              " %8s"
              % (oc, b9["rf"],
                 z["mean"].iloc[0],
                 z["mean"].iloc[0] - b9["rf"],
                 z["kind"].iloc[0]))

print("")
print("THE FINAL SETTINGS AND CALIBRATION")
if len(fz):
    print(fz[["outcome", "kind", "mtry", "k",
              "leaf", "mean", "sd",
              "slope", "slope_platt",
              "vs_late"]].round(4)
          .to_string(index=False))
    print("")
    print("  slope near 1 after Platt is the"
          " target. Script 109 found the")
    print("  forest under-confident at 1.33 to"
          " 1.67, which is the reverse of")
    print("  the usual pattern and follows from"
          " balanced_subsample inflating")
    print("  minority probabilities.")
print("")
print("saved", DEST, r.shape)
keep_awake(False)