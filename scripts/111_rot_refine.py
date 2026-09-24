"""Rotation forest refinement.

For a rotation forest the tuned parameter is the
subset size for each PCA rotation, not features
per split, so it is named SUBSET here and passed
explicitly. In script 110 the rotation forest
used its default n_subsets=3 whatever mtry was
printed. Small subsets mean many PCA fits per
tree, so the grid is 2, 4, 8, 16 and 32.

Stage 1 uses 100 trees and two seeds to select a
region; the ten-seed final at 300 trees gives the
reportable number.

ALSO TESTED
  leaf sizes 1, 2 and 3, below the lowest tested
    in script 110
  k of 15 and 20, beyond the edge of script 110
  depth, fixed at 12 in script 110
  a compromise test: one shared setting across
    outcomes against the per-outcome optima
  a gap-rule check, since the rule was fitted on
    weighted rank averaging and a learner that
    sees both modalities jointly may not follow
    it

Run in venv (analysis). About an hour.

OUTPUT FILES
  data\\processed\\rot_refine.csv
  results\\rot_refine_log.txt
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats as st
from scipy import linalg as sla
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
GRID_SEEDS = [42, 7]
SEEDS10 = [42, 7, 13, 1, 2, 3, 5, 8, 21, 99]
NFOLD = 5
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CCA_REG = 0.1
# subset size for each PCA rotation. Small
# values mean many rotations per tree and are
# expensive.
SUBSET = [2, 4, 8, 16, 32]
K_GRID = [1, 2, 3, 5, 10, 15, 20]
LEAF = [1, 2, 3, 5, 10, 25]
DEPTH = [6, 12, None]
NTREE_GRID, NTREE_FINAL = 100, 300
CT_LO, CT_HI = -48.0, 24.0
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(PROC, "rot_refine.csv")
# script 110, ten seeds
PREV = {"death_30d_inhosp":
        {"late": 0.8823, "best": 0.9182},
        "death_30d":
        {"late": 0.8822, "best": 0.9048},
        "composite_30d":
        {"late": 0.8449, "best": 0.8614},
        "cv_first":
        {"late": 0.8064, "best": 0.8340}}


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


class RotationForest:
    """Each tree is fitted on a PCA ROTATION of a
    random partition of the features.

    `subset` is the number of FEATURES PER
    ROTATION. The previous version floored it at
    2, which made subset 1 and 2 identical; here
    a subset of 1 is handled explicitly as an
    identity column so every value is distinct."""

    def __init__(self, n_estimators=100,
                 subset=8, leaf=5, depth=12,
                 seed=42):
        self.n = n_estimators
        self.subset = subset
        self.leaf = leaf
        self.depth = depth
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        size = int(min(max(1, self.subset), p))
        self.rot_, self.tr_ = [], []
        for _ in range(self.n):
            cols = rng.permutation(p)
            R = np.zeros((p, p))
            for i in range(0, p, size):
                idx = cols[i:i + size]
                if len(idx) < 2:
                    # a single feature keeps its
                    # own axis, so subset=1 is a
                    # genuine setting rather than
                    # a duplicate of subset=2
                    for j in idx:
                        R[j, j] = 1.0
                    continue
                rows = rng.choice(
                    n, max(10, int(0.75 * n)),
                    replace=True)
                try:
                    pc = PCA(
                        n_components=len(idx),
                        random_state=int(
                            rng.integers(1e6)))
                    pc.fit(X[np.ix_(rows, idx)])
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
            out += t.predict_proba(X @ R)[:, 1]
        out /= len(self.tr_)
        return np.column_stack([1 - out, out])


def oof_rot(Xa, Xb, y, grp, seed, k=3,
            subset=8, leaf=5, depth=12,
            ntree=NTREE_GRID):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        At, Ae = prep_fold(Xa[tr], Xa[te])
        Bt, Be = prep_fold(Xb[tr], Xb[te])
        Zt, Ze = build(At, Ae, Bt, Be, k)
        m = RotationForest(ntree, subset, leaf,
                           depth, seed)
        m.fit(Zt, y[tr])
        p[te] = m.predict_proba(Ze)[:, 1]
    return p


def oof_single(X, y, grp, seed):
    """One modality with an L2 head, so the gap
    can be measured."""
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        Xt, Xe = prep_fold(X[tr], X[te])
        best, bc = -1.0, 1.0
        for c in CS:
            try:
                m = LogisticRegression(
                    C=c, max_iter=3000)
                m.fit(Xt, y[tr])
                s = roc_auc_score(
                    y[tr],
                    m.predict_proba(Xt)[:, 1])
                if s > best:
                    best, bc = s, c
            except Exception:
                continue
        m = LogisticRegression(C=bc,
                               max_iter=3000)
        m.fit(Xt, y[tr])
        p[te] = m.predict_proba(Xe)[:, 1]
    return p


def late_from(pa, pb, y):
    pa, pb = _rank(pa), _rank(pb)
    bs, bv = -1.0, pa
    for w in np.arange(0, 1.001, 0.05):
        v = w * pb + (1 - w) * pa
        a = roc_auc_score(y, v)
        if a > bs:
            bs, bv = a, v
    return bv


def platt_oof(p, y, grp, seed=42):
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
print("  EHR %d   CTPA %d" % (len(EH),
                              len(CT)))
print("  subset sizes:", SUBSET,
      " (features per PCA rotation)")
print("  grid: %d trees, %d seeds; final: %d"
      " trees, %d seeds"
      % (NTREE_GRID, len(GRID_SEEDS),
         NTREE_FINAL, len(SEEDS10)))
print("  NOTE: EHR + CTPA only. The improved"
      " ECG modality is not here.", flush=True)

rows, gap_rows = [], []
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
    pv = PREV.get(oc, {})
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())))
    if pv:
        print("  script 110 best %.4f,"
              " late %.4f"
              % (pv["best"], pv["late"]),
              flush=True)

    pa = _rank(oof_single(Xa, y, grp, 42))
    pb = _rank(oof_single(Xb, y, grp, 42))
    a_e = roc_auc_score(y, pa)
    a_c = roc_auc_score(y, pb)
    ref = roc_auc_score(y, late_from(pa, pb, y))
    gap = abs(a_e - a_c)
    print("  ehr %.4f   ctpa %.4f   gap %.4f"
          "   late %.4f"
          % (a_e, a_c, gap, ref))
    print("  the gap rule predicts %+.4f for"
          " LATE fusion"
          % (0.036 - 0.243 * gap), flush=True)
    rows.append({
        "outcome": oc, "test": "ref",
        "subset": np.nan, "k": np.nan,
        "leaf": np.nan, "depth": np.nan,
        "mean": ref, "sd": np.nan,
        "vs_late": 0.0})

    # ---- STAGE 1: subset size x k ----
    print("")
    print("  SUBSET SIZE x k, gain over late")
    print("  %-7s" % "subset", end="")
    for kk in K_GRID:
        print(" %8s" % ("k=%d" % kk), end="")
    print("")
    best = None
    for sb in SUBSET:
        line = "  %-7d" % sb
        for kk in K_GRID:
            try:
                aa = np.array([
                    roc_auc_score(
                        y, oof_rot(
                            Xa, Xb, y, grp, s,
                            kk, sb, 5, 12))
                    for s in GRID_SEEDS])
            except Exception:
                line += " %8s" % "fail"
                continue
            line += " %+8.4f" % (aa.mean()
                                 - ref)
            rows.append({
                "outcome": oc,
                "test": "subset_k",
                "subset": sb, "k": kk,
                "leaf": 5, "depth": 12,
                "mean": aa.mean(),
                "sd": (aa.std(ddof=1)
                       if len(aa) > 1
                       else np.nan),
                "vs_late": aa.mean() - ref})
            if best is None or \
                    aa.mean() > best[0]:
                best = (aa.mean(), sb, kk)
        print(line, flush=True)
    bsb = best[1] if best else 8
    bk = best[2] if best else 3
    if best:
        e1 = bsb in (SUBSET[0], SUBSET[-1])
        e2 = bk in (K_GRID[0], K_GRID[-1])
        print("  best subset=%d k=%d  %.4f%s"
              % (bsb, bk, best[0],
                 "   (at a grid edge)"
                 if (e1 or e2) else ""),
              flush=True)
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)

    # ---- STAGE 2: leaf x depth ----
    print("")
    print("  LEAF x DEPTH at subset=%d k=%d"
          % (bsb, bk))
    print("  three of four outcomes chose"
          " leaf 5, the lowest tested before")
    print("  %-6s" % "leaf", end="")
    for dp in DEPTH:
        print(" %8s" % ("d=%s" % dp), end="")
    print("")
    best2 = None
    for lf in LEAF:
        line = "  %-6d" % lf
        for dp in DEPTH:
            try:
                aa = np.array([
                    roc_auc_score(
                        y, oof_rot(
                            Xa, Xb, y, grp, s,
                            bk, bsb, lf, dp))
                    for s in GRID_SEEDS])
            except Exception:
                line += " %8s" % "fail"
                continue
            line += " %+8.4f" % (aa.mean()
                                 - ref)
            rows.append({
                "outcome": oc,
                "test": "leaf_depth",
                "subset": bsb, "k": bk,
                "leaf": lf,
                "depth": (-1 if dp is None
                          else dp),
                "mean": aa.mean(),
                "sd": (aa.std(ddof=1)
                       if len(aa) > 1
                       else np.nan),
                "vs_late": aa.mean() - ref})
            if best2 is None or \
                    aa.mean() > best2[0]:
                best2 = (aa.mean(), lf, dp)
        print(line, flush=True)
    blf = best2[1] if best2 else 5
    bdp = best2[2] if best2 else 12
    print("  best leaf=%d depth=%s  %.4f%s"
          % (blf, bdp,
             best2[0] if best2 else np.nan,
             "   (leaf at the grid edge)"
             if blf == LEAF[0] else ""),
          flush=True)
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)

    # ---- STAGE 3: final, ten seeds ----
    print("")
    print("  PER-OUTCOME OPTIMUM, ten seeds,"
          " %d trees" % NTREE_FINAL)
    aa, br, sl, pl = [], [], [], []
    for s in SEEDS10:
        p = oof_rot(Xa, Xb, y, grp, s, bk,
                    bsb, blf, bdp,
                    NTREE_FINAL)
        aa.append(roc_auc_score(y, p))
        b_, s_ = calib(p, y)
        br.append(b_)
        sl.append(s_)
        pc = platt_oof(p, y, grp, s)
        ok = np.isfinite(pc)
        pl.append(calib(pc[ok], y[ok]))
    aa = np.array(aa)
    print("    %.4f (SD %.4f, range"
          " %.4f-%.4f)   %+.4f vs late"
          % (aa.mean(), aa.std(ddof=1),
             aa.min(), aa.max(),
             aa.mean() - ref))
    print("    slope %.3f -> %.3f after Platt"
          % (np.nanmean(sl),
             np.nanmean([x[1] for x in pl])),
          flush=True)
    rows.append({
        "outcome": oc, "test": "per_outcome",
        "subset": bsb, "k": bk, "leaf": blf,
        "depth": (-1 if bdp is None else bdp),
        "mean": aa.mean(),
        "sd": aa.std(ddof=1), "lo": aa.min(),
        "hi": aa.max(),
        "brier": np.nanmean(br),
        "slope": np.nanmean(sl),
        "slope_platt": np.nanmean(
            [x[1] for x in pl]),
        "vs_late": aa.mean() - ref})
    gap_rows.append({
        "outcome": oc, "gap": gap,
        "auc_ehr": a_e, "auc_ctpa": a_c,
        "late": ref, "rot": aa.mean(),
        "late_gain": ref - max(a_e, a_c),
        "rot_gain": aa.mean() - max(a_e, a_c),
        "rule_pred": 0.036 - 0.243 * gap})

    pd.DataFrame(rows).to_csv(DEST,
                              index=False)
    print("")
    print("  %s done in %.1f min  (saved)"
          % (oc, (time.time() - tb) / 60),
          flush=True)

# ---- STAGE 4: the compromise test ----
print("")
print("#" * 74)
print("THE COMPROMISE TEST")
print("  four different optima from a grid at"
      " two seeds each, on 76 to 219 events,")
print("  may be noise or signal. Fitting ONE"
      " setting on all four separates them.")
print("#" * 74, flush=True)

m = pd.DataFrame(rows)
g = m[m["test"] == "subset_k"]
cm_sb = int(g.groupby("subset")["vs_late"]
            .mean().idxmax())
cm_k = int(g.groupby("k")["vs_late"]
           .mean().idxmax())
ld = m[m["test"] == "leaf_depth"]
cm_lf = int(ld.groupby("leaf")["vs_late"]
            .mean().idxmax())
cm_dp_ = ld.groupby("depth")["vs_late"] \
    .mean().idxmax()
cm_dp = None if cm_dp_ == -1 else int(cm_dp_)
print("")
print("  compromise: subset=%d k=%d leaf=%d"
      " depth=%s" % (cm_sb, cm_k, cm_lf,
                     cm_dp), flush=True)

for oc in OUTS:
    if oc not in D.columns:
        continue
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    if y.sum() < 25:
        continue
    Xa = d[EH].values.astype(float)
    Xb = d[CT].values.astype(float)
    aa = np.array([roc_auc_score(
        y, oof_rot(Xa, Xb, y, grp, s, cm_k,
                   cm_sb, cm_lf, cm_dp,
                   NTREE_FINAL))
        for s in SEEDS10])
    po = m[(m["outcome"] == oc)
           & (m["test"] == "per_outcome")]
    if len(po):
        d_ = aa.mean() - po["mean"].iloc[0]
        sd = max(aa.std(ddof=1),
                 po["sd"].iloc[0])
        print("  %-18s compromise %.4f"
              "   per-outcome %.4f   %+.4f"
              "  =  %.1f seed SDs"
              % (oc, aa.mean(),
                 po["mean"].iloc[0], d_,
                 abs(d_) / max(sd, 1e-9)),
              flush=True)
        rows.append({
            "outcome": oc,
            "test": "compromise",
            "subset": cm_sb, "k": cm_k,
            "leaf": cm_lf,
            "depth": (-1 if cm_dp is None
                      else cm_dp),
            "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_late": np.nan,
            "vs_per_outcome": d_})

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)
gp = pd.DataFrame(gap_rows)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("SUBSET SIZE x k, gain over late")
for oc in OUTS:
    z = r[(r["test"] == "subset_k")
          & (r["outcome"] == oc)]
    if not len(z):
        continue
    print("")
    print("  " + oc)
    print(z.pivot_table(index="subset",
                        columns="k",
                        values="vs_late")
          .round(4).to_string())

print("")
print("MARGINAL OPTIMA")
z = r[r["test"] == "subset_k"]
if len(z):
    print("  by subset size")
    print(z.groupby("subset")["vs_late"].mean()
          .round(4).to_string())
    print("  by k")
    print(z.groupby("k")["vs_late"].mean()
          .round(4).to_string())
z = r[r["test"] == "leaf_depth"]
if len(z):
    print("  by leaf")
    print(z.groupby("leaf")["vs_late"].mean()
          .round(4).to_string())
    print("  by depth  (-1 is unrestricted)")
    print(z.groupby("depth")["vs_late"].mean()
          .round(4).to_string())

print("")
print("=" * 74)
print("COMPROMISE vs PER-OUTCOME")
c = r[r["test"] == "compromise"]
p_ = r[r["test"] == "per_outcome"]
if len(c) and len(p_):
    mm2 = p_.merge(
        c[["outcome", "mean",
           "vs_per_outcome"]],
        on="outcome", suffixes=("_po", "_cm"))
    print(mm2[["outcome", "subset", "k",
               "leaf", "mean_po", "mean_cm",
               "vs_per_outcome"]].round(4)
          .to_string(index=False))
    md = mm2["vs_per_outcome"].abs().mean()
    print("")
    print("  mean absolute loss from using one"
          " setting: %.4f" % md)
    if md < 0.005:
        print("  -> the per-outcome differences"
              " were noise. Report the single")
        print("     compromise setting.")
    else:
        print("  -> the differences are real."
              " Some outcomes genuinely want")
        print("     different settings, which"
              " is itself a finding.")

print("")
print("=" * 74)
print("DOES THE GAP RULE HOLD FOR ROTATION"
      " FOREST?")
print("  the rule was fitted on weighted rank"
      " averaging, where each modality")
print("  yields a prediction and the"
      " combination is a weighted sum.")
print("  Rotation forest never forms a"
      " per-modality prediction.")
if len(gp):
    print("")
    print(gp.round(4).to_string(index=False))
    print("")
    print("  %-18s %9s %9s %9s"
          % ("outcome", "predicted", "late",
             "rot"))
    for _, x in gp.iterrows():
        print("  %-18s %+9.4f %+9.4f %+9.4f"
              % (x["outcome"], x["rule_pred"],
                 x["late_gain"],
                 x["rot_gain"]))
    if len(gp) > 2 and np.std(gp["gap"]) > 1e-9:
        for nm, col in (("late", "late_gain"),
                        ("rot", "rot_gain")):
            rr, pp = st.pearsonr(gp["gap"],
                                 gp[col])
            print("")
            print("  gap vs %s gain:"
                  " r = %+.3f (p = %.3f)"
                  % (nm, rr, pp))
        err_l = (gp["late_gain"]
                 - gp["rule_pred"]).abs().mean()
        err_r = (gp["rot_gain"]
                 - gp["rule_pred"]).abs().mean()
        print("")
        print("  mean absolute error of the"
              " rule: late %.4f   rot %.4f"
              % (err_l, err_r))
        if err_r > 2 * err_l:
            print("  -> the rule governs late"
                  " fusion specifically. A")
            print("     learner that sees both"
                  " modalities jointly is not")
            print("     bound by it, which"
                  " scopes the rule rather than")
            print("     refuting it.")

print("")
print("AGAINST SCRIPT 110")
if len(p_):
    for oc in OUTS:
        z = p_[p_["outcome"] == oc]
        pv = PREV.get(oc, {})
        if len(z) and pv:
            print("  %-18s here %.4f   there"
                  " %.4f   %+.4f"
                  % (oc, z["mean"].iloc[0],
                     pv["best"],
                     z["mean"].iloc[0]
                     - pv["best"]))
    print("")
    print("  script 110's rotation forest used"
          " the default subset size of 3")
    print("  regardless of its printed mtry,"
          " so these are not strictly")
    print("  comparable settings.")
print("")
print("  NEXT: none of this includes the"
      " improved ECG modality.")
print("")
print("saved", DEST, r.shape)
keep_awake(False)