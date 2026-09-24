"""Three changes in one run: the CTPA C grid,
the cv_first leaf range, and the
missing-modality comparison at a swept leaf.

Scripts 118 to 127 used leaf=1. Script 117 swept
leaf and found 10 to 25 best, and script 128
measured the cost of leaf 1 at 0.0066 to 0.0245.

A  CTPA C GRID. When ens replaced L2 on the CTPA
   modality in script 128, its internal L2 used
   the five-value default C grid rather than
   CT_CS = logspace(-4, 4, 10). The wide grid is
   restored and the two are compared.
B  cv_first LEAF. Script 128's leaf sweep for
   cv_first rose up to 50, the top of the grid,
   so 75 and 100 are added for cv_first and
   composite_30d.
C  MISSING MODALITY. Script 118's variants all
   ran at leaf 1 and found ehr_ecg ahead of
   nan_ind. Rerun at the swept leaf:
     complete  the 1,636 with a CTPA
     ehr_ecg   all 3,501, CTPA ignored
     nan_ind   all 3,501, CTPA left missing plus
               an indicator; no imputation
     masked    all 3,501, a fixed fraction of
               trees built without the CTPA block
   nan_ind is compared with complete on the same
   complete cases, so the only difference is
   whether the 1,865 CTPA-free patients were
   available for training.

Parallel across seeds. Timing probe first.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\leaf_fix_missing.csv
  results\\leaf_fix_missing_log.txt
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import linalg as sla
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
RAW = os.path.join("data", "raw", "ecg")
SEEDS = [42, 7, 13, 1, 2]
NFOLD = 5
NJOBS = max(1, min(10, (os.cpu_count() or 4)
                   - 2))
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CT_CS = list(np.logspace(-4, 4, 10))
K, REG, NTREE = 10, 0.5, 600
DRAW3 = (14, 8, 16)
DRAW2 = (20, 8)
MASK_FREE = 0.5
# leaf grids: extended only where script 128 hit
# an edge
LEAF_EXT = {"cv_first": [25, 50, 75, 100],
            "composite_30d": [10, 25, 50,
                              75, 100]}
LEAF_MISS = [1, 10, 25]
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(
    PROC, "leaf_fix_missing.csv")
# script 128, leaf swept, three modalities
S128 = {"death_30d_inhosp": 0.9153,
        "death_30d": 0.9000,
        "composite_30d": 0.8610,
        "cv_first": 0.8179}
# script 118, all variants at leaf 1
S118 = {"death_30d":
        {"ehr_ecg": 0.8916, "nan_ind": 0.8622,
         "complete": 0.8857},
        "death_30d_inhosp":
        {"ehr_ecg": 0.9051, "nan_ind": 0.8877,
         "complete": 0.9037},
        "composite_30d":
        {"ehr_ecg": 0.8443, "nan_ind": 0.8240,
         "complete": 0.8462},
        "cv_first":
        {"ehr_ecg": 0.7865, "nan_ind": 0.7892,
         "complete": 0.7854}}

DROP_EXACT = ("rec", "t_flag", "vm_noise",
              "vm_base", "vm_rpeak", "net_i",
              "net_avf", "apen", "tp_win_ms",
              "t_win_ms", "t_peak_ms",
              "tp_fallback", "qrs_hit_limit",
              "p_hit_limit", "t_peak_edge",
              "t_edge_leads", "n_rr_dropped",
              "any_af", "tpe_spread_ms",
              "tpe_ok")
DROP_PREFIX = ("tpk_ms_",)
PERM_ONLY = ("tpe_ms_mean", "tpe_ms_worst",
             "tpe_qt_mean", "tpe_qt_worst")
NEG_BAD = ("s_wave_i", "q_wave_iii", "st_min",
           "t_v1", "t_v2", "t_v3", "t_v4",
           "t_v5", "t_v6", "t_wave_iii")
ANG = ("qrs_axis", "t_axis", "p_axis")


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


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


def prep_keepnan(Xtr, Xte, htr, hte):
    """Standardise on the rows that have the
    modality, then restore NaN elsewhere. No
    imputation anywhere."""
    A = np.asarray(Xtr, dtype=float).copy()
    B = np.asarray(Xte, dtype=float).copy()
    sub = A[htr]
    if len(sub) < 10:
        return (np.full_like(A, np.nan),
                np.full_like(B, np.nan))
    mu = np.nanmean(sub, axis=0)
    sd = np.nanstd(sub, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9),
                  sd, 1.0)
    A = (A - mu) / sd
    B = (B - mu) / sd
    A[~htr] = np.nan
    B[~hte] = np.nan
    return A, B


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


def ccaaug(Bt, Be, k, reg):
    m = len(Bt)
    pt = [[b] for b in Bt]
    pe = [[b] for b in Be]
    for i in range(m):
        for jj in range(i + 1, m):
            W1, W2 = rcca(Bt[i], Bt[jj],
                          reg, k)
            if W1 is None:
                continue
            mi = Bt[i].mean(0)
            mj = Bt[jj].mean(0)
            pt[i].append((Bt[i] - mi) @ W1)
            pe[i].append((Be[i] - mi) @ W1)
            pt[jj].append((Bt[jj] - mj) @ W2)
            pe[jj].append((Be[jj] - mj) @ W2)
    ft = [x for grp in pt for x in grp]
    fe = [x for grp in pe for x in grp]
    sizes = [sum(x.shape[1] for x in grp)
             for grp in pt]
    return (np.column_stack(ft),
            np.column_stack(fe),
            tuple(np.cumsum(sizes)[:-1]))


def ccaaug_nan(At, Ae, Bt, Be, Ct, Ce,
               htr, hte, k, reg):
    """Three blocks with the CTPA block left
    MISSING for patients without one, plus an
    availability indicator on the EHR side.

    The EHR-ECG variates are fitted on all rows;
    the CTPA variates only on rows that HAVE a
    CTPA, then applied to everyone with NaN
    restored where the modality is absent."""
    pa, pb, pc = [At], [Bt], [Ct]
    qa, qb, qc = [Ae], [Be], [Ce]
    W1, W2 = rcca(At, Bt, reg, k)
    if W1 is not None:
        ma, mb = At.mean(0), Bt.mean(0)
        pa.append((At - ma) @ W1)
        qa.append((Ae - ma) @ W1)
        pb.append((Bt - mb) @ W2)
        qb.append((Be - mb) @ W2)
    if htr.sum() > 50:
        A2 = np.nan_to_num(At[htr])
        C2 = np.nan_to_num(Ct[htr])
        V1, V2 = rcca(A2, C2, reg, k)
        if V1 is not None:
            mu, mv = A2.mean(0), C2.mean(0)
            za = (np.nan_to_num(At) - mu) @ V1
            ze = (np.nan_to_num(Ae) - mu) @ V1
            zc = (np.nan_to_num(Ct) - mv) @ V2
            zd = (np.nan_to_num(Ce) - mv) @ V2
            za[~htr] = np.nan
            ze[~hte] = np.nan
            zc[~htr] = np.nan
            zd[~hte] = np.nan
            pa.append(za)
            qa.append(ze)
            pc.append(zc)
            qc.append(zd)
    pa.append(htr.astype(float).reshape(-1, 1))
    qa.append(hte.astype(float).reshape(-1, 1))
    Zt = np.column_stack(pa + pb + pc)
    Ze = np.column_stack(qa + qb + qc)
    b0 = sum(x.shape[1] for x in pa)
    b1 = b0 + sum(x.shape[1] for x in pb)
    return Zt, Ze, (b0, b1)


def L_l2(Xt, yt, Xe, grid=None):
    gg = grid if grid else CS
    best, bc = -1.0, 1.0
    for c in gg:
        try:
            m = LogisticRegression(
                C=c, max_iter=3000)
            m.fit(Xt, yt)
            a = roc_auc_score(
                yt, m.predict_proba(Xt)[:, 1])
            if a > best:
                best, bc = a, c
        except Exception:
            continue
    m = LogisticRegression(C=bc, max_iter=3000)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1], bc


def L_ens(Xt, yt, Xe, grid=None):
    """ens: rank average of L2 and a forest. The
    grid argument is now PASSED THROUGH to the L2
    component, which script 128 failed to do for
    the CTPA block."""
    a, bc = L_l2(Xt, yt, Xe, grid)
    m = RandomForestClassifier(
        n_estimators=500, max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42, n_jobs=1)
    m.fit(Xt, yt)
    b = m.predict_proba(Xe)[:, 1]
    return 0.5 * (_rank(a) + _rank(b)), bc


class BlockForest:
    """mask=True builds a fixed fraction of trees
    WITHOUT the CTPA block, so every patient has
    voters. Script 118's version had none, and
    the without-CTPA patients scored exactly
    0.5000."""

    def __init__(self, n_estimators=600,
                 draws=(14, 8, 16),
                 bounds=(38, 312), leaf=25,
                 depth=None, mask=False,
                 mask_free=MASK_FREE, seed=42):
        self.n = n_estimators
        self.draws = draws
        self.bounds = bounds
        self.leaf = leaf
        self.depth = depth
        self.mask = mask
        self.mask_free = mask_free
        self.seed = seed

    def fit(self, X, y, have=None):
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        cuts = ([0] + [int(min(b, p))
                       for b in self.bounds]
                + [p])
        blk = [np.arange(cuts[i], cuts[i + 1])
               for i in range(len(cuts) - 1)]
        self.sel_, self.tr_, self.uses_ = \
            [], [], []
        nfree = (int(round(self.mask_free
                           * self.n))
                 if self.mask else 0)
        for ti in range(self.n):
            use_ct = not (self.mask
                          and ti < nfree)
            parts = []
            for bi, (ix, w) in enumerate(
                    zip(blk, self.draws)):
                if len(ix) == 0 or w <= 0:
                    continue
                if bi == 2 and not use_ct:
                    continue
                k = int(min(max(1, w), len(ix)))
                parts.append(rng.choice(
                    ix, k, replace=False))
            if not parts:
                continue
            cols = np.concatenate(parts)
            uses_ct = (use_ct
                       and len(blk) > 2
                       and len(blk[2]) > 0
                       and self.draws[2] > 0)
            rows = rng.choice(n, n,
                              replace=True)
            if (self.mask and uses_ct
                    and have is not None):
                rows = rows[have[rows]]
                if (len(rows) < 30
                        or y[rows].sum() < 3):
                    continue
            t = DecisionTreeClassifier(
                max_depth=self.depth,
                min_samples_leaf=self.leaf,
                class_weight="balanced",
                random_state=int(
                    rng.integers(1e6)))
            t.fit(X[np.ix_(rows, cols)],
                  y[rows])
            self.sel_.append(cols)
            self.tr_.append(t)
            self.uses_.append(uses_ct)
        return self

    def predict_proba(self, X, have=None):
        num = np.zeros(len(X))
        den = np.zeros(len(X))
        for cols, t, uc in zip(
                self.sel_, self.tr_,
                self.uses_):
            q = t.predict_proba(
                X[:, cols])[:, 1]
            if (self.mask and uc
                    and have is not None):
                num[have] += q[have]
                den[have] += 1.0
            else:
                num += q
                den += 1.0
        out = np.where(
            den > 0,
            num / np.where(den > 0, den, 1.0),
            0.5)
        return np.column_stack([1 - out, out])


def platt_oof(p, y, grp, seed=42):
    ok = np.isfinite(p)
    x = _logit(np.clip(p, 1e-6,
                       1 - 1e-6)).reshape(-1, 1)
    out = np.full(len(y), np.nan)
    idx = np.where(ok)[0]
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(x[idx], y[idx],
                           grp[idx]):
        try:
            m = LogisticRegression(
                C=1e6, max_iter=2000)
            m.fit(x[idx][tr], y[idx][tr])
            out[idx[te]] = m.predict_proba(
                x[idx][te])[:, 1]
        except Exception:
            out[idx[te]] = p[idx[te]]
    return out


def calib(p, y, sub=None):
    ok = np.isfinite(p)
    if sub is not None:
        ok = ok & sub
    if ok.sum() < 30:
        return np.nan, np.nan
    q = np.clip(p[ok], 1e-6, 1 - 1e-6)
    try:
        b = brier_score_loss(y[ok], q)
        x = _logit(q).reshape(-1, 1)
        m = LogisticRegression(C=1e6,
                               max_iter=2000)
        m.fit(x, y[ok])
        return b, float(m.coef_[0][0])
    except Exception:
        return np.nan, np.nan


def auc_on(p, y, sub=None):
    ok = np.isfinite(p)
    if sub is not None:
        ok = ok & sub
    if ok.sum() < 30 or y[ok].sum() < 5:
        return np.nan
    return roc_auc_score(y[ok], p[ok])


def par(fn, seeds=SEEDS):
    return Parallel(n_jobs=NJOBS,
                    backend="loky")(
        delayed(fn)(s) for s in seeds)


# ---------------- cohort --------------------
rec = pd.read_csv(os.path.join(
    PROC, "exp0_logits_record.csv"))
eidx = pd.read_csv(os.path.join(
    RAW, "ecg_record_index.csv"))
eidx["rec"] = eidx["rec"].astype(str).apply(
    lambda s: s if s.startswith("files/")
    else "files/" + s.replace("\\", "/")
    .strip("/"))
LG = [c for c in rec.columns
      if c.startswith("scp_")]
adm = pd.read_csv(os.path.join(
    FIG, "index_admission_times.csv"))
adm["admittime"] = pd.to_datetime(
    adm["admittime"], errors="coerce")
E = eidx.merge(rec, on="rec", how="inner")
E["t"] = pd.to_datetime(
    E["ecg_charttime"], errors="coerce")
E = E.merge(adm[["hadm_id", "admittime"]],
            on="hadm_id", how="left")
E["h"] = ((E["t"] - E["admittime"])
          .dt.total_seconds() / 3600.0)
E = E[(E["h"] >= ECG_LO)
      & (E["h"] <= ECG_HI)].copy()
ECGA = E.groupby(["subject_id", "hadm_id"],
                 as_index=False)[LG].mean()
keep_rec = set(E["rec"])

m12 = pd.read_csv(os.path.join(
    PROC, "ecg_derived_v12_record.csv"))
m12["rec"] = m12["rec"].astype(str).apply(
    lambda s: s if s.startswith("files/")
    else "files/" + s.replace("\\", "/")
    .strip("/"))
m12 = m12[m12["rec"].isin(keep_rec)]
cols = [c for c in m12.columns
        if c not in DROP_EXACT
        and not c.startswith(DROP_PREFIX)
        and pd.api.types.is_numeric_dtype(
            m12[c])]
j = eidx[["subject_id", "hadm_id",
          "rec"]].merge(
    m12[["rec"] + cols], on="rec",
    how="inner")
g = j.groupby(["subject_id", "hadm_id"])
pos = [c for c in cols
       if not c.startswith(NEG_BAD)
       and c not in ANG]
neg = [c for c in cols
       if c.startswith(NEG_BAD)]
parts = [g[cols].mean().add_suffix("_mean")]
if pos:
    parts.append(g[pos].max()
                 .add_suffix("_worst"))
if neg:
    parts.append(g[neg].min()
                 .add_suffix("_worst"))
MA = pd.concat(parts, axis=1).reset_index()
for c in ANG:
    if c not in j.columns:
        continue
    rad = np.radians(j[c])
    tmp = j[["subject_id", "hadm_id"]].copy()
    tmp["s"], tmp["c"] = np.sin(rad), np.cos(rad)
    gg = tmp.groupby(["subject_id",
                      "hadm_id"]).mean()
    cm = np.degrees(np.arctan2(
        gg["s"], gg["c"])).rename(
        c + "_circ").reset_index()
    MA = MA.merge(cm, on=["subject_id",
                          "hadm_id"],
                  how="left")
    MA = MA.drop(columns=[c + "_mean",
                          c + "_worst"],
                 errors="ignore")
MC = [c for c in MA.columns
      if c not in ("subject_id", "hadm_id")]
cov = MA[MC].notna().mean()
MC = [c for c in MC if cov[c] >= MINCOV
      and c not in PERM_ONLY]

lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
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

# LEFT join so the CTPA-free patients survive
D = ECGA.merge(
    MA[["hadm_id"] + MC], on="hadm_id",
    how="left").merge(
    lab, on=["subject_id", "hadm_id"],
    how="inner").merge(
    mim_ehr[["hadm_id"] + EH], on="hadm_id",
    how="inner").merge(
    mm[["hadm_id"] + CT], on="hadm_id",
    how="left")
D["has_ctpa"] = D[CT].notna().all(
    axis=1).astype(int)
ECGC = LG + MC
BLK = {"ehr": EH, "ecg": ECGC, "ctpa": CT}
MODS = ["ehr", "ecg", "ctpa"]
print("")
print("cohort:", len(D))
print("  with CTPA %d   without %d"
      % (int(D["has_ctpa"].sum()),
         int((1 - D["has_ctpa"]).sum())))
print("  EHR %d   ECG %d   CTPA %d"
      % (len(EH), len(ECGC), len(CT)))
print("")
print("  A: CTPA C grid restored to"
      " logspace(-4,4,10); script 128 used the")
print("     five-value default by accident")
print("  B: leaf extended to 75 and 100 for"
      " cv_first and composite_30d")
print("  C: missing-modality variants at leaf",
      LEAF_MISS, "rather than 1", flush=True)

# ---------- TIMING PROBE ----------
print("")
print("TIMING PROBE")
n_full = len(D)
Xp = np.random.rand(n_full, 400)
yp = (np.random.rand(n_full) > 0.90) \
    .astype(int)
hp = np.random.rand(n_full) > 0.53
t = time.time()
BlockForest(n_estimators=NTREE, leaf=25,
            bounds=(38, 312),
            seed=42).fit(Xp, yp)
t_blk = time.time() - t
t = time.time()
L_ens(np.random.rand(n_full, len(CT)), yp,
      np.random.rand(100, len(CT)), CT_CS)
t_ens = time.time() - t
nA = 2 * len(OUTS)
nB = sum(len(v) for v in LEAF_EXT.values())
nC = len(LEAF_MISS) * 4 * len(OUTS)
per = t_blk
stA = (nA * 3 * t_ens * NFOLD * len(SEEDS)
       / 60.0)
stB = (nB * per * NFOLD * len(SEEDS) / 60.0)
stC = (nC * per * NFOLD * len(SEEDS) / 60.0)
print("  block forest %d trees: %5.1f s"
      % (NTREE, t_blk))
print("  ens on the CTPA block: %5.1f s"
      % t_ens)
print("")
print("  stage A %.0f min, B %.0f min, C %.0f"
      " min, serial" % (stA, stB, stC))
print("  with %d-way parallelism about %.0f"
      " min" % (NJOBS,
                (stA + stB + stC) / NJOBS))
print("  Ctrl+C now if too long. Results save"
      " after every stage.")
time.sleep(5)

rows = []
t0 = time.time()

# ================= STAGE A ==================
print("")
print("#" * 74)
print("STAGE A: THE CTPA C GRID")
print("  script 128 called L_ens without the")
print("  grid, so its L2 used the 5-value")
print("  default. CTPA is the modality whose")
print("  optimal C sat outside that range.")
print("#" * 74, flush=True)

cc = D[D.has_ctpa == 1].reset_index(drop=True)
for oc in OUTS:
    if oc not in cc.columns:
        continue
    d, y = f.labels(cc, oc)
    grp = d["subject_id"].values
    if y.sum() < 25:
        continue
    X = d[CT].values.astype(float)
    print("")
    print("  %s   n=%d ev=%d"
          % (oc, len(y), int(y.sum())))
    for tag, gd in (("narrow", CS),
                    ("wide", CT_CS)):
        def one(s, gd=gd):
            p = np.zeros(len(y))
            cs = []
            cv = StratifiedGroupKFold(
                n_splits=NFOLD, shuffle=True,
                random_state=s)
            for tr, te in cv.split(
                    np.zeros((len(y), 1)), y,
                    grp):
                Xt, Xe = prep_fold(X[tr],
                                   X[te])
                q, bc = L_ens(Xt, y[tr], Xe,
                              gd)
                p[te] = q
                cs.append(bc)
            return p, cs
        try:
            res = par(one)
        except Exception as exc:
            print("    %-7s FAILED %s"
                  % (tag, repr(exc)[:40]))
            continue
        aa = np.array([roc_auc_score(y, r[0])
                       for r in res])
        allc = [c for r in res for c in r[1]]
        print("    %-7s %.4f (SD %.4f)   C"
              " chosen: %s"
              % (tag, aa.mean(),
                 aa.std(ddof=1),
                 ", ".join("%.0e" % c for c
                           in sorted(set(
                               allc)))),
              flush=True)
        rows.append({
            "stage": "A", "outcome": oc,
            "name": tag, "mean": aa.mean(),
            "sd": aa.std(ddof=1)})
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)

# ================= STAGE B ==================
print("")
print("#" * 74)
print("STAGE B: LEAF EXTENDED PAST THE EDGE")
print("  script 128's cv_first sweep rose")
print("  throughout: 0.7933, 0.7976, 0.8067,")
print("  0.8135, 0.8179 for leaf 1 to 50, with")
print("  50 the top of the grid")
print("#" * 74, flush=True)

for oc, leaves in LEAF_EXT.items():
    if oc not in cc.columns:
        continue
    d, y = f.labels(cc, oc)
    grp = d["subject_id"].values
    if y.sum() < 25:
        continue
    Xd = {k: d[v].values.astype(float)
          for k, v in BLK.items()}
    print("")
    print("  %s   n=%d ev=%d   script 128 best"
          " %.4f"
          % (oc, len(y), int(y.sum()),
             S128.get(oc, np.nan)))
    print("    %-6s %9s %9s" % ("leaf", "AUC",
                                "SD"))
    best = None
    for lf in leaves:
        def one(s, lf=lf):
            p = np.zeros(len(y))
            cv = StratifiedGroupKFold(
                n_splits=NFOLD, shuffle=True,
                random_state=s)
            for tr, te in cv.split(
                    np.zeros((len(y), 1)), y,
                    grp):
                Bt, Be = [], []
                for m_ in MODS:
                    a, b = prep_fold(
                        Xd[m_][tr], Xd[m_][te])
                    Bt.append(a)
                    Be.append(b)
                Zt, Ze, bd = ccaaug(Bt, Be, K,
                                    REG)
                bf = BlockForest(
                    n_estimators=NTREE,
                    draws=DRAW3, bounds=bd,
                    leaf=lf, depth=None,
                    seed=s)
                bf.fit(Zt, y[tr])
                p[te] = bf.predict_proba(
                    Ze)[:, 1]
            return p
        try:
            ps = par(one)
        except Exception:
            continue
        aa = np.array([roc_auc_score(y, p)
                       for p in ps])
        print("    %-6d %9.4f %9.4f"
              % (lf, aa.mean(),
                 aa.std(ddof=1)), flush=True)
        rows.append({
            "stage": "B", "outcome": oc,
            "leaf": lf, "mean": aa.mean(),
            "sd": aa.std(ddof=1)})
        if best is None or aa.mean() > best[0]:
            best = (aa.mean(), lf)
    if best:
        edge = ("   (still at the grid edge)"
                if best[1] == leaves[-1]
                else "")
        print("    best leaf=%d  %.4f  %+.4f vs"
              " script 128%s"
              % (best[1], best[0],
                 best[0] - S128.get(oc, np.nan),
                 edge), flush=True)
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)

# ================= STAGE C ==================
print("")
print("#" * 74)
print("STAGE C: MISSING MODALITY AT A CORRECT"
      " LEAF")
print("  script 118 ran every variant at leaf 1 and")
print("  concluded ehr_ecg beats nan_ind by")
print("  -0.0294, -0.0203, -0.0174, with")
print("  cv_first the exception at +0.0027.")
print("  If the ordering flips here, the")
print("  deployment conclusion flips with it.")
print("#" * 74, flush=True)

for oc in OUTS:
    if oc not in D.columns:
        continue
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    have = d["has_ctpa"].values == 1
    if y.sum() < 25:
        continue
    Xd = {k: d[v].values.astype(float)
          for k, v in BLK.items()}
    pv = S118.get(oc, {})
    tb = time.time()
    print("")
    print("  %s   n=%d ev=%d   with CTPA %d"
          " (ev %d)"
          % (oc, len(y), int(y.sum()),
             int(have.sum()),
             int(y[have].sum())))
    print("    script 118 (leaf 1): ehr_ecg"
          " %.4f, nan_ind %.4f, complete %.4f"
          % (pv.get("ehr_ecg", np.nan),
             pv.get("nan_ind", np.nan),
             pv.get("complete", np.nan)),
          flush=True)

    def arm(s, lf, which):
        p = np.full(len(y), np.nan)
        keep = (np.where(have)[0]
                if which == "complete"
                else np.arange(len(y)))
        yk, gk = y[keep], grp[keep]
        hk = have[keep]
        cv = StratifiedGroupKFold(
            n_splits=NFOLD, shuffle=True,
            random_state=s)
        for tr, te in cv.split(
                np.zeros((len(yk), 1)), yk,
                gk):
            Bt, Be = [], []
            for m_ in MODS:
                a, b = prep_fold(
                    Xd[m_][keep][tr],
                    Xd[m_][keep][te])
                Bt.append(a)
                Be.append(b)
            if which == "ehr_ecg":
                Zt, Ze, bd = ccaaug(
                    Bt[:2], Be[:2], K, REG)
                dr = DRAW2
                msk = False
            elif which == "complete":
                Zt, Ze, bd = ccaaug(Bt, Be, K,
                                    REG)
                dr = DRAW3
                msk = False
            else:
                Ct, Ce = prep_keepnan(
                    Xd["ctpa"][keep][tr],
                    Xd["ctpa"][keep][te],
                    hk[tr], hk[te])
                Zt, Ze, bd = ccaaug_nan(
                    Bt[0], Be[0], Bt[1], Be[1],
                    Ct, Ce, hk[tr], hk[te],
                    K, REG)
                dr = DRAW3
                msk = (which == "masked")
            bf = BlockForest(
                n_estimators=NTREE, draws=dr,
                bounds=bd, leaf=lf,
                depth=None, mask=msk, seed=s)
            bf.fit(Zt, yk[tr],
                   have=hk[tr] if msk else None)
            p[keep[te]] = bf.predict_proba(
                Ze, have=hk[te] if msk
                else None)[:, 1]
        return p

    print("")
    print("    %-6s %-10s %9s %9s %9s"
          % ("leaf", "arm", "AUC all",
             "AUC cc", "AUC no-ct"))
    store = {}
    for lf in LEAF_MISS:
        for which in ("complete", "ehr_ecg",
                      "nan_ind", "masked"):
            try:
                ps = par(lambda s, lf=lf,
                         which=which:
                         arm(s, lf, which))
            except Exception as exc:
                print("    %-6d %-10s FAILED %s"
                      % (lf, which,
                         repr(exc)[:35]))
                continue
            aa = np.nanmean([auc_on(p, y)
                             for p in ps])
            ccv = np.nanmean([auc_on(p, y, have)
                              for p in ps])
            nn = np.nanmean([auc_on(p, y, ~have)
                             for p in ps])
            store[(lf, which)] = ps[0]
            print("    %-6d %-10s %9.4f %9.4f"
                  " %9.4f"
                  % (lf, which, aa, ccv, nn),
                  flush=True)
            rows.append({
                "stage": "C", "outcome": oc,
                "leaf": lf, "name": which,
                "mean": aa, "mean_cc": ccv,
                "mean_noct": nn})
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)

    # the decisive comparisons
    print("")
    print("    THE COMPARISONS THAT MATTER")
    for lf in LEAF_MISS:
        a = store.get((lf, "nan_ind"))
        b = store.get((lf, "ehr_ecg"))
        c = store.get((lf, "complete"))
        if a is None or b is None:
            continue
        d1 = auc_on(a, y) - auc_on(b, y)
        line = ("    leaf %-3d  nan_ind minus"
                " ehr_ecg (full cohort) %+.4f"
                % (lf, d1))
        if c is not None:
            ok = (np.isfinite(a)
                  & np.isfinite(c) & have)
            if ok.sum() > 50:
                gn, lo, hi, _ = f.boot_diff(
                    y[ok], _rank(a[ok]),
                    _rank(c[ok]), grp[ok])
                line += ("\n              nan_ind"
                         " minus complete (same"
                         " patients) %+.4f"
                         " [%+.4f,%+.4f] %s"
                         % (gn, lo, hi,
                            "*" if (lo > 0
                                    or hi < 0)
                            else ""))
                rows.append({
                    "stage": "C2",
                    "outcome": oc, "leaf": lf,
                    "name": "nan_vs_complete",
                    "mean": gn, "lo": lo,
                    "hi": hi,
                    "sig": int(lo > 0
                               or hi < 0)})
        print(line, flush=True)

    # masked fix check and calibration
    b25 = store.get((25, "masked"))
    if b25 is not None:
        print("")
        print("    masked without-CTPA AUROC"
              " %.4f  (script 118 gave exactly"
              % auc_on(b25, y, ~have))
        print("    0.5000 because those patients"
              " received no votes)")
    n25 = store.get((25, "nan_ind"))
    if n25 is not None:
        b0, s0 = calib(n25, y, have)
        b1, s1 = calib(n25, y, ~have)
        pc = platt_oof(n25, y, grp)
        b2, s2 = calib(pc, y, have)
        b3, s3 = calib(pc, y, ~have)
        print("")
        print("    CALIBRATION BY AVAILABILITY")
        print("      with CTPA    slope %.3f ->"
              " %.3f" % (s0, s2))
        print("      without CTPA slope %.3f ->"
              " %.3f" % (s1, s3), flush=True)
        rows.append({
            "stage": "C3", "outcome": oc,
            "name": "calib", "slope": s0,
            "slope_platt": s2,
            "slope_noct": s1,
            "slope_noct_platt": s3})

    pd.DataFrame(rows).to_csv(DEST,
                              index=False)
    print("")
    print("    %s done in %.1f min  (saved)"
          % (oc, (time.time() - tb) / 60),
          flush=True)

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))

print("")
print("=" * 74)
print("STAGE A: DOES THE WIDE C GRID MATTER?")
a_ = r[r["stage"] == "A"]
if len(a_):
    print(a_.pivot_table(index="name",
                         columns="outcome",
                         values="mean")
          .round(4).to_string())
    w = a_[a_["name"] == "wide"]
    n = a_[a_["name"] == "narrow"]
    if len(w) and len(n):
        m = w.merge(n, on="outcome",
                    suffixes=("_w", "_n"))
        m["d"] = m["mean_w"] - m["mean_n"]
        print("")
        print("  wide minus narrow:")
        for _, x in m.iterrows():
            print("    %-18s %+.4f"
                  % (x["outcome"], x["d"]))
        print("  mean %+.4f" % m["d"].mean())

print("")
print("STAGE B: LEAF PAST 50")
b_ = r[r["stage"] == "B"]
if len(b_):
    print(b_.pivot_table(index="leaf",
                         columns="outcome",
                         values="mean")
          .round(4).to_string())
    print("")
    for oc in b_["outcome"].unique():
        z = b_[b_["outcome"] == oc]
        x = z.loc[z["mean"].idxmax()]
        print("  %-18s best leaf %d  %.4f"
              "   %+.4f vs script 128"
              % (oc, x["leaf"], x["mean"],
                 x["mean"] - S128.get(oc,
                                      np.nan)))

print("")
print("=" * 74)
print("STAGE C: MISSING MODALITY BY LEAF")
c_ = r[r["stage"] == "C"]
if len(c_):
    for lf in LEAF_MISS:
        z = c_[c_["leaf"] == lf]
        if not len(z):
            continue
        print("")
        print("  leaf %d, full-cohort AUROC"
              % lf)
        print(z.pivot_table(index="name",
                            columns="outcome",
                            values="mean")
              .round(4).to_string())

print("")
print("DOES THE ORDERING FLIP AT A CORRECT"
      " LEAF?")
print("  script 118 at leaf 1: ehr_ecg beat"
      " nan_ind by -0.0294, -0.0203, -0.0174,")
print("  with cv_first the only exception")
if len(c_):
    for oc in OUTS:
        for lf in LEAF_MISS:
            a = c_[(c_["outcome"] == oc)
                   & (c_["leaf"] == lf)
                   & (c_["name"] == "nan_ind")]
            b = c_[(c_["outcome"] == oc)
                   & (c_["leaf"] == lf)
                   & (c_["name"] == "ehr_ecg")]
            if len(a) and len(b):
                d1 = (a["mean"].iloc[0]
                      - b["mean"].iloc[0])
                print("  %-18s leaf %-3d"
                      " nan_ind minus ehr_ecg"
                      " %+.4f %s"
                      % (oc, lf, d1,
                         "(reversed)"
                         if d1 > 0 else ""))

print("")
print("nan_ind vs complete, SAME PATIENTS")
print("  the only difference is whether the"
      " 1,865 CTPA-free patients were")
print("  available for training")
c2 = r[r["stage"] == "C2"]
if len(c2):
    print(c2[["outcome", "leaf", "mean", "lo",
              "hi", "sig"]].round(4)
          .to_string(index=False))
    print("")
    print("  significant: %d of %d"
          % (int(c2["sig"].fillna(0).sum()),
             len(c2)))

print("")
print("CALIBRATION BY AVAILABILITY")
c3 = r[r["stage"] == "C3"]
if len(c3):
    print(c3[["outcome", "slope",
              "slope_platt", "slope_noct",
              "slope_noct_platt"]].round(3)
          .to_string(index=False))
print("")
print("saved", DEST, r.shape)
keep_awake(False)