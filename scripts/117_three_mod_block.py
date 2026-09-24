"""Three-modality block forest: EHR + ECG +
CTPA.

Settings carried from the pairwise models:
  EHR+CTPA (scripts 104 to 112): cca_aug as the
    representation, low max_features
  EHR+ECG (scripts 113 to 116): block-stratified
    sampling, with the strong EHR block drawing
    most of its columns and the weak ECG block
    few; CCA reg=0.5; 600 trees

Where the pairings disagreed, the setting is
swept here rather than inherited:
  leaf   5 to 25 on EHR+CTPA, 1 on EHR+ECG
  depth  6 on EHR+CTPA, unrestricted on EHR+ECG
  CCA k  3 to 10 on EHR+CTPA, 20 on EHR+ECG. With
         three blocks there are three pairwise
         variate sets, so k costs 6k columns.

Stage B2 sweeps max_features within the block
draw, which neither pairing tested. Stage E
compares rotation against no rotation at matched
max_features; in script 110 the rotation forest
evaluated every feature per split while the
random forest used 8.

Requiring a CTPA report cuts the cohort to about
1,636 admissions with 83 to 219 events.

Parallel across seeds. Results are saved after
every stage.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\three_mod_block.csv
  results\\three_mod_block_log.txt
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
SEEDS10 = [42, 7, 13, 1, 2, 3, 5, 8, 21, 99]
NFOLD = 5
NJOBS = max(1, min(10, (os.cpu_count() or 4)
                   - 2))
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CT_CS = list(np.logspace(-4, 4, 10))
# inherited starting point from script 116
N_EHR0, N_ECG0, N_CT0 = 28, 8, 16
LEAF0, DEPTH0 = 1, None
K0, REG0, NTREE0 = 20, 0.5, 600
MFEAT0 = None
# stage grids
A_EHR = [14, 20, 28, 38]
A_ECG = [4, 8, 16]
A_CT = [8, 16, 24, 46]
LEAF = [1, 2, 5, 10, 25]
DEPTH = [6, 12, None]
# fraction of the drawn columns; None is
# sklearn's sqrt default
MFEATS = [None, 0.25, 0.50, 1.00]
CCA_KS = [5, 10, 20]
CCA_REGS = [0.1, 0.5]
NTREES = [300, 600, 1200]
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(
    PROC, "three_mod_block.csv")
BEST_C = {"cv_first": 1e-4,
          "composite_30d": 1e-3,
          "death_30d": 1e-3,
          "death_30d_inhosp": 3e-4}
# script 88, late fusion, three modalities
S88 = {"death_30d": 0.8764,
       "composite_30d": 0.8404,
       "death_30d_inhosp": 0.8801,
       "cv_first": np.nan}

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


def _pca_block(S):
    S = S - S.mean(0)
    n = max(len(S) - 1, 1)
    try:
        w, V = np.linalg.eigh(S.T @ S / n)
        return V[:, ::-1]
    except Exception:
        return np.eye(S.shape[1])


class BlockForest3:
    """Each tree draws columns separately from
    THREE blocks: n_a from EHR-side, n_b from
    ECG-side, n_c from CTPA-side.

    mfeat is max_features WITHIN the drawn
    columns, as a fraction. None keeps sklearn's
    sqrt default, which on a 52-column draw is 7.
    Neither pairing ever varied this once block
    sampling existed, and mtry mattered
    enormously on EHR+CTPA before it did.

    rotate applies a PCA rotation to the drawn
    columns, matching what script 110 did on
    EHR+CTPA. It is tested ONCE at matched
    mfeat, because script 110's rotation forest
    had no max_features set while the random
    forest it beat used 8, so that comparison
    was not like for like."""

    def __init__(self, n_estimators=600,
                 n_a=28, n_b=8, n_c=16,
                 bounds=(38, 312), leaf=1,
                 depth=None, mfeat=None,
                 rotate=False, size=3,
                 boot=0.75, seed=42):
        self.n = n_estimators
        self.n_a = n_a
        self.n_b = n_b
        self.n_c = n_c
        self.bounds = bounds
        self.leaf = leaf
        self.depth = depth
        self.mfeat = mfeat
        self.rotate = rotate
        self.size = size
        self.boot = boot
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        b0 = int(min(self.bounds[0], p))
        b1 = int(min(self.bounds[1], p))
        blk = [np.arange(0, b0),
               np.arange(b0, b1),
               np.arange(b1, p)]
        want = [self.n_a, self.n_b, self.n_c]
        nboot = max(10, int(round(
            self.boot * n)))
        self.sel_, self.rot_, self.tr_ = \
            [], [], []
        for _ in range(self.n):
            parts = []
            for ix, w in zip(blk, want):
                if len(ix) == 0:
                    continue
                k = int(min(max(1, w), len(ix)))
                parts.append(rng.choice(
                    ix, k, replace=False))
            cols = np.concatenate(parts)
            Z = X[:, cols]
            bl = []
            if self.rotate:
                order = rng.permutation(
                    len(cols))
                for i in range(0, len(cols),
                               self.size):
                    idx = order[i:i + self.size]
                    if len(idx) < 2:
                        continue
                    rw = rng.choice(
                        n, nboot, replace=True)
                    bl.append((idx, _pca_block(
                        Z[np.ix_(rw, idx)])))
                Z2 = Z.copy()
                for idx, C in bl:
                    Z2[:, idx] = Z[:, idx] @ C
                Z = Z2
            mf = None
            if self.mfeat is not None:
                mf = int(min(max(
                    1, round(self.mfeat
                             * len(cols))),
                    len(cols)))
            rows = rng.choice(n, n,
                              replace=True)
            t = DecisionTreeClassifier(
                max_depth=self.depth,
                min_samples_leaf=self.leaf,
                max_features=mf,
                class_weight="balanced",
                random_state=int(
                    rng.integers(1e6)))
            t.fit(Z[rows], y[rows])
            self.sel_.append(cols)
            self.rot_.append(bl)
            self.tr_.append(t)
        return self

    def predict_proba(self, X):
        out = np.zeros(len(X))
        for cols, bl, t in zip(
                self.sel_, self.rot_,
                self.tr_):
            Z = X[:, cols]
            if bl:
                Z2 = Z.copy()
                for idx, C in bl:
                    Z2[:, idx] = Z[:, idx] @ C
                Z = Z2
            out += t.predict_proba(Z)[:, 1]
        out /= len(self.tr_)
        return np.column_stack([1 - out, out])


def learn_l2(Xt, yt, Xe, gt=None, C=None,
             grid=None):
    if C is not None and grid is None:
        m = LogisticRegression(C=C,
                               max_iter=3000)
        m.fit(Xt, yt)
        return m.predict_proba(Xe)[:, 1]
    gg = grid if grid else CS
    best, bc = -1.0, 1.0
    try:
        kk = min(3, max(2, int(yt.sum()) // 10))
        icv = StratifiedGroupKFold(
            n_splits=kk, shuffle=True,
            random_state=42)
        g = (gt if gt is not None
             else np.arange(len(yt)))
        for c in gg:
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


def rep3(At, Ae, Bt, Be, Ct, Ce, k, reg):
    """Three raw blocks, each followed by its
    canonical variates against the other two.

    PAIRWISE variates for all three pairs, so
    cross-modal directions exist for EHR-ECG,
    EHR-CTPA and ECG-CTPA. The two-modality
    shuffle control showed these carry genuine
    cross-modal association rather than acting
    as noise regularisation.

    Layout, so block boundaries are known:
      [EHR | EHR-ECG var | EHR-CTPA var]
      [ECG | ECG-EHR var | ECG-CTPA var]
      [CTPA | CTPA-EHR var | CTPA-ECG var]"""
    pa, pb, pc = [At], [Bt], [Ct]
    qa, qb, qc = [Ae], [Be], [Ce]
    for (U, V, u2, v2, ia, ib) in (
            (At, Bt, Ae, Be, 0, 1),
            (At, Ct, Ae, Ce, 0, 2),
            (Bt, Ct, Be, Ce, 1, 2)):
        W1, W2 = rcca(U, V, reg, k)
        if W1 is None:
            continue
        mu, mv = U.mean(0), V.mean(0)
        tgt_t = {0: pa, 1: pb, 2: pc}
        tgt_e = {0: qa, 1: qb, 2: qc}
        tgt_t[ia].append((U - mu) @ W1)
        tgt_e[ia].append((u2 - mu) @ W1)
        tgt_t[ib].append((V - mv) @ W2)
        tgt_e[ib].append((v2 - mv) @ W2)
    Zt = np.column_stack(pa + pb + pc)
    Ze = np.column_stack(qa + qb + qc)
    b0 = sum(x.shape[1] for x in pa)
    b1 = b0 + sum(x.shape[1] for x in pb)
    return Zt, Ze, (b0, b1)


def oof_block3(Xa, Xb, Xc, y, grp, seed,
               na=N_EHR0, nb=N_ECG0,
               nc=N_CT0, leaf=LEAF0,
               depth=DEPTH0, mfeat=MFEAT0,
               k=K0, reg=REG0, ntree=NTREE0,
               rotate=False):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        At, Ae = prep_fold(Xa[tr], Xa[te])
        Bt, Be = prep_fold(Xb[tr], Xb[te])
        Ct, Ce = prep_fold(Xc[tr], Xc[te])
        Zt, Ze, bd = rep3(At, Ae, Bt, Be,
                          Ct, Ce, k, reg)
        m = BlockForest3(
            n_estimators=ntree, n_a=na,
            n_b=nb, n_c=nc, bounds=bd,
            leaf=leaf, depth=depth,
            mfeat=mfeat, rotate=rotate,
            seed=seed)
        m.fit(Zt, y[tr])
        p[te] = m.predict_proba(Ze)[:, 1]
    return p


def oof_one(X, y, grp, seed, C=None,
            grid=None):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        Xt, Xe = prep_fold(X[tr], X[te])
        p[te] = learn_l2(Xt, y[tr], Xe,
                         grp[tr], C, grid)
    return p


def par_auc(fn, y, seeds=SEEDS):
    ps = Parallel(n_jobs=NJOBS,
                  backend="loky")(
        delayed(fn)(s) for s in seeds)
    return np.array([roc_auc_score(y, p)
                     for p in ps])


def late3(pe, pg, pc, y):
    bs, bv = -1.0, pe
    for w1 in np.arange(0, 1.001, 0.05):
        for w2 in np.arange(0, 1.001 - w1,
                            0.05):
            w3 = 1.0 - w1 - w2
            v = w1 * pe + w2 * pg + w3 * pc
            a = roc_auc_score(y, v)
            if a > bs:
                bs, bv = a, v
    return bv


def late2(pa, pb, y):
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
keep = set(E["rec"])

m12 = pd.read_csv(os.path.join(
    PROC, "ecg_derived_v12_record.csv"))
m12["rec"] = m12["rec"].astype(str).apply(
    lambda s: s if s.startswith("files/")
    else "files/" + s.replace("\\", "/")
    .strip("/"))
m12 = m12[m12["rec"].isin(keep)]
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

D = ECGA.merge(
    MA[["hadm_id"] + MC], on="hadm_id",
    how="left").merge(
    lab, on=["subject_id", "hadm_id"],
    how="inner").merge(
    mim_ehr[["hadm_id"] + EH], on="hadm_id",
    how="inner").merge(
    mm[["hadm_id"] + CT], on="hadm_id",
    how="inner")
ECGC = LG + MC
print("")
print("cohort:", len(D))
print("  EHR %d   ECG %d   CTPA %d"
      % (len(EH), len(ECGC), len(CT)))
print("  the CTPA requirement drops this from"
      " 3,501 to %d, the thin regime where"
      % len(D))
print("  cv_first failed on EHR+ECG")
print("  start: %d+%d+%d leaf=%d depth=%s"
      " k=%d mfeat=sqrt"
      % (N_EHR0, N_ECG0, N_CT0, LEAF0,
         DEPTH0, K0))
print("  parallel over up to %d seeds on %d"
      " cores" % (NJOBS, os.cpu_count() or 0),
      flush=True)

# ---------- TIMING PROBE ----------
print("")
print("TIMING PROBE")
npro = min(len(D), 1700)
ncol = len(EH) + len(ECGC) + len(CT) + 6 * K0
Xp = np.random.rand(npro, ncol)
yp = (np.random.rand(npro) > 0.90).astype(int)
t = time.time()
BlockForest3(n_estimators=NTREE0, n_a=28,
             n_b=8, n_c=16,
             bounds=(68, 380), leaf=1,
             depth=None,
             seed=42).fit(Xp, yp)
per = time.time() - t
t = time.time()
BlockForest3(n_estimators=200, n_a=28, n_b=8,
             n_c=16, bounds=(68, 380),
             leaf=1, depth=None, rotate=True,
             seed=42).fit(Xp, yp)
per_rot = (time.time() - t) * (NTREE0 / 200.0)
nA = (len(A_EHR) + len(A_ECG) + len(A_CT))
nB = len(LEAF) * len(DEPTH)
nB2 = len(MFEATS)
nC = len(CCA_KS) + len(CCA_REGS)
nD = len(NTREES)
cells = ((nA + nB + nB2 + nC + nD)
         * len(OUTS))
ser = cells * per * NFOLD * len(SEEDS) / 60.0
fin = (len(SEEDS10) * NFOLD * per
       * len(OUTS) * 2 / 60.0)
rot = (len(SEEDS) * NFOLD * per_rot
       * len(OUTS) / 60.0)
print("  %d trees on %d x %d: %5.1f s"
      % (NTREE0, npro, ncol, per))
print("  with rotation:          %5.1f s"
      % per_rot)
print("")
print("  %d grid cells; serial %.0f min,"
      " parallel about %.0f min"
      % (cells, ser + fin + rot,
         (ser + fin + rot) / NJOBS))
print("  Ctrl+C now if too long. Results save"
      " after every stage.")
time.sleep(5)

rows = []
t0 = time.time()
BEST = {}

for oc in OUTS:
    if oc not in D.columns:
        continue
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    if y.sum() < 25:
        continue
    Xa = d[EH].values.astype(float)
    Xb = d[ECGC].values.astype(float)
    Xc = d[CT].values.astype(float)
    cb = BEST_C.get(oc, 1e-3)
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())))

    pe = _rank(oof_one(Xa, y, grp, 42))
    pg = _rank(oof_one(Xb, y, grp, 42, cb))
    pc_ = _rank(oof_one(Xc, y, grp, 42,
                        grid=CT_CS))
    a_e = roc_auc_score(y, pe)
    a_g = roc_auc_score(y, pg)
    a_c = roc_auc_score(y, pc_)
    L2 = late2(pe, pc_, y)
    L3 = late3(pe, pg, pc_, y)
    ref2 = roc_auc_score(y, L2)
    ref = roc_auc_score(y, L3)
    print("  ehr %.4f   ecg %.4f   ctpa %.4f"
          % (a_e, a_g, a_c))
    print("  late 2-mod (ehr+ctpa) %.4f"
          "   late 3-mod %.4f" % (ref2, ref))
    print("  script 88 three-modality late:"
          " %.4f" % S88.get(oc, np.nan),
          flush=True)
    for nm, v in (("ehr", a_e), ("ecg", a_g),
                  ("ctpa", a_c),
                  ("late_2mod", ref2),
                  ("late_3mod", ref)):
        rows.append({
            "outcome": oc, "stage": "ref",
            "method": nm, "mean": v,
            "sd": np.nan, "vs_late": v - ref})

    # ---- STAGE A: the three block draws ----
    print("")
    print("  STAGE A: block draws, one at a"
          " time from the inherited point")
    cur = dict(na=N_EHR0, nb=N_ECG0,
               nc=N_CT0)
    for pname, grid, lab_ in (
            ("na", A_EHR, "n_ehr"),
            ("nb", A_ECG, "n_ecg"),
            ("nc", A_CT, "n_ctpa")):
        print("")
        print("    %-8s %9s %9s %9s"
              % (lab_, "AUC", "SD", "vs late"))
        bv, bm = cur[pname], -1.0
        for v in grid:
            cfg = dict(cur)
            cfg[pname] = v
            try:
                aa = par_auc(
                    lambda s, cfg=cfg:
                    oof_block3(Xa, Xb, Xc, y,
                               grp, s, **cfg),
                    y)
            except Exception as exc:
                print("    %-8s FAILED %s"
                      % (v, repr(exc)[:35]))
                continue
            mark = ("  (start)"
                    if v == cur[pname] else "")
            print("    %-8d %9.4f %9.4f"
                  " %+9.4f%s"
                  % (v, aa.mean(),
                     aa.std(ddof=1),
                     aa.mean() - ref, mark),
                  flush=True)
            rows.append({
                "outcome": oc, "stage": "A",
                "method": lab_, "value": v,
                "mean": aa.mean(),
                "sd": aa.std(ddof=1),
                "vs_late": aa.mean() - ref})
            if aa.mean() > bm:
                bm, bv = aa.mean(), v
        cur[pname] = bv
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)
    na_, nb_, nc_ = cur["na"], cur["nb"], \
        cur["nc"]
    print("")
    print("  best draws %d+%d+%d"
          % (na_, nb_, nc_), flush=True)

    # ---- STAGE B: leaf x depth ----
    print("")
    print("  STAGE B: leaf x depth")
    print("  THE FIRST DISAGREEMENT. EHR+CTPA"
          " at 1,636 wanted leaf 5 to 25 and")
    print("  depth 6; EHR+ECG at 3,501 wanted"
          " leaf 1 and unrestricted. This")
    print("  cohort is back at 1,636.")
    print("  %-6s" % "leaf", end="")
    for dp in DEPTH:
        print(" %9s" % ("d=%s" % dp), end="")
    print("")
    bB = None
    for lf in LEAF:
        line = "  %-6d" % lf
        for dp in DEPTH:
            try:
                aa = par_auc(
                    lambda s, lf=lf, dp=dp:
                    oof_block3(Xa, Xb, Xc, y,
                               grp, s, na_,
                               nb_, nc_, lf,
                               dp), y)
            except Exception:
                line += " %9s" % "fail"
                continue
            line += " %+9.4f" % (aa.mean()
                                 - ref)
            rows.append({
                "outcome": oc, "stage": "B",
                "method": "leaf_depth",
                "leaf": lf,
                "depth": (-1 if dp is None
                          else dp),
                "mean": aa.mean(),
                "sd": aa.std(ddof=1),
                "vs_late": aa.mean() - ref})
            if bB is None or aa.mean() > bB[0]:
                bB = (aa.mean(), lf, dp)
        print(line, flush=True)
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)
    lf_, dp_ = (bB[1], bB[2]) if bB \
        else (LEAF0, DEPTH0)
    print("  best leaf=%d depth=%s"
          % (lf_, dp_), flush=True)

    # ---- STAGE B2: max_features ----
    print("")
    print("  STAGE B2: max_features within the"
          " drawn columns")
    print("  NEITHER PAIRING EVER TESTED THIS"
          " once block sampling existed.")
    print("  mtry mattered enormously on"
          " EHR+CTPA before it did, and the")
    print("  sqrt default on a %d-column draw"
          " is %d."
          % (na_ + nb_ + nc_,
             int(round(np.sqrt(na_ + nb_
                               + nc_)))))
    print("    %-10s %6s %9s %9s"
          % ("mfeat", "cols", "AUC",
             "vs late"))
    bmf, bmfm = MFEAT0, -1.0
    ndraw = na_ + nb_ + nc_
    for mf in MFEATS:
        try:
            aa = par_auc(
                lambda s, mf=mf:
                oof_block3(Xa, Xb, Xc, y, grp,
                           s, na_, nb_, nc_,
                           lf_, dp_, mf), y)
        except Exception:
            continue
        nc2 = (int(round(np.sqrt(ndraw)))
               if mf is None
               else int(round(mf * ndraw)))
        tag = ("sqrt" if mf is None
               else "%.2f" % mf)
        print("    %-10s %6d %9.4f %+9.4f"
              % (tag, nc2, aa.mean(),
                 aa.mean() - ref), flush=True)
        rows.append({
            "outcome": oc, "stage": "B2",
            "method": "mfeat",
            "value": (-1.0 if mf is None
                      else mf),
            "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_late": aa.mean() - ref})
        if aa.mean() > bmfm:
            bmfm, bmf = aa.mean(), mf
    print("  best mfeat=%s"
          % ("sqrt" if bmf is None
             else "%.2f" % bmf), flush=True)
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)

    # ---- STAGE C: CCA rank and reg ----
    print("")
    print("  STAGE C: CCA rank and"
          " regularisation")
    print("  THE SECOND DISAGREEMENT. EHR+CTPA"
          " found k 3 to 10; EHR+ECG found")
    print("  20. Here k costs 6k columns"
          " rather than 2k, since there are")
    print("  three pairwise variate sets.")
    bk, bkm = K0, -1.0
    print("    %-8s %9s %9s" % ("k", "AUC",
                                "vs late"))
    for k in CCA_KS:
        try:
            aa = par_auc(
                lambda s, k=k:
                oof_block3(Xa, Xb, Xc, y, grp,
                           s, na_, nb_, nc_,
                           lf_, dp_, bmf, k),
                y)
        except Exception:
            continue
        print("    %-8d %9.4f %+9.4f"
              % (k, aa.mean(),
                 aa.mean() - ref), flush=True)
        rows.append({
            "outcome": oc, "stage": "C",
            "method": "cca_k", "value": k,
            "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_late": aa.mean() - ref})
        if aa.mean() > bkm:
            bkm, bk = aa.mean(), k
    br, brm = REG0, -1.0
    print("    %-8s %9s %9s" % ("reg", "AUC",
                                "vs late"))
    for rg in CCA_REGS:
        try:
            aa = par_auc(
                lambda s, rg=rg:
                oof_block3(Xa, Xb, Xc, y, grp,
                           s, na_, nb_, nc_,
                           lf_, dp_, bmf, bk,
                           rg), y)
        except Exception:
            continue
        print("    %-8.2f %9.4f %+9.4f"
              % (rg, aa.mean(),
                 aa.mean() - ref), flush=True)
        rows.append({
            "outcome": oc, "stage": "C",
            "method": "cca_reg", "value": rg,
            "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_late": aa.mean() - ref})
        if aa.mean() > brm:
            brm, br = aa.mean(), rg
    print("  best k=%d reg=%.2f" % (bk, br),
          flush=True)
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)

    # ---- STAGE D: tree count ----
    print("")
    print("  STAGE D: tree count")
    bt, btm = NTREE0, -1.0
    print("    %-8s %9s %9s" % ("ntree",
                                "AUC",
                                "vs late"))
    for nt in NTREES:
        try:
            aa = par_auc(
                lambda s, nt=nt:
                oof_block3(Xa, Xb, Xc, y, grp,
                           s, na_, nb_, nc_,
                           lf_, dp_, bmf, bk,
                           br, nt), y)
        except Exception:
            continue
        print("    %-8d %9.4f %+9.4f"
              % (nt, aa.mean(),
                 aa.mean() - ref), flush=True)
        rows.append({
            "outcome": oc, "stage": "D",
            "method": "ntree", "value": nt,
            "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_late": aa.mean() - ref})
        if aa.mean() > btm:
            btm, bt = aa.mean(), nt
    print("  best ntree=%d" % bt, flush=True)

    # ---- STAGE E: rotation, then final ----
    print("")
    print("  STAGE E: ROTATION AT MATCHED"
          " max_features")
    print("  script 110 gave rotation forest"
          " 0.8748 vs rf 0.8683 on EHR+CTPA,")
    print("  but its trees had no max_features"
          " while the rf used 8.")
    try:
        ar = par_auc(
            lambda s: oof_block3(
                Xa, Xb, Xc, y, grp, s, na_,
                nb_, nc_, lf_, dp_, bmf, bk,
                br, bt, True), y)
        print("    with rotation %.4f"
              " (SD %.4f)"
              % (ar.mean(), ar.std(ddof=1)),
              flush=True)
        rows.append({
            "outcome": oc, "stage": "E",
            "method": "rotation",
            "mean": ar.mean(),
            "sd": ar.std(ddof=1),
            "vs_late": ar.mean() - ref})
    except Exception:
        ar = None

    print("")
    print("  FINAL, ten seeds")
    print("    %d+%d+%d leaf=%d depth=%s"
          " mfeat=%s k=%d reg=%.2f ntree=%d"
          % (na_, nb_, nc_, lf_, dp_,
             "sqrt" if bmf is None
             else "%.2f" % bmf, bk, br, bt))
    aa = par_auc(
        lambda s: oof_block3(
            Xa, Xb, Xc, y, grp, s, na_, nb_,
            nc_, lf_, dp_, bmf, bk, br, bt),
        y, seeds=SEEDS10)
    p = oof_block3(Xa, Xb, Xc, y, grp, 42,
                   na_, nb_, nc_, lf_, dp_,
                   bmf, bk, br, bt)
    b0, s0 = calib(p, y)
    pcal = platt_oof(p, y, grp, 42)
    ok = np.isfinite(pcal)
    b1, s1 = calib(pcal[ok], y[ok])
    gn, lo, hi, _ = f.boot_diff(
        y, _rank(p), _rank(L3), grp)
    g2, l2_, h2, _ = f.boot_diff(
        y, _rank(p), _rank(L2), grp)
    print("    %.4f (SD %.4f, range"
          " %.4f-%.4f)"
          % (aa.mean(), aa.std(ddof=1),
             aa.min(), aa.max()))
    print("    vs late 3-mod %+.4f"
          " [%+.4f,%+.4f] %s"
          % (gn, lo, hi,
             "*" if (lo > 0 or hi < 0)
             else ""))
    print("    vs late 2-mod %+.4f"
          " [%+.4f,%+.4f] %s"
          % (g2, l2_, h2,
             "*" if (l2_ > 0 or h2 < 0)
             else ""))
    print("    slope %.3f -> %.3f after Platt"
          % (s0, s1))
    if ar is not None:
        print("    rotation would give %+.4f"
              " against this"
              % (ar.mean() - aa.mean()),
              flush=True)
    BEST[oc] = (na_, nb_, nc_, lf_, dp_, bmf,
                bk, br, bt, aa.mean(), ref,
                ref2)
    rows.append({
        "outcome": oc, "stage": "E",
        "method": "final", "na": na_,
        "nb": nb_, "nc": nc_, "leaf": lf_,
        "depth": (-1 if dp_ is None else dp_),
        "mfeat": (-1.0 if bmf is None
                  else bmf),
        "k": bk, "reg": br, "ntree": bt,
        "mean": aa.mean(),
        "sd": aa.std(ddof=1), "lo": aa.min(),
        "hi": aa.max(), "boot": gn,
        "boot_lo": lo, "boot_hi": hi,
        "sig": int(lo > 0 or hi < 0),
        "boot2": g2, "sig2": int(l2_ > 0
                                 or h2 < 0),
        "slope": s0, "slope_platt": s1,
        "vs_late": aa.mean() - ref,
        "vs_late2": aa.mean() - ref2})

    pd.DataFrame(rows).to_csv(DEST,
                              index=False)
    print("")
    print("  %s done in %.1f min  (saved)"
          % (oc, (time.time() - tb) / 60),
          flush=True)

# ---- STAGE F: the compromise setting ----
print("")
print("#" * 74)
print("STAGE F: ONE SETTING FOR ALL OUTCOMES")
print("  on EHR+ECG the compromise cost only"
      " 0.0012, so the per-outcome optima")
print("  were noise. This tests the same here.")
print("#" * 74, flush=True)

m = pd.DataFrame(rows)
gA = m[m["stage"] == "A"]


def marg(meth, default):
    z = gA[gA["method"] == meth]
    if not len(z):
        return default
    return int(z.groupby("value")["vs_late"]
               .mean().idxmax())


cm_na = marg("n_ehr", N_EHR0)
cm_nb = marg("n_ecg", N_ECG0)
cm_nc = marg("n_ctpa", N_CT0)
gB = m[m["stage"] == "B"]
cm_lf = int(gB.groupby("leaf")["vs_late"]
            .mean().idxmax()) if len(gB) \
    else LEAF0
cm_dp_ = (gB.groupby("depth")["vs_late"]
          .mean().idxmax() if len(gB) else -1)
cm_dp = None if cm_dp_ == -1 else int(cm_dp_)
gM = m[m["method"] == "mfeat"]
cm_mf_ = (gM.groupby("value")["vs_late"]
          .mean().idxmax() if len(gM) else -1.0)
cm_mf = None if cm_mf_ < 0 else float(cm_mf_)
gK = m[m["method"] == "cca_k"]
cm_k = int(gK.groupby("value")["vs_late"]
           .mean().idxmax()) if len(gK) else K0
gR = m[m["method"] == "cca_reg"]
cm_rg = float(gR.groupby("value")["vs_late"]
              .mean().idxmax()) if len(gR) \
    else REG0
gD = m[m["stage"] == "D"]
cm_nt = int(gD.groupby("value")["vs_late"]
            .mean().idxmax()) if len(gD) \
    else NTREE0
print("")
print("  compromise: %d+%d+%d leaf=%d"
      " depth=%s mfeat=%s k=%d reg=%.2f"
      " ntree=%d"
      % (cm_na, cm_nb, cm_nc, cm_lf, cm_dp,
         "sqrt" if cm_mf is None
         else "%.2f" % cm_mf, cm_k, cm_rg,
         cm_nt), flush=True)

for oc in OUTS:
    if oc not in D.columns or oc not in BEST:
        continue
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    Xa = d[EH].values.astype(float)
    Xb = d[ECGC].values.astype(float)
    Xc = d[CT].values.astype(float)
    (na_, nb_, nc_, lf_, dp_, bmf, bk, br,
     bt, po, ref, ref2) = BEST[oc]
    aa = par_auc(
        lambda s: oof_block3(
            Xa, Xb, Xc, y, grp, s, cm_na,
            cm_nb, cm_nc, cm_lf, cm_dp,
            cm_mf, cm_k, cm_rg, cm_nt), y,
        seeds=SEEDS10)
    print("  %-18s compromise %.4f"
          "   per-outcome %.4f   %+.4f"
          "   late3 %.4f   late2 %.4f"
          % (oc, aa.mean(), po,
             aa.mean() - po, ref, ref2),
          flush=True)
    rows.append({
        "outcome": oc, "stage": "F",
        "method": "compromise", "na": cm_na,
        "nb": cm_nb, "nc": cm_nc,
        "leaf": cm_lf,
        "depth": (-1 if cm_dp is None
                  else cm_dp),
        "mfeat": (-1.0 if cm_mf is None
                  else cm_mf),
        "k": cm_k, "reg": cm_rg,
        "ntree": cm_nt, "mean": aa.mean(),
        "sd": aa.std(ddof=1),
        "vs_late": aa.mean() - ref,
        "vs_late2": aa.mean() - ref2,
        "vs_per_outcome": aa.mean() - po})

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("STAGE A: THE THREE BLOCK DRAWS")
a = r[r["stage"] == "A"]
for meth in ("n_ehr", "n_ecg", "n_ctpa"):
    z = a[a["method"] == meth]
    if not len(z):
        continue
    print("")
    print("  " + meth)
    print(z.pivot_table(index="value",
                        columns="outcome",
                        values="vs_late")
          .round(4).to_string())
    print("  marginal")
    print(z.groupby("value")["vs_late"].mean()
          .round(4).to_string())

print("")
print("=" * 74)
print("THE THREE DISAGREEMENTS RESOLVED")
b = r[r["stage"] == "B"]
if len(b):
    print("")
    print("  LEAF  (EHR+CTPA wanted 5-25,"
          " EHR+ECG wanted 1)")
    print(b.groupby("leaf")["vs_late"].mean()
          .round(4).to_string())
    print("")
    print("  DEPTH  (EHR+CTPA wanted 6,"
          " EHR+ECG unrestricted; -1 is")
    print("  unrestricted)")
    print(b.groupby("depth")["vs_late"].mean()
          .round(4).to_string())
zk = r[r["method"] == "cca_k"]
if len(zk):
    print("")
    print("  CCA k  (EHR+CTPA wanted 3-10,"
          " EHR+ECG wanted 20)")
    print(zk.groupby("value")["vs_late"].mean()
          .round(4).to_string())

print("")
print("THE PARAMETER NEITHER PAIRING TESTED")
zm = r[r["method"] == "mfeat"]
if len(zm):
    print("  max_features within the draw"
          "  (-1.0 is sklearn's sqrt default)")
    print(zm.pivot_table(index="value",
                         columns="outcome",
                         values="vs_late")
          .round(4).to_string())
    print("  marginal")
    print(zm.groupby("value")["vs_late"].mean()
          .round(4).to_string())
    rg_ = zm["mean"].max() - zm["mean"].min()
    print("")
    print("  range %.4f   mean seed SD %.4f"
          "   ratio %.1f"
          % (rg_, zm["sd"].mean(),
             rg_ / max(zm["sd"].mean(), 1e-9)))
    print("  a ratio below about 2 means it"
          " does not matter here")

print("")
print("CCA reg AND TREE COUNT")
for meth in ("cca_reg", "ntree"):
    z = r[r["method"] == meth]
    if not len(z):
        continue
    print("  %-8s" % meth)
    print(z.groupby("value")["vs_late"].mean()
          .round(4).to_string())

print("")
print("=" * 74)
print("ROTATION, RE-TESTED AT MATCHED"
      " max_features")
rr = r[r["method"] == "rotation"]
ee = r[(r["stage"] == "E")
       & (r["method"] == "final")]
if len(rr) and len(ee):
    for oc in OUTS:
        a1 = rr[rr["outcome"] == oc]
        a2 = ee[ee["outcome"] == oc]
        if len(a1) and len(a2):
            print("  %-18s no rotation %.4f"
                  "   with rotation %.4f"
                  "   %+.4f"
                  % (oc, a2["mean"].iloc[0],
                     a1["mean"].iloc[0],
                     a1["mean"].iloc[0]
                     - a2["mean"].iloc[0]))
    print("")
    print("  script 110's +0.0065 on EHR+CTPA"
          " was measured with no max_features")
    print("  on the rotation forest and 8 on"
          " the random forest")

print("")
print("=" * 74)
print("FINAL, TEN SEEDS")
if len(ee):
    print(ee[["outcome", "na", "nb", "nc",
              "leaf", "depth", "mfeat", "k",
              "reg", "ntree", "mean", "sd",
              "vs_late", "vs_late2", "sig",
              "sig2"]].round(4)
          .to_string(index=False))

print("")
print("DOES THE THIRD MODALITY PAY?")
print("  vs_late2 compares against EHR+CTPA"
      " late fusion, which is the question")
print("  the dissertation turned on (it"
      " gained 0.0002 there)")
if len(ee):
    for _, x in ee.iterrows():
        print("  %-18s block %.4f"
              "   vs late 3-mod %+.4f %s"
              "   vs late 2-mod %+.4f %s"
              % (x["outcome"], x["mean"],
                 x["vs_late"],
                 "*" if x.get("sig") else " ",
                 x["vs_late2"],
                 "*" if x.get("sig2")
                 else " "))

print("")
print("COMPROMISE vs PER-OUTCOME")
c = r[r["stage"] == "F"]
if len(c):
    print(c[["outcome", "mean", "sd",
             "vs_late", "vs_late2",
             "vs_per_outcome"]].round(4)
          .to_string(index=False))
    md = c["vs_per_outcome"].abs().mean()
    print("")
    print("  mean absolute loss from one"
          " setting: %.4f" % md)
    if md < 0.005:
        print("  -> report the compromise")
    else:
        print("  -> the differences are real")

print("")
print("WINS OVER LATE FUSION")
for stg, nm in (("E", "per-outcome"),
                ("F", "compromise")):
    z = r[(r["stage"] == stg)
          & (r["method"].isin(
              ["final", "compromise"]))]
    if len(z):
        print("  %-12s %d of %d vs late 3-mod,"
              "  %d of %d vs late 2-mod"
              % (nm, int((z["vs_late"] > 0)
                         .sum()), len(z),
                 int((z["vs_late2"] > 0)
                     .sum()), len(z)))
print("")
print("saved", DEST, r.shape)
keep_awake(False)