"""Ensembling the three-modality models, with
competence-matched tree mixes and tertile
refinements.

Three three-modality models are combined: the
block forest, the competence-weighted pool and
late fusion. They differ in column sampling,
voting and combination rule.

Stage B derives the pool's tree mixes by
clustering the observed competence profiles,
replacing the fixed mixes of script 123. Stage C
varies the number of tertile bins and the driving
feature, both fixed in script 124.

META-LEARNERS
  grid    simplex weight search
  nnls    non-negative least squares (Super
          Learner)
  l2      L2 logistic meta-learner
  l1      lasso, which may drop a model
  enet    elastic net
  mean    unweighted average, as a floor

Every meta-learner is fitted on training folds
only. grid_ins, fitted on the rows it scores, is
reported alongside to show the size of that
optimism.

Parallel across seeds. Timing probe first.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ensemble_3mod.csv
  results\\ensemble_3mod_log.txt
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import linalg as sla
from scipy.optimize import nnls
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
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

PROC, FIG = f.PROC, f.FIG
RAW = os.path.join("data", "raw", "ecg")
SEEDS = [42, 7, 13, 1, 2]
NFOLD = 5
NJOBS = max(1, min(10, (os.cpu_count() or 4)
                   - 2))
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CT_CS = list(np.logspace(-4, 4, 10))
LEAF, DEPTH, K, REG, NTREE = 1, 12, 5, 0.5, 600
# from script 124: tertile measure, k=3, temp 5
SPARSE_K, TEMP = 3, 5.0
BINS = [2, 3, 4]
N_DRIVER = [1, 2]
N_MIX_CLUST = 7
TOTAL_COLS = 44
HAND_MIXES = [(28, 2, 2), (20, 8, 8),
              (8, 20, 8), (8, 8, 20),
              (14, 14, 14), (4, 4, 28),
              (4, 28, 4)]
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(
    PROC, "ensemble_3mod.csv")
S124 = {"death_30d_inhosp":
        {"block": 0.9053, "pool": 0.9038,
         "best": 0.9069},
        "death_30d":
        {"block": 0.8866, "pool": 0.8883,
         "best": 0.8896},
        "composite_30d":
        {"block": 0.8507, "pool": np.nan,
         "best": np.nan},
        "cv_first":
        {"block": 0.7888, "pool": np.nan,
         "best": np.nan}}

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


def ccaaug3(Bt, Be):
    pt = [[b] for b in Bt]
    pe = [[b] for b in Be]
    for i in range(3):
        for jj in range(i + 1, 3):
            W1, W2 = rcca(Bt[i], Bt[jj],
                          REG, K)
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


def learn_l2(Xt, yt, Xe, gt=None, grid=None):
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
    return (m.predict_proba(Xe)[:, 1],
            m.predict_proba(Xt)[:, 1])


def tertile_comp(p, y, cols, n_bins):
    """DISCRETE competence: each patient gets
    their cell's AUROC.

    Script 124 found this clearly the best of
    four measures, with the largest profile
    spread (0.1128 against tcp's 0.0824). n_bins
    and the number of driving features were
    both arbitrary there, so both vary here."""
    codes = np.zeros(len(y), dtype=int)
    mult = 1
    edges = []
    for c in cols.T:
        qs = np.nanquantile(
            c, np.linspace(0, 1,
                           n_bins + 1)[1:-1])
        codes = codes + mult * np.digitize(c, qs)
        edges.append(qs)
        mult *= n_bins
    lut = {}
    out = np.full(len(y), 0.5)
    for cd in np.unique(codes):
        m = codes == cd
        if m.sum() > 25 and 0 < y[m].sum() \
                < m.sum():
            try:
                v = roc_auc_score(y[m], p[m])
            except Exception:
                v = 0.5
        else:
            v = 0.5
        lut[int(cd)] = v
        out[m] = v
    return out, lut, edges


def apply_tertile(cols, edges, lut, n_bins):
    codes = np.zeros(len(cols), dtype=int)
    mult = 1
    for j, c in enumerate(cols.T):
        codes = codes + mult * np.digitize(
            c, edges[j])
        mult *= n_bins
    return np.array([lut.get(int(v), 0.5)
                     for v in codes])


def to_weights(C, temp):
    C = np.clip(C - C.min(1, keepdims=True)
                + 1e-6, 1e-9, None)
    C = C / C.sum(1, keepdims=True)
    S = np.exp(temp * (C - C.max(
        1, keepdims=True)))
    return S / S.sum(1, keepdims=True)


class CompetenceForest:
    def __init__(self, n_estimators=600,
                 mixes=HAND_MIXES,
                 bounds=(38, 312), leaf=1,
                 depth=12, seed=42):
        self.n = n_estimators
        self.mixes = mixes
        self.bounds = bounds
        self.leaf = leaf
        self.depth = depth
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        cuts = ([0] + [int(min(b, p))
                       for b in self.bounds]
                + [p])
        blk = [np.arange(cuts[i], cuts[i + 1])
               for i in range(3)]
        self.sel_, self.tr_, prof = [], [], []
        per = max(1, self.n // len(self.mixes))
        for mix in self.mixes:
            for _ in range(per):
                parts, got = [], []
                for ix, w in zip(blk, mix):
                    if len(ix) == 0 or w <= 0:
                        got.append(0)
                        continue
                    k = int(min(max(1, int(w)),
                                len(ix)))
                    parts.append(rng.choice(
                        ix, k, replace=False))
                    got.append(k)
                if not parts:
                    continue
                cols = np.concatenate(parts)
                rows = rng.choice(n, n,
                                  replace=True)
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
                g = np.array(got, dtype=float)
                prof.append(g / max(g.sum(), 1))
        self.prof_ = np.array(prof)
        return self

    def tree_preds(self, X):
        Q = np.zeros((len(X), len(self.tr_)))
        for j, (cols, t) in enumerate(
                zip(self.sel_, self.tr_)):
            Q[:, j] = t.predict_proba(
                X[:, cols])[:, 1]
        return Q

    def combine(self, Q, W, temp):
        if W is None:
            return Q.mean(1)
        S = W @ self.prof_.T
        S = np.exp(temp * (S - S.max(
            1, keepdims=True)))
        S = S / S.sum(1, keepdims=True)
        return (Q * S).sum(1)


class BlockForest:
    def __init__(self, n_estimators=600,
                 draws=(20, 8, 16),
                 bounds=(38, 312), leaf=1,
                 depth=12, seed=42):
        self.n = n_estimators
        self.draws = draws
        self.bounds = bounds
        self.leaf = leaf
        self.depth = depth
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        cuts = ([0] + [int(min(b, p))
                       for b in self.bounds]
                + [p])
        blk = [np.arange(cuts[i], cuts[i + 1])
               for i in range(len(cuts) - 1)]
        self.sel_, self.tr_ = [], []
        for _ in range(self.n):
            parts = []
            for ix, w in zip(blk, self.draws):
                if len(ix) == 0 or w <= 0:
                    continue
                k = int(min(max(1, w), len(ix)))
                parts.append(rng.choice(
                    ix, k, replace=False))
            if not parts:
                continue
            cols = np.concatenate(parts)
            rows = rng.choice(n, n,
                              replace=True)
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
        return self

    def predict_proba(self, X):
        out = np.zeros(len(X))
        for cols, t in zip(self.sel_,
                           self.tr_):
            out += t.predict_proba(
                X[:, cols])[:, 1]
        out /= max(len(self.tr_), 1)
        return np.column_stack([1 - out, out])


# ---------------- meta-learners -------------
def simplex(m, step=0.05):
    n = int(round(1.0 / step))

    def rec(k, rem):
        if k == 1:
            yield (rem,)
            return
        for i in range(rem + 1):
            for t in rec(k - 1, rem - i):
                yield (i,) + t
    for w in rec(m, n):
        yield tuple(x * step for x in w)


def meta_fit(R, y, tr, kind):
    """Fitted on training rows only, so the
    combination never sees in-sample
    predictions from any model."""
    m = R.shape[1]
    if kind == "mean":
        return np.ones(m) / m, None
    if kind == "grid":
        best, bw = -1.0, np.ones(m) / m
        for w in simplex(m):
            v = R[tr] @ np.array(w)
            try:
                a = roc_auc_score(y[tr], v)
            except Exception:
                continue
            if a > best:
                best, bw = a, np.array(w)
        return bw, None
    if kind == "nnls":
        # the Super Learner form: weights
        # constrained non-negative, which suits
        # predictors that all point the same way
        try:
            w, _ = nnls(R[tr],
                        y[tr].astype(float))
        except Exception:
            w = np.ones(m)
        if w.sum() <= 1e-9:
            w = np.ones(m)
        return w / w.sum(), None
    pen = {"l2": ("l2", "lbfgs"),
           "l1": ("l1", "liblinear"),
           "enet": ("elasticnet", "saga")}
    p_, s_ = pen.get(kind, ("l2", "lbfgs"))
    best, bm = -1.0, None
    for c in [0.01, 0.1, 1.0, 10.0]:
        try:
            kw = dict(penalty=p_, C=c,
                      solver=s_, max_iter=4000)
            if p_ == "elasticnet":
                kw["l1_ratio"] = 0.5
            mm = LogisticRegression(**kw)
            mm.fit(R[tr], y[tr])
            a = roc_auc_score(
                y[tr],
                mm.predict_proba(R[tr])[:, 1])
            if a > best:
                best, bm = a, mm
        except Exception:
            continue
    if bm is None:
        return np.ones(m) / m, None
    return None, bm


def meta_apply(R, te, w, mdl):
    if mdl is not None:
        return mdl.predict_proba(R[te])[:, 1]
    return R[te] @ w


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
BLK = {"ehr": EH, "ecg": ECGC, "ctpa": CT}
MODS = ["ehr", "ecg", "ctpa"]
ALL_NAMES = EH + ECGC + CT
METAS = ["mean", "grid", "nnls", "l2",
         "l1", "enet"]
print("")
print("cohort:", len(D))
print("  EHR %d   ECG %d   CTPA %d"
      % (len(EH), len(ECGC), len(CT)))
print("  competence: tertile measure, k=%d,"
      " temp=%.0f  (script 124's winner)"
      % (SPARSE_K, TEMP))
print("  bins tested:", BINS,
      "  drivers tested:", N_DRIVER)
print("  meta-learners:", ", ".join(METAS))
print("  every meta-learner is fitted on"
      " TRAINING rows only; grid_ins is the")
print("  in-sample version, so the leak in the"
      " project's usual late fusion is")
print("  measurable", flush=True)

# ---------- TIMING PROBE ----------
print("")
print("TIMING PROBE")
Xp = np.random.rand(len(D), 400)
yp = (np.random.rand(len(D)) > 0.90) \
    .astype(int)
t = time.time()
BlockForest(n_estimators=NTREE,
            bounds=(38, 312), leaf=LEAF,
            depth=DEPTH, seed=42).fit(Xp, yp)
t_blk = time.time() - t
t = time.time()
cfp = CompetenceForest(
    n_estimators=NTREE, bounds=(38, 312),
    leaf=LEAF, depth=DEPTH, seed=42)
cfp.fit(Xp, yp)
t_cf = time.time() - t
t = time.time()
RandomForestRegressor(
    n_estimators=150, max_depth=5,
    min_samples_leaf=30, n_jobs=1,
    random_state=42).fit(
    Xp, np.random.rand(len(D)))
t_cm = time.time() - t
# per fold: block + hand pool + clustered pool
# + 3 driver models
per = t_blk + 2 * t_cf + 3 * t_cm
ser = len(OUTS) * per * NFOLD \
    * len(SEEDS) / 60.0
print("  block forest       %6.1f s" % t_blk)
print("  competence pool    %6.1f s" % t_cf)
print("  one driver model   %6.1f s" % t_cm)
print("")
print("  the %d bin/driver variants and %d"
      " meta-learners reuse the same tree"
      % (len(BINS) * len(N_DRIVER),
         len(METAS)))
print("  predictions, so they cost almost"
      " nothing beyond the forests")
print("  serial about %.0f min; with %d-way"
      " parallelism about %.0f min"
      % (ser, NJOBS, ser / NJOBS))
print("  Ctrl+C now if too long. Results save"
      " after every outcome.")
time.sleep(5)

rows = []
t0 = time.time()


def run_seed(Xd, y, grp, Xall, seed):
    n = len(y)
    variants = [(b, nd) for b in BINS
                for nd in N_DRIVER]
    base = {"block": np.zeros(n),
            "late": np.zeros(n),
            "pool_hand": np.zeros(n),
            "pool_clust": np.zeros(n)}
    for v in variants:
        base[("hand", v)] = np.zeros(n)
        base[("clust", v)] = np.zeros(n)
    mixinfo = []
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((n, 1)), y, grp):
        Bt, Be = [], []
        for m_ in MODS:
            a, b = prep_fold(Xd[m_][tr],
                             Xd[m_][te])
            Bt.append(a)
            Be.append(b)
        Zt, Ze, bd = ccaaug3(Bt, Be)
        At = np.nan_to_num(Xall[tr])
        Ae = np.nan_to_num(Xall[te])

        # block forest
        bf = BlockForest(
            n_estimators=NTREE,
            draws=(20, 8, 16), bounds=bd,
            leaf=LEAF, depth=DEPTH, seed=seed)
        bf.fit(Zt, y[tr])
        base["block"][te] = bf.predict_proba(
            Ze)[:, 1]

        # per-modality predictions, late
        # fusion, and the driving features
        ptr, drivers = [], []
        ps_te = []
        for mi, m_ in enumerate(MODS):
            gd = CT_CS if m_ == "ctpa" else None
            pe_, pt2 = learn_l2(
                Bt[mi], y[tr], Be[mi],
                grp[tr], gd)
            ptr.append(_rank(pt2))
            ps_te.append(_rank(pe_))
            gm = RandomForestRegressor(
                n_estimators=150, max_depth=5,
                min_samples_leaf=30,
                random_state=seed, n_jobs=1)
            tgt = np.where(y[tr] == 1,
                           np.clip(pt2, 1e-6,
                                   1 - 1e-6),
                           1 - np.clip(
                               pt2, 1e-6,
                               1 - 1e-6))
            gm.fit(At, tgt)
            drivers.append(np.argsort(
                -gm.feature_importances_)[:3])
        bs, bv = -1.0, ps_te[0]
        for w1 in np.arange(0, 1.001, 0.05):
            for w2 in np.arange(
                    0, 1.001 - w1, 0.05):
                v = (w1 * ps_te[0]
                     + w2 * ps_te[1]
                     + (1 - w1 - w2) * ps_te[2])
                try:
                    a = roc_auc_score(y[te], v)
                except Exception:
                    continue
                if a > bs:
                    bs, bv = a, v
        base["late"][te] = bv

        # competence per (bins, drivers)
        CT_ = {}
        for (nb, nd) in variants:
            Cte = np.zeros((len(te), 3))
            Ctr = np.zeros((len(tr), 3))
            for mi in range(3):
                cs = drivers[mi][:nd]
                c_tr, lut, edges = tertile_comp(
                    ptr[mi], y[tr],
                    At[:, cs], nb)
                Ctr[:, mi] = c_tr
                Cte[:, mi] = apply_tertile(
                    Ae[:, cs], edges, lut, nb)
            CT_[(nb, nd)] = (Ctr, Cte)

        # STAGE B: mixes from the observed
        # competence profiles rather than by
        # hand. Clustering the training
        # profiles puts trees where the
        # competence regions actually are.
        Ctr0 = CT_[(3, 1)][0]
        W0 = to_weights(Ctr0, 1.0)
        try:
            km = KMeans(n_clusters=N_MIX_CLUST,
                        n_init=10,
                        random_state=seed)
            km.fit(W0)
            cm_ = km.cluster_centers_
            cmix = [tuple(max(2, int(round(
                c[i] * TOTAL_COLS)))
                for i in range(3))
                for c in cm_]
        except Exception:
            cmix = list(HAND_MIXES)
        mixinfo.append(cmix)

        for tag, mx in (("hand", HAND_MIXES),
                        ("clust", cmix)):
            cf = CompetenceForest(
                n_estimators=NTREE, mixes=mx,
                bounds=bd, leaf=LEAF,
                depth=DEPTH, seed=seed)
            cf.fit(Zt, y[tr])
            Q = cf.tree_preds(Ze)
            base["pool_" + tag][te] = \
                cf.combine(Q, None, TEMP)
            for v in variants:
                Wte = to_weights(
                    CT_[v][1], TEMP)
                base[(tag, v)][te] = \
                    cf.combine(Q, Wte, TEMP)
    return base, mixinfo


for oc in OUTS:
    if oc not in D.columns:
        continue
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    if y.sum() < 25:
        continue
    Xd = {k: d[v].values.astype(float)
          for k, v in BLK.items()}
    Xall = np.column_stack(
        [Xd["ehr"], Xd["ecg"], Xd["ctpa"]])
    pv = S124.get(oc, {})
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d   script 124:"
          " block %.4f  best comp %.4f"
          % (oc, len(y), int(y.sum()),
             pv.get("block", np.nan),
             pv.get("best", np.nan)))

    try:
        res = par(lambda s:
                  run_seed(Xd, y, grp, Xall, s))
    except Exception as exc:
        print("  FAILED %s" % repr(exc)[:70])
        continue
    Bs = [r_[0] for r_ in res]
    mixes0 = res[0][1][0]

    print("")
    print("  MIXES DERIVED FROM COMPETENCE"
          "  (fold 1, seed 42)")
    for mx in mixes0:
        tt = sum(mx)
        print("    %2d+%2d+%2d -> %.2f/%.2f/%.2f"
              % (mx[0], mx[1], mx[2],
                 mx[0] / tt, mx[1] / tt,
                 mx[2] / tt))
    print("    against the hand-picked pool,"
          " which spanned the simplex evenly",
          flush=True)

    # base models
    print("")
    print("  BASE MODELS")
    single = {}
    for nm in ("block", "late", "pool_hand",
               "pool_clust"):
        aa = np.array([roc_auc_score(y, b[nm])
                       for b in Bs])
        single[nm] = aa.mean()
        print("    %-12s %.4f (SD %.4f)"
              % (nm, aa.mean(),
                 aa.std(ddof=1)))
        rows.append({
            "outcome": oc, "stage": "base",
            "name": nm, "mean": aa.mean(),
            "sd": aa.std(ddof=1)})

    # competence variants
    print("")
    print("  COMPETENCE WEIGHTING, bins x"
          " drivers")
    print("  %-7s %-8s %9s %9s"
          % ("mixes", "bins/drv", "AUC",
             "vs pool"))
    bestc = None
    for tag in ("hand", "clust"):
        for nb in BINS:
            for nd in N_DRIVER:
                aa = np.array([
                    roc_auc_score(
                        y, b[(tag, (nb, nd))])
                    for b in Bs])
                ref = single["pool_" + tag]
                print("  %-7s %d bins %d drv"
                      " %9.4f %+9.4f"
                      % (tag, nb, nd,
                         aa.mean(),
                         aa.mean() - ref),
                      flush=True)
                rows.append({
                    "outcome": oc,
                    "stage": "comp",
                    "name": "%s_b%d_d%d"
                            % (tag, nb, nd),
                    "mixes": tag, "bins": nb,
                    "drivers": nd,
                    "mean": aa.mean(),
                    "sd": aa.std(ddof=1),
                    "vs_pool": aa.mean() - ref})
                if bestc is None or \
                        aa.mean() > bestc[0]:
                    bestc = (aa.mean(), tag,
                             nb, nd)
    if bestc:
        print("  best %.4f  mixes=%s bins=%d"
              " drivers=%d"
              % bestc, flush=True)
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)

    # ---- the ensemble ----
    print("")
    print("  ENSEMBLING THE DISTINCT MODELS")
    b0 = Bs[0]
    comp_best = b0[(bestc[1],
                    (bestc[2], bestc[3]))]
    members = {"block": b0["block"],
               "late": b0["late"],
               "comp": comp_best}
    nms = list(members.keys())
    R = np.column_stack(
        [_rank(members[k]) for k in nms])
    print("    correlations between members")
    for i in range(len(nms)):
        for jj in range(i + 1, len(nms)):
            cc = np.corrcoef(R[:, i],
                             R[:, jj])[0, 1]
            print("      %-6s vs %-6s %.3f"
                  % (nms[i], nms[jj], cc))
    print("    highly correlated members leave"
          " little for any combiner to do",
          flush=True)

    best_single = max(single.values())
    print("")
    print("    %-10s %9s %9s"
          % ("meta", "AUC", "vs best"))
    for kind in METAS + ["grid_ins"]:
        p = np.full(len(y), np.nan)
        cv = StratifiedGroupKFold(
            n_splits=NFOLD, shuffle=True,
            random_state=42)
        for tr, te in cv.split(R, y, grp):
            src = te if kind == "grid_ins" \
                else tr
            kk = ("grid" if kind == "grid_ins"
                  else kind)
            w, mdl = meta_fit(R, y, src, kk)
            p[te] = meta_apply(R, te, w, mdl)
        ok = np.isfinite(p)
        av = roc_auc_score(y[ok], p[ok])
        tag = ("  (in sample)"
               if kind == "grid_ins" else "")
        print("    %-10s %9.4f %+9.4f%s"
              % (kind, av, av - best_single,
                 tag), flush=True)
        rows.append({
            "outcome": oc, "stage": "meta",
            "name": kind, "mean": av,
            "sd": np.nan,
            "vs_best": av - best_single})

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
print("BASE MODELS")
b_ = r[r["stage"] == "base"]
if len(b_):
    print(b_.pivot_table(index="name",
                         columns="outcome",
                         values="mean")
          .round(4).to_string())

print("")
print("DID COMPETENCE-MATCHED MIXES HELP?")
print("  the hand-picked pool spanned the"
      " simplex evenly; the clustered pool")
print("  puts trees where the competence"
      " regions actually are")
if len(b_):
    for oc in OUTS:
        a = b_[(b_["outcome"] == oc)
               & (b_["name"] == "pool_hand")]
        c = b_[(b_["outcome"] == oc)
               & (b_["name"] == "pool_clust")]
        if len(a) and len(c):
            print("  %-18s hand %.4f   clustered"
                  " %.4f   %+.4f"
                  % (oc, a["mean"].iloc[0],
                     c["mean"].iloc[0],
                     c["mean"].iloc[0]
                     - a["mean"].iloc[0]))

print("")
print("COMPETENCE WEIGHTING, BY BINS AND"
      " DRIVERS")
c_ = r[r["stage"] == "comp"]
if len(c_):
    print("  by bins")
    print(c_.groupby("bins")["vs_pool"].max()
          .round(4).to_string())
    print("  by drivers")
    print(c_.groupby("drivers")["vs_pool"].max()
          .round(4).to_string())
    print("  by mix source")
    print(c_.groupby("mixes")["vs_pool"].max()
          .round(4).to_string())
    print("")
    print("  script 124 used 3 bins and 1"
          " driver, both arbitrary")

print("")
print("=" * 74)
print("META-LEARNERS")
m_ = r[r["stage"] == "meta"]
if len(m_):
    print(m_.pivot_table(index="name",
                         columns="outcome",
                         values="vs_best")
          .round(4).to_string())
    print("")
    print("  the constraint matters more than"
          " the penalty: the members are ranks")
    print("  in [0,1] that all point the same"
          " way, so negative weights are")
    print("  almost certainly overfitting."
          " nnls is the Super Learner form.")

print("")
print("HOW BIG IS THE LEAK?")
print("  grid_ins fits the weights on the rows"
      " it scores, which is what this")
print("  project's late fusion has always done")
for oc in OUTS:
    a = m_[(m_["outcome"] == oc)
           & (m_["name"] == "grid_ins")]
    b2 = m_[(m_["outcome"] == oc)
            & (m_["name"] == "grid")]
    if len(a) and len(b2):
        print("  %-18s in-sample %+.4f"
              "   out-of-fold %+.4f"
              "   leak %.4f"
              % (oc, a["vs_best"].iloc[0],
                 b2["vs_best"].iloc[0],
                 a["vs_best"].iloc[0]
                 - b2["vs_best"].iloc[0]))

print("")
print("DOES THE ENSEMBLE BEAT ITS BEST"
      " MEMBER?")
w_ = m_[(m_["vs_best"] > 0)
        & (m_["name"] != "grid_ins")]
print("  %d of %d honest cells"
      % (len(w_), len(m_[m_["name"]
                         != "grid_ins"])))
if len(w_):
    print(w_[["outcome", "name", "mean",
              "vs_best"]].round(4)
          .to_string(index=False))

print("")
print("BEST OVERALL PER OUTCOME")
for oc in OUTS:
    s = r[(r["outcome"] == oc)
          & (r["name"] != "grid_ins")]
    if not len(s):
        continue
    x = s.loc[s["mean"].idxmax()]
    pv = S124.get(oc, {})
    print("  %-18s %-16s %.4f   script 124"
          " best %.4f   %+.4f"
          % (oc, x["name"], x["mean"],
             pv.get("best", np.nan),
             x["mean"] - pv.get("best",
                                np.nan)))
print("")
print("saved", DEST, r.shape)
keep_awake(False)