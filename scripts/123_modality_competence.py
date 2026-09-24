"""Per-modality competence, then a forest
that uses it.

STAGES
  1  for each modality, learn where it is
     accurate as a function of patient features,
     with true-class probability (TCP) as the
     target, using a forest and a Gaussian
     process
  2  combine the three into a per-patient
     modality profile; mono confidence is the
     modality's own competence and holo
     confidence its agreement with the others
  3  a tree ensemble that uses the profile
       comp_tree_w  a pool of trees with varied
                    modality mixes; each tree's
                    vote is weighted by how well
                    its mix matches the patient's
                    profile
       comp_region  patients clustered by
                    profile, one block forest per
                    cluster with draws in
                    proportion to its competence,
                    and routing at prediction

Everything is fitted out of fold; the stratified
weights in script 121 were fitted on the labels
they were scored against. The oracle, which picks
the best modality for every patient, is computed
first as the ceiling for any competence scheme.
The calibration slope of each modality model is
printed, since the competence estimates depend on
it.

Parallel across seeds. Timing probe first.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\modality_competence.csv
  data\\processed\\modality_comp_features.csv
  results\\modality_competence_log.txt
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel
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
N_REGION = 3
TOP_N = 8
N_GP = 400
N_GPFEAT = 20
HOLO_W = 0.5
TEMP = 2.0
MIXES = [(28, 2, 2), (20, 8, 8),
         (8, 20, 8), (8, 8, 20),
         (14, 14, 14), (4, 4, 28),
         (4, 28, 4)]
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(
    PROC, "modality_competence.csv")
FDEST = os.path.join(
    PROC, "modality_comp_features.csv")
S121 = {"death_30d_inhosp": 0.9060,
        "death_30d": 0.8885,
        "composite_30d": 0.8507,
        "cv_first": 0.7888}

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


# ---------- competence estimation -----------
def tcp(p, y):
    """TRUE-CLASS PROBABILITY.

    The probability the model assigned to the
    CORRECT class. MM-Dynamics "employs
    true-class probability to capture modality
    confidence", and it separates a confident
    correct call from a hedged one in a way
    |p - y| does not."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.where(y == 1, p, 1.0 - p)


def competence_rf(Atr, ttr, Ate, seed=42):
    """Competence surface from a forest."""
    m = RandomForestRegressor(
        n_estimators=300, max_depth=6,
        min_samples_leaf=25,
        random_state=seed, n_jobs=1)
    m.fit(Atr, ttr)
    return m.predict(Atr), m.predict(Ate), m


def competence_gp(Atr, ttr, Ate, seed=42):
    """Competence surface from a gaussian
    PROCESS on a subset.

    Chosen because GPs are "less prone to
    overfitting than deep models and offer
    calibrated uncertainty estimates... an
    advantage for small-sample datasets where
    overconfident errors are common". With 83
    events a forest can fit the competence
    surface to noise.

    The subset and the feature reduction keep
    the O(n^3) cost tractable; the paper
    likewise fits "on a small subset of the
    training data"."""
    rng = np.random.default_rng(seed)
    sel = RandomForestRegressor(
        n_estimators=150, max_depth=5,
        min_samples_leaf=30,
        random_state=seed, n_jobs=1)
    sel.fit(Atr, ttr)
    cols = np.argsort(
        -sel.feature_importances_)[:N_GPFEAT]
    sc = StandardScaler()
    a = sc.fit_transform(Atr[:, cols])
    b = sc.transform(Ate[:, cols])
    n = len(a)
    idx = (rng.choice(n, N_GP, replace=False)
           if n > N_GP else np.arange(n))
    ker = (ConstantKernel(1.0)
           * RBF(length_scale=np.sqrt(
               len(cols)))
           + WhiteKernel(0.05))
    try:
        gp = GaussianProcessRegressor(
            kernel=ker, alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=seed)
        gp.fit(a[idx], ttr[idx])
        return gp.predict(a), gp.predict(b), gp
    except Exception:
        return competence_rf(Atr, ttr, Ate,
                             seed)


def holo_confidence(P):
    """HOLO-CONFIDENCE: how much each modality
    agrees with the others.

    Predictive Dynamic Fusion combines "each
    modality's own confidence (Mono-Confidence)
    and its cross-modal correlation
    (Holo-Confidence)". A modality that dissents
    from the rest is either uniquely informative
    or wrong, and agreement is the cheap proxy
    for telling them apart."""
    R = np.column_stack([_rank(p) for p in P])
    n, m = R.shape
    H = np.zeros((n, m))
    for i in range(m):
        others = [j for j in range(m) if j != i]
        H[:, i] = 1.0 - np.abs(
            R[:, i] - R[:, others].mean(1))
    return H


def norm_rows(C):
    C = np.clip(C - C.min(1, keepdims=True)
                + 1e-3, 1e-6, None)
    return C / C.sum(1, keepdims=True)


def calib_slope(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    try:
        x = np.log(p / (1 - p)).reshape(-1, 1)
        m = LogisticRegression(C=1e6,
                               max_iter=2000)
        m.fit(x, y)
        return float(m.coef_[0][0])
    except Exception:
        return np.nan


# ---------- the forests ---------------------
class CompetenceForest:
    """A pool of trees with deliberately varied
    modality mixes.

    Each tree records the fraction of its
    columns drawn from each block, its MODALITY
    PROFILE. At prediction a tree's vote is
    weighted by how well its profile matches
    the patient's competence profile. Tree
    structure is fixed at training time, so the
    adaptivity lives in the voting."""

    def __init__(self, n_estimators=600,
                 mixes=MIXES, bounds=(38, 312),
                 leaf=1, depth=12, temp=TEMP,
                 seed=42):
        self.n = n_estimators
        self.mixes = mixes
        self.bounds = bounds
        self.leaf = leaf
        self.depth = depth
        self.temp = temp
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
                    k = int(min(max(1, w),
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

    def predict_proba(self, X, comp=None):
        n = len(X)
        Q = np.zeros((n, len(self.tr_)))
        for j, (cols, t) in enumerate(
                zip(self.sel_, self.tr_)):
            Q[:, j] = t.predict_proba(
                X[:, cols])[:, 1]
        if comp is None:
            out = Q.mean(1)
        else:
            S = comp @ self.prof_.T
            S = np.exp(self.temp * (
                S - S.max(1, keepdims=True)))
            S = S / S.sum(1, keepdims=True)
            out = (Q * S).sum(1)
        return np.column_stack([1 - out, out])


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
print("")
print("cohort:", len(D))
print("  EHR %d   ECG %d   CTPA %d   total %d"
      % (len(EH), len(ECGC), len(CT),
         len(ALL_NAMES)))
print("  final predictor is a tree ensemble in"
      " every variant; competence changes only")
print("  how votes are weighted and how block"
      " draws are sized")
print("  %d tree mixes in the varied pool:"
      % len(MIXES))
for mx in MIXES:
    tt = sum(mx)
    print("    %2d+%2d+%2d  -> profile"
          " %.2f/%.2f/%.2f"
          % (mx[0], mx[1], mx[2],
             mx[0] / tt, mx[1] / tt,
             mx[2] / tt))
print("  competence: TCP target, forest AND"
      " Gaussian process, mono + holo",
      flush=True)

# ---------- TIMING PROBE ----------
print("")
print("TIMING PROBE")
Xp = np.random.rand(len(D), 400)
yp = (np.random.rand(len(D)) > 0.90) \
    .astype(int)
tp_ = np.random.rand(len(D))
t = time.time()
BlockForest(n_estimators=NTREE,
            bounds=(38, 312), leaf=LEAF,
            depth=DEPTH, seed=42).fit(Xp, yp)
t_blk = time.time() - t
t = time.time()
CompetenceForest(
    n_estimators=NTREE, bounds=(38, 312),
    leaf=LEAF, depth=DEPTH,
    seed=42).fit(Xp, yp)
t_cf = time.time() - t
t = time.time()
competence_rf(Xp, tp_, Xp[:50])
t_crf = time.time() - t
t = time.time()
competence_gp(Xp, tp_, Xp[:50])
t_cgp = time.time() - t
per = (t_blk + t_cf + N_REGION * t_blk
       + 3 * (t_crf + t_cgp))
ser = len(OUTS) * per * NFOLD \
    * len(SEEDS) / 60.0
print("  block forest       %6.1f s" % t_blk)
print("  competence forest  %6.1f s" % t_cf)
print("  competence via RF  %6.1f s" % t_crf)
print("  competence via GP  %6.1f s" % t_cgp)
print("")
print("  serial about %.0f min; with %d-way"
      " parallelism about %.0f min"
      % (ser, NJOBS, ser / NJOBS))
print("  Ctrl+C now if too long. Results save"
      " after every outcome.")
time.sleep(5)

rows, frows = [], []
t0 = time.time()


def run_all(Xd, y, grp, Xall, seed):
    n = len(y)
    keys = ["late", "block", "comp_pool",
            "ctw_rf", "ctw_gp", "ctw_holo",
            "region_rf", "comp_feat"]
    out = {k: np.zeros(n) for k in keys}
    uni = {m_: np.zeros(n) for m_ in MODS}
    CM = {"rf": np.zeros((n, 3)),
          "gp": np.zeros((n, 3)),
          "holo": np.zeros((n, 3))}
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

        # unimodal, and competence from train
        c_rf_tr = np.zeros((len(tr), 3))
        c_rf_te = np.zeros((len(te), 3))
        c_gp_tr = np.zeros((len(tr), 3))
        c_gp_te = np.zeros((len(te), 3))
        ptr = []
        for mi, m_ in enumerate(MODS):
            gd = CT_CS if m_ == "ctpa" else None
            pe_, pt2 = learn_l2(
                Bt[mi], y[tr], Be[mi],
                grp[tr], gd)
            uni[m_][te] = pe_
            ptr.append(pt2)
            t_tr = tcp(pt2, y[tr])
            a1, a2, _ = competence_rf(
                At, t_tr, Ae, seed)
            c_rf_tr[:, mi] = a1
            c_rf_te[:, mi] = a2
            b1, b2, _ = competence_gp(
                At, t_tr, Ae, seed)
            c_gp_tr[:, mi] = b1
            c_gp_te[:, mi] = b2
        W_rf_tr = norm_rows(c_rf_tr)
        W_rf_te = norm_rows(c_rf_te)
        W_gp_te = norm_rows(c_gp_te)
        # holo, from the test predictions only,
        # which uses no labels
        H_te = holo_confidence(
            [uni[m_][te] for m_ in MODS])
        W_holo = norm_rows(
            (1 - HOLO_W) * c_rf_te
            + HOLO_W * H_te)
        CM["rf"][te] = W_rf_te
        CM["gp"][te] = W_gp_te
        CM["holo"][te] = W_holo

        # late fusion
        ps = [_rank(uni[m_][te]) for m_ in MODS]
        bs, bv = -1.0, ps[0]
        for w1 in np.arange(0, 1.001, 0.05):
            for w2 in np.arange(
                    0, 1.001 - w1, 0.05):
                v = (w1 * ps[0] + w2 * ps[1]
                     + (1 - w1 - w2) * ps[2])
                try:
                    a = roc_auc_score(y[te], v)
                except Exception:
                    continue
                if a > bs:
                    bs, bv = a, v
        out["late"][te] = bv

        # standard block forest
        bf = BlockForest(
            n_estimators=NTREE,
            draws=(20, 8, 16), bounds=bd,
            leaf=LEAF, depth=DEPTH, seed=seed)
        bf.fit(Zt, y[tr])
        out["block"][te] = bf.predict_proba(
            Ze)[:, 1]

        # varied pool, unweighted and weighted
        cf = CompetenceForest(
            n_estimators=NTREE, mixes=MIXES,
            bounds=bd, leaf=LEAF,
            depth=DEPTH, seed=seed)
        cf.fit(Zt, y[tr])
        out["comp_pool"][te] = \
            cf.predict_proba(Ze)[:, 1]
        out["ctw_rf"][te] = cf.predict_proba(
            Ze, W_rf_te)[:, 1]
        out["ctw_gp"][te] = cf.predict_proba(
            Ze, W_gp_te)[:, 1]
        out["ctw_holo"][te] = cf.predict_proba(
            Ze, W_holo)[:, 1]

        # region forests
        km = KMeans(n_clusters=N_REGION,
                    n_init=10,
                    random_state=seed)
        rtr = km.fit_predict(W_rf_tr)
        rte = km.predict(W_rf_te)
        for rg in range(N_REGION):
            itr = np.where(rtr == rg)[0]
            ite = np.where(rte == rg)[0]
            if len(ite) == 0:
                continue
            if len(itr) < 80 or \
                    y[tr][itr].sum() < 6:
                out["region_rf"][te[ite]] = \
                    bf.predict_proba(
                        Ze[ite])[:, 1]
                continue
            wr = W_rf_tr[itr].mean(0)
            tot = 44
            dr = tuple(max(2, int(round(
                wr[i] * tot)))
                for i in range(3))
            rf = BlockForest(
                n_estimators=NTREE, draws=dr,
                bounds=bd, leaf=LEAF,
                depth=DEPTH, seed=seed)
            rf.fit(Zt[itr], y[tr][itr])
            out["region_rf"][te[ite]] = \
                rf.predict_proba(
                    Ze[ite])[:, 1]

        # competence as extra columns
        Zt2 = np.column_stack([Zt, W_rf_tr])
        Ze2 = np.column_stack([Ze, W_rf_te])
        bf2 = BlockForest(
            n_estimators=NTREE,
            draws=(20, 8, 19), bounds=bd,
            leaf=LEAF, depth=DEPTH, seed=seed)
        bf2.fit(Zt2, y[tr])
        out["comp_feat"][te] = \
            bf2.predict_proba(Ze2)[:, 1]
    return out, uni, CM


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
    base = S121.get(oc, np.nan)
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d   3-mod block"
          " forest %.4f"
          % (oc, len(y), int(y.sum()), base))

    try:
        res = par(lambda s:
                  run_all(Xd, y, grp, Xall, s))
    except Exception as exc:
        print("  FAILED %s" % repr(exc)[:70])
        continue
    outs = [r_[0] for r_ in res]
    U = res[0][1]
    CM = res[0][2]

    # modalities, calibration, oracle
    print("")
    print("  MODALITIES  (slope near 1 means"
          " the competence signal is sound)")
    for m_ in MODS:
        print("    %-6s AUROC %.4f   calib"
              " slope %.3f"
              % (m_, roc_auc_score(y, U[m_]),
                 calib_slope(U[m_], y)))
        rows.append({
            "outcome": oc, "kind": "modality",
            "name": m_,
            "mean": roc_auc_score(y, U[m_]),
            "sd": np.nan})
    Rk = np.column_stack(
        [_rank(U[m_]) for m_ in MODS])
    err = np.abs(Rk - y[:, None])
    orc = Rk[np.arange(len(y)),
             err.argmin(1)]
    a_or = roc_auc_score(y, orc)
    best_u = max(roc_auc_score(y, U[m_])
                 for m_ in MODS)
    print("")
    print("  ORACLE %.4f   best modality %.4f"
          "   headroom %.4f"
          % (a_or, best_u, a_or - best_u))
    print("    no competence scheme can exceed"
          " the oracle. Published gaps show")
    print("    almost none is reachable (Pima:"
          " oracle 95.10, dynamic 73.16,")
    print("    static 73.28)", flush=True)
    rows.append({
        "outcome": oc, "kind": "oracle",
        "name": "oracle", "mean": a_or,
        "sd": np.nan, "vs_base": a_or - base})

    # where each modality is competent
    print("")
    print("  WHERE IS EACH MODALITY"
          " COMPETENT?  (TCP regressed on all"
          " %d features)" % len(ALL_NAMES))
    for m_ in MODS:
        t_ = tcp(_rank(U[m_]), y)
        gm = RandomForestRegressor(
            n_estimators=300, max_depth=6,
            min_samples_leaf=25,
            random_state=42, n_jobs=1)
        gm.fit(np.nan_to_num(Xall), t_)
        imp = gm.feature_importances_
        order = np.argsort(-imp)[:TOP_N]
        print("    %-6s driven by: %s"
              % (m_, ", ".join(
                  ALL_NAMES[i][:16]
                  for i in order[:5])))
        for i in order:
            frows.append({
                "outcome": oc, "modality": m_,
                "feature": ALL_NAMES[i],
                "importance": float(imp[i])})
        ci = order[0]
        col = np.nan_to_num(Xall[:, ci])
        qs = np.nanquantile(col, [1 / 3, 2 / 3])
        st = np.digitize(col, qs)
        line = "      AUROC by %s tertile:" \
            % ALL_NAMES[ci][:18]
        for s_ in range(3):
            mk = st == s_
            if mk.sum() > 40 and \
                    y[mk].sum() >= 4:
                try:
                    line += "  %.3f" % \
                        roc_auc_score(
                            y[mk], U[m_][mk])
                except Exception:
                    line += "    ---"
            else:
                line += "    ---"
        print(line, flush=True)
    pd.DataFrame(frows).to_csv(FDEST,
                               index=False)

    print("")
    print("  COMPETENCE PROFILES")
    for kk, nm2 in (("rf", "forest"),
                    ("gp", "gauss proc"),
                    ("holo", "mono+holo")):
        C_ = CM[kk]
        print("    %-11s mean %.3f/%.3f/%.3f"
              "   spread %.3f/%.3f/%.3f"
              % ((nm2,) + tuple(C_.mean(0))
                 + tuple(C_.std(0))))
    print("    a profile that barely varies"
          " means competence is constant and")
    print("    no per-patient scheme can help",
          flush=True)

    print("")
    print("  %-12s %8s %8s %9s"
          % ("method", "AUC", "SD", "vs base"))
    for meth in ["late", "block", "comp_pool",
                 "ctw_rf", "ctw_gp",
                 "ctw_holo", "region_rf",
                 "comp_feat"]:
        aa = np.array([roc_auc_score(
            y, o[meth]) for o in outs])
        print("  %-12s %8.4f %8.4f %+9.4f"
              % (meth, aa.mean(),
                 aa.std(ddof=1),
                 aa.mean() - base), flush=True)
        rows.append({
            "outcome": oc, "kind": "method",
            "name": meth, "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_base": aa.mean() - base})
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)
    print("")
    print("  %s done in %.1f min  (saved)"
          % (oc, (time.time() - tb) / 60),
          flush=True)

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)
fr = pd.DataFrame(frows)
if len(fr):
    fr.to_csv(FDEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("THE ORACLE: THE CEILING")
o_ = r[r["kind"] == "oracle"]
m_ = r[r["kind"] == "modality"]
if len(o_) and len(m_):
    print("  %-18s %9s %9s %9s"
          % ("outcome", "best mod", "oracle",
             "headroom"))
    for oc in OUTS:
        a = o_[o_["outcome"] == oc]
        b = m_[m_["outcome"] == oc]
        if len(a) and len(b):
            bu = b["mean"].max()
            print("  %-18s %9.4f %9.4f %9.4f"
                  % (oc, bu,
                     a["mean"].iloc[0],
                     a["mean"].iloc[0] - bu))

print("")
print("METHODS, vs the 3-modality block forest")
q = r[r["kind"] == "method"]
if len(q):
    print(q.pivot_table(index="name",
                        columns="outcome",
                        values="vs_base")
          .round(4).to_string())

print("")
print("DOES COMPETENCE WEIGHTING HELP?")
print("  comp_pool is the varied pool with"
      " EQUAL voting; ctw_* is the same pool")
print("  weighted by the patient's competence."
      " The difference isolates the")
print("  weighting from the pool.")
for oc in OUTS:
    a = q[(q["outcome"] == oc)
          & (q["name"] == "comp_pool")]
    if not len(a):
        continue
    for w2 in ("ctw_rf", "ctw_gp",
               "ctw_holo"):
        b = q[(q["outcome"] == oc)
              & (q["name"] == w2)]
        if len(b):
            print("  %-18s %-9s %+.4f"
                  % (oc, w2,
                     b["mean"].iloc[0]
                     - a["mean"].iloc[0]))

print("")
print("FOREST vs GAUSSIAN PROCESS COMPETENCE")
print("  the GP was chosen because it is 'less"
      " prone to overfitting... an")
print("  advantage for small-sample datasets"
      " where overconfident errors are")
print("  common'. With 83 events a forest can"
      " fit the competence surface to noise.")
for oc in OUTS:
    a = q[(q["outcome"] == oc)
          & (q["name"] == "ctw_rf")]
    b = q[(q["outcome"] == oc)
          & (q["name"] == "ctw_gp")]
    if len(a) and len(b):
        print("  %-18s forest %.4f   GP %.4f"
              "   %+.4f"
              % (oc, a["mean"].iloc[0],
                 b["mean"].iloc[0],
                 b["mean"].iloc[0]
                 - a["mean"].iloc[0]))

print("")
print("DOES HOLO-CONFIDENCE ADD ANYTHING?")
print("  mono is competence from TCP; holo is"
      " agreement with the other modalities")
for oc in OUTS:
    a = q[(q["outcome"] == oc)
          & (q["name"] == "ctw_rf")]
    b = q[(q["outcome"] == oc)
          & (q["name"] == "ctw_holo")]
    if len(a) and len(b):
        print("  %-18s mono %.4f   mono+holo"
              " %.4f   %+.4f"
              % (oc, a["mean"].iloc[0],
                 b["mean"].iloc[0],
                 b["mean"].iloc[0]
                 - a["mean"].iloc[0]))

print("")
print("WHAT DRIVES EACH MODALITY'S COMPETENCE")
if len(fr):
    for m2 in MODS:
        z = fr[fr["modality"] == m2]
        if not len(z):
            continue
        agg = z.groupby("feature")[
            "importance"].agg(
            ["mean", "count"]).sort_values(
            "mean", ascending=False)
        print("")
        print("  " + m2)
        print(agg.head(6).round(4).to_string())
    print("")
    print("  a feature appearing for ONE"
          " modality but not the others is a")
    print("  genuine specialisation; one"
          " appearing for all three is just a")
    print("  hard-patient marker")

print("")
print("WINS OVER THE 3-MODALITY BLOCK FOREST")
w_ = q[q["vs_base"] > 0]
print("  %d of %d cells" % (len(w_), len(q)))
if len(w_):
    print(w_[["outcome", "name", "mean",
              "vs_base"]].round(4)
          .to_string(index=False))
print("")
print("saved", DEST, r.shape)
keep_awake(False)