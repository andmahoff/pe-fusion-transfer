"""Sparsity, temperature and alternative
competence measures for the competence weighting.

In script 123 the competence profiles varied by
only 0.006 to 0.026 around 0.33, so the weights
were nearly uniform. Single-feature tertiles
showed much larger differences: CTPA AUROC fell
from 0.809 to 0.615 across heart-failure
tertiles, ECG from 0.783 to 0.661 across
haemoglobin, and EHR from 0.865 to 0.786 across
RDW.

FOUR CHANGES
  FIX 1  sparsity: the competence model uses the
         top k features per modality, k in 1, 3,
         5 and 10, with all features as control
  FIX 2  temperature: the softmax temperature,
         fixed at 2.0 in script 123, swept at 1,
         2, 5, 10, 20 and a hard argmax
  FIX 3  competence measure: TCP, absolute error,
         tertile competence and local
         neighbourhood accuracy
  FIX 4  amplification: each profile is rescaled
         to a target spread before the softmax,
         separating a flat estimate from a wrong
         one

Two outcomes only, the two with the largest
competence spread.

Parallel across seeds. Timing probe first.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\sparse_competence.csv
  results\\sparse_competence_log.txt
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
from sklearn.neighbors import NearestNeighbors
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
SPARSE_K = [1, 3, 5, 10, 0]
TEMPS = [1.0, 2.0, 5.0, 10.0, 20.0, -1.0]
MEASURES = ["tcp", "err", "tertile", "local"]
AMPS = [0.0, 0.10, 0.25]
KNN = 40
MIXES = [(28, 2, 2), (20, 8, 8),
         (8, 20, 8), (8, 8, 20),
         (14, 14, 14), (4, 4, 28),
         (4, 28, 4)]
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
# the two outcomes with the largest competence
# spread in script 123
OUTS = ["death_30d_inhosp", "death_30d"]
DEST = os.path.join(
    PROC, "sparse_competence.csv")
# script 123: block forest, and comp_pool
S123 = {"death_30d_inhosp":
        {"block": 0.9053, "pool": 0.9038},
        "death_30d":
        {"block": 0.8866, "pool": 0.8883}}

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


# ---------- competence measures -------------
def comp_target(p, y, measure, Xs=None):
    """FIX 3: four of the eight measures the
    dynamic-selection review surveys, since
    "choosing the best measure to calculate the
    level of competence correctly is not
    straightforward".

    tcp      true-class probability, script
             123's choice
    err      1 minus absolute error
    tertile  DISCRETE competence: the modality's
             AUROC within the patient's tertile
             of the driving feature. This is the
             form that demonstrably works, since
             the tertile diagnostic recovered a
             0.19 spread where the smooth model
             recovered 0.02.
    local    accuracy in the k-nearest
             neighbourhood, the OLA measure"""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if measure == "tcp":
        return np.where(y == 1, p, 1.0 - p)
    if measure == "err":
        return 1.0 - np.abs(p - y)
    if measure == "local" and Xs is not None:
        n = len(y)
        nn = NearestNeighbors(
            n_neighbors=min(KNN + 1,
                            n)).fit(Xs)
        _, idx = nn.kneighbors(Xs)
        idx = idx[:, 1:]
        e = np.abs(p - y)
        return 1.0 - e[idx].mean(1)
    # tertile: computed by the caller, since it
    # needs the driving feature
    return np.where(y == 1, p, 1.0 - p)


def tertile_competence(p, y, col, n_bins=3):
    """DISCRETE competence: each patient gets
    their tertile's AUROC as the competence
    value.

    This is the exact form of the script 123
    diagnostic that worked. It cannot be flat
    unless the modality genuinely performs the
    same everywhere."""
    qs = np.nanquantile(
        col, np.linspace(0, 1, n_bins + 1)[1:-1])
    st = np.digitize(col, qs)
    out = np.full(len(y), 0.5)
    for s in range(n_bins):
        m = st == s
        if m.sum() > 30 and 0 < y[m].sum() \
                < m.sum():
            try:
                out[m] = roc_auc_score(y[m],
                                       p[m])
            except Exception:
                pass
    return out, st, qs


def amplify(C, target):
    """FIX 4: rescale each profile so its spread
    matches a target.

    Separates 'the competence estimate is flat'
    from 'the competence estimate is wrong'. If
    amplification helps, the ordering was right
    and only the magnitude was too small."""
    if target <= 0:
        return C
    mu = C.mean(1, keepdims=True)
    sd = C.std(1, keepdims=True)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return mu + (C - mu) * (target / sd)


def to_weights(C, temp):
    """temp < 0 means a hard argmax, the fully
    sparse limit."""
    C = np.clip(C - C.min(1, keepdims=True)
                + 1e-6, 1e-9, None)
    C = C / C.sum(1, keepdims=True)
    if temp < 0:
        W = np.zeros_like(C)
        W[np.arange(len(C)), C.argmax(1)] = 1.0
        return W
    S = np.exp(temp * (C - C.max(
        1, keepdims=True)))
    return S / S.sum(1, keepdims=True)


class CompetenceForest:
    """A pool of trees with varied modality
    mixes. Each records its modality profile,
    and votes are weighted by how well that
    profile matches the patient's competence."""

    def __init__(self, n_estimators=600,
                 mixes=MIXES, bounds=(38, 312),
                 leaf=1, depth=12, seed=42):
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
        if temp < 0:
            out = np.zeros(len(Q))
            am = S.argmax(1)
            for i in range(len(Q)):
                out[i] = Q[i, am[i]]
            return out
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
print("")
print("  script 123 profile spread was 0.006"
      " to 0.026; the tertile diagnostic")
print("  recovered 0.19. A ONE-feature"
      " competence model works where a")
print("  338-feature one does not, so sparsity"
      " is the main fix here.")
print("  sparse k:", SPARSE_K, " (0 = all)")
print("  temperatures:", TEMPS,
      " (-1 = hard argmax)")
print("  measures:", MEASURES)
print("  amplification targets:", AMPS,
      flush=True)

# ---------- TIMING PROBE ----------
print("")
print("TIMING PROBE")
Xp = np.random.rand(len(D), 400)
yp = (np.random.rand(len(D)) > 0.90) \
    .astype(int)
t = time.time()
cfp = CompetenceForest(
    n_estimators=NTREE, bounds=(38, 312),
    leaf=LEAF, depth=DEPTH, seed=42)
cfp.fit(Xp, yp)
t_cf = time.time() - t
t = time.time()
cfp.tree_preds(Xp[:400])
t_pred = time.time() - t
t = time.time()
RandomForestRegressor(
    n_estimators=200, max_depth=5,
    min_samples_leaf=30, n_jobs=1,
    random_state=42).fit(
    Xp, np.random.rand(len(D)))
t_comp = time.time() - t
# one fold: one pool fit, 3 modality models,
# and competence for each k and measure
ncomp = len(SPARSE_K) * len(MEASURES) * 3
per = t_cf + t_pred + ncomp * t_comp * 0.3
ser = len(OUTS) * per * NFOLD \
    * len(SEEDS) / 60.0
print("  competence pool fit  %6.1f s" % t_cf)
print("  tree predictions     %6.1f s"
      % t_pred)
print("  one competence model %6.1f s"
      % t_comp)
print("")
print("  the %d weighting combinations reuse"
      " ONE pool fit per fold, so they are"
      % (len(SPARSE_K) * len(TEMPS)
         * len(MEASURES) * len(AMPS)))
print("  nearly free once the trees exist")
print("  serial about %.0f min; with %d-way"
      " parallelism about %.0f min"
      % (ser, NJOBS, ser / NJOBS))
print("  Ctrl+C now if too long. Results save"
      " after every outcome.")
time.sleep(5)

rows = []
t0 = time.time()


def run_seed(Xd, y, grp, Xall, seed):
    """One seed. The pool is fitted once per
    fold and every weighting combination reuses
    its tree predictions, so the grid costs
    almost nothing beyond the forests."""
    n = len(y)
    combos = [(sk, ms, tp_, am)
              for sk in SPARSE_K
              for ms in MEASURES
              for tp_ in TEMPS
              for am in AMPS]
    P = {c: np.zeros(n) for c in combos}
    P["pool"] = np.zeros(n)
    P["block"] = np.zeros(n)
    spread = {}
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
        Ats = StandardScaler().fit_transform(At)

        cf = CompetenceForest(
            n_estimators=NTREE, mixes=MIXES,
            bounds=bd, leaf=LEAF,
            depth=DEPTH, seed=seed)
        cf.fit(Zt, y[tr])
        Q = cf.tree_preds(Ze)
        P["pool"][te] = cf.combine(Q, None, 1.0)

        bf = BlockForest(
            n_estimators=NTREE,
            draws=(20, 8, 16), bounds=bd,
            leaf=LEAF, depth=DEPTH, seed=seed)
        bf.fit(Zt, y[tr])
        P["block"][te] = bf.predict_proba(
            Ze)[:, 1]

        # per-modality training predictions
        ptr, drivers = [], []
        for mi, m_ in enumerate(MODS):
            gd = CT_CS if m_ == "ctpa" else None
            _, pt2 = learn_l2(
                Bt[mi], y[tr], Be[mi],
                grp[tr], gd)
            ptr.append(_rank(pt2))
            # driving feature for this modality,
            # from train only
            gm = RandomForestRegressor(
                n_estimators=200, max_depth=5,
                min_samples_leaf=30,
                random_state=seed, n_jobs=1)
            gm.fit(At, comp_target(
                pt2, y[tr], "tcp"))
            drivers.append(int(np.argmax(
                gm.feature_importances_)))

        # competence per (sparse k, measure)
        CC = {}
        for sk in SPARSE_K:
            for ms in MEASURES:
                Ctr = np.zeros((len(tr), 3))
                Cte = np.zeros((len(te), 3))
                for mi in range(3):
                    if ms == "tertile":
                        ci = drivers[mi]
                        c_tr, st, qs = \
                            tertile_competence(
                                ptr[mi], y[tr],
                                At[:, ci])
                        ste = np.digitize(
                            Ae[:, ci], qs)
                        lut = {}
                        for s_ in range(3):
                            mk = st == s_
                            lut[s_] = (
                                c_tr[mk][0]
                                if mk.sum()
                                else 0.5)
                        Ctr[:, mi] = c_tr
                        Cte[:, mi] = np.array(
                            [lut.get(int(v),
                                     0.5)
                             for v in ste])
                        continue
                    t_tr = comp_target(
                        ptr[mi], y[tr], ms,
                        Ats)
                    if sk == 0:
                        cols_ = np.arange(
                            At.shape[1])
                    else:
                        gm2 = \
                            RandomForestRegressor(
                                n_estimators=150,
                                max_depth=5,
                                min_samples_leaf=30,
                                random_state=seed,
                                n_jobs=1)
                        gm2.fit(At, t_tr)
                        cols_ = np.argsort(
                            -gm2.feature_importances_
                        )[:sk]
                    md = RandomForestRegressor(
                        n_estimators=200,
                        max_depth=4,
                        min_samples_leaf=30,
                        random_state=seed,
                        n_jobs=1)
                    md.fit(At[:, cols_], t_tr)
                    Ctr[:, mi] = md.predict(
                        At[:, cols_])
                    Cte[:, mi] = md.predict(
                        Ae[:, cols_])
                CC[(sk, ms)] = Cte
                key = (sk, ms)
                spread.setdefault(key, [])
                W0 = to_weights(Cte, 1.0)
                spread[key].append(
                    float(W0.std(0).mean()))

        for (sk, ms, tp_, am) in combos:
            Cte = CC[(sk, ms)]
            Ca = amplify(Cte, am)
            W = to_weights(Ca, tp_)
            P[(sk, ms, tp_, am)][te] = \
                cf.combine(Q, W, tp_)
    return P, spread


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
    pv = S123.get(oc, {})
    base = pv.get("pool", np.nan)
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())))
    print("  script 123: block %.4f   pool"
          " %.4f   (pool is the baseline here,"
          % (pv.get("block", np.nan), base))
    print("  since every variant uses the same"
          " pool and differs only in weighting)",
          flush=True)

    try:
        res = par(lambda s:
                  run_seed(Xd, y, grp, Xall, s))
    except Exception as exc:
        print("  FAILED %s" % repr(exc)[:70])
        continue
    Ps = [r_[0] for r_ in res]
    SPs = [r_[1] for r_ in res]

    for nm in ("pool", "block"):
        aa = np.array([roc_auc_score(
            y, p[nm]) for p in Ps])
        print("  %-10s %.4f (SD %.4f)"
              % (nm, aa.mean(),
                 aa.std(ddof=1)))
        rows.append({
            "outcome": oc, "sparse_k": np.nan,
            "measure": nm, "temp": np.nan,
            "amp": np.nan, "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_pool": aa.mean() - base,
            "spread": np.nan})

    # profile spread, the diagnostic
    print("")
    print("  PROFILE SPREAD BY SPARSITY AND"
          " MEASURE")
    print("  script 123 got 0.006 to 0.026."
          " Higher here means sparsity")
    print("  recovered the variation the full"
          " model could not.")
    print("  %-8s %-9s %9s" % ("sparse_k",
                               "measure",
                               "spread"))
    for sk in SPARSE_K:
        for ms in MEASURES:
            vals = [np.mean(sp[(sk, ms)])
                    for sp in SPs
                    if (sk, ms) in sp]
            if vals:
                print("  %-8s %-9s %9.4f"
                      % ("all" if sk == 0
                         else sk, ms,
                         float(np.mean(vals))),
                      flush=True)

    print("")
    print("  WEIGHTING RESULTS, vs the"
          " unweighted pool")
    best = None
    for sk in SPARSE_K:
        for ms in MEASURES:
            line = "  k=%-4s %-8s" % (
                "all" if sk == 0 else sk, ms)
            for tp_ in TEMPS:
                vals = []
                for am in AMPS:
                    aa = np.array([
                        roc_auc_score(
                            y, p[(sk, ms, tp_,
                                  am)])
                        for p in Ps])
                    vals.append(aa.mean())
                    sp = float(np.mean(
                        [np.mean(s[(sk, ms)])
                         for s in SPs
                         if (sk, ms) in s]))
                    rows.append({
                        "outcome": oc,
                        "sparse_k": sk,
                        "measure": ms,
                        "temp": tp_, "amp": am,
                        "mean": aa.mean(),
                        "sd": aa.std(ddof=1),
                        "vs_pool": aa.mean()
                        - base, "spread": sp})
                    if best is None or \
                            aa.mean() > best[0]:
                        best = (aa.mean(), sk,
                                ms, tp_, am)
                line += " %+7.4f" % (
                    max(vals) - base)
            print(line + "   (best over"
                  " amplification)", flush=True)
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)
    if best:
        print("")
        print("  BEST  %.4f  %+.4f vs pool"
              "   k=%s measure=%s temp=%s"
              " amp=%.2f"
              % (best[0], best[0] - base,
                 "all" if best[1] == 0
                 else best[1], best[2],
                 "argmax" if best[3] < 0
                 else best[3], best[4]),
              flush=True)

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
q = r[r["sparse_k"].notna()]

print("")
print("=" * 74)
print("FIX 1: DOES SPARSITY RECOVER THE"
      " VARIATION?")
print("  the tertile diagnostic recovered a"
      " 0.19 spread from ONE feature; the")
print("  338-feature model recovered 0.02")
if len(q):
    print("")
    print("  profile spread by sparse_k")
    print(q.groupby("sparse_k")["spread"]
          .mean().round(4).to_string())
    print("")
    print("  gain over the pool by sparse_k")
    print(q.groupby("sparse_k")["vs_pool"]
          .max().round(4).to_string())

print("")
print("FIX 2: DOES TEMPERATURE HELP?")
print("  script 123 hardcoded 2.0. -1 is a hard"
      " argmax, the fully sparse limit.")
if len(q):
    print(q.groupby("temp")["vs_pool"].max()
          .round(4).to_string())

print("")
print("FIX 3: WHICH COMPETENCE MEASURE?")
print("  the dynamic-selection review surveys"
      " eight and notes that choosing one")
print("  'is not straightforward'")
if len(q):
    print(q.groupby("measure")["vs_pool"].max()
          .round(4).to_string())
    print("")
    print("  spread by measure")
    print(q.groupby("measure")["spread"].mean()
          .round(4).to_string())

print("")
print("FIX 4: DOES AMPLIFICATION HELP?")
print("  if it does, the competence ordering"
      " was right and only the magnitude")
print("  was too small. If it does not, the"
      " estimate itself is wrong.")
if len(q):
    print(q.groupby("amp")["vs_pool"].max()
          .round(4).to_string())

print("")
print("=" * 74)
print("DOES SPREAD PREDICT GAIN?")
print("  the whole diagnosis rests on this: a"
      " flat profile cannot help, so a")
print("  positive relationship confirms the"
      " mechanism and a flat one refutes it")
if len(q) > 5:
    from scipy import stats as st_
    x = q["spread"].values
    yv = q["vs_pool"].values
    ok = np.isfinite(x) & np.isfinite(yv)
    if ok.sum() > 5 and np.std(x[ok]) > 1e-9:
        rr, pp = st_.pearsonr(x[ok], yv[ok])
        sr, sp2 = st_.spearmanr(x[ok], yv[ok])
        print("  Pearson %+.3f (p=%.4f)"
              "   Spearman %+.3f (p=%.4f)"
              "   on %d cells"
              % (rr, pp, sr, sp2,
                 int(ok.sum())))

print("")
print("BEST CELL PER OUTCOME")
for oc in OUTS:
    s = q[q["outcome"] == oc]
    if not len(s):
        continue
    x = s.loc[s["mean"].idxmax()]
    print("  %-18s %.4f   %+.4f vs pool"
          "   k=%s %s temp=%s amp=%.2f"
          % (oc, x["mean"], x["vs_pool"],
             "all" if x["sparse_k"] == 0
             else int(x["sparse_k"]),
             x["measure"],
             "argmax" if x["temp"] < 0
             else x["temp"], x["amp"]))

print("")
print("WINS OVER THE UNWEIGHTED POOL")
w_ = q[q["vs_pool"] > 0]
print("  %d of %d cells" % (len(w_), len(q)))
if len(w_):
    print(w_.sort_values(
        "vs_pool", ascending=False)[
        ["outcome", "sparse_k", "measure",
         "temp", "amp", "mean",
         "vs_pool"]].head(12).round(4)
        .to_string(index=False))
else:
    print("  none. The competence mechanism is"
          " real and measurable, but it")
    print("  cannot be estimated per patient at"
          " this event count, which is")
    print("  itself the finding.")
print("")
print("saved", DEST, r.shape)
keep_awake(False)