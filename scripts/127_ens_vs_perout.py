"""ens for every modality against per-outcome
winners, with combination rules and RF+CCA.

In script 126, ens (L2 and a forest,
rank-averaged) had the best marginal on all three
modality blocks, and better base learners gave
better late fusion on every outcome.

BASE SETS
  l2        L2 logistic regression throughout
  ens_all   ens for every modality
  per_out   the winning learner for each modality
            and outcome. It is selected on the
            same data, so the gap against ens_all
            shows how much of its edge is
            selection.

COMBINATION RULES
  five rules: rank, logit and probability
  averaging, the last two with and without
  calibration. A weighted rank average is not on
  a probability scale, so only its calibrated
  Brier score is meaningful.

RF+CCA on every combination, in two variants.
Per-modality predictions enter the block forest
as extra features, taken from inner
cross-validation inside each training fold so no
row sees its own prediction.

Parallel across seeds. Timing probe first.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ens_vs_perout.csv
  results\\ens_vs_perout_log.txt
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
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

HAVE_CAT = False
try:
    from catboost import CatBoostClassifier
    HAVE_CAT = True
except Exception:
    print("catboost unavailable")

PROC, FIG = f.PROC, f.FIG
RAW = os.path.join("data", "raw", "ecg")
SEEDS = [42, 7, 13, 1, 2]
NFOLD = 5
NINNER = 4
NJOBS = max(1, min(10, (os.cpu_count() or 4)
                   - 2))
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CT_CS = list(np.logspace(-4, 4, 10))
LEAF, DEPTH, NTREE = 1, 12, 600
# script 126: k=10 interior, 14+8+16 best
KS = [5, 8, 10, 13, 16]
REGS = [0.1, 0.5]
DRAW3 = (14, 8, 16)
DRAW2 = {"ehr": 14, "ecg": 8, "ctpa": 16}
RULES = ["rank", "prob", "prob_cal",
         "logit", "logit_cal"]
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(
    PROC, "ens_vs_perout.csv")
# script 126, per-outcome per-modality winners
PER_OUT = {
    "death_30d_inhosp": {"ehr": "rf",
                         "ecg": "ens",
                         "ctpa": "ens"},
    "death_30d": {"ehr": "catboost",
                  "ecg": "ens",
                  "ctpa": "ens"},
    "composite_30d": {"ehr": "gb",
                      "ecg": "ens",
                      "ctpa": "ens"},
    "cv_first": {"ehr": "ens", "ecg": "ens",
                 "ctpa": "l2"}}
S126 = {"death_30d_inhosp":
        {"late_l2": 0.8692, "late_b": 0.9101,
         "nnls": 0.9099, "blk": 0.8969},
        "death_30d":
        {"late_l2": 0.8856, "late_b": 0.9029,
         "nnls": 0.9041, "blk": 0.8946},
        "composite_30d":
        {"late_l2": 0.8492, "late_b": 0.8656,
         "nnls": 0.8684, "blk": 0.8562},
        "cv_first":
        {"late_l2": 0.8109, "late_b": 0.8239,
         "nnls": 0.8239, "blk": 0.7967}}

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


# ---------------- base learners -------------
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
    return m.predict_proba(Xe)[:, 1]


def L_rf(Xt, yt, Xe, grid=None):
    m = RandomForestClassifier(
        n_estimators=500, max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42, n_jobs=1)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def L_et(Xt, yt, Xe, grid=None):
    m = ExtraTreesClassifier(
        n_estimators=500, max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt", bootstrap=False,
        class_weight="balanced_subsample",
        random_state=42, n_jobs=1)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def L_gb(Xt, yt, Xe, grid=None):
    m = HistGradientBoostingClassifier(
        max_depth=3, max_iter=200,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def L_cat(Xt, yt, Xe, grid=None):
    if not HAVE_CAT:
        return L_gb(Xt, yt, Xe, grid)
    m = CatBoostClassifier(
        iterations=400, depth=4,
        learning_rate=0.05, l2_leaf_reg=6.0,
        random_seed=42, verbose=0,
        allow_writing_files=False)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def L_ens(Xt, yt, Xe, grid=None):
    """L2 and a forest, rank-averaged. The best
    marginal on all three modalities in script
    126, winning all four outcomes on ECG and
    losing only cv_first on CTPA by 0.0016."""
    a = _rank(L_l2(Xt, yt, Xe, grid))
    b = _rank(L_rf(Xt, yt, Xe, grid))
    return 0.5 * (a + b)


LFN = {"l2": L_l2, "rf": L_rf, "et": L_et,
       "gb": L_gb, "catboost": L_cat,
       "ens": L_ens}


class BlockForest:
    def __init__(self, n_estimators=600,
                 draws=(14, 8, 16),
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


# ---------------- combination ---------------
def platt(p_tr, y_tr, p_te):
    x = _logit(p_tr).reshape(-1, 1)
    z = _logit(p_te).reshape(-1, 1)
    try:
        m = LogisticRegression(C=1e6,
                               max_iter=2000)
        m.fit(x, y_tr)
        return m.predict_proba(z)[:, 1]
    except Exception:
        return p_te


def transform(ps_tr, ps_te, y_tr, rule):
    """rank discards the score distribution
    entirely; prob keeps it raw; the _cal forms
    Platt-scale each model first, which is what
    the genomics study credits for stacking's
    edge; logit averages log-odds, which the
    deep-ensemble paper reports highest at every
    ensemble size."""
    if rule == "rank":
        return (np.column_stack(
            [_rank(p) for p in ps_tr]),
            np.column_stack(
                [_rank(p) for p in ps_te]))
    if rule == "prob":
        return (np.column_stack(ps_tr),
                np.column_stack(ps_te))
    if rule == "logit":
        return (np.column_stack(
            [_logit(p) for p in ps_tr]),
            np.column_stack(
                [_logit(p) for p in ps_te]))
    ct, ce = [], []
    for a, b in zip(ps_tr, ps_te):
        ct.append(platt(a, y_tr, a))
        ce.append(platt(a, y_tr, b))
    if rule == "prob_cal":
        return (np.column_stack(ct),
                np.column_stack(ce))
    return (np.column_stack(
        [_logit(p) for p in ct]),
        np.column_stack(
            [_logit(p) for p in ce]))


def grid_w(M_tr, y_tr, M_te, step=0.05):
    m = M_tr.shape[1]
    n = int(round(1.0 / step))

    def rec(k, rem):
        if k == 1:
            yield (rem,)
            return
        for i in range(rem + 1):
            for t in rec(k - 1, rem - i):
                yield (i,) + t
    best, bw = -1.0, np.ones(m) / m
    for w in rec(m, n):
        w = np.array([x * step for x in w])
        try:
            a = roc_auc_score(y_tr, M_tr @ w)
        except Exception:
            continue
        if a > best:
            best, bw = a, w
    return M_te @ bw


def nnls_oof(R, y, grp, seed=42):
    p = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(R, y, grp):
        try:
            w, _ = nnls(R[tr],
                        y[tr].astype(float))
        except Exception:
            w = np.ones(R.shape[1])
        if w.sum() <= 1e-9:
            w = np.ones(R.shape[1])
        p[te] = R[te] @ (w / w.sum())
    return p


def inner_oof(X, y, grp, fn, seed, grid=None):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NINNER, shuffle=True,
        random_state=seed + 1)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        Xt, Xe = prep_fold(X[tr], X[te])
        p[te] = fn(Xt, y[tr], Xe, grid)
    return p


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


def calib(p, y):
    ok = np.isfinite(p)
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
COMBOS = [("ehr", "ecg"), ("ehr", "ctpa"),
          ("ecg", "ctpa"),
          ("ehr", "ecg", "ctpa")]
BASESETS = ["l2", "ens_all", "per_out"]
print("")
print("cohort:", len(D))
print("  EHR %d   ECG %d   CTPA %d"
      % (len(EH), len(ECGC), len(CT)))
print("")
print("  BASE SETS")
print("    l2       the inherited baseline")
print("    ens_all  ens everywhere. Best"
      " marginal on all three modalities in")
print("             script 126, and no"
      " selection was involved.")
print("    per_out  the winner per modality AND"
      " outcome, SELECTED on the same")
print("             data, so optimistic by"
      " construction:")
for oc in OUTS:
    print("             %-18s %s" % (
        oc, ", ".join(
            "%s=%s" % (m_, PER_OUT[oc][m_])
            for m_ in MODS)))
print("")
print("  k grid:", KS,
      "  (script 126 found 10 interior)")
print("  draws:", DRAW3,
      "  (beat 20+8+16 on three of four)")
print("  rules:", ", ".join(RULES),
      flush=True)

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
L_ens(np.random.rand(len(D), len(ECGC)), yp,
      np.random.rand(100, len(ECGC)))
t_ens = time.time() - t
stA = (len(KS) * len(REGS) * t_blk * NFOLD
       * len(SEEDS) * len(OUTS) / 60.0)
stB = ((3 * 2 * (NINNER + 1) * t_ens
        + len(COMBOS) * 2 * t_blk)
       * NFOLD * len(SEEDS) * len(OUTS)
       / 60.0)
print("  block forest %d trees: %5.1f s"
      % (NTREE, t_blk))
print("  ens on the ECG block:  %5.1f s"
      % t_ens)
print("")
print("  stage A about %.0f min, stages B-C"
      " about %.0f min, serial"
      % (stA, stB))
print("  with %d-way parallelism about %.0f"
      " min total"
      % (NJOBS, (stA + stB) / NJOBS))
print("  Ctrl+C now if too long. Results save"
      " after every stage.")
time.sleep(5)

rows = []
t0 = time.time()

for oc in OUTS:
    if oc not in D.columns:
        continue
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    if y.sum() < 25:
        continue
    Xd = {k: d[v].values.astype(float)
          for k, v in BLK.items()}
    pv = S126.get(oc, {})
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())))
    print("  script 126: late_l2 %.4f,"
          " late_best %.4f, nnls %.4f"
          % (pv.get("late_l2", np.nan),
             pv.get("late_b", np.nan),
             pv.get("nnls", np.nan)),
          flush=True)

    # ---- STAGE A: k and reg, refined ----
    print("")
    print("  STAGE A: CCA k x reg at draws"
          " %d+%d+%d" % DRAW3)
    print("  %-6s" % "k", end="")
    for rg in REGS:
        print(" %10s" % ("reg=%.1f" % rg),
              end="")
    print("")
    bestk, bestr, bkm = 10, 0.1, -1.0
    for k_ in KS:
        line = "  %-6d" % k_
        for rg in REGS:
            def one(s, k_=k_, rg=rg):
                p = np.zeros(len(y))
                cv = StratifiedGroupKFold(
                    n_splits=NFOLD,
                    shuffle=True,
                    random_state=s)
                for tr, te in cv.split(
                        np.zeros((len(y), 1)),
                        y, grp):
                    Bt, Be = [], []
                    for m_ in MODS:
                        a, b = prep_fold(
                            Xd[m_][tr],
                            Xd[m_][te])
                        Bt.append(a)
                        Be.append(b)
                    Zt, Ze, bd = ccaaug(
                        Bt, Be, k_, rg)
                    bf = BlockForest(
                        n_estimators=NTREE,
                        draws=DRAW3,
                        bounds=bd, leaf=LEAF,
                        depth=DEPTH, seed=s)
                    bf.fit(Zt, y[tr])
                    p[te] = bf.predict_proba(
                        Ze)[:, 1]
                return p
            try:
                ps = par(one)
            except Exception:
                line += " %10s" % "fail"
                continue
            aa = np.array([roc_auc_score(y, p)
                           for p in ps])
            line += " %10.4f" % aa.mean()
            rows.append({
                "outcome": oc, "stage": "A",
                "k": k_, "reg": rg,
                "mean": aa.mean(),
                "sd": aa.std(ddof=1)})
            if aa.mean() > bkm:
                bkm, bestk, bestr = \
                    aa.mean(), k_, rg
        print(line, flush=True)
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)
    print("  best k=%d reg=%.1f  %.4f"
          % (bestk, bestr, bkm), flush=True)

    # ---- STAGES B-C ----
    print("")
    print("  STAGES B-C: base sets,"
          " combination rules, RF+CCA")

    def big(s):
        n = len(y)
        out = {}
        M = {}
        for bs in BASESETS:
            for m_ in MODS:
                M[(bs, m_)] = np.zeros(n)
            for rl in RULES:
                out[("late", bs, rl)] = \
                    np.zeros(n)
        for cb in COMBOS:
            nm = "+".join(cb)
            for v in ("plain", "stack"):
                out[("rf", nm, v)] = \
                    np.zeros(n)
        cv = StratifiedGroupKFold(
            n_splits=NFOLD, shuffle=True,
            random_state=s)
        for tr, te in cv.split(
                np.zeros((n, 1)), y, grp):
            Bt, Be = [], []
            for m_ in MODS:
                a, b = prep_fold(Xd[m_][tr],
                                 Xd[m_][te])
                Bt.append(a)
                Be.append(b)
            ptr = {bs: [] for bs in BASESETS}
            pte = {bs: [] for bs in BASESETS}
            for mi, m_ in enumerate(MODS):
                gd = (CT_CS if m_ == "ctpa"
                      else None)
                for bs in BASESETS:
                    if bs == "l2":
                        ln = "l2"
                    elif bs == "ens_all":
                        ln = "ens"
                    else:
                        ln = PER_OUT[oc][m_]
                    fn = LFN[ln]
                    gg = gd if ln == "l2" \
                        else None
                    ptr[bs].append(inner_oof(
                        Xd[m_][tr], y[tr],
                        grp[tr], fn, s, gg))
                    q = fn(Bt[mi], y[tr],
                           Be[mi], gg)
                    pte[bs].append(q)
                    M[(bs, m_)][te] = q
            for bs in BASESETS:
                for rl in RULES:
                    A_, B_ = transform(
                        ptr[bs], pte[bs],
                        y[tr], rl)
                    out[("late", bs, rl)][te] \
                        = grid_w(A_, y[tr], B_)
            for cb in COMBOS:
                nm = "+".join(cb)
                idx = [MODS.index(c)
                       for c in cb]
                bt = [Bt[i] for i in idx]
                be = [Be[i] for i in idx]
                Zt, Ze, bd = ccaaug(
                    bt, be, bestk, bestr)
                dr = tuple(DRAW2[c]
                           for c in cb)
                bf = BlockForest(
                    n_estimators=NTREE,
                    draws=dr, bounds=bd,
                    leaf=LEAF, depth=DEPTH,
                    seed=s)
                bf.fit(Zt, y[tr])
                out[("rf", nm, "plain")][te] \
                    = bf.predict_proba(
                        Ze)[:, 1]
                Zt2 = np.column_stack(
                    [Zt] + [ptr["ens_all"][i]
                            .reshape(-1, 1)
                            for i in idx])
                Ze2 = np.column_stack(
                    [Ze] + [pte["ens_all"][i]
                            .reshape(-1, 1)
                            for i in idx])
                dr2 = tuple(
                    list(dr[:-1])
                    + [dr[-1] + len(idx)])
                bf2 = BlockForest(
                    n_estimators=NTREE,
                    draws=dr2, bounds=bd,
                    leaf=LEAF, depth=DEPTH,
                    seed=s)
                bf2.fit(Zt2, y[tr])
                out[("rf", nm, "stack")][te] \
                    = bf2.predict_proba(
                        Ze2)[:, 1]
        return out, M

    try:
        res = par(big)
    except Exception as exc:
        print("  FAILED %s" % repr(exc)[:70])
        continue
    Os = [r_[0] for r_ in res]
    M0 = res[0][1]

    print("")
    print("  MODALITY AUROC BY BASE SET")
    print("  %-9s %9s %9s %9s %9s"
          % ("baseset", "ehr", "ecg", "ctpa",
             "meancorr"))
    for bs in BASESETS:
        vals = [roc_auc_score(y, M0[(bs, m_)])
                for m_ in MODS]
        R = np.column_stack(
            [_rank(M0[(bs, m_)])
             for m_ in MODS])
        cc = [np.corrcoef(R[:, i],
                          R[:, jj])[0, 1]
              for i in range(3)
              for jj in range(i + 1, 3)]
        print("  %-9s %9.4f %9.4f %9.4f %9.3f"
              % (bs, vals[0], vals[1],
                 vals[2], float(np.mean(cc))),
              flush=True)
        for mi, m_ in enumerate(MODS):
            rows.append({
                "outcome": oc, "stage": "mod",
                "baseset": bs, "modality": m_,
                "mean": vals[mi]})
        rows.append({
            "outcome": oc, "stage": "corr",
            "baseset": bs,
            "mean": float(np.mean(cc))})

    print("")
    print("  LATE FUSION: BASE SET x RULE")
    print("  %-10s" % "rule", end="")
    for bs in BASESETS:
        print(" %10s" % bs, end="")
    print("")
    bestlate = None
    for rl in RULES:
        line = "  %-10s" % rl
        for bs in BASESETS:
            aa = np.array([roc_auc_score(
                y, o[("late", bs, rl)])
                for o in Os])
            line += " %10.4f" % aa.mean()
            rows.append({
                "outcome": oc, "stage": "B",
                "rule": rl, "baseset": bs,
                "mean": aa.mean(),
                "sd": aa.std(ddof=1)})
            if bestlate is None or \
                    aa.mean() > bestlate[0]:
                bestlate = (aa.mean(), bs, rl)
        print(line, flush=True)
    print("  best %.4f  %s / %s"
          % bestlate, flush=True)
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)

    print("")
    print("  RF+CCA BY COMBINATION  (k=%d"
          " reg=%.1f)" % (bestk, bestr))
    print("  %-16s %10s %10s %9s"
          % ("combo", "plain", "stacked",
             "diff"))
    for cb in COMBOS:
        nm = "+".join(cb)
        a1 = np.array([roc_auc_score(
            y, o[("rf", nm, "plain")])
            for o in Os]).mean()
        a2 = np.array([roc_auc_score(
            y, o[("rf", nm, "stack")])
            for o in Os]).mean()
        print("  %-16s %10.4f %10.4f %+9.4f"
              % (nm, a1, a2, a2 - a1),
              flush=True)
        for v, val in (("plain", a1),
                       ("stack", a2)):
            rows.append({
                "outcome": oc, "stage": "C",
                "combo": nm, "variant": v,
                "mean": val})

    # ---- final ensemble and calibration ----
    print("")
    print("  FINAL: nnls OF THE BEST LATE AND"
          " THE BEST BLOCK FOREST")
    pl = Os[0][("late", bestlate[1],
                bestlate[2])]
    pb = Os[0][("rf", "ehr+ecg+ctpa",
                "plain")]
    R = np.column_stack(
        [_rank(pl), _rank(pb)])
    pe_ = nnls_oof(R, y, grp)
    for nm, p in (("late_best", pl),
                  ("block", pb),
                  ("nnls", pe_)):
        ok = np.isfinite(p)
        av = roc_auc_score(y[ok], p[ok])
        b0, s0 = calib(p, y)
        pc = platt_oof(p, y, grp)
        b1, s1 = calib(pc, y)
        print("    %-10s %.4f   slope %.3f ->"
              " %.3f   Brier %.5f -> %.5f"
              % (nm, av, s0, s1, b0, b1))
        rows.append({
            "outcome": oc, "stage": "D",
            "name": nm, "mean": av,
            "slope": s0, "slope_platt": s1,
            "brier": b0, "brier_platt": b1,
            "vs_126": av - pv.get("nnls",
                                  np.nan)})
    print("    a weighted rank average is not"
          " on a probability scale, so its")
    print("    RAW Brier is meaningless; only"
          " the calibrated one is reportable",
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

print("")
print("=" * 74)
print("STAGE A: CCA k x reg")
a_ = r[r["stage"] == "A"]
if len(a_):
    print(a_.pivot_table(index="k",
                         columns="outcome",
                         values="mean")
          .round(4).to_string())
    print("")
    print("  marginal by k")
    print(a_.groupby("k")["mean"].mean()
          .round(4).to_string())
    print("  marginal by reg")
    print(a_.groupby("reg")["mean"].mean()
          .round(4).to_string())

print("")
print("=" * 74)
print("MODALITY AUROC BY BASE SET")
m_ = r[r["stage"] == "mod"]
if len(m_):
    for mm_ in MODS:
        z = m_[m_["modality"] == mm_]
        if not len(z):
            continue
        print("")
        print("  " + mm_)
        print(z.pivot_table(index="baseset",
                            columns="outcome",
                            values="mean")
              .round(4).to_string())

print("")
print("ens_all AGAINST per_out AT THE MODALITY"
      " LEVEL")
print("  per_out is selected on this data, so"
      " any edge it shows is partly")
print("  selection rather than a real"
      " difference")
if len(m_):
    for mm_ in MODS:
        a1 = m_[(m_["modality"] == mm_)
                & (m_["baseset"] == "ens_all")]
        a2 = m_[(m_["modality"] == mm_)
                & (m_["baseset"] == "per_out")]
        if len(a1) and len(a2):
            print("  %-6s ens_all %.4f   per_out"
                  " %.4f   %+.4f"
                  % (mm_, a1["mean"].mean(),
                     a2["mean"].mean(),
                     a2["mean"].mean()
                     - a1["mean"].mean()))

print("")
print("MEMBER CORRELATION BY BASE SET")
c_ = r[r["stage"] == "corr"]
if len(c_):
    print(c_.pivot_table(index="baseset",
                         columns="outcome",
                         values="mean")
          .round(3).to_string())
    print("")
    print("  a base set whose members correlate"
          " less leaves more for fusion")

print("")
print("=" * 74)
print("LATE FUSION: BASE SET x RULE")
b_ = r[r["stage"] == "B"]
if len(b_):
    print(b_.pivot_table(index="rule",
                         columns="baseset",
                         values="mean")
          .round(4).to_string())
    print("")
    print("  marginal by rule")
    print(b_.groupby("rule")["mean"].mean()
          .round(4).to_string())
    print("  marginal by base set")
    print(b_.groupby("baseset")["mean"].mean()
          .round(4).to_string())
    print("")
    print("  published comparisons report"
          " logit averaging highest at every")
    print("  ensemble size, uncalibrated"
          " probability worst, calibrated")
    print("  probability between")

print("")
print("DOES ens_all MATCH per_out AFTER"
      " FUSION?")
print("  if it does, use ens_all: one learner,"
      " no selection, and honest")
if len(b_):
    for oc in OUTS:
        z = b_[b_["outcome"] == oc]
        a1 = z[z["baseset"] == "ens_all"]
        a2 = z[z["baseset"] == "per_out"]
        a3 = z[z["baseset"] == "l2"]
        if len(a1) and len(a2):
            print("  %-18s l2 %.4f   ens_all"
                  " %.4f   per_out %.4f"
                  % (oc,
                     a3["mean"].max()
                     if len(a3) else np.nan,
                     a1["mean"].max(),
                     a2["mean"].max()))

print("")
print("RF+CCA BY COMBINATION")
cc_ = r[r["stage"] == "C"]
if len(cc_):
    for v in ("plain", "stack"):
        z = cc_[cc_["variant"] == v]
        if not len(z):
            continue
        print("")
        print("  " + v)
        print(z.pivot_table(index="combo",
                            columns="outcome",
                            values="mean")
              .round(4).to_string())

print("")
print("FINAL MODELS")
d_ = r[r["stage"] == "D"]
if len(d_):
    print(d_[["outcome", "name", "mean",
              "vs_126", "slope",
              "slope_platt",
              "brier_platt"]].round(4)
          .to_string(index=False))

print("")
print("BEST OVERALL PER OUTCOME")
for oc in OUTS:
    s = r[(r["outcome"] == oc)
          & r["stage"].isin(["A", "B", "C",
                             "D"])]
    if not len(s):
        continue
    x = s.loc[s["mean"].idxmax()]
    pv = S126.get(oc, {})
    print("  %-18s %.4f   script 126 nnls"
          " %.4f   %+.4f"
          % (oc, x["mean"],
             pv.get("nnls", np.nan),
             x["mean"] - pv.get("nnls",
                                np.nan)))
print("")
print("saved", DEST, r.shape)
keep_awake(False)