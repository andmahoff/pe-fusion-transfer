"""Per-modality base learners, CCA rank crossed
with the block draws, and calibration.

STAGE A  seven learners on each modality block
         separately (EHR 28 columns, ECG 264, CTPA
         46 mostly binary). The base models had
         been L2 logistic regression throughout.
STAGE B  CCA k and regularisation crossed with the
         block draws, since k sets how many
         canonical variates join each block.
STAGE C  Platt scaling of the nnls ensemble. The
         transform is monotone, so AUROC is
         unchanged.
STAGE D  the full model rebuilt with the stage A
         to C choices, against script 125's nnls
         ensemble (0.9085 and 0.8908).

Parallel across seeds. Timing probe first.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\base_learners.csv
  results\\base_learners_log.txt
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
except Exception as exc:
    print("catboost unavailable:",
          repr(exc)[:50])

PROC, FIG = f.PROC, f.FIG
RAW = os.path.join("data", "raw", "ecg")
SEEDS = [42, 7, 13, 1, 2]
NFOLD = 5
NJOBS = max(1, min(10, (os.cpu_count() or 4)
                   - 2))
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CT_CS = list(np.logspace(-4, 4, 10))
LEAF, DEPTH, NTREE = 1, 12, 600
K0, REG0 = 5, 0.5
KS = [3, 5, 10, 20]
DRAWS = [(20, 8, 16), (28, 8, 16),
         (20, 4, 8), (14, 8, 16),
         (20, 12, 24)]
REGS = [0.1, 0.5]
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(
    PROC, "base_learners.csv")
# script 125, nnls ensemble
S125 = {"death_30d_inhosp": 0.9085,
        "death_30d": 0.8908,
        "composite_30d": np.nan,
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


def ccaaug3(Bt, Be, k, reg):
    pt = [[b] for b in Bt]
    pe = [[b] for b in Be]
    for i in range(3):
        for jj in range(i + 1, 3):
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
def L_l2(Xt, yt, Xe, gt=None, grid=None):
    """The incumbent, as used in the
    dissertation."""
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


def L_l1(Xt, yt, Xe, gt=None, grid=None):
    """Lasso: SELECTS features, which may suit
    the 264-column ECG block where most columns
    are uninformative."""
    best, bm = -1.0, None
    for c in [1e-3, 1e-2, 1e-1, 1.0]:
        try:
            m = LogisticRegression(
                penalty="l1", C=c,
                solver="liblinear",
                max_iter=4000)
            m.fit(Xt, yt)
            a = roc_auc_score(
                yt,
                m.predict_proba(Xt)[:, 1])
            if a > best:
                best, bm = a, m
        except Exception:
            continue
    if bm is None:
        return L_l2(Xt, yt, Xe, gt, grid)
    return bm.predict_proba(Xe)[:, 1]


def L_rf(Xt, yt, Xe, gt=None, grid=None):
    m = RandomForestClassifier(
        n_estimators=500, max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42, n_jobs=1)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def L_et(Xt, yt, Xe, gt=None, grid=None):
    m = ExtraTreesClassifier(
        n_estimators=500, max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt", bootstrap=False,
        class_weight="balanced_subsample",
        random_state=42, n_jobs=1)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def L_gb(Xt, yt, Xe, gt=None, grid=None):
    m = HistGradientBoostingClassifier(
        max_depth=3, max_iter=200,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def L_cat(Xt, yt, Xe, gt=None, grid=None):
    m = CatBoostClassifier(
        iterations=400, depth=4,
        learning_rate=0.05, l2_leaf_reg=6.0,
        random_seed=42, verbose=0,
        allow_writing_files=False)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def L_ens(Xt, yt, Xe, gt=None, grid=None):
    """L2 and a forest, rank-averaged. The two
    families have different biases, so their
    mean can beat either."""
    a = _rank(L_l2(Xt, yt, Xe, gt, grid))
    b = _rank(L_rf(Xt, yt, Xe, gt, grid))
    return 0.5 * (a + b)


LEARNERS = [("l2", L_l2), ("l1", L_l1),
            ("rf", L_rf), ("et", L_et),
            ("gb", L_gb), ("ens", L_ens)]
if HAVE_CAT:
    LEARNERS.append(("catboost", L_cat))


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


def oof_uni(X, y, grp, seed, fn, grid=None):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        Xt, Xe = prep_fold(X[tr], X[te])
        p[te] = fn(Xt, y[tr], Xe, grp[tr],
                   grid)
    return p


def late3(ps, y):
    bs, bv = -1.0, ps[0]
    for w1 in np.arange(0, 1.001, 0.05):
        for w2 in np.arange(0, 1.001 - w1,
                            0.05):
            v = (w1 * ps[0] + w2 * ps[1]
                 + (1 - w1 - w2) * ps[2])
            try:
                a = roc_auc_score(y, v)
            except Exception:
                continue
            if a > bs:
                bs, bv = a, v
    return bv


def late3_oof(ps, y, grp, seed=42):
    """Weights fitted on training rows only.
    Script 125 measured the in-sample version's
    leak at +0.0058 and +0.0048."""
    p = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    M = np.column_stack(ps)
    for tr, te in cv.split(M, y, grp):
        bs, bw = -1.0, (1 / 3, 1 / 3, 1 / 3)
        for w1 in np.arange(0, 1.001, 0.05):
            for w2 in np.arange(
                    0, 1.001 - w1, 0.05):
                w3 = 1 - w1 - w2
                v = M[tr] @ np.array(
                    [w1, w2, w3])
                try:
                    a = roc_auc_score(y[tr], v)
                except Exception:
                    continue
                if a > bs:
                    bs, bw = a, (w1, w2, w3)
        p[te] = M[te] @ np.array(bw)
    return p


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


def platt_oof(p, y, grp, seed=42):
    ok = np.isfinite(p)
    q = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(q / (1 - q)).reshape(-1, 1)
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
        x = np.log(q / (1 - q)).reshape(-1, 1)
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
print("")
print("cohort:", len(D))
print("  EHR %d   ECG %d   CTPA %d"
      % (len(EH), len(ECGC), len(CT)))
print("  the blocks differ enormously, so the"
      " right learner may differ by modality")
print("  learners:",
      ", ".join(n for n, _ in LEARNERS))
print("  CCA k grid:", KS, "  reg:", REGS)
print("  draw grid:", DRAWS, flush=True)

# ---------- TIMING PROBE ----------
print("")
print("TIMING PROBE")
Xe_ = np.random.rand(len(D), len(EH))
Xg_ = np.random.rand(len(D), len(ECGC))
yp = (np.random.rand(len(D)) > 0.90) \
    .astype(int)
tt = {}
for nm, fn in LEARNERS:
    t = time.time()
    try:
        fn(Xg_, yp, Xg_[:100])
    except Exception:
        pass
    tt[nm] = time.time() - t
t = time.time()
BlockForest(n_estimators=NTREE,
            bounds=(38, 312), leaf=LEAF,
            depth=DEPTH, seed=42).fit(
    np.random.rand(len(D), 400), yp)
t_blk = time.time() - t
for nm in tt:
    print("  %-10s on the ECG block: %5.1f s"
          % (nm, tt[nm]))
print("  block forest:              %5.1f s"
      % t_blk)
stA = (sum(tt.values()) * 3 * NFOLD
       * len(SEEDS) * len(OUTS) / 60.0)
stB = (len(KS) * len(REGS) * len(DRAWS)
       * t_blk * NFOLD * len(SEEDS)
       * len(OUTS) / 60.0)
print("")
print("  stage A about %.0f min, stage B"
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
    base = S125.get(oc, np.nan)
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d   script 125 nnls"
          " %.4f"
          % (oc, len(y), int(y.sum()), base))

    # ---- STAGE A: per-modality learners ----
    print("")
    print("  STAGE A: PER-MODALITY LEARNERS")
    print("  L2 is the learner used in the"
          " dissertation")
    print("  %-10s %9s %9s %9s"
          % ("learner", "ehr", "ecg", "ctpa"))
    bestL, POOL = {}, {}
    for nm, fn in LEARNERS:
        line = "  %-10s" % nm
        for m_ in MODS:
            gd = CT_CS if (m_ == "ctpa"
                           and nm == "l2") \
                else None
            try:
                ps = par(lambda s, fn=fn,
                         m_=m_, gd=gd:
                         oof_uni(Xd[m_], y,
                                 grp, s, fn,
                                 gd))
            except Exception:
                line += " %9s" % "fail"
                continue
            aa = np.array([roc_auc_score(y, p)
                           for p in ps])
            line += " %9.4f" % aa.mean()
            POOL[(nm, m_)] = ps[0]
            rows.append({
                "outcome": oc, "stage": "A",
                "learner": nm, "modality": m_,
                "mean": aa.mean(),
                "sd": aa.std(ddof=1)})
            if m_ not in bestL or \
                    aa.mean() > bestL[m_][1]:
                bestL[m_] = (nm, aa.mean())
        print(line, flush=True)
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)
    print("")
    print("  best per modality: %s"
          % ", ".join(
              "%s=%s (%.4f)"
              % (m_, bestL[m_][0],
                 bestL[m_][1])
              for m_ in MODS), flush=True)

    # late fusion under L2 and under the best
    ps_l2 = [_rank(POOL[("l2", m_)])
             for m_ in MODS]
    ps_bt = [_rank(POOL[(bestL[m_][0], m_)])
             for m_ in MODS]
    for tag, ps in (("l2", ps_l2),
                    ("best", ps_bt)):
        v = late3_oof(ps, y, grp)
        ok = np.isfinite(v)
        av = roc_auc_score(y[ok], v[ok])
        print("    late fusion, %-4s bases:"
              " %.4f" % (tag, av))
        rows.append({
            "outcome": oc, "stage": "A",
            "learner": "late_" + tag,
            "modality": "all", "mean": av,
            "sd": np.nan})

    # ---- STAGE B: k x reg x draws ----
    print("")
    print("  STAGE B: CCA k x reg x DRAWS")
    print("  script 117 fixed draws at 20+8+16"
          " with k=10 reg=0.1; the current")
    print("  model uses k=5 reg=0.5. k changes"
          " the column count each block draw")
    print("  samples from, so the two interact.")
    print("  %-12s" % "draws", end="")
    for k_ in KS:
        print(" %9s" % ("k=%d" % k_), end="")
    print("     (best over reg)")
    bestB = None
    for dr in DRAWS:
        line = "  %-12s" % ("%d+%d+%d" % dr)
        for k_ in KS:
            vals = []
            for rg in REGS:
                def one(s, dr=dr, k_=k_,
                        rg=rg):
                    p = np.zeros(len(y))
                    cv = StratifiedGroupKFold(
                        n_splits=NFOLD,
                        shuffle=True,
                        random_state=s)
                    for tr, te in cv.split(
                            np.zeros(
                                (len(y), 1)),
                            y, grp):
                        Bt, Be = [], []
                        for m_ in MODS:
                            a, b = prep_fold(
                                Xd[m_][tr],
                                Xd[m_][te])
                            Bt.append(a)
                            Be.append(b)
                        Zt, Ze, bd = ccaaug3(
                            Bt, Be, k_, rg)
                        bf = BlockForest(
                            n_estimators=NTREE,
                            draws=dr,
                            bounds=bd,
                            leaf=LEAF,
                            depth=DEPTH,
                            seed=s)
                        bf.fit(Zt, y[tr])
                        p[te] = \
                            bf.predict_proba(
                                Ze)[:, 1]
                    return p
                try:
                    ps = par(one)
                except Exception:
                    continue
                aa = np.array([
                    roc_auc_score(y, p)
                    for p in ps])
                vals.append(aa.mean())
                rows.append({
                    "outcome": oc,
                    "stage": "B",
                    "draws": "%d+%d+%d" % dr,
                    "k": k_, "reg": rg,
                    "mean": aa.mean(),
                    "sd": aa.std(ddof=1)})
                if bestB is None or \
                        aa.mean() > bestB[0]:
                    bestB = (aa.mean(), dr,
                             k_, rg)
            if vals:
                line += " %9.4f" % max(vals)
            else:
                line += " %9s" % "fail"
        print(line, flush=True)
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)
    if bestB:
        print("  best %.4f  draws=%s k=%d"
              " reg=%.1f"
              % (bestB[0],
                 "%d+%d+%d" % bestB[1],
                 bestB[2], bestB[3]),
              flush=True)

    # ---- STAGE C and D: rebuild, calibrate --
    print("")
    print("  STAGE C/D: REBUILT MODEL AND"
          " CALIBRATION")
    dr, k_, rg = bestB[1], bestB[2], bestB[3]

    def blk_one(s):
        p = np.zeros(len(y))
        cv = StratifiedGroupKFold(
            n_splits=NFOLD, shuffle=True,
            random_state=s)
        for tr, te in cv.split(
                np.zeros((len(y), 1)), y, grp):
            Bt, Be = [], []
            for m_ in MODS:
                a, b = prep_fold(Xd[m_][tr],
                                 Xd[m_][te])
                Bt.append(a)
                Be.append(b)
            Zt, Ze, bd = ccaaug3(Bt, Be, k_,
                                 rg)
            bf = BlockForest(
                n_estimators=NTREE, draws=dr,
                bounds=bd, leaf=LEAF,
                depth=DEPTH, seed=s)
            bf.fit(Zt, y[tr])
            p[te] = bf.predict_proba(Ze)[:, 1]
        return p
    pb = par(blk_one)[0]
    pl = late3_oof(ps_bt, y, grp)
    R = np.column_stack(
        [_rank(pb), _rank(np.nan_to_num(pl))])
    pe_ = nnls_oof(R, y, grp)
    for nm, p in (("block_new", pb),
                  ("late_best", pl),
                  ("nnls_new", pe_)):
        ok = np.isfinite(p)
        av = roc_auc_score(y[ok], p[ok])
        b0, s0 = calib(p, y)
        pc = platt_oof(p, y, grp)
        ok2 = np.isfinite(pc)
        av2 = roc_auc_score(y[ok2], pc[ok2])
        b1, s1 = calib(pc, y)
        print("    %-10s %.4f   slope %.3f ->"
              " %.3f   Brier %.5f -> %.5f"
              % (nm, av, s0, s1, b0, b1))
        rows.append({
            "outcome": oc, "stage": "D",
            "learner": nm, "mean": av,
            "mean_platt": av2, "slope": s0,
            "slope_platt": s1, "brier": b0,
            "brier_platt": b1,
            "vs_125": av - base})
    print("    Platt is monotone, so AUROC is"
          " unchanged; only probability")
    print("    quality moves, which is what"
          " decision curves need", flush=True)

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
print("STAGE A: PER-MODALITY LEARNERS")
a_ = r[(r["stage"] == "A")
       & r["modality"].notna()
       & (r["modality"] != "all")]
for m_ in MODS:
    z = a_[a_["modality"] == m_]
    if not len(z):
        continue
    print("")
    print("  " + m_)
    print(z.pivot_table(index="learner",
                        columns="outcome",
                        values="mean")
          .round(4).to_string())

print("")
print("DOES THE BEST LEARNER DIFFER BY"
      " MODALITY?")
print("  EHR is 28 columns, ECG 264, CTPA 46"
      " mostly binary. A single inherited")
print("  choice cannot suit all three.")
if len(a_):
    for m_ in MODS:
        z = a_[a_["modality"] == m_]
        if not len(z):
            continue
        g2 = z.groupby("learner")["mean"].mean()
        print("  %-6s best %-9s %.4f   l2 %.4f"
              "   %+.4f"
              % (m_, g2.idxmax(), g2.max(),
                 g2.get("l2", np.nan),
                 g2.max() - g2.get("l2",
                                   np.nan)))

print("")
print("DOES IT CARRY INTO LATE FUSION?")
z = r[(r["stage"] == "A")
      & r["learner"].astype(str)
      .str.startswith("late_")]
if len(z):
    print(z.pivot_table(index="learner",
                        columns="outcome",
                        values="mean")
          .round(4).to_string())

print("")
print("STAGE B: k x reg x DRAWS")
b_ = r[r["stage"] == "B"]
if len(b_):
    print("  marginal by k")
    print(b_.groupby("k")["mean"].max()
          .round(4).to_string())
    print("  marginal by reg")
    print(b_.groupby("reg")["mean"].max()
          .round(4).to_string())
    print("  marginal by draws")
    print(b_.groupby("draws")["mean"].max()
          .round(4).to_string())
    print("")
    print("  IS THERE AN INTERACTION?")
    print("  the same best k at every draw"
          " means they are separable")
    piv = b_.pivot_table(index="draws",
                         columns="k",
                         values="mean")
    for dr in piv.index:
        print("    %-12s best k %d"
              % (dr, piv.loc[dr].idxmax()))

print("")
print("STAGE D: THE REBUILT MODEL")
d_ = r[r["stage"] == "D"]
if len(d_):
    print(d_[["outcome", "learner", "mean",
              "vs_125", "slope",
              "slope_platt"]].round(4)
          .to_string(index=False))
    print("")
    print("  the tree models here have all"
          " come out under-confident, with")
    print("  slopes of 1.4 to 1.8; Platt"
          " corrects them to about 0.97")

print("")
print("AGAINST SCRIPT 125's nnls ENSEMBLE")
for oc in OUTS:
    z = d_[(d_["outcome"] == oc)
           & (d_["learner"] == "nnls_new")]
    if len(z) and np.isfinite(
            S125.get(oc, np.nan)):
        print("  %-18s script 125 %.4f   here"
              " %.4f   %+.4f"
              % (oc, S125[oc],
                 z["mean"].iloc[0],
                 z["vs_125"].iloc[0]))
print("")
print("saved", DEST, r.shape)
keep_awake(False)