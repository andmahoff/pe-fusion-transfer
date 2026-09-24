"""Block-stratified feature sampling for
EHR + ECG.

In script 113 the forests drew max_features=8 of
312 columns (2.6%), about a third of the
proportion (8 of 94) that won on EHR+CTPA. The
312 columns are 28 EHR and 264 ECG features plus
canonical variates, so a random draw of 8 holds
about one EHR feature, although EHR is by far the
stronger modality.

This script compares
  proportional max_features, with the proportion
    extended to 0.40 and 0.60
  block sampling, which draws a set number of
    columns from each modality for every tree

Seeds run in parallel with joblib. Each
configuration seeds its own RNG, so the results
match a serial run exactly. Inner forests use
n_jobs=1.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\blockmtry.csv
  results\\blockmtry_log.txt
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
NJOBS = max(1, min(len(SEEDS),
                   (os.cpu_count() or 4) - 2))
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CCA_REG, CCA_K = 0.1, 10
NTREE = 300
PROP = [0.085, 0.15, 0.25, 0.40, 0.60]
BLOCK = [(4, 4), (4, 8), (8, 8), (8, 16),
         (14, 8), (14, 16)]
ECG_LO, ECG_HI = -12.0, 48.0
MINCOV = 0.20
OUTS = ["death_30d", "death_30d_inhosp",
        "composite_30d", "cv_first"]
DEST = os.path.join(PROC, "blockmtry.csv")
BEST_C = {"cv_first": 1e-4,
          "composite_30d": 1e-3,
          "death_30d": 1e-3,
          "death_30d_inhosp": 3e-4}
S113 = {"death_30d":
        {"late": 0.8700, "rf8": 0.8481,
         "rot8": 0.8445},
        "death_30d_inhosp":
        {"late": 0.8719, "rf8": 0.8738,
         "rot8": 0.8671},
        "composite_30d":
        {"late": 0.8421, "rf8": 0.8151,
         "rot8": 0.8158},
        "cv_first":
        {"late": 0.8221, "rf8": 0.7634,
         "rot8": 0.7741}}

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


class BlockForest:
    """Each tree is fitted on a column subset
    drawn separately from each modality: n_a from
    the EHR block, n_b from the ECG block.

    sklearn's max_features draws from the whole
    pool, so with 38 EHR-side and 274 ECG-side
    columns a draw of 8 gives about 1 EHR feature
    while the EHR modality carries 0.86 of the
    signal. Drawing per block guarantees the
    strong modality is always a candidate.

    Drawn per TREE rather than per SPLIT, which is
    the standard way to get the effect without a
    custom splitter, and gives every tree
    guaranteed EHR access throughout its depth.

    rotate=True applies a PCA rotation to the
    drawn columns first: the rotation forest
    version, on the same footing."""

    def __init__(self, n_estimators=300,
                 n_a=4, n_b=8, p_a=38,
                 leaf=5, depth=12,
                 rotate=False, size=3,
                 boot=0.75, seed=42):
        self.n = n_estimators
        self.n_a = n_a
        self.n_b = n_b
        self.p_a = p_a
        self.leaf = leaf
        self.depth = depth
        self.rotate = rotate
        self.size = size
        self.boot = boot
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        ia = np.arange(min(self.p_a, p))
        ib = np.arange(min(self.p_a, p), p)
        na = int(min(self.n_a, len(ia)))
        nb = int(min(self.n_b, len(ib)))
        nboot = max(10, int(round(
            self.boot * n)))
        self.sel_, self.rot_, self.tr_ = \
            [], [], []
        for _ in range(self.n):
            cols = np.concatenate([
                rng.choice(ia, na,
                           replace=False),
                rng.choice(ib, nb,
                           replace=False)])
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
                    rows = rng.choice(
                        n, nboot, replace=True)
                    bl.append((idx, _pca_block(
                        Z[np.ix_(rows, idx)])))
                Z2 = Z.copy()
                for idx, C in bl:
                    Z2[:, idx] = Z[:, idx] @ C
                Z = Z2
            rows = rng.choice(n, n,
                              replace=True)
            t = DecisionTreeClassifier(
                max_depth=self.depth,
                min_samples_leaf=self.leaf,
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


def learn_l2(Xt, yt, Xe, gt=None, C=None):
    if C is not None:
        m = LogisticRegression(C=C,
                               max_iter=3000)
        m.fit(Xt, yt)
        return m.predict_proba(Xe)[:, 1]
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


def rep_ccaaug(At, Ae, Bt, Be):
    """EHR block first, then its canonical
    variates, then ECG and its variates. So the
    block boundary for BlockForest is
    n_ehr + k."""
    A_, B_ = rcca(At, Bt, CCA_REG, CCA_K)
    if A_ is None:
        return (np.column_stack([At, Bt]),
                np.column_stack([Ae, Be]),
                At.shape[1])
    ma, mb = At.mean(0), Bt.mean(0)
    return (np.column_stack([
        At, (At - ma) @ A_, Bt,
        (Bt - mb) @ B_]),
        np.column_stack([
            Ae, (Ae - ma) @ A_, Be,
            (Be - mb) @ B_]),
        At.shape[1] + A_.shape[1])


def oof_prop(Xa, Xb, y, grp, seed, frac):
    """Random forest with max_features as a
    FRACTION, so the proportion matches what won
    on EHR+CTPA. n_jobs=1 because the outer seed
    loop already has the cores."""
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        At, Ae = prep_fold(Xa[tr], Xa[te])
        Bt, Be = prep_fold(Xb[tr], Xb[te])
        Zt, Ze, _ = rep_ccaaug(At, Ae, Bt, Be)
        mf = max(1, int(round(
            frac * Zt.shape[1])))
        m = RandomForestClassifier(
            n_estimators=NTREE,
            max_features=mf, max_depth=12,
            min_samples_leaf=5,
            class_weight=
            "balanced_subsample",
            random_state=42, n_jobs=1)
        m.fit(Zt, y[tr])
        p[te] = m.predict_proba(Ze)[:, 1]
    return p


def oof_block(Xa, Xb, y, grp, seed, na, nb,
              rotate=False):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        At, Ae = prep_fold(Xa[tr], Xa[te])
        Bt, Be = prep_fold(Xb[tr], Xb[te])
        Zt, Ze, pa = rep_ccaaug(At, Ae, Bt, Be)
        m = BlockForest(
            n_estimators=NTREE, n_a=na,
            n_b=nb, p_a=pa, leaf=5, depth=12,
            rotate=rotate, seed=seed)
        m.fit(Zt, y[tr])
        p[te] = m.predict_proba(Ze)[:, 1]
    return p


def oof_one(X, y, grp, seed, C=None):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        Xt, Xe = prep_fold(X[tr], X[te])
        p[te] = learn_l2(Xt, y[tr], Xe,
                         grp[tr], C)
    return p


def par_auc(fn, y, seeds=SEEDS, **kw):
    """Evaluate one configuration across seeds
    IN PARALLEL.

    Each seed's computation is unchanged and the
    RNG is seeded per configuration, so the
    numbers are identical to the serial version
    to the last digit. Only the order of
    completion differs."""
    ps = Parallel(n_jobs=NJOBS,
                  backend="loky")(
        delayed(fn)(seed=s, **kw)
        for s in seeds)
    return np.array([roc_auc_score(y, p)
                     for p in ps])


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


# ---------------- ECG features --------------
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
D = ECGA.merge(
    MA[["hadm_id"] + MC], on="hadm_id",
    how="left").merge(
    lab, on=["subject_id", "hadm_id"],
    how="inner").merge(
    mim_ehr[["hadm_id"] + EH], on="hadm_id",
    how="inner")
ECGC = LG + MC
ncol = len(EH) + len(ECGC) + 2 * CCA_K
print("")
print("cohort:", len(D))
print("  EHR %d   ECG %d   with variates %d"
      % (len(EH), len(ECGC), ncol))
print("  a random draw of 8 from %d gives %.1f"
      " EHR-side features on average"
      % (ncol, 8.0 * (len(EH) + CCA_K) / ncol))
print("  while the EHR modality carries about"
      " 0.86 of the signal")
print("  parallel over %d seeds on %d cores"
      % (NJOBS, os.cpu_count() or 0),
      flush=True)

# ---------- TIMING PROBE ----------
print("")
print("TIMING PROBE")
npro = min(len(D), 3500)
Xp = np.random.rand(npro, ncol)
yp = (np.random.rand(npro) > 0.89).astype(int)
t = time.time()
RandomForestClassifier(
    n_estimators=NTREE, max_features=27,
    max_depth=12, min_samples_leaf=5,
    n_jobs=1, random_state=42).fit(Xp, yp)
t_prop = time.time() - t
t = time.time()
BlockForest(n_estimators=NTREE, n_a=4, n_b=8,
            p_a=38, seed=42).fit(Xp, yp)
t_blk = time.time() - t
t = time.time()
BlockForest(n_estimators=100, n_a=4, n_b=8,
            p_a=38, rotate=True,
            seed=42).fit(Xp, yp)
t_brot = (time.time() - t) * (NTREE / 100.0)
print("  prop rf    %d trees: %6.1f s"
      % (NTREE, t_prop))
print("  block      %d trees: %6.1f s"
      % (NTREE, t_blk))
print("  block+rot  %d trees: %6.1f s"
      % (NTREE, t_brot))
ser = ((len(PROP) * t_prop
        + len(BLOCK) * (t_blk + t_brot))
       * NFOLD * len(SEEDS) * len(OUTS)
       / 60.0)
print("")
print("  serial would be %.0f min; with %d-way"
      " parallelism about %.0f min"
      % (ser, NJOBS, ser / NJOBS))
print("  Ctrl+C now if too long. Results save"
      " after each outcome.")
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
    Xa = d[EH].values.astype(float)
    Xb = d[ECGC].values.astype(float)
    cb = BEST_C.get(oc, 1e-3)
    pv = S113.get(oc, {})
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())))

    pa = _rank(oof_one(Xa, y, grp, 42))
    pb = _rank(oof_one(Xb, y, grp, 42, cb))
    a_e = roc_auc_score(y, pa)
    ref = roc_auc_score(y, late_from(pa, pb, y))
    print("  ehr %.4f   late %.4f" % (a_e, ref))
    print("  script 113 at mfeat=8:"
          " rf %.4f, rot %.4f"
          % (pv.get("rf8", np.nan),
             pv.get("rot8", np.nan)),
          flush=True)
    rows.append({
        "outcome": oc, "method": "late_wsrc",
        "na": np.nan, "nb": np.nan,
        "mean": ref, "sd": np.nan,
        "vs_late": 0.0, "vs_ehr": ref - a_e})
    rows.append({
        "outcome": oc, "method": "ehr",
        "na": np.nan, "nb": np.nan,
        "mean": a_e, "sd": np.nan,
        "vs_late": a_e - ref, "vs_ehr": 0.0})

    print("")
    print("  PROPORTIONAL max_features"
          "  (0.085 matches script 110)")
    print("  %-10s %6s %8s %8s %9s"
          % ("frac", "cols", "AUC", "SD",
             "vs late"))
    for fr in PROP:
        mf = max(1, int(round(fr * ncol)))
        try:
            aa = par_auc(oof_prop, y, Xa=Xa,
                         Xb=Xb, y_=None,
                         grp=grp, frac=fr) \
                if False else par_auc(
                    lambda seed, fr=fr:
                    oof_prop(Xa, Xb, y, grp,
                             seed, fr), y)
        except Exception as exc:
            print("  %-10s FAILED %s"
                  % (fr, repr(exc)[:40]))
            continue
        print("  %-10.3f %6d %8.4f %8.4f"
              " %+9.4f"
              % (fr, mf, aa.mean(),
                 aa.std(ddof=1),
                 aa.mean() - ref), flush=True)
        rows.append({
            "outcome": oc,
            "method": "prop_rf",
            "na": mf, "nb": np.nan,
            "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_late": aa.mean() - ref,
            "vs_ehr": aa.mean() - a_e})
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)

    for rot, lab_ in ((False, "block"),
                      (True, "block_rot")):
        print("")
        print("  %s  (n_ehr, n_ecg drawn"
              " separately per tree)" % lab_)
        print("  %-12s %8s %8s %9s"
              % ("ehr+ecg", "AUC", "SD",
                 "vs late"))
        for na, nb in BLOCK:
            try:
                aa = par_auc(
                    lambda seed, na=na, nb=nb,
                    rot=rot:
                    oof_block(Xa, Xb, y, grp,
                              seed, na, nb,
                              rot), y)
            except Exception as exc:
                print("  %-12s FAILED %s"
                      % ("%d+%d" % (na, nb),
                         repr(exc)[:40]))
                continue
            print("  %-12s %8.4f %8.4f %+9.4f"
                  % ("%d+%d" % (na, nb),
                     aa.mean(),
                     aa.std(ddof=1),
                     aa.mean() - ref),
                  flush=True)
            rows.append({
                "outcome": oc, "method": lab_,
                "na": na, "nb": nb,
                "mean": aa.mean(),
                "sd": aa.std(ddof=1),
                "vs_late": aa.mean() - ref,
                "vs_ehr": aa.mean() - a_e})
            pd.DataFrame(rows).to_csv(
                DEST, index=False)

    # calibration on the best block setting
    try:
        z = [x for x in rows
             if x["outcome"] == oc
             and x["method"] == "block"]
        if z:
            b_ = max(z, key=lambda x: x["mean"])
            p = oof_block(Xa, Xb, y, grp, 42,
                          int(b_["na"]),
                          int(b_["nb"]))
            b0, s0 = calib(p, y)
            pc = platt_oof(p, y, grp, 42)
            ok = np.isfinite(pc)
            b1, s1 = calib(pc[ok], y[ok])
            print("")
            print("  block %d+%d calibration:"
                  " slope %.3f -> %.3f"
                  "   Brier %.5f -> %.5f"
                  % (int(b_["na"]),
                     int(b_["nb"]), s0, s1,
                     b0, b1), flush=True)
    except Exception:
        pass

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
print("BEST OF EACH METHOD, gain over late")
b = r.loc[r.groupby(
    ["outcome", "method"])["mean"].idxmax()]
print(b.pivot_table(index="method",
                    columns="outcome",
                    values="vs_late")
      .round(4).to_string())

print("")
print("DOES MATCHING THE PROPORTION RECOVER"
      " THE ADVANTAGE?")
print("  script 113 used 8 of 312, or 2.6%,"
      " and lost by 0.022 to 0.070.")
print("  Script 110's winner was 8 of 94, or"
      " 8.5%. The 0.085 row is the")
print("  like-for-like rerun.")
p_ = r[r["method"] == "prop_rf"]
if len(p_):
    print(p_.pivot_table(index="na",
                         columns="outcome",
                         values="vs_late")
          .round(4).to_string())
    print("")
    print("  marginal by column count")
    print(p_.groupby("na")["vs_late"].mean()
          .round(4).to_string())

print("")
print("DOES BLOCK SAMPLING BEAT IT?")
print("  raising max_features helps only by"
      " accident, since it drags in more ECG")
print("  features alongside the EHR ones."
      " Block sampling guarantees the strong")
print("  modality is always a candidate while"
      " keeping the total small.")
for meth in ("block", "block_rot"):
    z = r[r["method"] == meth]
    if not len(z):
        continue
    z = z.copy()
    z["cfg"] = (z["na"].astype(int).astype(str)
                + "+"
                + z["nb"].astype(int)
                .astype(str))
    print("")
    print("  " + meth)
    print(z.pivot_table(index="cfg",
                        columns="outcome",
                        values="vs_late")
          .round(4).to_string())

print("")
print("COST PER UNIT OF GAIN")
print("  block 4+8 uses 12 columns per tree;"
      " prop 0.25 uses 78. If they reach")
print("  similar AUROC, block sampling is the"
      " better explanation of the mechanism.")
for oc in OUTS:
    s = r[r["outcome"] == oc]
    pb_ = s[s["method"] == "prop_rf"]
    bb_ = s[s["method"] == "block"]
    if len(pb_) and len(bb_):
        x = pb_.loc[pb_["mean"].idxmax()]
        z = bb_.loc[bb_["mean"].idxmax()]
        print("  %-18s prop %d cols %.4f"
              "   block %d+%d %.4f   %+.4f"
              % (oc, int(x["na"]), x["mean"],
                 int(z["na"]), int(z["nb"]),
                 z["mean"], z["mean"]
                 - x["mean"]))

print("")
print("WINS OVER late:wsrc")
w = r[(r["vs_late"] > 0)
      & (~r["method"].isin(
          ["late_wsrc", "ehr"]))]
tot = len(r[~r["method"].isin(
    ["late_wsrc", "ehr"])])
print("  %d of %d cells" % (len(w), tot))
if len(w):
    print(w[["outcome", "method", "na", "nb",
             "mean", "vs_late"]].round(4)
          .to_string(index=False))

print("")
print("BEST PER OUTCOME")
for oc in OUTS:
    s = r[(r["outcome"] == oc)
          & (~r["method"].isin(
              ["late_wsrc", "ehr"]))]
    lt = r[(r["outcome"] == oc)
           & (r["method"] == "late_wsrc")]
    if not len(s) or not len(lt):
        continue
    x = s.loc[s["mean"].idxmax()]
    pv = S113.get(oc, {})
    print("  %-18s %-10s %.4f   late %.4f"
          "   %+.4f   (script 113 best %.4f)"
          % (oc, x["method"], x["mean"],
             lt["mean"].iloc[0], x["vs_late"],
             max(pv.get("rf8", 0),
                 pv.get("rot8", 0))))
print("")
print("saved", DEST, r.shape)
keep_awake(False)