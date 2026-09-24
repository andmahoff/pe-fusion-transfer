"""EHR share in three-modality block sampling,
across four cohort strategies.

In EHR+ECG the winning draw was 28+8, 78% EHR
columns per tree. Script 117's three-modality
winner was 20+8+16, 45% EHR, and its
one-at-a-time sweep tested EHR shares of only 27%
to 61%, with the CTPA and ECG draws both best at
the lowest value tested.

CONFIGURATIONS, chosen by EHR share
  28+4+4   78%, the two-modality optimum
  28+2+2   88%
  38+4+4   83%
  20+4+4   71%
  28+4+2   and 28+2+4, to see whether the small
           remainder should favour ECG or CTPA
  28+8+8   64%
  20+8+16  45%, script 117's winner, as reference

COHORT STRATEGIES
  complete  the 1,636 with a CTPA (script 117)
  nan_ind   all 3,501, CTPA left missing plus an
            indicator (script 118)
  masked    all 3,501, half the trees built
            without the CTPA block (script 119)
  ccs_rank  complete-case pattern submodels,
            merged by rank (script 119). The
            Platt-merged variants are left out:
            they calibrated on in-sample
            predictions, which are near-perfect
            for a deep forest.

The reference is ehr_ecg, which ignores CTPA:
0.8916, 0.9051, 0.8443 and 0.7865 on the full
cohort in script 118.

Settings otherwise from script 117: leaf 1,
depth 12, CCA k 5, reg 0.5, 600 trees.

Parallel across seeds. Timing probe first.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\share_sweep.csv
  results\\share_sweep_log.txt
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
MASK_FREE = 0.5
# (n_ehr, n_ecg, n_ctpa)
CONFIGS = [(28, 4, 4), (28, 2, 2),
           (38, 4, 4), (20, 4, 4),
           (28, 4, 2), (28, 2, 4),
           (28, 8, 8), (20, 8, 16)]
STRATS = ["complete", "nan_ind", "masked",
          "ccs_rank"]
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
OUTS = ["death_30d", "death_30d_inhosp",
        "composite_30d", "cv_first"]
DEST = os.path.join(PROC, "share_sweep.csv")
BEST_C = {"cv_first": 1e-4,
          "composite_30d": 1e-3,
          "death_30d": 1e-3,
          "death_30d_inhosp": 3e-4}
# script 118, ehr_ecg full cohort: the bar
BAR = {"death_30d": 0.8916,
       "death_30d_inhosp": 0.9051,
       "composite_30d": 0.8443,
       "cv_first": 0.7865}

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


def prep_keepnan(Xtr, Xte, htr, hte):
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


class BlockForest3:
    """Block draws with native NaN handling.

    mask=True builds a fixed fraction of trees
    WITHOUT the CTPA block, so every patient has
    voters. Script 118's version had none, so the
    without-CTPA patients scored exactly 0.5000."""

    def __init__(self, n_estimators=600,
                 draws=(28, 4, 4),
                 bounds=(38, 312), leaf=1,
                 depth=12, mask=False,
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
        b0 = int(min(self.bounds[0], p))
        b1 = int(min(self.bounds[1], p))
        blk = [np.arange(0, b0),
               np.arange(b0, b1),
               np.arange(b1, p)]
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
                kk = int(min(max(1, w),
                             len(ix)))
                parts.append(rng.choice(
                    ix, kk, replace=False))
            if not parts:
                continue
            cols = np.concatenate(parts)
            uses_ct = (use_ct
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


def build_rep(At, Ae, Bt, Be, Ct, Ce,
              htr, hte, use_ct, keep_nan,
              add_ind):
    pa, pb, pc = [At], [Bt], []
    qa, qb, qc = [Ae], [Be], []
    W1, W2 = rcca(At, Bt, REG, K)
    if W1 is not None:
        ma, mb = At.mean(0), Bt.mean(0)
        pa.append((At - ma) @ W1)
        qa.append((Ae - ma) @ W1)
        pb.append((Bt - mb) @ W2)
        qb.append((Be - mb) @ W2)
    if use_ct:
        pc.append(Ct)
        qc.append(Ce)
        if htr.sum() > 50:
            A2 = np.nan_to_num(At[htr])
            C2 = np.nan_to_num(Ct[htr])
            V1, V2 = rcca(A2, C2, REG, K)
            if V1 is not None:
                mu, mv = A2.mean(0), C2.mean(0)
                za = (np.nan_to_num(At)
                      - mu) @ V1
                ze = (np.nan_to_num(Ae)
                      - mu) @ V1
                zc = (np.nan_to_num(Ct)
                      - mv) @ V2
                zd = (np.nan_to_num(Ce)
                      - mv) @ V2
                if keep_nan:
                    za[~htr] = np.nan
                    ze[~hte] = np.nan
                    zc[~htr] = np.nan
                    zd[~hte] = np.nan
                pa.append(za)
                qa.append(ze)
                pc.append(zc)
                qc.append(zd)
    if add_ind:
        pa.append(htr.astype(float)
                  .reshape(-1, 1))
        qa.append(hte.astype(float)
                  .reshape(-1, 1))
    Zt = np.column_stack(pa + pb + pc)
    Ze = np.column_stack(qa + qb + qc)
    b0 = sum(x.shape[1] for x in pa)
    b1 = b0 + sum(x.shape[1] for x in pb)
    return Zt, Ze, (b0, b1)


def fit_pred(Atr, Ate, Btr, Bte, Ctr, Cte,
             ytr, htr, hte, seed, draws,
             use_ct=True, keep_nan=False,
             add_ind=False, mask=False):
    At, Ae = prep_fold(Atr, Ate)
    Bt, Be = prep_fold(Btr, Bte)
    if keep_nan:
        Ct, Ce = prep_keepnan(Ctr, Cte, htr,
                              hte)
    elif use_ct:
        Ct, Ce = prep_fold(Ctr, Cte)
    else:
        Ct, Ce = Ctr, Cte
    Zt, Ze, bd = build_rep(
        At, Ae, Bt, Be, Ct, Ce, htr, hte,
        use_ct, keep_nan, add_ind)
    dr = (draws[0], draws[1],
          draws[2] if use_ct else 0)
    m = BlockForest3(
        n_estimators=NTREE, draws=dr,
        bounds=bd, leaf=LEAF, depth=DEPTH,
        mask=mask, seed=seed)
    m.fit(Zt, ytr, have=htr if mask else None)
    return m.predict_proba(
        Ze, have=hte if mask else None)[:, 1]


def oof_strat(Xa, Xb, Xc, y, grp, have, seed,
              draws, strat):
    p = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    if strat == "complete":
        keep = np.where(have)[0]
        yk, gk = y[keep], grp[keep]
        for tr, te in cv.split(
                np.zeros((len(yk), 1)), yk,
                gk):
            p[keep[te]] = fit_pred(
                Xa[keep][tr], Xa[keep][te],
                Xb[keep][tr], Xb[keep][te],
                Xc[keep][tr], Xc[keep][te],
                yk[tr], have[keep][tr],
                have[keep][te], seed, draws)
        return p
    if strat in ("nan_ind", "masked"):
        for tr, te in cv.split(
                np.zeros((len(y), 1)), y,
                grp):
            p[te] = fit_pred(
                Xa[tr], Xa[te], Xb[tr],
                Xb[te], Xc[tr], Xc[te], y[tr],
                have[tr], have[te], seed,
                draws, True, True,
                strat == "nan_ind",
                strat == "masked")
        return p
    # ccs_rank: submodel per pattern, the
    # reduced one trained on every row, merged
    # by rank within subgroup
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        tr_h = tr[have[tr]]
        te_h = te[have[te]]
        te_n = te[~have[te]]
        if (len(te_h) and len(tr_h) > 50
                and y[tr_h].sum() >= 5):
            q = fit_pred(
                Xa[tr_h], Xa[te_h], Xb[tr_h],
                Xb[te_h], Xc[tr_h], Xc[te_h],
                y[tr_h], have[tr_h],
                have[te_h], seed, draws)
            p[te_h] = _rank(q)
        if len(te_n) and len(tr) > 50:
            q = fit_pred(
                Xa[tr], Xa[te_n], Xb[tr],
                Xb[te_n], Xc[tr], Xc[te_n],
                y[tr], have[tr], have[te_n],
                seed, draws, False)
            p[te_n] = _rank(q)
    return p


def oof_one(X, y, grp, seed, C=None,
            grid=None, sub=None):
    idx = (np.arange(len(y)) if sub is None
           else np.where(sub)[0])
    p = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(idx), 1)), y[idx],
            grp[idx]):
        Xt, Xe = prep_fold(X[idx][tr],
                           X[idx][te])
        p[idx[te]] = learn_l2(
            Xt, y[idx][tr], Xe,
            grp[idx][tr], C, grid)
    return p


def par(fn, seeds=SEEDS):
    return Parallel(n_jobs=NJOBS,
                    backend="loky")(
        delayed(fn)(s) for s in seeds)


def auc_on(p, y, sub=None):
    ok = np.isfinite(p)
    if sub is not None:
        ok = ok & sub
    if ok.sum() < 30 or y[ok].sum() < 5:
        return np.nan
    return roc_auc_score(y[ok], p[ok])


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
    how="left")
D["has_ctpa"] = D[CT].notna().all(
    axis=1).astype(int)
ECGC = LG + MC
print("")
print("cohort:", len(D),
      "  with CTPA %d   without %d"
      % (int(D["has_ctpa"].sum()),
         int((1 - D["has_ctpa"]).sum())))
print("  EHR %d   ECG %d   CTPA %d"
      % (len(EH), len(ECGC), len(CT)))
print("")
print("  CONFIGURATIONS BY EHR SHARE")
print("  %-12s %6s %8s" % ("draws", "total",
                           "EHR %"))
for cfg in CONFIGS:
    tot = sum(cfg)
    print("  %-12s %6d %7.0f%%"
          % ("%d+%d+%d" % cfg, tot,
             100.0 * cfg[0] / tot))
print("  script 117's winner was 20+8+16 at"
      " 45%%; EHR+ECG won at 28+8 = 78%%",
      flush=True)

# ---------- TIMING PROBE ----------
print("")
print("TIMING PROBE")
npro = min(len(D), 3500)
ncol = len(EH) + len(ECGC) + len(CT) + 4 * K + 1
Xp = np.random.rand(npro, ncol)
yp = (np.random.rand(npro) > 0.89).astype(int)
hp = np.random.rand(npro) > 0.53
t = time.time()
BlockForest3(n_estimators=NTREE,
             draws=(28, 4, 4),
             bounds=(38 + K, 312), leaf=LEAF,
             depth=DEPTH, seed=42).fit(Xp, yp)
t_small = time.time() - t
t = time.time()
BlockForest3(n_estimators=NTREE,
             draws=(20, 8, 16),
             bounds=(38 + K, 312), leaf=LEAF,
             depth=DEPTH, seed=42).fit(Xp, yp)
t_big = time.time() - t
per = 0.5 * (t_small + t_big)
# complete uses ~47% of rows; ccs fits two
# forests per fold
wts = {"complete": 0.47, "nan_ind": 1.0,
       "masked": 1.0, "ccs_rank": 1.5}
cells = len(CONFIGS) * len(OUTS)
ser = sum(wts[s] for s in STRATS) * cells \
    * per * NFOLD * len(SEEDS) / 60.0
print("  %d trees, 28+4+4 : %5.1f s"
      % (NTREE, t_small))
print("  %d trees, 20+8+16: %5.1f s"
      % (NTREE, t_big))
print("")
print("  %d configs x %d strategies x %d"
      " outcomes = %d cells"
      % (len(CONFIGS), len(STRATS),
         len(OUTS), cells * len(STRATS)))
print("  serial %.0f min; with %d-way"
      " parallelism about %.0f min"
      % (ser, NJOBS, ser / NJOBS))
print("  Ctrl+C now if too long. Results save"
      " after every strategy.")
time.sleep(5)

rows = []
t0 = time.time()

for oc in OUTS:
    if oc not in D.columns:
        continue
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    have = d["has_ctpa"].values == 1
    if y.sum() < 25:
        continue
    Xa = d[EH].values.astype(float)
    Xb = d[ECGC].values.astype(float)
    Xc = d[CT].values.astype(float)
    bar = BAR.get(oc, np.nan)
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d   with CTPA %d"
          % (oc, len(y), int(y.sum()),
             int(have.sum())))
    print("  the bar is ehr_ecg at %.4f,"
          " which ignores CTPA entirely"
          % bar, flush=True)

    for strat in STRATS:
        print("")
        print("  %s" % strat)
        print("  %-12s %6s %9s %9s %9s %9s"
              % ("draws", "EHR%", "AUC all",
                 "AUC cc", "AUC no-ct",
                 "vs bar"))
        for cfg in CONFIGS:
            try:
                ps = par(lambda s, cfg=cfg,
                         st=strat:
                         oof_strat(
                             Xa, Xb, Xc, y,
                             grp, have, s,
                             cfg, st))
            except Exception as exc:
                print("  %-12s FAILED %s"
                      % ("%d+%d+%d" % cfg,
                         repr(exc)[:40]))
                continue
            aa = np.array([auc_on(p, y)
                           for p in ps])
            cc = np.array([auc_on(p, y, have)
                           for p in ps])
            nn = np.array([auc_on(p, y, ~have)
                           for p in ps])
            sh = 100.0 * cfg[0] / sum(cfg)
            base = (np.nanmean(cc)
                    if strat == "complete"
                    else np.nanmean(aa))
            print("  %-12s %5.0f%% %9.4f %9.4f"
                  " %9.4f %+9.4f"
                  % ("%d+%d+%d" % cfg, sh,
                     np.nanmean(aa),
                     np.nanmean(cc),
                     np.nanmean(nn),
                     base - bar), flush=True)
            rows.append({
                "outcome": oc,
                "strategy": strat,
                "draws": "%d+%d+%d" % cfg,
                "n_ehr": cfg[0],
                "n_ecg": cfg[1],
                "n_ctpa": cfg[2],
                "ehr_share": sh,
                "mean_all": np.nanmean(aa),
                "sd_all": np.nanstd(aa,
                                    ddof=1),
                "mean_cc": np.nanmean(cc),
                "mean_noct": np.nanmean(nn),
                "vs_bar": base - bar})
            pd.DataFrame(rows).to_csv(
                DEST, index=False)

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
print("GAIN OVER ehr_ecg, BY CONFIGURATION")
for st in STRATS:
    z = r[r["strategy"] == st]
    if not len(z):
        continue
    print("")
    print("  " + st)
    print(z.pivot_table(index="draws",
                        columns="outcome",
                        values="vs_bar")
          .round(4).to_string())

print("")
print("DOES A HIGHER EHR SHARE HELP?")
print("  script 117 tested 27 to 61 percent"
      " and never reached the 78 percent")
print("  that won on EHR+ECG")
if len(r):
    z = r.groupby("ehr_share")["vs_bar"] \
        .mean()
    print(z.round(4).to_string())
    bs = z.idxmax()
    print("")
    print("  best share %.0f%%%s" % (bs,
          "   (at the grid edge)"
          if bs >= 83 else ""))
    from scipy import stats as st_
    x = r["ehr_share"].values
    yv = r["vs_bar"].values
    ok = np.isfinite(x) & np.isfinite(yv)
    if ok.sum() > 5 and np.std(x[ok]) > 0:
        rr, pp = st_.pearsonr(x[ok], yv[ok])
        print("  share vs gain: r = %+.3f"
              " (p = %.4f) over %d cells"
              % (rr, pp, int(ok.sum())))

print("")
print("ECG vs CTPA IN THE SMALL REMAINDER")
print("  28+4+2 gives the remainder to ECG;"
      " 28+2+4 gives it to CTPA")
for oc in OUTS:
    a = r[(r["outcome"] == oc)
          & (r["draws"] == "28+4+2")]
    b = r[(r["outcome"] == oc)
          & (r["draws"] == "28+2+4")]
    if len(a) and len(b):
        print("  %-18s ecg-heavy %.4f"
              "   ctpa-heavy %.4f   %+.4f"
              % (oc, a["vs_bar"].mean(),
                 b["vs_bar"].mean(),
                 a["vs_bar"].mean()
                 - b["vs_bar"].mean()))

print("")
print("BEST STRATEGY")
if len(r):
    print(r.groupby("strategy")["vs_bar"]
          .max().round(4).to_string())

print("")
print("=" * 74)
print("BEST CELL PER OUTCOME")
for oc in OUTS:
    s = r[r["outcome"] == oc]
    if not len(s):
        continue
    x = s.loc[s["vs_bar"].idxmax()]
    print("  %-18s %-10s %-10s %.4f"
          "   bar %.4f   %+.4f"
          % (oc, x["strategy"], x["draws"],
             x["mean_all"] if
             x["strategy"] != "complete"
             else x["mean_cc"],
             BAR.get(oc, np.nan),
             x["vs_bar"]))

print("")
print("WINS OVER ehr_ecg")
w = r[r["vs_bar"] > 0]
print("  %d of %d cells" % (len(w), len(r)))
if len(w):
    print(w[["outcome", "strategy", "draws",
             "ehr_share", "vs_bar"]].round(4)
          .to_string(index=False))
else:
    print("  none. Four routes have now said"
          " the CTPA modality does not pay:")
    print("  late fusion, joint learning on"
          " complete cases, imputation-free")
    print("  full-cohort learning, and pattern"
          " submodels. The share question is")
    print("  now closed too.")
print("")
print("saved", DEST, r.shape)
keep_awake(False)