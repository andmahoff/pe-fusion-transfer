"""Pattern submodels for the missing CTPA
(Mercaldo and Blume, 2020).

One submodel for patients with all three
modalities and one for those with EHR and ECG
only, in two variants:
  PS   each submodel trains only on its own
       pattern, so the no-CTPA model uses the
       1,865 without one
  CCS  each submodel trains on every patient who
       has at least its predictors, so the
       no-CTPA model uses all 3,501

Scores from two submodels are on different
scales, so each is Platt-calibrated within its
own subgroup before merging, and a
rank-within-subgroup merge is tested alongside.

The masked variant builds a fixed fraction of
trees without the CTPA block. In script 118 every
retained tree used CTPA, so patients without one
received no votes and scored 0.5000.

Parallel across seeds. Results are saved after
every outcome.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\pattern_submodels.csv
  results\\pattern_submodels_log.txt
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
NFOLD = 5
NJOBS = max(1, min(10, (os.cpu_count() or 4)
                   - 2))
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CT_CS = list(np.logspace(-4, 4, 10))
N_EHR, N_ECG, N_CT = 28, 8, 16
LEAF, DEPTH, K, REG, NTREE = 1, 12, 5, 0.5, 600
# fraction of trees built without the CTPA block
# in the masked variant, so every patient has voters
MASK_FREE = 0.5
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
OUTS = ["death_30d", "death_30d_inhosp",
        "composite_30d", "cv_first"]
DEST = os.path.join(
    PROC, "pattern_submodels.csv")
BEST_C = {"cv_first": 1e-4,
          "composite_30d": 1e-3,
          "death_30d": 1e-3,
          "death_30d_inhosp": 3e-4}
# script 118, complete-case AUROC
S118 = {"death_30d":
        {"complete": 0.8857, "ehr_ecg": 0.8960,
         "nan_ind": 0.8801, "late2": 0.8884},
        "death_30d_inhosp":
        {"complete": 0.9037, "ehr_ecg": 0.9130,
         "nan_ind": 0.8985, "late2": 0.8728},
        "composite_30d":
        {"complete": 0.8462, "ehr_ecg": 0.8542,
         "nan_ind": 0.8448, "late2": 0.8500},
        "cv_first":
        {"complete": 0.7854, "ehr_ecg": 0.7888,
         "nan_ind": 0.7960, "late2": 0.8207}}

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


class BlockForestNaN:
    """Block draws with native NaN handling.

    mask=True builds a FIXED FRACTION of trees
    without the CTPA block, so every patient has
    voters. In script 118 every retained tree
    used CTPA, so the without-CTPA patients got
    no votes and scored exactly 0.5000."""

    def __init__(self, n_estimators=600,
                 draws=(28, 8, 16),
                 bounds=(38, 312), leaf=1,
                 depth=12, mask=False,
                 mask_free=MASK_FREE,
                 seed=42):
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
        for t_i in range(self.n):
            use_ct = True
            if self.mask and t_i < nfree:
                use_ct = False
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
            uses_ct = (use_ct and len(blk[2]) > 0
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
        out = np.where(den > 0, num
                       / np.where(den > 0, den,
                                  1.0), 0.5)
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


def fit_predict(Atr, Ate, Btr, Bte, Ctr, Cte,
                ytr, htr, hte, seed,
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
    dr = (N_EHR, N_ECG,
          N_CT if use_ct else 0)
    m = BlockForestNaN(
        n_estimators=NTREE, draws=dr,
        bounds=bd, leaf=LEAF, depth=DEPTH,
        mask=mask, seed=seed)
    m.fit(Zt, ytr,
          have=htr if mask else None)
    return m.predict_proba(
        Ze, have=hte if mask else None)[:, 1]


def platt_sub(p_tr, y_tr, p_te):
    """Platt scaling fitted within one pattern.

    Two submodels produce scores on different
    scales, and script 49's hybrid_two failed
    for exactly that reason. Calibrating each
    within its own subgroup puts them on a
    common probability scale before merging."""
    q = np.clip(p_tr, 1e-6, 1 - 1e-6)
    x = np.log(q / (1 - q)).reshape(-1, 1)
    r = np.clip(p_te, 1e-6, 1 - 1e-6)
    z = np.log(r / (1 - r)).reshape(-1, 1)
    try:
        m = LogisticRegression(C=1e6,
                               max_iter=2000)
        m.fit(x, y_tr)
        return m.predict_proba(z)[:, 1]
    except Exception:
        return p_te


def oof_arm(Xa, Xb, Xc, y, grp, have, seed,
            arm):
    """arm: complete | ehr_ecg | nan_ind |
    masked | ps_platt | ps_rank | ccs_platt |
    ccs_rank"""
    p = np.full(len(y), np.nan)
    if arm in ("complete",):
        keep = np.where(have)[0]
    else:
        keep = np.arange(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)

    if arm in ("complete", "ehr_ecg",
               "nan_ind", "masked"):
        yk, gk = y[keep], grp[keep]
        Ak, Bk, Ck = Xa[keep], Xb[keep], \
            Xc[keep]
        hk = have[keep]
        use_ct = arm != "ehr_ecg"
        keep_nan = arm in ("nan_ind", "masked")
        for tr, te in cv.split(
                np.zeros((len(yk), 1)), yk,
                gk):
            p[keep[te]] = fit_predict(
                Ak[tr], Ak[te], Bk[tr], Bk[te],
                Ck[tr], Ck[te], yk[tr], hk[tr],
                hk[te], seed, use_ct, keep_nan,
                arm == "nan_ind",
                arm == "masked")
        return p

    # ---- pattern submodels ----
    # submodel 1: patients with a CTPA, all
    #   three modalities
    # submodel 2: patients without, EHR+ECG
    #   PS  trains it on the no-CTPA rows only
    #   CCS trains it on every row, since they
    #       all have "at least the observed
    #       predictors of the pattern"
    ccs = arm.startswith("ccs")
    rankmode = arm.endswith("rank")
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        tr_h = tr[have[tr]]
        te_h = te[have[te]]
        tr_n = tr if ccs else tr[~have[tr]]
        te_n = te[~have[te]]
        # submodel 1, three modalities
        if len(te_h) and len(tr_h) > 50 \
                and y[tr_h].sum() >= 5:
            q_tr = fit_predict(
                Xa[tr_h], Xa[tr_h], Xb[tr_h],
                Xb[tr_h], Xc[tr_h], Xc[tr_h],
                y[tr_h], have[tr_h], have[tr_h],
                seed, True, False, False, False)
            q_te = fit_predict(
                Xa[tr_h], Xa[te_h], Xb[tr_h],
                Xb[te_h], Xc[tr_h], Xc[te_h],
                y[tr_h], have[tr_h], have[te_h],
                seed, True, False, False, False)
            p[te_h] = (_rank(q_te) if rankmode
                       else platt_sub(
                           q_tr, y[tr_h], q_te))
        # submodel 2, EHR + ECG only
        if len(te_n) and len(tr_n) > 50 \
                and y[tr_n].sum() >= 5:
            q_tr = fit_predict(
                Xa[tr_n], Xa[tr_n], Xb[tr_n],
                Xb[tr_n], Xc[tr_n], Xc[tr_n],
                y[tr_n], have[tr_n], have[tr_n],
                seed, False, False, False,
                False)
            q_te = fit_predict(
                Xa[tr_n], Xa[te_n], Xb[tr_n],
                Xb[te_n], Xc[tr_n], Xc[te_n],
                y[tr_n], have[tr_n], have[te_n],
                seed, False, False, False,
                False)
            p[te_n] = (_rank(q_te) if rankmode
                       else platt_sub(
                           q_tr, y[tr_n], q_te))
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


def late2(pa, pb, y):
    bs, bv = -1.0, pa
    for w in np.arange(0, 1.001, 0.05):
        v = w * pb + (1 - w) * pa
        a = roc_auc_score(y, v)
        if a > bs:
            bs, bv = a, v
    return bv


def calib(p, y, sub=None):
    ok = np.isfinite(p)
    if sub is not None:
        ok = ok & sub
    if ok.sum() < 30:
        return np.nan, np.nan
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
print("cohort:", len(D))
print("  with CTPA %d   without %d"
      % (int(D["has_ctpa"].sum()),
         int((1 - D["has_ctpa"]).sum())))
print("  EHR %d   ECG %d   CTPA %d"
      % (len(EH), len(ECGC), len(CT)))
print("  PS  trains the no-CTPA submodel on the"
      " %d without one"
      % int((1 - D["has_ctpa"]).sum()))
print("  CCS trains it on all %d, since they"
      " all have the observed predictors"
      % len(D))
print("  masked now builds %.0f%% of trees"
      " without the CTPA block, so every"
      % (100 * MASK_FREE))
print("  patient has voters", flush=True)

rows = []
t0 = time.time()
ARMS = ["complete", "ehr_ecg", "nan_ind",
        "masked", "ps_platt", "ps_rank",
        "ccs_platt", "ccs_rank"]

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
    cb = BEST_C.get(oc, 1e-3)
    pv = S118.get(oc, {})
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d   with CTPA %d"
          " (ev %d)"
          % (oc, len(y), int(y.sum()),
             int(have.sum()),
             int(y[have].sum())))

    pe = _rank(oof_one(Xa, y, grp, 42))
    pc_ = oof_one(Xc, y, grp, 42, grid=CT_CS,
                  sub=have)
    L2 = late2(_rank(pe[have]),
               _rank(pc_[have]), y[have])
    ref2 = roc_auc_score(y[have], L2)
    print("  late ehr+ctpa, complete cases:"
          " %.4f" % ref2)
    print("  script 118: complete %.4f,"
          " ehr_ecg %.4f, nan_ind %.4f"
          % (pv.get("complete", np.nan),
             pv.get("ehr_ecg", np.nan),
             pv.get("nan_ind", np.nan)),
          flush=True)

    print("")
    print("  %-11s %9s %9s %9s"
          % ("arm", "AUC all", "AUC cc",
             "AUC no-ct"))
    store = {}
    for arm in ARMS:
        try:
            ps = par(lambda s, arm=arm:
                     oof_arm(Xa, Xb, Xc, y,
                             grp, have, s,
                             arm))
        except Exception as exc:
            print("  %-11s FAILED %s"
                  % (arm, repr(exc)[:45]))
            continue
        aa = np.array([auc_on(p, y)
                       for p in ps])
        cc = np.array([auc_on(p, y, have)
                       for p in ps])
        nn = np.array([auc_on(p, y, ~have)
                       for p in ps])
        store[arm] = ps[0]
        print("  %-11s %9.4f %9.4f %9.4f"
              % (arm, np.nanmean(aa),
                 np.nanmean(cc),
                 np.nanmean(nn)), flush=True)
        rows.append({
            "outcome": oc, "arm": arm,
            "mean_all": np.nanmean(aa),
            "sd_all": np.nanstd(aa, ddof=1),
            "mean_cc": np.nanmean(cc),
            "mean_noct": np.nanmean(nn),
            "vs_late2": np.nanmean(cc) - ref2})
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)

    # the masked fix, verified
    if "masked" in store:
        v = auc_on(store["masked"], y, ~have)
        print("")
        print("  MASKED FIX CHECK: without-CTPA"
              " AUROC %.4f" % v)
        print("    script 118 gave exactly"
              " 0.5000 here because those")
        print("    patients received no votes"
              " at all")

    # the routing question
    best_full = None
    for arm in ("ehr_ecg", "ps_platt",
                "ps_rank", "ccs_platt",
                "ccs_rank"):
        z = [x for x in rows
             if x["outcome"] == oc
             and x["arm"] == arm]
        if z and np.isfinite(z[0]["mean_all"]):
            if (best_full is None
                    or z[0]["mean_all"]
                    > best_full[1]):
                best_full = (arm,
                             z[0]["mean_all"])
    if best_full:
        ez = [x for x in rows
              if x["outcome"] == oc
              and x["arm"] == "ehr_ecg"]
        print("")
        print("  best full-cohort variant: %s %.4f"
              % best_full)
        if ez:
            print("    against ehr_ecg %.4f,"
                  " so %+.4f"
                  % (ez[0]["mean_all"],
                     best_full[1]
                     - ez[0]["mean_all"]),
                  flush=True)

    # bootstrap: best routed vs ehr_ecg
    if ("ccs_platt" in store
            and "ehr_ecg" in store):
        a = store["ccs_platt"]
        b = store["ehr_ecg"]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 100:
            gn, lo, hi, _ = f.boot_diff(
                y[ok], _rank(a[ok]),
                _rank(b[ok]), grp[ok])
            print("")
            print("  ccs_platt vs ehr_ecg,"
                  " full cohort")
            print("    %+.4f [%+.4f,%+.4f] %s"
                  % (gn, lo, hi,
                     "*" if (lo > 0 or hi < 0)
                     else ""), flush=True)
            rows.append({
                "outcome": oc,
                "arm": "ccs_vs_ehrecg",
                "mean_all": gn, "lo": lo,
                "hi": hi,
                "sig": int(lo > 0 or hi < 0)})

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
q = r[r["arm"].isin(ARMS)]

print("")
print("=" * 74)
print("AUROC ON THE FULL COHORT")
print("  'complete' cannot score the patients"
      " it never saw, so its column here is")
print("  complete-case only")
if len(q):
    print(q.pivot_table(index="arm",
                        columns="outcome",
                        values="mean_all")
          .round(4).to_string())

print("")
print("AUROC ON THE COMPLETE CASES")
if len(q):
    print(q.pivot_table(index="arm",
                        columns="outcome",
                        values="mean_cc")
          .round(4).to_string())

print("")
print("AUROC ON THE PATIENTS WITHOUT A CTPA")
if len(q):
    print(q.pivot_table(index="arm",
                        columns="outcome",
                        values="mean_noct")
          .round(4).to_string())

print("")
print("=" * 74)
print("DO PATTERN SUBMODELS HELP?")
print("  Mercaldo and Blume show PS minimises"
      " expected loss when each pattern-")
print("  specific loss is minimised, and report"
      " it beating zero-imputation, mean-")
print("  imputation, complete-case analysis and"
      " multiple imputation.")
print("")
print("  %-18s %9s %9s %9s %9s"
      % ("outcome", "ehr_ecg", "ps", "ccs",
         "best gain"))
for oc in OUTS:
    s = q[q["outcome"] == oc]
    if not len(s):
        continue

    def gv(a):
        z = s[s["arm"] == a]
        return (z["mean_all"].iloc[0]
                if len(z) else np.nan)
    e = gv("ehr_ecg")
    ps_ = np.nanmax([gv("ps_platt"),
                     gv("ps_rank")])
    cs_ = np.nanmax([gv("ccs_platt"),
                     gv("ccs_rank")])
    print("  %-18s %9.4f %9.4f %9.4f %+9.4f"
          % (oc, e, ps_, cs_,
             np.nanmax([ps_, cs_]) - e))

print("")
print("PS vs CCS")
print("  CCS trains the reduced submodel on"
      " every row rather than only the")
print("  no-CTPA rows, and script 118 showed"
      " cohort size dominates here")
for oc in OUTS:
    s = q[q["outcome"] == oc]
    a = s[s["arm"] == "ps_platt"]
    b = s[s["arm"] == "ccs_platt"]
    if len(a) and len(b):
        print("  %-18s ps %.4f   ccs %.4f"
              "   %+.4f"
              % (oc, a["mean_all"].iloc[0],
                 b["mean_all"].iloc[0],
                 b["mean_all"].iloc[0]
                 - a["mean_all"].iloc[0]))

print("")
print("PLATT vs RANK MERGING")
print("  script 49's hybrid_two failed on scale"
      " mismatch; both of these fix it")
print("  differently")
for meth in ("ps", "ccs"):
    for oc in OUTS:
        s = q[q["outcome"] == oc]
        a = s[s["arm"] == meth + "_platt"]
        b = s[s["arm"] == meth + "_rank"]
        if len(a) and len(b):
            print("  %-4s %-18s platt %.4f"
                  "   rank %.4f   %+.4f"
                  % (meth, oc,
                     a["mean_all"].iloc[0],
                     b["mean_all"].iloc[0],
                     b["mean_all"].iloc[0]
                     - a["mean_all"].iloc[0]))

print("")
print("BOOTSTRAP: ccs_platt vs ehr_ecg")
z = r[r["arm"] == "ccs_vs_ehrecg"]
if len(z):
    print(z[["outcome", "mean_all", "lo",
             "hi", "sig"]].round(4)
          .to_string(index=False))
    print("")
    print("  significant: %d of %d"
          % (int(z["sig"].fillna(0).sum()),
             len(z)))
    print("  ehr_ecg ignores CTPA entirely, so"
          " a win for ccs_platt is the first")
    print("  evidence in this project that the"
          " CTPA modality adds anything")
print("")
print("saved", DEST, r.shape)
keep_awake(False)