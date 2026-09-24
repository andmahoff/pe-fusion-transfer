"""Missing CTPA on the full cohort, without
imputation.

Requiring a CTPA report limits the three-modality
model to 1,636 admissions, while the EHR + ECG
model uses all 3,501. This script trains on all
3,501 and uses CTPA only where it exists. The
missingness methods of script 94 combined
per-modality predictions; a block forest works on
features instead, and scikit-learn trees (1.3
onwards) send missing values down a learned
default branch.

FIVE VARIANTS
  complete  block forest on the 1,636 with CTPA
  ehr_ecg   block forest on all 3,501, ignoring
            CTPA
  nan       all 3,501, CTPA columns left missing
  nan_ind   the same plus a has_ctpa indicator,
            since whether a CTPA was ordered may
            itself be informative
  masked    trees that would draw CTPA columns
            for a patient without one fall back
            to the other blocks

The key comparison is nan_ind against complete:
the same method and features, differing only in
whether the 1,865 CTPA-free patients are used for
training. Calibration is reported split by CTPA
availability.

Settings from script 117: leaf 1, depth 12, CCA
k 5, sqrt max_features.

Parallel across seeds. Results are saved after
every outcome.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\missing_modality.csv
  results\\missing_modality_log.txt
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
# from script 117 marginals
N_EHR, N_ECG, N_CT = 28, 8, 16
LEAF, DEPTH, K, REG, NTREE = 1, 12, 5, 0.5, 600
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
OUTS = ["death_30d", "death_30d_inhosp",
        "composite_30d", "cv_first"]
DEST = os.path.join(
    PROC, "missing_modality.csv")
BEST_C = {"cv_first": 1e-4,
          "composite_30d": 1e-3,
          "death_30d": 1e-3,
          "death_30d_inhosp": 3e-4}
# script 116, EHR+ECG on the full cohort
S116 = {"death_30d": 0.8938,
        "death_30d_inhosp": 0.9199,
        "composite_30d": 0.8529,
        "cv_first": 0.8065}

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


def prep_keepnan(Xtr, Xte, have_tr, have_te):
    """Standardise using only the rows that have
    the modality, then restore NaN for the rest.

    No imputation anywhere: the missing rows stay
    missing and the tree learns a default branch
    for them."""
    A = np.asarray(Xtr, dtype=float).copy()
    B = np.asarray(Xte, dtype=float).copy()
    sub = A[have_tr]
    if len(sub) < 10:
        return np.full_like(A, np.nan), \
            np.full_like(B, np.nan)
    mu = np.nanmean(sub, axis=0)
    sd = np.nanstd(sub, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9),
                  sd, 1.0)
    A = (A - mu) / sd
    B = (B - mu) / sd
    A[~have_tr] = np.nan
    B[~have_te] = np.nan
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
    """Block-stratified draws with native NaN
    handling.

    scikit-learn 1.3's DecisionTreeClassifier
    sends missing values down a learned default
    branch, so no imputation is needed. This is
    what lets the 1,865 CTPA-free patients train
    the EHR and ECG structure while the CTPA
    columns simply are not available for them.

    mask=True makes a tree that drew CTPA columns
    fall back to the other blocks for patients
    without one, which is expert routing without
    a second model."""

    def __init__(self, n_estimators=600,
                 draws=(28, 8, 16),
                 bounds=(38, 312), leaf=1,
                 depth=12, mask=False,
                 seed=42):
        self.n = n_estimators
        self.draws = draws
        self.bounds = bounds
        self.leaf = leaf
        self.depth = depth
        self.mask = mask
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
        for _ in range(self.n):
            parts = []
            for ix, w in zip(blk, self.draws):
                if len(ix) == 0:
                    continue
                k = int(min(max(1, w), len(ix)))
                parts.append(rng.choice(
                    ix, k, replace=False))
            cols = np.concatenate(parts)
            uses_ct = bool(len(parts) > 2
                           and len(parts[2]))
            rows = rng.choice(n, n,
                              replace=True)
            if (self.mask and uses_ct
                    and have is not None):
                # this tree needs CTPA, so train
                # it only on patients who have it
                rows = rows[have[rows]]
                if len(rows) < 30 or \
                        y[rows].sum() < 3:
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
                # a CTPA tree only votes on
                # patients who have one
                num[have] += q[have]
                den[have] += 1.0
            else:
                num += q
                den += 1.0
        den = np.where(den > 0, den, 1.0)
        out = num / den
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
              have_tr, have_te, use_ct,
              keep_nan, add_ind):
    """EHR and ECG blocks with their canonical
    variates, then CTPA if used.

    The EHR-ECG variates are fitted on ALL rows.
    The CTPA variates are fitted only on rows
    that HAVE a CTPA, then applied to everyone,
    with the result set to NaN where the modality
    is absent."""
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
        sub = have_tr
        if sub.sum() > 50:
            A2 = np.nan_to_num(At[sub])
            C2 = np.nan_to_num(Ct[sub])
            V1, V2 = rcca(A2, C2, REG, K)
            if V1 is not None:
                mu = A2.mean(0)
                mv = C2.mean(0)
                za = (np.nan_to_num(At)
                      - mu) @ V1
                ze = (np.nan_to_num(Ae)
                      - mu) @ V1
                zc = (np.nan_to_num(Ct)
                      - mv) @ V2
                zd = (np.nan_to_num(Ce)
                      - mv) @ V2
                if keep_nan:
                    za[~have_tr] = np.nan
                    ze[~have_te] = np.nan
                    zc[~have_tr] = np.nan
                    zd[~have_te] = np.nan
                pa.append(za)
                qa.append(ze)
                pc.append(zc)
                qc.append(zd)
    if add_ind:
        pa.append(have_tr.astype(float)
                  .reshape(-1, 1))
        qa.append(have_te.astype(float)
                  .reshape(-1, 1))
    Zt = np.column_stack(pa + pb + pc)
    Ze = np.column_stack(qa + qb + qc)
    b0 = sum(x.shape[1] for x in pa)
    b1 = b0 + sum(x.shape[1] for x in pb)
    return Zt, Ze, (b0, b1)


def oof_arm(Xa, Xb, Xc, y, grp, have, seed,
            arm):
    """arm: complete | ehr_ecg | nan | nan_ind
    | masked"""
    if arm == "complete":
        keep = np.where(have)[0]
    else:
        keep = np.arange(len(y))
    yk, gk = y[keep], grp[keep]
    Ak, Bk, Ck = Xa[keep], Xb[keep], Xc[keep]
    hk = have[keep]
    use_ct = arm != "ehr_ecg"
    keep_nan = arm in ("nan", "nan_ind",
                       "masked")
    add_ind = arm == "nan_ind"
    mask = arm == "masked"
    p = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(yk), 1)), yk, gk):
        At, Ae = prep_fold(Ak[tr], Ak[te])
        Bt, Be = prep_fold(Bk[tr], Bk[te])
        if keep_nan:
            Ct, Ce = prep_keepnan(
                Ck[tr], Ck[te], hk[tr],
                hk[te])
        else:
            Ct, Ce = prep_fold(Ck[tr], Ck[te])
        Zt, Ze, bd = build_rep(
            At, Ae, Bt, Be, Ct, Ce, hk[tr],
            hk[te], use_ct, keep_nan, add_ind)
        dr = (N_EHR, N_ECG,
              N_CT if use_ct else 0)
        m = BlockForestNaN(
            n_estimators=NTREE, draws=dr,
            bounds=bd, leaf=LEAF,
            depth=DEPTH, mask=mask,
            seed=seed)
        m.fit(Zt, yk[tr],
              have=hk[tr] if mask else None)
        p[keep[te]] = m.predict_proba(
            Ze, have=hk[te] if mask
            else None)[:, 1]
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
        ok &= sub
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


def calib(p, y, sub=None):
    ok = np.isfinite(p)
    if sub is not None:
        ok &= sub
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

# LEFT join on CTPA, so the 1,865 without one
# are kept
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
print("  with CTPA:    %d (%.1f%%)"
      % (int(D["has_ctpa"].sum()),
         100.0 * D["has_ctpa"].mean()))
print("  without CTPA: %d"
      % int((1 - D["has_ctpa"]).sum()))
print("  EHR %d   ECG %d   CTPA %d"
      % (len(EH), len(ECGC), len(CT)))
print("  script 117 used only the complete"
      " cases and lost on all four outcomes")
print("  script 116 on EHR+ECG used all of"
      " them and won on three")
print("  settings from script 117 marginals:"
      " leaf=%d depth=%s k=%d"
      % (LEAF, DEPTH, K), flush=True)

rows = []
t0 = time.time()
ARMS = ["complete", "ehr_ecg", "nan",
        "nan_ind", "masked"]

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
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d"
          "   with CTPA %d (ev %d)"
          % (oc, len(y), int(y.sum()),
             int(have.sum()),
             int(y[have].sum())))

    # references on the complete cases, so the
    # script 117 comparison is like for like
    pe = _rank(oof_one(Xa, y, grp, 42))
    pg = _rank(oof_one(Xb, y, grp, 42, cb))
    pc_ = oof_one(Xc, y, grp, 42, grid=CT_CS,
                  sub=have)
    a_e_all = auc_on(pe, y)
    a_e_cc = auc_on(pe, y, have)
    L2 = late2(_rank(pe[have]),
               _rank(pc_[have]), y[have])
    ref2 = roc_auc_score(y[have], L2)
    print("  ehr: all %.4f, complete cases"
          " %.4f" % (a_e_all, a_e_cc))
    print("  late ehr+ctpa on complete cases:"
          " %.4f" % ref2)
    print("  script 116 EHR+ECG full cohort:"
          " %.4f" % S116.get(oc, np.nan),
          flush=True)
    rows.append({
        "outcome": oc, "arm": "ehr_only",
        "n_train": len(y), "mean_all": a_e_all,
        "mean_cc": a_e_cc, "sd": np.nan})
    rows.append({
        "outcome": oc, "arm": "late_ehr_ctpa",
        "n_train": int(have.sum()),
        "mean_all": np.nan, "mean_cc": ref2,
        "sd": np.nan})

    print("")
    print("  %-10s %7s %9s %9s %9s"
          % ("arm", "trains", "AUC all",
             "AUC cc", "AUC no-ct"))
    store = {}
    for arm in ARMS:
        try:
            ps = par(lambda s, arm=arm:
                     oof_arm(Xa, Xb, Xc, y,
                             grp, have, s,
                             arm))
        except Exception as exc:
            print("  %-10s FAILED %s"
                  % (arm, repr(exc)[:45]))
            continue
        aa = np.array([auc_on(p, y)
                       for p in ps])
        cc = np.array([auc_on(p, y, have)
                       for p in ps])
        nn = np.array([auc_on(p, y, ~have)
                       for p in ps])
        ntr = (int(have.sum())
               if arm == "complete"
               else len(y))
        store[arm] = ps[0]
        print("  %-10s %7d %9.4f %9.4f %9.4f"
              % (arm, ntr,
                 np.nanmean(aa),
                 np.nanmean(cc),
                 np.nanmean(nn)), flush=True)
        rows.append({
            "outcome": oc, "arm": arm,
            "n_train": ntr,
            "mean_all": np.nanmean(aa),
            "sd": np.nanstd(aa, ddof=1),
            "mean_cc": np.nanmean(cc),
            "mean_noct": np.nanmean(nn),
            "vs_late2": np.nanmean(cc) - ref2})
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)

    # THE comparison: nan_ind vs complete, on
    # the same complete-case patients
    if "nan_ind" in store and "complete" in store:
        a = store["nan_ind"]
        b = store["complete"]
        ok = np.isfinite(a) & np.isfinite(b) \
            & have
        if ok.sum() > 50:
            gn, lo, hi, _ = f.boot_diff(
                y[ok], _rank(a[ok]),
                _rank(b[ok]), grp[ok])
            print("")
            print("  THE COMPARISON: training on"
                  " all %d vs only the %d with"
                  % (len(y), int(have.sum())))
            print("  a CTPA, scored on the same"
                  " complete cases")
            print("    %+.4f [%+.4f,%+.4f] %s"
                  % (gn, lo, hi,
                     "*" if (lo > 0 or hi < 0)
                     else ""), flush=True)
            rows.append({
                "outcome": oc,
                "arm": "nan_ind_vs_complete",
                "n_train": len(y),
                "mean_all": gn, "lo": lo,
                "hi": hi,
                "sig": int(lo > 0 or hi < 0)})

    # calibration split by CTPA availability
    if "nan_ind" in store:
        p = store["nan_ind"]
        b1, s1 = calib(p, y, have)
        b2, s2 = calib(p, y, ~have)
        pcal = platt_oof(p, y, grp, 42)
        b3, s3 = calib(pcal, y, have)
        b4, s4 = calib(pcal, y, ~have)
        print("")
        print("  CALIBRATION BY AVAILABILITY")
        print("    a model that knows what it"
              " is missing should not be")
        print("    over-confident on the"
              " patients lacking a modality")
        print("    with CTPA    slope %.3f ->"
              " %.3f   Brier %.5f -> %.5f"
              % (s1, s3, b1, b3))
        print("    without CTPA slope %.3f ->"
              " %.3f   Brier %.5f -> %.5f"
              % (s2, s4, b2, b4), flush=True)
        rows.append({
            "outcome": oc,
            "arm": "calib_nan_ind",
            "slope_cc": s1, "slope_noct": s2,
            "slope_cc_platt": s3,
            "slope_noct_platt": s4,
            "brier_cc": b1, "brier_noct": b2})

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
print("AUROC ON THE COMPLETE CASES")
print("  the only fair cross-variant comparison,"
      " since 'complete' cannot score the")
print("  patients it never saw")
q = r[r["arm"].isin(ARMS)]
if len(q):
    print(q.pivot_table(index="arm",
                        columns="outcome",
                        values="mean_cc")
          .round(4).to_string())

print("")
print("AUROC ON THE FULL COHORT")
print("  'complete' is absent here by"
      " construction")
if len(q):
    print(q.pivot_table(index="arm",
                        columns="outcome",
                        values="mean_all")
          .round(4).to_string())

print("")
print("AUROC ON THE PATIENTS WITHOUT A CTPA")
if len(q):
    print(q.pivot_table(index="arm",
                        columns="outcome",
                        values="mean_noct")
          .round(4).to_string())

print("")
print("GAIN OVER LATE EHR+CTPA"
      " (complete cases)")
if len(q):
    print(q.pivot_table(index="arm",
                        columns="outcome",
                        values="vs_late2")
          .round(4).to_string())

print("")
print("=" * 74)
print("DOES THE EXTRA TRAINING DATA HELP?")
print("  nan_ind trains on all %d; complete"
      " trains on the subset with a CTPA."
      % len(D))
print("  Both are scored on the same complete"
      " cases, so the difference is purely")
print("  the 1,865 extra patients.")
z = r[r["arm"] == "nan_ind_vs_complete"]
if len(z):
    print(z[["outcome", "mean_all", "lo",
             "hi", "sig"]].round(4)
          .to_string(index=False))
    print("")
    print("  significant: %d of %d"
          % (int(z["sig"].fillna(0).sum()),
             len(z)))

print("")
print("AGAINST SCRIPT 116 (EHR+ECG, full"
      " cohort)")
for oc in OUTS:
    s = q[(q["outcome"] == oc)
          & (q["arm"] == "nan_ind")]
    e = q[(q["outcome"] == oc)
          & (q["arm"] == "ehr_ecg")]
    if len(s) and len(e):
        print("  %-18s ehr_ecg %.4f"
              "   nan_ind %.4f   %+.4f"
              "   (script 116 %.4f)"
              % (oc, e["mean_all"].iloc[0],
                 s["mean_all"].iloc[0],
                 s["mean_all"].iloc[0]
                 - e["mean_all"].iloc[0],
                 S116.get(oc, np.nan)))
print("")
print("  ehr_ecg vs nan_ind isolates whether"
      " CTPA adds anything once the cohort")
print("  is held at full size")

print("")
print("CALIBRATION BY AVAILABILITY")
c = r[r["arm"] == "calib_nan_ind"]
if len(c):
    print(c[["outcome", "slope_cc",
             "slope_noct", "slope_cc_platt",
             "slope_noct_platt"]].round(3)
          .to_string(index=False))
    print("")
    print("  a larger slope on the without-CTPA"
          " rows means the model is more")
    print("  under-confident there")
print("")
print("saved", DEST, r.shape)
keep_awake(False)