"""Full tuning of the EHR + ECG block forest.

max_features (script 114) and the block draw
(script 115) were tuned on this pairing. The other
settings came from EHR+CTPA, which has 94 columns
rather than 312 and fewer events, so they are
tuned here. Rotation is left out, since it made
no consistent difference in scripts 114 and 115.

STAGES, each starting from the previous winner
  A  n_ehr x n_ecg, with n_ecg extended to 24
     and 32
  B  leaf x depth
  C  CCA rank and regularisation
  D  tree count
  E  the combined setting, ten seeds, with
     calibration
  F  one compromise setting across all four
     outcomes, since per-outcome optima chosen
     from a grid are partly noise

The ECG modality is 71 SCP logits plus the v12
measurements, strict Tp-e only, with C fixed per
outcome.

Run in venv (analysis). Parallel across seeds.
Overnight.

OUTPUT FILES
  data\\processed\\ehr_ecg_tune.csv
  results\\ehr_ecg_tune_log.txt
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
NTREE = 300
N_EHR = [14, 20, 28, 38]
N_ECG = [8, 16, 24, 32]
LEAF = [1, 2, 5, 10, 25]
DEPTH = [6, 12, None]
CCA_KS = [5, 10, 20, 30]
CCA_REGS = [0.01, 0.1, 0.5]
NTREES = [150, 300, 600]
ECG_LO, ECG_HI = -12.0, 48.0
MINCOV = 0.20
OUTS = ["death_30d", "death_30d_inhosp",
        "composite_30d", "cv_first"]
DEST = os.path.join(PROC,
                    "ehr_ecg_tune.csv")
BEST_C = {"cv_first": 1e-4,
          "composite_30d": 1e-3,
          "death_30d": 1e-3,
          "death_30d_inhosp": 3e-4}
# script 115 where available, else script 114
S115 = {"death_30d":
        {"late": 0.8700, "best": 0.8932},
        "death_30d_inhosp":
        {"late": 0.8719, "best": 0.9123},
        "composite_30d":
        {"late": 0.8421, "best": 0.8496},
        "cv_first":
        {"late": 0.8221, "best": 0.8027}}

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


class BlockForest:
    """Each tree is fitted on a column subset
    drawn separately from each modality: n_a from
    the EHR-side block, n_b from the ECG-side
    block.

    sklearn's max_features draws from the whole
    pool, so with 38 EHR-side and 274 ECG-side
    columns a draw of 8 gives about 1 EHR feature
    while the EHR modality carries 0.86 of the
    signal. Drawing per block guarantees the
    strong modality is always a candidate.

    Rotation has been removed: it failed to
    distinguish itself four times once the
    sampling regime was equalised."""

    def __init__(self, n_estimators=300,
                 n_a=20, n_b=8, p_a=38,
                 leaf=5, depth=12, seed=42):
        self.n = n_estimators
        self.n_a = n_a
        self.n_b = n_b
        self.p_a = p_a
        self.leaf = leaf
        self.depth = depth
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        ia = np.arange(min(self.p_a, p))
        ib = np.arange(min(self.p_a, p), p)
        na = int(min(self.n_a, len(ia)))
        nb = int(min(self.n_b, len(ib)))
        self.sel_, self.tr_ = [], []
        for _ in range(self.n):
            cols = np.concatenate([
                rng.choice(ia, na,
                           replace=False),
                rng.choice(ib, nb,
                           replace=False)])
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


def rep_ccaaug(At, Ae, Bt, Be, k, reg):
    """EHR block first, then its canonical
    variates, then ECG and its variates, so the
    block boundary is n_ehr + k."""
    A_, B_ = rcca(At, Bt, reg, k)
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


def oof_block(Xa, Xb, y, grp, seed, na, nb,
              leaf=5, depth=12, k=10,
              reg=0.1, ntree=NTREE):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        At, Ae = prep_fold(Xa[tr], Xa[te])
        Bt, Be = prep_fold(Xb[tr], Xb[te])
        Zt, Ze, pa = rep_ccaaug(At, Ae, Bt,
                                Be, k, reg)
        m = BlockForest(
            n_estimators=ntree, n_a=na,
            n_b=nb, p_a=pa, leaf=leaf,
            depth=depth, seed=seed)
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


def par_auc(fn, y, seeds=SEEDS):
    """Across seeds IN PARALLEL. Each seed's
    computation is unchanged and the RNG is
    seeded per configuration, so results match
    the serial version to the last digit."""
    ps = Parallel(n_jobs=NJOBS,
                  backend="loky")(
        delayed(fn)(s) for s in seeds)
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
print("")
print("cohort:", len(D))
print("  EHR %d   ECG %d  (%d logits + %d v12"
      " measurements, strict Tp-e only)"
      % (len(EH), len(ECGC), len(LG), len(MC)))
print("  rotation dropped: failed four times"
      " once sampling was equalised")
print("  parallel over up to %d seeds on %d"
      " cores" % (NJOBS, os.cpu_count() or 0),
      flush=True)

# ---------- TIMING PROBE ----------
print("")
print("TIMING PROBE")
npro = min(len(D), 3500)
Xp = np.random.rand(npro,
                    len(EH) + len(ECGC) + 20)
yp = (np.random.rand(npro) > 0.89).astype(int)
t = time.time()
BlockForest(n_estimators=NTREE, n_a=20, n_b=8,
            p_a=38, seed=42).fit(Xp, yp)
t1 = time.time() - t
t = time.time()
BlockForest(n_estimators=NTREE, n_a=38,
            n_b=32, p_a=38, seed=42).fit(Xp,
                                          yp)
t2 = time.time() - t
per = 0.5 * (t1 + t2)
nA = len(N_EHR) * len(N_ECG)
nB = len(LEAF) * len(DEPTH)
nC = len(CCA_KS) + len(CCA_REGS)
nD = len(NTREES)
cells = (nA + nB + nC + nD) * len(OUTS)
ser = cells * per * NFOLD * len(SEEDS) / 60.0
fin = (len(SEEDS10) * NFOLD * per
       * len(OUTS) * 2 / 60.0)
print("  small config %d trees: %5.1f s"
      % (NTREE, t1))
print("  large config %d trees: %5.1f s"
      % (NTREE, t2))
print("")
print("  %d grid cells; serial %.0f min,"
      " parallel about %.0f min"
      % (cells, ser + fin,
         (ser + fin) / NJOBS))
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
    cb = BEST_C.get(oc, 1e-3)
    pv = S115.get(oc, {})
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())))

    pe = _rank(oof_one(Xa, y, grp, 42))
    pg = _rank(oof_one(Xb, y, grp, 42, cb))
    a_e = roc_auc_score(y, pe)
    ref = roc_auc_score(y, late_from(pe, pg, y))
    print("  ehr %.4f   late %.4f"
          "   previous best %.4f"
          % (a_e, ref, pv.get("best", np.nan)),
          flush=True)
    rows.append({
        "outcome": oc, "stage": "ref",
        "method": "late_wsrc", "mean": ref,
        "sd": np.nan, "vs_late": 0.0})

    # ---- STAGE A: n_ehr x n_ecg ----
    print("")
    print("  STAGE A: n_ehr x n_ecg,"
          " ECG extended to 24 and 32")
    print("  the ecg=8 vs 16 gap narrows as"
          " n_ehr rises and crosses over at")
    print("  n_ehr=28, so this tests whether"
          " that continues")
    print("  %-7s" % "n_ehr", end="")
    for nb in N_ECG:
        print(" %9s" % ("ecg=%d" % nb),
              end="")
    print("")
    bA = None
    for na in N_EHR:
        line = "  %-7d" % na
        for nb in N_ECG:
            try:
                aa = par_auc(
                    lambda s, na=na, nb=nb:
                    oof_block(Xa, Xb, y, grp,
                              s, na, nb), y)
            except Exception:
                line += " %9s" % "fail"
                continue
            line += " %+9.4f" % (aa.mean()
                                 - ref)
            rows.append({
                "outcome": oc, "stage": "A",
                "method": "block",
                "na": na, "nb": nb,
                "mean": aa.mean(),
                "sd": aa.std(ddof=1),
                "vs_late": aa.mean() - ref})
            if bA is None or aa.mean() > bA[0]:
                bA = (aa.mean(), na, nb)
        print(line, flush=True)
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)
    na_, nb_ = (bA[1], bA[2]) if bA else (20, 8)
    print("  best %d+%d  %.4f" % (na_, nb_,
                                  bA[0] if bA
                                  else np.nan),
          flush=True)

    # ---- STAGE B: leaf x depth ----
    print("")
    print("  STAGE B: leaf x depth at %d+%d"
          % (na_, nb_))
    print("  leaf=5 depth=12 were inherited"
          " from script 111 on 94 columns;")
    print("  this cohort has 312 and 2.6x the"
          " events")
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
                    oof_block(Xa, Xb, y, grp,
                              s, na_, nb_,
                              lf, dp), y)
            except Exception:
                line += " %9s" % "fail"
                continue
            line += " %+9.4f" % (aa.mean()
                                 - ref)
            rows.append({
                "outcome": oc, "stage": "B",
                "method": "block", "na": na_,
                "nb": nb_, "leaf": lf,
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
    lf_, dp_ = (bB[1], bB[2]) if bB else (5, 12)
    print("  best leaf=%d depth=%s  %.4f"
          % (lf_, dp_,
             bB[0] if bB else np.nan),
          flush=True)

    # ---- STAGE C: CCA rank, then reg ----
    print("")
    print("  STAGE C: CCA rank and"
          " regularisation")
    print("  k=10 was chosen on EHR+CTPA where"
          " the second block had 46 features;")
    print("  here it has 264, so more"
          " directions may be warranted")
    print("    %-8s %9s %9s" % ("k", "AUC",
                                "vs late"))
    bk, bkm = 10, -1.0
    for k in CCA_KS:
        try:
            aa = par_auc(
                lambda s, k=k:
                oof_block(Xa, Xb, y, grp, s,
                          na_, nb_, lf_, dp_,
                          k), y)
        except Exception:
            continue
        print("    %-8d %9.4f %+9.4f"
              % (k, aa.mean(),
                 aa.mean() - ref), flush=True)
        rows.append({
            "outcome": oc, "stage": "C",
            "method": "cca_k", "na": na_,
            "nb": nb_, "leaf": lf_,
            "depth": (-1 if dp_ is None
                      else dp_), "k": k,
            "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_late": aa.mean() - ref})
        if aa.mean() > bkm:
            bkm, bk = aa.mean(), k
    print("    %-8s %9s %9s" % ("reg", "AUC",
                                "vs late"))
    br, brm = 0.1, -1.0
    for rg in CCA_REGS:
        try:
            aa = par_auc(
                lambda s, rg=rg:
                oof_block(Xa, Xb, y, grp, s,
                          na_, nb_, lf_, dp_,
                          bk, rg), y)
        except Exception:
            continue
        print("    %-8.2f %9.4f %+9.4f"
              % (rg, aa.mean(),
                 aa.mean() - ref), flush=True)
        rows.append({
            "outcome": oc, "stage": "C",
            "method": "cca_reg", "na": na_,
            "nb": nb_, "leaf": lf_,
            "depth": (-1 if dp_ is None
                      else dp_), "k": bk,
            "reg": rg, "mean": aa.mean(),
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
    bt, btm = NTREE, -1.0
    print("    %-8s %9s %9s" % ("ntree",
                                "AUC",
                                "vs late"))
    for nt in NTREES:
        try:
            aa = par_auc(
                lambda s, nt=nt:
                oof_block(Xa, Xb, y, grp, s,
                          na_, nb_, lf_, dp_,
                          bk, br, nt), y)
        except Exception:
            continue
        print("    %-8d %9.4f %+9.4f"
              % (nt, aa.mean(),
                 aa.mean() - ref), flush=True)
        rows.append({
            "outcome": oc, "stage": "D",
            "method": "ntree", "na": na_,
            "nb": nb_, "leaf": lf_,
            "depth": (-1 if dp_ is None
                      else dp_), "k": bk,
            "reg": br, "ntree": nt,
            "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_late": aa.mean() - ref})
        if aa.mean() > btm:
            btm, bt = aa.mean(), nt
    print("  best ntree=%d" % bt, flush=True)

    # ---- STAGE E: final, ten seeds ----
    print("")
    print("  STAGE E: FINAL, ten seeds"
          "  %d+%d leaf=%d depth=%s k=%d"
          " reg=%.2f ntree=%d"
          % (na_, nb_, lf_, dp_, bk, br, bt))
    aa = par_auc(
        lambda s: oof_block(
            Xa, Xb, y, grp, s, na_, nb_, lf_,
            dp_, bk, br, bt), y,
        seeds=SEEDS10)
    p = oof_block(Xa, Xb, y, grp, 42, na_,
                  nb_, lf_, dp_, bk, br, bt)
    b0, s0 = calib(p, y)
    pc = platt_oof(p, y, grp, 42)
    ok = np.isfinite(pc)
    b1, s1 = calib(pc[ok], y[ok])
    print("    %.4f (SD %.4f, range"
          " %.4f-%.4f)   %+.4f vs late"
          % (aa.mean(), aa.std(ddof=1),
             aa.min(), aa.max(),
             aa.mean() - ref))
    print("    slope %.3f -> %.3f after Platt"
          "   Brier %.5f -> %.5f"
          % (s0, s1, b0, b1))
    print("    previous best %.4f, so %+.4f"
          % (pv.get("best", np.nan),
             aa.mean() - pv.get("best",
                                np.nan)),
          flush=True)
    BEST[oc] = (na_, nb_, lf_, dp_, bk, br,
                bt, aa.mean(), ref)
    rows.append({
        "outcome": oc, "stage": "E",
        "method": "final", "na": na_,
        "nb": nb_, "leaf": lf_,
        "depth": (-1 if dp_ is None else dp_),
        "k": bk, "reg": br, "ntree": bt,
        "mean": aa.mean(),
        "sd": aa.std(ddof=1), "lo": aa.min(),
        "hi": aa.max(), "slope": s0,
        "slope_platt": s1, "brier": b0,
        "brier_platt": b1,
        "vs_late": aa.mean() - ref})

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
print("  four per-outcome optima selected from"
      " a grid are partly noise. If one")
print("  setting loses little, report it"
      " instead.")
print("#" * 74, flush=True)

m = pd.DataFrame(rows)
gA = m[m["stage"] == "A"]
cm_na = int(gA.groupby("na")["vs_late"]
            .mean().idxmax())
cm_nb = int(gA.groupby("nb")["vs_late"]
            .mean().idxmax())
gB = m[m["stage"] == "B"]
cm_lf = int(gB.groupby("leaf")["vs_late"]
            .mean().idxmax())
cm_dp_ = gB.groupby("depth")["vs_late"] \
    .mean().idxmax()
cm_dp = None if cm_dp_ == -1 else int(cm_dp_)
gC = m[m["method"] == "cca_k"]
cm_k = int(gC.groupby("k")["vs_late"]
           .mean().idxmax()) if len(gC) else 10
gR = m[m["method"] == "cca_reg"]
cm_rg = float(gR.groupby("reg")["vs_late"]
              .mean().idxmax()) if len(gR) \
    else 0.1
gD = m[m["stage"] == "D"]
cm_nt = int(gD.groupby("ntree")["vs_late"]
            .mean().idxmax()) if len(gD) \
    else NTREE
print("")
print("  compromise: %d+%d leaf=%d depth=%s"
      " k=%d reg=%.2f ntree=%d"
      % (cm_na, cm_nb, cm_lf, cm_dp, cm_k,
         cm_rg, cm_nt), flush=True)

for oc in OUTS:
    if oc not in D.columns or oc not in BEST:
        continue
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    Xa = d[EH].values.astype(float)
    Xb = d[ECGC].values.astype(float)
    na_, nb_, lf_, dp_, bk, br, bt, po, ref \
        = BEST[oc]
    aa = par_auc(
        lambda s: oof_block(
            Xa, Xb, y, grp, s, cm_na, cm_nb,
            cm_lf, cm_dp, cm_k, cm_rg, cm_nt),
        y, seeds=SEEDS10)
    print("  %-18s compromise %.4f"
          "   per-outcome %.4f   %+.4f"
          "   late %.4f"
          % (oc, aa.mean(), po,
             aa.mean() - po, ref), flush=True)
    rows.append({
        "outcome": oc, "stage": "F",
        "method": "compromise", "na": cm_na,
        "nb": cm_nb, "leaf": cm_lf,
        "depth": (-1 if cm_dp is None
                  else cm_dp), "k": cm_k,
        "reg": cm_rg, "ntree": cm_nt,
        "mean": aa.mean(),
        "sd": aa.std(ddof=1),
        "vs_late": aa.mean() - ref,
        "vs_per_outcome": aa.mean() - po})

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("STAGE A: n_ehr x n_ecg")
a = r[r["stage"] == "A"]
for oc in OUTS:
    z = a[a["outcome"] == oc]
    if not len(z):
        continue
    print("")
    print("  " + oc)
    print(z.pivot_table(index="na",
                        columns="nb",
                        values="vs_late")
          .round(4).to_string())
if len(a):
    print("")
    print("  marginal by n_ehr")
    print(a.groupby("na")["vs_late"].mean()
          .round(4).to_string())
    print("  marginal by n_ecg")
    print(a.groupby("nb")["vs_late"].mean()
          .round(4).to_string())

print("")
print("STAGE B: leaf x depth")
b = r[r["stage"] == "B"]
for oc in OUTS:
    z = b[b["outcome"] == oc]
    if not len(z):
        continue
    print("")
    print("  " + oc)
    print(z.pivot_table(index="leaf",
                        columns="depth",
                        values="vs_late")
          .round(4).to_string())
if len(b):
    print("")
    print("  marginal by leaf")
    print(b.groupby("leaf")["vs_late"].mean()
          .round(4).to_string())
    print("  marginal by depth"
          "  (-1 is unrestricted)")
    print(b.groupby("depth")["vs_late"].mean()
          .round(4).to_string())

print("")
print("STAGE C: CCA rank and regularisation")
for meth, ix in (("cca_k", "k"),
                 ("cca_reg", "reg")):
    z = r[r["method"] == meth]
    if not len(z):
        continue
    print("")
    print("  " + meth)
    print(z.pivot_table(index=ix,
                        columns="outcome",
                        values="vs_late")
          .round(4).to_string())
    print("  marginal")
    print(z.groupby(ix)["vs_late"].mean()
          .round(4).to_string())

print("")
print("STAGE D: tree count")
z = r[r["stage"] == "D"]
if len(z):
    print(z.pivot_table(index="ntree",
                        columns="outcome",
                        values="vs_late")
          .round(4).to_string())
    print("  marginal")
    print(z.groupby("ntree")["vs_late"].mean()
          .round(4).to_string())

print("")
print("=" * 74)
print("FINAL, TEN SEEDS")
e = r[r["stage"] == "E"]
if len(e):
    print(e[["outcome", "na", "nb", "leaf",
             "depth", "k", "reg", "ntree",
             "mean", "sd", "vs_late",
             "slope_platt"]].round(4)
          .to_string(index=False))
    print("")
    print("  AGAINST THE PREVIOUS BEST")
    for oc in OUTS:
        z = e[e["outcome"] == oc]
        pv = S115.get(oc, {})
        if len(z) and pv:
            print("    %-18s late %.4f"
                  "   before %.4f   now %.4f"
                  "   %+.4f"
                  % (oc, pv["late"],
                     pv["best"],
                     z["mean"].iloc[0],
                     z["mean"].iloc[0]
                     - pv["best"]))

print("")
print("COMPROMISE vs PER-OUTCOME")
c = r[r["stage"] == "F"]
if len(c):
    print(c[["outcome", "mean", "sd",
             "vs_late",
             "vs_per_outcome"]].round(4)
          .to_string(index=False))
    md = c["vs_per_outcome"].abs().mean()
    print("")
    print("  mean absolute loss from one"
          " setting: %.4f" % md)
    if md < 0.005:
        print("  -> the per-outcome"
              " differences were noise."
              " Report the compromise.")
    else:
        print("  -> the differences are real,"
              " which is itself a finding.")

print("")
print("WINS OVER late:wsrc, FINAL MODELS")
for stg, nm in (("E", "per-outcome"),
                ("F", "compromise")):
    z = r[r["stage"] == stg]
    if len(z):
        print("  %-12s %d of %d outcomes"
              % (nm, int((z["vs_late"] > 0)
                         .sum()), len(z)))
print("")
print("saved", DEST, r.shape)
keep_awake(False)