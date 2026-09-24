"""EHR + ECG under the tree learners.

No CTPA report is needed, so the cohort is about
3,501 admissions with 397 death events, against
152 on the three-modality cohort. The ECG
modality is the improved one: the 71 SCP logits
plus the v12 waveform measurements, with C fixed
per outcome. Script 88 gives the late-fusion
baseline for this pairing.

Both forests use the same max_features, so the
rotation forest is compared with the random
forest under the same splitting rule; in script
110 the rotation forest's trees evaluated every
feature at each split. Stage 2 sweeps
max_features for the rotation forest, including
the unrestricted setting.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ehr_ecg_learners.csv
  results\\ehr_ecg_learners_log.txt
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as sla
from sklearn.decomposition import PCA
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
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CCA_REG, CCA_K = 0.1, 10
PCA_K = 8
NTREE = 300
MFEAT = 8
MFEAT_GRID = [8, 16, 32, None]
MFEAT_OUT = "death_30d"
ECG_LO, ECG_HI = -12.0, 48.0
MINCOV = 0.20
OUTS = ["death_30d", "composite_30d",
        "cv_first", "death_30d_inhosp"]
DEST = os.path.join(
    PROC, "ehr_ecg_learners.csv")
# script 88, late fusion, improved ECG
S88 = {"death_30d": {"ehr": 0.8308,
                     "gain": 0.0078},
       "composite_30d": {"ehr": 0.8042,
                         "gain": 0.0090},
       "cv_first": {"ehr": 0.7982,
                    "gain": 0.0093},
       "death_30d_inhosp": {"ehr": 0.8321,
                            "gain": 0.0057}}
BEST_C = {"cv_first": 1e-4,
          "composite_30d": 1e-3,
          "death_30d": 1e-3,
          "death_30d_inhosp": 3e-4}

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


class RotationForest:
    """Partition form, subset 3, matching the
    default script 110 actually used.

    max_features MATTERS, and setting it is the
    fairness fix described at the top. mfeat None
    reproduces the old unrestricted behaviour,
    which stage 2 tests against."""

    def __init__(self, n_estimators=300,
                 size=3, boot=0.75, leaf=5,
                 depth=12, mfeat=8, seed=42):
        self.n = n_estimators
        self.size = size
        self.boot = boot
        self.leaf = leaf
        self.depth = depth
        self.mfeat = mfeat
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        nb = max(10, int(round(self.boot * n)))
        mf = (None if self.mfeat is None
              else int(min(max(1, self.mfeat),
                           p)))
        self.bl_, self.tr_ = [], []
        for _ in range(self.n):
            cols = rng.permutation(p)
            bl = []
            for i in range(0, p, self.size):
                idx = cols[i:i + self.size]
                if len(idx) < 2:
                    continue
                rows = rng.choice(n, nb,
                                  replace=True)
                bl.append((idx, _pca_block(
                    X[np.ix_(rows, idx)])))
            Z = X.copy()
            for idx, C in bl:
                Z[:, idx] = X[:, idx] @ C
            rows = rng.choice(n, n,
                              replace=True)
            t = DecisionTreeClassifier(
                max_depth=self.depth,
                min_samples_leaf=self.leaf,
                max_features=mf,
                class_weight="balanced",
                random_state=int(
                    rng.integers(1e6)))
            t.fit(Z[rows], y[rows])
            self.bl_.append(bl)
            self.tr_.append(t)
        return self

    def predict_proba(self, X):
        out = np.zeros(len(X))
        for bl, t in zip(self.bl_, self.tr_):
            Z = X.copy()
            for idx, C in bl:
                Z[:, idx] = X[:, idx] @ C
            out += t.predict_proba(Z)[:, 1]
        out /= len(self.tr_)
        return np.column_stack([1 - out, out])


# ---------------- learners ------------------
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


def learn_rf(Xt, yt, Xe, gt=None):
    m = RandomForestClassifier(
        n_estimators=NTREE,
        max_features=MFEAT, max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42, n_jobs=-1)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def learn_rot(Xt, yt, Xe, gt=None,
              mfeat=MFEAT):
    m = RotationForest(n_estimators=NTREE,
                       size=3, leaf=5,
                       depth=12, mfeat=mfeat,
                       seed=42)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


# ---------------- representations -----------
def rep_raw(At, Ae, Bt, Be, yt):
    return (np.column_stack([At, Bt]),
            np.column_stack([Ae, Be]))


def rep_ccaaug(At, Ae, Bt, Be, yt):
    A_, B_ = rcca(At, Bt, CCA_REG, CCA_K)
    if A_ is None:
        return rep_raw(At, Ae, Bt, Be, yt)
    ma, mb = At.mean(0), Bt.mean(0)
    return (np.column_stack([
        At, Bt, (At - ma) @ A_,
        (Bt - mb) @ B_]),
        np.column_stack([
            Ae, Be, (Ae - ma) @ A_,
            (Be - mb) @ B_]))


def rep_pca(At, Ae, Bt, Be, yt):
    Xt = np.column_stack([At, Bt])
    Xe = np.column_stack([Ae, Be])
    k = int(min(PCA_K, Xt.shape[1] - 1,
                len(yt) - 1))
    pc = PCA(n_components=k, random_state=42)
    return (pc.fit_transform(Xt),
            pc.transform(Xe))


def oof(Xa, Xb, y, grp, repfn, learner, seed,
        **kw):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        At, Ae = prep_fold(Xa[tr], Xa[te])
        Bt, Be = prep_fold(Xb[tr], Xb[te])
        Zt, Ze = repfn(At, Ae, Bt, Be, y[tr])
        p[te] = learner(Zt, y[tr], Ze,
                        grp[tr], **kw)
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
print("  EHR %d   ECG %d   total %d"
      % (len(EH), len(ECGC),
         len(EH) + len(ECGC)))
print("  no CTPA requirement, so this cohort"
      " is about 2.6x the three-modality one")
print("  the ECG modality is the improved one:"
      " 71 logits + v12 measurements")
print("  both forests use max_features=%d, so"
      " the families are comparable" % MFEAT,
      flush=True)

# ---------- TIMING PROBE ----------
print("")
print("TIMING PROBE")
npro = min(len(D), 3500)
Xp = np.random.rand(npro,
                    len(EH) + len(ECGC) + 20)
yp = (np.random.rand(npro) > 0.89).astype(int)
t = time.time()
learn_rf(Xp, yp, Xp[:100])
t_rf = time.time() - t
t = time.time()
RotationForest(n_estimators=50, size=3,
               leaf=5, depth=12,
               mfeat=MFEAT,
               seed=42).fit(Xp, yp)
t_rot = (time.time() - t) * (NTREE / 50.0)
t = time.time()
RotationForest(n_estimators=20, size=3,
               leaf=5, depth=12, mfeat=None,
               seed=42).fit(Xp, yp)
t_rot_n = (time.time() - t) * (NTREE / 20.0)
print("  rf              %d trees: %7.1f s"
      % (NTREE, t_rf))
print("  rot mfeat=%-4d  %d trees: %7.1f s"
      % (MFEAT, NTREE, t_rot))
print("  rot mfeat=None  %d trees: %7.1f s"
      "  (unrestricted)"
      % (NTREE, t_rot_n))
print("  speedup from setting max_features:"
      " %.0fx" % (t_rot_n / max(t_rot, 1e-9)))
per_seed = NFOLD * (2 * t_rf + t_rot)
est1 = (per_seed * len(SEEDS)
        * len(OUTS) / 60.0)
est2 = (NFOLD * len(SEEDS)
        * (2 * t_rot + t_rot_n
           + t_rot * 1.3) / 60.0)
print("")
print("  stage 1 (5 methods x 4 outcomes):"
      " %.0f min" % est1)
print("  stage 2 (mfeat sweep, 1 outcome):"
      " %.0f min" % est2)
print("  PROJECTED TOTAL: %.0f minutes"
      " (%.1f hours)"
      % (est1 + est2, (est1 + est2) / 60))
print("  Ctrl+C now if too long. Results save"
      " after each outcome.")
time.sleep(5)

rows = []
t0 = time.time()

# ================ STAGE 1 ===================
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
    s8 = S88.get(oc, {})
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())))

    pa = _rank(oof_one(Xa, y, grp, 42))
    pb = _rank(oof_one(Xb, y, grp, 42, cb))
    a_e = roc_auc_score(y, pa)
    a_g = roc_auc_score(y, pb)
    ref = roc_auc_score(y, late_from(pa, pb, y))
    gap = abs(a_e - a_g)
    print("  ehr %.4f   ecg %.4f   gap %.4f"
          % (a_e, a_g, gap))
    print("  late %.4f   the gap rule predicts"
          " %+.4f" % (ref, 0.036 - 0.243 * gap))
    if s8:
        print("  script 88: ehr %.4f, late"
              " fusion gained %+.4f"
              % (s8["ehr"], s8["gain"]),
              flush=True)
    for nm, v in (("ehr", a_e), ("ecg", a_g),
                  ("late_wsrc", ref)):
        rows.append({
            "outcome": oc, "stage": 1,
            "method": nm, "mfeat": np.nan,
            "n": len(y), "ev": int(y.sum()),
            "mean": v, "sd": np.nan,
            "vs_late": v - ref,
            "vs_ehr": v - a_e})

    print("")
    print("  %-16s %8s %8s %9s %9s"
          % ("method", "AUC", "SD",
             "vs late", "vs ehr"))
    for nm, repfn, learner in (
            ("early/l2", rep_raw, learn_l2),
            ("pca8/l2", rep_pca, learn_l2),
            ("raw/rf", rep_raw, learn_rf),
            ("cca_aug/rf", rep_ccaaug,
             learn_rf),
            ("cca_aug/rot", rep_ccaaug,
             learn_rot)):
        try:
            aa = np.array([
                roc_auc_score(
                    y, oof(Xa, Xb, y, grp,
                           repfn, learner, s))
                for s in SEEDS])
        except Exception as exc:
            print("  %-16s FAILED %s"
                  % (nm, repr(exc)[:40]))
            continue
        print("  %-16s %8.4f %8.4f %+9.4f"
              " %+9.4f"
              % (nm, aa.mean(),
                 aa.std(ddof=1),
                 aa.mean() - ref,
                 aa.mean() - a_e), flush=True)
        rows.append({
            "outcome": oc, "stage": 1,
            "method": nm, "mfeat": np.nan,
            "n": len(y), "ev": int(y.sum()),
            "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_late": aa.mean() - ref,
            "vs_ehr": aa.mean() - a_e})
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)

    try:
        p = oof(Xa, Xb, y, grp, rep_ccaaug,
                learn_rf, 42)
        b0, s0 = calib(p, y)
        pc = platt_oof(p, y, grp, 42)
        ok = np.isfinite(pc)
        b1, s1 = calib(pc[ok], y[ok])
        print("")
        print("  cca_aug/rf calibration:"
              " slope %.3f -> %.3f after Platt"
              "   Brier %.5f -> %.5f"
              % (s0, s1, b0, b1), flush=True)
    except Exception:
        pass

    pd.DataFrame(rows).to_csv(DEST,
                              index=False)
    print("")
    print("  %s done in %.1f min  (saved)"
          % (oc, (time.time() - tb) / 60),
          flush=True)

# ================ STAGE 2 ===================
print("")
print("#" * 74)
print("STAGE 2: DOES max_features HELP OR HURT"
      " THE ROTATION FOREST?")
print("  script 110's random forest gave a"
      " MONOTONE mtry marginal, so more")
print("  restriction helped there. But a"
      " rotation forest already gets its")
print("  diversity from the rotations, so"
      " subsampling features too could")
print("  over-randomise. mfeat=None is the"
      " old unrestricted behaviour.")
print("#" * 74, flush=True)

oc = MFEAT_OUT
if oc in D.columns:
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    Xa = d[EH].values.astype(float)
    Xb = d[ECGC].values.astype(float)
    z = [x for x in rows
         if x["outcome"] == oc
         and x["method"] == "late_wsrc"]
    ref = z[0]["mean"] if z else np.nan
    tb = time.time()
    print("")
    print("  %s   late %.4f" % (oc, ref))
    print("  %-10s %8s %8s %9s"
          % ("mfeat", "AUC", "SD", "vs late"))
    for mf in MFEAT_GRID:
        try:
            aa = np.array([
                roc_auc_score(
                    y, oof(Xa, Xb, y, grp,
                           rep_ccaaug,
                           learn_rot, s,
                           mfeat=mf))
                for s in SEEDS])
        except Exception as exc:
            print("  %-10s FAILED %s"
                  % (str(mf), repr(exc)[:40]))
            continue
        tag = ("  (unrestricted)"
               if mf is None else "")
        print("  %-10s %8.4f %8.4f %+9.4f%s"
              % (str(mf), aa.mean(),
                 aa.std(ddof=1),
                 aa.mean() - ref, tag),
              flush=True)
        rows.append({
            "outcome": oc, "stage": 2,
            "method": "cca_aug/rot",
            "mfeat": (-1 if mf is None
                      else mf),
            "n": len(y), "ev": int(y.sum()),
            "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_late": aa.mean() - ref,
            "vs_ehr": np.nan})
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)
    print("")
    print("  stage 2: %.1f min"
          % ((time.time() - tb) / 60),
          flush=True)

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
s1 = r[r["stage"] == 1]

print("")
print("=" * 74)
print("AUROC BY METHOD")
print(s1.pivot_table(index="method",
                     columns="outcome",
                     values="mean")
      .round(4).to_string())

print("")
print("GAIN OVER late:wsrc")
q = s1[~s1["method"].isin(
    ["ehr", "ecg", "late_wsrc"])]
if len(q):
    print(q.pivot_table(index="method",
                        columns="outcome",
                        values="vs_late")
          .round(4).to_string())

print("")
print("GAIN OVER EHR ALONE")
print("  script 88's late fusion with the"
      " improved ECG gained +0.0093, +0.0090,")
print("  +0.0078 and +0.0057. Anything beating"
      " those is a better use of the ECG")
print("  modality than weighted rank"
      " averaging.")
q2 = s1[s1["method"] != "ehr"]
if len(q2):
    print(q2.pivot_table(index="method",
                         columns="outcome",
                         values="vs_ehr")
          .round(4).to_string())

print("")
print("AVERAGE RANK ACROSS OUTCOMES")
rk = s1[s1["method"] != "ehr"].pivot_table(
    index="method", columns="outcome",
    values="mean").rank(
    ascending=False).mean(axis=1)
print(rk.sort_values().round(2).to_string())

print("")
print("=" * 74)
print("STAGE 2: max_features ON THE ROTATION"
      " FOREST")
s2 = r[r["stage"] == 2]
if len(s2):
    print(s2[["mfeat", "mean", "sd",
              "vs_late"]].round(4)
          .to_string(index=False))
    b = s2.loc[s2["mean"].idxmax()]
    n_ = s2[s2["mfeat"] == -1]
    print("")
    print("  best mfeat %s at %.4f"
          % ("None" if b["mfeat"] == -1
             else int(b["mfeat"]), b["mean"]))
    if len(n_):
        d_ = b["mean"] - n_["mean"].iloc[0]
        print("  unrestricted %.4f, so %+.4f"
              % (n_["mean"].iloc[0], d_))
        if b["mfeat"] == -1:
            print("  -> the unrestricted version"
                  " is better, so script 110's")
            print("     rotation-forest"
                  " advantage was partly the")
            print("     splitting regime rather"
                  " than the rotations.")
        else:
            print("  -> restricting features"
                  " helps or ties, so the fair")
            print("     comparison stands and"
                  " the rotations are doing")
            print("     the work.")

print("")
print("=" * 74)
print("DOES THE ECG MODALITY EARN ITS PLACE?")
for oc in OUTS:
    s = s1[s1["outcome"] == oc]
    e = s[s["method"] == "ehr"]
    b = s[~s["method"].isin(["ehr", "ecg"])]
    if not len(e) or not len(b):
        continue
    x = b.loc[b["mean"].idxmax()]
    s8 = S88.get(oc, {})
    print("  %-18s ehr %.4f -> %-14s %.4f"
          "   %+.4f   (script 88 late %+.4f)"
          % (oc, e["mean"].iloc[0],
             x["method"], x["mean"],
             x["vs_ehr"], s8.get("gain",
                                 np.nan)))

print("")
print("saved", DEST, r.shape)
keep_awake(False)