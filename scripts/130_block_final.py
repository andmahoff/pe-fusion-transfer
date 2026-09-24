"""Remaining settings of the three-modality block
forest, including rotation, in one staged run.

In three_mod_block.csv, rotation beat the plain
forest on all four outcomes by 0.0013 to 0.0039,
each within one seed SD, with rotation fixed at
size=3 and boot=0.75.

ROTATION PARAMETERS
  size      PCA subset size
  boot      bootstrap fraction for each PCA fit
  rot_frac  fraction of trees rotated
  within    rotate within each modality block
            rather than across the drawn columns,
            which mixes the modalities in one PCA

OTHER SETTINGS
  A  class weighting; every block forest so far
     used balanced class weights and came out
     under-confident (slopes 1.3 to 2.6)
  B  ccp_alpha pruning, as an alternative to a
     fixed leaf size
  C  row bootstrap fraction
  D  OOB-weighted voting
  E  draws crossed with leaf
  F  cv_first leaf up to 300
  G  CCA reg up to 2.0
  H  tree count up to 2400

The final stage runs ten seeds and a paired
bootstrap of rotation against no rotation.

Parallel across seeds. Timing probe first.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\block_final.csv
  results\\block_final_log.txt
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
K0, REG0, NTREE0 = 10, 0.5, 600
DRAWS = [(14, 8, 16), (20, 8, 16),
         (14, 4, 8), (20, 12, 24),
         (28, 8, 16)]
LEAF_MAIN = [5, 10, 25, 50]
LEAF_CV = [50, 100, 150, 200, 300]
CW = ["balanced", None]
CCP = [0.0, 1e-4, 1e-3, 1e-2]
BOOT = [0.6, 0.8, 1.0]
REGS = [0.1, 0.5, 1.0, 2.0]
NTREES = [600, 1200, 2400]
# rotation grids
ROT_SIZE = [2, 3, 5, 8]
ROT_BOOT = [0.5, 0.75, 1.0]
ROT_FRAC = [0.0, 0.5, 1.0]
ROT_WITHIN = [False, True]
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(PROC, "block_final.csv")
# three_mod_block.csv
S117 = {"death_30d_inhosp":
        {"plain": 0.9143, "rot": 0.9156},
        "death_30d":
        {"plain": 0.9036, "rot": 0.9053},
        "composite_30d":
        {"plain": 0.8605, "rot": 0.8631},
        "cv_first":
        {"plain": 0.8180, "rot": 0.8219}}

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


def _pca_block(S):
    """Eigenvectors of the small covariance,
    largest first. Cheaper than sklearn's PCA
    for blocks of 2 to 8 columns."""
    S = S - S.mean(0)
    n = max(len(S) - 1, 1)
    try:
        w, V = np.linalg.eigh(S.T @ S / n)
        return V[:, ::-1]
    except Exception:
        return np.eye(S.shape[1])


class BlockForest:
    """Block-stratified column sampling with every
    remaining knob exposed, rotation included.

    cw        class_weight. "balanced" is what
              every earlier script used, and the
              literature links it to the 1.3-2.6
              calibration slopes seen throughout.
    ccp       minimal cost-complexity pruning.
    boot      rows drawn per tree, as a fraction
              of n.
    oob_w     weight each tree's vote by its
              out-of-bag AUROC.
    rot_frac  fraction of trees that get a PCA
              rotation. 0.0 is no rotation, 1.0 is
              what three_mod_block did.
    rot_size  columns per PCA block.
    rot_boot  rows sampled for each PCA fit.
    within    rotate WITHIN each modality block
              rather than across the drawn
              columns. Across mixes the modalities
              into the same PCA blocks, which
              undoes the block structure."""

    def __init__(self, n_estimators=600,
                 draws=(14, 8, 16),
                 bounds=(38, 312), leaf=25,
                 depth=None, cw="balanced",
                 ccp=0.0, boot=1.0,
                 oob_w=False, rot_frac=0.0,
                 rot_size=3, rot_boot=0.75,
                 within=False, seed=42):
        self.n = n_estimators
        self.draws = draws
        self.bounds = bounds
        self.leaf = leaf
        self.depth = depth
        self.cw = cw
        self.ccp = ccp
        self.boot = boot
        self.oob_w = oob_w
        self.rot_frac = rot_frac
        self.rot_size = rot_size
        self.rot_boot = rot_boot
        self.within = within
        self.seed = seed

    def _rot(self, Z, parts, rng, n):
        """Returns [(local_idx, C), ...].

        within=True builds PCA blocks inside each
        modality's drawn columns separately, so no
        rotation ever mixes two modalities.
        within=False permutes all drawn columns
        together, which is what three_mod_block
        did."""
        nb = max(10, int(round(self.rot_boot
                               * n)))
        groups = []
        if self.within:
            off = 0
            for pr in parts:
                loc = np.arange(off,
                                off + len(pr))
                off += len(pr)
                o = rng.permutation(loc)
                for i in range(0, len(o),
                               self.rot_size):
                    g = o[i:i + self.rot_size]
                    if len(g) >= 2:
                        groups.append(g)
        else:
            o = rng.permutation(Z.shape[1])
            for i in range(0, len(o),
                           self.rot_size):
                g = o[i:i + self.rot_size]
                if len(g) >= 2:
                    groups.append(g)
        out = []
        for g in groups:
            rows = rng.choice(len(Z), nb,
                              replace=True)
            out.append((g, _pca_block(
                Z[np.ix_(rows, g)])))
        return out

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        cuts = ([0] + [int(min(b, p))
                       for b in self.bounds]
                + [p])
        blk = [np.arange(cuts[i], cuts[i + 1])
               for i in range(len(cuts) - 1)]
        nb = max(20, int(round(self.boot * n)))
        nrot = int(round(self.rot_frac * self.n))
        self.sel_, self.tr_ = [], []
        self.rot_, self.w_ = [], []
        for ti in range(self.n):
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
            Z = X[:, cols]
            rt = []
            if ti < nrot:
                rt = self._rot(Z, parts, rng, n)
                Z2 = Z.copy()
                for g, C in rt:
                    Z2[:, g] = Z[:, g] @ C
                Z = Z2
            rows = rng.choice(n, nb,
                              replace=True)
            if len(np.unique(y[rows])) < 2:
                continue
            t = DecisionTreeClassifier(
                max_depth=self.depth,
                min_samples_leaf=self.leaf,
                ccp_alpha=self.ccp,
                class_weight=self.cw,
                random_state=int(
                    rng.integers(1e6)))
            t.fit(Z[rows], y[rows])
            wt = 1.0
            if self.oob_w:
                oob = np.setdiff1d(
                    np.arange(n), rows)
                if (len(oob) > 30
                        and 0 < y[oob].sum()
                        < len(oob)):
                    try:
                        a = roc_auc_score(
                            y[oob],
                            t.predict_proba(
                                Z[oob])[:, 1])
                        wt = max(a - 0.5, 1e-3)
                    except Exception:
                        wt = 1e-3
                else:
                    wt = 1e-3
            self.sel_.append(cols)
            self.rot_.append(rt)
            self.tr_.append(t)
            self.w_.append(wt)
        self.w_ = np.array(self.w_)
        if self.w_.sum() > 0:
            self.w_ = self.w_ / self.w_.sum()
        return self

    def predict_proba(self, X):
        out = np.zeros(len(X))
        if not self.tr_:
            out[:] = 0.5
            return np.column_stack(
                [1 - out, out])
        for cols, rt, t, w in zip(
                self.sel_, self.rot_,
                self.tr_, self.w_):
            Z = X[:, cols]
            if rt:
                Z2 = Z.copy()
                for g, C in rt:
                    Z2[:, g] = Z[:, g] @ C
                Z = Z2
            out += w * t.predict_proba(Z)[:, 1]
        return np.column_stack([1 - out, out])


def oof(Xd, MODS, y, grp, seed, cfg):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        Bt, Be = [], []
        for m_ in MODS:
            a, b = prep_fold(Xd[m_][tr],
                             Xd[m_][te])
            Bt.append(a)
            Be.append(b)
        Zt, Ze, bd = ccaaug(
            Bt, Be, cfg.get("k", K0),
            cfg.get("reg", REG0))
        bf = BlockForest(
            n_estimators=cfg.get("ntree",
                                 NTREE0),
            draws=cfg.get("draws",
                          (14, 8, 16)),
            bounds=bd,
            leaf=cfg.get("leaf", 25),
            depth=cfg.get("depth", None),
            cw=cfg.get("cw", "balanced"),
            ccp=cfg.get("ccp", 0.0),
            boot=cfg.get("boot", 1.0),
            oob_w=cfg.get("oob_w", False),
            rot_frac=cfg.get("rot_frac", 0.0),
            rot_size=cfg.get("rot_size", 3),
            rot_boot=cfg.get("rot_boot", 0.75),
            within=cfg.get("within", False),
            seed=seed)
        bf.fit(Zt, y[tr])
        p[te] = bf.predict_proba(Ze)[:, 1]
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
print("")
print("cohort:", len(D))
print("  EHR %d   ECG %d   CTPA %d"
      % (len(EH), len(ECGC), len(CT)))
print("")
print("  STAGES")
print("   E   draws x leaf, NEVER crossed")
print("   A   class_weight balanced vs None")
print("   B   ccp_alpha pruning")
print("   C   row bootstrap fraction")
print("   D   OOB-weighted voting")
print("   R1  rotated fraction x within/across")
print("   R2  rot_size x rot_boot")
print("   G   CCA reg to 2.0")
print("   H   tree count to 2400")
print("   final  ten seeds + paired bootstrap"
      " of rotation vs none", flush=True)

# ---------- TIMING PROBE ----------
print("")
print("TIMING PROBE")
Xp = np.random.rand(len(D), 400)
yp = (np.random.rand(len(D)) > 0.90) \
    .astype(int)
t = time.time()
BlockForest(n_estimators=NTREE0, leaf=25,
            bounds=(38, 312),
            seed=42).fit(Xp, yp)
t_pl = time.time() - t
t = time.time()
BlockForest(n_estimators=NTREE0, leaf=25,
            bounds=(38, 312), rot_frac=1.0,
            seed=42).fit(Xp, yp)
t_rot = time.time() - t
per = 0.5 * (t_pl + t_rot)
nE = len(DRAWS) * len(LEAF_MAIN)
nR1 = len(ROT_FRAC) * len(ROT_WITHIN)
nR2 = len(ROT_SIZE) * len(ROT_BOOT)
cells = ((nE + len(CW) + len(CCP)
          + len(BOOT) + 2 + nR1 + nR2
          + len(REGS) + len(NTREES))
         * len(OUTS))
ser = cells * per * NFOLD * len(SEEDS) / 60.0
fin = (len(SEEDS10) * NFOLD * per * 2
       * len(OUTS) / 60.0)
print("  plain    %d trees: %5.1f s"
      % (NTREE0, t_pl))
print("  rotated  %d trees: %5.1f s"
      % (NTREE0, t_rot))
print("")
print("  %d cells; serial about %.0f min,"
      " parallel about %.0f min"
      % (cells, ser + fin,
         (ser + fin) / NJOBS))
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
    pv = S117.get(oc, {})
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())))
    print("  three_mod_block: plain %.4f,"
          " +rotation %.4f (gain %+.4f, inside"
          % (pv.get("plain", np.nan),
             pv.get("rot", np.nan),
             pv.get("rot", 0) - pv.get("plain",
                                       0)))
    print("  one seed SD, tested once at a"
          " selected setting)", flush=True)

    cfg = {"draws": (14, 8, 16), "leaf": 25,
           "depth": None, "k": K0,
           "reg": REG0, "ntree": NTREE0,
           "cw": "balanced", "ccp": 0.0,
           "boot": 1.0, "oob_w": False,
           "rot_frac": 0.0, "rot_size": 3,
           "rot_boot": 0.75, "within": False}

    def run(c, seeds=SEEDS):
        ps = par(lambda s, c=c:
                 oof(Xd, MODS, y, grp, s, c),
                 seeds)
        aa = np.array([roc_auc_score(y, p)
                       for p in ps])
        return aa, ps[0]

    def sweep(stage, key, grid, label,
              fmt="%-10s"):
        print("")
        print("  STAGE %s: %s" % (stage,
                                  label))
        print("    " + fmt % key
              + " %9s %9s" % ("AUC", "SD"))
        bv, bm = cfg[key], -1.0
        for v in grid:
            c = dict(cfg)
            c[key] = v
            try:
                aa, p0 = run(c)
            except Exception as exc:
                print("    " + fmt % str(v)
                      + " FAILED %s"
                      % repr(exc)[:30])
                continue
            print("    " + fmt % str(v)
                  + " %9.4f %9.4f"
                  % (aa.mean(),
                     aa.std(ddof=1)),
                  flush=True)
            rows.append({
                "outcome": oc, "stage": stage,
                "name": key, "value": str(v),
                "mean": aa.mean(),
                "sd": aa.std(ddof=1)})
            if aa.mean() > bm:
                bm, bv = aa.mean(), v
        cfg[key] = bv
        print("    best %s" % str(bv),
              flush=True)
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)
        return bm

    # ---- STAGE E: draws x leaf ----
    print("")
    print("  STAGE E: DRAWS x LEAF")
    print("  never crossed; leaf matters by up"
          " to 0.0245 so the draw optimum may")
    print("  differ at leaf 25")
    lg = (LEAF_CV if oc == "cv_first"
          else LEAF_MAIN)
    print("  %-12s" % "draws", end="")
    for lf in lg:
        print(" %9s" % ("lf=%d" % lf), end="")
    print("")
    bE = None
    for dr in DRAWS:
        line = "  %-12s" % ("%d+%d+%d" % dr)
        for lf in lg:
            c = dict(cfg)
            c["draws"], c["leaf"] = dr, lf
            try:
                aa, _ = run(c)
            except Exception:
                line += " %9s" % "fail"
                continue
            line += " %9.4f" % aa.mean()
            rows.append({
                "outcome": oc, "stage": "E",
                "name": "draws_leaf",
                "draws": "%d+%d+%d" % dr,
                "leaf": lf, "mean": aa.mean(),
                "sd": aa.std(ddof=1)})
            if bE is None or aa.mean() > bE[0]:
                bE = (aa.mean(), dr, lf)
        print(line, flush=True)
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)
    if bE:
        cfg["draws"], cfg["leaf"] = bE[1], bE[2]
        print("  best %d+%d+%d leaf=%d  %.4f%s"
              % (bE[1] + (bE[2], bE[0],
                          "   (leaf at the grid edge)"
                          if bE[2] == lg[-1]
                          else "")), flush=True)

    sweep("A", "cw", CW,
          "CLASS WEIGHTING ('balanced' may"
          " explain the 1.3-2.6 slopes seen"
          " throughout)")
    sweep("B", "ccp", CCP,
          "ccp_alpha PRUNING")
    sweep("C", "boot", BOOT,
          "ROW BOOTSTRAP FRACTION")
    sweep("D", "oob_w", [False, True],
          "OOB-WEIGHTED VOTING")

    # ---- STAGE R1: rotated fraction x scope --
    print("")
    print("  STAGE R1: ROTATED FRACTION x SCOPE")
    print("  three_mod_block used frac=1.0 and"
          " across-block rotation, which mixes")
    print("  EHR, ECG and CTPA into the same PCA"
          " blocks and undoes the block")
    print("  structure. within=True keeps each"
          " modality's rotation separate.")
    print("  %-10s" % "rot_frac", end="")
    for wi in ROT_WITHIN:
        print(" %12s" % ("within=%s" % wi),
              end="")
    print("")
    bR = None
    for rf in ROT_FRAC:
        line = "  %-10.1f" % rf
        for wi in ROT_WITHIN:
            c = dict(cfg)
            c["rot_frac"], c["within"] = rf, wi
            try:
                aa, _ = run(c)
            except Exception:
                line += " %12s" % "fail"
                continue
            line += " %12.4f" % aa.mean()
            rows.append({
                "outcome": oc, "stage": "R1",
                "name": "frac_within",
                "rot_frac": rf,
                "within": int(wi),
                "mean": aa.mean(),
                "sd": aa.std(ddof=1)})
            if bR is None or aa.mean() > bR[0]:
                bR = (aa.mean(), rf, wi)
        print(line, flush=True)
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)
    if bR:
        cfg["rot_frac"], cfg["within"] = \
            bR[1], bR[2]
        print("  best frac=%.1f within=%s  %.4f"
              % (bR[1], bR[2], bR[0]),
              flush=True)

    # ---- STAGE R2: rot_size x rot_boot ----
    if cfg["rot_frac"] > 0:
        print("")
        print("  STAGE R2: rot_size x rot_boot")
        print("  both were fixed at 3 and 0.75"
              " before. rot_boot")
        print("  controls how much the rotations"
              " differ between trees, which is")
        print("  the rotation forest's main"
              " diversity lever.")
        print("  %-10s" % "rot_size", end="")
        for rb in ROT_BOOT:
            print(" %10s" % ("bt=%.2f" % rb),
                  end="")
        print("")
        bR2 = None
        for rs in ROT_SIZE:
            line = "  %-10d" % rs
            for rb in ROT_BOOT:
                c = dict(cfg)
                c["rot_size"] = rs
                c["rot_boot"] = rb
                try:
                    aa, _ = run(c)
                except Exception:
                    line += " %10s" % "fail"
                    continue
                line += " %10.4f" % aa.mean()
                rows.append({
                    "outcome": oc,
                    "stage": "R2",
                    "name": "size_boot",
                    "rot_size": rs,
                    "rot_boot": rb,
                    "mean": aa.mean(),
                    "sd": aa.std(ddof=1)})
                if bR2 is None or \
                        aa.mean() > bR2[0]:
                    bR2 = (aa.mean(), rs, rb)
            print(line, flush=True)
            pd.DataFrame(rows).to_csv(
                DEST, index=False)
        if bR2:
            cfg["rot_size"] = bR2[1]
            cfg["rot_boot"] = bR2[2]
            print("  best size=%d boot=%.2f"
                  "  %.4f"
                  % (bR2[1], bR2[2], bR2[0]),
                  flush=True)
    else:
        print("")
        print("  STAGE R2 SKIPPED: stage R1"
              " chose no rotation")

    sweep("G", "reg", REGS,
          "CCA reg EXTENDED (only 0.1 and 0.5"
          " tested before, 0.5 won at the top"
          " of the grid)")
    sweep("H", "ntree", NTREES,
          "TREE COUNT (1200 won a 300/600/1200"
          " grid, again at the top)")

    # ---- FINAL: ten seeds, rotation tested ----
    print("")
    print("  FINAL, TEN SEEDS")
    print("    %d+%d+%d leaf=%d cw=%s ccp=%.5f"
          " boot=%.1f oob=%s"
          % (cfg["draws"] + (cfg["leaf"],
                             str(cfg["cw"]),
                             cfg["ccp"],
                             cfg["boot"],
                             str(cfg["oob_w"]))))
    print("    rot_frac=%.1f within=%s size=%d"
          " rot_boot=%.2f reg=%.1f ntree=%d"
          % (cfg["rot_frac"],
             str(cfg["within"]),
             cfg["rot_size"], cfg["rot_boot"],
             cfg["reg"], cfg["ntree"]))
    cr = dict(cfg)
    cp_ = dict(cfg)
    cp_["rot_frac"] = 0.0
    if cr["rot_frac"] == 0.0:
        cr["rot_frac"] = 1.0
    ar, pr = run(cr, SEEDS10)
    ap, pp = run(cp_, SEEDS10)
    print("")
    print("    %-12s %8s %8s %9s"
          % ("model", "AUC", "SD", "vs 117"))
    for nm, aa, ref in (
            ("rotation", ar,
             pv.get("rot", np.nan)),
            ("no rotation", ap,
             pv.get("plain", np.nan))):
        print("    %-12s %8.4f %8.4f %+9.4f"
              % (nm, aa.mean(),
                 aa.std(ddof=1),
                 aa.mean() - ref))
    gn, lo, hi, _ = f.boot_diff(
        y, _rank(pr), _rank(pp), grp)
    print("")
    print("    PAIRED BOOTSTRAP, rotation minus"
          " none: %+.4f [%+.4f,%+.4f] %s"
          % (gn, lo, hi,
             "*" if (lo > 0 or hi < 0)
             else ""))
    print("    three_mod_block never did this;"
          " it compared single runs")
    keep_rot = ar.mean() >= ap.mean()
    best_p = pr if keep_rot else pp
    best_a = ar if keep_rot else ap
    b0, s0 = calib(best_p, y)
    pc = platt_oof(best_p, y, grp)
    b1, s1 = calib(pc, y)
    print("")
    print("    chosen: %s   slope %.3f -> %.3f"
          "   Brier %.5f -> %.5f"
          % ("rotation" if keep_rot
             else "no rotation", s0, s1,
             b0, b1), flush=True)
    rows.append({
        "outcome": oc, "stage": "final",
        "name": ("rotation" if keep_rot
                 else "no_rotation"),
        "draws": "%d+%d+%d" % cfg["draws"],
        "leaf": cfg["leaf"],
        "cw": str(cfg["cw"]),
        "ccp": cfg["ccp"], "boot": cfg["boot"],
        "oob_w": int(cfg["oob_w"]),
        "rot_frac": (cr["rot_frac"]
                     if keep_rot else 0.0),
        "within": int(cfg["within"]),
        "rot_size": cfg["rot_size"],
        "rot_boot": cfg["rot_boot"],
        "reg": cfg["reg"],
        "ntree": cfg["ntree"],
        "mean": best_a.mean(),
        "sd": best_a.std(ddof=1),
        "mean_rot": ar.mean(),
        "mean_plain": ap.mean(),
        "boot_diff": gn, "lo": lo, "hi": hi,
        "sig": int(lo > 0 or hi < 0),
        "slope": s0, "slope_platt": s1,
        "brier": b0, "brier_platt": b1,
        "vs_117": best_a.mean()
        - max(pv.get("rot", 0),
              pv.get("plain", 0))})

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
print("STAGE E: DRAWS x LEAF")
e_ = r[r["stage"] == "E"]
if len(e_):
    print("  marginal by draws")
    print(e_.groupby("draws")["mean"].mean()
          .round(4).to_string())
    print("  marginal by leaf")
    print(e_.groupby("leaf")["mean"].mean()
          .round(4).to_string())
    print("")
    print("  IS THERE AN INTERACTION?")
    for oc in OUTS:
        z = e_[e_["outcome"] == oc]
        if not len(z):
            continue
        piv = z.pivot_table(index="draws",
                            columns="leaf",
                            values="mean")
        print("    %-18s best leaf per draw: %s"
              % (oc, [int(piv.loc[d].idxmax())
                      for d in piv.index]))

print("")
print("=" * 74)
print("ROTATION")
r1 = r[r["stage"] == "R1"]
if len(r1):
    print("  rotated fraction x scope, marginal")
    print(r1.groupby("rot_frac")["mean"].mean()
          .round(4).to_string())
    print("  by scope (1 = within-block)")
    print(r1.groupby("within")["mean"].mean()
          .round(4).to_string())
    print("")
    print("  three_mod_block used frac=1.0 and"
          " within=0. If within=1 wins, keeping")
    print("  each modality's rotation separate"
          " matters.")
r2 = r[r["stage"] == "R2"]
if len(r2):
    print("")
    print("  rot_size x rot_boot")
    print(r2.pivot_table(index="rot_size",
                         columns="rot_boot",
                         values="mean")
          .round(4).to_string())
    print("  marginal by size (was hardcoded 3)")
    print(r2.groupby("rot_size")["mean"].mean()
          .round(4).to_string())
    print("  marginal by boot (was hardcoded"
          " 0.75)")
    print(r2.groupby("rot_boot")["mean"].mean()
          .round(4).to_string())

print("")
print("OTHER STAGES")
for st, lab2 in (("A", "class weighting"),
                 ("B", "ccp_alpha"),
                 ("C", "row bootstrap"),
                 ("D", "OOB weighting"),
                 ("G", "CCA reg"),
                 ("H", "tree count")):
    z = r[r["stage"] == st]
    if not len(z):
        continue
    print("")
    print("  " + lab2)
    print(z.pivot_table(index="value",
                        columns="outcome",
                        values="mean")
          .round(4).to_string())
    rg = z["mean"].max() - z["mean"].min()
    print("  range %.4f   mean seed SD %.4f"
          "   ratio %.1f"
          % (rg, z["sd"].mean(),
             rg / max(z["sd"].mean(), 1e-9)))

print("")
print("=" * 74)
print("FINAL MODELS")
fz = r[r["stage"] == "final"]
if len(fz):
    print(fz[["outcome", "name", "draws",
              "leaf", "cw", "ccp", "boot",
              "rot_frac", "within",
              "rot_size", "rot_boot", "reg",
              "ntree", "mean", "sd",
              "vs_117"]].round(4)
          .to_string(index=False))
    print("")
    print("IS ROTATION REAL?")
    print("  three_mod_block found +0.0013 to"
          " +0.0039, all inside one seed SD,")
    print("  from a single run per outcome. This"
          " is ten seeds and a paired")
    print("  bootstrap.")
    print(fz[["outcome", "mean_rot",
              "mean_plain", "boot_diff", "lo",
              "hi", "sig"]].round(4)
          .to_string(index=False))
    print("")
    print("  significant: %d of %d"
          % (int(fz["sig"].fillna(0).sum()),
             len(fz)))
    print("")
    print("CALIBRATION")
    print(fz[["outcome", "cw", "slope",
              "slope_platt", "brier",
              "brier_platt"]].round(4)
          .to_string(index=False))
print("")
print("saved", DEST, r.shape)
keep_awake(False)