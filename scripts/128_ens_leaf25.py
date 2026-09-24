"""Script 127 rerun with leaf and depth swept in
the block-forest model.

Script 127 fixed leaf=1 and depth=12. Script 116's
leaf marginal rises from leaf 1 to leaf 25, and
script 127's block model scored 0.008 to 0.025
below the same architecture in
three_mod_block.csv.

CARRIED FORWARD from script 127
  k=10, reg=0.5, an interior optimum
  draws 14+8+16
  ens_all as the base set
  logit averaging as the combination rule

Stacking the modality predictions into the block
forest is dropped; in script 127 it helped only
ECG+CTPA.

Both the late-fusion model and the block forest
are Platt-calibrated: the weighted rank average
is over-confident (slopes 0.107 to 0.283) and the
forest under-confident (1.44 to 1.75). Only the
calibrated Brier score is reported.

Parallel across seeds. Timing probe first.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ens_leaf25.csv
  results\\ens_leaf25_log.txt
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
NINNER = 4
NJOBS = max(1, min(10, (os.cpu_count() or 4)
                   - 2))
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CT_CS = list(np.logspace(-4, 4, 10))
K, REG, NTREE = 10, 0.5, 600
DRAW3 = (14, 8, 16)
DRAW2 = {"ehr": 14, "ecg": 8, "ctpa": 16}
LEAF = [1, 5, 10, 25, 50]
DEPTH = [6, 12, None]
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(PROC, "ens_leaf25.csv")
# script 127, which used leaf=1
S127 = {"death_30d_inhosp":
        {"block": 0.8969, "late": 0.9061,
         "nnls": 0.9067, "best": 0.9138},
        "death_30d":
        {"block": 0.8955, "late": 0.9080,
         "nnls": 0.9075, "best": 0.9080},
        "composite_30d":
        {"block": 0.8572, "late": 0.8672,
         "nnls": 0.8695, "best": 0.8695},
        "cv_first":
        {"block": 0.7934, "late": 0.8159,
         "nnls": 0.8154, "best": 0.8159}}
# three_mod_block.csv, the architecture done right
S117 = {"death_30d_inhosp": 0.9143,
        "death_30d": 0.9036,
        "composite_30d": 0.8605,
        "cv_first": 0.8180}

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


def L_ens(Xt, yt, Xe, grid=None):
    """ens: rank average of L2 and a forest. Best
    marginal on all three modalities in
    base_learners.csv: EHR +0.0212, ECG +0.0156,
    CTPA +0.0094 over L2."""
    a = _rank(L_l2(Xt, yt, Xe, grid))
    b = _rank(L_rf(Xt, yt, Xe, grid))
    return 0.5 * (a + b)


class BlockForest:
    def __init__(self, n_estimators=600,
                 draws=(14, 8, 16),
                 bounds=(38, 312), leaf=25,
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
print("")
print("cohort:", len(D))
print("  EHR %d   ECG %d   CTPA %d"
      % (len(EH), len(ECGC), len(CT)))
print("")
print("  script 127 used leaf=1; script 116's")
print("  leaf marginal rises to leaf 25")
print("  leaf grid:", LEAF, " depth:", DEPTH)
print("  carried forward: k=%d reg=%.1f draws"
      " %d+%d+%d, ens_all bases, logit rule"
      % ((K, REG) + DRAW3), flush=True)

# ---------- TIMING PROBE ----------
print("")
print("TIMING PROBE")
Xp = np.random.rand(len(D), 400)
yp = (np.random.rand(len(D)) > 0.90) \
    .astype(int)
t = time.time()
BlockForest(n_estimators=NTREE, leaf=25,
            depth=12, bounds=(38, 312),
            seed=42).fit(Xp, yp)
t_blk = time.time() - t
t = time.time()
L_ens(np.random.rand(len(D), len(ECGC)), yp,
      np.random.rand(100, len(ECGC)))
t_ens = time.time() - t
nA = len(LEAF) * len(DEPTH)
stA = (nA * t_blk * NFOLD * len(SEEDS)
       * len(OUTS) / 60.0)
stB = ((3 * (NINNER + 1) * t_ens + 2 * t_blk)
       * NFOLD * len(SEEDS10) * len(OUTS)
       / 60.0)
print("  block forest %d trees: %5.1f s"
      % (NTREE, t_blk))
print("  ens on the ECG block:  %5.1f s"
      % t_ens)
print("")
print("  stage A %.0f min, stage B %.0f min,"
      " serial" % (stA, stB))
print("  with %d-way parallelism about %.0f"
      " min" % (NJOBS, (stA + stB) / NJOBS))
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
    pv = S127.get(oc, {})
    ref117 = S117.get(oc, np.nan)
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())))
    print("  script 127 (leaf=1): block %.4f,"
          " nnls %.4f, best %.4f"
          % (pv.get("block", np.nan),
             pv.get("nnls", np.nan),
             pv.get("best", np.nan)))
    print("  three_mod_block.csv, same"
          " architecture done right: %.4f"
          % ref117, flush=True)

    # ---- STAGE A: leaf x depth ----
    print("")
    print("  STAGE A: LEAF x DEPTH")
    print("  %-6s" % "leaf", end="")
    for dp in DEPTH:
        print(" %10s" % ("d=%s" % dp), end="")
    print("")
    best = None
    for lf in LEAF:
        line = "  %-6d" % lf
        for dp in DEPTH:
            def one(s, lf=lf, dp=dp):
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
                        Bt, Be, K, REG)
                    bf = BlockForest(
                        n_estimators=NTREE,
                        draws=DRAW3,
                        bounds=bd, leaf=lf,
                        depth=dp, seed=s)
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
                "leaf": lf,
                "depth": (-1 if dp is None
                          else dp),
                "mean": aa.mean(),
                "sd": aa.std(ddof=1)})
            if best is None or \
                    aa.mean() > best[0]:
                best = (aa.mean(), lf, dp)
        print(line, flush=True)
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)
    blf, bdp = (best[1], best[2]) if best \
        else (25, 12)
    print("  best leaf=%d depth=%s  %.4f"
          % (blf, bdp,
             best[0] if best else np.nan))
    print("  script 127 used leaf=1, which"
          " scored %.4f here"
          % next((r["mean"] for r in rows
                  if r["outcome"] == oc
                  and r.get("leaf") == 1
                  and r.get("depth") == 12),
                 np.nan), flush=True)

    # ---- STAGE B: the full model, 10 seeds ----
    print("")
    print("  STAGE B: FULL MODEL, TEN SEEDS")
    print("  ens_all bases, logit late fusion,"
          " block forest at leaf=%d depth=%s"
          % (blf, bdp))

    def big(s):
        n = len(y)
        out = {"late": np.zeros(n),
               "block": np.zeros(n)}
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
            ptr, pte = [], []
            for mi, m_ in enumerate(MODS):
                ptr.append(inner_oof(
                    Xd[m_][tr], y[tr],
                    grp[tr], L_ens, s))
                pte.append(L_ens(
                    Bt[mi], y[tr], Be[mi]))
            # LOGIT averaging: marginal 0.8606,
            # ahead of rank 0.8598 and every
            # calibrated form
            A_ = np.column_stack(
                [_logit(p) for p in ptr])
            B_ = np.column_stack(
                [_logit(p) for p in pte])
            out["late"][te] = grid_w(A_, y[tr],
                                     B_)
            Zt, Ze, bd = ccaaug(Bt, Be, K, REG)
            bf = BlockForest(
                n_estimators=NTREE,
                draws=DRAW3, bounds=bd,
                leaf=blf, depth=bdp, seed=s)
            bf.fit(Zt, y[tr])
            out["block"][te] = \
                bf.predict_proba(Ze)[:, 1]
        return out

    try:
        Os = par(big, seeds=SEEDS10)
    except Exception as exc:
        print("  FAILED %s" % repr(exc)[:70])
        continue
    al = np.array([roc_auc_score(y, o["late"])
                   for o in Os])
    ab = np.array([roc_auc_score(y, o["block"])
                   for o in Os])
    pl, pb = Os[0]["late"], Os[0]["block"]
    R = np.column_stack([_rank(pl), _rank(pb)])
    pe_ = nnls_oof(R, y, grp)
    an = roc_auc_score(y[np.isfinite(pe_)],
                       pe_[np.isfinite(pe_)])
    print("")
    print("  %-10s %8s %8s %9s %9s"
          % ("model", "AUC", "SD", "vs 127",
             "vs 117"))
    for nm, v, sd in (("late", al.mean(),
                       al.std(ddof=1)),
                      ("block", ab.mean(),
                       ab.std(ddof=1)),
                      ("nnls", an, np.nan)):
        print("  %-10s %8.4f %8.4f %+9.4f"
              " %+9.4f"
              % (nm, v, sd,
                 v - pv.get(nm if nm != "late"
                            else "late",
                            np.nan),
                 v - ref117))
        rows.append({
            "outcome": oc, "stage": "B",
            "name": nm, "leaf": blf,
            "depth": (-1 if bdp is None
                      else bdp),
            "mean": v, "sd": sd,
            "vs_127": v - pv.get(
                nm if nm != "late" else "late",
                np.nan),
            "vs_117": v - ref117})

    # calibration
    print("")
    print("  CALIBRATION")
    print("  a weighted logit average is not on"
          " a probability scale, so its RAW")
    print("  Brier is meaningless; the block"
          " forest is under-confident. Both")
    print("  need Platt, and only the corrected"
          " figures are reportable.")
    for nm, p in (("late", pl), ("block", pb),
                  ("nnls", pe_)):
        b0, s0 = calib(p, y)
        pc = platt_oof(p, y, grp)
        b1, s1 = calib(pc, y)
        print("    %-6s slope %6.3f -> %.3f"
              "   Brier %.5f -> %.5f"
              % (nm, s0, s1, b0, b1))
        rows.append({
            "outcome": oc, "stage": "C",
            "name": nm, "slope": s0,
            "slope_platt": s1, "brier": b0,
            "brier_platt": b1})

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
print("STAGE A: LEAF x DEPTH")
a_ = r[r["stage"] == "A"]
if len(a_):
    print(a_.pivot_table(index="leaf",
                         columns="outcome",
                         values="mean")
          .round(4).to_string())
    print("")
    print("  marginal by leaf")
    print(a_.groupby("leaf")["mean"].mean()
          .round(4).to_string())
    print("  marginal by depth"
          "  (-1 is unrestricted)")
    print(a_.groupby("depth")["mean"].mean()
          .round(4).to_string())
    print("")
    print("  script 116's verified marginal"
          " RISES to leaf 25. If this agrees,")
    print("  the setting is confirmed on a"
          " second cohort.")

print("")
print("=" * 74)
print("STAGE B: THE MODELS")
b_ = r[r["stage"] == "B"]
if len(b_):
    print(b_.pivot_table(index="name",
                         columns="outcome",
                         values="mean")
          .round(4).to_string())
    print("")
    print("  AGAINST SCRIPT 127 (leaf=1)")
    print(b_.pivot_table(index="name",
                         columns="outcome",
                         values="vs_127")
          .round(4).to_string())
    print("")
    print("  AGAINST three_mod_block.csv")
    print(b_.pivot_table(index="name",
                         columns="outcome",
                         values="vs_117")
          .round(4).to_string())

print("")
print("DID FIXING THE LEAF RECOVER THE GAP?")
print("  script 127's block model was 0.008 to"
      " 0.025 below the same architecture in")
print("  three_mod_block.csv. A vs_117 near"
      " zero means the gap was the leaf.")
if len(b_):
    z = b_[b_["name"] == "block"]
    for _, x in z.iterrows():
        print("  %-18s block %.4f   vs 117"
              " %+.4f   vs 127 %+.4f"
              % (x["outcome"], x["mean"],
                 x["vs_117"], x["vs_127"]))

print("")
print("CALIBRATION")
c_ = r[r["stage"] == "C"]
if len(c_):
    print(c_[["outcome", "name", "slope",
              "slope_platt", "brier",
              "brier_platt"]].round(4)
          .to_string(index=False))

print("")
print("BEST PER OUTCOME")
for oc in OUTS:
    s = b_[b_["outcome"] == oc]
    pv = S127.get(oc, {})
    if not len(s):
        continue
    x = s.loc[s["mean"].idxmax()]
    print("  %-18s %-6s %.4f   script 127 best"
          " %.4f   %+.4f"
          % (oc, x["name"], x["mean"],
             pv.get("best", np.nan),
             x["mean"] - pv.get("best",
                                np.nan)))
print("")
print("saved", DEST, r.shape)
keep_awake(False)