"""All four modality combinations on one cohort,
with model-level fusion and competence analysis.

EHR+ECG was measured on 3,501 patients (script
116) and EHR+CTPA and the three-modality model on
1,636 (scripts 110, 111 and 117). Here every
combination is fitted and scored on the same
1,636 patients, including ECG+CTPA with a tree
learner, a plain random forest on all three
modalities, and PCA with a tree on three blocks.

STAGE C looks for where each model is competent,
without nominating a variable:
  ola, lca   dynamic ensemble selection: each
             model is weighted by its error among
             the k nearest neighbours (LCA uses
             only neighbours whose label matches
             the model's prediction), in EHR space
             and in the full feature space
  comp_tree  a shallow tree predicting which model
             wins, whose printed splits show where
             each model is competent
  diff_imp   a forest on the signed error
             difference, ranking features by how
             much they separate the models
  auto_strat stratified weighting on the feature
             diff_imp ranks first
  des_meta   competence scores passed as features
             to a forest alongside the base
             predictions

Parallel across seeds. Timing probe first.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\same_cohort.csv
  data\\processed\\same_cohort_competence.csv
  results\\same_cohort_log.txt
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import linalg as sla
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.neighbors import NearestNeighbors
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
PCA_K = 8
KNN = 40
TREE_DEPTH = 3
TOP_N = 10
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(PROC, "same_cohort.csv")
CDEST = os.path.join(
    PROC, "same_cohort_competence.csv")
BEST_C = {"cv_first": 1e-4,
          "composite_30d": 1e-3,
          "death_30d": 1e-3,
          "death_30d_inhosp": 3e-4}
DRAW = {"ehr": 20, "ecg": 8, "ctpa": 16}
COMBOS = [("ehr", "ecg"), ("ehr", "ctpa"),
          ("ecg", "ctpa"),
          ("ehr", "ecg", "ctpa")]

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
    """Draws a fixed number of columns from each
    block, so no block is crowded out by a
    larger one."""

    def __init__(self, n_estimators=600,
                 draws=(20, 8), bounds=(38,),
                 leaf=1, depth=12, seed=42):
        self.n = n_estimators
        self.draws = draws
        self.bounds = bounds
        self.leaf = leaf
        self.depth = depth
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        cuts = ([0]
                + [int(min(b, p))
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


def make_rep(Bt, Be, mode):
    """mode: raw | ccaaug | pca.

    ccaaug appends PAIRWISE canonical variates
    for every pair of blocks, so cross-modal
    directions exist between all of them."""
    if mode == "raw":
        Zt = np.column_stack(Bt)
        Ze = np.column_stack(Be)
        bd = np.cumsum([b.shape[1]
                        for b in Bt])[:-1]
        return Zt, Ze, tuple(bd)
    if mode == "pca":
        Zt = np.column_stack(Bt)
        Ze = np.column_stack(Be)
        k = int(min(PCA_K, Zt.shape[1] - 1,
                    len(Zt) - 1))
        pc = PCA(n_components=k,
                 random_state=42)
        return (pc.fit_transform(Zt),
                pc.transform(Ze), ())
    pt = [[b] for b in Bt]
    pe = [[b] for b in Be]
    nb = len(Bt)
    for i in range(nb):
        for jj in range(i + 1, nb):
            W1, W2 = rcca(Bt[i], Bt[jj],
                          REG, K)
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


def oof_combo(Xs, y, grp, seed, names,
              method):
    """method: late | block | rf_raw | rf_pca
    | rf_ccaaug"""
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    if method == "late":
        ps = []
        for X, nm in zip(Xs, names):
            q = np.zeros(len(y))
            gd = (CT_CS if nm == "ctpa"
                  else None)
            for tr, te in cv.split(
                    np.zeros((len(y), 1)), y,
                    grp):
                Xt, Xe = prep_fold(X[tr],
                                   X[te])
                q[te] = learn_l2(
                    Xt, y[tr], Xe, grp[tr],
                    None, gd)
            ps.append(_rank(q))
        if len(ps) == 2:
            bs, bv = -1.0, ps[0]
            for w in np.arange(0, 1.001,
                               0.05):
                v = w * ps[1] + (1 - w) * ps[0]
                a = roc_auc_score(y, v)
                if a > bs:
                    bs, bv = a, v
            return bv
        bs, bv = -1.0, ps[0]
        for w1 in np.arange(0, 1.001, 0.05):
            for w2 in np.arange(
                    0, 1.001 - w1, 0.05):
                v = (w1 * ps[0] + w2 * ps[1]
                     + (1 - w1 - w2) * ps[2])
                a = roc_auc_score(y, v)
                if a > bs:
                    bs, bv = a, v
        return bv
    mode = ("ccaaug"
            if method in ("block",
                          "rf_ccaaug")
            else ("pca" if method == "rf_pca"
                  else "raw"))
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        Bt, Be = [], []
        for X in Xs:
            a, b = prep_fold(X[tr], X[te])
            Bt.append(a)
            Be.append(b)
        Zt, Ze, bd = make_rep(Bt, Be, mode)
        if method == "block":
            dr = tuple(DRAW[n] for n in names)
            m = BlockForest(
                n_estimators=NTREE, draws=dr,
                bounds=bd, leaf=LEAF,
                depth=DEPTH, seed=seed)
        else:
            m = RandomForestClassifier(
                n_estimators=NTREE,
                max_depth=DEPTH,
                min_samples_leaf=LEAF,
                class_weight=
                "balanced_subsample",
                random_state=seed, n_jobs=1)
        m.fit(Zt, y[tr])
        p[te] = m.predict_proba(Ze)[:, 1]
    return p


def par(fn, seeds=SEEDS):
    return Parallel(n_jobs=NJOBS,
                    backend="loky")(
        delayed(fn)(s) for s in seeds)


# ---------- competence machinery ------------
def des_weights(P, y, Xref, mode="ola",
                k=KNN):
    """Per-patient weights from the region of
    COMPETENCE.

    No variable is nominated: the region is the
    k nearest neighbours across WHATEVER space
    Xref spans, so every column contributes to
    deciding who counts as similar.

    ola  uses the whole region.
    lca  uses only neighbours whose label
         matches the model's own prediction,
         the a-posteriori variant."""
    n = len(y)
    nn = NearestNeighbors(
        n_neighbors=min(k + 1, n)).fit(Xref)
    _, idx = nn.kneighbors(Xref)
    idx = idx[:, 1:]
    comp = np.zeros((n, len(P)))
    for mi, p in enumerate(P):
        err = np.abs(p - y)
        if mode == "ola":
            comp[:, mi] = 1.0 - err[idx].mean(1)
        else:
            pred = (p >= 0.5).astype(int)
            same = y[idx] == pred[:, None]
            num = np.where(
                same, 1.0 - err[idx],
                0.0).sum(1)
            den = np.maximum(same.sum(1), 1)
            comp[:, mi] = num / den
    comp = np.clip(comp, 1e-6, None)
    return comp / comp.sum(1, keepdims=True)


def competence_target(P, y):
    """Signed error difference. Positive means
    model 1 was closer to the truth for that
    patient."""
    return np.abs(P[0] - y) - np.abs(P[1] - y)


def competence_tree(P, y, Xall, names,
                    depth=TREE_DEPTH):
    """A shallow tree predicting which model
    WINS from every available feature.

    This is the automatic, readable version of
    the subgroup hypothesis: rather than
    nominating age, it searches all 338 columns
    and the printed splits say where each model
    is competent."""
    z = competence_target(P, y)
    w = np.abs(z)
    lab = (z > 0).astype(int)
    keep = w > np.quantile(w, 0.25)
    if keep.sum() < 100:
        keep = np.ones(len(z), bool)
    t = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=max(
            30, int(0.05 * keep.sum())),
        random_state=42)
    try:
        t.fit(Xall[keep], lab[keep],
              sample_weight=w[keep])
    except Exception:
        return None, None
    try:
        txt = export_text(
            t, feature_names=list(names),
            max_depth=depth, decimals=2)
    except Exception:
        txt = ""
    return t, txt


def difference_importance(P, y, Xall, names,
                          top=TOP_N):
    """Rank every feature by how much it
    separates the two models' competence.

    Regressing the signed error difference on
    all features and reading the importances
    answers 'where do they diverge' without any
    variable being chosen in advance."""
    z = competence_target(P, y)
    m = RandomForestRegressor(
        n_estimators=400, max_depth=6,
        min_samples_leaf=20,
        random_state=42, n_jobs=1)
    try:
        m.fit(Xall, z)
    except Exception:
        return []
    imp = m.feature_importances_
    order = np.argsort(-imp)[:top]
    return [(names[i], float(imp[i]))
            for i in order]


def strat_weights(P, y, strat):
    """Fusion weights fitted within strata. If
    one model is accurate for one kind of
    patient and another elsewhere, a global
    weight is wrong for both."""
    n = len(y)
    W = np.full((n, len(P)), 1.0 / len(P))
    for s in np.unique(strat):
        m = strat == s
        if m.sum() < 50 or y[m].sum() < 5:
            continue
        best, bw = -1.0, None
        for w in np.arange(0, 1.001, 0.05):
            v = w * P[1][m] + (1 - w) * P[0][m]
            try:
                a = roc_auc_score(y[m], v)
            except Exception:
                continue
            if a > best:
                best, bw = a, (1 - w, w)
        if bw:
            W[m] = np.array(bw)
    return W


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
ALL_NAMES = EH + ECGC + CT
print("")
print("cohort:", len(D), " EVERY combination"
      " is fitted and scored here")
print("  EHR %d   ECG %d   CTPA %d   total %d"
      % (len(EH), len(ECGC), len(CT),
         len(ALL_NAMES)))
print("  stage C searches all %d features for"
      " competence regions; nothing is"
      % len(ALL_NAMES))
print("  nominated by hand", flush=True)

# ---------- TIMING PROBE ----------
print("")
print("TIMING PROBE")
npro = len(D)
Xp = np.random.rand(npro, 400)
yp = (np.random.rand(npro) > 0.90).astype(int)
t = time.time()
BlockForest(n_estimators=NTREE,
            draws=(20, 8, 16),
            bounds=(38, 312), leaf=LEAF,
            depth=DEPTH, seed=42).fit(Xp, yp)
t_blk = time.time() - t
t = time.time()
RandomForestClassifier(
    n_estimators=NTREE, max_depth=DEPTH,
    min_samples_leaf=LEAF, n_jobs=1,
    random_state=42).fit(Xp, yp)
t_rf = time.time() - t
per = 0.6 * t_blk + 0.4 * t_rf
cells = len(COMBOS) * 5 * len(OUTS)
ser = cells * per * NFOLD * len(SEEDS) / 60.0
print("  block forest  %d trees: %5.1f s"
      % (NTREE, t_blk))
print("  random forest %d trees: %5.1f s"
      % (NTREE, t_rf))
print("")
print("  stage A: %d cells" % cells)
print("  serial about %.0f min; with %d-way"
      " parallelism about %.0f min"
      % (ser, NJOBS, ser / NJOBS))
print("  stages B and C reuse stage A's"
      " predictions and are nearly free")
print("  Ctrl+C now if too long. Results save"
      " after every outcome.")
time.sleep(5)

rows, crows = [], []
t0 = time.time()
METHODS = ["late", "block", "rf_raw",
           "rf_pca", "rf_ccaaug"]

for oc in OUTS:
    if oc not in D.columns:
        continue
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    if y.sum() < 25:
        continue
    X = {k: d[v].values.astype(float)
         for k, v in BLK.items()}
    Xall = np.nan_to_num(np.column_stack(
        [X["ehr"], X["ecg"], X["ctpa"]]))
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())))

    uni = {}
    for nm in ("ehr", "ecg", "ctpa"):
        gd = CT_CS if nm == "ctpa" else None
        q = np.zeros(len(y))
        cv = StratifiedGroupKFold(
            n_splits=NFOLD, shuffle=True,
            random_state=42)
        for tr, te in cv.split(
                np.zeros((len(y), 1)), y, grp):
            Xt, Xe = prep_fold(X[nm][tr],
                               X[nm][te])
            q[te] = learn_l2(Xt, y[tr], Xe,
                             grp[tr], None, gd)
        uni[nm] = q
        print("  %-5s %.4f"
              % (nm, roc_auc_score(y, q)))
        rows.append({
            "outcome": oc, "stage": "uni",
            "combo": nm, "method": "l2",
            "mean": roc_auc_score(y, q),
            "sd": np.nan})

    # ---- STAGE A ----
    print("")
    print("  STAGE A: every combination on"
          " THIS cohort")
    print("  %-16s %-10s %8s %8s"
          % ("combo", "method", "AUC", "SD"))
    store = {}
    for cb in COMBOS:
        nm = "+".join(cb)
        for meth in METHODS:
            try:
                ps = par(lambda s, cb=cb,
                         meth=meth:
                         oof_combo(
                             [X[k] for k in cb],
                             y, grp, s, cb,
                             meth))
            except Exception as exc:
                print("  %-16s %-10s FAILED %s"
                      % (nm, meth,
                         repr(exc)[:35]))
                continue
            aa = np.array([roc_auc_score(y, p)
                           for p in ps])
            store[(nm, meth)] = ps[0]
            print("  %-16s %-10s %8.4f %8.4f"
                  % (nm, meth, aa.mean(),
                     aa.std(ddof=1)),
                  flush=True)
            rows.append({
                "outcome": oc, "stage": "A",
                "combo": nm, "method": meth,
                "mean": aa.mean(),
                "sd": aa.std(ddof=1)})
            pd.DataFrame(rows).to_csv(
                DEST, index=False)

    a = store.get(("ehr+ecg", "block"))
    b = store.get(("ehr+ctpa", "block"))
    c3 = store.get(("ehr+ecg+ctpa", "block"))
    base3 = (roc_auc_score(y, c3)
             if c3 is not None else np.nan)

    # ---- STAGE B ----
    if a is not None and b is not None:
        print("")
        print("  STAGE B: fusing the"
              " two-modality models")
        ra, rb = _rank(a), _rank(b)
        res = {"mean": 0.5 * (ra + rb)}
        bs, bv = -1.0, ra
        for w in np.arange(0, 1.001, 0.05):
            v = w * rb + (1 - w) * ra
            s_ = roc_auc_score(y, v)
            if s_ > bs:
                bs, bv = s_, v
        res["grid"] = bv
        for nm2, v in res.items():
            av = roc_auc_score(y, v)
            print("    %-12s %.4f   vs 3-mod"
                  " block %.4f   %+.4f"
                  % (nm2, av, base3,
                     av - base3), flush=True)
            rows.append({
                "outcome": oc, "stage": "B",
                "combo": "models",
                "method": nm2, "mean": av,
                "sd": np.nan})

    # ---- STAGE C ----
    if a is not None and b is not None:
        print("")
        print("  STAGE C: WHERE IS EACH MODEL"
              " COMPETENT?")
        print("  nothing below nominates a"
              " variable: the regions are")
        print("  found from all %d features"
              % len(ALL_NAMES))
        P = [np.asarray(a), np.asarray(b)]
        Rk = [_rank(P[0]), _rank(P[1])]
        Xehr = StandardScaler().fit_transform(
            np.nan_to_num(X["ehr"]))
        Xfull = StandardScaler().fit_transform(
            Xall)

        # DES on two different spaces
        for space, Xref in (("ehr", Xehr),
                            ("full", Xfull)):
            for mode in ("ola", "lca"):
                W = des_weights(P, y, Xref,
                                mode)
                v = (W[:, 0] * Rk[0]
                     + W[:, 1] * Rk[1])
                av = roc_auc_score(y, v)
                print("    %-14s %.4f   %+.4f"
                      % ("%s/%s" % (mode,
                                    space),
                         av, av - base3),
                      flush=True)
                rows.append({
                    "outcome": oc,
                    "stage": "C",
                    "combo": "des",
                    "method": "%s_%s"
                              % (mode, space),
                    "mean": av, "sd": np.nan})

        # the competence tree, printed
        print("")
        print("    COMPETENCE TREE  (which"
              " model wins, from all features)")
        tmod, txt = competence_tree(
            P, y, Xall, ALL_NAMES)
        if txt:
            for line in txt.split("\n")[:25]:
                if line.strip():
                    print("      " + line)
            print("      class 1 = EHR+CTPA"
                  " model wins that patient")

        # importance ranking, no variable
        # nominated
        print("")
        print("    FEATURES THAT SEPARATE"
              " COMPETENCE  (top %d of %d)"
              % (TOP_N, len(ALL_NAMES)))
        imps = difference_importance(
            P, y, Xall, ALL_NAMES)
        for nm3, iv in imps:
            print("      %-34s %.4f"
                  % (nm3[:34], iv))
            crows.append({
                "outcome": oc,
                "feature": nm3,
                "importance": iv})
        pd.DataFrame(crows).to_csv(
            CDEST, index=False)

        # stratify on whatever ranked first
        if imps:
            top_name = imps[0][0]
            ci = ALL_NAMES.index(top_name)
            col = Xall[:, ci]
            qs = np.nanquantile(col,
                                [1 / 3, 2 / 3])
            strat = np.digitize(col, qs)
            W = strat_weights(Rk, y, strat)
            v = (W[:, 0] * Rk[0]
                 + W[:, 1] * Rk[1])
            av = roc_auc_score(y, v)
            print("")
            print("    AUTO-STRATIFIED on %s"
                  % top_name[:40])
            print("      %.4f   %+.4f vs 3-mod"
                  " block" % (av, av - base3))
            for s_ in np.unique(strat):
                m_ = strat == s_
                if m_.sum():
                    print("      tertile %d:"
                          " weight on EHR+CTPA"
                          " %.2f  (n=%d)"
                          % (s_ + 1,
                             W[m_, 1].mean(),
                             int(m_.sum())))
            print("      a weight that varies"
                  " across tertiles is the"
                  " subgroup")
            print("      hypothesis holding;"
                  " a flat one is it failing",
                  flush=True)
            rows.append({
                "outcome": oc, "stage": "C",
                "combo": "des",
                "method": "auto_strat",
                "mean": av, "sd": np.nan,
                "note": top_name})

        # competence as features for a forest
        C_ = des_weights(P, y, Xfull, "ola")
        M = np.column_stack([Rk[0], Rk[1], C_])
        if imps:
            M = np.column_stack(
                [M, Xall[:, [ALL_NAMES.index(n)
                             for n, _ in
                             imps[:5]]]])
        q = np.zeros(len(y))
        cv = StratifiedGroupKFold(
            n_splits=NFOLD, shuffle=True,
            random_state=42)
        for tr, te in cv.split(M, y, grp):
            mm2 = RandomForestClassifier(
                n_estimators=300, max_depth=4,
                min_samples_leaf=20,
                class_weight=
                "balanced_subsample",
                random_state=42, n_jobs=1)
            mm2.fit(M[tr], y[tr])
            q[te] = mm2.predict_proba(
                M[te])[:, 1]
        av = roc_auc_score(y, q)
        print("")
        print("    des_meta %.4f   %+.4f"
              "   (competence + top features"
              " AS FEATURES in a forest)"
              % (av, av - base3), flush=True)
        rows.append({
            "outcome": oc, "stage": "C",
            "combo": "des",
            "method": "des_meta",
            "mean": av, "sd": np.nan})

    pd.DataFrame(rows).to_csv(DEST,
                              index=False)
    print("")
    print("  %s done in %.1f min  (saved)"
          % (oc, (time.time() - tb) / 60),
          flush=True)

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)
cdf = pd.DataFrame(crows)
if len(cdf):
    cdf.to_csv(CDEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("STAGE A: EVERY COMBINATION, ONE COHORT")
print("  EHR+ECG was previously measured on"
      " 3,501 patients and EHR+CTPA on 1,636,")
print("  so the two were never commensurable")
a_ = r[r["stage"] == "A"]
for meth in METHODS:
    z = a_[a_["method"] == meth]
    if not len(z):
        continue
    print("")
    print("  " + meth)
    print(z.pivot_table(index="combo",
                        columns="outcome",
                        values="mean")
          .round(4).to_string())

print("")
print("BEST METHOD PER COMBINATION")
if len(a_):
    print(a_.pivot_table(index="combo",
                         columns="method",
                         values="mean")
          .round(4).to_string())

print("")
print("DOES THE THIRD MODALITY PAY ON ONE"
      " COHORT?")
for oc in OUTS:
    z = a_[a_["outcome"] == oc]
    if not len(z):
        continue
    for meth in ("late", "block"):
        s = z[z["method"] == meth]

        def gv(c):
            x = s[s["combo"] == c]
            return (x["mean"].iloc[0]
                    if len(x) else np.nan)
        b2 = np.nanmax([gv("ehr+ecg"),
                        gv("ehr+ctpa"),
                        gv("ecg+ctpa")])
        b3 = gv("ehr+ecg+ctpa")
        print("  %-18s %-6s best 2-mod %.4f"
              "   3-mod %.4f   %+.4f"
              % (oc, meth, b2, b3, b3 - b2))

print("")
print("ECG+CTPA, NEVER TESTED WITH A TREE"
      " BEFORE")
z = a_[a_["combo"] == "ecg+ctpa"]
if len(z):
    print(z.pivot_table(index="method",
                        columns="outcome",
                        values="mean")
          .round(4).to_string())

print("")
print("DOES PCA HELP WITH THREE BLOCKS?")
for oc in OUTS:
    z = a_[(a_["outcome"] == oc)
           & (a_["combo"] == "ehr+ecg+ctpa")]
    if not len(z):
        continue

    def gm(m):
        x = z[z["method"] == m]
        return (x["mean"].iloc[0]
                if len(x) else np.nan)
    print("  %-18s rf_raw %.4f   rf_pca %.4f"
          "   rf_ccaaug %.4f   block %.4f"
          % (oc, gm("rf_raw"), gm("rf_pca"),
             gm("rf_ccaaug"), gm("block")))

print("")
print("=" * 74)
print("STAGE B: FUSING THE TWO-MODALITY MODELS")
b_ = r[r["stage"] == "B"]
if len(b_):
    print(b_.pivot_table(index="method",
                         columns="outcome",
                         values="mean")
          .round(4).to_string())

print("")
print("STAGE C: COMPETENCE-AWARE FUSION")
c_ = r[r["stage"] == "C"]
if len(c_):
    print(c_.pivot_table(index="method",
                         columns="outcome",
                         values="mean")
          .round(4).to_string())
    print("")
    print("  ola_ehr and lca_ehr define"
          " similarity on 28 of %d columns;"
          % len(ALL_NAMES))
    print("  ola_full and lca_full use all of"
          " them. If full beats ehr, the")
    print("  competence regions are not"
          " EHR-shaped.")

print("")
print("FEATURES THAT SEPARATE COMPETENCE,"
      " ACROSS OUTCOMES")
if len(cdf):
    agg = cdf.groupby("feature")[
        "importance"].agg(["mean", "count"])
    agg = agg.sort_values("mean",
                          ascending=False)
    print(agg.head(15).round(4).to_string())
    print("")
    print("  a feature appearing in several"
          " outcomes with high importance is")
    print("  a real competence boundary rather"
          " than one outcome's noise")

print("")
print("EVERYTHING AGAINST THE 3-MODALITY BLOCK"
      " FOREST")
for oc in OUTS:
    base = a_[(a_["outcome"] == oc)
              & (a_["combo"] == "ehr+ecg+ctpa")
              & (a_["method"] == "block")]
    alt = r[(r["outcome"] == oc)
            & (r["stage"].isin(["B", "C"]))]
    if not len(base) or not len(alt):
        continue
    bb = base["mean"].iloc[0]
    x = alt.loc[alt["mean"].idxmax()]
    print("  %-18s 3-mod block %.4f"
          "   best alt %-14s %.4f   %+.4f"
          % (oc, bb, x["method"], x["mean"],
             x["mean"] - bb))
print("")
print("saved", DEST, r.shape)
keep_awake(False)