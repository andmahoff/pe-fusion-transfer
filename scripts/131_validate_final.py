"""Validation of the final three-modality model:
plateau map, nested cross-validation and
decision curves.

FINAL CONFIGURATION, from block_final.csv
  draws 14+4+8, leaf 10 (100 for cv_first),
  balanced class weights, ccp 0, row bootstrap
  1.0, no OOB weighting, k=10, reg 0.5, and no
  rotation (every paired bootstrap interval
  spanned zero). 600 trees, since 2400 gave no
  gain.

A  PLATEAU MAP. Several settings in block_final
   won by less than one seed SD, so the near-tied
   settings are rerun at ten seeds and
   bootstrapped against the winner.
B  NESTED CROSS-VALIDATION. The draw and leaf
   selection is repeated inside each outer fold,
   as TRIPOD+AI asks for tuning steps, and the
   stage reports how often the inner search picks
   each configuration.
D  DECISION CURVES. Net benefit across threshold
   probabilities, after Platt calibration, since
   the raw forest is under-confident (slopes 1.49
   to 2.03).

Stage C, a bootstrap stability check, is not run:
it would take most of the runtime, and stage A
measures the same flatness more cheaply.

Parallel across seeds. Timing probe first.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\validate_final.csv
  data\\processed\\validate_final_dca.csv
  results\\validate_final_log.txt
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
NINNER = 4
NJOBS = max(1, min(10, (os.cpu_count() or 4)
                   - 2))
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CT_CS = list(np.logspace(-4, 4, 10))
K0, REG0 = 10, 0.5
NTREE = 600
FINAL = {"draws": (14, 4, 8), "leaf": 10,
         "cw": "balanced", "ccp": 0.0,
         "boot": 1.0, "oob_w": False,
         "k": K0, "reg": REG0,
         "ntree": NTREE}
LEAF_BY = {"cv_first": 100}
PLATEAU_DRAWS = [(14, 4, 8), (14, 8, 16),
                 (20, 8, 16), (20, 12, 24)]
PLATEAU_REG = [0.1, 0.5]
INNER_DRAWS = [(14, 4, 8), (14, 8, 16),
               (20, 8, 16)]
INNER_LEAF_MAIN = [5, 10, 25]
INNER_LEAF_CV = [50, 100, 150]
THRESH = np.arange(0.01, 0.51, 0.01)
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(
    PROC, "validate_final.csv")
DCA = os.path.join(
    PROC, "validate_final_dca.csv")
S130 = {"death_30d_inhosp": 0.9147,
        "death_30d": 0.9054,
        "composite_30d": 0.8641,
        "cv_first": 0.8284}

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


class BlockForest:
    def __init__(self, n_estimators=600,
                 draws=(14, 4, 8),
                 bounds=(38, 312), leaf=10,
                 depth=None, cw="balanced",
                 ccp=0.0, boot=1.0,
                 seed=42):
        self.n = n_estimators
        self.draws = draws
        self.bounds = bounds
        self.leaf = leaf
        self.depth = depth
        self.cw = cw
        self.ccp = ccp
        self.boot = boot
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        cuts = ([0] + [int(min(b, p))
                       for b in self.bounds]
                + [p])
        blk = [np.arange(cuts[i], cuts[i + 1])
               for i in range(len(cuts) - 1)]
        nb = max(20, int(round(self.boot * n)))
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
            t.fit(X[np.ix_(rows, cols)],
                  y[rows])
            self.sel_.append(cols)
            self.tr_.append(t)
        return self

    def predict_proba(self, X):
        out = np.zeros(len(X))
        if not self.tr_:
            out[:] = 0.5
            return np.column_stack(
                [1 - out, out])
        for cols, t in zip(self.sel_,
                           self.tr_):
            out += t.predict_proba(
                X[:, cols])[:, 1]
        out /= len(self.tr_)
        return np.column_stack([1 - out, out])


def fit_pred(Bt, Be, ytr, cfg, seed):
    Zt, Ze, bd = ccaaug(Bt, Be, cfg["k"],
                        cfg["reg"])
    bf = BlockForest(
        n_estimators=cfg["ntree"],
        draws=cfg["draws"], bounds=bd,
        leaf=cfg["leaf"], cw=cfg["cw"],
        ccp=cfg["ccp"], boot=cfg["boot"],
        seed=seed)
    bf.fit(Zt, ytr)
    return bf.predict_proba(Ze)[:, 1]


def oof(Xd, MODS, y, grp, seed, cfg):
    """Flat out-of-fold: the configuration is
    FIXED, so this carries the selection
    optimism that nested CV removes."""
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
        p[te] = fit_pred(Bt, Be, y[tr], cfg,
                         seed)
    return p


def nested_oof(Xd, MODS, y, grp, seed, base,
               draws_grid, leaf_grid):
    """NESTED: draws and leaf are re-selected by
    an INNER cross-validation inside each outer
    training fold, so the outer prediction never
    sees a choice made with its own labels.

    The returned picks are the per-fold
    selections, which show how stable the
    tuning is without needing a bootstrap."""
    p = np.zeros(len(y))
    picks = []
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        ytr, gtr = y[tr], grp[tr]
        best, bcfg = -1.0, None
        icv = StratifiedGroupKFold(
            n_splits=NINNER, shuffle=True,
            random_state=seed + 1)
        splits = list(icv.split(
            np.zeros((len(ytr), 1)), ytr, gtr))
        for dr in draws_grid:
            for lf in leaf_grid:
                c = dict(base)
                c["draws"], c["leaf"] = dr, lf
                q = np.zeros(len(ytr))
                for i2, j2 in splits:
                    Bt, Be = [], []
                    for m_ in MODS:
                        a, b = prep_fold(
                            Xd[m_][tr][i2],
                            Xd[m_][tr][j2])
                        Bt.append(a)
                        Be.append(b)
                    q[j2] = fit_pred(
                        Bt, Be, ytr[i2], c,
                        seed)
                try:
                    s = roc_auc_score(ytr, q)
                except Exception:
                    continue
                if s > best:
                    best, bcfg = s, (dr, lf)
        if bcfg is None:
            bcfg = (base["draws"],
                    base["leaf"])
        picks.append(bcfg)
        c = dict(base)
        c["draws"], c["leaf"] = bcfg
        Bt, Be = [], []
        for m_ in MODS:
            a, b = prep_fold(Xd[m_][tr],
                             Xd[m_][te])
            Bt.append(a)
            Be.append(b)
        p[te] = fit_pred(Bt, Be, ytr, c, seed)
    return p, picks


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


def net_benefit(p, y, th):
    """Net benefit = TP/n - FP/n * th/(1-th).

    Needs CALIBRATED probabilities, since the
    threshold is on the probability scale rather
    than the ranking."""
    ok = np.isfinite(p)
    p2, y2 = p[ok], y[ok]
    n = len(y2)
    out = []
    for t in th:
        pos = p2 >= t
        tp = float((pos & (y2 == 1)).sum())
        fp = float((pos & (y2 == 0)).sum())
        out.append(tp / n
                   - (fp / n) * (t / (1 - t)))
    return np.array(out)


def nb_all(y, th):
    prev = y.mean()
    return np.array([prev - (1 - prev)
                     * (t / (1 - t))
                     for t in th])


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
print("  no further tuning; stage C (bootstrap")
print("  stability) is not run")
print("  Tree count cut 2400 -> %d (the"
      " marginal spread was 0.0005 against a"
      % NTREE)
print("  seed SD of 0.0042).", flush=True)

# ---------- TIMING PROBE ----------
print("")
print("TIMING PROBE")
Xp = np.random.rand(len(D), 400)
yp = (np.random.rand(len(D)) > 0.90) \
    .astype(int)
t = time.time()
BlockForest(n_estimators=NTREE, leaf=10,
            bounds=(38, 312),
            seed=42).fit(Xp, yp)
t_fit = time.time() - t
nA = len(PLATEAU_DRAWS) * len(PLATEAU_REG)
nInner = len(INNER_DRAWS) * len(INNER_LEAF_MAIN)
stA = (nA * t_fit * NFOLD * len(SEEDS10)
       * len(OUTS) / 60.0)
stB = ((nInner * NINNER + 1) * t_fit * NFOLD
       * len(SEEDS) * len(OUTS) / 60.0)
print("  one forest fit (%d trees): %5.1f s"
      % (NTREE, t_fit))
print("")
print("  stage A %.0f min, stage B %.0f min,"
      " serial" % (stA, stB))
print("  stage D reuses stage B's predictions"
      " and is free")
print("  with %d-way parallelism about %.0f"
      " min" % (NJOBS, (stA + stB) / NJOBS))
print("  Ctrl+C now if too long. Results save"
      " after every stage.")
time.sleep(5)

rows, dca_rows = [], []
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
    base = dict(FINAL)
    base["leaf"] = LEAF_BY.get(oc,
                               FINAL["leaf"])
    lg = (INNER_LEAF_CV if oc == "cv_first"
          else INNER_LEAF_MAIN)
    ref = S130.get(oc, np.nan)
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d   previously"
          " reported %.4f"
          % (oc, len(y), int(y.sum()), ref))

    # ---- STAGE A: the plateau ----
    print("")
    print("  STAGE A: PLATEAU MAP")
    print("  draws 14+4+8 beat 14+8+16 by"
          " 0.0008 against SDs of 0.0014 to")
    print("  0.0055, and scripts 117, 128 and"
          " 130 each chose a different draw")
    print("  %-10s %-5s %9s %9s"
          % ("draws", "reg", "AUC", "SD"))
    cells = []
    for dr in PLATEAU_DRAWS:
        for rg in PLATEAU_REG:
            c = dict(base)
            c["draws"], c["reg"] = dr, rg
            try:
                ps = par(lambda s, c=c:
                         oof(Xd, MODS, y, grp,
                             s, c), SEEDS10)
            except Exception as exc:
                print("  %-10s %-5.1f FAILED %s"
                      % ("%d+%d+%d" % dr, rg,
                         repr(exc)[:30]))
                continue
            aa = np.array([roc_auc_score(y, p)
                           for p in ps])
            cells.append((aa.mean(),
                          aa.std(ddof=1), dr,
                          rg, ps[0]))
            print("  %-10s %-5.1f %9.4f %9.4f"
                  % ("%d+%d+%d" % dr, rg,
                     aa.mean(),
                     aa.std(ddof=1)),
                  flush=True)
            rows.append({
                "outcome": oc, "stage": "A",
                "draws": "%d+%d+%d" % dr,
                "reg": rg, "mean": aa.mean(),
                "sd": aa.std(ddof=1)})
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)
    if cells:
        cells.sort(key=lambda z: -z[0])
        top = cells[0]
        ntied = 0
        for c2 in cells:
            gn, lo, hi, _ = f.boot_diff(
                y, _rank(top[4]),
                _rank(c2[4]), grp)
            tied = not (lo > 0 or hi < 0)
            ntied += int(tied)
            rows.append({
                "outcome": oc, "stage": "A2",
                "draws": "%d+%d+%d" % c2[2],
                "reg": c2[3], "mean": c2[0],
                "boot_diff": gn, "lo": lo,
                "hi": hi,
                "sig": int(not tied)})
        print("")
        print("  best %s reg=%.1f  %.4f"
              % ("%d+%d+%d" % top[2], top[3],
                 top[0]))
        print("  %d of %d settings are"
              " STATISTICALLY TIED with it"
              % (ntied, len(cells)))
        print("  spread across the plateau:"
              " %.4f"
              % (cells[0][0] - cells[-1][0]),
              flush=True)
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)

    # ---- STAGE B: nested CV ----
    print("")
    print("  STAGE B: NESTED CROSS-VALIDATION")
    print("  draws and leaf re-selected by an"
          " INNER CV inside each outer fold, so")
    print("  the outer prediction never sees a"
          " choice made with its own labels")
    pn, af = None, None
    try:
        res = par(lambda s:
                  nested_oof(Xd, MODS, y, grp,
                             s, base,
                             INNER_DRAWS, lg))
        an = np.array([roc_auc_score(y, r[0])
                       for r in res])
        flat_picks = [p for r in res
                      for p in r[1]]
        pn = res[0][0]
        ps = par(lambda s: oof(Xd, MODS, y,
                               grp, s, base))
        af = np.array([roc_auc_score(y, p)
                       for p in ps])
        print("    flat (tuned on these folds)"
              "   %.4f (SD %.4f)"
              % (af.mean(), af.std(ddof=1)))
        print("    NESTED                      "
              "  %.4f (SD %.4f)"
              % (an.mean(), an.std(ddof=1)))
        print("    OPTIMISM                    "
              " %+.4f"
              % (af.mean() - an.mean()))
        print("    script 125 measured this at"
              " +0.0048 to +0.0058 for the")
        print("    late-fusion weights alone",
              flush=True)
        rows.append({
            "outcome": oc, "stage": "B",
            "name": "flat", "mean": af.mean(),
            "sd": af.std(ddof=1)})
        rows.append({
            "outcome": oc, "stage": "B",
            "name": "nested",
            "mean": an.mean(),
            "sd": an.std(ddof=1),
            "boot_diff": af.mean() - an.mean()})
        cnt = pd.Series(
            ["%d+%d+%d/lf%d" % (a[0] + (a[1],))
             for a in flat_picks]).value_counts()
        print("")
        print("    the inner search picked, over"
              " %d selections:" % len(flat_picks))
        for k2, v2 in cnt.items():
            print("      %-18s %2d (%3.0f%%)"
                  % (k2, v2,
                     100.0 * v2
                     / len(flat_picks)))
        print("    %d distinct configurations;"
              " the modal one wins %.0f%%"
              % (len(cnt),
                 100.0 * cnt.iloc[0]
                 / len(flat_picks)))
        print("    a low modal share means the"
              " surface is flat and the chosen")
        print("    configuration is close to"
              " arbitrary", flush=True)
        rows.append({
            "outcome": oc, "stage": "B2",
            "name": "selection",
            "mean": float(len(cnt)),
            "modal_pct": 100.0 * cnt.iloc[0]
            / len(flat_picks),
            "modal_cfg": cnt.index[0]})
    except Exception as exc:
        print("  FAILED %s" % repr(exc)[:60])
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)

    # ---- STAGE D: decision curves ----
    print("")
    print("  STAGE D: DECISION CURVES")
    print("  net benefit needs calibrated"
          " probabilities, not just a ranking,")
    print("  and the raw forest is"
          " UNDER-confident (slopes 1.5 to 2.0)")
    if pn is None:
        pn = par(lambda s: oof(Xd, MODS, y,
                               grp, s, base))[0]
    pc = platt_oof(pn, y, grp)
    b0, s0 = calib(pn, y)
    b1, s1 = calib(pc, y)
    print("    slope %.3f -> %.3f   Brier"
          " %.5f -> %.5f" % (s0, s1, b0, b1))
    nbm = net_benefit(pc, y, THRESH)
    nbt = nb_all(y, THRESH)
    nbn = np.zeros(len(THRESH))
    for i, t2 in enumerate(THRESH):
        dca_rows.append({
            "outcome": oc, "threshold": t2,
            "nb_model": nbm[i],
            "nb_treat_all": nbt[i],
            "nb_treat_none": nbn[i],
            "prevalence": float(y.mean())})
    over = nbm - np.maximum(nbt, nbn)
    ok = over > 0
    lo_t = hi_t = np.nan
    if ok.any():
        lo_t, hi_t = THRESH[ok][0], THRESH[ok][-1]
        print("    beats both default strategies"
              " from threshold %.2f to %.2f"
              % (lo_t, hi_t))
        print("    peak advantage %.4f at %.2f"
              % (over.max(),
                 THRESH[over.argmax()]))
        print("    at that threshold this is"
              " about %.1f extra true positives"
              % (over.max() * len(y)))
        print("    per %d patients, with no"
              " increase in false positives"
              % len(y))
    else:
        print("    NEVER beats treat-all or"
              " treat-none at any threshold")
    print("    prevalence %.3f. The curve"
          " cannot say which threshold to use."
          % y.mean(), flush=True)
    rows.append({
        "outcome": oc, "stage": "D",
        "name": "dca", "slope": s0,
        "slope_platt": s1, "brier": b0,
        "brier_platt": b1, "lo": lo_t,
        "hi": hi_t,
        "mean": (over.max() if ok.any()
                 else 0.0)})

    pd.DataFrame(rows).to_csv(DEST,
                              index=False)
    pd.DataFrame(dca_rows).to_csv(DCA,
                                  index=False)
    print("")
    print("  %s done in %.1f min  (saved)"
          % (oc, (time.time() - tb) / 60),
          flush=True)

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)
pd.DataFrame(dca_rows).to_csv(DCA,
                              index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))

print("")
print("=" * 74)
print("STAGE A: HOW BIG IS THE PLATEAU?")
a2 = r[r["stage"] == "A2"]
if len(a2):
    for oc in OUTS:
        z = a2[a2["outcome"] == oc]
        if not len(z):
            continue
        tied = int((z["sig"] == 0).sum())
        print("  %-18s %d of %d settings tied"
              " with the best; spread %.4f"
              % (oc, tied, len(z),
                 z["mean"].max()
                 - z["mean"].min()))
    print("")
    print("  a large tied set means the choice"
          " of draws and reg is arbitrary, and")
    print("  the write-up should report a"
          " plateau rather than an optimum")
a_ = r[r["stage"] == "A"]
if len(a_):
    print("")
    print("  marginal by draws")
    print(a_.groupby("draws")["mean"].mean()
          .round(4).to_string())
    print("  marginal by reg")
    print(a_.groupby("reg")["mean"].mean()
          .round(4).to_string())

print("")
print("=" * 74)
print("STAGE B: NESTED CV AND OPTIMISM")
b_ = r[r["stage"] == "B"]
if len(b_):
    print(b_.pivot_table(index="name",
                         columns="outcome",
                         values="mean")
          .round(4).to_string())
    print("")
    print("  optimism, flat minus nested")
    z = b_[b_["name"] == "nested"]
    for _, x in z.iterrows():
        print("    %-18s %+.4f"
              % (x["outcome"], x["boot_diff"]))
    print("")
    print("  TRIPOD+AI asks whether all model"
          " building steps including")
    print("  hyperparameter tuning were replayed"
          " during internal evaluation. The")
    print("  nested row replays them; the flat"
          " row does not.")

print("")
print("SELECTION STABILITY, FROM THE INNER"
      " SEARCH")
b2 = r[r["stage"] == "B2"]
if len(b2):
    print(b2[["outcome", "mean", "modal_pct",
              "modal_cfg"]].round(1)
          .to_string(index=False))
    print("")
    print("  'mean' is the number of distinct"
          " configurations chosen. A modal")
    print("  share near 100 means stable tuning;"
          " near 30 means the surface is")
    print("  flat and the selection is close to"
          " arbitrary.")

print("")
print("STAGE D: DECISION CURVES")
d_ = r[r["stage"] == "D"]
if len(d_):
    print(d_[["outcome", "slope",
              "slope_platt", "brier",
              "brier_platt", "lo", "hi",
              "mean"]].round(4)
          .to_string(index=False))
    print("")
    print("  lo and hi bound the thresholds"
          " where the model beats both default")
    print("  strategies; 'mean' is the peak net"
          " benefit advantage. Full curves are")
    print("  in the dca CSV, ready to plot.")

print("")
print("=" * 74)
print("THE HONEST FIGURES")
if len(b_):
    z = b_[b_["name"] == "nested"]
    for _, x in z.iterrows():
        print("  %-18s nested %.4f   previously"
              " reported %.4f   %+.4f"
              % (x["outcome"], x["mean"],
                 S130.get(x["outcome"], np.nan),
                 x["mean"] - S130.get(
                     x["outcome"], np.nan)))
    print("")
    print("  these are the numbers to report.")
print("")
print("saved", DEST, r.shape)
print("saved", DCA)
keep_awake(False)