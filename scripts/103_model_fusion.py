"""Combines fusion strategies as an ensemble,
and measures how similarly they rank patients.

The dissertation reported early, intermediate and
late fusion at 0.8715, 0.8718 and 0.8709. Similar
AUROCs can come from models that rank patients
the same way, where combining adds nothing, or
from models that rank them differently.

PASS 1 builds the base models once, on one
outcome at one seed, and prints their similarity
structure. Set CORR_ONLY = True to stop there.
PASS 2 runs the combiners on every outcome.

COMBINERS
  simple mean of ranks
  weighted rank grid, the usual late fusion but
    over models
  Super Learner: non-negative least squares
  stacking with a logistic meta-learner
  diversity-pruned: the best model plus the
    member least correlated with it

Also tests whether diversity between models
predicts combination gain, the gap rule applied
to models rather than modalities.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\model_fusion.csv
  data\\processed\\model_diversity.csv
  results\\model_fusion_log.txt
"""
import os
import sys
import time
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as st
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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

# set True to stop after the correlation pass
CORR_ONLY = False

PROC, FIG = f.PROC, f.FIG
RAW = os.path.join("data", "raw", "ecg")
SEEDS = [42, 7, 13]
NFOLD = 5
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
EPOCHS, BATCH, LR, WD = 150, 64, 1e-3, 1e-4
DROP = 0.8
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(PROC,
                    "model_fusion.csv")
DDEST = os.path.join(PROC,
                     "model_diversity.csv")

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
torch.set_num_threads(max(1, os.cpu_count() - 1))


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


def fit_pred(Xtr, ytr, Xte, gtr=None):
    im = SimpleImputer(strategy="median")
    a = im.fit_transform(Xtr)
    b = im.transform(Xte)
    sc = StandardScaler()
    a, b = sc.fit_transform(a), sc.transform(b)
    best, bc = -1.0, 1.0
    try:
        kk = min(3, max(2, int(ytr.sum()) // 10))
        icv = StratifiedGroupKFold(
            n_splits=kk, shuffle=True,
            random_state=42)
        g = (gtr if gtr is not None
             else np.arange(len(ytr)))
        for c in CS:
            q = np.zeros(len(ytr))
            for t2, v2 in icv.split(a, ytr, g):
                m = LogisticRegression(
                    C=c, max_iter=3000)
                m.fit(a[t2], ytr[t2])
                q[v2] = m.predict_proba(
                    a[v2])[:, 1]
            s = roc_auc_score(ytr, q)
            if s > best:
                best, bc = s, c
    except Exception:
        bc = 1.0
    m = LogisticRegression(C=bc, max_iter=3000)
    m.fit(a, ytr)
    return m.predict_proba(b)[:, 1]


class Enc(nn.Module):
    def __init__(self, nin, w, lat, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, w), nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(w, lat))

    def forward(self, x):
        return self.net(x)


class JointNet(nn.Module):
    def __init__(self, na, nb, drop,
                 w=32, lat=16):
        super().__init__()
        self.ea = Enc(na, w, lat, drop)
        self.eb = Enc(nb, w, lat, drop)
        self.head = nn.Sequential(
            nn.Linear(2 * lat, w), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(w, 1))

    def forward(self, xa, xb):
        return self.head(torch.cat(
            [self.ea(xa), self.eb(xb)],
            dim=1)).squeeze(1)


def oof_inter(Xa, Xb, y, grp, seed):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        At, Ae = prep_fold(Xa[tr], Xa[te])
        Bt, Be = prep_fold(Xb[tr], Xb[te])
        torch.manual_seed(seed)
        m = JointNet(At.shape[1], Bt.shape[1],
                     DROP)
        opt = torch.optim.Adam(
            m.parameters(), lr=LR,
            weight_decay=WD)
        lf = nn.BCEWithLogitsLoss()
        ta = torch.tensor(At,
                          dtype=torch.float32)
        tb = torch.tensor(Bt,
                          dtype=torch.float32)
        ty = torch.tensor(y[tr],
                          dtype=torch.float32)
        m.train()
        for _ in range(EPOCHS):
            idx = torch.randperm(len(tr))
            for i in range(0, len(tr), BATCH):
                b_ = idx[i:i + BATCH]
                if len(b_) < 4:
                    continue
                opt.zero_grad()
                lf(m(ta[b_], tb[b_]),
                   ty[b_]).backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            o = m(torch.tensor(
                Ae, dtype=torch.float32),
                torch.tensor(
                    Be, dtype=torch.float32))
        p[te] = torch.sigmoid(o).numpy()
    return p


def oof_raw(X, y, grp, seed):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        zt, ze = prep_fold(X[tr], X[te])
        p[te] = fit_pred(zt, y[tr], ze,
                         grp[tr])
    return p


def oof_pca(X, y, grp, k, seed):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        zt, ze = prep_fold(X[tr], X[te])
        kk = int(min(k, zt.shape[1] - 1,
                     len(tr) - 1))
        pc = PCA(n_components=kk,
                 random_state=42)
        p[te] = fit_pred(
            pc.fit_transform(zt), y[tr],
            pc.transform(ze), grp[tr])
    return p


def wgrid(ps, y, step=0.05):
    k = len(ps)
    if k == 1:
        return ps[0], (1.0,)
    n = int(round(1.0 / step))

    def rec(m, rem):
        if m == 1:
            yield (rem,)
            return
        for i in range(rem + 1):
            for t in rec(m - 1, rem - i):
                yield (i,) + t
    best, bv, bw = -1.0, ps[0], None
    for w in rec(k, n):
        w = tuple(x * step for x in w)
        v = sum(wi * p for wi, p in zip(w, ps))
        a = roc_auc_score(y, v)
        if a > best:
            best, bv, bw = a, v, w
    return bv, bw


def super_learner(ps, y):
    """Non-negative least squares weights, then
    normalised: the Super Learner form."""
    A = np.column_stack(ps)
    try:
        w, _ = nnls(A, y.astype(float))
    except Exception:
        w = np.ones(len(ps))
    if w.sum() <= 1e-9:
        w = np.ones(len(ps)) / len(ps)
    else:
        w = w / w.sum()
    return A @ w, tuple(w)


def diversity(pa, pb, y):
    """Pairwise diversity at the median
    threshold, as used in diversity-guided
    stacking: disagreement, Yule's Q, kappa,
    and the joint-miss rate on events."""
    a = (pa >= np.median(pa)).astype(int)
    b = (pb >= np.median(pb)).astype(int)
    ca = (a == y).astype(int)
    cb = (b == y).astype(int)
    n11 = float(((ca == 1) & (cb == 1)).sum())
    n00 = float(((ca == 0) & (cb == 0)).sum())
    n10 = float(((ca == 1) & (cb == 0)).sum())
    n01 = float(((ca == 0) & (cb == 1)).sum())
    n = float(len(y))
    dis = (n10 + n01) / n
    den = n11 * n00 + n01 * n10
    q = ((n11 * n00 - n01 * n10) / den
         if den > 1e-9 else 0.0)
    po = (n11 + n00) / n
    pe = (((n11 + n10) * (n11 + n01)
           + (n01 + n00) * (n10 + n00))
          / (n * n))
    kap = ((po - pe) / (1 - pe)
           if abs(1 - pe) > 1e-9 else 0.0)
    ev = y == 1
    jm = (float(((pa[ev] < np.median(pa))
                 & (pb[ev] < np.median(pb)))
                .mean()) if ev.sum() else
          np.nan)
    return dis, q, kap, jm


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
    mim_ehr[["hadm_id"] + EH],
    on="hadm_id", how="inner").merge(
    mm[["hadm_id"] + CT], on="hadm_id",
    how="inner")
ECGC = LG + MC
print("cohort:", len(D))
print("  EHR %d   ECG %d   CTPA %d"
      % (len(EH), len(ECGC), len(CT)),
      flush=True)

# ======== PASS 1: CORRELATION FIRST =========
print("")
print("#" * 74)
print("PASS 1: HOW SIMILAR ARE THE MODELS?")
print("  measured on %s at seed 42, BEFORE any"
      " combining. If the strategies" % OUTS[0])
print("  correlate above 0.95 they are the same"
      " model in different clothes and")
print("  the rest of this script answers a"
      " question with no content.")
print("#" * 74, flush=True)

t0 = time.time()
d0, y0 = f.labels(D, OUTS[0])
g0 = d0["subject_id"].values
Xe0 = d0[EH].values.astype(float)
Xg0 = d0[ECGC].values.astype(float)
Xc0 = d0[CT].values.astype(float)
Q = {}
for nm, fn in (
        ("ehr", lambda: oof_raw(
            Xe0, y0, g0, 42)),
        ("ecg", lambda: oof_raw(
            Xg0, y0, g0, 42)),
        ("ctpa", lambda: oof_raw(
            Xc0, y0, g0, 42)),
        ("early", lambda: oof_raw(
            np.column_stack(
                [Xe0, Xg0, Xc0]), y0, g0,
            42)),
        ("pca10", lambda: oof_pca(
            np.column_stack([Xe0, Xc0]),
            y0, g0, 10, 42)),
        ("inter", lambda: oof_inter(
            Xe0, Xc0, y0, g0, 42))):
    try:
        Q[nm] = _rank(fn())
        print("  built %-6s  AUC %.4f"
              % (nm, roc_auc_score(y0, Q[nm])),
              flush=True)
    except Exception as exc:
        print("  %s FAILED %s"
              % (nm, repr(exc)[:50]))
if all(k in Q for k in ("ehr", "ecg", "ctpa")):
    v, _ = wgrid([Q["ehr"], Q["ecg"],
                  Q["ctpa"]], y0)
    Q["late"] = _rank(v)
    print("  built late    AUC %.4f"
          % roc_auc_score(y0, Q["late"]),
          flush=True)

print("")
print("  %-8s %-8s %7s %7s %7s %7s"
      % ("a", "b", "spear", "err_r", "kappa",
         "disag"))
pairs = []
for a, b in itertools.combinations(
        list(Q.keys()), 2):
    sr = float(st.spearmanr(
        Q[a], Q[b]).statistic)
    er = float(st.pearsonr(
        y0 - Q[a], y0 - Q[b]).statistic)
    dis, qq, kap, _ = diversity(Q[a], Q[b], y0)
    print("  %-8s %-8s %7.3f %7.3f %7.3f"
          " %7.3f"
          % (a, b, sr, er, kap, dis),
          flush=True)
    pairs.append((a, b, sr))

strat = [x for x in pairs
         if x[0] in ("early", "inter", "late")
         and x[1] in ("early", "inter",
                      "late")]
print("")
print("  VERDICT  (%.1f min)"
      % ((time.time() - t0) / 60))
if strat:
    ms = float(np.mean([x[2] for x in strat]))
    print("  mean Spearman between the three"
          " strategies: %.3f" % ms)
    for a, b, s_ in strat:
        print("    %-6s vs %-6s  %.3f"
              % (a, b, s_))
    print("")
    if ms >= 0.95:
        print("  -> they are the same model in"
              " different clothes. That")
        print("     explains the dissertation's"
              " 0.8715 / 0.8718 / 0.8709")
        print("     and why mechanism choice"
              " turned out irrelevant.")
    elif ms >= 0.85:
        print("  -> largely redundant. Expect a"
              " combination gain near zero.")
    else:
        print("  -> they rank patients genuinely"
              " differently, so the")
        print("     combining below is worth"
              " reading.")
if pairs:
    lo = min(pairs, key=lambda x: x[2])
    print("  least similar pair overall: %s and"
          " %s at %.3f"
          % (lo[0], lo[1], lo[2]))

if CORR_ONLY:
    print("")
    print("CORR_ONLY is set, stopping here."
          " Set it to False and rerun for the")
    print("full combining experiment.")
    keep_awake(False)
    sys.exit(0)

print("")
print("  continuing to the full run ...",
      flush=True)

# ============ PASS 2: THE FULL RUN ==========
rows, drows = [], []

for oc in OUTS:
    if oc not in D.columns:
        continue
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    if y.sum() < 25:
        continue
    Xe = d[EH].values.astype(float)
    Xg = d[ECGC].values.astype(float)
    Xc = d[CT].values.astype(float)
    Xall = np.column_stack([Xe, Xg, Xc])
    Xec = np.column_stack([Xe, Xc])
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())),
          flush=True)

    P = {}
    print("  building base models ...",
          flush=True)
    for nm, fn in (
            ("ehr", lambda s: oof_raw(
                Xe, y, grp, s)),
            ("ecg", lambda s: oof_raw(
                Xg, y, grp, s)),
            ("ctpa", lambda s: oof_raw(
                Xc, y, grp, s)),
            ("early", lambda s: oof_raw(
                Xall, y, grp, s)),
            ("pca10", lambda s: oof_pca(
                Xec, y, grp, 10, s)),
            ("inter", lambda s: oof_inter(
                Xe, Xc, y, grp, s))):
        try:
            P[nm] = [_rank(fn(s))
                     for s in SEEDS]
        except Exception as exc:
            print("    %s FAILED %s"
                  % (nm, repr(exc)[:50]))
    if all(k in P for k in
           ("ehr", "ecg", "ctpa")):
        lf_ = []
        for i in range(len(SEEDS)):
            v, _ = wgrid([P["ehr"][i],
                          P["ecg"][i],
                          P["ctpa"][i]], y)
            lf_.append(_rank(v))
        P["late"] = lf_

    names = list(P.keys())
    print("  %d base models" % len(names))
    print("")
    print("  %-8s %8s %8s"
          % ("model", "AUC", "SD"))
    aucs = {}
    for nm in names:
        v = np.array([roc_auc_score(y, p)
                      for p in P[nm]])
        aucs[nm] = v.mean()
        print("  %-8s %8.4f %8.4f"
              % (nm, v.mean(),
                 v.std(ddof=1)), flush=True)
        rows.append({
            "outcome": oc, "method": nm,
            "kind": "base", "mean": v.mean(),
            "sd": v.std(ddof=1),
            "vs_best": np.nan,
            "members": "", "weights": ""})

    print("")
    print("  PAIRWISE SIMILARITY  (seed 42)")
    print("  %-8s %-8s %7s %7s %7s %7s %7s"
          % ("a", "b", "spear", "err_r",
             "disag", "YuleQ", "kappa"))
    for a, b in itertools.combinations(
            names, 2):
        pa, pb = P[a][0], P[b][0]
        sr = float(st.spearmanr(
            pa, pb).statistic)
        er = float(st.pearsonr(
            y - pa, y - pb).statistic)
        dis, q, kap, jm = diversity(pa, pb, y)
        print("  %-8s %-8s %7.3f %7.3f %7.3f"
              " %7.3f %7.3f"
              % (a, b, sr, er, dis, q, kap),
              flush=True)
        drows.append({
            "outcome": oc, "a": a, "b": b,
            "spearman": sr, "err_corr": er,
            "disagree": dis, "yule_q": q,
            "kappa": kap, "joint_miss": jm,
            "auc_a": aucs[a], "auc_b": aucs[b],
            "gap": abs(aucs[a] - aucs[b])})

    print("")
    print("  COMBINING THE MODELS")
    best_base = max(aucs, key=aucs.get)
    bb = aucs[best_base]
    print("    best single: %s %.4f"
          % (best_base, bb))
    print("    %-20s %8s %9s"
          % ("combiner", "AUC", "vs best"))
    combos = {
        "all": names,
        "strategies": [n for n in
                       ("early", "inter",
                        "late") if n in P],
        "top3": sorted(aucs, key=aucs.get,
                       reverse=True)[:3]}
    for cname, mem in combos.items():
        if len(mem) < 2:
            continue
        for meth in ("mean", "grid",
                     "superlearner", "stack"):
            aa, wlast = [], None
            try:
                for i in range(len(SEEDS)):
                    ps = [P[m][i] for m in mem]
                    if meth == "mean":
                        v = np.mean(ps, axis=0)
                    elif meth == "grid":
                        v, wlast = wgrid(ps, y)
                    elif meth == "superlearner":
                        v, wlast = \
                            super_learner(ps, y)
                    else:
                        Xm = np.column_stack(ps)
                        v = np.zeros(len(y))
                        cv = StratifiedGroupKFold(
                            n_splits=NFOLD,
                            shuffle=True,
                            random_state=SEEDS[i])
                        for tr, te in cv.split(
                                Xm, y, grp):
                            md = LogisticRegression(
                                C=1.0,
                                max_iter=2000)
                            md.fit(Xm[tr], y[tr])
                            v[te] = \
                                md.predict_proba(
                                    Xm[te])[:, 1]
                    aa.append(
                        roc_auc_score(y, v))
            except Exception as exc:
                print("    %s/%s FAILED %s"
                      % (cname, meth,
                         repr(exc)[:40]))
                continue
            v_ = np.array(aa)
            print("    %-20s %8.4f %+9.4f"
                  % ("%s/%s" % (cname, meth),
                     v_.mean(),
                     v_.mean() - bb),
                  flush=True)
            rows.append({
                "outcome": oc,
                "method": "%s/%s"
                          % (cname, meth),
                "kind": "combo",
                "mean": v_.mean(),
                "sd": v_.std(ddof=1),
                "vs_best": v_.mean() - bb,
                "members": "+".join(mem),
                "weights": (str(np.round(
                    wlast, 2))
                    if wlast is not None
                    else "")})

    if len(names) >= 2:
        pa = P[best_base][0]
        cors = {n: abs(float(st.spearmanr(
            pa, P[n][0]).statistic))
            for n in names if n != best_base}
        partner = min(cors, key=cors.get)
        aa = []
        for i in range(len(SEEDS)):
            v, _ = wgrid([P[best_base][i],
                          P[partner][i]], y)
            aa.append(roc_auc_score(y, v))
        v_ = np.array(aa)
        print("    %-20s %8.4f %+9.4f"
              "   (%s + %s, spearman %.3f)"
              % ("pruned/grid", v_.mean(),
                 v_.mean() - bb, best_base,
                 partner, cors[partner]),
              flush=True)
        rows.append({
            "outcome": oc,
            "method": "pruned/grid",
            "kind": "combo",
            "mean": v_.mean(),
            "sd": v_.std(ddof=1),
            "vs_best": v_.mean() - bb,
            "members": best_base + "+"
                       + partner,
            "weights": ""})

    pd.DataFrame(rows).to_csv(DEST,
                              index=False)
    pd.DataFrame(drows).to_csv(DDEST,
                               index=False)
    print("")
    print("  %s done in %.1f min  (saved)"
          % (oc, (time.time() - tb) / 60),
          flush=True)

r = pd.DataFrame(rows)
dv = pd.DataFrame(drows)
r.to_csv(DEST, index=False)
dv.to_csv(DDEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("HOW SIMILAR ARE THE STRATEGIES?")
s = dv[dv["a"].isin(["early", "inter", "late"])
       & dv["b"].isin(["early", "inter",
                       "late"])]
if len(s):
    print(s[["outcome", "a", "b", "spearman",
             "err_corr", "kappa"]].round(3)
          .to_string(index=False))
    print("")
    print("  mean Spearman: %.3f"
          % s["spearman"].mean())

print("")
print("ALL PAIRWISE SPEARMAN, averaged over"
      " outcomes")
if len(dv):
    print(dv.groupby(["a", "b"])["spearman"]
          .mean().reset_index()
          .sort_values("spearman").round(3)
          .to_string(index=False))

print("")
print("=" * 74)
print("DOES COMBINING HELP?")
c = r[r["kind"] == "combo"]
if len(c):
    print(c.pivot_table(index="method",
                        columns="outcome",
                        values="vs_best")
          .round(4).to_string())
    print("")
    print("  positive means the combination"
          " beats the best single model")
    print("  wins: %d of %d cells"
          % (int((c["vs_best"] > 0).sum()),
             len(c)))
    b = c.loc[c["vs_best"].idxmax()]
    print("  largest gain: %s on %s, %+.4f"
          % (b["method"], b["outcome"],
             b["vs_best"]))

print("")
print("DOES DIVERSITY PREDICT COMBINATION"
      " GAIN?")
print("  ensembles gain from members whose errors differ,")
print("  which is the gap rule applied to models")
print("  rather than modalities")
if len(dv) > 5 and len(c):
    pg = c[c["method"] == "all/grid"][
        ["outcome", "vs_best"]]
    md = dv.groupby("outcome").agg(
        spear=("spearman", "mean"),
        err=("err_corr", "mean"),
        dis=("disagree", "mean")).reset_index()
    z = md.merge(pg, on="outcome")
    if len(z) > 2:
        print("")
        print(z.round(4).to_string(index=False))
        for nm, col in (("Spearman", "spear"),
                        ("error corr", "err"),
                        ("disagreement",
                         "dis")):
            if np.std(z[col]) > 1e-9:
                sr, sp = st.spearmanr(
                    z[col], z["vs_best"])
                print("  %-14s vs gain:"
                      " rho %+.3f (p=%.3f)"
                      % (nm, sr, sp))
print("")
print("saved", DEST, r.shape)
keep_awake(False)