"""Three further fusion methods: cooperative
learning, supervised PCA and LOL.

  COOPERATIVE LEARNING (Ding and Tibshirani)
    minimises ||y - fX - fZ||^2 + rho*||fX - fZ||^2
    on an augmented design. At rho = 0 it is
    early fusion, which the self-check at the end
    verifies for every cell. The penalty rewards
    views that agree, where the gap rule says
    gains come from views that differ.

  SUPERVISED PCA (Bair, Hastie, Paul, Tibshirani)
    screen features by univariate association
    inside the fold, then decompose.

  LOL (Vogelstein et al.)
    the class mean-difference vector plus the top
    principal components, orthogonalised. Built
    for p >> n.

C is selected by inner cross-validation for every
method. Welch's t for the screening is computed
directly, with a normal-tail p-value, which is
adequate for ranking features.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\final_fusion.csv
  results\\final_fusion_log.txt
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings(
    "ignore", category=RuntimeWarning)

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
    print("  set the power plan too:"
          " powercfg /change"
          " standby-timeout-ac 0")
atexit.register(keep_awake, False)

PROC, FIG = f.PROC, f.FIG
RAW = os.path.join("data", "raw", "ecg")
SEEDS = [42, 7, 13, 1, 2, 3]
NFOLD = 5
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
RHOS = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]
SPCA_THR = [0.05, 0.10, 0.25, 0.50]
SPCA_K = [5, 10, 20]
LOL_K = [5, 10, 20, 30]
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(PROC,
                    "final_fusion.csv")

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


# ---------- cooperative learning ------------
def coop_fit_predict(Atr, Btr, ytr, Ate, Bte,
                     rho, gtr=None):
    """Cooperative learning with C selected by
    INNER CROSS-VALIDATION.

    minimise ||y - Xa.wa - Xb.wb||^2
             + rho * ||Xa.wa - Xb.wb||^2

    which equals an ordinary fit on

        X~ = [[Xa,     Xb   ],
              [-s.Xa,  s.Xb ]]
        y~ = [y, 0]        s = sqrt(rho)

    The earlier version chose C on the rows it
    had just fitted, so rho = 0 was an
    overfitted C rather than early fusion.
    Selection now matches fit_pred exactly, so
    rho = 0 MUST reproduce early fusion, and the
    self-check at the end verifies it."""
    s = np.sqrt(max(rho, 0.0))
    n = Atr.shape[0]
    top = np.column_stack([Atr, Btr])
    tote = np.column_stack([Ate, Bte])

    def build(idx):
        t = top[idx]
        yy = ytr[idx].astype(int)
        if s <= 0:
            return t, yy
        b = np.column_stack(
            [-s * Atr[idx], s * Btr[idx]])
        return (np.vstack([t, b]),
                np.concatenate([
                    yy, np.zeros(len(idx),
                                 dtype=int)]))

    best, bc = -1.0, 1.0
    try:
        kk = min(3, max(2,
                        int(ytr.sum()) // 10))
        icv = StratifiedGroupKFold(
            n_splits=kk, shuffle=True,
            random_state=42)
        g = (gtr if gtr is not None
             else np.arange(n))
        for c in CS:
            q = np.zeros(n)
            for t2, v2 in icv.split(top, ytr,
                                    g):
                Xa_, ya_ = build(t2)
                m = LogisticRegression(
                    C=c, max_iter=3000)
                m.fit(Xa_, ya_)
                q[v2] = m.predict_proba(
                    top[v2])[:, 1]
            a_ = roc_auc_score(ytr, q)
            if a_ > best:
                best, bc = a_, c
    except Exception:
        bc = 1.0
    Xa_, ya_ = build(np.arange(n))
    m = LogisticRegression(C=bc, max_iter=3000)
    m.fit(Xa_, ya_)
    return m.predict_proba(tote)[:, 1]


def oof_coop(Xa, Xb, y, grp, rho, seed):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        At, Ae = prep_fold(Xa[tr], Xa[te])
        Bt, Be = prep_fold(Xb[tr], Xb[te])
        p[te] = coop_fit_predict(
            At, Bt, y[tr], Ae, Be, rho,
            grp[tr])
    return p


# ---------- supervised PCA ------------------
def spca_transform(Xtr, ytr, Xte, thr, k):
    """Screen by univariate association first,
    then decompose. The screening happens inside
    the fold, which is the step usually got
    wrong."""
    Xt, Xe = prep_fold(Xtr, Xte)
    ia = ytr == 1
    ib = ~ia
    # Welch's t computed directly. scipy's
    # ttest_ind fires a catastrophic-cancellation
    # warning on the near-constant binary CTPA
    # flags and floods the log. The statistic is
    # identical; only the p-value uses a normal
    # tail, which is fine for ranking features.
    if ia.sum() >= 3 and ib.sum() >= 3:
        ma, mb = Xt[ia].mean(0), Xt[ib].mean(0)
        va = Xt[ia].var(0, ddof=1)
        vb = Xt[ib].var(0, ddof=1)
        se = np.sqrt(va / ia.sum()
                     + vb / ib.sum())
        se = np.where(se > 1e-12, se, np.inf)
        pv = 2.0 * st.norm.sf(
            np.abs(ma - mb) / se)
    else:
        pv = np.ones(Xt.shape[1])
    pv = np.nan_to_num(np.asarray(
        pv, dtype=float), nan=1.0)
    sel = pv <= thr
    if sel.sum() < 3:
        idx = np.argsort(pv)[:max(3, k)]
        sel = np.zeros(Xt.shape[1], bool)
        sel[idx] = True
    kk = int(min(k, sel.sum() - 1,
                 len(ytr) - 1))
    if kk < 1:
        return (Xt[:, sel], Xe[:, sel],
                int(sel.sum()))
    pc = PCA(n_components=kk, random_state=42)
    return (pc.fit_transform(Xt[:, sel]),
            pc.transform(Xe[:, sel]),
            int(sel.sum()))


def oof_spca(X, y, grp, thr, k, seed):
    p = np.zeros(len(y))
    nsel = []
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        zt, ze, ns = spca_transform(
            X[tr], y[tr], X[te], thr, k)
        nsel.append(ns)
        p[te] = fit_pred(zt, y[tr], ze,
                         grp[tr])
    return p, float(np.mean(nsel))


# ---------- LOL -----------------------------
def lol_transform(Xtr, ytr, Xte, k):
    """Class mean-difference vector plus the top
    principal components, orthogonalised."""
    Xt, Xe = prep_fold(Xtr, Xte)
    d = (Xt[ytr == 1].mean(0)
         - Xt[ytr == 0].mean(0))
    nd = np.linalg.norm(d)
    if nd < 1e-9:
        d = np.ones(Xt.shape[1])
        nd = np.linalg.norm(d)
    d = (d / nd).reshape(-1, 1)
    kk = int(min(k - 1, Xt.shape[1] - 1,
                 len(ytr) - 2))
    if kk >= 1:
        pc = PCA(n_components=kk,
                 random_state=42)
        pc.fit(Xt)
        W = np.column_stack(
            [d, pc.components_.T])
    else:
        W = d
    Q, _ = np.linalg.qr(W)
    return Xt @ Q, Xe @ Q


def oof_lol(X, y, grp, k, seed):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        zt, ze = lol_transform(
            X[tr], y[tr], X[te], k)
        p[te] = fit_pred(zt, y[tr], ze,
                         grp[tr])
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


def oof_late(blocks, y, grp, seed):
    ps = []
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for X in blocks:
        q = np.zeros(len(y))
        for tr, te in cv.split(
                np.zeros((len(y), 1)), y, grp):
            zt, ze = prep_fold(X[tr], X[te])
            q[te] = fit_pred(zt, y[tr], ze,
                             grp[tr])
        ps.append(_rank(q))
    bs, bv = -1.0, ps[0]
    for w in np.arange(0, 1.001, 0.05):
        v = w * ps[1] + (1 - w) * ps[0]
        a = roc_auc_score(y, v)
        if a > bs:
            bs, bv = a, v
    return bv


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
BLK = {"ehr": EH, "ecg": ECGC, "ctpa": CT}
PAIRS = [("ehr", "ctpa"), ("ehr", "ecg"),
         ("ecg", "ctpa")]
print("")
print("cohort:", len(D))
print("  EHR %d   ECG %d   CTPA %d"
      % (len(EH), len(ECGC), len(CT)),
      flush=True)

rows = []
t0 = time.time()

for oc in OUTS:
    if oc not in D.columns:
        continue
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    if y.sum() < 25:
        continue
    X = {k: d[v].values.astype(float)
         for k, v in BLK.items()}
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())),
          flush=True)

    for a, b in PAIRS:
        pair = a + "+" + b
        Xa, Xb = X[a], X[b]
        Xc = np.column_stack([Xa, Xb])
        nf = Xc.shape[1]
        print("")
        print("  " + "=" * 68)
        print("  %-10s %d features"
              % (pair, nf), flush=True)

        la = np.array([roc_auc_score(
            y, oof_late([Xa, Xb], y, grp, s))
            for s in SEEDS])
        ref = la.mean()
        ea = np.array([roc_auc_score(
            y, oof_raw(Xc, y, grp, s))
            for s in SEEDS])
        print("    %-18s %8s %8s %9s"
              % ("method", "AUC", "SD",
                 "vs late"))
        for nm, v in (("late_wsrc", la),
                      ("early", ea)):
            print("    %-18s %8.4f %8.4f"
                  " %+9.4f"
                  % (nm, v.mean(),
                     v.std(ddof=1),
                     v.mean() - ref),
                  flush=True)
            rows.append({
                "outcome": oc, "pair": pair,
                "method": nm, "param": np.nan,
                "k": np.nan, "nfeat": nf,
                "n_selected": np.nan,
                "mean": v.mean(),
                "sd": v.std(ddof=1),
                "vs_late": v.mean() - ref})

        for rho in RHOS:
            try:
                aa = np.array([roc_auc_score(
                    y, oof_coop(Xa, Xb, y,
                                grp, rho, s))
                    for s in SEEDS])
            except Exception as exc:
                print("      coop rho=%.1f"
                      " FAILED %s"
                      % (rho, repr(exc)[:50]))
                continue
            note = ("  (must equal early)"
                    if rho == 0 else "")
            print("    %-18s %8.4f %8.4f"
                  " %+9.4f%s"
                  % ("coop rho=%.1f" % rho,
                     aa.mean(), aa.std(ddof=1),
                     aa.mean() - ref, note),
                  flush=True)
            rows.append({
                "outcome": oc, "pair": pair,
                "method": "coop",
                "param": rho, "k": np.nan,
                "nfeat": nf,
                "n_selected": np.nan,
                "mean": aa.mean(),
                "sd": aa.std(ddof=1),
                "vs_late": aa.mean() - ref})

        for thr in SPCA_THR:
            for k in SPCA_K:
                try:
                    res = [oof_spca(
                        Xc, y, grp, thr, k, s)
                        for s in SEEDS]
                    aa = np.array(
                        [roc_auc_score(y, p)
                         for p, _ in res])
                    ns = float(np.mean(
                        [n for _, n in res]))
                except Exception as exc:
                    print("      spca FAILED",
                          repr(exc)[:50])
                    continue
                print("    %-18s %8.4f %8.4f"
                      " %+9.4f   %.0f of %d"
                      " kept"
                      % ("spca p<%.2f k%d"
                         % (thr, k), aa.mean(),
                         aa.std(ddof=1),
                         aa.mean() - ref, ns,
                         nf), flush=True)
                rows.append({
                    "outcome": oc,
                    "pair": pair,
                    "method": "spca",
                    "param": thr, "k": k,
                    "nfeat": nf,
                    "n_selected": ns,
                    "mean": aa.mean(),
                    "sd": aa.std(ddof=1),
                    "vs_late": aa.mean() - ref})

        for k in LOL_K:
            try:
                aa = np.array([roc_auc_score(
                    y, oof_lol(Xc, y, grp,
                               k, s))
                    for s in SEEDS])
            except Exception as exc:
                print("      lol FAILED",
                      repr(exc)[:50])
                continue
            print("    %-18s %8.4f %8.4f"
                  " %+9.4f"
                  % ("lol k%d" % k, aa.mean(),
                     aa.std(ddof=1),
                     aa.mean() - ref),
                  flush=True)
            rows.append({
                "outcome": oc, "pair": pair,
                "method": "lol", "param": k,
                "k": k, "nfeat": nf,
                "n_selected": np.nan,
                "mean": aa.mean(),
                "sd": aa.std(ddof=1),
                "vs_late": aa.mean() - ref})

    pd.DataFrame(rows).to_csv(DEST,
                              index=False)
    print("")
    print("  %s done in %.1f min  (saved,"
          " %d rows)"
          % (oc, (time.time() - tb) / 60,
             len(rows)), flush=True)

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("SELF-CHECK: coop rho=0 vs early")
print("  at rho = 0 the augmented rows vanish,"
      " so these must match.")
bad = 0
for p in ("ehr+ctpa", "ehr+ecg", "ecg+ctpa"):
    for oc in OUTS:
        z = r[(r["pair"] == p)
              & (r["outcome"] == oc)]
        e = z[z["method"] == "early"]
        c0 = z[(z["method"] == "coop")
               & (z["param"] == 0.0)]
        if len(e) and len(c0):
            df = (c0["mean"].iloc[0]
                  - e["mean"].iloc[0])
            ok = abs(df) < 1e-6
            bad += int(not ok)
            print("    %-10s %-18s %+.6f  %s"
                  % (p, oc, df,
                     "OK" if ok
                     else "MISMATCH"))
print("")
if bad:
    print("  %d MISMATCHES: the cooperative"
          " results are not trustworthy." % bad)
else:
    print("  all cells match, so the"
          " cooperative results are sound")

print("")
print("BEST OF EACH METHOD, vs late:wsrc")
bf = r.loc[r.groupby(
    ["outcome", "pair", "method"]
)["mean"].idxmax()]
for p in ("ehr+ctpa", "ehr+ecg", "ecg+ctpa"):
    s = bf[bf["pair"] == p]
    if not len(s):
        continue
    print("")
    print("  " + p)
    print(s.pivot_table(index="method",
                        columns="outcome",
                        values="vs_late")
          .round(4).to_string())

print("")
print("AVERAGE RANK ACROSS ALL CELLS")
rk = bf.pivot_table(
    index="method",
    columns=["pair", "outcome"],
    values="mean").rank(
    ascending=False).mean(axis=1)
print(rk.sort_values().round(2).to_string())

print("")
print("=" * 74)
print("COOPERATIVE LEARNING: DOES AGREEMENT"
      " BETWEEN VIEWS HELP?")
print("  rho penalises disagreement, so it"
      " rewards views that agree. The gap")
print("  rule says gains come from views that"
      " differ, so rho > 0 winning would")
print("  need the rule qualified.")
c = r[r["method"] == "coop"]
if len(c):
    print("")
    print(c.pivot_table(
        index="param",
        columns=["pair", "outcome"],
        values="vs_late").round(4).to_string())
    print("")
    for p in ("ehr+ctpa", "ehr+ecg",
              "ecg+ctpa"):
        for oc in OUTS:
            s = c[(c["pair"] == p)
                  & (c["outcome"] == oc)]
            if not len(s):
                continue
            b = s.loc[s["mean"].idxmax()]
            z = s[s["param"] == 0.0]
            print("    %-10s %-18s best"
                  " rho=%.1f %+.4f"
                  "   rho=0 %+.4f"
                  % (p, oc, b["param"],
                     b["vs_late"],
                     z["vs_late"].iloc[0]
                     if len(z) else np.nan))
    idx = c.groupby(["pair", "outcome"])[
        "mean"].idxmax()
    nz = int((c.loc[idx]["param"] > 0).sum())
    tot = len(idx)
    print("")
    print("    cells preferring rho > 0:"
          " %d of %d" % (nz, tot))
    if nz <= tot // 3:
        print("    -> agreement-seeking mostly"
              " does not help, consistent")
        print("       with the gap rule")

print("")
print("SUPERVISED PCA: DOES SCREENING FIX"
      " PLAIN PCA?")
print("  plain PCA lost wherever ECG was"
      " present, since its leading components")
print("  track amplitude variance rather than"
      " signal.")
s = r[r["method"] == "spca"]
if len(s):
    for p in ("ehr+ctpa", "ehr+ecg",
              "ecg+ctpa"):
        for oc in OUTS:
            z = s[(s["pair"] == p)
                  & (s["outcome"] == oc)]
            if not len(z):
                continue
            b = z.loc[z["mean"].idxmax()]
            print("    %-10s %-18s p<%.2f k%.0f"
                  "  %+.4f   %.0f of %d kept"
                  % (p, oc, b["param"],
                     b["k"], b["vs_late"],
                     b["n_selected"],
                     b["nfeat"]))

print("")
print("WINS OVER late:wsrc")
w = bf[bf["vs_late"] > 0]
print("  %d of %d cells" % (len(w), len(bf)))
if len(w):
    print(w[["outcome", "pair", "method",
             "param", "mean", "vs_late"]]
          .round(4).to_string(index=False))

print("")
print("  with these three, sixteen fusion"
      " families have now been tested")
print("")
print("saved", DEST, r.shape)
keep_awake(False)