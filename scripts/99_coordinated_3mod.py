"""Coordinated representations on the MIMIC
three-modality cohort: every pairing of EHR, ECG
and CTPA, and all three together.

Methods as in script 98 (early, PCA, CCA, kernel
CCA, PLS), with late:wsrc as the reference. The
ECG block keeps columns down to 20% coverage, so
it contains missing values; clean_fit and
clean_apply fill them with medians from the
training fold only, before any transform. Each
method is wrapped so one failure does not stop
the run, and results are written after each
outcome.

PCA maximises variance rather than
discrimination, so on the wide ECG block its
leading components may track amplitude and
lead-voltage terms rather than prognosis. PLS
maximises covariance with the outcome and is the
comparison for that.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\coord3.csv
  data\\processed\\coord3_cancorr.csv
  results\\coordinated_3mod_log.txt
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import linalg as sla
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
RAW = os.path.join("data", "raw", "ecg")
SEEDS = [42, 7, 13]
NFOLD = 5
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
REGS = [0.1, 0.5]
NCOMP = [5, 10, 20]
PCA_K = [10, 20, 40]
PLS_K = [5, 10, 20]
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
OUTS = ["death_30d", "composite_30d",
        "cv_first", "death_30d_inhosp"]
DEST = os.path.join(PROC, "coord3.csv")
CDEST = os.path.join(PROC,
                     "coord3_cancorr.csv")

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


# ------------- NaN handling, centralised ----
def clean_fit(A):
    """Returns cleaned array and the medians,
    taken from this array only."""
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


def rcca(X, Y, reg=0.1, k=10):
    """Regularised CCA, closed form. Uses no
    labels, so all patients contribute."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    n = len(X)
    p, q = X.shape[1], Y.shape[1]
    Cxx = X.T @ X / n + reg * np.eye(p)
    Cyy = Y.T @ Y / n + reg * np.eye(q)
    Cxy = X.T @ Y / n
    try:
        Kx = sla.fractional_matrix_power(
            Cxx, -0.5).real
        Ky = sla.fractional_matrix_power(
            Cyy, -0.5).real
        U, s, Vt = np.linalg.svd(
            Kx @ Cxy @ Ky,
            full_matrices=False)
    except Exception:
        return None, None, None
    k = int(min(k, len(s)))
    return (Kx @ U[:, :k], Ky @ Vt[:k].T,
            s[:k])


def gcca(views, reg=0.1, k=10):
    """MAX-VAR generalised CCA for three or more
    views. CCA is defined for two and does not
    extend; this does."""
    Ws, Zs = [], []
    for V in views:
        V = V - V.mean(0)
        n, p = V.shape
        C = V.T @ V / n + reg * np.eye(p)
        try:
            K = sla.fractional_matrix_power(
                C, -0.5).real
        except Exception:
            return None, None
        Ws.append(K)
        Zs.append(V @ K)
    try:
        U, s, Vt = np.linalg.svd(
            np.column_stack(Zs),
            full_matrices=False)
    except Exception:
        return None, None
    k = int(min(k, Vt.shape[0]))
    P = Vt[:k].T
    out, i = [], 0
    for W, V in zip(Ws, views):
        p = V.shape[1]
        out.append(W @ P[i:i + p])
        i += p
    return out, s[:k]


def kcca_feats(Xtr, Ytr, Xte, Yte, reg=0.1,
               k=10, m=600, seed=0):
    """Kernel CCA. Inputs are cleaned here, so
    the caller cannot pass NaN into
    rbf_kernel."""
    Xtr, mx = clean_fit(Xtr)
    Xte = clean_apply(Xte, mx)
    Ytr, my = clean_fit(Ytr)
    Yte = clean_apply(Yte, my)
    rng = np.random.default_rng(seed)
    n = len(Xtr)
    idx = (rng.choice(n, m, replace=False)
           if n > m else np.arange(n))
    gx = 1.0 / max(Xtr.shape[1], 1)
    gy = 1.0 / max(Ytr.shape[1], 1)
    Ktr_x = rbf_kernel(Xtr, Xtr[idx], gx)
    Ktr_y = rbf_kernel(Ytr, Ytr[idx], gy)
    Kte_x = rbf_kernel(Xte, Xtr[idx], gx)
    Kte_y = rbf_kernel(Yte, Ytr[idx], gy)
    cx, cy = Ktr_x.mean(0), Ktr_y.mean(0)
    A, B, s = rcca(Ktr_x - cx, Ktr_y - cy,
                   reg, k)
    if A is None:
        return None, None, None
    return (np.column_stack([
        (Ktr_x - cx) @ A,
        (Ktr_y - cy) @ B]),
        np.column_stack([
            (Kte_x - cx) @ A,
            (Kte_y - cy) @ B]), s)


def fit_pred(Xtr, ytr, Xte, gtr=None):
    im = SimpleImputer(strategy="median")
    a = im.fit_transform(Xtr)
    b = im.transform(Xte)
    sc = StandardScaler()
    a, b = sc.fit_transform(a), sc.transform(b)
    best, bc = -1.0, 1.0
    try:
        kk = min(3, max(2,
                        int(ytr.sum()) // 10))
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
    m = LogisticRegression(C=bc,
                           max_iter=3000)
    m.fit(a, ytr)
    return m.predict_proba(b)[:, 1]


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
CT = [c for c in f.CTPA46
      if c in mm.columns]

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
BLOCKS = {"ehr": EH, "ecg": ECGC, "ctpa": CT}
PAIRS = [("ehr", "ecg"), ("ecg", "ctpa"),
         ("ehr", "ctpa")]

print("three-modality cohort:", len(D))
print("  EHR %d   ECG %d   CTPA %d"
      % (len(EH), len(ECGC), len(CT)))
print("  ehr+ecg %d   ecg+ctpa %d"
      "   ehr+ctpa %d   all three %d"
      % (len(EH) + len(ECGC),
         len(ECGC) + len(CT),
         len(EH) + len(CT),
         len(EH) + len(ECGC) + len(CT)))
print("  INSPECT has no ECG, so only the M2M"
      " direction exists here")
nn = D[ECGC].isna().sum().sum()
print("  NaN cells in the ECG block: %d"
      "  (cleaned inside every transform)"
      % int(nn), flush=True)

rows, crows = [], []
t0 = time.time()


def evaluate(transform, y, grp, nm, pair, oc,
             extra=None):
    """Grouped OOF. transform(tr, te, seed)
    returns (Xtr, Xte) with the fit done on the
    TRAINING rows only. Wrapped so one failure
    cannot end the run."""
    try:
        aa = []
        for s in SEEDS:
            p = np.zeros(len(y))
            cv = StratifiedGroupKFold(
                n_splits=NFOLD, shuffle=True,
                random_state=s)
            for tr, te in cv.split(
                    np.zeros((len(y), 1)),
                    y, grp):
                Xtr, Xte = transform(tr, te, s)
                if Xtr is None:
                    p[te] = 0.5
                    continue
                p[te] = fit_pred(
                    Xtr, y[tr], Xte, grp[tr])
            aa.append(roc_auc_score(y, p))
        v = np.array(aa)
    except Exception as exc:
        print("      %s FAILED: %s"
              % (nm, repr(exc)[:70]),
              flush=True)
        return np.nan, np.nan
    d_ = {"outcome": oc, "pair": pair,
          "method": nm, "auc": v.mean(),
          "sd": v.std(ddof=1)}
    if extra:
        d_.update(extra)
    rows.append(d_)
    return v.mean(), v.std(ddof=1)


for oc in OUTS:
    if oc not in D.columns:
        continue
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    if y.sum() < 30:
        continue
    X = {k: d[v].values.astype(float)
         for k, v in BLOCKS.items()}
    tb = time.time()

    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())),
          flush=True)

    uni = {}
    for k in BLOCKS:
        m_, _ = evaluate(
            lambda tr, te, s, k=k:
                (X[k][tr], X[k][te]),
            y, grp, "uni_" + k, "single", oc)
        uni[k] = m_
    print("  unimodal:  ehr %.4f   ecg %.4f"
          "   ctpa %.4f"
          % (uni["ehr"], uni["ecg"],
             uni["ctpa"]), flush=True)

    for a, b in PAIRS:
        pair = a + "+" + b
        Xa, Xb = X[a], X[b]
        nf = Xa.shape[1] + Xb.shape[1]
        bu = max(uni[a], uni[b])
        gap = abs(uni[a] - uni[b])
        print("")
        print("  " + "=" * 68)
        print("  %-10s %d features   gap %.4f"
              "   gap rule predicts %+.4f"
              % (pair, nf, gap,
                 0.036 - 0.243 * gap),
              flush=True)

        # late:wsrc
        try:
            aa = []
            for s in SEEDS:
                pa = np.zeros(len(y))
                pb = np.zeros(len(y))
                cv = StratifiedGroupKFold(
                    n_splits=NFOLD,
                    shuffle=True,
                    random_state=s)
                for tr, te in cv.split(
                        Xa, y, grp):
                    pa[te] = fit_pred(
                        Xa[tr], y[tr], Xa[te],
                        grp[tr])
                    pb[te] = fit_pred(
                        Xb[tr], y[tr], Xb[te],
                        grp[tr])
                pa, pb = _rank(pa), _rank(pb)
                bs = -1.0
                for w_ in np.arange(
                        0, 1.001, 0.05):
                    q = roc_auc_score(
                        y, w_ * pb
                        + (1 - w_) * pa)
                    bs = max(bs, q)
                aa.append(bs)
            a_late = float(np.mean(aa))
            rows.append({
                "outcome": oc, "pair": pair,
                "method": "late_wsrc",
                "auc": a_late,
                "sd": float(np.std(aa,
                                   ddof=1))})
        except Exception as exc:
            a_late = np.nan
            print("    late_wsrc FAILED",
                  repr(exc)[:60])

        print("    %-18s %8s %9s"
              % ("method", "AUC", "vs uni"))
        print("    %-18s %8.4f %+9.4f"
              % ("late_wsrc", a_late,
                 a_late - bu), flush=True)

        m_, _ = evaluate(
            lambda tr, te, s, Xa=Xa, Xb=Xb:
                (np.column_stack(
                    [Xa[tr], Xb[tr]]),
                 np.column_stack(
                     [Xa[te], Xb[te]])),
            y, grp, "early", pair, oc)
        print("    %-18s %8.4f %+9.4f"
              % ("early", m_, m_ - bu),
              flush=True)

        for k in PCA_K:
            if k >= nf:
                continue

            def fpca(tr, te, s, Xa=Xa, Xb=Xb,
                     k=k):
                Etr, md = clean_fit(
                    np.column_stack(
                        [Xa[tr], Xb[tr]]))
                Ete = clean_apply(
                    np.column_stack(
                        [Xa[te], Xb[te]]), md)
                sc = StandardScaler()
                pc = PCA(n_components=k,
                         random_state=42)
                return (pc.fit_transform(
                    sc.fit_transform(Etr)),
                    pc.transform(
                        sc.transform(Ete)))
            m_, _ = evaluate(
                fpca, y, grp, "pca%d" % k,
                pair, oc, {"k": k})
            print("    %-18s %8.4f %+9.4f"
                  % ("pca%d" % k, m_, m_ - bu),
                  flush=True)

        bcca = None
        for reg in REGS:
            for k in NCOMP:
                def fcca(tr, te, s, Xa=Xa,
                         Xb=Xb, reg=reg, k=k,
                         aug=False):
                    Atr, ma = clean_fit(Xa[tr])
                    Ate = clean_apply(Xa[te],
                                      ma)
                    Btr, mb = clean_fit(Xb[tr])
                    Bte = clean_apply(Xb[te],
                                      mb)
                    A_, B_, _ = rcca(Atr, Btr,
                                     reg, k)
                    if A_ is None:
                        return None, None
                    ca, cb = Atr.mean(0), \
                        Btr.mean(0)
                    ztr = np.column_stack([
                        (Atr - ca) @ A_,
                        (Btr - cb) @ B_])
                    zte = np.column_stack([
                        (Ate - ca) @ A_,
                        (Bte - cb) @ B_])
                    if aug:
                        ztr = np.column_stack(
                            [Atr, Btr, ztr])
                        zte = np.column_stack(
                            [Ate, Bte, zte])
                    return ztr, zte
                m_, _ = evaluate(
                    fcca, y, grp,
                    "cca_r%.1f_k%d" % (reg, k),
                    pair, oc,
                    {"reg": reg, "k": k})
                if np.isfinite(m_) and (
                        bcca is None
                        or m_ > bcca[1]):
                    bcca = ("cca r%.1f k%d"
                            % (reg, k), m_)
                evaluate(
                    lambda tr, te, s, reg=reg,
                    k=k, Xa=Xa, Xb=Xb:
                    fcca(tr, te, s, Xa, Xb,
                         reg, k, True),
                    y, grp,
                    "ccaaug_r%.1f_k%d"
                    % (reg, k), pair, oc,
                    {"reg": reg, "k": k})
        if bcca:
            print("    %-18s %8.4f %+9.4f"
                  % (bcca[0], bcca[1],
                     bcca[1] - bu), flush=True)

        for reg in (0.1, 0.5):
            def fk(tr, te, s, Xa=Xa, Xb=Xb,
                   reg=reg):
                return kcca_feats(
                    Xa[tr], Xb[tr], Xa[te],
                    Xb[te], reg, 10,
                    seed=s)[:2]
            m_, _ = evaluate(
                fk, y, grp,
                "kcca_r%.1f" % reg, pair, oc,
                {"reg": reg})
            if np.isfinite(m_):
                print("    %-18s %8.4f %+9.4f"
                      % ("kcca r%.1f" % reg,
                         m_, m_ - bu),
                      flush=True)

        for k in PLS_K:
            def fpls(tr, te, s, Xa=Xa, Xb=Xb,
                     k=k):
                Etr, md = clean_fit(
                    np.column_stack(
                        [Xa[tr], Xb[tr]]))
                Ete = clean_apply(
                    np.column_stack(
                        [Xa[te], Xb[te]]), md)
                pl = PLSRegression(
                    n_components=k, scale=True)
                pl.fit(Etr, y[tr].astype(float))
                return (pl.transform(Etr),
                        pl.transform(Ete))
            m_, _ = evaluate(
                fpls, y, grp, "pls%d" % k,
                pair, oc, {"k": k})
            print("    %-18s %8.4f %+9.4f"
                  % ("pls%d" % k, m_, m_ - bu),
                  flush=True)

        try:
            Atr, _ = clean_fit(Xa)
            Btr, _ = clean_fit(Xb)
            _, _, cc = rcca(Atr, Btr, 0.1, 10)
            if cc is not None:
                crows.append({
                    "outcome": oc,
                    "pair": pair, "nfeat": nf,
                    "cancorr1": float(cc[0]),
                    "cancorr_mean":
                        float(cc.mean()),
                    "shared_var": float(
                        np.mean(cc ** 2)),
                    "gap": gap,
                    "best_uni": bu,
                    "late_gain": a_late - bu})
                print("    canonical corrs:",
                      np.round(cc[:5], 3),
                      flush=True)
        except Exception:
            pass

    # ---------------- ALL THREE -------------
    bu3 = max(uni.values())
    print("")
    print("  " + "=" * 68)
    print("  ALL THREE   %d features"
          % sum(X[k].shape[1] for k in X))
    print("  CCA is defined for TWO views;"
          " GCCA is the extension", flush=True)

    try:
        aa = []
        for s in SEEDS:
            ps = {}
            cv = StratifiedGroupKFold(
                n_splits=NFOLD, shuffle=True,
                random_state=s)
            for k in BLOCKS:
                q = np.zeros(len(y))
                for tr, te in cv.split(
                        X[k], y, grp):
                    q[te] = fit_pred(
                        X[k][tr], y[tr],
                        X[k][te], grp[tr])
                ps[k] = _rank(q)
            bs = -1.0
            for w1 in np.arange(0, 1.001, 0.05):
                for w2 in np.arange(
                        0, 1.001 - w1, 0.05):
                    w3 = 1.0 - w1 - w2
                    q = roc_auc_score(
                        y, w1 * ps["ehr"]
                        + w2 * ps["ecg"]
                        + w3 * ps["ctpa"])
                    bs = max(bs, q)
            aa.append(bs)
        a3 = float(np.mean(aa))
        rows.append({"outcome": oc,
                     "pair": "all3",
                     "method": "late_wsrc",
                     "auc": a3,
                     "sd": float(np.std(
                         aa, ddof=1))})
        print("    %-18s %8.4f %+9.4f"
              % ("late_wsrc", a3, a3 - bu3),
              flush=True)
    except Exception as exc:
        print("    late_wsrc FAILED",
              repr(exc)[:60])

    m_, _ = evaluate(
        lambda tr, te, s:
            (np.column_stack(
                [X[k][tr] for k in BLOCKS]),
             np.column_stack(
                 [X[k][te] for k in BLOCKS])),
        y, grp, "early", "all3", oc)
    print("    %-18s %8.4f %+9.4f"
          % ("early", m_, m_ - bu3),
          flush=True)

    for reg in REGS:
        for k in NCOMP:
            def fg(tr, te, s, reg=reg, k=k,
                   aug=False):
                vs, mds = [], []
                for q in BLOCKS:
                    v, md = clean_fit(X[q][tr])
                    vs.append(v)
                    mds.append(md)
                Ps, _ = gcca(vs, reg, k)
                if Ps is None:
                    return None, None
                mus = [v.mean(0) for v in vs]
                ztr = np.column_stack([
                    (v - mu) @ P for v, mu, P
                    in zip(vs, mus, Ps)])
                vt = [clean_apply(X[q][te], md)
                      for q, md in
                      zip(BLOCKS, mds)]
                zte = np.column_stack([
                    (v - mu) @ P for v, mu, P
                    in zip(vt, mus, Ps)])
                if aug:
                    ztr = np.column_stack(
                        [np.column_stack(vs),
                         ztr])
                    zte = np.column_stack(
                        [np.column_stack(vt),
                         zte])
                return ztr, zte
            m_, _ = evaluate(
                fg, y, grp,
                "gcca_r%.1f_k%d" % (reg, k),
                "all3", oc,
                {"reg": reg, "k": k})
            print("    %-18s %8.4f %+9.4f"
                  % ("gcca r%.1f k%d"
                     % (reg, k), m_,
                     m_ - bu3), flush=True)
            evaluate(
                lambda tr, te, s, reg=reg,
                k=k: fg(tr, te, s, reg, k,
                        True),
                y, grp,
                "gccaaug_r%.1f_k%d"
                % (reg, k), "all3", oc,
                {"reg": reg, "k": k})

    for k in PLS_K:
        def fp3(tr, te, s, k=k):
            Etr, md = clean_fit(
                np.column_stack(
                    [X[q][tr] for q in BLOCKS]))
            Ete = clean_apply(
                np.column_stack(
                    [X[q][te] for q in BLOCKS]),
                md)
            pl = PLSRegression(
                n_components=k, scale=True)
            pl.fit(Etr, y[tr].astype(float))
            return (pl.transform(Etr),
                    pl.transform(Ete))
        m_, _ = evaluate(fp3, y, grp,
                         "pls%d" % k, "all3",
                         oc, {"k": k})
        print("    %-18s %8.4f %+9.4f"
              % ("pls%d" % k, m_, m_ - bu3),
              flush=True)

    # write after each outcome, so a crash
    # costs at most one block
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)
    pd.DataFrame(crows).to_csv(CDEST,
                               index=False)
    print("")
    print("  %s done in %.1f min"
          "   (saved, %d rows so far)"
          % (oc, (time.time() - tb) / 60,
             len(rows)), flush=True)

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)
c = pd.DataFrame(crows)
c.to_csv(CDEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("BEST OF EACH FAMILY, BY PAIRING")
r = r[r["auc"].notna()].copy()
r["fam"] = r["method"].str.replace(
    r"_r[\d.]+|_k\d+|\d+$", "", regex=True)
bf = r.loc[r.groupby(
    ["outcome", "pair", "fam"]
)["auc"].idxmax()]
for p in ("ehr+ecg", "ecg+ctpa",
          "ehr+ctpa", "all3"):
    s = bf[bf["pair"] == p]
    if not len(s):
        continue
    print("")
    print("  " + p)
    print(s.pivot_table(index="fam",
                        columns="outcome",
                        values="auc")
          .round(4).to_string())

print("")
print("AVERAGE RANK WITHIN EACH PAIRING")
for p in ("ehr+ecg", "ecg+ctpa",
          "ehr+ctpa", "all3"):
    s = bf[bf["pair"] == p]
    if not len(s):
        continue
    rk = s.pivot_table(
        index="fam", columns="outcome",
        values="auc").rank(
        ascending=False).mean(axis=1)
    print("")
    print("  " + p)
    print(rk.sort_values().round(2)
          .to_string())

print("")
print("=" * 74)
print("DOES COMPRESSION HELP MORE WHEN THERE")
print("IS MORE TO COMPRESS?")
print("  script 98 found PCA losing by 0.0108"
      " on 74 features; an earlier run of")
print("  this script showed it losing 0.054 to"
      " 0.133 on 292")
print("  %-10s %6s %12s %12s"
      % ("pair", "nfeat", "pca-early",
         "pls-early"))
NF = {"ehr+ctpa": len(EH) + len(CT),
      "ehr+ecg": len(EH) + len(ECGC),
      "ecg+ctpa": len(ECGC) + len(CT),
      "all3": len(EH) + len(ECGC) + len(CT)}
for p in ("ehr+ctpa", "ehr+ecg",
          "ecg+ctpa", "all3"):
    s = bf[bf["pair"] == p]
    pc = s[s["fam"] == "pca"]
    ea = s[s["fam"] == "early"]
    pl = s[s["fam"] == "pls"]
    if len(ea):
        print("  %-10s %6d %12s %12s"
              % (p, NF[p],
                 ("%+.4f" % (pc["auc"].mean()
                             - ea["auc"].mean())
                  if len(pc) else "-"),
                 ("%+.4f" % (pl["auc"].mean()
                             - ea["auc"].mean())
                  if len(pl) else "-")))
print("")
print("  PCA maximises variance, PLS maximises"
      " covariance with the outcome, so PLS")
print("  should be immune to the failure mode"
      " that sinks PCA on wide inputs")

print("")
print("ALL THREE MODALITIES")
s = bf[bf["pair"] == "all3"]
if len(s):
    print(s[["outcome", "fam", "auc"]]
          .round(4).to_string(index=False))

print("")
print("EVERYTHING vs late:wsrc")
for p in ("ehr+ecg", "ecg+ctpa",
          "ehr+ctpa", "all3"):
    for oc in OUTS:
        ss = r[(r["pair"] == p)
               & (r["outcome"] == oc)]
        lw = ss[ss["method"] == "late_wsrc"]
        if not len(ss) or not len(lw):
            continue
        b = ss.loc[ss["auc"].idxmax()]
        print("  %-10s %-16s best %-18s"
              " %.4f   late %.4f   %+.4f"
              % (p, oc, b["method"], b["auc"],
                 lw["auc"].iloc[0],
                 b["auc"] - lw["auc"].iloc[0]))

print("")
print("CANONICAL CORRELATIONS BY PAIRING")
if len(c):
    print(c.round(4).to_string(index=False))
    print("")
    print("  script 98 reached only 0.49 to"
          " 0.53 on ehr+ctpa, likely because")
    print("  CTPA is mostly binary flags;"
          " ehr+ecg is continuous throughout")
print("")
print("saved", DEST, r.shape)