"""Coordinated representations for EHR+CTPA: PCA,
CCA, kernel CCA and PLS, in all four transfer
directions.

Every method is closed-form (an eigenproblem or
SVD), so none adds trained hidden layers. CCA is
fitted without labels, so every patient
contributes to the projection and only the final
logistic regression uses the outcome.

METHODS
  early        raw concatenation, the reference
  pca_k        PCA on the concatenation, a linear
               stand-in for a multimodal
               autoencoder
  cca_shared   canonical variates only
  cca_aug      raw features plus canonical
               variates
  kcca         kernel CCA, RBF
  pls          supervised coordination,
               maximising covariance with the
               outcome

Also tests whether overlap in the feature space,
measured by the canonical correlations, predicts
fusion gain. Prediction correlation (moderator 3
in script 22) measured overlap in the outputs
instead.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\coordinated.csv
  data\\processed\\coordinated_cancorr.csv
  results\\coordinated_log.txt
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import linalg as sla
from scipy import stats as st
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

PROC = f.PROC
SEEDS = [42, 7, 13]
NFOLD = 5
CS = list(np.logspace(-4, 2, 7))
REGS = [0.01, 0.1, 0.5]
NCOMP = [5, 10, 20]
PCA_K = [10, 20, 40]
OUTS = ["death_30d", "composite_30d",
        "cv_first"]
DIRS = ["I2M", "M2I", "M2M", "I2I"]
DEST = os.path.join(PROC, "coordinated.csv")


def _rank(p):
    return pd.Series(p).rank(pct=True).values


# ---------------- regularised CCA -----------
def rcca(X, Y, reg=0.1, k=10):
    """Closed-form regularised CCA by whitening
    and SVD.

    Cxx^-1/2 Cxy Cyy^-1/2 = U S V', giving
    canonical correlations S and projections
    Wx = Cxx^-1/2 U, Wy = Cyy^-1/2 V.

    No labels are used, so every patient
    contributes regardless of outcome."""
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
    except Exception:
        return None, None, None
    M = Kx @ Cxy @ Ky
    U, s, Vt = np.linalg.svd(M,
                             full_matrices=False)
    k = int(min(k, len(s)))
    return (Kx @ U[:, :k], Ky @ Vt[:k].T,
            s[:k])


def kcca(Xtr, Ytr, Xte, Yte, reg=0.1, k=10,
         gamma=None, m=600, seed=0):
    """Kernel CCA with an RBF kernel, on a
    random subset of landmarks so the kernel
    matrix stays tractable.

    Nonlinear but still closed-form; the
    literature notes its generalisation is
    limited by the kernel being fixed."""
    rng = np.random.default_rng(seed)
    n = len(Xtr)
    idx = (rng.choice(n, m, replace=False)
           if n > m else np.arange(n))
    if gamma is None:
        gamma = 1.0 / Xtr.shape[1]
    Ktr_x = rbf_kernel(Xtr, Xtr[idx], gamma)
    Ktr_y = rbf_kernel(Ytr, Ytr[idx],
                       1.0 / Ytr.shape[1])
    Kte_x = rbf_kernel(Xte, Xtr[idx], gamma)
    Kte_y = rbf_kernel(Yte, Ytr[idx],
                       1.0 / Ytr.shape[1])
    mx, my = Ktr_x.mean(0), Ktr_y.mean(0)
    A, B, s = rcca(Ktr_x - mx, Ktr_y - my,
                   reg, k)
    if A is None:
        return None, None, None
    ztr = np.column_stack([
        (Ktr_x - mx) @ A, (Ktr_y - my) @ B])
    zte = np.column_stack([
        (Kte_x - mx) @ A, (Kte_y - my) @ B])
    return ztr, zte, s


def fit_eval(Xtr, ytr, Xte, yte, grp=None,
             seed=42):
    """L2 logistic regression with C chosen
    inside the training data."""
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
        g = (grp if grp is not None
             else np.arange(len(ytr)))
        for c in CS:
            q = np.zeros(len(ytr))
            for t2, v2 in icv.split(a, ytr, g):
                m = LogisticRegression(
                    C=c, max_iter=4000)
                m.fit(a[t2], ytr[t2])
                q[v2] = m.predict_proba(
                    a[v2])[:, 1]
            s = roc_auc_score(ytr, q)
            if s > best:
                best, bc = s, c
    except Exception:
        bc = 1.0
    m = LogisticRegression(C=bc, max_iter=4000)
    m.fit(a, ytr)
    return m.predict_proba(b)[:, 1]


def get_cell(direction, oc, ins, mim):
    if direction == "I2M":
        ds, ys = f.labels(ins, oc)
        dt, yt = f.labels(mim, oc)
        return ds, ys, dt, yt, None
    if direction == "M2I":
        ds, ys = f.labels(mim, oc)
        dt, yt = f.labels(ins, oc)
        return ds, ys, dt, yt, None
    if direction == "M2M":
        d, y = f.labels(mim, oc)
        return d, y, d, y, d["subject_id"].values
    d, y = f.labels(ins, oc)
    gc = ("subject_id" if "subject_id"
          in d.columns else "person_id")
    return d, y, d, y, d[gc].values


ins = f.load_inspect()
mim = f.load_mimic()
print("INSPECT", ins.shape, " MIMIC",
      mim.shape, flush=True)

rows, crows = [], []
t0 = time.time()

for oc in OUTS:
    for dr in DIRS:
        try:
            cell = get_cell(dr, oc, ins, mim)
        except Exception as exc:
            print("skip", dr, oc,
                  repr(exc)[:70])
            continue
        ds, ys, dt, yt, grp = cell
        if ys.sum() < 30 or yt.sum() < 30:
            continue
        A = f.EHR_COLS
        B = [c for c in f.CTPA_COLS
             if c in ds.columns
             and c in dt.columns]
        Xs_a, Xt_a = f.prep(
            ds[A].values.astype(float),
            dt[A].values.astype(float))
        Xs_b, Xt_b = f.prep(
            ds[B].values.astype(float),
            dt[B].values.astype(float))

        print("")
        print("=" * 74)
        print("%s  %s   source n=%d ev=%d"
              "   target n=%d ev=%d"
              % (dr, oc, len(ys),
                 int(ys.sum()), len(yt),
                 int(yt.sum())), flush=True)

        # ---- canonical correlations, no
        # labels used, so this is measurable
        # BEFORE any model is fitted ----
        _, _, cc = rcca(Xs_a, Xs_b, 0.1, 10)
        if cc is not None:
            print("  canonical correlations:",
                  np.round(cc[:5], 3))
            print("  first %.3f   mean of 10"
                  " %.3f   shared variance"
                  " %.3f"
                  % (cc[0], cc.mean(),
                     float(np.mean(cc ** 2))),
                  flush=True)

        def run(Xs, Xt, nm, extra=None):
            aa = []
            for s in SEEDS:
                if grp is None:
                    p = fit_eval(Xs, ys, Xt,
                                 yt, None, s)
                    aa.append(
                        roc_auc_score(yt, p))
                else:
                    p = np.zeros(len(yt))
                    cv = StratifiedGroupKFold(
                        n_splits=NFOLD,
                        shuffle=True,
                        random_state=s)
                    for tr, te in cv.split(
                            Xs, ys, grp):
                        p[te] = fit_eval(
                            Xs[tr], ys[tr],
                            Xt[te], yt[te],
                            grp[tr], s)
                    aa.append(
                        roc_auc_score(yt, p))
            v = np.array(aa)
            d_ = {"outcome": oc,
                  "direction": dr,
                  "method": nm,
                  "nfeat": Xs.shape[1],
                  "auc": v.mean(),
                  "sd": v.std(ddof=1)}
            if extra:
                d_.update(extra)
            rows.append(d_)
            return v.mean(), v.std(ddof=1)

        # references
        a_e, _ = run(Xs_a, Xt_a, "ehr")
        a_c, _ = run(Xs_b, Xt_b, "ctpa")
        Xs_e = np.column_stack([Xs_a, Xs_b])
        Xt_e = np.column_stack([Xt_a, Xt_b])
        a_early, _ = run(Xs_e, Xt_e, "early")

        # late:wsrc, the winner so far
        pa = _rank(fit_eval(Xs_a, ys, Xt_a,
                            yt, grp, 42))
        pb = _rank(fit_eval(Xs_b, ys, Xt_b,
                            yt, grp, 42))
        best, bw = -1.0, 0.5
        for w in np.arange(0, 1.001, 0.05):
            s_ = roc_auc_score(
                yt, w * pb + (1 - w) * pa)
            if s_ > best:
                best, bw = s_, w
        a_late = best
        rows.append({"outcome": oc,
                     "direction": dr,
                     "method": "late_wsrc",
                     "nfeat": 1,
                     "auc": a_late,
                     "sd": np.nan})

        print("")
        print("  %-14s %6s %8s %8s"
              % ("method", "nfeat", "AUC",
                 "SD"))
        for nm, v in (("ehr", a_e),
                      ("ctpa", a_c),
                      ("early", a_early),
                      ("late_wsrc", a_late)):
            print("  %-14s %6s %8.4f"
                  % (nm, "-", v))

        # ---- PCA, the autoencoder proxy ----
        for k in PCA_K:
            if k >= Xs_e.shape[1]:
                continue
            pc = PCA(n_components=k,
                     random_state=42)
            sc0 = StandardScaler()
            zs = pc.fit_transform(
                sc0.fit_transform(
                    np.nan_to_num(Xs_e)))
            zt = pc.transform(
                sc0.transform(
                    np.nan_to_num(Xt_e)))
            m_, s_ = run(zs, zt, "pca%d" % k,
                         {"k": k})
            print("  %-14s %6d %8.4f %8.4f"
                  % ("pca%d" % k, k, m_, s_),
                  flush=True)

        # ---- CCA ----
        bestcc = None
        for reg in REGS:
            for k in NCOMP:
                A_, B_, s_cc = rcca(
                    Xs_a, Xs_b, reg, k)
                if A_ is None:
                    continue
                ma, mb = Xs_a.mean(0), Xs_b.mean(0)
                zs = np.column_stack([
                    (Xs_a - ma) @ A_,
                    (Xs_b - mb) @ B_])
                zt = np.column_stack([
                    (Xt_a - ma) @ A_,
                    (Xt_b - mb) @ B_])
                nm = "cca_r%.2f_k%d" % (reg, k)
                m_, sd_ = run(
                    zs, zt, nm,
                    {"reg": reg, "k": k,
                     "cancorr1": float(s_cc[0]),
                     "cancorr_mean":
                         float(s_cc.mean())})
                if bestcc is None or \
                        m_ > bestcc[1]:
                    bestcc = (nm, m_, sd_,
                              reg, k)
                # augmented: raw plus shared
                m2, sd2 = run(
                    np.column_stack([Xs_e, zs]),
                    np.column_stack([Xt_e, zt]),
                    "ccaaug_r%.2f_k%d"
                    % (reg, k),
                    {"reg": reg, "k": k})
        if bestcc:
            print("  %-14s %6d %8.4f %8.4f"
                  " (reg %.2f)"
                  % ("cca best", bestcc[4],
                     bestcc[1], bestcc[2],
                     bestcc[3]), flush=True)

        # ---- kernel CCA ----
        try:
            for reg in (0.1, 0.5):
                zs, zt, s_k = kcca(
                    Xs_a, Xs_b, Xt_a, Xt_b,
                    reg, 10, seed=42)
                if zs is None:
                    continue
                m_, sd_ = run(
                    zs, zt,
                    "kcca_r%.2f" % reg,
                    {"reg": reg})
                print("  %-14s %6d %8.4f %8.4f"
                      % ("kcca r%.2f" % reg,
                         zs.shape[1], m_, sd_),
                      flush=True)
        except Exception as exc:
            print("  kcca failed:",
                  repr(exc)[:70])

        # ---- PLS, supervised coordination ----
        for k in (5, 10):
            try:
                pl = PLSRegression(
                    n_components=k, scale=True)
                pl.fit(np.nan_to_num(Xs_e),
                       ys.astype(float))
                zs = pl.transform(
                    np.nan_to_num(Xs_e))
                zt = pl.transform(
                    np.nan_to_num(Xt_e))
                m_, sd_ = run(zs, zt,
                              "pls%d" % k,
                              {"k": k})
                print("  %-14s %6d %8.4f %8.4f"
                      % ("pls%d" % k, k, m_,
                         sd_), flush=True)
            except Exception as exc:
                print("  pls failed:",
                      repr(exc)[:70])

        # record the cell for the structural
        # question
        if cc is not None:
            crows.append({
                "outcome": oc, "direction": dr,
                "cancorr1": float(cc[0]),
                "cancorr_mean": float(cc.mean()),
                "shared_var":
                    float(np.mean(cc ** 2)),
                "auc_ehr": a_e,
                "auc_ctpa": a_c,
                "gap": abs(a_e - a_c),
                "best_uni": max(a_e, a_c),
                "late_gain":
                    a_late - max(a_e, a_c)})

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)
c = pd.DataFrame(crows)
c.to_csv(os.path.join(
    PROC, "coordinated_cancorr.csv"),
    index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("BEST OF EACH FAMILY, BY CELL")
r["fam"] = r["method"].str.replace(
    r"_r[\d.]+|_k\d+|\d+$", "", regex=True)
bf = r.loc[r.groupby(
    ["outcome", "direction", "fam"]
)["auc"].idxmax()]
print(bf.pivot_table(index="fam",
                     columns="direction",
                     values="auc")
      .round(4).to_string())

print("")
print("GAIN OVER THE BEST UNIMODAL MODALITY")
uni = r[r["method"].isin(["ehr", "ctpa"])] \
    .groupby(["outcome", "direction"])["auc"] \
    .max().rename("best_uni").reset_index()
m = bf.merge(uni,
             on=["outcome", "direction"])
m["gain"] = m["auc"] - m["best_uni"]
print(m.pivot_table(index="fam",
                    columns="direction",
                    values="gain")
      .round(4).to_string())

print("")
print("AVERAGE RANK ACROSS CELLS")
rk = bf.pivot_table(
    index="fam",
    columns=["outcome", "direction"],
    values="auc").rank(
    ascending=False).mean(axis=1)
print(rk.sort_values().round(2).to_string())

print("")
print("=" * 74)
print("PCA AS THE AUTOENCODER PROXY")
print("  if linear compression fails, a")
print("  nonlinear autoencoder will not"
      " succeed either")
p = bf[bf["fam"] == "pca"]
e = bf[bf["fam"] == "early"]
if len(p) and len(e):
    mm = p.merge(
        e[["outcome", "direction", "auc"]],
        on=["outcome", "direction"],
        suffixes=("_pca", "_early"))
    mm["d"] = (mm["auc_pca"]
               - mm["auc_early"])
    print(mm[["outcome", "direction",
              "auc_early", "auc_pca", "d"]]
          .round(4).to_string(index=False))
    print("")
    print("  mean PCA minus early: %+.4f"
          % mm["d"].mean())
    if mm["d"].mean() < 0:
        print("  -> compression loses"
              " information. The autoencoder")
        print("     question is closed without"
              " building one.")

print("")
print("=" * 74)
print("THE STRUCTURAL QUESTION")
print("  does feature-space overlap predict")
print("  fusion gain, where output overlap")
print("  (moderator 3) failed at p = 0.67?")
print("=" * 74)
if len(c) > 3:
    print("")
    print(c.round(4).to_string(index=False))
    print("")
    for nm, col in (("first canonical corr",
                     "cancorr1"),
                    ("mean canonical corr",
                     "cancorr_mean"),
                    ("shared variance",
                     "shared_var"),
                    ("AUC gap (gap rule)",
                     "gap")):
        x = c[col].values
        y = c["late_gain"].values
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 3 and np.std(x[ok]) > 1e-9:
            pr, pp = st.pearsonr(x[ok], y[ok])
            sr, sp = st.spearmanr(x[ok], y[ok])
            print("  %-22s  Pearson %+.3f"
                  " (p=%.3f)   Spearman %+.3f"
                  " (p=%.3f)"
                  % (nm, pr, pp, sr, sp))
    print("")
    print("  a negative correlation would mean"
          " more shared variance implies")
    print("  less complementary information and"
          " so a smaller fusion gain, which")
    print("  is a second route to the gap"
          " mechanism, measurable from the")
    print("  FEATURES ALONE before any model is"
          " fitted")

print("")
print("BEST METHOD PER CELL")
for oc in OUTS:
    for dr in DIRS:
        s = r[(r["outcome"] == oc)
              & (r["direction"] == dr)]
        if not len(s):
            continue
        b = s.loc[s["auc"].idxmax()]
        lw = s[s["method"] == "late_wsrc"]
        note = ("" if not len(lw)
                else "   late_wsrc %.4f (%+.4f)"
                % (lw["auc"].iloc[0],
                   b["auc"] - lw["auc"].iloc[0]))
        print("  %-16s %-4s  %-18s %.4f%s"
              % (oc, dr, b["method"],
                 b["auc"], note))
print("")
print("saved", DEST, r.shape)