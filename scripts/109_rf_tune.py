"""Tunes the random forest on cca_aug.

The forest in script 108 used n_estimators=500,
max_depth=6, min_samples_leaf=10 and sklearn's
default max_features, none of them swept. mtry,
the number of predictors drawn at each split, is
usually the most influential. On the 94-column
cca_aug matrix, 20 of which are canonical
variates, the sqrt default draws about 10 columns
per split, so mtry controls how often the
cross-modal columns are considered. mtry and the
CCA rank are tuned together, since they interact.

ALSO TESTED
  min_samples_leaf, since a minimum of 10 may be
    too permissive at 83 events
  max_depth, for calibration as well as
    discrimination
  post-hoc calibration: Platt and isotonic on the
    out-of-fold predictions, since the
    dissertation reports decision curves
  cross-modal products: the strongest EHR feature
    times the strongest CTPA feature

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\rf_tune.csv
  results\\rf_tune_log.txt
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as sla
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier
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
SEEDS = [42, 7, 13, 1, 2]
SEEDS10 = [42, 7, 13, 1, 2, 3, 5, 8, 21, 99]
NFOLD = 5
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CCA_REG = 0.1
# mtry as a fraction of the column count, so it
# means the same thing as k changes
MTRY_FRAC = [0.10, 0.25, 0.40, 0.60, 1.00]
K_GRID = [5, 10, 20, 30]
LEAF = [5, 10, 25, 50, 100]
DEPTH = [4, 6, 8, 12, None]
NTREE = 500
CT_LO, CT_HI = -48.0, 24.0
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(PROC, "rf_tune.csv")
# script 108, ten seeds, untuned forest
BASE = {"death_30d_inhosp":
        {"late": 0.8823, "rf": 0.9105},
        "death_30d":
        {"late": 0.8822, "rf": 0.8994},
        "composite_30d":
        {"late": 0.8449, "rf": 0.8566},
        "cv_first":
        {"late": 0.8064, "rf": 0.8149}}


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


def build(At, Ae, Bt, Be, k, prod=0):
    """raw features plus canonical variates,
    optionally plus cross-modal products."""
    Xt = np.column_stack([At, Bt])
    Xe = np.column_stack([Ae, Be])
    A_, B_ = rcca(At, Bt, CCA_REG, k)
    if A_ is not None:
        ma, mb = At.mean(0), Bt.mean(0)
        Xt = np.column_stack([
            Xt, (At - ma) @ A_,
            (Bt - mb) @ B_])
        Xe = np.column_stack([
            Xe, (Ae - ma) @ A_,
            (Be - mb) @ B_])
    if prod > 0:
        # the strongest EHR columns times the
        # strongest CTPA columns. Trees split one
        # feature at a time, so a product is
        # expensive for them to represent, which
        # is the same argument that explains why
        # the canonical variates helped.
        va = np.abs(At).std(0)
        vb = np.abs(Bt).std(0)
        ia = np.argsort(-va)[:prod]
        ib = np.argsort(-vb)[:prod]
        pt, pe = [], []
        for i in ia:
            for j in ib:
                pt.append(At[:, i] * Bt[:, j])
                pe.append(Ae[:, i] * Be[:, j])
        Xt = np.column_stack([Xt] + pt)
        Xe = np.column_stack([Xe] + pe)
    return Xt, Xe


def make_rf(mfrac, leaf, depth, ncol,
            seed=42):
    mf = max(1, int(round(mfrac * ncol)))
    return RandomForestClassifier(
        n_estimators=NTREE,
        max_features=mf,
        max_depth=depth,
        min_samples_leaf=leaf,
        class_weight="balanced_subsample",
        random_state=seed, n_jobs=-1)


def oof_rf(Xa, Xb, y, grp, seed, k=10,
           mfrac=None, leaf=10, depth=6,
           prod=0):
    """mfrac None means sklearn's sqrt default,
    which on 94 columns draws about 10 and so
    considers fewer than two canonical
    variates per split."""
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        At, Ae = prep_fold(Xa[tr], Xa[te])
        Bt, Be = prep_fold(Xb[tr], Xb[te])
        Zt, Ze = build(At, Ae, Bt, Be, k,
                       prod)
        if mfrac is None:
            m = RandomForestClassifier(
                n_estimators=NTREE,
                max_depth=depth,
                min_samples_leaf=leaf,
                class_weight=
                "balanced_subsample",
                random_state=42, n_jobs=-1)
        else:
            m = make_rf(mfrac, leaf, depth,
                        Zt.shape[1])
        m.fit(Zt, y[tr])
        p[te] = m.predict_proba(Ze)[:, 1]
    return p


def oof_late(Xa, Xb, y, grp, seed):
    pa = np.zeros(len(y))
    pb = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        At, Ae = prep_fold(Xa[tr], Xa[te])
        Bt, Be = prep_fold(Xb[tr], Xb[te])
        for X_, Xt_, out in ((At, Ae, "a"),
                             (Bt, Be, "b")):
            best, bc = -1.0, 1.0
            for c in CS:
                try:
                    mm = LogisticRegression(
                        C=c, max_iter=3000)
                    mm.fit(X_, y[tr])
                    s = roc_auc_score(
                        y[tr],
                        mm.predict_proba(
                            X_)[:, 1])
                    if s > best:
                        best, bc = s, c
                except Exception:
                    continue
            mm = LogisticRegression(
                C=bc, max_iter=3000)
            mm.fit(X_, y[tr])
            q = mm.predict_proba(Xt_)[:, 1]
            if out == "a":
                pa[te] = q
            else:
                pb[te] = q
    pa, pb = _rank(pa), _rank(pb)
    bs, bv = -1.0, pa
    for w in np.arange(0, 1.001, 0.05):
        v = w * pb + (1 - w) * pa
        a = roc_auc_score(y, v)
        if a > bs:
            bs, bv = a, v
    return bv


def calib_stats(p, y):
    """Brier, calibration slope and intercept.
    A slope below 1 means over-confident."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    try:
        b = brier_score_loss(y, p)
    except Exception:
        b = np.nan
    try:
        x = np.log(p / (1 - p)).reshape(-1, 1)
        m = LogisticRegression(C=1e6,
                               max_iter=2000)
        m.fit(x, y)
        return b, float(m.coef_[0][0]), \
            float(m.intercept_[0])
    except Exception:
        return b, np.nan, np.nan


# ---------------- cohort --------------------
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
lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
D = mm[["subject_id", "hadm_id"] + CT].merge(
    mim_ehr[["hadm_id"] + EH], on="hadm_id",
    how="inner").merge(
    lab, on=["subject_id", "hadm_id"],
    how="inner")
print("")
print("cohort:", len(D))
print("  EHR %d   CTPA %d   raw %d"
      % (len(EH), len(CT), len(EH) + len(CT)))
print("  with k=10 variates: %d columns,"
      " of which 20 are canonical"
      % (len(EH) + len(CT) + 20))
print("  sqrt default draws %d per split, so"
      " under two canonical variates"
      % int(round(np.sqrt(
          len(EH) + len(CT) + 20))),
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
    Xa = d[EH].values.astype(float)
    Xb = d[CT].values.astype(float)
    bs_ = BASE.get(oc, {})
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())))
    if bs_:
        print("  untuned forest %.4f   late"
              " %.4f" % (bs_["rf"],
                         bs_["late"]),
              flush=True)

    la = np.array([roc_auc_score(
        y, oof_late(Xa, Xb, y, grp, s))
        for s in SEEDS])
    ref = la.mean()
    rows.append({
        "outcome": oc, "test": "ref",
        "mtry": np.nan, "k": np.nan,
        "leaf": np.nan, "depth": np.nan,
        "prod": 0, "mean": ref,
        "sd": la.std(ddof=1),
        "vs_late": 0.0})

    # baseline: the untuned forest
    aa = np.array([roc_auc_score(
        y, oof_rf(Xa, Xb, y, grp, s))
        for s in SEEDS])
    base = aa.mean()
    print("  reproduced untuned  %.4f"
          " (SD %.4f)   %+.4f vs late"
          % (base, aa.std(ddof=1),
             base - ref), flush=True)
    rows.append({
        "outcome": oc, "test": "untuned",
        "mtry": np.nan, "k": 10, "leaf": 10,
        "depth": 6, "prod": 0, "mean": base,
        "sd": aa.std(ddof=1),
        "vs_late": base - ref})

    # ---- TEST 1: mtry x CCA rank ----
    print("")
    print("  mtry x CCA RANK  (the two"
          " interact: more variates means more")
    print("  columns to draw from)")
    print("  %-8s" % "mtry", end="")
    for kk in K_GRID:
        print(" %9s" % ("k=%d" % kk), end="")
    print("      gain over late")
    best = None
    for mf in MTRY_FRAC:
        line = "  %-8s" % ("%.0f%%" % (100 * mf))
        for kk in K_GRID:
            try:
                aa = np.array([
                    roc_auc_score(
                        y, oof_rf(
                            Xa, Xb, y, grp, s,
                            k=kk, mfrac=mf))
                    for s in SEEDS])
            except Exception:
                line += " %9s" % "fail"
                continue
            line += " %+9.4f" % (aa.mean()
                                 - ref)
            rows.append({
                "outcome": oc,
                "test": "mtry_k",
                "mtry": mf, "k": kk,
                "leaf": 10, "depth": 6,
                "prod": 0, "mean": aa.mean(),
                "sd": aa.std(ddof=1),
                "vs_late": aa.mean() - ref})
            if best is None or \
                    aa.mean() > best[0]:
                best = (aa.mean(), mf, kk)
        print(line, flush=True)
    if best:
        print("  best mtry=%.0f%% k=%d  %.4f"
              "  %+.4f over the untuned forest"
              % (100 * best[1], best[2],
                 best[0], best[0] - base),
              flush=True)
    bmf = best[1] if best else 0.25
    bk = best[2] if best else 10
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)

    # ---- TEST 2: leaf size and depth ----
    print("")
    print("  LEAF SIZE x DEPTH, at the best"
          " mtry and k")
    print("  one clinical study tuned the leaf"
          " minimum to 161; ours was 10")
    print("  %-8s" % "leaf", end="")
    for dp in DEPTH:
        print(" %9s" % ("d=%s" % dp), end="")
    print("")
    best2 = None
    for lf in LEAF:
        line = "  %-8d" % lf
        for dp in DEPTH:
            try:
                aa = np.array([
                    roc_auc_score(
                        y, oof_rf(
                            Xa, Xb, y, grp, s,
                            k=bk, mfrac=bmf,
                            leaf=lf, depth=dp))
                    for s in SEEDS])
            except Exception:
                line += " %9s" % "fail"
                continue
            line += " %+9.4f" % (aa.mean()
                                 - ref)
            rows.append({
                "outcome": oc,
                "test": "leaf_depth",
                "mtry": bmf, "k": bk,
                "leaf": lf,
                "depth": (-1 if dp is None
                          else dp),
                "prod": 0, "mean": aa.mean(),
                "sd": aa.std(ddof=1),
                "vs_late": aa.mean() - ref})
            if best2 is None or \
                    aa.mean() > best2[0]:
                best2 = (aa.mean(), lf, dp)
        print(line, flush=True)
    if best2:
        print("  best leaf=%d depth=%s  %.4f"
              % (best2[1], best2[2], best2[0]),
              flush=True)
    blf = best2[1] if best2 else 10
    bdp = best2[2] if best2 else 6
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)

    # ---- TEST 3: cross-modal products ----
    print("")
    print("  CROSS-MODAL PRODUCTS, on top of"
          " the tuned setting")
    for pr in (0, 3, 5):
        try:
            aa = np.array([
                roc_auc_score(
                    y, oof_rf(Xa, Xb, y, grp,
                              s, k=bk,
                              mfrac=bmf,
                              leaf=blf,
                              depth=bdp,
                              prod=pr))
                for s in SEEDS])
        except Exception as exc:
            print("    prod=%d FAILED %s"
                  % (pr, repr(exc)[:40]))
            continue
        print("    prod=%-2d (%3d extra cols)"
              "  %.4f (SD %.4f)  %+.4f"
              % (pr, pr * pr, aa.mean(),
                 aa.std(ddof=1),
                 aa.mean() - ref), flush=True)
        rows.append({
            "outcome": oc, "test": "prod",
            "mtry": bmf, "k": bk, "leaf": blf,
            "depth": (-1 if bdp is None
                      else bdp), "prod": pr,
            "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_late": aa.mean() - ref})

    # ---- TEST 4: ten seeds and calibration --
    print("")
    print("  FINAL, TEN SEEDS, at mtry=%.0f%%"
          " k=%d leaf=%d depth=%s"
          % (100 * bmf, bk, blf, bdp))
    aa, br, sl, ic = [], [], [], []
    p42 = None
    for s in SEEDS10:
        p = oof_rf(Xa, Xb, y, grp, s, k=bk,
                   mfrac=bmf, leaf=blf,
                   depth=bdp)
        aa.append(roc_auc_score(y, p))
        b_, s_, i_ = calib_stats(p, y)
        br.append(b_)
        sl.append(s_)
        ic.append(i_)
        if s == 42:
            p42 = p
    aa = np.array(aa)
    print("    AUROC %.4f (SD %.4f, range"
          " %.4f-%.4f)"
          % (aa.mean(), aa.std(ddof=1),
             aa.min(), aa.max()))
    print("    Brier %.5f   slope %.3f"
          "   intercept %+.3f"
          % (np.nanmean(br), np.nanmean(sl),
             np.nanmean(ic)))
    print("    a slope below 1 means"
          " over-confident; forests usually are",
          flush=True)
    rows.append({
        "outcome": oc, "test": "final",
        "mtry": bmf, "k": bk, "leaf": blf,
        "depth": (-1 if bdp is None else bdp),
        "prod": 0, "mean": aa.mean(),
        "sd": aa.std(ddof=1),
        "lo": aa.min(), "hi": aa.max(),
        "brier": np.nanmean(br),
        "slope": np.nanmean(sl),
        "intercept": np.nanmean(ic),
        "vs_late": aa.mean() - ref})

    # post-hoc calibration, out of fold
    if p42 is not None:
        print("")
        print("  POST-HOC CALIBRATION")
        b0, s0, i0 = calib_stats(p42, y)
        print("    %-10s Brier %.5f   slope"
              " %.3f   AUROC %.4f"
              % ("raw", b0, s0,
                 roc_auc_score(y, p42)))
        for nm in ("platt", "isotonic"):
            q = np.full(len(y), np.nan)
            cv = StratifiedGroupKFold(
                n_splits=NFOLD, shuffle=True,
                random_state=42)
            for tr, te in cv.split(
                    p42.reshape(-1, 1), y,
                    grp):
                try:
                    if nm == "platt":
                        x = np.log(np.clip(
                            p42, 1e-6,
                            1 - 1e-6)
                            / (1 - np.clip(
                                p42, 1e-6,
                                1 - 1e-6)))
                        c = LogisticRegression(
                            C=1e6,
                            max_iter=2000)
                        c.fit(
                            x[tr].reshape(
                                -1, 1), y[tr])
                        q[te] = c.predict_proba(
                            x[te].reshape(
                                -1, 1))[:, 1]
                    else:
                        c = IsotonicRegression(
                            out_of_bounds=
                            "clip")
                        c.fit(p42[tr], y[tr])
                        q[te] = c.predict(
                            p42[te])
                except Exception:
                    q[te] = p42[te]
            ok = np.isfinite(q)
            b_, s_, i_ = calib_stats(q[ok],
                                     y[ok])
            print("    %-10s Brier %.5f"
                  "   slope %.3f   AUROC %.4f"
                  % (nm, b_, s_,
                     roc_auc_score(y[ok],
                                   q[ok])),
                  flush=True)
            rows.append({
                "outcome": oc,
                "test": "calib_" + nm,
                "mtry": bmf, "k": bk,
                "leaf": blf,
                "depth": (-1 if bdp is None
                          else bdp), "prod": 0,
                "mean": roc_auc_score(y[ok],
                                      q[ok]),
                "sd": np.nan, "brier": b_,
                "slope": s_, "intercept": i_,
                "vs_late": np.nan})

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
print("mtry x CCA RANK, gain over late")
m = r[r["test"] == "mtry_k"]
for oc in OUTS:
    z = m[m["outcome"] == oc]
    if not len(z):
        continue
    print("")
    print("  " + oc)
    print(z.pivot_table(index="mtry",
                        columns="k",
                        values="vs_late")
          .round(4).to_string())

print("")
print("DOES RAISING mtry HELP?")
print("  the sqrt default considers fewer than"
      " two canonical variates per split")
if len(m):
    z = m.groupby("mtry")["vs_late"].mean()
    print(z.round(4).to_string())
    print("")
    print("  best mtry fraction overall: %.0f%%"
          % (100 * z.idxmax()))

print("")
print("LEAF SIZE x DEPTH")
ld = r[r["test"] == "leaf_depth"]
for oc in OUTS:
    z = ld[ld["outcome"] == oc]
    if not len(z):
        continue
    print("")
    print("  " + oc)
    print(z.pivot_table(index="leaf",
                        columns="depth",
                        values="vs_late")
          .round(4).to_string())

print("")
print("CROSS-MODAL PRODUCTS")
pz = r[r["test"] == "prod"]
if len(pz):
    print(pz.pivot_table(index="prod",
                         columns="outcome",
                         values="vs_late")
          .round(4).to_string())
    print("")
    print("  prod=0 is the tuned forest"
          " without products")

print("")
print("=" * 74)
print("TUNED vs UNTUNED, ten seeds")
fz = r[r["test"] == "final"]
uz = r[r["test"] == "untuned"]
print("  %-18s %9s %9s %9s %9s"
      % ("outcome", "untuned", "tuned",
         "gain", "vs late"))
for oc in OUTS:
    a = uz[uz["outcome"] == oc]
    b = fz[fz["outcome"] == oc]
    if len(a) and len(b):
        print("  %-18s %9.4f %9.4f %+9.4f"
              " %+9.4f"
              % (oc, a["mean"].iloc[0],
                 b["mean"].iloc[0],
                 b["mean"].iloc[0]
                 - a["mean"].iloc[0],
                 b["vs_late"].iloc[0]))

print("")
print("THE TUNED SETTINGS")
if len(fz):
    print(fz[["outcome", "mtry", "k", "leaf",
              "depth", "mean", "sd"]]
          .round(4).to_string(index=False))
    print("")
    print("  a setting that is similar across"
          " outcomes is more trustworthy than")
    print("  four different optima, which would"
          " suggest fitting to noise")

print("")
print("CALIBRATION")
cz = r[r["test"].str.startswith("calib_")
       | (r["test"] == "final")]
if len(cz):
    print(cz[["outcome", "test", "brier",
              "slope", "intercept", "mean"]]
          .round(4).to_string(index=False))
    print("")
    print("  slope 1.0 and intercept 0.0 are"
          " ideal. Forests are usually")
    print("  over-confident, so a slope below 1"
          " on the raw rows would be expected")
    print("  and the calibrated rows should"
          " move it toward 1 without changing")
    print("  AUROC, since both transforms are"
          " monotone.")
print("")
print("saved", DEST, r.shape)
keep_awake(False)