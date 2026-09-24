"""Six learners crossed with six EHR+CTPA
representations. TabPFN runs separately in
script 107 because of its run time on CPU.

Script 105 found that PCA helps L2 logistic
regression and hurts every tree learner, so the
learner and the representation interact.

REPRESENTATIONS
  raw74     the 74 raw features
  pca8      PCA to 8 components
  spca      supervised PCA, p<0.25 k5
  lol8      LOL, 8 components
  cca       canonical variates only
  cca_aug   raw features plus canonical variates

REFERENCE ROWS, with an L2 head
  late_wsrc the incumbent
  coop      cooperative learning at its best rho;
            it augments rows rather than columns,
            so it cannot be paired with a tree

Every transform is fitted inside each training
fold, including the supervised ones.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\learner_repr.csv
  results\\learner_repr_log.txt
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats as st
from scipy import linalg as sla
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
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

HAVE_CAT = False
try:
    from catboost import CatBoostClassifier
    HAVE_CAT = True
except Exception as exc:
    print("catboost unavailable:",
          repr(exc)[:60])

PROC, FIG = f.PROC, f.FIG
SEEDS = [42, 7, 13, 1, 2]
NFOLD = 5
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
PCA_K = 8
SPCA_THR, SPCA_K = 0.25, 5
LOL_K = 8
CCA_REG, CCA_K = 0.1, 10
RHOS = [0.0, 0.5, 1.0, 2.0, 5.0]
CT_LO, CT_HI = -48.0, 24.0
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(PROC,
                    "learner_repr.csv")


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


def rcca(X, Y, reg=0.1, k=10):
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


# ---------------- the learners --------------
def learn_l2(Xt, yt, Xe, gt, extra=None):
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


def learn_plain(Xt, yt, Xe, gt, extra=None):
    m = LogisticRegression(C=1e6,
                           max_iter=5000)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def learn_ridge_var(Xt, yt, Xe, gt,
                    extra=None):
    """Shrinkage proportional to each column's
    explained variance, so the ordering is
    respected rather than ignored. Falls back to
    plain L2 when the representation carries no
    variance ratio."""
    if extra is None:
        return learn_l2(Xt, yt, Xe, gt)
    w = np.sqrt(np.clip(extra, 1e-9, None))
    w = w / w.max()
    return learn_l2(Xt * w, yt, Xe * w, gt)


def learn_gb(Xt, yt, Xe, gt, extra=None):
    m = HistGradientBoostingClassifier(
        max_depth=3, max_iter=150,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def learn_cat(Xt, yt, Xe, gt, extra=None):
    m = CatBoostClassifier(
        iterations=300, depth=3,
        learning_rate=0.05, l2_leaf_reg=6.0,
        random_seed=42, verbose=0,
        allow_writing_files=False)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


def learn_rf(Xt, yt, Xe, gt, extra=None):
    m = RandomForestClassifier(
        n_estimators=500, max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=42, n_jobs=-1)
    m.fit(Xt, yt)
    return m.predict_proba(Xe)[:, 1]


LEARNERS = [("l2", learn_l2),
            ("plain", learn_plain),
            ("ridge_var", learn_ridge_var),
            ("gb", learn_gb),
            ("rf", learn_rf)]
if HAVE_CAT:
    LEARNERS.append(("catboost", learn_cat))


# ---------------- representations -----------
def rep_raw(At, Ae, Bt, Be, yt):
    return (np.column_stack([At, Bt]),
            np.column_stack([Ae, Be]), None)


def rep_pca(At, Ae, Bt, Be, yt):
    Xt = np.column_stack([At, Bt])
    Xe = np.column_stack([Ae, Be])
    k = int(min(PCA_K, Xt.shape[1] - 1,
                len(yt) - 1))
    pc = PCA(n_components=k, random_state=42)
    return (pc.fit_transform(Xt),
            pc.transform(Xe),
            pc.explained_variance_ratio_)


def rep_spca(At, Ae, Bt, Be, yt):
    """Screen by Welch's t, THEN decompose.
    Computed directly because scipy's ttest_ind
    floods the log on the near-constant binary
    CTPA flags."""
    Xt = np.column_stack([At, Bt])
    Xe = np.column_stack([Ae, Be])
    ia, ib = yt == 1, yt == 0
    if ia.sum() >= 3 and ib.sum() >= 3:
        ma, mb = Xt[ia].mean(0), Xt[ib].mean(0)
        se = np.sqrt(
            Xt[ia].var(0, ddof=1) / ia.sum()
            + Xt[ib].var(0, ddof=1) / ib.sum())
        se = np.where(se > 1e-12, se, np.inf)
        pv = 2.0 * st.norm.sf(
            np.abs(ma - mb) / se)
    else:
        pv = np.ones(Xt.shape[1])
    pv = np.nan_to_num(pv, nan=1.0)
    sel = pv <= SPCA_THR
    if sel.sum() < SPCA_K + 1:
        idx = np.argsort(pv)[:SPCA_K + 1]
        sel = np.zeros(Xt.shape[1], bool)
        sel[idx] = True
    k = int(min(SPCA_K, sel.sum() - 1,
                len(yt) - 1))
    if k < 1:
        return Xt[:, sel], Xe[:, sel], None
    pc = PCA(n_components=k, random_state=42)
    return (pc.fit_transform(Xt[:, sel]),
            pc.transform(Xe[:, sel]),
            pc.explained_variance_ratio_)


def rep_lol(At, Ae, Bt, Be, yt):
    """Class mean-difference vector plus the top
    principal components, orthogonalised."""
    Xt = np.column_stack([At, Bt])
    Xe = np.column_stack([Ae, Be])
    d = (Xt[yt == 1].mean(0)
         - Xt[yt == 0].mean(0))
    nd = np.linalg.norm(d)
    if nd < 1e-9:
        d = np.ones(Xt.shape[1])
        nd = np.linalg.norm(d)
    d = (d / nd).reshape(-1, 1)
    k = int(min(LOL_K - 1, Xt.shape[1] - 1,
                len(yt) - 2))
    if k >= 1:
        pc = PCA(n_components=k,
                 random_state=42)
        pc.fit(Xt)
        W = np.column_stack(
            [d, pc.components_.T])
    else:
        W = d
    Q, _ = np.linalg.qr(W)
    return Xt @ Q, Xe @ Q, None


def rep_cca(At, Ae, Bt, Be, yt):
    A_, B_ = rcca(At, Bt, CCA_REG, CCA_K)
    if A_ is None:
        return rep_raw(At, Ae, Bt, Be, yt)
    ma, mb = At.mean(0), Bt.mean(0)
    return (np.column_stack([(At - ma) @ A_,
                             (Bt - mb) @ B_]),
            np.column_stack([(Ae - ma) @ A_,
                             (Be - mb) @ B_]),
            None)


def rep_ccaaug(At, Ae, Bt, Be, yt):
    zt, ze, _ = rep_cca(At, Ae, Bt, Be, yt)
    return (np.column_stack([At, Bt, zt]),
            np.column_stack([Ae, Be, ze]),
            None)


REPS = [("raw74", rep_raw),
        ("pca8", rep_pca),
        ("spca", rep_spca),
        ("lol8", rep_lol),
        ("cca", rep_cca),
        ("cca_aug", rep_ccaaug)]


def oof(Xa, Xb, y, grp, repfn, learner, seed):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        At, Ae = prep_fold(Xa[tr], Xa[te])
        Bt, Be = prep_fold(Xb[tr], Xb[te])
        Zt, Ze, extra = repfn(At, Ae, Bt, Be,
                              y[tr])
        p[te] = learner(Zt, y[tr], Ze,
                        grp[tr], extra)
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
        pa[te] = learn_l2(At, y[tr], Ae,
                          grp[tr])
        pb[te] = learn_l2(Bt, y[tr], Be,
                          grp[tr])
    pa, pb = _rank(pa), _rank(pb)
    bs, bv = -1.0, pa
    for w in np.arange(0, 1.001, 0.05):
        v = w * pb + (1 - w) * pa
        a = roc_auc_score(y, v)
        if a > bs:
            bs, bv = a, v
    return bv


def oof_coop(Xa, Xb, y, grp, rho, seed):
    """Cooperative learning augments rows, not
    columns, so it is a fitting procedure rather
    than a representation and cannot be paired
    with a tree. Kept as an L2 reference row."""
    p = np.zeros(len(y))
    s = np.sqrt(max(rho, 0.0))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        At, Ae = prep_fold(Xa[tr], Xa[te])
        Bt, Be = prep_fold(Xb[tr], Xb[te])
        top = np.column_stack([At, Bt])
        tote = np.column_stack([Ae, Be])
        if s > 0:
            Xi = np.vstack([
                top, np.column_stack(
                    [-s * At, s * Bt])])
            yi = np.concatenate([
                y[tr].astype(int),
                np.zeros(len(tr), dtype=int)])
        else:
            Xi, yi = top, y[tr].astype(int)
        best, bc = -1.0, 1.0
        for c in CS:
            try:
                m = LogisticRegression(
                    C=c, max_iter=3000)
                m.fit(Xi, yi)
                a_ = roc_auc_score(
                    y[tr],
                    m.predict_proba(top)[:, 1])
                if a_ > best:
                    best, bc = a_, c
            except Exception:
                continue
        m = LogisticRegression(C=bc,
                               max_iter=3000)
        m.fit(Xi, yi)
        p[te] = m.predict_proba(tote)[:, 1]
    return p


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
print("  EHR %d   CTPA %d   total %d"
      % (len(EH), len(CT), len(EH) + len(CT)))
print("  learners:",
      ", ".join(n for n, _ in LEARNERS))
print("  representations:",
      ", ".join(n for n, _ in REPS))
print("  TabPFN runs separately in script 107",
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
    tb = time.time()
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())),
          flush=True)

    la = np.array([roc_auc_score(
        y, oof_late(Xa, Xb, y, grp, s))
        for s in SEEDS])
    ref = la.mean()
    print("  late_wsrc (L2)   %.4f (SD %.4f)"
          % (ref, la.std(ddof=1)), flush=True)
    rows.append({
        "outcome": oc, "repr": "late",
        "learner": "l2", "mean": ref,
        "sd": la.std(ddof=1), "vs_late": 0.0})

    bc_ = None
    for rho in RHOS:
        try:
            aa = np.array([roc_auc_score(
                y, oof_coop(Xa, Xb, y, grp,
                            rho, s))
                for s in SEEDS])
        except Exception:
            continue
        rows.append({
            "outcome": oc,
            "repr": "coop_r%.1f" % rho,
            "learner": "l2", "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_late": aa.mean() - ref})
        if bc_ is None or aa.mean() > bc_[1]:
            bc_ = (rho, aa.mean(),
                   aa.std(ddof=1))
    if bc_:
        print("  coop best rho=%.1f  %.4f"
              " (SD %.4f)   %+.4f"
              % (bc_[0], bc_[1], bc_[2],
                 bc_[1] - ref), flush=True)

    for rname, repfn in REPS:
        print("")
        print("  %-8s  %-10s %8s %8s %9s"
              % (rname, "learner", "AUC",
                 "SD", "vs late"))
        for lname, fn in LEARNERS:
            try:
                aa = np.array([
                    roc_auc_score(
                        y, oof(Xa, Xb, y, grp,
                               repfn, fn, s))
                    for s in SEEDS])
            except Exception as exc:
                print("    %-20s FAILED %s"
                      % (lname,
                         repr(exc)[:40]))
                continue
            print("    %-8s %10s %8.4f"
                  " %8.4f %+9.4f"
                  % ("", lname, aa.mean(),
                     aa.std(ddof=1),
                     aa.mean() - ref),
                  flush=True)
            rows.append({
                "outcome": oc, "repr": rname,
                "learner": lname,
                "mean": aa.mean(),
                "sd": aa.std(ddof=1),
                "vs_late": aa.mean() - ref})
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
grid = r[~r["repr"].isin(["late"])
         & ~r["repr"].str.startswith("coop")]

print("")
print("=" * 74)
print("GAIN OVER late:wsrc, EVERY CELL")
for rname, _ in REPS:
    s = grid[grid["repr"] == rname]
    if not len(s):
        continue
    print("")
    print("  " + rname)
    print(s.pivot_table(index="learner",
                        columns="outcome",
                        values="vs_late")
          .round(4).to_string())

print("")
print("AVERAGE RANK ACROSS ALL 4 OUTCOMES")
piv = grid.copy()
piv["cell"] = (piv["repr"] + "/"
               + piv["learner"])
rk = piv.pivot_table(index="cell",
                     columns="outcome",
                     values="mean").rank(
    ascending=False).mean(axis=1)
print(rk.sort_values().head(15).round(2)
      .to_string())

print("")
print("BEST LEARNER WITHIN EACH"
      " REPRESENTATION")
for rname, _ in REPS:
    s = grid[grid["repr"] == rname]
    if not len(s):
        continue
    z = s.groupby("learner")["mean"].mean()
    print("  %-8s  %-10s mean %.4f"
          "   worst %-10s %.4f"
          % (rname, z.idxmax(), z.max(),
             z.idxmin(), z.min()))

print("")
print("BEST REPRESENTATION WITHIN EACH"
      " LEARNER")
print("  this is the interaction: script 105"
      " showed PCA helping L2 and hurting")
print("  every tree, so the best representation"
      " should differ by learner")
for lname, _ in LEARNERS:
    s = grid[grid["learner"] == lname]
    if not len(s):
        continue
    z = s.groupby("repr")["mean"].mean()
    print("  %-10s best %-8s %.4f"
          "   worst %-8s %.4f"
          % (lname, z.idxmax(), z.max(),
             z.idxmin(), z.min()))

print("")
print("=" * 74)
print("TOP TEN CELLS OVERALL")
b = grid.sort_values("mean",
                     ascending=False)
print(b[["outcome", "repr", "learner",
         "mean", "sd", "vs_late"]]
      .head(10).round(4)
      .to_string(index=False))

print("")
print("BEST PER OUTCOME")
for oc in OUTS:
    s = grid[grid["outcome"] == oc]
    if not len(s):
        continue
    x = s.loc[s["mean"].idxmax()]
    lt = r[(r["outcome"] == oc)
           & (r["repr"] == "late")]
    print("  %-18s %-8s / %-10s %.4f"
          "   late %.4f   %+.4f"
          % (oc, x["repr"], x["learner"],
             x["mean"],
             lt["mean"].iloc[0]
             if len(lt) else np.nan,
             x["vs_late"]))

print("")
print("WINS OVER late:wsrc")
w_ = grid[grid["vs_late"] > 0]
print("  %d of %d cells" % (len(w_),
                            len(grid)))
print("")
print("  NOTE: %d comparisons across four"
      " outcomes, so the single best cell"
      % len(grid))
print("  needs a ten-seed and bootstrap"
      " confirmation before it is trusted,")
print("  as the PCA result did.")
print("")
print("saved", DEST, r.shape)
keep_awake(False)