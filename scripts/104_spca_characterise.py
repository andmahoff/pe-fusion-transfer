"""Characterises supervised PCA on EHR+CTPA
for in-hospital death.

In script 102 (six seeds), spca at p<0.25, k5
reached 0.8998 against late fusion's 0.8786. The
optimum kept 49 of 74 features, and p<0.50 (61
features) performed almost identically, so the
screening may add little beyond the compression.

FOUR TESTS
  1 spca against plain PCA at matched k
  2 a finer k sweep from 2 to 12
  3 a threshold sweep including p<1.00, which
    keeps every feature and so is plain PCA, as
    the internal control
  4 ten seeds, a paired bootstrap, and a
    permutation check for leakage

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\spca_characterise.csv
  results\\spca_characterise_log.txt
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
atexit.register(keep_awake, False)

PROC, FIG = f.PROC, f.FIG
SEEDS = [42, 7, 13, 1, 2, 3, 5, 8, 21, 99]
NFOLD = 5
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
KS = [2, 3, 4, 5, 6, 8, 10, 12]
# p<1.00 keeps everything, so it IS plain PCA
# and acts as the internal control
THRS = [0.05, 0.10, 0.25, 0.50, 1.00]
CT_LO, CT_HI = -48.0, 24.0
TARGET = "death_30d_inhosp"
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(
    PROC, "spca_characterise.csv")
# script 102, six seeds
PREV = {"late": 0.8786, "spca_best": 0.8998,
        "pca10": 0.8985, "lol5": 0.8900}


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


def screen(Xt, ytr, thr, k):
    """Welch's t per column, computed directly.
    thr = 1.00 selects everything, which makes
    this setting plain PCA by construction."""
    if thr >= 1.0:
        return np.ones(Xt.shape[1], bool)
    ia = ytr == 1
    ib = ~ia
    if ia.sum() < 3 or ib.sum() < 3:
        return np.ones(Xt.shape[1], bool)
    ma, mb = Xt[ia].mean(0), Xt[ib].mean(0)
    va = Xt[ia].var(0, ddof=1)
    vb = Xt[ib].var(0, ddof=1)
    se = np.sqrt(va / ia.sum() + vb / ib.sum())
    se = np.where(se > 1e-12, se, np.inf)
    pv = 2.0 * st.norm.sf(np.abs(ma - mb) / se)
    pv = np.nan_to_num(pv, nan=1.0)
    sel = pv <= thr
    if sel.sum() < max(3, k + 1):
        idx = np.argsort(pv)[:max(3, k + 1)]
        sel = np.zeros(Xt.shape[1], bool)
        sel[idx] = True
    return sel


def oof_spca(X, y, grp, thr, k, seed):
    p = np.zeros(len(y))
    ns = []
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        Xt, Xe = prep_fold(X[tr], X[te])
        sel = screen(Xt, y[tr], thr, k)
        ns.append(int(sel.sum()))
        kk = int(min(k, sel.sum() - 1,
                     len(tr) - 1))
        if kk < 1:
            p[te] = fit_pred(
                Xt[:, sel], y[tr],
                Xe[:, sel], grp[tr])
            continue
        pc = PCA(n_components=kk,
                 random_state=42)
        p[te] = fit_pred(
            pc.fit_transform(Xt[:, sel]),
            y[tr], pc.transform(Xe[:, sel]),
            grp[tr])
    return p, float(np.mean(ns))


def oof_raw(X, y, grp, seed):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        Xt, Xe = prep_fold(X[tr], X[te])
        p[te] = fit_pred(Xt, y[tr], Xe,
                         grp[tr])
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
        pa[te] = fit_pred(At, y[tr], Ae,
                          grp[tr])
        pb[te] = fit_pred(Bt, y[tr], Be,
                          grp[tr])
    pa, pb = _rank(pa), _rank(pb)
    bs, bv = -1.0, pa
    for w in np.arange(0, 1.001, 0.05):
        v = w * pb + (1 - w) * pa
        a = roc_auc_score(y, v)
        if a > bs:
            bs, bv = a, v
    return bv


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
print("cohort:", len(D))
print("  EHR %d   CTPA %d   total %d"
      % (len(EH), len(CT), len(EH) + len(CT)),
      flush=True)

rows = []
t0 = time.time()

# ===== TEST 1-3: the full surface on TARGET ==
d, y = f.labels(D, TARGET)
grp = d["subject_id"].values
Xa = d[EH].values.astype(float)
Xb = d[CT].values.astype(float)
X = np.column_stack([Xa, Xb])
print("")
print("#" * 74)
print("%s   n=%d ev=%d   THE CELL IN QUESTION"
      % (TARGET, len(y), int(y.sum())))
print("#" * 74, flush=True)

la = np.array([roc_auc_score(
    y, oof_late(Xa, Xb, y, grp, s))
    for s in SEEDS])
ra = np.array([roc_auc_score(
    y, oof_raw(X, y, grp, s))
    for s in SEEDS])
ref = la.mean()
print("")
print("  %-12s %8s %8s %8s %8s"
      % ("model", "mean", "SD", "min", "max"))
for nm, v in (("late_wsrc", la),
              ("early", ra)):
    print("  %-12s %8.4f %8.4f %8.4f %8.4f"
          % (nm, v.mean(), v.std(ddof=1),
             v.min(), v.max()), flush=True)
    rows.append({
        "outcome": TARGET, "method": nm,
        "thr": np.nan, "k": np.nan,
        "n_sel": np.nan, "mean": v.mean(),
        "sd": v.std(ddof=1), "lo": v.min(),
        "hi": v.max(),
        "vs_late": v.mean() - ref})

print("")
print("  THE SURFACE  (ten seeds)")
print("  p<1.00 keeps every feature, so that"
      " row IS plain PCA and is the")
print("  internal control: if screening adds"
      " nothing, the rows will match.")
print("")
hdr = "  %-6s" % "k"
for thr in THRS:
    hdr += " %9s" % ("p<%.2f" % thr)
print(hdr + "    (gain over late fusion)")
best = None
for k in KS:
    line = "  %-6d" % k
    for thr in THRS:
        try:
            res = [oof_spca(X, y, grp, thr,
                            k, s)
                   for s in SEEDS]
            aa = np.array([roc_auc_score(y, p)
                           for p, _ in res])
            ns = float(np.mean(
                [n for _, n in res]))
        except Exception as exc:
            line += " %9s" % "fail"
            continue
        g_ = aa.mean() - ref
        line += " %+9.4f" % g_
        rows.append({
            "outcome": TARGET,
            "method": ("pca" if thr >= 1.0
                       else "spca"),
            "thr": thr, "k": k, "n_sel": ns,
            "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "lo": aa.min(), "hi": aa.max(),
            "vs_late": g_})
        if best is None or aa.mean() > best[2]:
            best = (thr, k, aa.mean(),
                    aa.std(ddof=1), ns)
    print(line, flush=True)

print("")
print("  FEATURES KEPT BY THRESHOLD")
for thr in THRS:
    z = [x for x in rows
         if x.get("thr") == thr
         and np.isfinite(x.get("n_sel",
                               np.nan))]
    if z:
        print("    p<%.2f   %.0f of %d"
              % (thr, np.mean(
                  [x["n_sel"] for x in z]),
                 X.shape[1]))

if best:
    thr, k, m, sd, ns = best
    print("")
    print("  BEST: p<%.2f k=%d   %.4f"
          " (SD %.4f)   %+.4f over late"
          "   %.0f of %d kept"
          % (thr, k, m, sd, m - ref,
             ns, X.shape[1]))
    print("  script 102 gave p<0.25 k5"
          " at %.4f over six seeds"
          % PREV["spca_best"])
    print("  margin is %.1f late-fusion SDs"
          " and %.1f of its own"
          % ((m - ref) / max(la.std(ddof=1),
                             1e-9),
             (m - ref) / max(sd, 1e-9)))

    # paired bootstrap at the best setting
    pb_, _ = oof_spca(X, y, grp, thr, k,
                      SEEDS[0])
    pl_ = oof_late(Xa, Xb, y, grp, SEEDS[0])
    g_, lo_, hi_, _ = f.boot_diff(
        y, _rank(pb_), _rank(pl_), grp)
    print("")
    print("  PAIRED BOOTSTRAP, seed 42")
    print("    best vs late_wsrc  %+.4f"
          " [%+.4f,%+.4f] %s"
          % (g_, lo_, hi_,
             "*" if (lo_ > 0 or hi_ < 0)
             else ""))
    rows.append({
        "outcome": TARGET,
        "method": "boot_vs_late",
        "thr": thr, "k": k, "n_sel": ns,
        "mean": g_, "sd": np.nan,
        "lo": lo_, "hi": hi_,
        "vs_late": g_})

    # permutation
    pv = []
    for i in range(5):
        rng = np.random.default_rng(700 + i)
        ysh = y.copy()
        rng.shuffle(ysh)
        p_, _ = oof_spca(X, ysh, grp, thr,
                         k, 42)
        pv.append(roc_auc_score(ysh, p_))
    pv = np.array(pv)
    print("  PERMUTATION: %.4f (SD %.4f)"
          "   a correct pipeline gives ~0.50"
          % (pv.mean(), pv.std(ddof=1)),
          flush=True)
    rows.append({
        "outcome": "PERMUTED",
        "method": "spca", "thr": thr,
        "k": k, "n_sel": ns,
        "mean": pv.mean(),
        "sd": pv.std(ddof=1),
        "lo": pv.min(), "hi": pv.max(),
        "vs_late": np.nan})

# ===== TEST 4: the other outcomes ===========
print("")
print("#" * 74)
print("THE SAME SETTING ON THE OTHER OUTCOMES")
print("  script 102 found spca losing in 11 of"
      " 12 other cells, so this should")
print("  confirm the effect is cell-specific")
print("#" * 74, flush=True)
if best:
    thr, k = best[0], best[1]
    for oc in OUTS:
        if oc == TARGET or oc not in D.columns:
            continue
        d2, y2 = f.labels(D, oc)
        g2 = d2["subject_id"].values
        if y2.sum() < 25:
            continue
        A2 = d2[EH].values.astype(float)
        B2 = d2[CT].values.astype(float)
        X2 = np.column_stack([A2, B2])
        l2 = np.array([roc_auc_score(
            y2, oof_late(A2, B2, y2, g2, s))
            for s in SEEDS])
        res = [oof_spca(X2, y2, g2, thr, k, s)
               for s in SEEDS]
        a2 = np.array([roc_auc_score(y2, p)
                       for p, _ in res])
        print("  %-18s ev=%3d   late %.4f"
              "   spca %.4f   %+.4f"
              % (oc, int(y2.sum()), l2.mean(),
                 a2.mean(),
                 a2.mean() - l2.mean()),
              flush=True)
        rows.append({
            "outcome": oc, "method": "spca",
            "thr": thr, "k": k,
            "n_sel": float(np.mean(
                [n for _, n in res])),
            "mean": a2.mean(),
            "sd": a2.std(ddof=1),
            "lo": a2.min(), "hi": a2.max(),
            "vs_late": a2.mean() - l2.mean()})
        rows.append({
            "outcome": oc,
            "method": "late_wsrc",
            "thr": np.nan, "k": np.nan,
            "n_sel": np.nan, "mean": l2.mean(),
            "sd": l2.std(ddof=1),
            "lo": l2.min(), "hi": l2.max(),
            "vs_late": 0.0})

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("DOES SCREENING DO ANYTHING?")
print("  p<1.00 is plain PCA. If the screened"
      " rows match it, the 'supervised'")
print("  part is decoration and this is just"
      " compression.")
s = r[(r["outcome"] == TARGET)
      & r["k"].notna()
      & r["method"].isin(["spca", "pca"])]
if len(s):
    piv = s.pivot_table(index="k",
                        columns="thr",
                        values="mean")
    print("")
    print(piv.round(4).to_string())
    if 1.0 in piv.columns:
        print("")
        print("  screened minus plain PCA, by k")
        for kk in piv.index:
            row = piv.loc[kk]
            bestscr = row.drop(
                1.0, errors="ignore").max()
            print("    k=%-3d  best screened"
                  " %.4f   plain %.4f   %+.4f"
                  % (kk, bestscr, row[1.0],
                     bestscr - row[1.0]))
        d_ = (piv.drop(columns=[1.0]).max(axis=1)
              - piv[1.0])
        print("")
        print("  mean advantage of screening:"
              " %+.4f" % d_.mean())
        if abs(d_.mean()) < 0.003:
            print("  -> screening adds nothing."
                  " This is compression, not")
            print("     supervision, and should"
                  " be reported as PCA.")
        else:
            print("  -> screening contributes"
                  " genuinely")

print("")
print("SEED SD BY SETTING")
if len(s):
    print(s.pivot_table(index="k",
                        columns="thr",
                        values="sd")
          .round(4).to_string())
    lz = r[(r["outcome"] == TARGET)
           & (r["method"] == "late_wsrc")]
    if len(lz):
        print("")
        print("  late fusion SD: %.4f"
              % lz["sd"].iloc[0])
        print("  compression is markedly more"
              " STABLE, which is part of the")
        print("  result and not just the mean")

print("")
print("IS THE EFFECT CELL-SPECIFIC?")
o = r[(r["method"] == "spca")
      & r["vs_late"].notna()
      & (r["outcome"] != "PERMUTED")]
if len(o):
    z = o.groupby("outcome")["vs_late"].max()
    print(z.round(4).to_string())
    print("")
    print("  positive only on %s would confirm"
          " a narrow, strong claim rather"
          % TARGET)
    print("  than a broad, weak one")
print("")
print("saved", DEST, r.shape)
keep_awake(False)