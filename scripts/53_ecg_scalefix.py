"""Fixes the rank-scale defect in script 52, then
tries stacking.

Script 52 ranked ecg_traj within the multi-ECG
subgroup while ranking every other modality
across the full cohort. Inside the subgroup the
trajectory score therefore spanned 0 to 1 while
the others occupied a compressed interior slice,
so the weight grid shrank trajectory to control
its influence. That is a defect, not a finding.

Fixes tested here:
  strat_fixed   - every component re-ranked
                  within the subgroup before
                  fusing
  stack         - logistic regression on the
                  modality scores plus a
                  has-trajectory flag and its
                  interactions, so the model
                  learns the weighting itself
  select        - per-stratum model selection
                  rather than blending

Also reconciles the 972 vs 754 cohort
discrepancy between scripts 48 and 52.
Run in venv (analysis).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
SEED, NFOLD = 42, 5
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
OUTS = ["death_30d", "death_30d_inhosp",
        "composite_30d", "cv_first"]


def _rank(p):
    """Rank ignoring NaN, NaN preserved."""
    s = pd.Series(p)
    return s.rank(pct=True).values


def rank_sub(p, m):
    """Rank within a mask only."""
    out = np.full(len(p), np.nan)
    if m.sum() > 1:
        out[m] = pd.Series(
            p[m]).rank(pct=True).values
    return out


def simplex(k, step):
    n = int(round(1.0 / step))

    def rec(m, rem):
        if m == 1:
            yield (rem,)
            return
        for i in range(rem + 1):
            for t in rec(m - 1, rem - i):
                yield (i,) + t
    for w in rec(k, n):
        yield tuple(x * step for x in w)


def grid_w(ps, y):
    k = len(ps)
    if k == 1:
        return (1.0,)
    step = 0.05 if k <= 3 else 0.10
    best, bw = -1.0, tuple([1.0 / k] * k)
    if len(np.unique(y)) < 2:
        return bw
    for w in simplex(k, step):
        a = roc_auc_score(
            y, sum(wi * p
                   for wi, p in zip(w, ps)))
        if a > best:
            best, bw = a, w
    return bw


def fit_fold(Xa, ya, ga, Xb):
    im = SimpleImputer(strategy="median")
    a = im.fit_transform(Xa)
    b = im.transform(Xb)
    sc = StandardScaler()
    a, b = sc.fit_transform(a), sc.transform(b)
    best, bc = -1.0, 0.1
    k = min(3, max(2, int(ya.sum()) // 8))
    try:
        icv = StratifiedGroupKFold(
            n_splits=k, shuffle=True,
            random_state=SEED)
        for c in CS:
            q = np.zeros(len(ya))
            for t2, v2 in icv.split(a, ya, ga):
                m = LogisticRegression(
                    C=c, max_iter=5000)
                m.fit(a[t2], ya[t2])
                q[v2] = m.predict_proba(
                    a[v2])[:, 1]
            s = roc_auc_score(ya, q)
            if s > best:
                best, bc = s, c
    except Exception:
        bc = 0.1
    m = LogisticRegression(C=bc, max_iter=5000)
    m.fit(a, ya)
    return m.predict_proba(b)[:, 1]


def oof(X, y, grp, folds, mask=None):
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        if mask is not None:
            tr = tr[mask[tr]]
            te = te[mask[te]]
        if len(te) == 0 or len(tr) < 40:
            continue
        if y[tr].sum() < 5:
            continue
        p[te] = fit_fold(X[tr], y[tr],
                         grp[tr], X[te])
    return p


def fuse_global(P, names, y, folds):
    out = np.zeros(len(y))
    ws = []
    for tr, te in folds:
        w = grid_w([P[n][tr] for n in names],
                   y[tr])
        ws.append(w)
        out[te] = sum(wi * P[n][te]
                      for wi, n in zip(w, names))
    return out, np.mean(ws, axis=0)


def fuse_strat_fixed(P, spec, y, folds, multi):
    """Every component re-ranked within the
    subgroup before fusing, and the fused score
    re-ranked within subgroup afterwards, so all
    strata land on one scale."""
    out = np.zeros(len(y))
    ws = {False: [], True: []}
    for tr, te in folds:
        for flag in (False, True):
            trm = tr[multi[tr] == flag]
            tem = te[multi[te] == flag]
            if len(tem) == 0:
                continue
            names = spec[flag]
            if len(trm) < 60 or \
                    y[trm].sum() < 10:
                names, trm = spec[False], tr
            # re-rank inside this stratum
            A = {}
            for n in names:
                v = P[n]
                mm = np.zeros(len(y), bool)
                mm[trm] = True
                mm[tem] = True
                A[n] = rank_sub(
                    np.nan_to_num(
                        v, nan=np.nanmedian(v)),
                    mm)
            w = grid_w([A[n][trm] for n in names],
                       y[trm])
            ws[flag].append(w)
            v = sum(wi * A[n][tem]
                    for wi, n in zip(w, names))
            out[tem] = pd.Series(v).rank(
                pct=True).values
    m = {k: (np.mean(v, axis=0) if v else None)
         for k, v in ws.items()}
    return out, m


def fuse_stack(P, base, y, folds, multi):
    """Logistic regression on the modality
    scores plus a has-trajectory flag and its
    interactions. The model learns when to use
    trajectory rather than being told."""
    out = np.zeros(len(y))
    med = np.nanmedian(P["ecg_t"])
    t = np.where(np.isfinite(P["ecg_t"]),
                 P["ecg_t"], med)
    cols = [P[n] for n in base]
    X = np.column_stack(
        cols + [t, multi.astype(float),
                t * multi,
                P["ehr"] * multi])
    for tr, te in folds:
        sc = StandardScaler()
        a = sc.fit_transform(X[tr])
        b = sc.transform(X[te])
        m = LogisticRegression(
            C=0.1, max_iter=5000)
        m.fit(a, y[tr])
        out[te] = m.predict_proba(b)[:, 1]
    return _rank(out)


def fuse_select(cands, y, folds, multi):
    """Per stratum, use whichever candidate
    scores best in the training fold."""
    out = np.zeros(len(y))
    picks = {False: [], True: []}
    for tr, te in folds:
        for flag in (False, True):
            trm = tr[multi[tr] == flag]
            tem = te[multi[te] == flag]
            if len(tem) == 0 or len(trm) < 40:
                continue
            best, bn = -1.0, None
            for nm, v in cands.items():
                vv = np.nan_to_num(
                    v, nan=np.nanmedian(v))
                if len(np.unique(y[trm])) < 2:
                    continue
                a = roc_auc_score(y[trm],
                                  vv[trm])
                if a > best:
                    best, bn = a, nm
            if bn is None:
                continue
            picks[flag].append(bn)
            vv = np.nan_to_num(
                cands[bn],
                nan=np.nanmedian(cands[bn]))
            out[tem] = pd.Series(
                vv[tem]).rank(pct=True).values
    return out, picks


traj = pd.read_csv(os.path.join(
    PROC, "ecg_trajectory.csv"))
stat = pd.read_csv(os.path.join(
    PROC, "ecg_derived_v3.csv"))
mim = f.load_mimic()
ins = f.load_inspect()

tcov = traj.notna().mean()
tcols = [c for c in traj.columns
         if c not in ("hadm_id", "subject_id")
         and tcov[c] >= 0.60]
scov = stat.notna().mean()
scols = [c for c in stat.columns
         if c not in ("subject_id", "hadm_id")
         and pd.api.types.is_numeric_dtype(
             stat[c])
         and scov[c] >= 0.60]
print("static cols:", len(scols),
      " trajectory cols:", len(tcols))

print("")
print("COHORT RECONCILIATION")
tj = mim[["hadm_id"]].merge(
    traj[["hadm_id"] + tcols],
    on="hadm_id", how="left")
n10 = int((tj[tcols].notna().sum(axis=1)
           >= 10).sum())
print("  admissions in traj file:", len(traj))
print("  in mimic cohort with >=10 traj"
      " features:", n10)
print("  (script 48 reported 972,"
      " script 52 reported 754)")
sq = pd.read_csv(os.path.join(
    FIG, "p_ecg_harm_death_30d.csv"))
both = mim[["hadm_id"]].merge(
    traj[["hadm_id"] + tcols], on="hadm_id",
    how="left")
mm = (both[tcols].notna().sum(axis=1)
      >= 10).values
mm &= mim["hadm_id"].isin(
    set(sq["hadm_id"])).values
print("  and with a stored p_ecg:",
      int(mm.sum()))

rows = []
for oc in OUTS:
    if oc not in mim.columns:
        continue
    d = mim
    if oc == "cv_first":
        d = mim[mim["death_first"] == 0]
    y = pd.to_numeric(d[oc], errors="coerce")
    y = y.fillna(0).astype(int).values
    grp = d["subject_id"].values

    a = d[["hadm_id"]].merge(
        stat[["hadm_id"] + scols],
        on="hadm_id", how="left")
    a = a.merge(traj[["hadm_id"] + tcols],
                on="hadm_id", how="left")
    ok = (a[scols].notna().sum(axis=1)
          >= 5).values
    yy, gg = y[ok], grp[ok]
    if yy.sum() < 30:
        continue
    Xs = a.loc[ok, scols].values.astype(float)
    Xt = a.loc[ok, tcols].values.astype(float)
    multi = (a.loc[ok, tcols].notna()
             .sum(axis=1) >= 10).values

    print("")
    print("=" * 66)
    print("%s   n=%d ev=%d   single %d (ev %d)"
          " | multi %d (ev %d)"
          % (oc, int(ok.sum()), int(yy.sum()),
             int((~multi).sum()),
             int(yy[~multi].sum()),
             int(multi.sum()),
             int(yy[multi].sum())))

    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    folds = list(cv.split(Xs, yy, gg))

    src = ("death_30d"
           if oc == "death_30d_inhosp" else oc)
    se, ysrc = f.labels(ins, src)
    p1, q1 = f.prep(
        se[f.EHR_COLS].values.astype(float),
        d.loc[ok, f.EHR_COLS].values
        .astype(float))
    p2, q2 = f.prep(
        se[f.CTPA_COLS].values.astype(float),
        d.loc[ok, f.CTPA_COLS].values
        .astype(float))

    P = {"ehr": _rank(f.fit_lr(
        p1, ysrc, "ehr").predict_proba(
        q1)[:, 1]),
        "ctpa": _rank(f.fit_lr(
            p2, ysrc, "ctpa").predict_proba(
            q2)[:, 1]),
        "ecg_s": _rank(oof(Xs, yy, gg, folds))}
    pt = oof(Xt, yy, gg, folds, mask=multi)
    P["ecg_t"] = pt

    print("  ehr %.4f | ctpa %.4f | ecg_s %.4f"
          % (roc_auc_score(yy, P["ehr"]),
             roc_auc_score(yy, P["ctpa"]),
             roc_auc_score(yy, P["ecg_s"])))
    fm = np.isfinite(pt)
    if fm.sum() > 50 and yy[fm].sum() > 10:
        print("  ecg_t on multi only: %.4f"
              " (n=%d ev=%d)"
              % (roc_auc_score(yy[fm], pt[fm]),
                 int(fm.sum()),
                 int(yy[fm].sum())))

    M = {}
    M["two"], _ = fuse_global(
        P, ["ctpa", "ehr"], yy, folds)
    M["global_static"], _ = fuse_global(
        P, ["ctpa", "ecg_s", "ehr"], yy, folds)
    M["strat_fixed"], wf = fuse_strat_fixed(
        P, {False: ["ctpa", "ecg_s", "ehr"],
            True: ["ctpa", "ecg_s", "ecg_t",
                   "ehr"]},
        yy, folds, multi)
    M["strat_fixed_ctl"], _ = fuse_strat_fixed(
        P, {False: ["ctpa", "ecg_s", "ehr"],
            True: ["ctpa", "ecg_s", "ehr"]},
        yy, folds, multi)
    M["stack"] = fuse_stack(
        P, ["ctpa", "ecg_s", "ehr"],
        yy, folds, multi)
    M["select"], picks = fuse_select(
        {"two": M["two"],
         "global_static": M["global_static"],
         "strat_fixed": M["strat_fixed"]},
        yy, folds, multi)

    base = M["two"]
    print("")
    print("  %-18s %7s %7s %s"
          % ("model", "AUC", "AP",
             "gain vs ctpa+ehr"))
    for nm in ["two", "global_static",
               "strat_fixed_ctl",
               "strat_fixed", "stack",
               "select"]:
        v = M[nm]
        au = roc_auc_score(yy, v)
        if nm == "two":
            g = lo = hi = 0.0
        else:
            g, lo, hi, _ = f.boot_diff(
                yy, v, base, gg)
        star = ("*" if (lo > 0 or hi < 0)
                else " ")
        print("  %-18s %.4f  %.4f  %+.4f"
              " [%+.4f,%+.4f] %s"
              % (nm, au,
                 average_precision_score(yy, v),
                 g, lo, hi, star))
        rows.append({
            "outcome": oc, "model": nm,
            "n": int(ok.sum()),
            "ev": int(yy.sum()), "auc": au,
            "gain": g, "lo": lo, "hi": hi,
            "sig": int(lo > 0 or hi < 0)})

    print("")
    print("  KEY CONTRAST: trajectory with"
          " scale fixed")
    g, lo, hi, _ = f.boot_diff(
        yy, M["strat_fixed"],
        M["strat_fixed_ctl"], gg)
    print("    strat_fixed vs its control"
          "  %+.4f [%+.4f,%+.4f] %s"
          % (g, lo, hi,
             "*" if (lo > 0 or hi < 0) else ""))
    rows.append({
        "outcome": oc,
        "model": "traj_effect_scalefixed",
        "n": int(ok.sum()),
        "ev": int(yy.sum()), "auc": np.nan,
        "gain": g, "lo": lo, "hi": hi,
        "sig": int(lo > 0 or hi < 0)})

    if wf.get(True) is not None:
        print("")
        print("  FITTED WEIGHTS (scale-fixed)")
        print("    single [ctpa,ecg_s,ehr] =",
              np.round(wf[False], 2))
        print("    multi  [ctpa,ecg_s,ecg_t,"
              "ehr] =", np.round(wf[True], 2))
        print("    (script 52 gave multi"
              " [0.05, 0.0, 0.05, 0.9])")

    print("")
    print("  WITHIN SUBGROUP vs ctpa+ehr")
    for gn, gm in (("single", ~multi),
                   ("multi", multi)):
        if yy[gm].sum() < 12:
            continue
        a0 = roc_auc_score(yy[gm], base[gm])
        line = "    %-7s ev=%3d  2mod %.4f" % (
            gn, int(yy[gm].sum()), a0)
        for nm in ["strat_fixed", "stack"]:
            av = roc_auc_score(yy[gm],
                               M[nm][gm])
            line += "  %s %.4f (%+.4f)" % (
                nm[:5], av, av - a0)
        print(line)

r = pd.DataFrame(rows)
r.to_csv(os.path.join(
    PROC, "ecg_scalefix.csv"), index=False)

print("")
print("=" * 66)
print("GAIN OVER ctpa+ehr")
print(r[r["model"] != "traj_effect_scalefixed"]
      .pivot_table(index="model",
                   columns="outcome",
                   values="gain")
      .round(4).to_string())
print("")
print("significant cells:")
print(r.groupby("model")["sig"]
      .agg(["sum", "count"]).to_string())

print("")
print("TRAJECTORY EFFECT WITH SCALE FIXED")
q = r[r["model"] == "traj_effect_scalefixed"]
print(q[["outcome", "gain", "lo", "hi",
         "sig"]].round(4).to_string(
    index=False))

print("")
print("saved ecg_scalefix.csv", r.shape)