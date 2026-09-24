"""Stratified fusion weights, so the trajectory
modality is not averaged away.

Script 49 fitted one weight vector across the
full cohort, where 54% of patients have no
trajectory features, so the weight grid
optimised for the majority.

Here each subgroup gets its own weights, fitted
inside each training fold:
  single-ECG  -> ctpa + ecg_static + ehr
  multi-ECG   -> ctpa + ecg_static + ecg_traj
                 + ehr

CONTROL: strat_static gives both groups their own
weights but no trajectory. If that alone recovers
the gain, stratification is doing the work rather
than trajectory.
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
    return pd.Series(p).rank(pct=True).values


def simplex(k, step):
    """Weight vectors on the k-simplex."""
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
        v = sum(wi * p for wi, p in zip(w, ps))
        a = roc_auc_score(y, v)
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
    """OOF over all rows, or over mask only with
    NaN elsewhere."""
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


def fuse_global(P, names, y, grp, folds):
    out = np.zeros(len(y))
    ws = []
    for tr, te in folds:
        w = grid_w([P[n][tr] for n in names],
                   y[tr])
        ws.append(w)
        out[te] = sum(
            wi * P[n][te]
            for wi, n in zip(w, names))
    return out, np.mean(ws, axis=0)


def fuse_strat(P, spec, y, grp, folds, multi):
    """spec = {False: [...], True: [...]}"""
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
                names = spec[False]
                trm = tr
            w = grid_w(
                [P[n][trm] for n in names],
                y[trm])
            ws[flag].append(w)
            out[tem] = sum(
                wi * P[n][tem]
                for wi, n in zip(w, names))
    m = {}
    for flag in (False, True):
        if ws[flag]:
            m[flag] = np.mean(ws[flag], axis=0)
    return out, m


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
    print("%s   n=%d  ev=%d" % (oc,
                                int(ok.sum()),
                                int(yy.sum())))
    print("  single %d (ev %d) | multi %d"
          " (ev %d)"
          % (int((~multi).sum()),
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

    P = {}
    P["ehr"] = _rank(f.fit_lr(
        p1, ysrc, "ehr").predict_proba(
        q1)[:, 1])
    P["ctpa"] = _rank(f.fit_lr(
        p2, ysrc, "ctpa").predict_proba(
        q2)[:, 1])
    P["ecg_s"] = _rank(
        oof(Xs, yy, gg, folds))
    pt = oof(Xt, yy, gg, folds, mask=multi)
    r = np.full(len(yy), np.nan)
    m2 = np.isfinite(pt)
    r[m2] = _rank(pt[m2])
    P["ecg_t"] = r

    print("  ehr %.4f | ctpa %.4f"
          " | ecg_static %.4f"
          % (roc_auc_score(yy, P["ehr"]),
             roc_auc_score(yy, P["ctpa"]),
             roc_auc_score(yy, P["ecg_s"])))
    if m2.sum() > 50 and yy[m2].sum() > 10:
        print("  ecg_traj (multi only) %.4f"
              " on n=%d ev=%d"
              % (roc_auc_score(yy[m2],
                               P["ecg_t"][m2]),
                 int(m2.sum()),
                 int(yy[m2].sum())))

    M = {}
    M["two"], w2 = fuse_global(
        P, ["ctpa", "ehr"], yy, gg, folds)
    M["global_static"], wg = fuse_global(
        P, ["ctpa", "ecg_s", "ehr"],
        yy, gg, folds)
    M["strat_static"], ws1 = fuse_strat(
        P, {False: ["ctpa", "ecg_s", "ehr"],
            True: ["ctpa", "ecg_s", "ehr"]},
        yy, gg, folds, multi)
    M["strat_traj"], ws2 = fuse_strat(
        P, {False: ["ctpa", "ecg_s", "ehr"],
            True: ["ctpa", "ecg_s", "ecg_t",
                   "ehr"]},
        yy, gg, folds, multi)
    M["strat_traj_only"], ws3 = fuse_strat(
        P, {False: ["ctpa", "ecg_s", "ehr"],
            True: ["ctpa", "ecg_t", "ehr"]},
        yy, gg, folds, multi)

    base = M["two"]
    ab = roc_auc_score(yy, base)
    print("")
    print("  %-18s %7s %7s %s"
          % ("model", "AUC", "AP",
             "gain vs ctpa+ehr"))
    for nm in ["two", "global_static",
               "strat_static", "strat_traj",
               "strat_traj_only"]:
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
            "gain_vs_two": g, "lo": lo,
            "hi": hi,
            "sig": int(lo > 0 or hi < 0)})

    print("")
    print("  KEY CONTRASTS")
    for nm, ref, lab in [
            ("strat_traj", "strat_static",
             "trajectory, stratification held"),
            ("strat_static", "global_static",
             "stratification, no trajectory"),
            ("strat_traj", "global_static",
             "both together")]:
        g, lo, hi, _ = f.boot_diff(
            yy, M[nm], M[ref], gg)
        star = ("*" if (lo > 0 or hi < 0)
                else " ")
        print("    %-34s %+.4f"
              " [%+.4f,%+.4f] %s"
              % (lab, g, lo, hi, star))
        rows.append({
            "outcome": oc,
            "model": nm + "_vs_" + ref,
            "n": int(ok.sum()),
            "ev": int(yy.sum()),
            "auc": np.nan, "gain_vs_two": g,
            "lo": lo, "hi": hi,
            "sig": int(lo > 0 or hi < 0)})

    print("")
    print("  WITHIN SUBGROUP (vs ctpa+ehr)")
    for gn, gm in (("single", ~multi),
                   ("multi", multi)):
        if yy[gm].sum() < 12:
            continue
        a0 = roc_auc_score(yy[gm], base[gm])
        print("    %-7s n=%4d ev=%3d"
              "  2mod %.4f"
              % (gn, int(gm.sum()),
                 int(yy[gm].sum()), a0))
        for nm in ["strat_static",
                   "strat_traj"]:
            av = roc_auc_score(yy[gm],
                               M[nm][gm])
            print("      %-16s %.4f  %+.4f"
                  % (nm, av, av - a0))

    if True in ws2:
        print("")
        print("  FITTED WEIGHTS")
        print("    single [ctpa,ecg_s,ehr] =",
              np.round(ws2.get(False,
                               []), 2))
        print("    multi  [ctpa,ecg_s,ecg_t,"
              "ehr] =",
              np.round(ws2.get(True, []), 2))

r = pd.DataFrame(rows)
r.to_csv(os.path.join(
    PROC, "ecg_stratified.csv"), index=False)

print("")
print("=" * 66)
print("GAIN OVER ctpa+ehr, BY MODEL")
print("=" * 66)
p = r[~r["model"].str.contains("_vs_")]
print(p.pivot_table(index="model",
                    columns="outcome",
                    values="gain_vs_two")
      .round(4).to_string())
print("")
print("significant cells:")
print(p.groupby("model")["sig"]
      .agg(["sum", "count"]).to_string())

print("")
print("KEY CONTRASTS ACROSS OUTCOMES")
q = r[r["model"].str.contains("_vs_")]
print(q.pivot_table(index="model",
                    columns="outcome",
                    values="gain_vs_two")
      .round(4).to_string())
print("")
print(q.groupby("model")["sig"]
      .agg(["sum", "count"]).to_string())

print("")
print("saved ecg_stratified.csv", r.shape)