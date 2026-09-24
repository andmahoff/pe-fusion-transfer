"""Hybrid ECG modality on the full cohort.

Patients with one recording use static features;
those with two or more also use trajectory
features. Two constructions:

  hybrid_gb  - one model over static plus
               trajectory, with trajectory left
               missing for single-ECG patients.
               HistGradientBoosting handles
               missing values natively, so it
               learns a separate path for them.
  hybrid_two - two models fitted inside each
               training fold, one per group,
               each scoring its own patients.

Then the three-modality fusion is run on the full
cohort with each ECG version, to test whether
adding trajectory for the 28% who have serial
ECGs changes the overall model.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
SEED, NFOLD = 42, 5
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
OUTS = ["death_30d", "death_30d_inhosp",
        "composite_30d", "cv_first"]
# late_best gap line from 20_gap_tuned.py, typed in
INT2, SL2 = 0.0360, -0.2425


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def mk_gb():
    return HistGradientBoostingClassifier(
        random_state=SEED, max_depth=3,
        learning_rate=0.05, max_iter=300,
        l2_regularization=1.0)


def fit_lr_fold(Xa, ya, ga, Xb):
    """Impute, scale, pick C by inner grouped CV,
    then score Xb."""
    im = SimpleImputer(strategy="median")
    a = im.fit_transform(Xa)
    b = im.transform(Xb)
    sc = StandardScaler()
    a, b = sc.fit_transform(a), sc.transform(b)
    best, bc = -1.0, 0.1
    k = min(3, int(ya.sum()) // 5)
    if k >= 2:
        icv = StratifiedGroupKFold(
            n_splits=k, shuffle=True,
            random_state=SEED)
        for c in CS:
            q = np.zeros(len(ya))
            try:
                for t2, v2 in icv.split(
                        a, ya, ga):
                    m = LogisticRegression(
                        C=c, max_iter=5000)
                    m.fit(a[t2], ya[t2])
                    q[v2] = m.predict_proba(
                        a[v2])[:, 1]
                s = roc_auc_score(ya, q)
            except Exception:
                continue
            if s > best:
                best, bc = s, c
    m = LogisticRegression(C=bc, max_iter=5000)
    m.fit(a, ya)
    return m.predict_proba(b)[:, 1]


def oof_lr(X, y, grp, folds):
    p = np.zeros(len(y))
    for tr, te in folds:
        p[te] = fit_lr_fold(
            X[tr], y[tr], grp[tr], X[te])
    return p


def oof_gb(X, y, grp, folds):
    """HistGB, NaN passed through untouched."""
    p = np.zeros(len(y))
    for tr, te in folds:
        m = mk_gb()
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def oof_two(Xs, Xt, y, grp, multi, folds):
    """Two models per fold: singles get static
    only, multis get static plus trajectory.
    Raw probabilities are concatenated, so both
    stay on a probability scale."""
    p = np.zeros(len(y))
    Xb = np.column_stack([Xs, Xt])
    for tr, te in folds:
        for flag in (False, True):
            trm = tr[multi[tr] == flag]
            tem = te[multi[te] == flag]
            if len(tem) == 0:
                continue
            if len(trm) < 40 or \
                    y[trm].sum() < 5:
                # too few to fit, fall back to
                # the static model on all
                p[tem] = fit_lr_fold(
                    Xs[tr], y[tr], grp[tr],
                    Xs[tem])
                continue
            A = Xb if flag else Xs
            p[tem] = fit_lr_fold(
                A[trm], y[trm], grp[trm],
                A[tem])
    return p


def gw(ps, y, step=0.05):
    k = len(ps)
    best, bw = -1.0, tuple([1.0 / k] * k)
    if k == 2:
        for w in np.arange(0, 1.001, step):
            a = roc_auc_score(
                y, w * ps[0] + (1 - w) * ps[1])
            if a > best:
                best, bw = a, (w, 1 - w)
    else:
        for w1 in np.arange(0, 1.001, step):
            for w2 in np.arange(
                    0, 1.001 - w1, step):
                a = roc_auc_score(
                    y, w1 * ps[0] + w2 * ps[1]
                    + (1 - w1 - w2) * ps[2])
                if a > best:
                    best, bw = a, (
                        w1, w2, 1 - w1 - w2)
    return bw


def wcv(ps, y, grp, folds):
    out = np.zeros(len(y))
    ws = []
    for tr, te in folds:
        w = gw([p[tr] for p in ps], y[tr])
        ws.append(w)
        out[te] = sum(wi * p[te]
                      for wi, p in zip(w, ps))
    return out, np.mean(ws, axis=0)


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
print("static cols:", len(scols))
print("trajectory cols:", len(tcols))

rows, fus, sub = [], [], []

for oc in OUTS:
    if oc not in mim.columns:
        continue
    d = mim
    if oc == "cv_first":
        d = mim[mim["death_first"] == 0]
    y = pd.to_numeric(d[oc], errors="coerce")
    y = y.fillna(0).astype(int).values
    grp = d["subject_id"].values

    fp = os.path.join(
        FIG, "p_ecg_harm_%s.csv" % oc)
    if not os.path.exists(fp):
        continue
    stq = pd.read_csv(fp)[
        ["hadm_id", "p_ecg"]].drop_duplicates(
        "hadm_id")

    a = d[["hadm_id"]].merge(
        stat[["hadm_id"] + scols],
        on="hadm_id", how="left")
    a = a.merge(traj[["hadm_id"] + tcols],
                on="hadm_id", how="left")
    a = a.merge(stq, on="hadm_id", how="left")

    # full cohort: needs static ECG features
    ok = (a[scols].notna().sum(axis=1)
          >= 5).values
    ok &= np.isfinite(a["p_ecg"]).values
    yy, gg = y[ok], grp[ok]
    if yy.sum() < 30:
        continue

    Xs = a.loc[ok, scols].values.astype(float)
    Xt = a.loc[ok, tcols].values.astype(float)
    ps = a.loc[ok, "p_ecg"].values
    multi = (a.loc[ok, tcols].notna()
             .sum(axis=1) >= 10).values

    print("")
    print("=" * 66)
    print("%s   n=%d  ev=%d" % (oc,
                                int(ok.sum()),
                                int(yy.sum())))
    print("  single-ECG: %d (ev %d, rate %.3f)"
          % (int((~multi).sum()),
             int(yy[~multi].sum()),
             float(yy[~multi].mean())))
    print("  multi-ECG : %d (ev %d, rate %.3f)"
          % (int(multi.sum()),
             int(yy[multi].sum()),
             float(yy[multi].mean())))

    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    folds = list(cv.split(Xs, yy, gg))

    E = {}
    E["stored"] = _rank(ps)
    E["static"] = _rank(
        oof_lr(Xs, yy, gg, folds))
    Xall = np.column_stack([Xs, Xt])
    E["hybrid_gb"] = _rank(
        oof_gb(Xall, yy, gg, folds))
    E["static_gb"] = _rank(
        oof_gb(Xs, yy, gg, folds))
    E["hybrid_two"] = _rank(
        oof_two(Xs, Xt, yy, gg, multi, folds))

    print("")
    print("  ECG MODALITY ON THE FULL COHORT")
    print("  %-12s %7s %7s %9s %9s"
          % ("version", "AUC", "AP",
             "single", "multi"))
    for nm in ["stored", "static", "static_gb",
               "hybrid_gb", "hybrid_two"]:
        v = E[nm]
        au = roc_auc_score(yy, v)
        a1 = (roc_auc_score(yy[~multi],
                            v[~multi])
              if yy[~multi].sum() > 5
              else np.nan)
        a2s = (roc_auc_score(yy[multi],
                             v[multi])
               if yy[multi].sum() > 5
               else np.nan)
        print("  %-12s %.4f  %.4f  %8.4f"
              "  %8.4f"
              % (nm, au,
                 average_precision_score(yy, v),
                 a1, a2s))
        rows.append({
            "outcome": oc, "version": nm,
            "n": int(ok.sum()),
            "ev": int(yy.sum()), "auc": au,
            "auc_single": a1, "auc_multi": a2s})
        sub.append({"outcome": oc,
                    "version": nm,
                    "single": a1, "multi": a2s})

    # three-modality fusion on the full cohort
    src = ("death_30d"
           if oc == "death_30d_inhosp" else oc)
    se, ysrc = f.labels(ins, src)
    p1, q1 = f.prep(
        se[f.EHR_COLS].values.astype(float),
        d.loc[ok, f.EHR_COLS].values
        .astype(float))
    p_ehr = _rank(f.fit_lr(
        p1, ysrc, "ehr").predict_proba(
        q1)[:, 1])
    p2, q2 = f.prep(
        se[f.CTPA_COLS].values.astype(float),
        d.loc[ok, f.CTPA_COLS].values
        .astype(float))
    p_ct = _rank(f.fit_lr(
        p2, ysrc, "ctpa").predict_proba(
        q2)[:, 1])
    a_ehr = roc_auc_score(yy, p_ehr)
    two, _ = wcv([p_ct, p_ehr], yy, gg, folds)
    a2f = roc_auc_score(yy, two)

    print("")
    print("  THREE-MODALITY, FULL COHORT")
    print("  EHR %.4f | ctpa+ehr %.4f"
          % (a_ehr, a2f))
    print("  %-12s %7s %8s %8s %8s %s"
          % ("ecg", "ecgAUC", "gap", "pred",
             "3mod", "gain vs ctpa+ehr"))
    keep3 = {}
    for nm in ["stored", "static",
               "hybrid_gb", "hybrid_two"]:
        v = E[nm]
        ae = roc_auc_score(yy, v)
        gp = a_ehr - ae
        p3, w3 = wcv([p_ct, v, p_ehr],
                     yy, gg, folds)
        keep3[nm] = p3
        a3 = roc_auc_score(yy, p3)
        gn, lo, hi, _ = f.boot_diff(
            yy, p3, two, gg)
        star = ("*" if (lo > 0 or hi < 0)
                else " ")
        print("  %-12s %.4f  %+.4f  %+.4f"
              "  %.4f  %+.4f"
              " [%+.4f,%+.4f] %s"
              % (nm, ae, gp,
                 INT2 + SL2 * gp, a3, gn,
                 lo, hi, star))
        fus.append({
            "outcome": oc, "ecg": nm,
            "n": int(ok.sum()),
            "ev": int(yy.sum()),
            "ecg_auc": ae, "gap": gp,
            "predicted": INT2 + SL2 * gp,
            "auc_2mod": a2f, "auc_3mod": a3,
            "gain": gn, "lo": lo, "hi": hi,
            "w_ecg": float(w3[1]),
            "sig": int(lo > 0 or hi < 0)})

    print("")
    print("  DOES THE HYBRID BEAT STATIC"
          " IN THE 3-MODALITY MODEL?")
    for nm in ["hybrid_gb", "hybrid_two"]:
        gn, lo, hi, _ = f.boot_diff(
            yy, keep3[nm], keep3["static"], gg)
        star = ("*" if (lo > 0 or hi < 0)
                else " ")
        print("    %-12s %+.4f"
              " [%+.4f,%+.4f] %s"
              % (nm, gn, lo, hi, star))
        fus.append({
            "outcome": oc,
            "ecg": nm + "_vs_static",
            "n": int(ok.sum()),
            "ev": int(yy.sum()),
            "ecg_auc": np.nan, "gap": np.nan,
            "predicted": np.nan,
            "auc_2mod": np.nan,
            "auc_3mod": np.nan, "gain": gn,
            "lo": lo, "hi": hi,
            "w_ecg": np.nan,
            "sig": int(lo > 0 or hi < 0)})

    print("")
    print("  GAIN WITHIN EACH SUBGROUP"
          " (hybrid_gb 3mod vs ctpa+ehr)")
    for nm, m in (("single", ~multi),
                  ("multi", multi)):
        if yy[m].sum() < 15:
            continue
        aa = roc_auc_score(yy[m], two[m])
        bb = roc_auc_score(
            yy[m], keep3["hybrid_gb"][m])
        print("    %-7s n=%4d ev=%3d"
              "  2mod %.4f -> 3mod %.4f"
              "  %+.4f"
              % (nm, int(m.sum()),
                 int(yy[m].sum()), aa, bb,
                 bb - aa))

r1 = pd.DataFrame(rows)
r2 = pd.DataFrame(fus)
r1.to_csv(os.path.join(
    PROC, "ecg_hybrid.csv"), index=False)
r2.to_csv(os.path.join(
    PROC, "ecg_hybrid_fusion.csv"),
    index=False)

print("")
print("=" * 66)
print("SUMMARY: FULL-COHORT THREE-MODALITY"
      " GAIN BY ECG VERSION")
print("=" * 66)
p = r2[~r2["ecg"].str.contains("_vs_")]
print(p.pivot_table(index="ecg",
                    columns="outcome",
                    values="gain")
      .round(4).to_string())

print("")
print("SIGNIFICANT CELLS PER VERSION")
print(p.groupby("ecg")["sig"].agg(
    ["sum", "count"]).to_string())

print("")
print("HYBRID VS STATIC, HEAD TO HEAD")
q = r2[r2["ecg"].str.contains("_vs_static")]
print(q[["outcome", "ecg", "gain", "lo",
         "hi", "sig"]].round(4)
      .to_string(index=False))

print("")
print("saved ecg_hybrid.csv,"
      " ecg_hybrid_fusion.csv")