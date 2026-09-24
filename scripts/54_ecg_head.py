"""Improve the ECG modality itself.

Tests, on the full cohort and split by whether a
patient has serial recordings:
  - regularisation strength, never tuned for
    this modality
  - class weighting, which the dissertation head
    used and which distorts the probability
    scale
  - gradient boosting over the features, against
    the current linear model
  - static plus trajectory combined, trajectory
    left missing where absent so HistGB routes
    it rather than imputing

Target: the gap line predicts a full-cohort
three-modality gain of +0.017 to +0.022 if this
modality reaches 0.78 to 0.80 against an EHR
modality at 0.857.
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
CS = [0.001, 0.003, 0.01, 0.03, 0.1,
      0.3, 1.0, 3.0]
OUTS = ["death_30d", "death_30d_inhosp",
        "composite_30d"]
DEST = os.path.join(PROC, "ecg_head.csv")


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def lr_fold(Xa, ya, ga, Xb, cw=None,
            fixed_c=None):
    im = SimpleImputer(strategy="median")
    a = im.fit_transform(Xa)
    b = im.transform(Xb)
    sc = StandardScaler()
    a, b = sc.fit_transform(a), sc.transform(b)
    if fixed_c is not None:
        bc = fixed_c
    else:
        best, bc = -1.0, 0.1
        k = min(3, max(2, int(ya.sum()) // 8))
        try:
            icv = StratifiedGroupKFold(
                n_splits=k, shuffle=True,
                random_state=SEED)
            for c in CS:
                q = np.zeros(len(ya))
                for t2, v2 in icv.split(
                        a, ya, ga):
                    m = LogisticRegression(
                        C=c, max_iter=5000,
                        class_weight=cw)
                    m.fit(a[t2], ya[t2])
                    q[v2] = m.predict_proba(
                        a[v2])[:, 1]
                s = roc_auc_score(ya, q)
                if s > best:
                    best, bc = s, c
        except Exception:
            bc = 0.1
    m = LogisticRegression(
        C=bc, max_iter=5000, class_weight=cw)
    m.fit(a, ya)
    return m.predict_proba(b)[:, 1], bc


def gb_fold(Xa, ya, Xb, depth=3, lr=0.05,
            leaves=15, l2=1.0):
    m = HistGradientBoostingClassifier(
        random_state=SEED, max_depth=depth,
        learning_rate=lr, max_iter=300,
        max_leaf_nodes=leaves,
        l2_regularization=l2,
        early_stopping=True,
        validation_fraction=0.15)
    m.fit(Xa, ya)
    return m.predict_proba(Xb)[:, 1]


def oof(X, y, grp, folds, kind="lr",
        cw=None, fixed_c=None, **kw):
    p = np.zeros(len(y))
    cs = []
    for tr, te in folds:
        if kind == "gb":
            p[te] = gb_fold(X[tr], y[tr],
                            X[te], **kw)
        else:
            p[te], c = lr_fold(
                X[tr], y[tr], grp[tr], X[te],
                cw=cw, fixed_c=fixed_c)
            cs.append(c)
    return p, cs


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
print("static %d  trajectory %d"
      % (len(scols), len(tcols)))

rows = []
for oc in OUTS:
    if oc not in mim.columns:
        continue
    d = mim
    y = pd.to_numeric(d[oc], errors="coerce")
    y = y.fillna(0).astype(int).values
    grp = d["subject_id"].values

    a = d[["hadm_id"]].merge(
        stat[["hadm_id"] + scols],
        on="hadm_id", how="left")
    a = a.merge(traj[["hadm_id"] + tcols],
                on="hadm_id", how="left")
    fp = os.path.join(
        FIG, "p_ecg_harm_%s.csv" % oc)
    if not os.path.exists(fp):
        continue
    stq = pd.read_csv(fp)[
        ["hadm_id", "p_ecg"]].drop_duplicates(
        "hadm_id")
    a = a.merge(stq, on="hadm_id", how="left")

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
    Xst = np.column_stack([Xs, Xt])
    Xall = np.column_stack(
        [Xs, Xt, ps.reshape(-1, 1)])

    print("")
    print("=" * 68)
    print("%s  n=%d ev=%d  (multi %d)"
          % (oc, int(ok.sum()), int(yy.sum()),
             int(multi.sum())))

    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    folds = list(cv.split(Xs, yy, gg))

    R = {"stored": (_rank(ps), None)}

    for nm, X, cw, fc in [
            ("static_C1", Xs, None, 1.0),
            ("static_Ctuned", Xs, None, None),
            ("static_bal", Xs, "balanced",
             None),
            ("st_Ctuned", Xst, None, None),
            ("st_bal", Xst, "balanced", None),
            ("all_Ctuned", Xall, None, None)]:
        p, cs = oof(X, yy, gg, folds,
                    kind="lr", cw=cw,
                    fixed_c=fc)
        R[nm] = (_rank(p), cs)

    for nm, X, kw in [
            ("static_gb", Xs, {}),
            ("st_gb", Xst, {}),
            ("st_gb_deep", Xst,
             {"depth": 5, "leaves": 31}),
            ("st_gb_slow", Xst,
             {"lr": 0.02, "l2": 3.0}),
            ("all_gb", Xall, {})]:
        p, _ = oof(X, yy, gg, folds,
                   kind="gb", **kw)
        R[nm] = (_rank(p), None)

    lin = max(["static_Ctuned", "st_Ctuned",
               "all_Ctuned"],
              key=lambda k: roc_auc_score(
                  yy, R[k][0]))
    gbb = max(["st_gb", "st_gb_deep",
               "st_gb_slow", "all_gb"],
              key=lambda k: roc_auc_score(
                  yy, R[k][0]))
    R["blend"] = (0.5 * (R[lin][0]
                         + R[gbb][0]), None)
    R["blend+stored"] = (
        (R[lin][0] + R[gbb][0]
         + R["stored"][0]) / 3.0, None)

    base = roc_auc_score(yy, R["stored"][0])
    print("  %-16s %7s %7s %8s %8s %s"
          % ("model", "AUC", "AP", "single",
             "multi", "C"))
    for nm in ["stored", "static_C1",
               "static_Ctuned", "static_bal",
               "st_Ctuned", "st_bal",
               "all_Ctuned", "static_gb",
               "st_gb", "st_gb_deep",
               "st_gb_slow", "all_gb",
               "blend", "blend+stored"]:
        v, cs = R[nm]
        au = roc_auc_score(yy, v)
        a1 = (roc_auc_score(yy[~multi],
                            v[~multi])
              if yy[~multi].sum() > 5
              else np.nan)
        a2 = (roc_auc_score(yy[multi],
                            v[multi])
              if yy[multi].sum() > 5
              else np.nan)
        ctxt = (str(sorted(set(cs)))
                if cs else "")
        mk = "*" if au > base else " "
        print("  %-16s %.4f  %.4f  %.4f"
              "  %.4f %s %s"
              % (nm, au,
                 average_precision_score(yy, v),
                 a1, a2, mk, ctxt))
        rows.append({
            "outcome": oc, "model": nm,
            "n": int(ok.sum()),
            "ev": int(yy.sum()), "auc": au,
            "auc_single": a1, "auc_multi": a2,
            "vs_stored": au - base})

    bn = max([k for k in R if k != "stored"],
             key=lambda k: roc_auc_score(
                 yy, R[k][0]))
    g, lo, hi, _ = f.boot_diff(
        yy, R[bn][0], R["stored"][0], gg)
    print("")
    print("  best (%s) vs stored: %+.4f"
          " [%+.4f,%+.4f] %s"
          % (bn, g, lo, hi,
             "*" if (lo > 0 or hi < 0) else ""))
    rows.append({
        "outcome": oc,
        "model": "BEST_vs_stored:" + bn,
        "n": int(ok.sum()),
        "ev": int(yy.sum()),
        "auc": roc_auc_score(yy, R[bn][0]),
        "auc_single": np.nan,
        "auc_multi": np.nan,
        "vs_stored": g, "lo": lo, "hi": hi})

    # what the late_best gap line from
    # 20_gap_tuned.py predicts at this level
    src = ("death_30d"
           if oc == "death_30d_inhosp" else oc)
    se, ysrc = f.labels(ins, src)
    p1, q1 = f.prep(
        se[f.EHR_COLS].values.astype(float),
        d.loc[ok, f.EHR_COLS].values
        .astype(float))
    a_ehr = roc_auc_score(
        yy, _rank(f.fit_lr(
            p1, ysrc, "ehr").predict_proba(
            q1)[:, 1]))
    ab = roc_auc_score(yy, R[bn][0])
    gp = a_ehr - ab
    print("  EHR %.4f | best ECG %.4f"
          " | gap %.4f -> predicted 3mod"
          " gain %+.4f"
          % (a_ehr, ab, gp,
             0.0360 - 0.2425 * gp))
    rows.append({
        "outcome": oc,
        "model": "GAP_CHECK",
        "n": int(ok.sum()),
        "ev": int(yy.sum()), "auc": ab,
        "auc_single": a_ehr,
        "auc_multi": gp,
        "vs_stored": 0.0360 - 0.2425 * gp})

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("=" * 68)
print("GAIN OVER THE STORED MODALITY")
q = r[~r["model"].str.startswith(
    ("BEST_", "GAP_"))]
print(q.pivot_table(index="model",
                    columns="outcome",
                    values="vs_stored")
      .round(4).to_string())

print("")
print("BEST PER OUTCOME")
for oc, s in q.groupby("outcome"):
    b = s.loc[s["auc"].idxmax()]
    print("  %-18s %-16s %.4f  (%+.4f)"
          % (oc, b["model"], b["auc"],
             b["vs_stored"]))

print("")
print("GAP CHECK")
g = r[r["model"] == "GAP_CHECK"]
print(g[["outcome", "auc", "auc_single",
         "auc_multi", "vs_stored"]]
      .rename(columns={
          "auc": "best_ecg",
          "auc_single": "ehr",
          "auc_multi": "gap",
          "vs_stored": "pred_3mod_gain"})
      .round(4).to_string(index=False))

print("")
print("saved", DEST, r.shape)