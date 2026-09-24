"""Tests whether serial ECGs carry information
that a single recording does not.

The earlier ECG scripts collapse multiple
recordings to one number. This builds trajectory
features instead: change between the first and
last recording, rate of change per hour, and
within-admission variability.

Restricted to the 1,551 admissions with two or
more recordings, so the comparison is fair.
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
RAW = os.path.join("data", "raw", "ecg")
SEED, NFOLD = 42, 5
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
OUTS = ["death_30d", "death_30d_inhosp",
        "composite_30d", "cv_first"]
# late_best gap line from 20_gap_tuned.py, typed in
INT2, SL2 = 0.0360, -0.2425

# measurements worth tracking over time
TRACK = ["hr_mean", "rr_sd", "rmssd", "pnn50",
         "rr_cv", "qrs_ms", "qt_ms",
         "qtc_hodges", "qrs_axis", "r_amp"]


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def oof(X, y, grp):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    for tr, te in cv.split(X, y, grp):
        im = SimpleImputer(strategy="median")
        a = im.fit_transform(X[tr])
        b = im.transform(X[te])
        sc = StandardScaler()
        a, b = sc.fit_transform(a), sc.transform(b)
        best, bc = -1.0, 0.1
        icv = StratifiedGroupKFold(
            n_splits=3, shuffle=True,
            random_state=SEED)
        for c in CS:
            q = np.zeros(len(tr))
            for t2, v2 in icv.split(
                    a, y[tr], grp[tr]):
                m = LogisticRegression(
                    C=c, max_iter=5000)
                m.fit(a[t2], y[tr][t2])
                q[v2] = m.predict_proba(
                    a[v2])[:, 1]
            s = roc_auc_score(y[tr], q)
            if s > best:
                best, bc = s, c
        m = LogisticRegression(C=bc,
                               max_iter=5000)
        m.fit(a, y[tr])
        p[te] = m.predict_proba(b)[:, 1]
    return p


rec = pd.read_csv(os.path.join(
    PROC, "ecg_derived_v3_record.csv"))
idx = pd.read_csv(os.path.join(
    RAW, "ecg_record_index.csv"))
print("record-level rows:", len(rec))

j = idx[["subject_id", "hadm_id", "rec",
         "ecg_charttime"]].merge(
    rec, on="rec", how="inner")
j["t"] = pd.to_datetime(
    j["ecg_charttime"], errors="coerce")
j = j.dropna(subset=["t"])
j = j.sort_values(["hadm_id", "t"])

n = j.groupby("hadm_id").size()
multi = set(n[n >= 2].index)
print("admissions total:", len(n))
print("with >=2 recordings:", len(multi))
print("recordings per admission:")
print(n.value_counts().head(6).to_string())

TRACK = [c for c in TRACK if c in j.columns]
print("")
print("tracking:", TRACK)

rows = []
for h, g in j[j["hadm_id"].isin(multi)] \
        .groupby("hadm_id"):
    g = g.sort_values("t")
    d = {"hadm_id": h,
         "subject_id": g["subject_id"].iloc[0],
         "n_rec": len(g)}
    span = (g["t"].iloc[-1]
            - g["t"].iloc[0]).total_seconds()
    d["span_h"] = span / 3600.0
    for c in TRACK:
        v = pd.to_numeric(g[c],
                          errors="coerce")
        ok = v.notna()
        if ok.sum() < 2:
            continue
        vv = v[ok].values
        tt = g.loc[ok, "t"]
        d[c + "_first"] = float(vv[0])
        d[c + "_last"] = float(vv[-1])
        d[c + "_delta"] = float(vv[-1] - vv[0])
        d[c + "_absdelta"] = float(
            abs(vv[-1] - vv[0]))
        d[c + "_within_sd"] = float(np.std(vv))
        d[c + "_range"] = float(
            vv.max() - vv.min())
        hrs = (tt.iloc[-1]
               - tt.iloc[0]).total_seconds()
        hrs = hrs / 3600.0
        if hrs > 0.5:
            d[c + "_slope"] = float(
                (vv[-1] - vv[0]) / hrs)
        if abs(vv[0]) > 1e-6:
            d[c + "_relchange"] = float(
                (vv[-1] - vv[0]) / abs(vv[0]))
    rows.append(d)

traj = pd.DataFrame(rows)
traj.to_csv(os.path.join(
    PROC, "ecg_trajectory.csv"), index=False)
print("")
print("trajectory features:", traj.shape)

cov = traj.notna().mean()
tcols = [c for c in traj.columns
         if c not in ("hadm_id", "subject_id")
         and cov[c] >= 0.60]
print("with >=60%% coverage:", len(tcols))

# static comparator: the same measurements
# aggregated, not tracked
stat = pd.read_csv(os.path.join(
    PROC, "ecg_derived_v3.csv"))
scov = stat.notna().mean()
scols = [c for c in stat.columns
         if c not in ("subject_id", "hadm_id")
         and pd.api.types.is_numeric_dtype(
             stat[c])
         and scov[c] >= 0.60]

mim = f.load_mimic()
ins = f.load_inspect()
out = []

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
        traj[["hadm_id"] + tcols],
        on="hadm_id", how="left")
    a = a.merge(stat[["hadm_id"] + scols],
                on="hadm_id", how="left")
    a = a.merge(stq, on="hadm_id", how="left")

    ok = (a[tcols].notna().sum(axis=1)
          >= 10).values
    ok &= np.isfinite(a["p_ecg"]).values
    yy, gg = y[ok], grp[ok]
    if yy.sum() < 25:
        print("")
        print("%s: only %d events on the"
              " multi-ECG subset, skipping"
              % (oc, int(yy.sum())))
        continue

    print("")
    print("=" * 62)
    print("%s  n=%d ev=%d  (>=2 recordings)"
          % (oc, int(ok.sum()), int(yy.sum())))

    ps = a.loc[ok, "p_ecg"].values
    Xt = a.loc[ok, tcols].values.astype(float)
    Xs = a.loc[ok, scols].values.astype(float)
    Xb = np.column_stack([Xs, Xt])

    R = {"stored": _rank(ps),
         "static": _rank(oof(Xs, yy, gg)),
         "traj": _rank(oof(Xt, yy, gg)),
         "static+traj": _rank(
             oof(Xb, yy, gg))}
    R["stored+traj"] = 0.5 * (
        R["stored"] + R["traj"])
    R["all"] = (R["stored"] + R["static"]
                + R["traj"]) / 3.0

    se, ysrc = f.labels(
        ins, "death_30d"
        if oc == "death_30d_inhosp" else oc)
    p, q = f.prep(
        se[f.EHR_COLS].values.astype(float),
        d.loc[ok, f.EHR_COLS].values
        .astype(float))
    a_ehr = roc_auc_score(
        yy, _rank(f.fit_lr(p, ysrc, "ehr")
                  .predict_proba(q)[:, 1]))
    print("  EHR reference: %.4f" % a_ehr)
    print("  %-14s %7s %7s %8s %10s"
          % ("model", "AUC", "AP", "gap",
             "pred 3mod"))
    base = roc_auc_score(yy, R["stored"])
    for nm in ["stored", "static", "traj",
               "static+traj", "stored+traj",
               "all"]:
        au = roc_auc_score(yy, R[nm])
        ap = average_precision_score(yy, R[nm])
        gp = a_ehr - au
        mk = " *" if au > base else "  "
        print("  %-14s %.4f  %.4f  %+.4f"
              "   %+.4f%s"
              % (nm, au, ap, gp,
                 INT2 + SL2 * gp, mk))
        out.append({
            "outcome": oc, "model": nm,
            "n": int(ok.sum()),
            "ev": int(yy.sum()), "auc": au,
            "ap": ap, "ehr": a_ehr, "gap": gp,
            "vs_stored": au - base})

r = pd.DataFrame(out)
r.to_csv(os.path.join(
    PROC, "ecg_trajectory_fit.csv"),
    index=False)

print("")
print("=" * 62)
print("DOES TRAJECTORY ADD ANYTHING?")
print("=" * 62)
for oc, sub in r.groupby("outcome"):
    b = sub[sub["model"] != "stored"]
    b = b.loc[b["auc"].idxmax()]
    s0 = float(sub[sub["model"] == "stored"]
               ["auc"].iloc[0])
    print("  %-18s stored %.4f -> %-12s"
          " %.4f  (%+.4f)"
          % (oc, s0, b["model"], b["auc"],
             b["vs_stored"]))

print("")
print("MEAN CHANGE OVER STORED, BY MODEL")
print(r[r["model"] != "stored"]
      .groupby("model")["vs_stored"]
      .agg(["mean", "max", "count"])
      .round(4).to_string())

print("")
print("BEST GAP AGAINST THE 0.145 THRESHOLD")
for oc, sub in r.groupby("outcome"):
    b = sub.loc[sub["auc"].idxmax()]
    print("  %-18s %.4f  gap %.4f  -> %s"
          % (oc, b["auc"], b["gap"],
             "clears" if b["gap"] < 0.145
             else "does not clear"))

print("")
print("saved ecg_trajectory_fit.csv", r.shape)