"""Tests whether the ECG modality can be improved
without new data: max against mean aggregation,
ECG count as a feature, and recalibration
before fusion, each reported against the
gap-mechanism ceiling.
Run in venv (analysis).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
SEED, NFOLD = 42, 5
# late_best gap line from 20_gap_tuned.py, typed in
INT, SLOPE = 0.0360, -0.2425


def _rank(p):
    return pd.Series(p).rank(pct=True).values


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


def wcv(R, y, grp, cb):
    ps = [R[t] for t in cb]
    out = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    X = np.column_stack(ps)
    for tr, te in cv.split(X, y, grp):
        w = gw([p[tr] for p in ps], y[tr])
        out[te] = sum(wi * p[te]
                      for wi, p in zip(w, ps))
    return out


def iso_cv(p, y, grp):
    """Out-of-fold isotonic recalibration."""
    out = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    for tr, te in cv.split(
            p.reshape(-1, 1), y, grp):
        m = IsotonicRegression(
            out_of_bounds="clip")
        m.fit(p[tr], y[tr])
        out[te] = m.predict(p[te])
    return out


keys = pd.read_csv(os.path.join(
    FIG, "fusion_cohort_keys.csv"))
print("fusion_cohort_keys:", keys.shape)
print("  n_ecg distribution:")
print(keys["n_ecg"].value_counts()
      .head(8).to_string())
print("  corr(p_ecg, p_ecg_max): %.3f"
      % float(keys[["p_ecg", "p_ecg_max"]]
              .corr().iloc[0, 1]))

mm = f.load_mimic()
ins = f.load_inspect()
rows = []

for oc in ["death_30d", "composite_30d"]:
    d, y = f.labels(mm, oc)
    grp = d["subject_id"].values
    se, ysrc = f.labels(ins, oc)

    a, b = f.prep(
        se[f.EHR_COLS].values.astype(float),
        d[f.EHR_COLS].values.astype(float))
    p_ehr = f.fit_lr(
        a, ysrc, "ehr").predict_proba(b)[:, 1]
    c, e = f.prep(
        se[f.CTPA_COLS].values.astype(float),
        d[f.CTPA_COLS].values.astype(float))
    p_ct = f.fit_lr(
        c, ysrc, "ctpa").predict_proba(e)[:, 1]

    st = pd.read_csv(os.path.join(
        FIG, "p_ecg_harm_%s.csv" % oc))
    st = st[["hadm_id", "p_ecg"]] \
        .drop_duplicates("hadm_id")
    k = d[["hadm_id"]].merge(
        keys[["hadm_id", "p_ecg", "p_ecg_max",
              "n_ecg"]].rename(
            columns={"p_ecg": "ecg_keys"}),
        on="hadm_id", how="left")
    k = k.merge(st, on="hadm_id", how="left")

    ok = (np.isfinite(k["p_ecg"])
          & np.isfinite(k["p_ecg_max"])).values
    yy = y[ok]
    gg = grp[ok]
    print("")
    print("=" * 58)
    print("%s  n=%d ev=%d"
          % (oc, int(ok.sum()), int(yy.sum())))

    base = {"ehr": _rank(p_ehr[ok]),
            "ctpa": _rank(p_ct[ok])}
    a_ehr = roc_auc_score(yy, base["ehr"])
    a_two = roc_auc_score(
        yy, wcv(base, yy, gg,
                ("ctpa", "ehr")))
    print("  ehr %.4f | ctpa+ehr %.4f"
          % (a_ehr, a_two))

    VAR = {
        "mean (current)": k["p_ecg"].values[ok],
        "max": k["p_ecg_max"].values[ok],
        "mean+max": 0.5 * (
            _rank(k["p_ecg"].values[ok])
            + _rank(k["p_ecg_max"].values[ok])),
        "iso-recal": iso_cv(
            k["p_ecg"].values[ok], yy, gg),
    }
    nz = k["n_ecg"].values[ok]
    VAR["weighted by n"] = (
        _rank(k["p_ecg"].values[ok])
        * np.clip(nz, 1, 5) / 5.0)

    print("")
    print("  %-16s %7s %7s %8s %8s"
          % ("ecg variant", "ecgAUC", "gap",
             "3mod", "vs 2mod"))
    for nm, v in VAR.items():
        v = np.asarray(v, dtype=float)
        if not np.isfinite(v).all():
            continue
        r = _rank(v)
        ae = roc_auc_score(yy, r)
        R = dict(base)
        R["ecg"] = r
        a3 = roc_auc_score(
            yy, wcv(R, yy, gg,
                    ("ctpa", "ecg", "ehr")))
        gap = a_ehr - ae
        print("  %-16s %.4f  %.4f  %.4f  %+.4f"
              % (nm, ae, gap, a3, a3 - a_two))
        rows.append({
            "outcome": oc, "variant": nm,
            "ecg_auc": ae, "gap": gap,
            "auc_3mod": a3,
            "vs_2mod": a3 - a_two,
            "predicted": INT + SLOPE * gap})

    print("")
    print("  gap-mechanism ceiling")
    for tgt in (0.75, 0.79, 0.83):
        g = a_ehr - tgt
        print("    if ecg reached %.2f:"
              " gap %.3f -> predicted"
              " gain %+.4f"
              % (tgt, g, INT + SLOPE * g))

r = pd.DataFrame(rows)
r.to_csv(os.path.join(PROC, "ecg_improve.csv"),
         index=False)
print("")
print(r.round(4).to_string(index=False))
print("")
print("saved ecg_improve.csv", r.shape)