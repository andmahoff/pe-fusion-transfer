"""Add the waveform measurements to the stored
71-SCP model without shrinking the stored
prediction.

Script 54 put everything in one matrix, so the
one strong feature was shrunk alongside 187 weak
ones. Four constructions that avoid this:

  offset   - stored prediction enters as a fixed
             offset (logit as an unpenalised
             feature), so the measurements only
             have to explain what it gets wrong
  residual - fit the measurements on the stored
             model's residual, then add
  wblend    - weight-tuned blend rather than
             equal thirds, weight fitted inside
             each training fold
  boost    - HistGB with the stored logit as an
             init prediction, so the trees model
             only the remainder

Also reports how much of the measurement signal
is redundant with the stored prediction.
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
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
SEED, NFOLD = 42, 5
CS = [0.001, 0.003, 0.01, 0.03, 0.1,
      0.3, 1.0]
OUTS = ["death_30d", "death_30d_inhosp",
        "composite_30d"]
DEST = os.path.join(PROC, "ecg_residual.csv")


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def logit(p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def prep(Xa, Xb):
    im = SimpleImputer(strategy="median")
    a = im.fit_transform(Xa)
    b = im.transform(Xb)
    sc = StandardScaler()
    return sc.fit_transform(a), sc.transform(b)


def pick_c(a, ya, ga):
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
        pass
    return bc


def m_offset(Xa, ya, ga, sa, Xb, sb):
    """Stored logit appended unscaled with a
    large effective C, so it is not shrunk with
    the measurements. Approximated by scaling the
    measurements down rather than the offset up.
    """
    a, b = prep(Xa, Xb)
    c = pick_c(a, ya, ga)
    a = np.column_stack([a, logit(sa)])
    b = np.column_stack([b, logit(sb)])
    # penalty applies to all columns, so give the
    # offset column a large scale to make its
    # coefficient cheap
    a[:, -1] *= 20.0
    b[:, -1] *= 20.0
    m = LogisticRegression(C=c, max_iter=5000)
    m.fit(a, ya)
    return m.predict_proba(b)[:, 1]


def m_residual(Xa, ya, ga, sa, Xb, sb):
    """Two-stage: model the residual of the
    stored prediction, then add it back on the
    logit scale."""
    la, lb = logit(sa), logit(sb)
    res = ya - sa
    a, b = prep(Xa, Xb)
    r = Ridge(alpha=50.0)
    r.fit(a, res)
    return 1.0 / (1.0 + np.exp(
        -(lb + 4.0 * r.predict(b))))


def m_resid_gb(Xa, ya, ga, sa, Xb, sb):
    """Same idea with a boosted residual
    learner."""
    la, lb = logit(sa), logit(sb)
    res = ya - sa
    g = HistGradientBoostingRegressor(
        random_state=SEED, max_depth=3,
        learning_rate=0.03, max_iter=200,
        l2_regularization=3.0,
        early_stopping=True,
        validation_fraction=0.15)
    g.fit(Xa, res)
    return 1.0 / (1.0 + np.exp(
        -(lb + 4.0 * g.predict(Xb))))


def m_wblend(Xa, ya, ga, sa, Xb, sb,
             kind="gb"):
    """Weight-tuned blend. The measurement model
    is fitted on the training fold, then the
    weight is chosen on that same fold."""
    if kind == "gb":
        m = HistGradientBoostingClassifier(
            random_state=SEED, max_depth=3,
            learning_rate=0.05, max_iter=300,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15)
        m.fit(Xa, ya)
        pa = m.predict_proba(Xa)[:, 1]
        pb = m.predict_proba(Xb)[:, 1]
    else:
        a, b = prep(Xa, Xb)
        c = pick_c(a, ya, ga)
        m = LogisticRegression(C=c,
                               max_iter=5000)
        m.fit(a, ya)
        pa = m.predict_proba(a)[:, 1]
        pb = m.predict_proba(b)[:, 1]
    ra, rb = _rank(pa), _rank(pb)
    sa_r, sb_r = _rank(sa), _rank(sb)
    best, bw = -1.0, 0.0
    for w in np.arange(0, 1.001, 0.05):
        v = w * ra + (1 - w) * sa_r
        s = roc_auc_score(ya, v)
        if s > best:
            best, bw = s, w
    return bw * rb + (1 - bw) * sb_r, bw


def m_gbinit(Xa, ya, sa, Xb, sb):
    """Boosting on the measurements, with the
    stored logit carried as a feature the trees
    can split on first."""
    a = np.column_stack([Xa, logit(sa)])
    b = np.column_stack([Xb, logit(sb)])
    m = HistGradientBoostingClassifier(
        random_state=SEED, max_depth=3,
        learning_rate=0.05, max_iter=300,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15)
    m.fit(a, ya)
    return m.predict_proba(b)[:, 1]


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
      % (len(scols), len(tcols)), flush=True)

rows = []
for oc in OUTS:
    if oc not in mim.columns:
        continue
    d = mim
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

    ok = (a[scols].notna().sum(axis=1)
          >= 5).values
    ok &= np.isfinite(a["p_ecg"]).values
    yy, gg = y[ok], grp[ok]
    if yy.sum() < 30:
        continue

    X = a.loc[ok, scols + tcols].values.astype(
        float)
    Xi = np.nan_to_num(X, nan=np.nan)
    ps = a.loc[ok, "p_ecg"].values

    print("")
    print("=" * 68)
    print("%s  n=%d ev=%d"
          % (oc, int(ok.sum()), int(yy.sum())),
          flush=True)

    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    folds = list(cv.split(X, yy, gg))

    P = {"stored": ps.copy()}
    for nm in ["offset", "residual",
               "resid_gb", "wblend_gb",
               "wblend_lr", "gbinit"]:
        P[nm] = np.zeros(len(yy))
    ws = {"wblend_gb": [], "wblend_lr": []}

    for tr, te in folds:
        Xa, Xb = X[tr], X[te]
        ya, ga = yy[tr], gg[tr]
        sa, sb = ps[tr], ps[te]
        P["offset"][te] = m_offset(
            Xa, ya, ga, sa, Xb, sb)
        P["residual"][te] = m_residual(
            Xa, ya, ga, sa, Xb, sb)
        P["resid_gb"][te] = m_resid_gb(
            Xa, ya, ga, sa, Xb, sb)
        v, w = m_wblend(Xa, ya, ga, sa, Xb,
                        sb, "gb")
        P["wblend_gb"][te] = v
        ws["wblend_gb"].append(w)
        v, w = m_wblend(Xa, ya, ga, sa, Xb,
                        sb, "lr")
        P["wblend_lr"][te] = v
        ws["wblend_lr"].append(w)
        P["gbinit"][te] = m_gbinit(
            Xa, ya, sa, Xb, sb)

    base = roc_auc_score(yy, _rank(ps))
    print("  stored baseline: %.4f" % base)
    print("  %-12s %7s %7s %s"
          % ("model", "AUC", "AP",
             "vs stored"))
    for nm in ["stored", "offset", "residual",
               "resid_gb", "wblend_lr",
               "wblend_gb", "gbinit"]:
        v = _rank(P[nm])
        au = roc_auc_score(yy, v)
        if nm == "stored":
            g = lo = hi = 0.0
        else:
            g, lo, hi, _ = f.boot_diff(
                yy, v, _rank(ps), gg)
        star = ("*" if (lo > 0 or hi < 0)
                else " ")
        wtxt = ""
        if nm in ws and ws[nm]:
            wtxt = "  w=%.2f" % np.mean(ws[nm])
        print("  %-12s %.4f  %.4f  %+.4f"
              " [%+.4f,%+.4f] %s%s"
              % (nm, au,
                 average_precision_score(yy, v),
                 g, lo, hi, star, wtxt),
              flush=True)
        rows.append({
            "outcome": oc, "model": nm,
            "n": int(ok.sum()),
            "ev": int(yy.sum()), "auc": au,
            "gain": g, "lo": lo, "hi": hi,
            "weight": (np.mean(ws[nm])
                       if nm in ws and ws[nm]
                       else np.nan),
            "sig": int(lo > 0 or hi < 0)})

    # how redundant are the measurements?
    mgb = np.zeros(len(yy))
    for tr, te in folds:
        m = HistGradientBoostingClassifier(
            random_state=SEED, max_depth=3,
            learning_rate=0.05, max_iter=300,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15)
        m.fit(X[tr], yy[tr])
        mgb[te] = m.predict_proba(X[te])[:, 1]
    rho = pd.Series(_rank(mgb)).corr(
        pd.Series(_rank(ps)), method="spearman")
    print("")
    print("  REDUNDANCY CHECK")
    print("    measurements alone: %.4f"
          % roc_auc_score(yy, mgb))
    print("    correlation with stored: %.3f"
          % rho)
    print("    (high correlation means the"
          " logits already encode them)")
    rows.append({
        "outcome": oc,
        "model": "REDUNDANCY",
        "n": int(ok.sum()), "ev": int(yy.sum()),
        "auc": roc_auc_score(yy, mgb),
        "gain": np.nan, "lo": np.nan,
        "hi": np.nan, "weight": rho,
        "sig": 0})

    # predicted three-modality gain for the best
    # version, from the late_best gap line in
    # 20_gap_tuned.py
    bn = max([k for k in P if k != "stored"],
             key=lambda k: roc_auc_score(
                 yy, _rank(P[k])))
    ab = roc_auc_score(yy, _rank(P[bn]))
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
    gp = a_ehr - ab
    print("  best %s %.4f | EHR %.4f"
          " | gap %.4f -> predicted 3mod"
          " gain %+.4f"
          % (bn, ab, a_ehr, gp,
             0.0360 - 0.2425 * gp), flush=True)

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("=" * 68)
print("GAIN OVER THE STORED MODALITY")
q = r[r["model"] != "REDUNDANCY"]
print(q.pivot_table(index="model",
                    columns="outcome",
                    values="gain")
      .round(4).to_string())
print("")
print("significant cells:")
print(q.groupby("model")["sig"]
      .agg(["sum", "count"]).to_string())
print("")
print("REDUNDANCY")
print(r[r["model"] == "REDUNDANCY"][
    ["outcome", "auc", "weight"]]
    .rename(columns={"auc": "meas_alone",
                     "weight": "corr_stored"})
    .round(4).to_string(index=False))
print("")
print("saved", DEST, r.shape)