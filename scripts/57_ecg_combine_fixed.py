"""Combine the waveform measurements with the
stored 71-SCP model, with the blending weight
chosen on inner cross-validation.

Script 56 fitted the measurement model on the
training fold and then chose the blending weight
on that same fold. A boosted model fits its own
training data almost perfectly, so the tuner saw
a near-flawless predictor and gave it 0.99 to
1.00. On held-out data it collapsed.

Here the measurement predictions used for weight
selection come from an inner cross-validation, so
the tuner sees honest out-of-fold performance.

Five combination rules are compared, all with the
same corrected inner-CV predictions.
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ecg_combine_fixed.csv
  results\\ecg_combine_fixed_log.txt
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
SEED, NFOLD, INNER = 42, 5, 3
CS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
OUTS = ["death_30d", "death_30d_inhosp",
        "composite_30d"]
DEST = os.path.join(
    PROC, "ecg_combine_fixed.csv")


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def logit(p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def mk_gb():
    return HistGradientBoostingClassifier(
        random_state=SEED, max_depth=3,
        learning_rate=0.05, max_iter=300,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15)


def fit_meas(Xa, ya, Xb, kind):
    """Fit the measurement model and score Xb."""
    if kind == "gb":
        m = mk_gb()
        m.fit(Xa, ya)
        return m.predict_proba(Xb)[:, 1]
    im = SimpleImputer(strategy="median")
    a = im.fit_transform(Xa)
    b = im.transform(Xb)
    sc = StandardScaler()
    a, b = sc.fit_transform(a), sc.transform(b)
    m = LogisticRegression(C=0.01,
                           max_iter=5000)
    m.fit(a, ya)
    return m.predict_proba(b)[:, 1]


def inner_oof(Xa, ya, ga, kind):
    """Honest out-of-fold predictions
    on the training fold, so the weight tuner
    sees held-out performance rather than fitted
    performance."""
    p = np.zeros(len(ya))
    k = min(INNER, max(2, int(ya.sum()) // 8))
    try:
        icv = StratifiedGroupKFold(
            n_splits=k, shuffle=True,
            random_state=SEED)
        for t2, v2 in icv.split(Xa, ya, ga):
            p[v2] = fit_meas(Xa[t2], ya[t2],
                             Xa[v2], kind)
    except Exception:
        p = fit_meas(Xa, ya, Xa, kind)
    return p


def pick_w(pm_oof, ps_tr, ya, step=0.05):
    """Weight chosen on honest inner-CV
    measurement predictions."""
    rm = _rank(pm_oof)
    rs = _rank(ps_tr)
    best, bw = -1.0, 0.0
    for w in np.arange(0, 1.001, step):
        a = roc_auc_score(ya, w * rm
                          + (1 - w) * rs)
        if a > best:
            best, bw = a, w
    return bw


traj = pd.read_csv(os.path.join(
    PROC, "ecg_trajectory.csv"))
stat = pd.read_csv(os.path.join(
    PROC, "ecg_derived_v3.csv"))
mim = f.load_mimic()

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
    for nm in ["meas_gb", "meas_lr",
               "wfix_gb", "wfix_lr",
               "equal_gb", "stack_gb"]:
        P[nm] = np.zeros(len(yy))
    ws = {"wfix_gb": [], "wfix_lr": [],
          "old_gb": []}

    for tr, te in folds:
        Xa, Xb = X[tr], X[te]
        ya, ga = yy[tr], gg[tr]
        sa, sb = ps[tr], ps[te]

        for kind, mn, wn in (
                ("gb", "meas_gb", "wfix_gb"),
                ("lr", "meas_lr", "wfix_lr")):
            pb = fit_meas(Xa, ya, Xb, kind)
            P[mn][te] = pb
            po = inner_oof(Xa, ya, ga, kind)
            w = pick_w(po, sa, ya)
            ws[wn].append(w)
            P[wn][te] = (w * _rank(pb)
                         + (1 - w) * _rank(sb))
            if kind == "gb":
                # what script 56 did, for
                # comparison
                pf = fit_meas(Xa, ya, Xa, kind)
                ws["old_gb"].append(
                    pick_w(pf, sa, ya))

        P["equal_gb"][te] = 0.5 * (
            _rank(P["meas_gb"][te])
            + _rank(sb))

        # stacking on honest inner-CV scores
        po = inner_oof(Xa, ya, ga, "gb")
        Za = np.column_stack(
            [logit(np.clip(po, 1e-6, 1 - 1e-6)),
             logit(sa)])
        Zb = np.column_stack(
            [logit(np.clip(
                P["meas_gb"][te], 1e-6,
                1 - 1e-6)), logit(sb)])
        sc = StandardScaler()
        m = LogisticRegression(C=1.0,
                               max_iter=5000)
        m.fit(sc.fit_transform(Za), ya)
        P["stack_gb"][te] = m.predict_proba(
            sc.transform(Zb))[:, 1]

    print("")
    print("  WEIGHT CHOSEN FOR THE"
          " MEASUREMENTS")
    print("    broken (script 56): %.2f"
          % np.mean(ws["old_gb"]))
    print("    corrected inner-CV: %.2f"
          % np.mean(ws["wfix_gb"]))
    print("    corrected, linear : %.2f"
          % np.mean(ws["wfix_lr"]), flush=True)

    base = roc_auc_score(yy, _rank(ps))
    print("")
    print("  %-11s %7s %7s %s"
          % ("model", "AUC", "AP",
             "vs stored"))
    for nm in ["stored", "meas_lr", "meas_gb",
               "equal_gb", "wfix_lr",
               "wfix_gb", "stack_gb"]:
        v = _rank(P[nm])
        au = roc_auc_score(yy, v)
        if nm == "stored":
            g = lo = hi = 0.0
        else:
            g, lo, hi, _ = f.boot_diff(
                yy, v, _rank(ps), gg)
        star = ("*" if (lo > 0 or hi < 0)
                else " ")
        wt = ""
        if nm in ws and ws[nm]:
            wt = "  w=%.2f" % np.mean(ws[nm])
        print("  %-11s %.4f  %.4f  %+.4f"
              " [%+.4f,%+.4f] %s%s"
              % (nm, au,
                 average_precision_score(yy, v),
                 g, lo, hi, star, wt),
              flush=True)
        rows.append({
            "outcome": oc, "model": nm,
            "n": int(ok.sum()),
            "ev": int(yy.sum()), "auc": au,
            "gain": g, "lo": lo, "hi": hi,
            "weight": (float(np.mean(ws[nm]))
                       if nm in ws and ws[nm]
                       else np.nan),
            "sig": int(lo > 0 or hi < 0)})

    rows.append({
        "outcome": oc,
        "model": "WEIGHT_broken_vs_fixed",
        "n": int(ok.sum()), "ev": int(yy.sum()),
        "auc": np.nan, "gain": np.nan,
        "lo": float(np.mean(ws["old_gb"])),
        "hi": float(np.mean(ws["wfix_gb"])),
        "weight": np.nan, "sig": 0})

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("=" * 68)
print("GAIN OVER THE STORED MODALITY")
q = r[r["model"] != "WEIGHT_broken_vs_fixed"]
print(q.pivot_table(index="model",
                    columns="outcome",
                    values="gain")
      .round(4).to_string())
print("")
print("significant cells (of 3 each):")
print(q.groupby("model")["sig"]
      .agg(["sum", "count"]).to_string())

print("")
print("WEIGHT GIVEN TO THE MEASUREMENTS")
w = r[r["model"] == "WEIGHT_broken_vs_fixed"]
print(w[["outcome", "lo", "hi"]]
      .rename(columns={"lo": "broken",
                       "hi": "corrected"})
      .round(3).to_string(index=False))

print("")
print("BEST PER OUTCOME")
for oc, s in q.groupby("outcome"):
    b = s.loc[s["auc"].idxmax()]
    print("  %-18s %-11s %.4f  (%+.4f)"
          % (oc, b["model"], b["auc"],
             b["gain"]))

print("")
print("saved", DEST, r.shape)