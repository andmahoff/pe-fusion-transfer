"""Tests whether the derived physiological
feature set improves the ECG modality.

Fits four representations by grouped CV on MIMIC:
  derived  - the v3 waveform measurements alone
  stored   - the existing 71-logit predictions
  both     - derived features plus the stored
             prediction as one more column
  fuse     - the two as separate models, rank
             averaged

Each is measured against the 0.79 the gap
relationship says the modality must reach for a
third modality to earn its place.
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

# columns that are ids or diagnostics, not
# features
DROP = ("subject_id", "hadm_id", "rec",
        "t_flag", "vm_noise", "vm_base",
        "vm_rpeak", "net_i", "net_avf")


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def mk_lr(c):
    return LogisticRegression(C=c,
                              max_iter=5000)


def mk_gb():
    return HistGradientBoostingClassifier(
        random_state=SEED, max_depth=3,
        learning_rate=0.05, max_iter=300,
        l2_regularization=1.0)


def oof(X, y, grp, model="lr"):
    """Out-of-fold predictions, grouped by
    subject. C is chosen inside each training
    fold, so no target leakage."""
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    for tr, te in cv.split(X, y, grp):
        im = SimpleImputer(strategy="median")
        a = im.fit_transform(X[tr])
        b = im.transform(X[te])
        sc = StandardScaler()
        a = sc.fit_transform(a)
        b = sc.transform(b)
        if model == "gb":
            m = mk_gb()
            m.fit(a, y[tr])
        else:
            best, bc = -1.0, 0.1
            icv = StratifiedGroupKFold(
                n_splits=3, shuffle=True,
                random_state=SEED)
            gi = grp[tr]
            for c in CS:
                q = np.zeros(len(tr))
                for t2, v2 in icv.split(
                        a, y[tr], gi):
                    mm = mk_lr(c)
                    mm.fit(a[t2], y[tr][t2])
                    q[v2] = mm.predict_proba(
                        a[v2])[:, 1]
                s = roc_auc_score(y[tr], q)
                if s > best:
                    best, bc = s, c
            m = mk_lr(bc)
            m.fit(a, y[tr])
        p[te] = m.predict_proba(b)[:, 1]
    return p


der = pd.read_csv(os.path.join(
    PROC, "ecg_derived_v3.csv"))
print("derived features:", der.shape)
feat = [c for c in der.columns
        if c not in DROP
        and not c.startswith("net_")
        and pd.api.types.is_numeric_dtype(
            der[c])]
print("usable feature columns:", len(feat))

cov = der[feat].notna().mean()
keep = [c for c in feat if cov[c] >= 0.50]
print("with >=50%% coverage:", len(keep))
print("dropped for sparsity:",
      [c for c in feat if c not in keep][:8])

mim = f.load_mimic()
ins = f.load_inspect()
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

    st_fp = os.path.join(
        FIG, "p_ecg_harm_%s.csv" % oc)
    if not os.path.exists(st_fp):
        continue
    stq = pd.read_csv(st_fp)[
        ["hadm_id", "p_ecg"]].drop_duplicates(
        "hadm_id")

    j = d[["hadm_id"]].merge(
        der[["hadm_id"] + keep],
        on="hadm_id", how="left")
    j = j.merge(stq, on="hadm_id", how="left")

    ok = (j[keep].notna().sum(axis=1) >= 5).values
    ok &= np.isfinite(j["p_ecg"]).values
    yy, gg = y[ok], grp[ok]
    if yy.sum() < 30:
        continue

    print("")
    print("=" * 62)
    print("%s  n=%d ev=%d"
          % (oc, int(ok.sum()), int(yy.sum())))

    Xd = j.loc[ok, keep].values.astype(float)
    ps = j.loc[ok, "p_ecg"].values
    Xb = np.column_stack([Xd, ps])

    res = {}
    res["stored"] = _rank(ps)
    for nm, X, md in (
            ("derived_lr", Xd, "lr"),
            ("derived_gb", Xd, "gb"),
            ("both_lr", Xb, "lr"),
            ("both_gb", Xb, "gb")):
        res[nm] = _rank(oof(X, yy, gg, md))

    best_der = max(
        ["derived_lr", "derived_gb"],
        key=lambda k: roc_auc_score(yy, res[k]))
    res["fuse"] = 0.5 * (res["stored"]
                         + res[best_der])

    # EHR reference for the gap
    se, ysrc = f.labels(ins, "death_30d"
                        if oc ==
                        "death_30d_inhosp"
                        else oc)
    a, b = f.prep(
        se[f.EHR_COLS].values.astype(float),
        d.loc[ok, f.EHR_COLS].values
        .astype(float))
    p_ehr = f.fit_lr(
        a, ysrc, "ehr").predict_proba(b)[:, 1]
    a_ehr = roc_auc_score(yy, _rank(p_ehr))

    print("  EHR reference: %.4f" % a_ehr)
    print("  %-12s %7s %7s %8s %10s"
          % ("model", "AUC", "AP", "gap",
             "pred 3mod"))
    for nm in ["stored", "derived_lr",
               "derived_gb", "both_lr",
               "both_gb", "fuse"]:
        au = roc_auc_score(yy, res[nm])
        ap = average_precision_score(
            yy, res[nm])
        gap = a_ehr - au
        pr = INT2 + SL2 * gap
        mark = " *" if au > roc_auc_score(
            yy, res["stored"]) else "  "
        print("  %-12s %.4f  %.4f  %+.4f"
              "   %+.4f%s"
              % (nm, au, ap, gap, pr, mark))
        rows.append({
            "outcome": oc, "model": nm,
            "n": int(ok.sum()),
            "ev": int(yy.sum()),
            "auc": au, "ap": ap,
            "ehr": a_ehr, "gap": gap,
            "pred_3mod": pr})

r = pd.DataFrame(rows)
r.to_csv(os.path.join(PROC, "ecg_fit.csv"),
         index=False)

print("")
print("=" * 62)
print("IMPROVEMENT OVER THE STORED MODALITY")
print("=" * 62)
for oc, sub in r.groupby("outcome"):
    s0 = float(sub[sub["model"] == "stored"]
               ["auc"].iloc[0])
    b = sub[sub["model"] != "stored"]
    b = b.loc[b["auc"].idxmax()]
    print("  %-18s stored %.4f -> %s %.4f"
          "  (%+.4f)"
          % (oc, s0, b["model"], b["auc"],
             b["auc"] - s0))

print("")
print("DOES IT CLEAR THE BAR?")
print("  a third modality needs the gap under"
      " ~0.145 to pay")
for oc, sub in r.groupby("outcome"):
    b = sub.loc[sub["auc"].idxmax()]
    verdict = ("YES" if b["gap"] < 0.145
               else "no")
    print("  %-18s best %.4f  gap %.4f"
          "  -> %s"
          % (oc, b["auc"], b["gap"], verdict))

print("")
print("saved ecg_fit.csv", r.shape)