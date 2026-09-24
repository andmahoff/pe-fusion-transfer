"""Test adjustments against the reproduced
dissertation ECG model, on the windowed cohort.

Baseline (Tables 8 and 43, plus the Section 3.4
matching window): 71 SCP logits, max over the
seven windows, mean over recordings, recordings
restricted to -12h to +48h relative to
admission, L2 logistic regression C = 1.0,
class_weight = "balanced", max_iter 5000,
StratifiedGroupKFold(5) grouped on subject,
seed 42.

That reproduction gives 0.7164, 0.7188, 0.7060
and 0.6983 against published 0.7207, 0.7229,
0.7063 and 0.6849, with matching event counts.

Adjustments tested:
  1 C tuning       the dissertation states no
                   search over C was performed
  2 class weight   balanced was used, and it
                   distorts the probability
                   scale before rank averaging
  3 boosting       over 71 named statements
  4 trajectory     change in each statement
                   across serial recordings

All comparisons share folds, so differences are
attributable to the change rather than to
partitioning. Multi-seed columns are reported
because the fold-seed SD is 0.003 to 0.007.
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ecg_adjust.csv
  data\\processed\\ecg_adjust_traj.csv
  data\\processed\\ecg_adjust_coefs.csv
  results\\ecg_adjust_log.txt
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
RAW = os.path.join("data", "raw", "ecg")
SEED, NFOLD = 42, 5
ALT_SEEDS = [42, 7, 13]
CS = [0.0001, 0.001, 0.003, 0.01, 0.03,
      0.1, 0.3, 1.0, 3.0, 10.0]
WIN_LO, WIN_HI = -12.0, 48.0
OUTS = ["composite_30d", "death_30d",
        "death_30d_inhosp", "cv_first"]
DEST = os.path.join(PROC, "ecg_adjust.csv")

PUB = {"composite_30d": 0.7207,
       "death_30d": 0.7229,
       "death_30d_inhosp": 0.7063,
       "cv_first": 0.6849}


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def lr_fold(Xa, ya, ga, Xb, C=1.0,
            cw="balanced", tune=False):
    im = SimpleImputer(strategy="median")
    a = im.fit_transform(Xa)
    b = im.transform(Xb)
    sc = StandardScaler()
    a, b = sc.fit_transform(a), sc.transform(b)
    c = C
    if tune:
        best = -1.0
        k = min(3, max(2, int(ya.sum()) // 10))
        try:
            icv = StratifiedGroupKFold(
                n_splits=k, shuffle=True,
                random_state=SEED)
            for cand in CS:
                q = np.zeros(len(ya))
                for t2, v2 in icv.split(
                        a, ya, ga):
                    m = LogisticRegression(
                        C=cand, max_iter=5000,
                        class_weight=cw)
                    m.fit(a[t2], ya[t2])
                    q[v2] = m.predict_proba(
                        a[v2])[:, 1]
                s = roc_auc_score(ya, q)
                if s > best:
                    best, c = s, cand
        except Exception:
            c = C
    m = LogisticRegression(
        C=c, max_iter=5000, class_weight=cw)
    m.fit(a, ya)
    return m.predict_proba(b)[:, 1], c


def gb_fold(Xa, ya, Xb, **kw):
    p = dict(random_state=SEED, max_depth=3,
             learning_rate=0.05, max_iter=300,
             l2_regularization=1.0,
             early_stopping=True,
             validation_fraction=0.15)
    p.update(kw)
    m = HistGradientBoostingClassifier(**p)
    m.fit(Xa, ya)
    return m.predict_proba(Xb)[:, 1]


def oof(X, y, folds, grp, kind="lr", **kw):
    p = np.zeros(len(y))
    cs = []
    for tr, te in folds:
        if kind == "gb":
            p[te] = gb_fold(X[tr], y[tr],
                            X[te], **kw)
        else:
            p[te], c = lr_fold(
                X[tr], y[tr], grp[tr],
                X[te], **kw)
            cs.append(c)
    return p, cs


# ---------- windowed features ----------
rec = pd.read_csv(os.path.join(
    PROC, "exp0_logits_record.csv"))
idx = pd.read_csv(os.path.join(
    RAW, "ecg_record_index.csv"))
idx["rec"] = idx["rec"].astype(str).apply(
    lambda s: s if s.startswith("files/")
    else "files/" + s.replace("\\", "/")
    .strip("/"))
LG = [c for c in rec.columns
      if c.startswith("scp_")]

adm = pd.read_csv(os.path.join(
    FIG, "index_admission_times.csv"))
adm["admittime"] = pd.to_datetime(
    adm["admittime"], errors="coerce")

R = idx.merge(rec, on="rec", how="inner")
R["ecg_t"] = pd.to_datetime(
    R["ecg_charttime"], errors="coerce")
R = R.merge(adm[["hadm_id", "admittime"]],
            on="hadm_id", how="left")
R["h_rel"] = ((R["ecg_t"] - R["admittime"])
              .dt.total_seconds() / 3600.0)
n0 = R["hadm_id"].nunique()
R = R[(R["h_rel"] >= WIN_LO)
      & (R["h_rel"] <= WIN_HI)].copy()
print("window %+.0fh to %+.0fh applied"
      % (WIN_LO, WIN_HI))
print("  recordings:", len(R))
print("  admissions:", R["hadm_id"].nunique(),
      "of", n0, " (published 3,507)",
      flush=True)

A = R.groupby(["subject_id", "hadm_id"],
              as_index=False)[LG].mean()

# ---------- trajectory, windowed ----------
Rs = R.dropna(subset=["ecg_t"]).sort_values(
    ["hadm_id", "ecg_t"])
rows = []
for h, g in Rs.groupby("hadm_id"):
    if len(g) < 2:
        continue
    d = {"hadm_id": h, "n_rec": len(g)}
    hrs = (g["ecg_t"].iloc[-1]
           - g["ecg_t"].iloc[0]).total_seconds()
    d["span_h"] = hrs / 3600.0
    for c in LG:
        v = g[c].values
        d[c + "_delta"] = float(v[-1] - v[0])
        d[c + "_sd"] = float(np.nanstd(v))
    rows.append(d)
TJ = pd.DataFrame(rows)
TJ.to_csv(os.path.join(
    PROC, "ecg_adjust_traj.csv"), index=False)
TR = [c for c in TJ.columns if c != "hadm_id"]
print("trajectory:", TJ.shape, flush=True)

lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
D = A.merge(lab, on=["subject_id", "hadm_id"],
            how="inner")
D = D.merge(TJ, on="hadm_id", how="left")
print("cohort:", len(D), flush=True)

res, coefs = [], []
for oc in OUTS:
    if oc not in D.columns:
        continue
    d = D
    if oc == "cv_first":
        d = D[D["death_first"] == 0]
    y = pd.to_numeric(d[oc], errors="coerce")
    y = y.fillna(0).astype(int).values
    grp = d["subject_id"].values
    X = d[LG].values.astype(float)
    Xt = d[TR].values.astype(float)
    multi = (d[TR].notna().sum(axis=1)
             >= 10).values

    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    folds = list(cv.split(X, y, grp))

    print("")
    print("=" * 74)
    print("%s  n=%d ev=%d  multi=%d"
          "   published %.4f"
          % (oc, len(y), int(y.sum()),
             int(multi.sum()), PUB[oc]),
          flush=True)

    P, CH = {}, {}
    P["baseline"], _ = oof(
        X, y, folds, grp, C=1.0,
        cw="balanced")
    P["no_classweight"], _ = oof(
        X, y, folds, grp, C=1.0, cw=None)
    P["C_tuned_bal"], CH["C_tuned_bal"] = oof(
        X, y, folds, grp, cw="balanced",
        tune=True)
    P["C_tuned_none"], CH["C_tuned_none"] = oof(
        X, y, folds, grp, cw=None, tune=True)
    P["boosting"], _ = oof(
        X, y, folds, grp, kind="gb")
    P["logit_traj"], CH["logit_traj"] = oof(
        np.column_stack([X, Xt]), y, folds,
        grp, cw=None, tune=True)
    bl = max(["C_tuned_none", "C_tuned_bal"],
             key=lambda k: roc_auc_score(
                 y, P[k]))
    P["blend_gb"] = 0.5 * (
        _rank(P[bl]) + _rank(P["boosting"]))

    base = _rank(P["baseline"])
    ab = roc_auc_score(y, base)
    print("  baseline %.4f  (published %.4f,"
          " diff %+.4f)"
          % (ab, PUB[oc], ab - PUB[oc]))
    print("")
    print("  %-16s %7s %7s %-26s %s"
          % ("adjustment", "AUC", "AP",
             "vs baseline", "3-seed"))

    for nm in ["baseline", "no_classweight",
               "C_tuned_bal", "C_tuned_none",
               "boosting", "logit_traj",
               "blend_gb"]:
        v = _rank(P[nm])
        au = roc_auc_score(y, v)
        if nm == "baseline":
            g = lo = hi = 0.0
        else:
            g, lo, hi, _ = f.boot_diff(
                y, v, base, grp)
        star = ("*" if (lo > 0 or hi < 0)
                else " ")

        # stability across fold seeds
        ss = []
        for s2 in ALT_SEEDS:
            cv2 = StratifiedGroupKFold(
                n_splits=NFOLD, shuffle=True,
                random_state=s2)
            fo = list(cv2.split(X, y, grp))
            if nm == "baseline":
                q, _ = oof(X, y, fo, grp,
                           C=1.0,
                           cw="balanced")
            elif nm == "no_classweight":
                q, _ = oof(X, y, fo, grp,
                           C=1.0, cw=None)
            elif nm == "C_tuned_bal":
                q, _ = oof(X, y, fo, grp,
                           cw="balanced",
                           tune=True)
            elif nm == "C_tuned_none":
                q, _ = oof(X, y, fo, grp,
                           cw=None, tune=True)
            elif nm == "boosting":
                q, _ = oof(X, y, fo, grp,
                           kind="gb")
            elif nm == "logit_traj":
                q, _ = oof(
                    np.column_stack([X, Xt]),
                    y, fo, grp, cw=None,
                    tune=True)
            else:
                q = None
            if q is not None:
                ss.append(roc_auc_score(
                    y, _rank(q)))
        sd = (float(np.std(ss, ddof=1))
              if len(ss) > 1 else np.nan)
        mn = (float(np.mean(ss))
              if ss else np.nan)

        ct = ("  C=%s" % sorted(set(CH[nm]))
              if nm in CH else "")
        print("  %-16s %.4f  %.4f  %+.4f"
              " [%+.4f,%+.4f] %s"
              "  %.4f+-%.4f%s"
              % (nm, au,
                 average_precision_score(y, v),
                 g, lo, hi, star, mn, sd, ct),
              flush=True)
        res.append({
            "outcome": oc, "model": nm,
            "n": len(y), "ev": int(y.sum()),
            "auc": au,
            "ap": average_precision_score(
                y, v),
            "published": PUB[oc],
            "gain": g, "lo": lo, "hi": hi,
            "seed_mean": mn, "seed_sd": sd,
            "C": (str(sorted(set(CH[nm])))
                  if nm in CH else ""),
            "sig": int(lo > 0 or hi < 0)})

    im = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Z = sc.fit_transform(im.fit_transform(X))
    cl = CH.get("C_tuned_none", [1.0])
    cb = sorted(set(cl))[len(set(cl)) // 2]
    mm = LogisticRegression(C=cb,
                            max_iter=5000)
    mm.fit(Z, y)
    co = mm.coef_[0]
    print("")
    print("  TOP STATEMENTS (C=%.4f,"
          " no class weight)" % cb)
    for k in np.argsort(-np.abs(co))[:10]:
        print("    %-12s %+.4f"
              % (LG[k].replace("scp_", ""),
                 co[k]))
        coefs.append({
            "outcome": oc,
            "statement":
                LG[k].replace("scp_", ""),
            "coef": float(co[k]), "C": cb})

    bn = max([k for k in P if k != "baseline"],
             key=lambda k: roc_auc_score(
                 y, _rank(P[k])))
    ba = roc_auc_score(y, _rank(P[bn]))
    print("")
    print("  best adjustment: %s  %.4f"
          "  (%+.4f over baseline,"
          " %+.4f over published)"
          % (bn, ba, ba - ab, ba - PUB[oc]),
          flush=True)

r = pd.DataFrame(res)
r.to_csv(DEST, index=False)
pd.DataFrame(coefs).to_csv(
    os.path.join(PROC,
                 "ecg_adjust_coefs.csv"),
    index=False)

print("")
print("=" * 74)
print("AUROC BY MODEL (windowed cohort)")
print(r.pivot_table(index="model",
                    columns="outcome",
                    values="auc")
      .round(4).to_string())
print("")
print("GAIN OVER THE REPRODUCED BASELINE")
print(r.pivot_table(index="model",
                    columns="outcome",
                    values="gain")
      .round(4).to_string())
print("")
print("significant cells (of 4 each):")
print(r.groupby("model")["sig"]
      .agg(["sum", "count"]).to_string())
print("")
print("THREE-SEED MEAN, for stability")
print(r.pivot_table(index="model",
                    columns="outcome",
                    values="seed_mean")
      .round(4).to_string())
print("")
print("SELECTED C VALUES")
print(r[r["C"] != ""][["outcome", "model",
                       "C"]]
      .to_string(index=False))
print("")
print("BEST PER OUTCOME")
for oc, s in r.groupby("outcome"):
    b = s.loc[s["auc"].idxmax()]
    print("  %-18s %-16s %.4f"
          "  (%+.4f vs baseline,"
          " published %.4f)"
          % (oc, b["model"], b["auc"],
             b["gain"], b["published"]))
print("")
print("saved", DEST, r.shape)