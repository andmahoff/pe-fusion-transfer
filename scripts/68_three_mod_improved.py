"""Three-modality model with the improved ECG
modality.

Reproduces the dissertation's WMEAN3-CTPA
(published 0.8715, reproduced 0.8720) and reruns
it with the ECG modality tuned, to see whether
the third modality earns its place once it is
properly regularised.

Dissertation protocol:
  cohort   presentation window -48h to +24h,
           46-feature CTPA, EHR zero-shot from
           INSPECT, ECG and CTPA fitted by
           grouped CV on MIMIC
  fusion   weighted rank averaging, weights on a
           0.05 simplex grid, fitted inside each
           training fold
  seeds    six, matching Table 20b

ECG variants compared:
  ecg_pub    C = 1.0, class_weight balanced
             (as published)
  ecg_tuned  C searched inside each fold,
             class_weight balanced
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\three_mod_improved.csv
  results\\three_mod_improved_log.txt
"""
import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
RAW = os.path.join("data", "raw", "ecg")
SEEDS = [42, 7, 13, 1, 2, 3]
NFOLD = 5
CS = [0.0001, 0.001, 0.003, 0.01, 0.03,
      0.1, 0.3, 1.0, 3.0, 10.0]
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
OUTS = ["death_30d", "composite_30d",
        "cv_first"]
DEST = os.path.join(
    PROC, "three_mod_improved.csv")
PUB3 = {"death_30d": 0.8715}


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def lr_fold(Xa, ya, ga, Xb, C=1.0, cw=None,
            tune=False):
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
                random_state=42)
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
    return m.predict_proba(b)[:, 1]


def oof(X, y, grp, seed, C=1.0, cw=None,
        tune=False):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(X, y, grp):
        p[te] = lr_fold(
            X[tr], y[tr], grp[tr], X[te],
            C=C, cw=cw, tune=tune)
    return p


def grid_w(ps, y, step=0.05):
    k = len(ps)
    if k == 1:
        return (1.0,)
    n = int(round(1.0 / step))

    def rec(m, rem):
        if m == 1:
            yield (rem,)
            return
        for i in range(rem + 1):
            for t in rec(m - 1, rem - i):
                yield (i,) + t
    best, bw = -1.0, tuple([1.0 / k] * k)
    if len(np.unique(y)) < 2:
        return bw
    for w in rec(k, n):
        w = tuple(x * step for x in w)
        a = roc_auc_score(
            y, sum(wi * p
                   for wi, p in zip(w, ps)))
        if a > best:
            best, bw = a, w
    return bw


def wcv(ps, y, grp, seed):
    out = np.zeros(len(y))
    ws = []
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    X = np.column_stack(ps)
    for tr, te in cv.split(X, y, grp):
        w = grid_w([p[tr] for p in ps], y[tr])
        ws.append(w)
        out[te] = sum(wi * p[te]
                      for wi, p in zip(w, ps))
    return out, np.mean(ws, axis=0)


# ---------- ECG features, windowed ----------
rec = pd.read_csv(os.path.join(
    PROC, "exp0_logits_record.csv"))
eidx = pd.read_csv(os.path.join(
    RAW, "ecg_record_index.csv"))
eidx["rec"] = eidx["rec"].astype(str).apply(
    lambda s: s if s.startswith("files/")
    else "files/" + s.replace("\\", "/")
    .strip("/"))
LG = [c for c in rec.columns
      if c.startswith("scp_")]
adm = pd.read_csv(os.path.join(
    FIG, "index_admission_times.csv"))
adm["admittime"] = pd.to_datetime(
    adm["admittime"], errors="coerce")

E = eidx.merge(rec, on="rec", how="inner")
E["t"] = pd.to_datetime(
    E["ecg_charttime"], errors="coerce")
E = E.merge(adm[["hadm_id", "admittime"]],
            on="hadm_id", how="left")
E["h"] = ((E["t"] - E["admittime"])
          .dt.total_seconds() / 3600.0)
E = E[(E["h"] >= ECG_LO) & (E["h"] <= ECG_HI)]
ECG = E.groupby(["subject_id", "hadm_id"],
                as_index=False)[LG].mean()
print("ECG admissions:", len(ECG), flush=True)

# ---------- CTPA cohort, 46 features --------
mm = f.load_mimic_ctpa46()
idx = pd.read_csv(os.path.join(
    FIG, "ctpa_notes_index.csv"))
w = idx[(idx["h_before"] >= CT_LO)
        & (idx["h_before"] <= CT_HI)]
hs = set(w["idx_hadm"].dropna().astype(int))
mm = mm[mm["hadm_id"].isin(hs)]
v2 = pd.read_csv(os.path.join(
    FIG, "mimic_imp_features_v2.csv"),
    usecols=["hadm_id"])
mm = mm[mm["hadm_id"].isin(
    set(v2["hadm_id"]))]
CT46 = [c for c in f.CTPA46
        if c in mm.columns]
print("CTPA cohort:", len(mm),
      " features:", len(CT46),
      " (published 1,703)", flush=True)

D = mm.merge(
    ECG.drop(columns=["subject_id"]),
    on="hadm_id", how="inner")
print("three-modality cohort:", len(D),
      flush=True)

ins = f.load_inspect()
rows = []

for oc in OUTS:
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    if y.sum() < 30:
        continue
    src = oc
    se, ysrc = f.labels(ins, src)

    print("")
    print("=" * 74)
    print("%s   n=%d  ev=%d" % (oc, len(y),
                                int(y.sum())),
          flush=True)

    # EHR: zero-shot from INSPECT
    a1, b1 = f.prep(
        se[f.EHR_COLS].values.astype(float),
        d[f.EHR_COLS].values.astype(float))
    p_ehr = _rank(f.fit_lr(
        a1, ysrc, "ehr").predict_proba(
        b1)[:, 1])
    Xc = d[CT46].values.astype(float)
    Xe = d[LG].values.astype(float)

    acc = {k: [] for k in
           ["ehr", "ctpa", "ecg_pub",
            "ecg_tuned", "ctpa+ehr",
            "3mod_pub", "3mod_tuned"]}
    wt = {"3mod_pub": [], "3mod_tuned": []}

    for s in SEEDS:
        p_ct = _rank(oof(Xc, y, grp, s,
                         tune=True))
        p_pub = _rank(oof(
            Xe, y, grp, s, C=1.0,
            cw="balanced"))
        p_tun = _rank(oof(
            Xe, y, grp, s, cw="balanced",
            tune=True))
        two, _ = wcv([p_ct, p_ehr], y, grp, s)
        t3a, w3a = wcv([p_ct, p_pub, p_ehr],
                       y, grp, s)
        t3b, w3b = wcv([p_ct, p_tun, p_ehr],
                       y, grp, s)
        acc["ehr"].append(
            roc_auc_score(y, p_ehr))
        acc["ctpa"].append(
            roc_auc_score(y, p_ct))
        acc["ecg_pub"].append(
            roc_auc_score(y, p_pub))
        acc["ecg_tuned"].append(
            roc_auc_score(y, p_tun))
        acc["ctpa+ehr"].append(
            roc_auc_score(y, two))
        acc["3mod_pub"].append(
            roc_auc_score(y, t3a))
        acc["3mod_tuned"].append(
            roc_auc_score(y, t3b))
        wt["3mod_pub"].append(w3a)
        wt["3mod_tuned"].append(w3b)
        if s == SEEDS[0]:
            keep = (two, t3a, t3b)

    print("  %-12s %8s %8s   %s"
          % ("model", "mean", "SD",
             "per-seed range"))
    for k in ["ehr", "ctpa", "ecg_pub",
              "ecg_tuned", "ctpa+ehr",
              "3mod_pub", "3mod_tuned"]:
        v = np.array(acc[k])
        print("  %-12s %.4f  %.4f   %.4f-%.4f"
              % (k, v.mean(), v.std(ddof=1),
                 v.min(), v.max()), flush=True)
        rows.append({
            "outcome": oc, "model": k,
            "n": len(y), "ev": int(y.sum()),
            "mean": v.mean(),
            "sd": v.std(ddof=1),
            "lo": v.min(), "hi": v.max(),
            "n_seeds": len(v)})

    two, t3a, t3b = keep
    print("")
    print("  PAIRED BOOTSTRAP, seed 42")
    for nm, p in (("3mod_pub", t3a),
                  ("3mod_tuned", t3b)):
        g, lo, hi, _ = f.boot_diff(
            y, p, two, grp)
        print("    %-11s vs ctpa+ehr  %+.4f"
              " [%+.4f,%+.4f] %s"
              % (nm, g, lo, hi,
                 "*" if (lo > 0 or hi < 0)
                 else ""))
        rows.append({
            "outcome": oc,
            "model": nm + "_vs_two",
            "n": len(y), "ev": int(y.sum()),
            "mean": g, "sd": np.nan,
            "lo": lo, "hi": hi,
            "n_seeds": 1})
    g, lo, hi, _ = f.boot_diff(
        y, t3b, t3a, grp)
    print("    tuned vs published ECG   %+.4f"
          " [%+.4f,%+.4f] %s"
          % (g, lo, hi,
             "*" if (lo > 0 or hi < 0)
             else ""))
    rows.append({
        "outcome": oc,
        "model": "tuned_vs_pub_ecg",
        "n": len(y), "ev": int(y.sum()),
        "mean": g, "sd": np.nan, "lo": lo,
        "hi": hi, "n_seeds": 1})

    print("")
    print("  MEAN FUSION WEIGHTS"
          " [ctpa, ecg, ehr]")
    for k in ("3mod_pub", "3mod_tuned"):
        print("    %-11s %s"
              % (k, np.round(
                  np.mean(wt[k], axis=0), 2)))
    if oc in PUB3:
        print("")
        print("  published WMEAN3-CTPA: %.4f"
              % PUB3[oc])

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("=" * 74)
print("SIX-SEED MEAN AUROC")
q = r[~r["model"].str.contains("_vs_")]
print(q.pivot_table(index="model",
                    columns="outcome",
                    values="mean")
      .round(4).to_string())
print("")
print("SIX-SEED SD")
print(q.pivot_table(index="model",
                    columns="outcome",
                    values="sd")
      .round(4).to_string())
print("")
print("DOES THE THIRD MODALITY PAY?")
p = r[r["model"].str.contains("_vs_")]
print(p[["outcome", "model", "mean", "lo",
         "hi"]].round(4)
      .to_string(index=False))
print("")
print("saved", DEST, r.shape)