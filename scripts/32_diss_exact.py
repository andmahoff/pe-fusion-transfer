"""Exact reproduction of the dissertation's
WMEAN3-CTPA, using the decoupled EHR loader so
the cohort is not capped by the 38-feature CTPA
join. Rebuilds the presentation-window cohort and
the 46-feature CTPA set, fits CTPA by grouped CV
on MIMIC as the dissertation did, keeps the EHR
modality zero-shot from INSPECT, reuses the
stored ECG predictions, and fuses with target
cross-validated weights.
Run in venv (analysis).
"""
import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC = f.PROC
FIG = f.FIG
SCALE = "std"
SEED = 42
NFOLD = 5
WIN_LO, WIN_HI = -48.0, 24.0
REQUIRE_IMPRESSION = True

OUTS = ["death_30d", "composite_30d", "cv_first"]
REF = {"death_30d": 0.8715}


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def window_hadms():
    idx = pd.read_csv(os.path.join(
        FIG, "ctpa_notes_index.csv"))
    w = idx[(idx["h_before"] >= WIN_LO)
            & (idx["h_before"] <= WIN_HI)]
    return set(w["idx_hadm"].dropna().astype(int))


def grid_w(ps, y, step=0.05):
    k = len(ps)
    best, bw = -1.0, tuple([1.0 / k] * k)
    if k == 2:
        for w in np.arange(0, 1.001, step):
            a = roc_auc_score(
                y, w * ps[0] + (1 - w) * ps[1])
            if a > best:
                best, bw = a, (w, 1 - w)
    elif k == 3:
        for w1 in np.arange(0, 1.001, step):
            for w2 in np.arange(
                    0, 1.001 - w1, step):
                w3 = 1 - w1 - w2
                a = roc_auc_score(
                    y, w1 * ps[0] + w2 * ps[1]
                    + w3 * ps[2])
                if a > best:
                    best, bw = a, (w1, w2, w3)
    return bw


def wmean_cv(R, y, grp, cb):
    ps = [R[t] for t in cb]
    out = np.zeros(len(y))
    ws = []
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    X = np.column_stack(ps)
    for tr, te in cv.split(X, y, grp):
        w = grid_w([p[tr] for p in ps], y[tr])
        ws.append(w)
        out[te] = sum(wi * p[te]
                      for wi, p in zip(w, ps))
    return out, np.mean(ws, axis=0)


def ctpa_oof(d, y, cols, seed=SEED):
    X = d[cols].values.astype(float)
    grp = d["subject_id"].values
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(X, y, grp):
        a, b = f.prep(X[tr], X[te], SCALE)
        m = f.fit_lr(a, y[tr], "ctpa")
        p[te] = m.predict_proba(b)[:, 1]
    return p


mm = f.load_mimic_ctpa46()
print("EHR + 46-feature CTPA + labels:", len(mm))

win = window_hadms()
mm = mm[mm["hadm_id"].isin(win)]
print("in presentation window:", len(mm))

if REQUIRE_IMPRESSION:
    v2 = pd.read_csv(os.path.join(
        FIG, "mimic_imp_features_v2.csv"),
        usecols=["hadm_id"])
    keep = set(v2["hadm_id"])
    n0 = len(mm)
    mm = mm[mm["hadm_id"].isin(keep)]
    print("with a usable impression:", len(mm),
          "(dropped %d)" % (n0 - len(mm)))

print("  dissertation used n=1703,"
      " 157 death events")

ins = f.load_inspect()
cols46 = [c for c in f.CTPA46
          if c in mm.columns]
print("  CTPA features available:", len(cols46),
      "of 46")

ecg = {}
for oc in OUTS:
    fp = os.path.join(
        FIG, "p_ecg_harm_%s.csv" % oc)
    if os.path.exists(fp):
        d = pd.read_csv(fp)
        ecg[oc] = d[["hadm_id", "p_ecg"]] \
            .drop_duplicates("hadm_id")

rows = []
for oc in OUTS:
    d, y = f.labels(mm, oc)
    grp = d["subject_id"].values
    se, ysrc = f.labels(ins, oc)

    print("")
    print("=" * 64)
    print("%s   n=%d  ev=%d"
          % (oc, len(y), int(y.sum())))

    a, b = f.prep(
        se[f.EHR_COLS].values.astype(float),
        d[f.EHR_COLS].values.astype(float),
        SCALE)
    me = f.fit_lr(a, ysrc, "ehr")
    P = {"ehr": me.predict_proba(b)[:, 1],
         "ctpa": ctpa_oof(d, y, cols46)}

    if oc in ecg:
        e = d[["hadm_id"]].merge(
            ecg[oc], on="hadm_id", how="left")
        P["ecg"] = e["p_ecg"].values

    for t in sorted(P):
        print("  %-5s coverage %.3f"
              % (t, float(np.isfinite(
                  P[t]).mean())))
    print("")
    print("  %-16s %5s %4s %8s %8s %s"
          % ("model", "n", "ev", "AUC", "AP",
             "weights"))

    names = sorted(P)
    combos = []
    for r_ in range(1, len(names) + 1):
        combos += list(combinations(names, r_))

    for cb in combos:
        ok = np.ones(len(y), dtype=bool)
        for t in cb:
            ok &= np.isfinite(P[t])
        if ok.sum() < 200 or y[ok].sum() < 30:
            continue
        yy = y[ok]
        gg = grp[ok]
        R = {t: _rank(P[t][ok]) for t in cb}
        if len(cb) == 1:
            p, w = R[cb[0]], None
        else:
            p, w = wmean_cv(R, yy, gg, cb)
        auc = roc_auc_score(yy, p)
        ap = average_precision_score(yy, p)
        print("  %-16s %5d %4d  %.4f  %.4f  %s"
              % ("+".join(cb), int(ok.sum()),
                 int(yy.sum()), auc, ap,
                 str([round(x, 2) for x in w])
                 if w is not None else ""))
        rows.append({
            "outcome": oc,
            "model": "+".join(cb),
            "n_mod": len(cb),
            "n": int(ok.sum()),
            "ev": int(yy.sum()), "auc": auc,
            "ap": ap,
            "weights": (str([round(x, 2)
                             for x in w])
                        if w is not None else "")})

r = pd.DataFrame(rows)
r.to_csv(os.path.join(PROC,
                      "diss_exact.csv"),
         index=False)

print("")
print("=" * 64)
print("DIRECT COMPARISON, death_30d")
print("=" * 64)
d = r[r["outcome"] == "death_30d"].sort_values(
    "auc", ascending=False)
print(d.round(4).to_string(index=False))
print("")
print("  dissertation WMEAN3-CTPA: %.4f"
      " on n=1703, 157 events"
      % REF["death_30d"])
t = d[d["model"] == "ctpa+ecg+ehr"]
if len(t):
    v = float(t["auc"].iloc[0])
    print("  reproduction:             %.4f"
          " on n=%d, %d events"
          % (v, int(t["n"].iloc[0]),
             int(t["ev"].iloc[0])))
    print("  difference:               %+.4f"
          % (v - REF["death_30d"]))
t2 = d[d["model"] == "ctpa+ehr"]
if len(t) and len(t2):
    print("")
    print("  two vs three modalities:"
          " %.4f vs %.4f  (%+.4f)"
          % (float(t2["auc"].iloc[0]), v,
             float(t2["auc"].iloc[0]) - v))

print("")
print("ALL OUTCOMES, best model")
for oc, sub in r.groupby("outcome"):
    b = sub.loc[sub["auc"].idxmax()]
    print("  %-14s %-16s %.4f  n=%d ev=%d"
          % (oc, b["model"], b["auc"],
             b["n"], b["ev"]))

print("")
print("saved diss_exact.csv", r.shape)