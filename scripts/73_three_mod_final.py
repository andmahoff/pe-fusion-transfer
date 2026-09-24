"""Three-modality model with the optimised ECG
modality, compared against the dissertation.

ECG configuration per outcome, from the C sweeps
in scripts 70 and 72. Class weighting balanced
throughout, matching the published head:
  cv_first          logits + measurements, C=1e-4
  composite_30d     logits alone,           C=1e-2
  death_30d_inhosp  logits alone,           C=3e-3
  death_30d         logits alone,           C=3e-2

Published head for comparison: logits alone,
C = 1.0.

Everything else follows the dissertation:
  cohort   CTPA presentation window -48h to +24h,
           46-feature CTPA fitted by grouped CV
           on MIMIC, EHR zero-shot from INSPECT
  ECG      -12h to +48h matching window, max over
           the seven signal windows, mean over
           recordings
  fusion   weighted rank averaging, 0.05 simplex
           grid, weights fitted inside each
           training fold
  seeds    six, matching Table 20b

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\three_mod_final.csv
  data\\processed\\three_mod_final_weights.csv
  results\\three_mod_final_log.txt
"""
import os
import sys
import time
import numpy as np
import pandas as pd
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
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
CT_CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0,
         1e1, 1e2]
DROP = ("rec", "t_flag", "vm_noise",
        "vm_base", "vm_rpeak", "net_i",
        "net_avf")
DEST = os.path.join(
    PROC, "three_mod_final.csv")

# outcome -> (use measurements, C)
BEST = {"cv_first": (True, 1e-4),
        "composite_30d": (False, 1e-2),
        "death_30d_inhosp": (False, 3e-3),
        "death_30d": (False, 3e-2)}
OUTS = ["death_30d", "composite_30d",
        "cv_first"]

# dissertation three-modality figures
PUB3 = {"death_30d": 0.8715,
        "composite_30d": 0.8389}
PUB_ECG = {"death_30d": 0.7229,
           "composite_30d": 0.7207,
           "cv_first": 0.6849,
           "death_30d_inhosp": 0.7063}


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def oof_lr(X, y, grp, seed, C, cw="balanced",
           tune_grid=None):
    p = np.zeros(len(y))
    cs = []
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(X, y, grp):
        im = SimpleImputer(strategy="median")
        a = im.fit_transform(X[tr])
        b = im.transform(X[te])
        sc = StandardScaler()
        a, b = sc.fit_transform(a), sc.transform(b)
        c = C
        if tune_grid:
            best = -1.0
            k = min(3, max(2,
                           int(y[tr].sum())
                           // 10))
            try:
                icv = StratifiedGroupKFold(
                    n_splits=k, shuffle=True,
                    random_state=42)
                for cand in tune_grid:
                    q = np.zeros(len(tr))
                    for t2, v2 in icv.split(
                            a, y[tr], grp[tr]):
                        m = LogisticRegression(
                            C=cand,
                            max_iter=6000,
                            class_weight=cw)
                        m.fit(a[t2],
                              y[tr][t2])
                        q[v2] = \
                            m.predict_proba(
                                a[v2])[:, 1]
                    s = roc_auc_score(
                        y[tr], q)
                    if s > best:
                        best, c = s, cand
            except Exception:
                c = C
        cs.append(c)
        m = LogisticRegression(
            C=c, max_iter=6000,
            class_weight=cw)
        m.fit(a, y[tr])
        p[te] = m.predict_proba(b)[:, 1]
    return p, cs


def grid_w(ps, y, step=0.05):
    k = len(ps)
    if k == 1:
        return (1.0,)
    n = int(round(1.0 / step))

    def rec_(m, rem):
        if m == 1:
            yield (rem,)
            return
        for i in range(rem + 1):
            for t in rec_(m - 1, rem - i):
                yield (i,) + t
    best, bw = -1.0, tuple([1.0 / k] * k)
    if len(np.unique(y)) < 2:
        return bw
    for w in rec_(k, n):
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
        w = grid_w([p[tr] for p in ps],
                   y[tr])
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
E = E[(E["h"] >= ECG_LO)
      & (E["h"] <= ECG_HI)].copy()
ECGA = E.groupby(["subject_id", "hadm_id"],
                 as_index=False)[LG].mean()
keep_rec = set(E["rec"])
print("ECG admissions:", len(ECGA))

mv = pd.read_csv(os.path.join(
    PROC, "ecg_derived_v3_record.csv"))
mv["rec"] = mv["rec"].astype(str).apply(
    lambda s: s if s.startswith("files/")
    else "files/" + s.replace("\\", "/")
    .strip("/"))
mv = mv[mv["rec"].isin(keep_rec)]
MC = [c for c in mv.columns
      if c not in DROP
      and pd.api.types.is_numeric_dtype(
          mv[c])]
MV = eidx[["subject_id", "hadm_id",
           "rec"]].merge(
    mv[["rec"] + MC], on="rec", how="inner")
MA = MV.groupby(["subject_id", "hadm_id"],
                as_index=False)[MC].agg(
    ["mean", "max"])
MA.columns = ["%s_%s" % (a, b) if b else a
              for a, b in MA.columns]
MA = MA.reset_index()
MEAS = [c for c in MA.columns
        if c not in ("subject_id",
                     "hadm_id")]
cov = MA[MEAS].notna().mean()
MEAS = [c for c in MEAS if cov[c] >= 0.50]
print("measurement columns:", len(MEAS),
      flush=True)

# ---------- CTPA cohort ----------
mm = f.load_mimic_ctpa46()
cidx = pd.read_csv(os.path.join(
    FIG, "ctpa_notes_index.csv"))
w = cidx[(cidx["h_before"] >= CT_LO)
         & (cidx["h_before"] <= CT_HI)]
hs = set(w["idx_hadm"].dropna().astype(int))
mm = mm[mm["hadm_id"].isin(hs)]
v2 = pd.read_csv(os.path.join(
    FIG, "mimic_imp_features_v2.csv"),
    usecols=["hadm_id"])
mm = mm[mm["hadm_id"].isin(set(v2["hadm_id"]))]
CT46 = [c for c in f.CTPA46
        if c in mm.columns]
print("CTPA cohort:", len(mm),
      " features:", len(CT46),
      " (published 1,703)")

D = mm.merge(ECGA.drop(columns=["subject_id"]),
             on="hadm_id", how="inner")
D = D.merge(MA[["hadm_id"] + MEAS],
            on="hadm_id", how="left")
print("three-modality cohort:", len(D),
      flush=True)

ins = f.load_inspect()
rows, wrows = [], []
t0 = time.time()

for oc in OUTS:
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    if y.sum() < 30:
        continue
    se, ysrc = f.labels(ins, oc)
    use_meas, cbest = BEST[oc]

    print("")
    print("=" * 78)
    print("%s   n=%d  ev=%d" % (oc, len(y),
                                int(y.sum())))
    print("  optimised ECG: %s, C=%.0e"
          % ("logits + measurements"
             if use_meas else "logits alone",
             cbest), flush=True)

    a1, b1 = f.prep(
        se[f.EHR_COLS].values.astype(float),
        d[f.EHR_COLS].values.astype(float))
    p_ehr = _rank(f.fit_lr(
        a1, ysrc, "ehr").predict_proba(
        b1)[:, 1])
    Xc = d[CT46].values.astype(float)
    XL = d[LG].values.astype(float)
    XO = (np.column_stack(
        [XL, d[MEAS].values.astype(float)])
        if use_meas else XL)

    acc = {k: [] for k in
           ["ehr", "ctpa", "ecg_pub",
            "ecg_opt", "ctpa+ehr",
            "3mod_pub", "3mod_opt"]}
    wa = {"3mod_pub": [], "3mod_opt": []}
    keep = None

    for s in SEEDS:
        p_ct, _ = oof_lr(Xc, y, grp, s, 1.0,
                         cw=None,
                         tune_grid=CT_CS)
        p_ct = _rank(p_ct)
        pp, _ = oof_lr(XL, y, grp, s, 1.0)
        pp = _rank(pp)
        po, _ = oof_lr(XO, y, grp, s, cbest)
        po = _rank(po)
        two, _ = wcv([p_ct, p_ehr], y, grp, s)
        t3a, w3a = wcv([p_ct, pp, p_ehr],
                       y, grp, s)
        t3b, w3b = wcv([p_ct, po, p_ehr],
                       y, grp, s)
        for k, v in (("ehr", p_ehr),
                     ("ctpa", p_ct),
                     ("ecg_pub", pp),
                     ("ecg_opt", po),
                     ("ctpa+ehr", two),
                     ("3mod_pub", t3a),
                     ("3mod_opt", t3b)):
            acc[k].append(roc_auc_score(y, v))
        wa["3mod_pub"].append(w3a)
        wa["3mod_opt"].append(w3b)
        if s == SEEDS[0]:
            keep = (two, t3a, t3b, po, pp)

    print("")
    print("  %-11s %8s %8s %8s"
          % ("model", "mean", "SD", "seed42"))
    for k in ["ehr", "ctpa", "ecg_pub",
              "ecg_opt", "ctpa+ehr",
              "3mod_pub", "3mod_opt"]:
        v = np.array(acc[k])
        print("  %-11s %.4f  %.4f  %.4f"
              % (k, v.mean(),
                 v.std(ddof=1), v[0]),
              flush=True)
        rows.append({
            "outcome": oc, "model": k,
            "n": len(y), "ev": int(y.sum()),
            "mean": v.mean(),
            "sd": v.std(ddof=1),
            "seed42": v[0]})

    two, t3a, t3b, po, pp = keep
    print("")
    print("  PAIRED BOOTSTRAP, seed 42")
    for nm, p, ref, rn in (
            ("3mod_pub", t3a, two,
             "ctpa+ehr"),
            ("3mod_opt", t3b, two,
             "ctpa+ehr"),
            ("3mod_opt", t3b, t3a,
             "3mod_pub"),
            ("ecg_opt", po, pp,
             "ecg_pub")):
        g, lo, hi, _ = f.boot_diff(
            y, p, ref, grp)
        print("    %-10s vs %-9s %+.4f"
              " [%+.4f,%+.4f] %s"
              % (nm, rn, g, lo, hi,
                 "*" if (lo > 0 or hi < 0)
                 else ""))
        rows.append({
            "outcome": oc,
            "model": nm + "_vs_" + rn,
            "n": len(y), "ev": int(y.sum()),
            "mean": g, "sd": np.nan,
            "seed42": np.nan,
            "lo": lo, "hi": hi,
            "sig": int(lo > 0 or hi < 0)})

    print("")
    print("  MEAN WEIGHTS [ctpa, ecg, ehr]")
    for k in ("3mod_pub", "3mod_opt"):
        mw = np.mean(wa[k], axis=0)
        print("    %-9s %s"
              % (k, np.round(mw, 2)))
        wrows.append({
            "outcome": oc, "model": k,
            "w_ctpa": mw[0], "w_ecg": mw[1],
            "w_ehr": mw[2]})

    if oc in PUB3:
        print("")
        print("  dissertation WMEAN3-CTPA:"
              " %.4f" % PUB3[oc])
        print("  reproduced (published ECG):"
              " %.4f" % np.mean(
                  acc["3mod_pub"]))
        print("  optimised ECG:            "
              " %.4f  (%+.4f vs published)"
              % (np.mean(acc["3mod_opt"]),
                 np.mean(acc["3mod_opt"])
                 - PUB3[oc]))

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)
pd.DataFrame(wrows).to_csv(
    os.path.join(
        PROC,
        "three_mod_final_weights.csv"),
    index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 78)
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
print("PAIRED COMPARISONS, seed 42")
p = r[r["model"].str.contains("_vs_")]
print(p[["outcome", "model", "mean", "lo",
         "hi", "sig"]].round(4)
      .to_string(index=False))
print("")
print("FUSION WEIGHTS")
print(pd.DataFrame(wrows).round(2)
      .to_string(index=False))
print("")
print("=" * 78)
print("AGAINST THE DISSERTATION")
print("  %-16s %10s %10s %10s %9s"
      % ("outcome", "published",
         "reproduced", "optimised",
         "change"))
for oc in OUTS:
    s = q[q["outcome"] == oc]
    if not len(s) or oc not in PUB3:
        continue
    a = float(s[s["model"] == "3mod_pub"]
              ["mean"].iloc[0])
    b = float(s[s["model"] == "3mod_opt"]
              ["mean"].iloc[0])
    print("  %-16s %10.4f %10.4f %10.4f"
          " %+9.4f"
          % (oc, PUB3[oc], a, b,
             b - PUB3[oc]))
print("")
print("  ECG modality alone")
print("  %-16s %10s %10s %9s"
      % ("outcome", "published",
         "optimised", "change"))
for oc in OUTS:
    s = q[q["outcome"] == oc]
    if not len(s):
        continue
    b = float(s[s["model"] == "ecg_opt"]
              ["mean"].iloc[0])
    print("  %-16s %10.4f %10.4f %+9.4f"
          % (oc, PUB_ECG[oc], b,
             b - PUB_ECG[oc]))
print("")
print("saved", DEST, r.shape)