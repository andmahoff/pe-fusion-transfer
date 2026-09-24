"""Final fusion: the optimised ECG modality in
both the three-modality and the EHR+ECG models.

ECG configuration, from script 87:
  L+v12+strict, meaning the 71 SCP logits plus
  the v12 measurements plus the strict Tp-e
  columns, C fixed per outcome at its swept
  optimum:
    cv_first          1e-4
    composite_30d     1e-3
    death_30d         1e-3
    death_30d_inhosp  3e-4
  class weighting balanced, as published.

MODEL 1  THREE-MODALITY on the CTPA
         presentation-window cohort, reproducing
         WMEAN3-CTPA. Published: death_30d
         0.8715, composite_30d 0.8389.

MODEL 2  EHR + ECG on the full ECG cohort, which
         needs no scan. Table 23 reported EHR+ECG
         at 0.7845 on cv_first against EHR alone
         at 0.7916, so the published ECG modality
         was harming that pairing. This tests
         whether the optimised one reverses it.

Six seeds, C fixed rather than swept.
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\fusion_final.csv
  data\\processed\\fusion_final_weights.csv
  results\\fusion_final_log.txt
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
MINCOV = 0.20
DEST = os.path.join(PROC,
                    "fusion_final.csv")

BEST_C = {"cv_first": 1e-4,
          "composite_30d": 1e-3,
          "death_30d": 1e-3,
          "death_30d_inhosp": 3e-4}
OUTS = ["death_30d", "composite_30d",
        "cv_first", "death_30d_inhosp"]

PUB3 = {"death_30d": 0.8715,
        "composite_30d": 0.8389}
PUB_ECG = {"death_30d": 0.7229,
           "composite_30d": 0.7207,
           "cv_first": 0.6849,
           "death_30d_inhosp": 0.7063}
PUB_CT = {"cv_first": {"ehr": 0.7916,
                       "ecg": 0.6574,
                       "ctpa": 0.5761,
                       "ehr_ecg": 0.7845,
                       "ehr_ctpa": 0.7906}}

DROP_EXACT = ("rec", "t_flag", "vm_noise",
              "vm_base", "vm_rpeak", "net_i",
              "net_avf", "apen", "tp_win_ms",
              "t_win_ms", "t_peak_ms",
              "tp_fallback", "qrs_hit_limit",
              "p_hit_limit", "t_peak_edge",
              "t_edge_leads", "n_rr_dropped",
              "any_af", "tpe_spread_ms",
              "tpe_ok")
DROP_PREFIX = ("tpk_ms_",)
PERM_ONLY = ("tpe_ms_mean", "tpe_ms_worst",
             "tpe_qt_mean", "tpe_qt_worst")
NEG_BAD = ("s_wave_i", "q_wave_iii", "st_min",
           "t_v1", "t_v2", "t_v3", "t_v4",
           "t_v5", "t_v6", "t_wave_iii")
ANG = ("qrs_axis", "t_axis", "p_axis")


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def oof(X, y, grp, seed, C, cw="balanced",
        tune=None):
    p = np.zeros(len(y))
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
        if tune:
            best = -1.0
            k = min(3, max(2,
                           int(y[tr].sum())
                           // 10))
            try:
                icv = StratifiedGroupKFold(
                    n_splits=k, shuffle=True,
                    random_state=42)
                for cand in tune:
                    qq = np.zeros(len(tr))
                    for t2, v2 in icv.split(
                            a, y[tr], grp[tr]):
                        mm = LogisticRegression(
                            C=cand,
                            max_iter=8000,
                            class_weight=cw)
                        mm.fit(a[t2],
                               y[tr][t2])
                        qq[v2] = \
                            mm.predict_proba(
                                a[v2])[:, 1]
                    s = roc_auc_score(
                        y[tr], qq)
                    if s > best:
                        best, c = s, cand
            except Exception:
                c = C
        m = LogisticRegression(
            C=c, max_iter=8000,
            class_weight=cw)
        m.fit(a, y[tr])
        p[te] = m.predict_proba(b)[:, 1]
    return p


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
keep = set(E["rec"])
print("ECG admissions:", len(ECGA), flush=True)

m12 = pd.read_csv(os.path.join(
    PROC, "ecg_derived_v12_record.csv"))
m12["rec"] = m12["rec"].astype(str).apply(
    lambda s: s if s.startswith("files/")
    else "files/" + s.replace("\\", "/")
    .strip("/"))
m12 = m12[m12["rec"].isin(keep)]
cols = [c for c in m12.columns
        if c not in DROP_EXACT
        and not c.startswith(DROP_PREFIX)
        and pd.api.types.is_numeric_dtype(
            m12[c])]
j = eidx[["subject_id", "hadm_id",
          "rec"]].merge(
    m12[["rec"] + cols], on="rec",
    how="inner")
g = j.groupby(["subject_id", "hadm_id"])
pos = [c for c in cols
       if not c.startswith(NEG_BAD)
       and c not in ANG]
neg = [c for c in cols
       if c.startswith(NEG_BAD)]
parts = [g[cols].mean().add_suffix("_mean")]
if pos:
    parts.append(g[pos].max()
                 .add_suffix("_worst"))
if neg:
    parts.append(g[neg].min()
                 .add_suffix("_worst"))
MA = pd.concat(parts, axis=1).reset_index()
for c in ANG:
    if c not in j.columns:
        continue
    rad = np.radians(j[c])
    tmp = j[["subject_id", "hadm_id"]].copy()
    tmp["s"], tmp["c"] = np.sin(rad), np.cos(rad)
    gg = tmp.groupby(["subject_id",
                      "hadm_id"]).mean()
    cm = np.degrees(np.arctan2(
        gg["s"], gg["c"])).rename(
        c + "_circ").reset_index()
    MA = MA.merge(cm, on=["subject_id",
                          "hadm_id"],
                  how="left")
    MA = MA.drop(columns=[c + "_mean",
                          c + "_worst"],
                 errors="ignore")
MC = [c for c in MA.columns
      if c not in ("subject_id", "hadm_id")]
cov = MA[MC].notna().mean()
MC = [c for c in MC if cov[c] >= MINCOV
      and c not in PERM_ONLY]
print("measurement columns:", len(MC),
      " (permissive Tp-e excluded, strict"
      " retained)", flush=True)

lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))

# ---- the EHR columns live in the MIMIC
# cohort, not in the ECG or label files ----
mim_ehr = f.load_mimic_ehr()
EH = [c for c in f.EHR_COLS
      if c in mim_ehr.columns]
print("EHR columns found:", len(EH),
      "of", len(f.EHR_COLS))
if len(EH) < len(f.EHR_COLS):
    print("  missing:",
          [c for c in f.EHR_COLS
           if c not in mim_ehr.columns][:8])

FULL = ECGA.merge(
    MA[["hadm_id"] + MC], on="hadm_id",
    how="left").merge(
    lab, on=["subject_id", "hadm_id"],
    how="inner").merge(
    mim_ehr[["hadm_id"] + EH],
    on="hadm_id", how="inner")
print("full ECG cohort:", len(FULL),
      " (requires both an ECG and EHR"
      " features)", flush=True)

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
CT = mm.merge(
    ECGA.drop(columns=["subject_id"]),
    on="hadm_id", how="inner").merge(
    MA[["hadm_id"] + MC], on="hadm_id",
    how="left")
print("CTPA cohort:", len(CT),
      " (published 1,703)", flush=True)

ins = f.load_inspect()
rows, wrows = [], []
t0 = time.time()

# ============ MODEL 1: THREE-MODALITY ========
print("")
print("#" * 78)
print("MODEL 1: THREE-MODALITY, CTPA cohort")
print("#" * 78, flush=True)

for oc in OUTS:
    if oc not in CT.columns:
        continue
    d, y = f.labels(CT, oc)
    grp = d["subject_id"].values
    if y.sum() < 30:
        continue
    se, ysrc = f.labels(
        ins, "death_30d"
        if oc == "death_30d_inhosp" else oc)
    cb = BEST_C[oc]

    print("")
    print("=" * 78)
    print("%s   n=%d ev=%d   ECG C=%.0e"
          % (oc, len(y), int(y.sum()), cb),
          flush=True)

    a1, b1 = f.prep(
        se[f.EHR_COLS].values.astype(float),
        d[f.EHR_COLS].values.astype(float))
    p_ehr = _rank(f.fit_lr(
        a1, ysrc, "ehr").predict_proba(
        b1)[:, 1])
    Xc = d[CT46].values.astype(float)
    XL = d[LG].values.astype(float)
    XO = np.column_stack(
        [XL, d[MC].values.astype(float)])

    acc = {k: [] for k in
           ["ehr", "ctpa", "ecg_pub",
            "ecg_opt", "ctpa+ehr",
            "3mod_pub", "3mod_opt"]}
    wa = {"3mod_pub": [], "3mod_opt": []}
    keep2 = None
    for s in SEEDS:
        p_ct = _rank(oof(Xc, y, grp, s, 1.0,
                         cw=None,
                         tune=CT_CS))
        pp = _rank(oof(XL, y, grp, s, 1.0))
        po = _rank(oof(XO, y, grp, s, cb))
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
            keep2 = (two, t3a, t3b, po, pp)

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
            "model_type": "3mod",
            "outcome": oc, "model": k,
            "n": len(y), "ev": int(y.sum()),
            "mean": v.mean(),
            "sd": v.std(ddof=1),
            "seed42": v[0]})

    two, t3a, t3b, po, pp = keep2
    print("")
    print("  PAIRED BOOTSTRAP, seed 42")
    for nm, p, ref, rn in (
            ("3mod_pub", t3a, two, "ctpa+ehr"),
            ("3mod_opt", t3b, two, "ctpa+ehr"),
            ("3mod_opt", t3b, t3a, "3mod_pub"),
            ("ecg_opt", po, pp, "ecg_pub")):
        gn, lo, hi, _ = f.boot_diff(
            y, p, ref, grp)
        print("    %-10s vs %-9s %+.4f"
              " [%+.4f,%+.4f] %s"
              % (nm, rn, gn, lo, hi,
                 "*" if (lo > 0 or hi < 0)
                 else ""))
        rows.append({
            "model_type": "3mod",
            "outcome": oc,
            "model": nm + "_vs_" + rn,
            "n": len(y), "ev": int(y.sum()),
            "mean": gn, "sd": np.nan,
            "seed42": np.nan, "lo": lo,
            "hi": hi,
            "sig": int(lo > 0 or hi < 0)})

    print("")
    print("  WEIGHTS [ctpa, ecg, ehr]")
    for k in ("3mod_pub", "3mod_opt"):
        mw = np.mean(wa[k], axis=0)
        print("    %-9s %s" % (
            k, np.round(mw, 2)))
        wrows.append({
            "model_type": "3mod",
            "outcome": oc, "model": k,
            "w_ctpa": mw[0], "w_ecg": mw[1],
            "w_ehr": mw[2]})
    if oc in PUB3:
        print("  dissertation %.4f"
              "   reproduced %.4f"
              "   optimised %.4f  (%+.4f)"
              % (PUB3[oc],
                 np.mean(acc["3mod_pub"]),
                 np.mean(acc["3mod_opt"]),
                 np.mean(acc["3mod_opt"])
                 - PUB3[oc]))

# ============ MODEL 2: EHR + ECG =============
print("")
print("#" * 78)
print("MODEL 2: EHR + ECG, full ECG cohort")
print("#" * 78, flush=True)

for oc in OUTS:
    if oc not in FULL.columns:
        continue
    d, y = f.labels(FULL, oc)
    grp = d["subject_id"].values
    if y.sum() < 30:
        continue
    se, ysrc = f.labels(
        ins, "death_30d"
        if oc == "death_30d_inhosp" else oc)
    cb = BEST_C[oc]

    print("")
    print("=" * 78)
    print("%s   n=%d ev=%d   ECG C=%.0e"
          % (oc, len(y), int(y.sum()), cb),
          flush=True)

    # use the EHR columns actually present
    a1, b1 = f.prep(
        se[EH].values.astype(float),
        d[EH].values.astype(float))
    p_ehr = _rank(f.fit_lr(
        a1, ysrc, "ehr").predict_proba(
        b1)[:, 1])
    XL = d[LG].values.astype(float)
    XO = np.column_stack(
        [XL, d[MC].values.astype(float)])

    acc = {k: [] for k in
           ["ehr", "ecg_pub", "ecg_opt",
            "ehr+ecg_pub", "ehr+ecg_opt"]}
    wa = {"ehr+ecg_pub": [],
          "ehr+ecg_opt": []}
    keep2 = None
    for s in SEEDS:
        pp = _rank(oof(XL, y, grp, s, 1.0))
        po = _rank(oof(XO, y, grp, s, cb))
        fa, wfa = wcv([pp, p_ehr], y, grp, s)
        fb, wfb = wcv([po, p_ehr], y, grp, s)
        for k, v in (("ehr", p_ehr),
                     ("ecg_pub", pp),
                     ("ecg_opt", po),
                     ("ehr+ecg_pub", fa),
                     ("ehr+ecg_opt", fb)):
            acc[k].append(roc_auc_score(y, v))
        wa["ehr+ecg_pub"].append(wfa)
        wa["ehr+ecg_opt"].append(wfb)
        if s == SEEDS[0]:
            keep2 = (fa, fb, po, pp, p_ehr)

    print("  %-13s %8s %8s %8s"
          % ("model", "mean", "SD", "seed42"))
    for k in ["ehr", "ecg_pub", "ecg_opt",
              "ehr+ecg_pub", "ehr+ecg_opt"]:
        v = np.array(acc[k])
        print("  %-13s %.4f  %.4f  %.4f"
              % (k, v.mean(),
                 v.std(ddof=1), v[0]),
              flush=True)
        rows.append({
            "model_type": "ehr_ecg",
            "outcome": oc, "model": k,
            "n": len(y), "ev": int(y.sum()),
            "mean": v.mean(),
            "sd": v.std(ddof=1),
            "seed42": v[0]})

    fa, fb, po, pp, pe = keep2
    print("")
    print("  PAIRED BOOTSTRAP, seed 42")
    for nm, p, ref, rn in (
            ("ehr+ecg_pub", fa, pe, "ehr"),
            ("ehr+ecg_opt", fb, pe, "ehr"),
            ("ehr+ecg_opt", fb, fa,
             "ehr+ecg_pub"),
            ("ecg_opt", po, pp, "ecg_pub")):
        gn, lo, hi, _ = f.boot_diff(
            y, p, ref, grp)
        print("    %-12s vs %-12s %+.4f"
              " [%+.4f,%+.4f] %s"
              % (nm, rn, gn, lo, hi,
                 "*" if (lo > 0 or hi < 0)
                 else ""))
        rows.append({
            "model_type": "ehr_ecg",
            "outcome": oc,
            "model": nm + "_vs_" + rn,
            "n": len(y), "ev": int(y.sum()),
            "mean": gn, "sd": np.nan,
            "seed42": np.nan, "lo": lo,
            "hi": hi,
            "sig": int(lo > 0 or hi < 0)})

    print("")
    print("  WEIGHTS [ecg, ehr]")
    for k in ("ehr+ecg_pub", "ehr+ecg_opt"):
        mw = np.mean(wa[k], axis=0)
        print("    %-12s %s" % (
            k, np.round(mw, 2)))
        wrows.append({
            "model_type": "ehr_ecg",
            "outcome": oc, "model": k,
            "w_ctpa": np.nan,
            "w_ecg": mw[0], "w_ehr": mw[1]})

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)
pd.DataFrame(wrows).to_csv(
    os.path.join(
        PROC, "fusion_final_weights.csv"),
    index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 78)
print("SIX-SEED MEAN AUROC")
q = r[~r["model"].str.contains("_vs_")]
for mt, lab_ in (("3mod", "THREE-MODALITY"),
                 ("ehr_ecg", "EHR + ECG")):
    s = q[q["model_type"] == mt]
    if len(s):
        print("")
        print(lab_)
        print(s.pivot_table(index="model",
                            columns="outcome",
                            values="mean")
              .round(4).to_string())

print("")
print("SIX-SEED SD")
for mt, lab_ in (("3mod", "THREE-MODALITY"),
                 ("ehr_ecg", "EHR + ECG")):
    s = q[q["model_type"] == mt]
    if len(s):
        print("")
        print(lab_)
        print(s.pivot_table(index="model",
                            columns="outcome",
                            values="sd")
              .round(4).to_string())

print("")
print("PAIRED COMPARISONS, seed 42")
p = r[r["model"].str.contains("_vs_")]
print(p[["model_type", "outcome", "model",
         "mean", "lo", "hi", "sig"]]
      .round(4).to_string(index=False))
print("")
print("  significant cells: %d of %d"
      % (int(p["sig"].fillna(0).sum()),
         len(p)))

print("")
print("FUSION WEIGHTS")
print(pd.DataFrame(wrows).round(2)
      .to_string(index=False))

print("")
print("=" * 78)
print("AGAINST THE DISSERTATION")
print("")
print("  THREE-MODALITY")
print("  %-16s %10s %11s %10s %9s"
      % ("outcome", "published",
         "reproduced", "optimised",
         "change"))
for oc in OUTS:
    s = q[(q["model_type"] == "3mod")
          & (q["outcome"] == oc)]
    if not len(s) or oc not in PUB3:
        continue
    a = float(s[s["model"] == "3mod_pub"]
              ["mean"].iloc[0])
    b = float(s[s["model"] == "3mod_opt"]
              ["mean"].iloc[0])
    print("  %-16s %10.4f %11.4f %10.4f"
          " %+9.4f"
          % (oc, PUB3[oc], a, b,
             b - PUB3[oc]))

print("")
print("  ECG MODALITY ALONE, full cohort")
print("  %-16s %10s %10s %9s"
      % ("outcome", "published",
         "optimised", "change"))
for oc in OUTS:
    s = q[(q["model_type"] == "ehr_ecg")
          & (q["outcome"] == oc)]
    if not len(s):
        continue
    b = float(s[s["model"] == "ecg_opt"]
              ["mean"].iloc[0])
    print("  %-16s %10.4f %10.4f %+9.4f"
          % (oc, PUB_ECG[oc], b,
             b - PUB_ECG[oc]))

print("")
print("  cv_first ON THE CTPA COHORT"
      " (Table 23, n=1,555)")
s = q[(q["model_type"] == "3mod")
      & (q["outcome"] == "cv_first")]
if len(s) and "cv_first" in PUB_CT:
    pc = PUB_CT["cv_first"]
    for k, lb in (("ehr", "ehr"),
                  ("ecg_pub", "ecg"),
                  ("ctpa", "ctpa"),
                  ("ctpa+ehr", "ehr_ctpa")):
        v = s[s["model"] == k]
        if len(v) and lb in pc:
            print("  %-12s published %.4f"
                  "   here %.4f   %+.4f"
                  % (k, pc[lb],
                     v["mean"].iloc[0],
                     v["mean"].iloc[0]
                     - pc[lb]))
    print("  published EHR+ECG %.4f, BELOW"
          " EHR alone at %.4f  (%+.4f)"
          % (pc["ehr_ecg"], pc["ehr"],
             pc["ehr_ecg"] - pc["ehr"]))

print("")
print("DOES THE OPTIMISED ECG REVERSE THE"
      " EHR+ECG HARM?")
print("  %-18s %8s %10s %10s"
      % ("outcome", "ehr", "+published",
         "+optimised"))
for oc in OUTS:
    s = q[(q["model_type"] == "ehr_ecg")
          & (q["outcome"] == oc)]
    if not len(s):
        continue
    e = float(s[s["model"] == "ehr"]
              ["mean"].iloc[0])
    a = float(s[s["model"] == "ehr+ecg_pub"]
              ["mean"].iloc[0])
    b = float(s[s["model"] == "ehr+ecg_opt"]
              ["mean"].iloc[0])
    print("  %-18s %8.4f %10.4f %10.4f"
          % (oc, e, a, b))
    print("  %-18s %8s %+10.4f %+10.4f"
          % ("", "", a - e, b - e))

print("")
print("  a difference under about 0.008 sits"
      " inside the noise floor measured in")
print("  script 87 from two scores correlating"
      " at 0.977")
print("")
print("saved", DEST, r.shape)