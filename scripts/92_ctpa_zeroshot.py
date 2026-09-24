"""CTPA modality trained on INSPECT, tested on
MIMIC, and the fusion models built on it.

Three CTPA variants, so the feature-set effect and
the transfer cost are separated:

  mimic46    MIMIC-internal grouped CV on all 46
             Table 9 features. The dissertation
             protocol.
  mimic38    MIMIC-internal CV on the 38 shared
             features, dropping the 8
             support-device flags. A fair
             comparator for the zero-shot variant.
  zeroshot   fitted on INSPECT (n = 3,300) and
             applied to MIMIC with no target
             labels, on the same 38 features.

The 8 device flags are MIMIC-only: INSPECT
impressions do not mention support devices in a
comparable way, so a zero-shot model cannot use
them. Comparing zeroshot against mimic46 would
therefore confound transfer cost with feature
count; mimic38 is the correct baseline.

The dissertation reports the CTPA transfer cost
as 0.012 to 0.018 AUROC, against 0.047 to 0.095
for structured measurements, so a small gap is
expected.

EHR is zero-shot INSPECT throughout, as
published. ECG is the optimised v12 modality.
Six seeds.
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ctpa_zeroshot.csv
  data\\processed\\ctpa_zeroshot_weights.csv
  results\\ctpa_zeroshot_log.txt
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
# Table 43: ten log-spaced values, 1e-4 to 1e4
CT_CS = list(np.logspace(-4, 4, 10))
MINCOV = 0.20
DEST = os.path.join(PROC,
                    "ctpa_zeroshot.csv")
BEST_C = {"cv_first": 1e-4,
          "composite_30d": 1e-3,
          "death_30d": 1e-3,
          "death_30d_inhosp": 3e-4}
OUTS = ["death_30d", "composite_30d",
        "cv_first", "death_30d_inhosp"]
PUB3 = {"death_30d": 0.8715,
        "composite_30d": 0.8389}
PUB_CTPA = {"death_30d": 0.781,
            "composite_30d": 0.7370,
            "cv_first": 0.581}

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


def oof(X, y, grp, seed, C, cw=None,
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
                        mm2 = LogisticRegression(
                            C=cand,
                            max_iter=5000,
                            class_weight=cw)
                        mm2.fit(a[t2],
                                y[tr][t2])
                        qq[v2] = \
                            mm2.predict_proba(
                                a[v2])[:, 1]
                    s = roc_auc_score(
                        y[tr], qq)
                    if s > best:
                        best, c = s, cand
            except Exception:
                c = C
        m = LogisticRegression(
            C=c, max_iter=5000,
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


# ---------- ECG, optimised v12 ----------
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
CT = mm.merge(
    ECGA.drop(columns=["subject_id"]),
    on="hadm_id", how="inner").merge(
    MA[["hadm_id"] + MC], on="hadm_id",
    how="left")

ins = f.load_inspect()
CT46 = [c for c in f.CTPA46
        if c in CT.columns]
# the 38 shared with INSPECT
CT38 = [c for c in f.CTPA_COLS
        if c in CT.columns
        and c in ins.columns]
DEV = [c for c in CT46 if c not in CT38]
print("FEATURE SETS")
print("  MIMIC 46:", len(CT46))
print("  shared with INSPECT:", len(CT38))
print("  MIMIC-only (devices):", len(DEV))
print("   ", DEV)
print("")
print("  INSPECT n:", len(ins))
print("  CTPA cohort:", len(CT),
      " (published 1,703)", flush=True)

rows, wrows = [], []
t0 = time.time()

for oc in OUTS:
    if oc not in CT.columns:
        continue
    d, y = f.labels(CT, oc)
    grp = d["subject_id"].values
    if y.sum() < 25:
        continue
    src = ("death_30d"
           if oc == "death_30d_inhosp" else oc)
    se, ysrc = f.labels(ins, src)
    cb = BEST_C[oc]

    print("")
    print("=" * 78)
    print("%s   n=%d ev=%d"
          "   INSPECT n=%d ev=%d"
          % (oc, len(y), int(y.sum()),
             len(ysrc), int(ysrc.sum())),
          flush=True)

    # EHR, zero-shot as published
    a1, b1 = f.prep(
        se[f.EHR_COLS].values.astype(float),
        d[f.EHR_COLS].values.astype(float))
    p_ehr = _rank(f.fit_lr(
        a1, ysrc, "ehr").predict_proba(
        b1)[:, 1])

    # CTPA zero-shot: fitted on INSPECT once,
    # so it has no fold or seed variation
    a2, b2 = f.prep(
        se[CT38].values.astype(float),
        d[CT38].values.astype(float))
    p_zs = _rank(f.fit_lr(
        a2, ysrc, "ctpa").predict_proba(
        b2)[:, 1])

    X46 = d[CT46].values.astype(float)
    X38 = d[CT38].values.astype(float)
    XL = d[LG].values.astype(float)
    XO = np.column_stack(
        [XL, d[MC].values.astype(float)])

    acc = {k: [] for k in
           ["ehr", "ecg_opt",
            "ctpa_mimic46", "ctpa_mimic38",
            "ctpa_zeroshot",
            "2mod_mimic46", "2mod_mimic38",
            "2mod_zeroshot",
            "3mod_mimic46", "3mod_zeroshot"]}
    wa = {"3mod_mimic46": [],
          "3mod_zeroshot": []}
    keep2 = None
    for s in SEEDS:
        p46 = _rank(oof(X46, y, grp, s, 1.0,
                        tune=CT_CS))
        p38 = _rank(oof(X38, y, grp, s, 1.0,
                        tune=CT_CS))
        po = _rank(oof(XO, y, grp, s, cb,
                       cw="balanced"))
        t46, _ = wcv([p46, p_ehr], y, grp, s)
        t38, _ = wcv([p38, p_ehr], y, grp, s)
        tzs, _ = wcv([p_zs, p_ehr], y, grp, s)
        m46, w46 = wcv([p46, po, p_ehr],
                       y, grp, s)
        mzs, wzs = wcv([p_zs, po, p_ehr],
                       y, grp, s)
        for k, v in (("ehr", p_ehr),
                     ("ecg_opt", po),
                     ("ctpa_mimic46", p46),
                     ("ctpa_mimic38", p38),
                     ("ctpa_zeroshot", p_zs),
                     ("2mod_mimic46", t46),
                     ("2mod_mimic38", t38),
                     ("2mod_zeroshot", tzs),
                     ("3mod_mimic46", m46),
                     ("3mod_zeroshot", mzs)):
            acc[k].append(roc_auc_score(y, v))
        wa["3mod_mimic46"].append(w46)
        wa["3mod_zeroshot"].append(wzs)
        if s == SEEDS[0]:
            keep2 = (p46, p38, t46, t38, tzs,
                     m46, mzs)

    print("  %-16s %8s %8s   %s"
          % ("model", "mean", "SD",
             "published"))
    PUBMAP = {"ctpa_mimic46":
              PUB_CTPA.get(oc),
              "3mod_mimic46": PUB3.get(oc)}
    for k in ["ehr", "ecg_opt",
              "ctpa_mimic46", "ctpa_mimic38",
              "ctpa_zeroshot",
              "2mod_mimic46", "2mod_mimic38",
              "2mod_zeroshot",
              "3mod_mimic46",
              "3mod_zeroshot"]:
        v = np.array(acc[k])
        pb = PUBMAP.get(k)
        pt = ("  %.4f (%+.4f)"
              % (pb, v.mean() - pb)
              if pb else "")
        print("  %-16s %.4f  %.4f%s"
              % (k, v.mean(),
                 v.std(ddof=1), pt),
              flush=True)
        rows.append({
            "outcome": oc, "model": k,
            "n": len(y), "ev": int(y.sum()),
            "mean": v.mean(),
            "sd": v.std(ddof=1),
            "published": pb})

    # transfer cost, matched on features
    tc = (np.mean(acc["ctpa_mimic38"])
          - np.mean(acc["ctpa_zeroshot"]))
    fc = (np.mean(acc["ctpa_mimic46"])
          - np.mean(acc["ctpa_mimic38"]))
    print("")
    print("  DECOMPOSITION")
    print("    device flags worth: %+.4f"
          " (46 minus 38, both MIMIC)" % fc)
    print("    transfer cost:      %+.4f"
          " (mimic38 minus zeroshot)" % tc)
    print("    dissertation reports the CTPA"
          " transfer cost as 0.012 to 0.018")
    rows.append({
        "outcome": oc,
        "model": "TRANSFER_COST",
        "n": len(y), "ev": int(y.sum()),
        "mean": tc, "sd": np.nan,
        "published": np.nan})
    rows.append({
        "outcome": oc,
        "model": "DEVICE_FLAGS",
        "n": len(y), "ev": int(y.sum()),
        "mean": fc, "sd": np.nan,
        "published": np.nan})

    p46, p38, t46, t38, tzs, m46, mzs = keep2
    print("")
    print("  PAIRED BOOTSTRAP, seed 42")
    for nm, p, ref, rn in (
            ("ctpa_zeroshot", p_zs, p38,
             "ctpa_mimic38"),
            ("2mod_zeroshot", tzs, p_ehr,
             "ehr"),
            ("2mod_zeroshot", tzs, t46,
             "2mod_mimic46"),
            ("3mod_zeroshot", mzs, tzs,
             "2mod_zeroshot"),
            ("3mod_zeroshot", mzs, m46,
             "3mod_mimic46")):
        gn, lo, hi, _ = f.boot_diff(
            y, p, ref, grp)
        print("    %-14s vs %-14s %+.4f"
              " [%+.4f,%+.4f] %s"
              % (nm, rn, gn, lo, hi,
                 "*" if (lo > 0 or hi < 0)
                 else ""))
        rows.append({
            "outcome": oc,
            "model": nm + "_vs_" + rn,
            "n": len(y), "ev": int(y.sum()),
            "mean": gn, "sd": np.nan,
            "published": np.nan,
            "lo": lo, "hi": hi,
            "sig": int(lo > 0 or hi < 0)})

    print("")
    print("  WEIGHTS [ctpa, ecg, ehr]")
    for k in ("3mod_mimic46", "3mod_zeroshot"):
        mw = np.mean(wa[k], axis=0)
        print("    %-14s %s"
              % (k, np.round(mw, 2)))
        wrows.append({
            "outcome": oc, "model": k,
            "w_ctpa": mw[0], "w_ecg": mw[1],
            "w_ehr": mw[2]})

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)
pd.DataFrame(wrows).to_csv(
    os.path.join(
        PROC,
        "ctpa_zeroshot_weights.csv"),
    index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 78)
print("SIX-SEED MEAN AUROC")
q = r[~r["model"].str.contains("_vs_|_COST|"
                               "_FLAGS")]
print(q.pivot_table(index="model",
                    columns="outcome",
                    values="mean")
      .round(4).to_string())

print("")
print("ZERO-SHOT TRANSFER COST")
print("  mimic38 minus zeroshot, so the"
      " feature sets are matched")
tc = r[r["model"] == "TRANSFER_COST"]
print(tc[["outcome", "mean"]].round(4)
      .to_string(index=False))
if len(tc):
    print("  mean %.4f   dissertation reports"
          " 0.012 to 0.018 for CTPA text"
          % tc["mean"].mean())

print("")
print("WHAT THE 8 DEVICE FLAGS ADD")
fc = r[r["model"] == "DEVICE_FLAGS"]
print(fc[["outcome", "mean"]].round(4)
      .to_string(index=False))
print("  the dissertation reports support"
      " devices adding +0.072 on in-hospital")
print("  death and little elsewhere")

print("")
print("FUSION: MIMIC-INTERNAL vs ZERO-SHOT")
print("  %-18s %10s %10s %9s"
      % ("outcome", "mimic46",
         "zeroshot", "cost"))
for oc in OUTS:
    s = q[q["outcome"] == oc]
    for pre in ("2mod", "3mod"):
        a = s[s["model"] == pre + "_mimic46"]
        b = s[s["model"] == pre + "_zeroshot"]
        if len(a) and len(b):
            print("  %-18s %10.4f %10.4f"
                  " %+9.4f   (%s)"
                  % (oc, a["mean"].iloc[0],
                     b["mean"].iloc[0],
                     b["mean"].iloc[0]
                     - a["mean"].iloc[0],
                     pre))

print("")
print("PAIRED COMPARISONS")
p = r[r["model"].str.contains("_vs_")]
print(p[["outcome", "model", "mean", "lo",
         "hi", "sig"]].round(4)
      .to_string(index=False))

print("")
print("FUSION WEIGHTS")
print(pd.DataFrame(wrows).round(2)
      .to_string(index=False))
print("")
print("saved", DEST, r.shape)