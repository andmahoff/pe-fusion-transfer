"""Train each modality on its full cohort, test
only on the three-modality intersection.

The dissertation fitted every modality strictly
within the 1,703 sub-cohort (Section 3.7:
"three-modality models were compared with other
models trained strictly within the same subgroup
rather than against the full dataset"). That was
a fair-comparison choice, but it discards
training data for no gain in validity: the
sub-cohort is nested inside the full cohort, so
the extra patients can train the model provided
no test subject appears in training.

Two regimes, identical test patients:

  WITHIN   each modality fitted inside the
           three-modality cohort, as published
  FULL     each modality fitted on its own full
           cohort, predictions taken for the
           test rows only

Grouping is on subject_id across the full cohort,
so a test patient cannot appear in training
through an admission that lacks a CTPA report.

Superset sizes:
  ECG    3,501 admissions with an ECG, against
         1,636 in the intersection
  CTPA   all admissions with a windowed report,
         which exceeds the intersection because
         that also requires an ECG
  EHR    zero-shot from INSPECT, so unaffected

Six seeds. Run in venv (analysis).

OUTPUT FILES
  data\\processed\\supercohort.csv
  data\\processed\\supercohort_weights.csv
  results\\supercohort_log.txt
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
                    "supercohort.csv")
BEST_C = {"cv_first": 1e-4,
          "composite_30d": 1e-3,
          "death_30d": 1e-3,
          "death_30d_inhosp": 3e-4}
OUTS = ["death_30d", "composite_30d",
        "cv_first", "death_30d_inhosp"]
PUB3 = {"death_30d": 0.8715,
        "composite_30d": 0.8389}

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


def oof_super(Xf, yf, gf, test_mask, seed, C,
              cw=None, tune=None):
    """Fit on the full cohort with grouped CV,
    return predictions for the test rows only.

    A test subject is never in training, because
    the folds are built over the full cohort and
    grouped on subject_id.
    """
    p = np.full(len(yf), np.nan)
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(Xf, yf, gf):
        im = SimpleImputer(strategy="median")
        a = im.fit_transform(Xf[tr])
        b = im.transform(Xf[te])
        sc = StandardScaler()
        a, b = sc.fit_transform(a), sc.transform(b)
        c = C
        if tune:
            best = -1.0
            k = min(3, max(2,
                           int(yf[tr].sum())
                           // 10))
            try:
                icv = StratifiedGroupKFold(
                    n_splits=k, shuffle=True,
                    random_state=42)
                for cand in tune:
                    qq = np.zeros(len(tr))
                    for t2, v2 in icv.split(
                            a, yf[tr], gf[tr]):
                        mm2 = LogisticRegression(
                            C=cand,
                            max_iter=8000,
                            class_weight=cw)
                        mm2.fit(a[t2],
                                yf[tr][t2])
                        qq[v2] = \
                            mm2.predict_proba(
                                a[v2])[:, 1]
                    s = roc_auc_score(
                        yf[tr], qq)
                    if s > best:
                        best, c = s, cand
            except Exception:
                c = C
        m = LogisticRegression(
            C=c, max_iter=8000,
            class_weight=cw)
        m.fit(a, yf[tr])
        p[te] = m.predict_proba(b)[:, 1]
    return p[test_mask]


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


# ---------- ECG features ----------
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

# ---------- cohorts ----------
lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
mim_ehr = f.load_mimic_ehr()
EH = [c for c in f.EHR_COLS
      if c in mim_ehr.columns]

# ECG superset
ECG_FULL = ECGA.merge(
    MA[["hadm_id"] + MC], on="hadm_id",
    how="left").merge(
    lab, on=["subject_id", "hadm_id"],
    how="inner").merge(
    mim_ehr[["hadm_id"] + EH],
    on="hadm_id", how="inner")

# CTPA superset: every windowed report, with no
# requirement for an ECG
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
CT_FULL = mm[mm["hadm_id"].isin(
    set(v2["hadm_id"]))].copy()
CT46 = [c for c in f.CTPA46
        if c in CT_FULL.columns]

# intersection: the test cohort
inter = (set(ECG_FULL["hadm_id"])
         & set(CT_FULL["hadm_id"]))
print("COHORTS")
print("  ECG superset :", len(ECG_FULL))
print("  CTPA superset:", len(CT_FULL))
print("  intersection :", len(inter),
      " (the dissertation used 1,703)")
print("  extra ECG training rows:",
      len(ECG_FULL) - len(inter))
print("  extra CTPA training rows:",
      len(CT_FULL) - len(inter))
print("  CTPA features:", len(CT46),
      flush=True)

ECG_FULL["in_test"] = ECG_FULL["hadm_id"] \
    .isin(inter).astype(int)
CT_FULL["in_test"] = CT_FULL["hadm_id"] \
    .isin(inter).astype(int)

ins = f.load_inspect()
rows, wrows = [], []
t0 = time.time()

for oc in OUTS:
    if oc not in ECG_FULL.columns:
        continue

    # full cohorts, with the test flag
    de, ye = f.labels(ECG_FULL, oc)
    ge = de["subject_id"].values
    me = de["in_test"].values == 1
    dc, yc = f.labels(CT_FULL, oc)
    gc = dc["subject_id"].values
    mc = dc["in_test"].values == 1

    if me.sum() < 100 or ye[me].sum() < 25:
        continue
    # align the two test subsets by hadm_id
    he = de.loc[me, "hadm_id"].values
    hc = dc.loc[mc, "hadm_id"].values
    common = np.intersect1d(he, hc)
    ie = np.isin(he, common)
    ic = np.isin(hc, common)
    order_e = np.argsort(he[ie])
    order_c = np.argsort(hc[ic])
    y = ye[me][ie][order_e]
    grp = ge[me][ie][order_e]

    se, ysrc = f.labels(
        ins, "death_30d"
        if oc == "death_30d_inhosp" else oc)
    cb = BEST_C[oc]

    print("")
    print("=" * 76)
    print("%s   test n=%d ev=%d"
          "   ECG train n=%d   CTPA train n=%d"
          % (oc, len(y), int(y.sum()),
             len(ye), len(yc)), flush=True)

    # EHR: zero-shot, so identical either way
    dt = de.loc[me].iloc[ie].iloc[order_e]
    a1, b1 = f.prep(
        se[f.EHR_COLS].values.astype(float),
        dt[f.EHR_COLS].values.astype(float))
    p_ehr = _rank(f.fit_lr(
        a1, ysrc, "ehr").predict_proba(
        b1)[:, 1])

    XEf = np.column_stack(
        [de[LG].values.astype(float),
         de[MC].values.astype(float)])
    XCf = dc[CT46].values.astype(float)
    # WITHIN: restrict to the test rows first
    XEw = XEf[me][ie][order_e]
    XCw = XCf[mc][ic][order_c]
    yw, gw = y, grp

    acc = {}
    for k in ("ecg_within", "ecg_full",
              "ctpa_within", "ctpa_full",
              "2mod_within", "2mod_full",
              "3mod_within", "3mod_full",
              "ehr"):
        acc[k] = []
    wa = {"3mod_within": [], "3mod_full": []}
    keep2 = None

    for s in SEEDS:
        # WITHIN: fitted inside the test cohort
        pew = _rank(oof_super(
            XEw, yw, gw,
            np.ones(len(yw), bool), s, cb,
            cw="balanced"))
        pcw = _rank(oof_super(
            XCw, yw, gw,
            np.ones(len(yw), bool), s, 1.0,
            tune=CT_CS))
        # FULL: fitted on the superset, taken
        # for the test rows only
        pef = oof_super(XEf, ye, ge, me, s, cb,
                        cw="balanced")
        pef = _rank(pef[ie][order_e])
        pcf = oof_super(XCf, yc, gc, mc, s, 1.0,
                        tune=CT_CS)
        pcf = _rank(pcf[ic][order_c])

        tw, _ = wcv([pcw, p_ehr], y, grp, s)
        tf, _ = wcv([pcf, p_ehr], y, grp, s)
        m3w, w3w = wcv([pcw, pew, p_ehr],
                       y, grp, s)
        m3f, w3f = wcv([pcf, pef, p_ehr],
                       y, grp, s)
        for k, v in (("ehr", p_ehr),
                     ("ecg_within", pew),
                     ("ecg_full", pef),
                     ("ctpa_within", pcw),
                     ("ctpa_full", pcf),
                     ("2mod_within", tw),
                     ("2mod_full", tf),
                     ("3mod_within", m3w),
                     ("3mod_full", m3f)):
            acc[k].append(roc_auc_score(y, v))
        wa["3mod_within"].append(w3w)
        wa["3mod_full"].append(w3f)
        if s == SEEDS[0]:
            keep2 = (tw, tf, m3w, m3f,
                     pew, pef, pcw, pcf)

    print("  %-13s %8s %8s   %s"
          % ("model", "mean", "SD",
             "full minus within"))
    for a_, b_ in (("ehr", None),
                   ("ecg_within", "ecg_full"),
                   ("ctpa_within", "ctpa_full"),
                   ("2mod_within", "2mod_full"),
                   ("3mod_within",
                    "3mod_full")):
        va = np.array(acc[a_])
        print("  %-13s %.4f  %.4f"
              % (a_, va.mean(),
                 va.std(ddof=1)))
        rows.append({
            "outcome": oc, "model": a_,
            "n": len(y), "ev": int(y.sum()),
            "mean": va.mean(),
            "sd": va.std(ddof=1)})
        if b_:
            vb = np.array(acc[b_])
            print("  %-13s %.4f  %.4f"
                  "   %+.4f"
                  % (b_, vb.mean(),
                     vb.std(ddof=1),
                     vb.mean() - va.mean()),
                  flush=True)
            rows.append({
                "outcome": oc, "model": b_,
                "n": len(y),
                "ev": int(y.sum()),
                "mean": vb.mean(),
                "sd": vb.std(ddof=1),
                "gain_vs_within":
                    vb.mean() - va.mean()})

    tw, tf, m3w, m3f, pew, pef, pcw, pcf = keep2
    print("")
    print("  PAIRED BOOTSTRAP, seed 42")
    for nm, p, ref, rn in (
            ("ecg_full", pef, pew,
             "ecg_within"),
            ("ctpa_full", pcf, pcw,
             "ctpa_within"),
            ("2mod_full", tf, tw,
             "2mod_within"),
            ("3mod_full", m3f, m3w,
             "3mod_within"),
            ("3mod_full", m3f, tf,
             "2mod_full")):
        gn, lo, hi, _ = f.boot_diff(
            y, p, ref, grp)
        print("    %-11s vs %-12s %+.4f"
              " [%+.4f,%+.4f] %s"
              % (nm, rn, gn, lo, hi,
                 "*" if (lo > 0 or hi < 0)
                 else ""))
        rows.append({
            "outcome": oc,
            "model": nm + "_vs_" + rn,
            "n": len(y), "ev": int(y.sum()),
            "mean": gn, "sd": np.nan,
            "lo": lo, "hi": hi,
            "sig": int(lo > 0 or hi < 0)})

    print("")
    print("  WEIGHTS [ctpa, ecg, ehr]")
    for k in ("3mod_within", "3mod_full"):
        mw = np.mean(wa[k], axis=0)
        print("    %-12s %s"
              % (k, np.round(mw, 2)))
        wrows.append({
            "outcome": oc, "model": k,
            "w_ctpa": mw[0], "w_ecg": mw[1],
            "w_ehr": mw[2]})
    if oc in PUB3:
        print("  dissertation %.4f"
              "   within %.4f   full %.4f"
              % (PUB3[oc],
                 np.mean(acc["3mod_within"]),
                 np.mean(acc["3mod_full"])))

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)
pd.DataFrame(wrows).to_csv(
    os.path.join(
        PROC, "supercohort_weights.csv"),
    index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 76)
print("SIX-SEED MEAN AUROC")
q = r[~r["model"].str.contains("_vs_")]
print(q.pivot_table(index="model",
                    columns="outcome",
                    values="mean")
      .round(4).to_string())

print("")
print("WHAT THE EXTRA TRAINING DATA BUYS")
gq = q[q["gain_vs_within"].notna()] \
    if "gain_vs_within" in q.columns \
    else pd.DataFrame()
if len(gq):
    print(gq.pivot_table(
        index="model", columns="outcome",
        values="gain_vs_within")
        .round(4).to_string())

print("")
print("PAIRED COMPARISONS, seed 42")
p = r[r["model"].str.contains("_vs_")]
print(p[["outcome", "model", "mean", "lo",
         "hi", "sig"]].round(4)
      .to_string(index=False))
print("")
print("  significant: %d of %d"
      % (int(p["sig"].fillna(0).sum()),
         len(p)))

print("")
print("FUSION WEIGHTS")
print(pd.DataFrame(wrows).round(2)
      .to_string(index=False))

print("")
print("AGAINST THE DISSERTATION")
print("  %-16s %10s %9s %9s"
      % ("outcome", "published",
         "within", "full"))
for oc in OUTS:
    s = q[q["outcome"] == oc]
    if not len(s) or oc not in PUB3:
        continue
    a = s[s["model"] == "3mod_within"]
    b = s[s["model"] == "3mod_full"]
    if len(a) and len(b):
        print("  %-16s %10.4f %9.4f %9.4f"
              % (oc, PUB3[oc],
                 a["mean"].iloc[0],
                 b["mean"].iloc[0]))
print("")
print("  the dissertation fitted every"
      " modality inside the sub-cohort")
print("  (Section 3.7); 'full' uses the"
      " nested superset for training only")
print("")
print("saved", DEST, r.shape)