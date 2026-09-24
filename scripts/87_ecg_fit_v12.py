"""Final fitting comparison, on v12 features.

Replaces script 82. Changes for v12:

  DROP LIST      diagnostic columns are excluded
                 explicitly. Script 82 left
                 t_win_ms, tp_win_ms, tpk_ms_*
                 and the hit-limit flags in the
                 feature matrix; t_win_ms in
                 particular is close to a proxy
                 for QT.
  Tp-e HANDLING  removed by prefix, not exact
                 name, so tpe_ms_strict,
                 tpe_qt_strict, qt_ms_strict and
                 qtc_strict cannot leak into the
                 "without Tp-e" set as they would
                 have in script 82.
  STRICT SETS    strict and permissive QT/Tp-e
                 tested as separate sets, since
                 v12 writes both.
  AGGREGATION    signed-worst and circular means
                 applied here, after the window,
                 as in script 82.

Feature sets:
  logits         71 SCP statement logits
  L+v7 / L+v8    the earlier versions, so the
                 progression is visible
  L+v12          the corrected measurements
  L+v12+tpe      plus Tp-e, permissive
  L+v12+strict   plus Tp-e, strict only
  L+shopp        the six flags
  L+daniel       the weighted score, logit RBBB
  L+daniel_wave  the weighted score, waveform
                 RBBB (they correlate 0.977, so
                 this should confirm it makes no
                 difference)
  L+dan+shopp    both

Five seeds, C swept over 13 values, class
weighting balanced.
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ecg_fit_v12.csv
  results\\ecg_fit_v12_log.txt
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
SEEDS = [42, 7, 13, 1, 2]
NFOLD = 5
WIN_LO, WIN_HI = -12.0, 48.0
CGRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4,
         3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
         1e-1, 3e-1, 1.0]
OUTS = ["cv_first", "death_30d_inhosp",
        "composite_30d", "death_30d"]
DEST = os.path.join(PROC, "ecg_fit_v12.csv")
PUB = {"composite_30d": 0.7207,
       "death_30d": 0.7229,
       "death_30d_inhosp": 0.7063,
       "cv_first": 0.6849}

# diagnostics, never features
DROP_EXACT = ("rec", "t_flag", "vm_noise",
              "vm_base", "vm_rpeak", "net_i",
              "net_avf", "apen",
              "tp_win_ms", "t_win_ms",
              "t_peak_ms", "tp_fallback",
              "qrs_hit_limit", "p_hit_limit",
              "t_peak_edge", "t_edge_leads",
              "n_rr_dropped", "any_af",
              "tpe_spread_ms", "tpe_ok")
DROP_PREFIX = ("tpk_ms_",)
# removed by prefix so the strict copies go too
TPE_PREFIX = ("tpe_ms", "tpe_qt",
              "qt_ms_strict", "qtc_strict")
STRICT_PREFIX = ("tpe_ms_strict",
                 "tpe_qt_strict",
                 "qt_ms_strict",
                 "qtc_strict")
SHOPP = ("shopp_hr100", "shopp_s1q3t3",
         "shopp_rbbb", "shopp_twi_v14",
         "shopp_ste_avr", "shopp_afib")
NEG_BAD = ("s_wave_i", "q_wave_iii", "st_min",
           "t_v1", "t_v2", "t_v3", "t_v4",
           "t_v5", "t_v6", "t_wave_iii")
ANG = ("qrs_axis", "t_axis", "p_axis")
MINCOV = 0.20


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def run(X, y, grp, seed, C):
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
        m = LogisticRegression(
            C=C, max_iter=8000,
            class_weight="balanced")
        m.fit(a, y[tr])
        p[te] = m.predict_proba(b)[:, 1]
    return roc_auc_score(y, p)


def sweep(X, y, grp):
    best = None
    for C in CGRID:
        v = np.array([run(X, y, grp, s, C)
                      for s in SEEDS])
        if best is None or v.mean() > best[1]:
            best = (C, v.mean(),
                    v.std(ddof=1))
    return best


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
R["t"] = pd.to_datetime(
    R["ecg_charttime"], errors="coerce")
R = R.merge(adm[["hadm_id", "admittime"]],
            on="hadm_id", how="left")
R["h"] = ((R["t"] - R["admittime"])
          .dt.total_seconds() / 3600.0)
R = R[(R["h"] >= WIN_LO)
      & (R["h"] <= WIN_HI)].copy()
A = R.groupby(["subject_id", "hadm_id"],
              as_index=False)[LG].mean()
keep = set(R["rec"])
print("windowed admissions:", len(A),
      " recordings:", len(R), flush=True)


def aggregate(j, cols, tag):
    g = j.groupby(["subject_id", "hadm_id"])
    pos = [c for c in cols
           if not c.startswith(NEG_BAD)
           and c not in ANG]
    neg = [c for c in cols
           if c.startswith(NEG_BAD)]
    parts = [g[cols].mean()
             .add_suffix("_mean")]
    if pos:
        parts.append(g[pos].max()
                     .add_suffix("_worst"))
    if neg:
        parts.append(g[neg].min()
                     .add_suffix("_worst"))
    ag = pd.concat(parts, axis=1).reset_index()
    for c in ANG:
        if c not in j.columns:
            continue
        rad = np.radians(j[c])
        tmp = j[["subject_id",
                 "hadm_id"]].copy()
        tmp["s"] = np.sin(rad)
        tmp["c"] = np.cos(rad)
        gg = tmp.groupby(["subject_id",
                          "hadm_id"]).mean()
        cm = np.degrees(np.arctan2(
            gg["s"], gg["c"])).rename(
            c + "_circ").reset_index()
        ag = ag.merge(cm,
                      on=["subject_id",
                          "hadm_id"],
                      how="left")
        ag = ag.drop(columns=[c + "_mean",
                              c + "_worst"],
                     errors="ignore")
    kc = [c for c in ag.columns
          if c not in ("subject_id",
                       "hadm_id")]
    cov = ag[kc].notna().mean()
    kc = [c for c in kc if cov[c] >= MINCOV]
    ag = ag.rename(columns={
        c: c + "__" + tag for c in kc})
    return (ag[["subject_id", "hadm_id"]
               + [c + "__" + tag for c in kc]],
            [c + "__" + tag for c in kc])


def load(fn, tag):
    p = os.path.join(PROC, fn)
    if not os.path.exists(p):
        print("  MISSING:", fn)
        return None, []
    m = pd.read_csv(p)
    m["rec"] = m["rec"].astype(str).apply(
        lambda s: s if s.startswith("files/")
        else "files/" + s.replace("\\", "/")
        .strip("/"))
    m = m[m["rec"].isin(keep)]
    cols = [c for c in m.columns
            if c not in DROP_EXACT
            and not c.startswith(DROP_PREFIX)
            and pd.api.types
            .is_numeric_dtype(m[c])]
    j = idx[["subject_id", "hadm_id",
             "rec"]].merge(
        m[["rec"] + cols], on="rec",
        how="inner")
    return aggregate(j, cols, tag)


V7, C7 = load("ecg_derived_v7_record.csv",
              "v7")
V8, C8 = load("ecg_derived_v8_record.csv",
              "v8")
V12, C12 = load(
    "ecg_derived_v12_record.csv", "v12")

# split v12 columns by prefix, not exact name
BASE12 = [c for c in C12
          if not c.startswith(TPE_PREFIX)]
TPE12 = [c for c in C12
         if c.startswith(("tpe_ms", "tpe_qt"))
         and not c.startswith(STRICT_PREFIX)]
STR12 = [c for c in C12
         if c.startswith(STRICT_PREFIX)]
SH12 = [c for c in C12
        if c.startswith(tuple(
            s + "_" for s in SHOPP))]
DAN = [c for c in C12
       if c.startswith(("daniel_score",
                        "daniel_ge",
                        "daniel_twi"))]
DANW = [c for c in C12
        if c.startswith("daniel_wave")]
print("v7 %d  v8 %d  v12 %d"
      % (len(C7), len(C8), len(C12)))
print("  base %d  tpe %d  strict %d"
      "  shopp %d  daniel %d  wave %d"
      % (len(BASE12), len(TPE12), len(STR12),
         len(SH12), len(DAN), len(DANW)),
      flush=True)
leak = [c for c in BASE12
        if c.startswith(STRICT_PREFIX)]
print("  strict columns leaking into base:",
      len(leak), "(script 82 would have had"
      " several)", flush=True)

lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
D = A.merge(lab, on=["subject_id", "hadm_id"],
            how="inner")
for fr in (V7, V8, V12):
    if fr is not None:
        D = D.merge(
            fr.drop(columns=["subject_id"]),
            on="hadm_id", how="left")
print("cohort:", len(D), flush=True)

nb = len([c for c in C12
          if any(c.startswith(x)
                 for x in NEG_BAD)
          and "_worst" in c])
na = len([c for c in C12 if "_circ" in c])
print("")
print("AGGREGATION: %d signed-worst,"
      " %d circular-mean" % (nb, na),
      flush=True)

res = []
t0 = time.time()

for oc in OUTS:
    d = D
    if oc == "cv_first":
        d = D[D["death_first"] == 0]
    y = pd.to_numeric(
        d[oc], errors="coerce").fillna(
        0).astype(int).values
    grp = d["subject_id"].values

    def blk(cols):
        return (d[cols].values.astype(float)
                if cols else None)
    XL = d[LG].values.astype(float)
    SETS = [("logits", XL)]
    for nm, cs in (("L+v7", C7),
                   ("L+v8", C8),
                   ("L+v12", BASE12),
                   ("L+shopp", SH12),
                   ("L+daniel", DAN),
                   ("L+daniel_wave", DANW)):
        if cs:
            SETS.append((nm, np.column_stack(
                [XL, blk(cs)])))
    if BASE12 and TPE12:
        SETS.append(("L+v12+tpe",
                     np.column_stack(
                         [XL, blk(BASE12),
                          blk(TPE12)])))
    if BASE12 and STR12:
        SETS.append(("L+v12+strict",
                     np.column_stack(
                         [XL, blk(BASE12),
                          blk(STR12)])))
    if DAN and SH12:
        SETS.append(("L+dan+shopp",
                     np.column_stack(
                         [XL, blk(DAN),
                          blk(SH12)])))

    print("")
    print("=" * 78)
    print("%s  n=%d ev=%d   published %.4f"
          % (oc, len(y), int(y.sum()),
             PUB[oc]), flush=True)
    print("  %-16s %5s %8s %8s %8s %s"
          % ("feature set", "nfeat", "peak C",
             "AUROC", "SD", "vs logits"))

    base = None
    for nm, X in SETS:
        C, m, s = sweep(X, y, grp)
        if nm == "logits":
            base = m
        edge = ("  EDGE"
                if C in (CGRID[0], CGRID[-1])
                else "")
        print("  %-16s %5d %8.0e %8.4f"
              " %8.4f  %+.4f%s"
              % (nm, X.shape[1], C, m, s,
                 m - base, edge), flush=True)
        res.append({
            "outcome": oc, "featset": nm,
            "nfeat": X.shape[1], "C": C,
            "auc": m, "sd": s,
            "vs_logits": m - base,
            "published": PUB[oc],
            "vs_published": m - PUB[oc],
            "at_edge": int(
                C in (CGRID[0], CGRID[-1]))})

    s_ = [x for x in res
          if x["outcome"] == oc]
    b = max(s_, key=lambda x: x["auc"])
    print("")
    print("  best: %s  %.4f  (%+.4f over"
          " logits, %+.4f over published)"
          % (b["featset"], b["auc"],
             b["vs_logits"],
             b["vs_published"]), flush=True)

r = pd.DataFrame(res)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 78)
print("PEAK AUROC BY FEATURE SET")
print(r.pivot_table(index="featset",
                    columns="outcome",
                    values="auc")
      .round(4).to_string())
print("")
print("GAIN OVER LOGITS ALONE")
print(r.pivot_table(index="featset",
                    columns="outcome",
                    values="vs_logits")
      .round(4).to_string())
print("")
print("GAIN OVER THE PUBLISHED FIGURE")
print(r.pivot_table(index="featset",
                    columns="outcome",
                    values="vs_published")
      .round(4).to_string())

print("")
print("v7 -> v8 -> v12 PROGRESSION")
p = r[r["featset"].isin(
    ["logits", "L+v7", "L+v8", "L+v12"])]
print(p.pivot_table(index="featset",
                    columns="outcome",
                    values="auc")
      .round(4).to_string())
for a_, b_ in (("L+v7", "L+v8"),
               ("L+v8", "L+v12")):
    x = r[r["featset"] == a_]
    z = r[r["featset"] == b_]
    if len(x) and len(z):
        dd = (z.set_index("outcome")["auc"]
              - x.set_index("outcome")["auc"])
        print("  %s -> %s   mean %+.4f"
              "   range %+.4f to %+.4f"
              % (a_, b_, dd.mean(),
                 dd.min(), dd.max()))
print("  v7->v8 was +0.0002 mean despite five"
      " genuine bug fixes; v8->v12 tests")
print("  whether twelve versions of landmark"
      " work changed prediction at all")

print("")
print("Tp-e: STRICT vs PERMISSIVE vs NONE")
for oc in OUTS:
    s = r[r["outcome"] == oc]
    a = s[s["featset"] == "L+v12"]
    b = s[s["featset"] == "L+v12+tpe"]
    c = s[s["featset"] == "L+v12+strict"]
    if len(a):
        line = "  %-18s none %.4f" % (
            oc, a["auc"].iloc[0])
        if len(b):
            line += "   perm %.4f (%+.4f)" % (
                b["auc"].iloc[0],
                b["auc"].iloc[0]
                - a["auc"].iloc[0])
        if len(c):
            line += "   strict %.4f (%+.4f)" % (
                c["auc"].iloc[0],
                c["auc"].iloc[0]
                - a["auc"].iloc[0])
        print(line)

print("")
print("DANIEL: LOGIT RBBB vs WAVEFORM RBBB")
print("  the two scores correlate 0.977, so"
      " any difference here is noise")
for oc in OUTS:
    s = r[r["outcome"] == oc]
    a = s[s["featset"] == "L+daniel"]
    b = s[s["featset"] == "L+daniel_wave"]
    if len(a) and len(b):
        print("  %-18s logit %.4f"
              "   waveform %.4f   %+.4f"
              % (oc, a["auc"].iloc[0],
                 b["auc"].iloc[0],
                 b["auc"].iloc[0]
                 - a["auc"].iloc[0]))

print("")
print("C AT THE PEAK")
print(r.pivot_table(index="featset",
                    columns="outcome",
                    values="C").to_string())
w = r.groupby("featset").agg(
    nfeat=("nfeat", "first"),
    medC=("C", "median"))
print("")
print("  columns vs optimal C:")
print(w.sort_values("nfeat").round(6)
      .to_string())
print("  cells at a grid edge: %d of %d"
      % (int(r["at_edge"].sum()), len(r)))

print("")
print("SEED SD")
print(r.pivot_table(index="featset",
                    columns="outcome",
                    values="sd")
      .round(4).to_string())

print("")
print("AVERAGE RANK ACROSS OUTCOMES")
rk = r.pivot_table(index="featset",
                   columns="outcome",
                   values="auc").rank(
    ascending=False).mean(axis=1)
print(rk.sort_values().round(2).to_string())

print("")
print("BEST PER OUTCOME")
for oc, s in r.groupby("outcome"):
    b = s.loc[s["auc"].idxmax()]
    print("  %-18s %-16s %.4f at C=%.0e"
          "  (published %.4f, %+.4f)"
          % (oc, b["featset"], b["auc"],
             b["C"], b["published"],
             b["vs_published"]))
print("")
print("saved", DEST, r.shape)