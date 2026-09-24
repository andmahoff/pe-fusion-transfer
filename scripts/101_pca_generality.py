"""Tests whether the PCA advantage on EHR+CTPA
for in-hospital death comes from the low event
count or from the kind of features compressed.

On death_30d_inhosp with EHR+CTPA, PCA to 8-10
components reached 0.9002 against late fusion's
0.8823 over ten seeds (script 100), and lost on
the other three outcomes. On EHR+ECG, with fewer
events per feature, PCA lost (script 99), which a
pure event-count explanation would not predict.

THREE TESTS
  1 every feature set on this outcome: each
    modality alone, each pairing, and all three.
    If compression helps only where ECG is
    absent, feature type is the explanation.
  2 subsampling: death_30d (152 events, where PCA
    loses) thinned to 120, 100 and 83 events on
    identical features. If the PCA advantage
    appears as events fall, event count is
    causal.
  3 the winning k range on all four outcomes.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\pca_generality.csv
  results\\pca_generality_log.txt
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats as st

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
RAW = os.path.join("data", "raw", "ecg")
SEEDS = [42, 7, 13, 1, 2, 3, 5, 8, 21, 99]
SUB_SEEDS = [42, 7, 13, 1, 2]
NFOLD = 5
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
KS = [5, 8, 10, 15, 20, 30]
KS_WIN = [8, 10]
SUB_EV = [152, 130, 110, 95, 83]
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
MINCOV = 0.20
TARGET = "death_30d_inhosp"
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(PROC,
                    "pca_generality.csv")

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


def clean_fit(A):
    A = np.asarray(A, dtype=float).copy()
    med = np.nanmedian(A, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(A)
    if bad.any():
        ii = np.where(bad)
        A[ii] = np.take(med, ii[1])
    return A, med


def clean_apply(A, med):
    A = np.asarray(A, dtype=float).copy()
    bad = ~np.isfinite(A)
    if bad.any():
        ii = np.where(bad)
        A[ii] = np.take(med, ii[1])
    return A


def fit_pred(Xtr, ytr, Xte, gtr=None):
    im = SimpleImputer(strategy="median")
    a = im.fit_transform(Xtr)
    b = im.transform(Xte)
    sc = StandardScaler()
    a, b = sc.fit_transform(a), sc.transform(b)
    best, bc = -1.0, 1.0
    try:
        kk = min(3, max(2, int(ytr.sum()) // 10))
        icv = StratifiedGroupKFold(
            n_splits=kk, shuffle=True,
            random_state=42)
        g = (gtr if gtr is not None
             else np.arange(len(ytr)))
        for c in CS:
            q = np.zeros(len(ytr))
            for t2, v2 in icv.split(a, ytr, g):
                m = LogisticRegression(
                    C=c, max_iter=3000)
                m.fit(a[t2], ytr[t2])
                q[v2] = m.predict_proba(
                    a[v2])[:, 1]
            s = roc_auc_score(ytr, q)
            if s > best:
                best, bc = s, c
    except Exception:
        bc = 1.0
    m = LogisticRegression(C=bc, max_iter=3000)
    m.fit(a, ytr)
    return m.predict_proba(b)[:, 1]


def oof_raw(X, y, grp, seed):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        Xtr, md = clean_fit(X[tr])
        Xte = clean_apply(X[te], md)
        p[te] = fit_pred(Xtr, y[tr], Xte,
                         grp[tr])
    return p


def oof_pca(X, y, grp, k, seed):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        Xtr, md = clean_fit(X[tr])
        Xte = clean_apply(X[te], md)
        sc = StandardScaler()
        pc = PCA(n_components=min(
            k, Xtr.shape[1] - 1,
            len(tr) - 1), random_state=42)
        ztr = pc.fit_transform(
            sc.fit_transform(Xtr))
        zte = pc.transform(sc.transform(Xte))
        p[te] = fit_pred(ztr, y[tr], zte,
                         grp[tr])
    return p


def oof_late(blocks, y, grp, seed):
    """Weighted rank averaging over however many
    blocks are supplied."""
    ps = []
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for X in blocks:
        q = np.zeros(len(y))
        for tr, te in cv.split(
                np.zeros((len(y), 1)), y, grp):
            Xtr, md = clean_fit(X[tr])
            Xte = clean_apply(X[te], md)
            q[te] = fit_pred(Xtr, y[tr], Xte,
                             grp[tr])
        ps.append(_rank(q))
    if len(ps) == 1:
        return ps[0]
    if len(ps) == 2:
        bs, bv = -1.0, ps[0]
        for w in np.arange(0, 1.001, 0.05):
            v = w * ps[1] + (1 - w) * ps[0]
            a = roc_auc_score(y, v)
            if a > bs:
                bs, bv = a, v
        return bv
    bs, bv = -1.0, ps[0]
    for w1 in np.arange(0, 1.001, 0.05):
        for w2 in np.arange(0, 1.001 - w1,
                            0.05):
            w3 = 1.0 - w1 - w2
            v = (w1 * ps[0] + w2 * ps[1]
                 + w3 * ps[2])
            a = roc_auc_score(y, v)
            if a > bs:
                bs, bv = a, v
    return bv


# ---------------- cohort --------------------
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

lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
mim_ehr = f.load_mimic_ehr()
EH = [c for c in f.EHR_COLS
      if c in mim_ehr.columns]
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
CT = [c for c in f.CTPA46 if c in mm.columns]

D = ECGA.merge(
    MA[["hadm_id"] + MC], on="hadm_id",
    how="left").merge(
    lab, on=["subject_id", "hadm_id"],
    how="inner").merge(
    mim_ehr[["hadm_id"] + EH],
    on="hadm_id", how="inner").merge(
    mm[["hadm_id"] + CT], on="hadm_id",
    how="inner")
ECGC = LG + MC
BLK = {"ehr": EH, "ecg": ECGC, "ctpa": CT}
SETS = [("ehr", ["ehr"]),
        ("ctpa", ["ctpa"]),
        ("ecg", ["ecg"]),
        ("ehr+ctpa", ["ehr", "ctpa"]),
        ("ehr+ecg", ["ehr", "ecg"]),
        ("ecg+ctpa", ["ecg", "ctpa"]),
        ("all3", ["ehr", "ecg", "ctpa"])]

print("cohort:", len(D))
print("  EHR %d   ECG %d   CTPA %d"
      % (len(EH), len(ECGC), len(CT)),
      flush=True)

rows = []
t0 = time.time()

# =========== TEST 1: every feature set =======
print("")
print("#" * 74)
print("TEST 1: EVERY FEATURE SET ON %s"
      % TARGET)
print("  if compression helps only where ECG"
      " is absent, feature type is the")
print("  explanation rather than event count")
print("#" * 74, flush=True)

d, y = f.labels(D, TARGET)
grp = d["subject_id"].values
nev = int(y.sum())
print("  n=%d  events=%d" % (len(y), nev),
      flush=True)

for nm, blks in SETS:
    Xs = [d[BLK[b]].values.astype(float)
          for b in blks]
    Xc = np.column_stack(Xs)
    nf = Xc.shape[1]
    epf = nev / nf
    has_ecg = "ecg" in blks
    print("")
    print("  %-10s %4d features"
          "   %.2f events per feature"
          "   ECG %s"
          % (nm, nf, epf,
             "yes" if has_ecg else "no"),
          flush=True)

    ra = np.array([roc_auc_score(
        y, oof_raw(Xc, y, grp, s))
        for s in SEEDS])
    la = (ra if len(blks) == 1
          else np.array([roc_auc_score(
              y, oof_late(Xs, y, grp, s))
              for s in SEEDS]))
    ref = la.mean()
    print("    %-10s %8.4f (SD %.4f)"
          % ("raw", ra.mean(),
             ra.std(ddof=1)))
    if len(blks) > 1:
        print("    %-10s %8.4f (SD %.4f)"
              % ("late", la.mean(),
                 la.std(ddof=1)), flush=True)
    for mn, v in (("raw", ra), ("late", la)):
        rows.append({
            "test": "sets", "featset": nm,
            "outcome": TARGET, "method": mn,
            "k": np.nan, "nfeat": nf,
            "events": nev, "ev_per_feat": epf,
            "has_ecg": int(has_ecg),
            "mean": v.mean(),
            "sd": v.std(ddof=1)})

    bk, bm = None, -1.0
    for k in KS:
        if k >= nf:
            continue
        aa = np.array([roc_auc_score(
            y, oof_pca(Xc, y, grp, k, s))
            for s in SEEDS])
        print("    %-10s %8.4f (SD %.4f)"
              "   %+.4f vs ref"
              % ("pca%d" % k, aa.mean(),
                 aa.std(ddof=1),
                 aa.mean() - ref), flush=True)
        rows.append({
            "test": "sets", "featset": nm,
            "outcome": TARGET,
            "method": "pca%d" % k, "k": k,
            "nfeat": nf, "events": nev,
            "ev_per_feat": epf,
            "has_ecg": int(has_ecg),
            "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_ref": aa.mean() - ref})
        if aa.mean() > bm:
            bk, bm = k, aa.mean()
    if bk is not None:
        print("    best k=%d  %.4f  %+.4f"
              " vs reference"
              % (bk, bm, bm - ref), flush=True)
        rows.append({
            "test": "best", "featset": nm,
            "outcome": TARGET,
            "method": "pca_best", "k": bk,
            "nfeat": nf, "events": nev,
            "ev_per_feat": epf,
            "has_ecg": int(has_ecg),
            "mean": bm, "sd": np.nan,
            "vs_ref": bm - ref})

# =========== TEST 2: subsampling =============
print("")
print("#" * 74)
print("TEST 2: SUBSAMPLING, THE CAUSAL TEST")
print("  death_30d on EHR+CTPA has 152 events"
      " and PCA loses by 0.0035.")
print("  Thinning to 83 events on identical"
      " features isolates event count")
print("  from everything else. If the advantage"
      " appears, event count is")
print("  causal; if it does not, it is not.")
print("#" * 74, flush=True)

d2, y2 = f.labels(D, "death_30d")
grp2 = d2["subject_id"].values
Xa = d2[EH].values.astype(float)
Xb = d2[CT].values.astype(float)
Xc2 = np.column_stack([Xa, Xb])
print("  full: n=%d events=%d"
      % (len(y2), int(y2.sum())), flush=True)

print("")
print("  %-8s %-10s %8s %8s %8s"
      % ("events", "method", "mean", "SD",
         "vs late"))
for ne in SUB_EV:
    if ne > int(y2.sum()):
        continue
    accs = {"late": [], "pca8": [],
            "pca10": [], "raw": []}
    for s in SUB_SEEDS:
        rng = np.random.default_rng(1000 + s)
        ev_idx = np.where(y2 == 1)[0]
        drop = int(y2.sum()) - ne
        if drop > 0:
            rm = rng.choice(ev_idx, drop,
                            replace=False)
            m_ = np.ones(len(y2), bool)
            m_[rm] = False
        else:
            m_ = np.ones(len(y2), bool)
        ys, gs = y2[m_], grp2[m_]
        Xs_ = Xc2[m_]
        As, Bs = Xa[m_], Xb[m_]
        accs["late"].append(roc_auc_score(
            ys, oof_late([As, Bs], ys, gs, s)))
        accs["raw"].append(roc_auc_score(
            ys, oof_raw(Xs_, ys, gs, s)))
        for k in KS_WIN:
            accs["pca%d" % k].append(
                roc_auc_score(
                    ys, oof_pca(Xs_, ys, gs,
                                k, s)))
    lm = float(np.mean(accs["late"]))
    for mn in ("late", "raw", "pca8",
               "pca10"):
        v = np.array(accs[mn])
        print("  %-8d %-10s %8.4f %8.4f"
              " %+8.4f"
              % (ne, mn, v.mean(),
                 v.std(ddof=1),
                 v.mean() - lm), flush=True)
        rows.append({
            "test": "subsample",
            "featset": "ehr+ctpa",
            "outcome": "death_30d",
            "method": mn,
            "k": (8 if mn == "pca8"
                  else 10 if mn == "pca10"
                  else np.nan),
            "nfeat": Xc2.shape[1],
            "events": ne,
            "ev_per_feat": ne / Xc2.shape[1],
            "has_ecg": 0, "mean": v.mean(),
            "sd": v.std(ddof=1),
            "vs_ref": v.mean() - lm})

# =========== TEST 3: all outcomes ============
print("")
print("#" * 74)
print("TEST 3: THE WINNING k RANGE ON EVERY"
      " OUTCOME")
print("#" * 74, flush=True)

for oc in OUTS:
    if oc not in D.columns:
        continue
    d3, y3 = f.labels(D, oc)
    g3 = d3["subject_id"].values
    if y3.sum() < 25:
        continue
    print("")
    print("  %s  ev=%d" % (oc, int(y3.sum())),
          flush=True)
    for nm, blks in SETS:
        if len(blks) == 1:
            continue
        Xs = [d3[BLK[b]].values.astype(float)
              for b in blks]
        Xc3 = np.column_stack(Xs)
        la = np.array([roc_auc_score(
            y3, oof_late(Xs, y3, g3, s))
            for s in SEEDS])
        best = None
        for k in KS_WIN:
            aa = np.array([roc_auc_score(
                y3, oof_pca(Xc3, y3, g3, k, s))
                for s in SEEDS])
            rows.append({
                "test": "outcomes",
                "featset": nm, "outcome": oc,
                "method": "pca%d" % k, "k": k,
                "nfeat": Xc3.shape[1],
                "events": int(y3.sum()),
                "ev_per_feat":
                    int(y3.sum())
                    / Xc3.shape[1],
                "has_ecg": int("ecg" in blks),
                "mean": aa.mean(),
                "sd": aa.std(ddof=1),
                "vs_ref": aa.mean()
                - la.mean()})
            if best is None or \
                    aa.mean() > best[1]:
                best = (k, aa.mean())
        rows.append({
            "test": "outcomes",
            "featset": nm, "outcome": oc,
            "method": "late", "k": np.nan,
            "nfeat": Xc3.shape[1],
            "events": int(y3.sum()),
            "ev_per_feat": int(y3.sum())
            / Xc3.shape[1],
            "has_ecg": int("ecg" in blks),
            "mean": la.mean(),
            "sd": la.std(ddof=1)})
        print("    %-10s late %.4f"
              "   best pca%d %.4f   %+.4f"
              % (nm, la.mean(), best[0],
                 best[1], best[1] - la.mean()),
              flush=True)
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("TEST 1: PCA ADVANTAGE BY FEATURE SET")
b = r[r["test"] == "best"]
if len(b):
    print("  %-10s %6s %8s %8s %6s %9s"
          % ("set", "nfeat", "ev/feat",
             "best k", "ECG", "vs ref"))
    for _, x in b.sort_values(
            "ev_per_feat").iterrows():
        print("  %-10s %6d %8.2f %8d %6s"
              " %+9.4f"
              % (x["featset"], x["nfeat"],
                 x["ev_per_feat"], x["k"],
                 "yes" if x["has_ecg"]
                 else "no", x["vs_ref"]))
    print("")
    wi = b[b["has_ecg"] == 1]["vs_ref"]
    wo = b[b["has_ecg"] == 0]["vs_ref"]
    print("  mean advantage WITH ECG:    %+.4f"
          % wi.mean())
    print("  mean advantage WITHOUT ECG: %+.4f"
          % wo.mean())
    if len(b) > 3:
        x = b["ev_per_feat"].values
        yv = b["vs_ref"].values
        ok = np.isfinite(x) & np.isfinite(yv)
        if ok.sum() > 3 and np.std(x[ok]) > 0:
            pr, pp = st.pearsonr(x[ok], yv[ok])
            sr, sp = st.spearmanr(x[ok],
                                  yv[ok])
            print("")
            print("  advantage vs events per"
                  " feature:")
            print("    Pearson %+.3f (p=%.3f)"
                  "   Spearman %+.3f (p=%.3f)"
                  % (pr, pp, sr, sp))

print("")
print("TEST 2: DOES THE ADVANTAGE APPEAR AS"
      " EVENTS FALL?")
s = r[r["test"] == "subsample"]
if len(s):
    print(s.pivot_table(index="events",
                        columns="method",
                        values="mean")
          .round(4).to_string())
    print("")
    print("  pca minus late, by event count")
    for ne in sorted(s["events"].unique()):
        z = s[s["events"] == ne]
        lm = z[z["method"] == "late"]
        for mn in ("pca8", "pca10"):
            q = z[z["method"] == mn]
            if len(q) and len(lm):
                print("    %-4d events   %-6s"
                      " %+.4f"
                      % (ne, mn,
                         q["mean"].iloc[0]
                         - lm["mean"].iloc[0]))
    print("")
    print("  a monotone rise as events fall"
          " means event count is causal")

print("")
print("TEST 3: EVERY OUTCOME")
o = r[r["test"] == "outcomes"]
if len(o):
    pv = o[o["method"] != "late"].copy()
    lt = o[o["method"] == "late"][
        ["featset", "outcome", "mean"]].rename(
        columns={"mean": "late"})
    mg = pv.merge(lt,
                  on=["featset", "outcome"])
    mg["adv"] = mg["mean"] - mg["late"]
    bb = mg.loc[mg.groupby(
        ["featset", "outcome"])["mean"].idxmax()]
    print(bb.pivot_table(index="featset",
                         columns="outcome",
                         values="adv")
          .round(4).to_string())
    print("")
    print("  positive means PCA beats late"
          " fusion in that cell")
    npos = int((bb["adv"] > 0).sum())
    print("  cells where PCA wins: %d of %d"
          % (npos, len(bb)))

print("")
print("saved", DEST, r.shape)