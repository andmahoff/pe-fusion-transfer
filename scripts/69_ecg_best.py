"""Full adjustment grid, five seeds, to build
the best ECG modality per outcome.

Grid, eight cells:
  class weight   balanced | none
  C              1.0 fixed | tuned in fold
  trajectory     absent | present

Selection uses the five-seed mean, matching the
dissertation's multi-seed protocol (Table 20b
reports five to six seeds). The full grid is
reported, not only the winner.

Baseline is the reproduced published model:
balanced weighting, C = 1.0, no trajectory,
windowed -12h to +48h cohort.
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ecg_best_grid.csv
  data\\processed\\p_ecg_best.csv
  results\\ecg_best_log.txt
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
SEEDS = [42, 7, 13, 1, 2]
NFOLD = 5
CS = [0.0003, 0.001, 0.003, 0.01, 0.03,
      0.1, 0.3, 1.0]
WIN_LO, WIN_HI = -12.0, 48.0
OUTS = ["composite_30d", "death_30d",
        "death_30d_inhosp", "cv_first"]
DEST = os.path.join(
    PROC, "ecg_best_grid.csv")
PUB = {"composite_30d": 0.7207,
       "death_30d": 0.7229,
       "death_30d_inhosp": 0.7063,
       "cv_first": 0.6849}

# label, class_weight, tune C, trajectory
GRID = [
    ("published", "balanced", False, False),
    ("cw_none", None, False, False),
    ("C_bal", "balanced", True, False),
    ("C_none", None, True, False),
    ("traj_bal", "balanced", False, True),
    ("traj_none", None, False, True),
    ("C_traj_bal", "balanced", True, True),
    ("C_traj_none", None, True, True),
]


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def fold(Xa, ya, ga, Xb, cw, tune):
    im = SimpleImputer(strategy="median")
    a = im.fit_transform(Xa)
    b = im.transform(Xb)
    sc = StandardScaler()
    a, b = sc.fit_transform(a), sc.transform(b)
    c = 1.0
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
            c = 1.0
    m = LogisticRegression(
        C=c, max_iter=5000, class_weight=cw)
    m.fit(a, ya)
    return m.predict_proba(b)[:, 1], c


def oof(X, y, grp, seed, cw, tune):
    p = np.zeros(len(y))
    cs = []
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(X, y, grp):
        p[te], c = fold(X[tr], y[tr],
                        grp[tr], X[te],
                        cw, tune)
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
print("windowed admissions:", len(A),
      flush=True)

Rs = R.dropna(subset=["t"]).sort_values(
    ["hadm_id", "t"])
rows = []
for h, g in Rs.groupby("hadm_id"):
    if len(g) < 2:
        continue
    d = {"hadm_id": h, "n_rec": len(g)}
    hrs = (g["t"].iloc[-1]
           - g["t"].iloc[0]).total_seconds()
    d["span_h"] = hrs / 3600.0
    for c in LG:
        v = g[c].values
        d[c + "_delta"] = float(v[-1] - v[0])
        d[c + "_sd"] = float(np.nanstd(v))
    rows.append(d)
TJ = pd.DataFrame(rows)
TR = [c for c in TJ.columns if c != "hadm_id"]
print("trajectory:", TJ.shape, flush=True)

lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
D = A.merge(lab, on=["subject_id", "hadm_id"],
            how="inner").merge(
    TJ, on="hadm_id", how="left")
print("cohort:", len(D), flush=True)

res, keep = [], []
t0 = time.time()

for oc in OUTS:
    d = D
    if oc == "cv_first":
        d = D[D["death_first"] == 0]
    y = pd.to_numeric(d[oc], errors="coerce")
    y = y.fillna(0).astype(int).values
    grp = d["subject_id"].values
    X0 = d[LG].values.astype(float)
    Xt = np.column_stack(
        [X0, d[TR].values.astype(float)])

    print("")
    print("=" * 74)
    print("%s  n=%d ev=%d   published %.4f"
          % (oc, len(y), int(y.sum()),
             PUB[oc]), flush=True)
    print("  %-13s %8s %8s %8s  %s"
          % ("configuration", "mean", "SD",
             "seed42", "C values"))

    store = {}
    for nm, cw, tune, traj in GRID:
        X = Xt if traj else X0
        aucs, cs_all, p42 = [], [], None
        for s in SEEDS:
            p, cs = oof(X, y, grp, s, cw, tune)
            aucs.append(roc_auc_score(y, p))
            cs_all += cs
            if s == 42:
                p42 = p
        v = np.array(aucs)
        store[nm] = (v.mean(), p42)
        ct = (str(sorted(set(cs_all)))
              if tune else "1.0")
        print("  %-13s %.4f  %.4f  %.4f  %s"
              % (nm, v.mean(),
                 v.std(ddof=1), v[0], ct),
              flush=True)
        res.append({
            "outcome": oc, "config": nm,
            "class_weight": str(cw),
            "tuned_C": tune,
            "trajectory": traj,
            "n": len(y), "ev": int(y.sum()),
            "seed_mean": v.mean(),
            "seed_sd": v.std(ddof=1),
            "seed42": v[0], "lo": v.min(),
            "hi": v.max(),
            "published": PUB[oc],
            "vs_published": v.mean() - PUB[oc],
            "C": ct})

    base = store["published"][0]
    bn = max(store, key=lambda k: store[k][0])
    print("")
    print("  best: %s  %.4f"
          "  (%+.4f over published protocol,"
          " %+.4f over published figure)"
          % (bn, store[bn][0],
             store[bn][0] - base,
             store[bn][0] - PUB[oc]))

    pb = _rank(store[bn][1])
    p0 = _rank(store["published"][1])
    g, lo, hi, _ = f.boot_diff(
        y, pb, p0, grp)
    print("  paired bootstrap at seed 42:"
          " %+.4f [%+.4f,%+.4f] %s"
          % (g, lo, hi,
             "*" if (lo > 0 or hi < 0)
             else ""), flush=True)

    o = d[["subject_id", "hadm_id"]].copy()
    o["p_ecg"] = store[bn][1]
    o["outcome"] = oc
    o["config"] = bn
    keep.append(o)

r = pd.DataFrame(res)
r.to_csv(DEST, index=False)
pd.concat(keep).to_csv(
    os.path.join(PROC, "p_ecg_best.csv"),
    index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("FIVE-SEED MEAN AUROC BY CONFIGURATION")
print(r.pivot_table(index="config",
                    columns="outcome",
                    values="seed_mean")
      .round(4).to_string())
print("")
print("FIVE-SEED SD (lower is more stable)")
print(r.pivot_table(index="config",
                    columns="outcome",
                    values="seed_sd")
      .round(4).to_string())
print("")
print("GAIN OVER THE PUBLISHED FIGURE")
print(r.pivot_table(index="config",
                    columns="outcome",
                    values="vs_published")
      .round(4).to_string())
print("")
print("BEST CONFIGURATION PER OUTCOME")
for oc, s in r.groupby("outcome"):
    b = s.loc[s["seed_mean"].idxmax()]
    p0 = float(s[s["config"] == "published"]
               ["seed_mean"].iloc[0])
    print("  %-18s %-13s %.4f"
          "  (published protocol %.4f,"
          " published figure %.4f)"
          % (oc, b["config"],
             b["seed_mean"], p0,
             b["published"]))
    print("      cw=%s  tunedC=%s  traj=%s"
          "  C=%s"
          % (b["class_weight"], b["tuned_C"],
             b["trajectory"], b["C"]))
print("")
print("AVERAGE RANK ACROSS OUTCOMES")
rk = r.pivot_table(index="config",
                   columns="outcome",
                   values="seed_mean").rank(
    ascending=False).mean(axis=1)
print(rk.sort_values().round(2).to_string())
print("")
print("saved", DEST, r.shape)
print("saved data/processed/p_ecg_best.csv")