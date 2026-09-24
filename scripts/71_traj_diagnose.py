"""Tests whether trajectory fails through overfitting
or through a lack of signal.

Four diagnostics:
  1 STANDALONE   fit on trajectory columns
                 alone, no logits, swept over C.
                 Near 0.50 means no signal;
                 0.60+ means signal that cannot
                 be combined.
  2 TRAIN vs OOF the overfitting signature. A
                 large gap means noise is being
                 fitted; a small gap means there
                 is nothing to fit.
  3 SUBGROUP     restrict to admissions that
                 actually have serial ECGs, so
                 the 56% imputed-constant rows
                 cannot dilute.
  4 PERMUTATION  shuffle the trajectory block
                 within the subgroup. If real
                 trajectory scores no better
                 than shuffled, it carries
                 nothing.

Five seeds throughout.
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\traj_diagnose.csv
  results\\traj_diagnose_log.txt
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
CGRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
OUTS = ["cv_first", "death_30d_inhosp",
        "composite_30d", "death_30d"]
DEST = os.path.join(PROC,
                    "traj_diagnose.csv")


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def run(X, y, grp, seed, C,
        cw="balanced", want_train=False):
    """Grouped OOF, and optionally the
    in-fold training AUC for comparison."""
    p = np.zeros(len(y))
    tr_auc = []
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
            class_weight=cw)
        m.fit(a, y[tr])
        p[te] = m.predict_proba(b)[:, 1]
        if want_train:
            tr_auc.append(roc_auc_score(
                y[tr],
                m.predict_proba(a)[:, 1]))
    oof = roc_auc_score(y, p)
    return (oof, float(np.mean(tr_auc))
            if want_train else np.nan)


def sweep(X, y, grp, want_train=False):
    """Best C by five-seed mean OOF."""
    best = None
    for C in CGRID:
        oo, tt = [], []
        for s in SEEDS:
            o, t = run(X, y, grp, s, C,
                       want_train=want_train)
            oo.append(o)
            tt.append(t)
        m = float(np.mean(oo))
        if best is None or m > best[1]:
            best = (C, m,
                    float(np.std(oo, ddof=1)),
                    float(np.nanmean(tt)))
    return best


# ---------- features ----------
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

Rs = R.dropna(subset=["t"]).sort_values(
    ["hadm_id", "t"])
rows = []
for h, g in Rs.groupby("hadm_id"):
    if len(g) < 2:
        continue
    d = {"hadm_id": h, "has_traj": 1}
    for c in LG:
        v = g[c].values
        d[c + "_delta"] = float(v[-1] - v[0])
        d[c + "_sd"] = float(np.nanstd(v))
    rows.append(d)
TJ = pd.DataFrame(rows)
TR = [c for c in TJ.columns
      if c not in ("hadm_id", "has_traj")]
print("logits %d   trajectory %d"
      % (len(LG), len(TR)))
print("admissions with serial ECGs:",
      len(TJ), flush=True)

lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
D = A.merge(lab, on=["subject_id", "hadm_id"],
            how="inner").merge(
    TJ, on="hadm_id", how="left")
D["has_traj"] = D["has_traj"].fillna(0)
print("full cohort:", len(D),
      "  with trajectory:",
      int(D["has_traj"].sum()), flush=True)

res = []
t0 = time.time()

for oc in OUTS:
    d0 = D
    if oc == "cv_first":
        d0 = D[D["death_first"] == 0]
    y0 = pd.to_numeric(
        d0[oc], errors="coerce").fillna(
        0).astype(int).values
    g0 = d0["subject_id"].values
    sub = d0["has_traj"].values == 1
    d1 = d0[sub]
    y1 = y0[sub]
    g1 = g0[sub]

    print("")
    print("=" * 76)
    print("%s   full n=%d ev=%d"
          "   serial-ECG n=%d ev=%d"
          % (oc, len(y0), int(y0.sum()),
             len(y1), int(y1.sum())),
          flush=True)
    if y1.sum() < 25:
        print("  too few events in the"
              " subgroup, skipping")
        continue

    XL0 = d0[LG].values.astype(float)
    XT0 = d0[TR].values.astype(float)
    XL1 = d1[LG].values.astype(float)
    XT1 = d1[TR].values.astype(float)
    XB1 = np.column_stack([XL1, XT1])

    print("")
    print("  1. STANDALONE  (is there any"
          " signal at all?)")
    c, m, s, t = sweep(XT1, y1, g1,
                       want_train=True)
    print("     trajectory only, serial-ECG"
          " subgroup")
    print("       OOF %.4f (SD %.4f)"
          "   train %.4f   gap %+.4f"
          "   C=%.0e" % (m, s, t, t - m, c))
    verdict = ("NO SIGNAL" if m < 0.55
               else "WEAK SIGNAL" if m < 0.60
               else "REAL SIGNAL")
    print("       ->", verdict)
    res.append({"outcome": oc,
                "test": "standalone_traj",
                "n": len(y1),
                "ev": int(y1.sum()),
                "auc": m, "sd": s,
                "train": t, "gap": t - m,
                "C": c})

    print("")
    print("  2. OVERFITTING SIGNATURE")
    for nm, X in (("logits", XL1),
                  ("logits+traj", XB1)):
        c, m, s, t = sweep(X, y1, g1,
                           want_train=True)
        print("     %-12s OOF %.4f"
              "   train %.4f   gap %+.4f"
              "   C=%.0e"
              % (nm, m, t, t - m, c))
        res.append({"outcome": oc,
                    "test": "overfit_" + nm,
                    "n": len(y1),
                    "ev": int(y1.sum()),
                    "auc": m, "sd": s,
                    "train": t, "gap": t - m,
                    "C": c})

    print("")
    print("  3. SUBGROUP COMPARISON"
          "  (fair, no imputed rows)")
    cl, ml, sl, _ = sweep(XL1, y1, g1)
    cb, mb, sb, _ = sweep(XB1, y1, g1)
    print("     logits       %.4f (SD %.4f)"
          " at C=%.0e" % (ml, sl, cl))
    print("     logits+traj  %.4f (SD %.4f)"
          " at C=%.0e" % (mb, sb, cb))
    print("     trajectory contributes"
          " %+.4f" % (mb - ml))
    res.append({"outcome": oc,
                "test": "subgroup_delta",
                "n": len(y1),
                "ev": int(y1.sum()),
                "auc": mb - ml, "sd": np.nan,
                "train": np.nan,
                "gap": np.nan, "C": cb})

    print("")
    print("  4. PERMUTATION  (shuffled"
          " trajectory, 3 draws)")
    perm = []
    for k in range(3):
        rng = np.random.default_rng(100 + k)
        XP = np.column_stack(
            [XL1, XT1[rng.permutation(
                len(XT1))]])
        _, mp, _, _ = sweep(XP, y1, g1)
        perm.append(mp)
    pm = float(np.mean(perm))
    print("     shuffled     %.4f"
          "  (real %.4f, difference %+.4f)"
          % (pm, mb, mb - pm))
    if mb - pm < 0.005:
        print("     -> real trajectory is no"
              " better than shuffled")
    else:
        print("     -> real trajectory beats"
              " shuffled")
    res.append({"outcome": oc,
                "test": "permutation",
                "n": len(y1),
                "ev": int(y1.sum()),
                "auc": pm, "sd": np.nan,
                "train": np.nan,
                "gap": mb - pm, "C": np.nan})

r = pd.DataFrame(res)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 76)
print("1. STANDALONE TRAJECTORY AUROC")
q = r[r["test"] == "standalone_traj"]
print(q[["outcome", "n", "ev", "auc", "sd",
         "train", "gap"]].round(4)
      .to_string(index=False))
print("")
print("  0.50 means no signal;"
      " a large train-OOF gap means"
      " overfitting")

print("")
print("2. OVERFITTING GAP (train minus OOF)")
q = r[r["test"].str.startswith("overfit_")]
print(q.pivot_table(index="test",
                    columns="outcome",
                    values="gap")
      .round(4).to_string())

print("")
print("3. TRAJECTORY CONTRIBUTION IN THE"
      " SERIAL-ECG SUBGROUP")
q = r[r["test"] == "subgroup_delta"]
print(q[["outcome", "n", "ev", "auc"]]
      .rename(columns={"auc": "delta"})
      .round(4).to_string(index=False))

print("")
print("4. REAL MINUS SHUFFLED TRAJECTORY")
q = r[r["test"] == "permutation"]
print(q[["outcome", "auc", "gap"]]
      .rename(columns={"auc": "shuffled",
                       "gap": "real_minus_shuf"})
      .round(4).to_string(index=False))

print("")
print("VERDICT")
st = r[r["test"] == "standalone_traj"]
sg = r[r["test"] == "subgroup_delta"]
pmv = r[r["test"] == "permutation"]
print("  mean standalone AUROC: %.4f"
      % st["auc"].mean())
print("  mean subgroup contribution: %+.4f"
      % sg["auc"].mean())
print("  mean real minus shuffled: %+.4f"
      % pmv["gap"].mean())
if st["auc"].mean() < 0.55:
    print("  -> trajectory carries no"
          " prognostic signal in this cohort")
elif sg["auc"].mean() < 0.0:
    print("  -> signal exists but does not"
          " survive alongside the logits")
else:
    print("  -> trajectory contributes in the"
          " serial-ECG subgroup")
print("")
print("saved", DEST, r.shape)