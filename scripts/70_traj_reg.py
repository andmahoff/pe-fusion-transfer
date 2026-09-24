"""Tests whether the trajectory model needs
heavier regularisation than the logit model.

Script 69 found trajectory catastrophic at
C = 1.0 but competitive once C was tuned, and the
selected values hit 0.0003, the bottom of that
grid. If the optimum lies below the floor, it was
never found.

Tests:
  1 an extended C grid down to 1e-6, on logits
    alone and on logits plus trajectory, so the
    two optima can be compared directly
  2 block scaling, dividing each block by the
    square root of its width, so the 142
    trajectory columns do not dominate the 71
    logit columns by count
  3 a narrower trajectory block, deltas only,
    dropping the within-admission SD columns

Five seeds throughout, matching the
dissertation's multi-seed protocol.
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\traj_reg.csv
  results\\traj_reg_log.txt
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
OUTS = ["cv_first", "death_30d_inhosp",
        "composite_30d", "death_30d"]
DEST = os.path.join(PROC, "traj_reg.csv")
PUB = {"composite_30d": 0.7207,
       "death_30d": 0.7229,
       "death_30d_inhosp": 0.7063,
       "cv_first": 0.6849}

# extended, four decades below the old floor
CGRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4,
         3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
         1e-1, 3e-1, 1.0]


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def prep_block(a, b, blocks, scale):
    """blocks is a list of column-count ints.
    scale divides each block by sqrt(width)."""
    if not scale:
        return a, b
    a2, b2 = a.copy(), b.copy()
    i = 0
    for n in blocks:
        k = np.sqrt(n)
        a2[:, i:i + n] /= k
        b2[:, i:i + n] /= k
        i += n
    return a2, b2


def run_fixed(X, y, grp, seed, C, cw,
              blocks=None, scale=False):
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
        if scale and blocks:
            a, b = prep_block(a, b, blocks,
                              True)
        m = LogisticRegression(
            C=C, max_iter=8000,
            class_weight=cw)
        m.fit(a, y[tr])
        p[te] = m.predict_proba(b)[:, 1]
    return roc_auc_score(y, p)


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
    d = {"hadm_id": h}
    for c in LG:
        v = g[c].values
        d[c + "_delta"] = float(v[-1] - v[0])
        d[c + "_sd"] = float(np.nanstd(v))
    rows.append(d)
TJ = pd.DataFrame(rows)
DL = [c for c in TJ.columns
      if c.endswith("_delta")]
SD = [c for c in TJ.columns
      if c.endswith("_sd")]
print("logits %d  delta %d  sd %d"
      % (len(LG), len(DL), len(SD)),
      flush=True)

lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
D = A.merge(lab, on=["subject_id", "hadm_id"],
            how="inner").merge(
    TJ, on="hadm_id", how="left")
print("cohort:", len(D), flush=True)

res = []
t0 = time.time()

for oc in OUTS:
    d = D
    if oc == "cv_first":
        d = D[D["death_first"] == 0]
    y = pd.to_numeric(d[oc], errors="coerce")
    y = y.fillna(0).astype(int).values
    grp = d["subject_id"].values

    X_lg = d[LG].values.astype(float)
    X_full = np.column_stack(
        [X_lg, d[DL].values.astype(float),
         d[SD].values.astype(float)])
    X_delta = np.column_stack(
        [X_lg, d[DL].values.astype(float)])

    VAR = [
        ("logits", X_lg, None, False),
        ("logits+traj", X_full,
         [len(LG), len(DL), len(SD)], False),
        ("logits+traj_scaled", X_full,
         [len(LG), len(DL), len(SD)], True),
        ("logits+delta", X_delta,
         [len(LG), len(DL)], False),
        ("logits+delta_scaled", X_delta,
         [len(LG), len(DL)], True),
    ]

    print("")
    print("=" * 76)
    print("%s  n=%d ev=%d   published %.4f"
          % (oc, len(y), int(y.sum()),
             PUB[oc]), flush=True)

    for vn, X, blocks, scale in VAR:
        curve = []
        for C in CGRID:
            aa = [run_fixed(X, y, grp, s, C,
                            "balanced",
                            blocks, scale)
                  for s in SEEDS]
            v = np.array(aa)
            curve.append((C, v.mean(),
                          v.std(ddof=1)))
            res.append({
                "outcome": oc, "variant": vn,
                "C": C, "mean": v.mean(),
                "sd": v.std(ddof=1),
                "nfeat": X.shape[1],
                "published": PUB[oc]})
        best = max(curve, key=lambda x: x[1])
        edge = ("  (at the grid edge)"
                if best[0] in (CGRID[0],
                               CGRID[-1])
                else "")
        print("")
        print("  %-20s %d features"
              % (vn, X.shape[1]))
        print("    peak C = %.0e   AUROC"
              " %.4f (SD %.4f)%s"
              % (best[0], best[1], best[2],
                 edge))
        line = "    curve: "
        for C, m, _ in curve:
            line += "%.4f " % m
        print(line, flush=True)

    sub = [x for x in res
           if x["outcome"] == oc]
    bl = max([x for x in sub
              if x["variant"] == "logits"],
             key=lambda x: x["mean"])
    bt = max([x for x in sub
              if x["variant"] != "logits"],
             key=lambda x: x["mean"])
    print("")
    print("  best logits-only : %.4f"
          " at C=%.0e" % (bl["mean"], bl["C"]))
    print("  best with traj   : %.4f"
          " at C=%.0e  (%s)"
          % (bt["mean"], bt["C"],
             bt["variant"]))
    print("  trajectory contributes %+.4f"
          % (bt["mean"] - bl["mean"]))
    print("  optimal C ratio, traj vs logits:"
          " %.1fx" % (bl["C"] / bt["C"])
          if bt["C"] > 0 else "")

r = pd.DataFrame(res)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 76)
print("PEAK AUROC BY VARIANT")
pk = r.loc[r.groupby(
    ["outcome", "variant"])["mean"].idxmax()]
print(pk.pivot_table(index="variant",
                     columns="outcome",
                     values="mean")
      .round(4).to_string())
print("")
print("C AT THE PEAK")
print(pk.pivot_table(index="variant",
                     columns="outcome",
                     values="C")
      .to_string())
print("")
print("DOES THE TRAJECTORY MODEL WANT A"
      " SMALLER C?")
for oc, s in pk.groupby("outcome"):
    a = s[s["variant"] == "logits"]["C"]
    b = s[s["variant"]
          == "logits+traj"]["C"]
    if len(a) and len(b):
        print("  %-18s logits %.0e   traj"
              " %.0e   ratio %.0fx"
              % (oc, a.iloc[0], b.iloc[0],
                 a.iloc[0] / b.iloc[0]))
print("")
print("BEST OVERALL PER OUTCOME")
for oc, s in pk.groupby("outcome"):
    b = s.loc[s["mean"].idxmax()]
    print("  %-18s %-20s %.4f at C=%.0e"
          "  (published %.4f, %+.4f)"
          % (oc, b["variant"], b["mean"],
             b["C"], b["published"],
             b["mean"] - b["published"]))
print("")
print("saved", DEST, r.shape)