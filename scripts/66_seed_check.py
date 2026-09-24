"""Tests whether the reproduction gap is fold-seed
variation. Runs the exact baseline across ten
fold seeds and checks whether the published
figure falls inside the spread.

The dissertation reports ECG fold-seed SD of
0.0045 to 0.0075 (Table 20b).
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ecg_seed_check.csv
  results\\ecg_seed_check_log.txt
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
RAW = os.path.join("data", "raw", "ecg")
SEEDS = [42, 7, 13, 1, 2, 3, 99, 123,
         2024, 8]
OUTS = ["composite_30d", "death_30d",
        "death_30d_inhosp", "cv_first"]
PUB = {"composite_30d": 0.7207,
       "death_30d": 0.7229,
       "death_30d_inhosp": 0.7063,
       "cv_first": 0.6849}
DEST = os.path.join(PROC,
                    "ecg_seed_check.csv")


def run(X, y, grp, seed):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(X, y, grp):
        sc = StandardScaler()
        a = sc.fit_transform(X[tr])
        b = sc.transform(X[te])
        m = LogisticRegression(
            C=1.0, max_iter=5000,
            class_weight="balanced")
        m.fit(a, y[tr])
        p[te] = m.predict_proba(b)[:, 1]
    return roc_auc_score(y, p)


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
R = idx[["subject_id", "hadm_id",
         "rec"]].merge(rec, on="rec",
                       how="inner")
A = R.groupby(["subject_id", "hadm_id"],
              as_index=False)[LG].mean()
lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
D = A.merge(lab, on=["subject_id",
                     "hadm_id"], how="inner")
print("admissions:", len(D), flush=True)

rows = []
for oc in OUTS:
    d = D
    if oc == "cv_first":
        d = D[D["death_first"] == 0]
    y = pd.to_numeric(
        d[oc], errors="coerce").fillna(
        0).astype(int).values
    grp = d["subject_id"].values
    X = d[LG].values.astype(float)

    aucs = []
    for s in SEEDS:
        aucs.append(run(X, y, grp, s))
    aucs = np.array(aucs)
    pub = PUB[oc]
    inside = (aucs.min() <= pub
              <= aucs.max())
    z = ((pub - aucs.mean())
         / max(aucs.std(ddof=1), 1e-9))

    print("")
    print("=" * 66)
    print("%s   published %.4f" % (oc, pub))
    print("  seed AUCs:",
          np.round(aucs, 4).tolist())
    print("  mean %.4f  SD %.4f"
          "  range %.4f to %.4f"
          % (aucs.mean(), aucs.std(ddof=1),
             aucs.min(), aucs.max()))
    print("  published sits %.2f SD from the"
          " mean" % z)
    print("  inside the seed range:",
          "YES" if inside else "NO",
          flush=True)
    for s, a in zip(SEEDS, aucs):
        rows.append({"outcome": oc,
                     "seed": s, "auc": a,
                     "published": pub})

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("=" * 66)
print("SUMMARY")
g = r.groupby("outcome")["auc"].agg(
    ["mean", "std", "min", "max"])
g["published"] = [PUB[i] for i in g.index]
g["inside"] = np.where(
    (g["min"] <= g["published"])
    & (g["published"] <= g["max"]),
    "yes", "no")
g["z"] = (g["published"] - g["mean"]) / g["std"]
print(g.round(4).to_string())
print("")
print("dissertation Table 20b reports ECG"
      " fold-seed SD of 0.0045 to 0.0075")
print("observed SD here: %.4f to %.4f"
      % (g["std"].min(), g["std"].max()))
n = int((g["inside"] == "yes").sum())
print("")
print("published figure inside the seed range"
      " for %d of %d outcomes" % (n, len(g)))
if n >= 3:
    print("-> the gap is fold-seed variation,"
          " not a different model")
else:
    print("-> something systematic differs")
print("")
print("saved", DEST, r.shape)