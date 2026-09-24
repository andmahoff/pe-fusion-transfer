"""Reproduce the dissertation ECG modality
exactly, per Tables 8 and 43.

  features   71 SCP statement logits.
             Per recording: element-wise MAX
             across the seven 250/125 windows.
             Per admission: MEAN across the
             recordings belonging to it.
             71 columns, not 142.
  learner    L2 logistic regression,
             C = 1.0 (no search performed),
             class_weight = "balanced",
             max_iter = 5000, lbfgs
  cv         StratifiedGroupKFold(5),
             grouped on subject_id, seed 42
  cohort     3,507 admissions with an ECG
             (3,131 for cv_first)

Targets from the dissertation results tables:
  composite_30d      0.7207   AP 0.3065
  death_30d          0.7229   AP 0.2422
  death_30d_inhosp   0.7063   AP 0.1537
  cv_first           0.6849   AP 0.1272
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ecg_exact.csv
  data\\processed\\p_ecg_exact.csv
  results\\ecg_exact_log.txt
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
RAW = os.path.join("data", "raw", "ecg")
SEED, NFOLD = 42, 5
DEST = os.path.join(PROC, "ecg_exact.csv")

TARGET = {"composite_30d": (0.7207, 0.3065,
                            3507, 547),
          "death_30d": (0.7229, 0.2422,
                        3507, 398),
          "death_30d_inhosp": (0.7063, 0.1537,
                               3507, 242),
          "cv_first": (0.6849, 0.1272,
                       3131, 171)}


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def fit_exact(X, y, grp, C=1.0,
              cw="balanced"):
    """Exactly as specified: C fixed, balanced
    weighting, grouped 5-fold, seed 42."""
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    for tr, te in cv.split(X, y, grp):
        sc = StandardScaler()
        a = sc.fit_transform(X[tr])
        b = sc.transform(X[te])
        m = LogisticRegression(
            C=C, max_iter=5000,
            class_weight=cw, penalty="l2",
            solver="lbfgs")
        m.fit(a, y[tr])
        p[te] = m.predict_proba(b)[:, 1]
    return p


# ---- record-level logits, max over windows ----
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
print("statements:", len(LG))

R = idx[["subject_id", "hadm_id",
         "rec"]].merge(rec, on="rec",
                       how="inner")
print("recordings:", len(R),
      " (dissertation used 6,004)")

# ---- MEAN across recordings per admission ----
A = R.groupby(["subject_id", "hadm_id"],
              as_index=False)[LG].mean()
A["n_ecg"] = R.groupby(
    ["subject_id", "hadm_id"]).size().values
print("admissions with an ECG:", len(A),
      " (dissertation used 3,507)")
print("features per admission:", len(LG),
      flush=True)

lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
print("label rows:", len(lab))
print("label cols:", [c for c in lab.columns
                      if c not in
                      ("subject_id",
                       "hadm_id")])

D = A.merge(lab, on=["subject_id", "hadm_id"],
            how="inner")
print("merged cohort:", len(D), flush=True)

rows, preds = [], []
for oc, (t_auc, t_ap, t_n, t_ev) in \
        TARGET.items():
    if oc not in D.columns:
        print("")
        print("skip %s: not in labels" % oc)
        continue
    d = D
    if oc == "cv_first" and \
            "death_first" in D.columns:
        d = D[D["death_first"] == 0]
    y = pd.to_numeric(d[oc], errors="coerce")
    y = y.fillna(0).astype(int).values
    grp = d["subject_id"].values
    X = d[LG].values.astype(float)

    print("")
    print("=" * 70)
    print("%s" % oc)
    print("  cohort   n=%d  events=%d"
          % (len(y), int(y.sum())))
    print("  target   n=%d  events=%d"
          % (t_n, t_ev), flush=True)

    p = fit_exact(X, y, grp)
    au = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    print("")
    print("  AUROC  %.4f   target %.4f"
          "   diff %+.4f"
          % (au, t_auc, au - t_auc))
    print("  AP     %.4f   target %.4f"
          "   diff %+.4f"
          % (ap, t_ap, ap - t_ap))
    verdict = ("EXACT" if abs(au - t_auc)
               < 0.005 else
               "CLOSE" if abs(au - t_auc)
               < 0.015 else "MISMATCH")
    print("  ->", verdict, flush=True)

    # cross-check against the stored file
    fp = os.path.join(
        FIG, "p_ecg_harm_%s.csv" % oc)
    co = np.nan
    if os.path.exists(fp):
        st = pd.read_csv(fp)[
            ["hadm_id",
             "p_ecg"]].drop_duplicates(
            "hadm_id")
        q = d[["hadm_id"]].copy()
        q["p_new"] = p
        q = q.merge(st, on="hadm_id",
                    how="inner")
        if len(q) > 100:
            co = float(pd.Series(
                _rank(q["p_new"].values)).corr(
                pd.Series(_rank(
                    q["p_ecg"].values)),
                method="spearman"))
            sa = roc_auc_score(
                pd.to_numeric(
                    d.set_index("hadm_id")
                    .loc[q["hadm_id"], oc]
                ).fillna(0).astype(int).values,
                _rank(q["p_ecg"].values))
            print("  stored p_ecg on the same"
                  " rows: %.4f" % sa)
            print("  correlation with stored:"
                  " %.3f" % co)

    rows.append({
        "outcome": oc, "n": len(y),
        "ev": int(y.sum()),
        "target_n": t_n, "target_ev": t_ev,
        "auc": au, "target_auc": t_auc,
        "diff_auc": au - t_auc, "ap": ap,
        "target_ap": t_ap,
        "diff_ap": ap - t_ap,
        "corr_stored": co,
        "verdict": verdict})

    o = d[["subject_id", "hadm_id"]].copy()
    o["p_ecg"] = p
    o["outcome"] = oc
    preds.append(o)

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)
if preds:
    pd.concat(preds).to_csv(
        os.path.join(PROC, "p_ecg_exact.csv"),
        index=False)

print("")
print("=" * 70)
print("REPRODUCTION SUMMARY")
print(r[["outcome", "n", "ev", "auc",
         "target_auc", "diff_auc",
         "corr_stored", "verdict"]]
      .round(4).to_string(index=False))
print("")
print("mean absolute AUROC difference: %.4f"
      % float(r["diff_auc"].abs().mean()))
n_ok = int((r["verdict"] == "EXACT").sum())
print("exact matches: %d of %d"
      % (n_ok, len(r)))
print("")
print("saved", DEST)
print("saved data/processed/p_ecg_exact.csv")