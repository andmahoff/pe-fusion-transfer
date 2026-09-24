"""Reproduce the exact dissertation ECG model.

Specification from the dissertation's script 82:
  features  71 SCP statement outputs per ECG
            record, max-aggregated over the
            seven 250/125 windows
  fit       L2 logistic regression at the
            record level (6,004 rows), C tuned
            by nested CV
  cv        StratifiedGroupKFold(5) grouped by
            subject_id
  aggregate per-record predictions averaged by
            mean to admission level

Script 62 aggregated logits to admission
level and fitted one model per admission,
which is a different estimator.

Two remaining ambiguities are tested as a grid:
raw logits vs sigmoid outputs, and class
weighting.
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\exp0_match.csv
  data\\processed\\p_ecg_reproduced.csv
  results\\exp0_match_log.txt
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
CS = [0.0001, 0.001, 0.003, 0.01, 0.03,
      0.1, 0.3, 1.0, 3.0, 10.0]
OUTS = ["death_30d", "death_30d_inhosp",
        "composite_30d"]
DEST = os.path.join(PROC, "exp0_match.csv")

TARGET = {"death_30d": 0.7367,
          "death_30d_inhosp": 0.7302,
          "composite_30d": 0.7357}


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def sig(x):
    return 1.0 / (1.0 + np.exp(
        -np.clip(x, -60, 60)))


def record_oof(X, y, grp, cw, cpol):
    """Fit at record level, grouped by subject.
    Returns per-record out-of-fold
    predictions."""
    p = np.zeros(len(y))
    cs = []
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    for tr, te in cv.split(X, y, grp):
        sc = StandardScaler()
        a = sc.fit_transform(X[tr])
        b = sc.transform(X[te])
        if cpol == "fixed1":
            c = 1.0
        else:
            best, c = -1.0, 0.01
            icv = StratifiedGroupKFold(
                n_splits=3, shuffle=True,
                random_state=SEED)
            for cand in CS:
                q = np.zeros(len(tr))
                try:
                    for t2, v2 in icv.split(
                            a, y[tr], grp[tr]):
                        mm = LogisticRegression(
                            C=cand,
                            max_iter=6000,
                            class_weight=cw)
                        mm.fit(a[t2], y[tr][t2])
                        q[v2] = \
                            mm.predict_proba(
                                a[v2])[:, 1]
                    s = roc_auc_score(y[tr], q)
                except Exception:
                    continue
                if s > best:
                    best, c = s, cand
        cs.append(c)
        m = LogisticRegression(
            C=c, max_iter=6000,
            class_weight=cw)
        m.fit(a, y[tr])
        p[te] = m.predict_proba(b)[:, 1]
    return p, cs


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

# record-level frame: one row per index entry,
# which is how the original ran (6,004 rows)
R = idx[["subject_id", "hadm_id",
         "rec"]].merge(
    rec, on="rec", how="inner")
print("record-level rows:", len(R),
      " (original had 6,004)")
print("admissions:", R["hadm_id"].nunique(),
      " subjects:",
      R["subject_id"].nunique())
print("statements:", len(LG), flush=True)

mim = f.load_mimic()
rows = []
best_overall = {}

for oc in OUTS:
    if oc not in mim.columns:
        continue
    lab = mim[["subject_id", "hadm_id",
               oc]].copy()
    lab[oc] = pd.to_numeric(
        lab[oc], errors="coerce").fillna(
        0).astype(int)

    D = R.merge(lab, on=["subject_id",
                         "hadm_id"],
                how="inner")
    y = D[oc].values
    grp = D["subject_id"].values
    if y.sum() < 30:
        continue

    fp = os.path.join(
        FIG, "p_ecg_harm_%s.csv" % oc)
    if not os.path.exists(fp):
        continue
    stq = pd.read_csv(fp)[
        ["hadm_id", "p_ecg"]].drop_duplicates(
        "hadm_id")

    Xr = D[LG].values.astype(float)
    Xs = sig(Xr)
    tgt = TARGET.get(oc, np.nan)

    print("")
    print("=" * 74)
    print("%s   records=%d  admissions=%d"
          "  record-level events=%d"
          % (oc, len(D),
             D["hadm_id"].nunique(),
             int(y.sum())))
    print("  stored target AUC %.4f" % tgt,
          flush=True)
    print("  %-28s %7s %9s %7s %6s"
          % ("configuration", "AUC",
             "vs target", "AP", "corr"))

    for sname, X in (("logit", Xr),
                     ("sigmoid", Xs)):
        for cwn, cw in (("none", None),
                        ("balanced",
                         "balanced")):
            for cpol in ("fixed1", "tuned"):
                pr, cs = record_oof(
                    X, y, grp, cw, cpol)
                D["_p"] = pr
                # MEAN aggregation to admission
                ad = D.groupby(
                    ["subject_id", "hadm_id"],
                    as_index=False)["_p"].mean()
                ad = ad.merge(
                    lab, on=["subject_id",
                             "hadm_id"],
                    how="inner")
                ad = ad.merge(
                    stq, on="hadm_id",
                    how="left")
                m = np.isfinite(
                    ad["p_ecg"]).values
                ya = ad.loc[m, oc].values
                pa = _rank(ad.loc[m,
                                  "_p"].values)
                sa = _rank(ad.loc[m,
                                  "p_ecg"].values)
                au = roc_auc_score(ya, pa)
                co = float(pd.Series(pa).corr(
                    pd.Series(sa),
                    method="spearman"))
                nm = "%s | cw=%s | C=%s" % (
                    sname, cwn, cpol)
                flag = ""
                if abs(au - tgt) < 0.01:
                    flag = "  (match)"
                elif abs(au - tgt) < 0.02:
                    flag = "  (close)"
                print("  %-28s %.4f  %+.4f"
                      "   %.4f  %.3f%s"
                      % (nm, au, au - tgt,
                         average_precision_score(
                             ya, pa), co, flag),
                      flush=True)
                rows.append({
                    "outcome": oc,
                    "scale": sname,
                    "class_weight": cwn,
                    "C_policy": cpol,
                    "C_chosen": str(
                        sorted(set(cs))),
                    "n_records": len(D),
                    "n_adm": int(m.sum()),
                    "ev": int(ya.sum()),
                    "auc": au, "target": tgt,
                    "diff": au - tgt,
                    "corr_stored": co,
                    "ap":
                        average_precision_score(
                            ya, pa)})
                k = (oc, abs(au - tgt))
                if (oc not in best_overall
                        or abs(au - tgt)
                        < best_overall[oc][0]):
                    best_overall[oc] = (
                        abs(au - tgt), nm,
                        ad[["subject_id",
                            "hadm_id",
                            "_p"]].copy())

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

# save the closest reproduction per outcome
out = []
for oc, (dd, nm, ad) in best_overall.items():
    a = ad.rename(columns={"_p": "p_ecg_repro"})
    a["outcome"] = oc
    a["config"] = nm
    out.append(a)
if out:
    pd.concat(out).to_csv(
        os.path.join(
            PROC, "p_ecg_reproduced.csv"),
        index=False)

print("")
print("=" * 74)
print("CLOSEST CONFIGURATION PER OUTCOME")
for oc, s in r.groupby("outcome"):
    b = s.loc[s["diff"].abs().idxmin()]
    print("  %-18s %s | cw=%s | C=%s"
          % (oc, b["scale"],
             b["class_weight"],
             b["C_policy"]))
    print("      AUC %.4f  target %.4f"
          "  (%+.4f)  corr %.3f  C=%s"
          % (b["auc"], b["target"],
             b["diff"], b["corr_stored"],
             b["C_chosen"]))

print("")
print("MEAN ABSOLUTE DIFFERENCE FROM TARGET")
g = r.groupby(["scale", "class_weight",
               "C_policy"])["diff"].apply(
    lambda x: x.abs().mean()).sort_values()
print(g.round(4).to_string())

print("")
print("MEAN CORRELATION WITH STORED p_ecg")
g2 = r.groupby(["scale", "class_weight",
                "C_policy"])[
    "corr_stored"].mean().sort_values(
    ascending=False)
print(g2.round(3).to_string())

print("")
print("saved", DEST, r.shape)
print("saved data/processed/"
      "p_ecg_reproduced.csv")