"""Reproduction check. Rebuild the 28-feature EHR
model from the dissertation's 126_expanded_retrain.py
using the rebuilt MIMIC vitals, and compare
against the dissertation's recorded figures. Run in venv (analysis).
"""
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# folder holding the dissertation's derived files
DIS = os.environ.get(
    "PE_DISS_DIR", os.path.join("..", "Dissertation"))
FIG = os.path.join(DIS, "figwork", "data")
DSET = os.path.join(DIS, "Datasets")
RAW = os.path.join("data", "raw", "inspect")
PROC = os.path.join("data", "processed")

SEEDS = [42, 7, 13]

TA = ["temp", "hr", "sbp", "dbp", "rr",
      "creatinine", "sodium", "potassium",
      "bun", "glucose", "calcium", "bicarb",
      "chloride", "hct", "plt", "wbc", "hgb",
      "aniongap", "rbc", "mchc", "mch", "mcv",
      "rdw"]
FLAGS = ["afib", "cancer", "copd",
         "heart_failure"]

# ---------- INSPECT ----------
ins = pd.read_csv(
    os.path.join(FIG, "inspect_expanded_feats.csv"))
ilab = pd.read_csv(
    os.path.join(FIG, "inspect_labels_final.csv"))
ifl = pd.read_csv(
    os.path.join(RAW, "inspect_flags.csv"))
per = pd.read_csv(
    os.path.join(DSET, "INSPECT_EHR", "person.csv"),
    usecols=["person_id", "year_of_birth"])

ins = ins.merge(ilab[["person_id", "pe_date",
                      "death_30d", "composite_30d",
                      "cv_first", "death_first"]],
                on="person_id", how="left")
ins = ins.merge(ifl, on="person_id", how="left")
for f in FLAGS:
    ins[f] = ins[f].fillna(0).astype(int)
ins = ins.merge(per, on="person_id", how="left")
py = pd.to_datetime(ins["pe_date"],
                    errors="coerce").dt.year
ins["age"] = py - ins["year_of_birth"]
ins.loc[(ins["age"] < 18) | (ins["age"] > 100),
        "age"] = np.nan
print("INSPECT rows:", len(ins))
print("  age median:", float(ins["age"].median()))
for f in FLAGS:
    print("  %-14s %.3f" % (f, ins[f].mean()))

# ---------- MIMIC ----------
mm = pd.read_csv(
    os.path.join(FIG, "mimic_expanded_feats.csv"))
vit = pd.read_csv(
    os.path.join(PROC, "mimic_vitals_rebuilt.csv"))
vit = vit.drop(columns=["subject_id"])
vit.columns = ["hadm_id"] + [
    c.replace("mean_", "")
    for c in vit.columns[1:]]
mm = mm.merge(vit, on="hadm_id", how="left")

mfl = pd.read_csv(
    os.path.join(FIG,
                 "mimic_condition_flags_timed.csv"))
keep = ["hadm_id"] + [f + "_prior_idx"
                      for f in FLAGS]
mfl = mfl[keep].drop_duplicates("hadm_id")
mfl.columns = ["hadm_id"] + FLAGS
mm = mm.merge(mfl, on="hadm_id", how="left")
for f in FLAGS:
    mm[f] = mm[f].fillna(0).astype(int)

coh = pd.read_csv(
    os.path.join(DSET, "MIMICIV",
                 "mimic_pe_mace_cohort.csv"),
    usecols=["hadm_id", "age_at_admit"])
coh = coh.dropna().drop_duplicates("hadm_id")
coh = coh.rename(columns={"age_at_admit": "age"})
mm = mm.merge(coh, on="hadm_id", how="left")

mlab = pd.read_csv(
    os.path.join(FIG, "mimic_labels_harmonised.csv"))
mm = mm.merge(mlab, on=["subject_id", "hadm_id"],
              how="inner")
print("")
print("MIMIC rows:", len(mm))
print("  age median:", float(mm["age"].median()))
for f in FLAGS:
    print("  %-14s %.3f" % (f, mm[f].mean()))

CELLS = {
    "current": ["temp", "hr", "sbp", "dbp", "rr",
                "creatinine", "hgb", "wbc",
                "neut_pct", "lymph_pct",
                "albumin"],
    "tierA": TA,
}
PAIRS = ["death_30d", "composite_30d", "cv_first"]
REF = {
    ("current", "death_30d"): 0.7946,
    ("current", "composite_30d"): 0.7700,
    ("current", "cv_first"): 0.7860,
    ("tierA", "death_30d"): 0.8234,
    ("tierA", "composite_30d"): 0.8003,
    ("tierA", "cv_first"): 0.7933,
}


def mk_lr():
    return Pipeline([
        ("im", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.1, max_iter=5000))])


rows = []
for oc in PAIRS:
    di, dm = ins, mm
    if oc == "cv_first":
        di = ins[ins["death_first"] == 0]
        dm = mm[mm["death_first"] == 0]
    yi = pd.to_numeric(di[oc], errors="coerce")
    yi = yi.fillna(0).astype(int).values
    ym = dm[oc].values.astype(int)
    print("")
    print("=" * 58)
    print("%s  INSPECT n=%d ev=%d |"
          " MIMIC n=%d ev=%d"
          % (oc, len(yi), int(yi.sum()),
             len(ym), int(ym.sum())))

    for cell, an in CELLS.items():
        use_age = cell == "tierA"
        ic = ["m7_" + a for a in an]
        mc = ["mi_" + a if "mi_" + a in dm.columns
              else a for a in an]
        ic = [c for c in ic if c in di.columns]
        mc = [c for c in mc if c in dm.columns]
        if len(ic) != len(mc):
            print("  %s: mismatch %d/%d"
                  % (cell, len(ic), len(mc)))
            continue
        extra = FLAGS + (["age"] if use_age else [])
        Xi = di[ic + extra].values.astype(float)
        Xm = dm[mc + extra].values.astype(float)

        m = mk_lr()
        m.fit(Xi, yi)
        p = m.predict_proba(Xm)[:, 1]
        a = roc_auc_score(ym, p)
        ap = average_precision_score(ym, p)

        cvs = []
        for s in SEEDS:
            cv = StratifiedGroupKFold(
                n_splits=5, shuffle=True,
                random_state=s)
            oof = np.zeros(len(yi))
            gi = di["person_id"].values
            for tr, te in cv.split(Xi, yi, gi):
                m2 = mk_lr()
                m2.fit(Xi[tr], yi[tr])
                oof[te] = m2.predict_proba(
                    Xi[te])[:, 1]
            cvs.append(roc_auc_score(yi, oof))

        ref = REF.get((cell, oc), np.nan)
        print("  %-9s ZS %.4f  (ref %.4f,"
              " diff %+.4f)  AP %.4f"
              "  insCV %.4f  %d feats"
              % (cell, a, ref, a - ref, ap,
                 float(np.mean(cvs)), Xi.shape[1]))
        rows.append({"outcome": oc, "cell": cell,
                     "nfeat": Xi.shape[1],
                     "zs_auc": a, "ref": ref,
                     "diff": a - ref, "ap": ap,
                     "ins_cv": float(np.mean(cvs))})

r = pd.DataFrame(rows)
r.to_csv(os.path.join(PROC, "repro_126.csv"),
         index=False)
print("")
print(r.round(4).to_string(index=False))