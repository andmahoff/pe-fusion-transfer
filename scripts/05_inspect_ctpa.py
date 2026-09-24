"""Build the INSPECT CTPA impression features using the
extractor from the dissertation's 113_inspect_train.py. Selects the index
scan per PE-cohort person (nearest CTPA within 30 days
of pe_date), applies the 38-feature regex set, and
validates against the recorded in-distribution figures.
Run in venv (analysis).
"""
import os
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# folder holding the dissertation's derived files
DIS = os.environ.get(
    "PE_DISS_DIR", os.path.join("..", "Dissertation"))
FIG = os.path.join(DIS, "figwork", "data")
RAW = os.path.join("data", "raw", "inspect")
PROC = os.path.join("data", "processed")
os.makedirs(PROC, exist_ok=True)
D = "_20250611.tsv"

SEEDS = [42, 7, 13]
GAP_D = 30

NEG = (r"\bno\b|\bnot\b|\bwithout\b|\bnone\b"
       r"|\bnor\b|\bnegative for\b|\babsent\b"
       r"|\bfree of\b|\bunremarkable\b"
       r"|\bresolved\b|\bruled out\b")

BASER = {
    "pe_pos": r"pulmonary embol|filling defect",
    "saddle": r"saddle",
    "central": r"\bcentral\b(?!\s*(?:line|venous"
               r"|cath|access))|main pulmonary",
    "lobar": r"lobar",
    "segmental": r"(?<!sub)segmental",
    "subseg": r"subsegmental",
    "bilateral": r"bilateral",
    "rv_strain": r"right ventric|rv[/ ]lv"
                 r"|right heart strain|rv strain",
    "septal_bow": r"septal bowing"
                  r"|septal flattening|d-?shaped",
    "reflux": r"reflux[^.]{0,25}"
              r"(?:ivc|vena cava|hepatic)",
    "mpa_enlarge": r"pulmonary arter\w+"
                   r"[^.]{0,30}(?:enlarg|dilat)",
    "infarct": r"infarct",
    "effusion": r"effusion",
    "malignancy": r"malignan|metasta|neoplas"
                  r"|carcinoma|\bmass\b",
    "consolid": r"consolidat|pneumonia",
    "atelect": r"atelecta",
    "edema": r"edema",
    "cardiomeg": r"cardiomegal",
    "adenopathy": r"adenopathy",
}

COMORB = {
    "emphysema": r"emphysem|\bcopd\b"
                 r"|centrilobular",
    "fibrosis": r"fibrosis|interstitial"
                r"|honeycomb|reticulation"
                r"|\bild\b",
    "bronchiect": r"bronchiectas",
    "mets": r"metasta|innumerable"
            r"|osseous lesion|lytic lesion"
            r"|sclerotic lesion",
    "lymphangitic": r"lymphangitic",
    "cirrhosis": r"cirrho|nodular (?:contour"
                 r"|liver)|hepatic steatosis",
    "ascites": r"ascites|peritoneal fluid",
    "pericard_eff": r"pericardial effusion"
                    r"|pericardial fluid",
    "aortic_ath": r"aortic (?:atheroscler"
                  r"|calcific)|atheroscler"
                  r"|calcified (?:aorta|plaque)",
    "aneurysm": r"aneurysm|dissect",
    "valve": r"valv\w+ (?:calcific|replace"
             r"|prosthe)|prosthetic valve"
             r"|annular calcific",
    "cachexia": r"cachexia|cachectic"
                r"|sarcopeni|muscle wasting",
    "obesity": r"obes|large body habitus"
               r"|body habitus",
    "renal": r"nephrostomy|atrophic kidney"
             r"|renal atroph|hydronephro",
    "pleural_thick": r"pleural thickening"
                     r"|pleural plaque",
    "vert_fx": r"compression (?:fracture|deform)"
               r"|vertebral fracture",
}


def norm(t):
    t = str(t)
    t = re.sub(r"<[^>]{1,20}>", " ", t)
    t = re.sub(r"_{2,}", " ", t)
    t = re.sub(r"^\s*IMPRESSION[S]?\s*:", " ",
               t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def clauses(s):
    s = re.sub(r"(\d)\.(\d)", r"\1<D>\2", s)
    ps = re.split(r"[.;:\n]|\b(?:but|however"
                  r"|although)\b", s)
    return [p.replace("<D>", ".").strip()
            for p in ps if p.strip()]


def hit(cs, pat, want_neg=False):
    for c in cs:
        m = re.search(pat, c)
        if not m:
            continue
        neg = bool(re.search(NEG, c[:m.start()]))
        if neg == want_neg:
            return 1
    return 0


def featurise(txt):
    low = txt.lower()
    cs = clauses(low)
    d = {}
    for k, p in BASER.items():
        d[k] = hit(cs, p)
    d["pe_neg"] = hit(cs, BASER["pe_pos"], True)
    d["txt_len"] = len(txt)
    d["n_sent"] = len(cs)
    for k, p in COMORB.items():
        d["cm_" + k] = hit(cs, p)
    return d


XC = (list(BASER.keys()) + ["pe_neg",
                            "txt_len", "n_sent"]
      + ["cm_" + k for k in COMORB])
print("features:", len(XC))


def make_model():
    return Pipeline([
        ("im", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=10, cv=3, scoring="roc_auc",
            max_iter=5000, n_jobs=-1))])


def evalcv(X, y, grp):
    aucs = []
    ap = np.nan
    for s in SEEDS:
        cv = StratifiedGroupKFold(
            n_splits=5, shuffle=True,
            random_state=s)
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y, grp):
            md = make_model()
            md.fit(X[tr], y[tr])
            oof[te] = md.predict_proba(
                X[te])[:, 1]
        aucs.append(roc_auc_score(y, oof))
        if s == 42:
            ap = average_precision_score(y, oof)
    return (float(np.mean(aucs)),
            float(np.std(aucs)), ap)


# ---------- assemble ----------
imp = pd.read_csv(
    os.path.join(RAW, "impressions" + D), sep="\t")
ilb = pd.read_csv(
    os.path.join(RAW, "labels" + D), sep="\t")
spl = pd.read_csv(
    os.path.join(RAW, "splits" + D), sep="\t")
mp = pd.read_csv(
    os.path.join(RAW, "study_mapping" + D),
    sep="\t")
hl = pd.read_csv(
    os.path.join(FIG, "inspect_labels_final.csv"))
hl["pdate"] = pd.to_datetime(hl["pe_date"],
                             errors="coerce")

g = imp.merge(ilb, on="impression_id")
g = g.merge(spl, on="impression_id")
g = g.merge(mp[["impression_id",
                "procedure_DATETIME"]],
            on="impression_id")
g["pdt"] = pd.to_datetime(
    g["procedure_DATETIME"], errors="coerce")
g["txt"] = g["impressions"].apply(norm)
g = g[g["txt"].str.len() > 0]
print("studies:", len(g))

rows = [featurise(t) for t in g["txt"]]
fx = pd.DataFrame(rows)
for c in XC:
    g[c] = fx[c].values

pe = g[g["person_id"].isin(set(hl["person_id"]))]
pe = pe.merge(
    hl[["person_id", "pdate", "death_30d",
        "composite_30d", "cv_first",
        "death_first"]],
    on="person_id", how="left")
pe["gap"] = (pe["pdt"] - pe["pdate"]).abs()
pe = pe[pe["gap"] <= pd.Timedelta(days=GAP_D)]
pe = pe.sort_values(["person_id", "gap"])
pe = pe.groupby("person_id",
                as_index=False).first()
print("PE cell persons:", len(pe), "(expect 3300)")

keep = (["person_id", "split", "1_month_mortality",
         "death_30d", "composite_30d", "cv_first",
         "death_first"] + XC)
keep = [c for c in keep if c in pe.columns]
out = pe[keep]
DEST = os.path.join(PROC,
                    "inspect_imp_features_v2.csv")
out.to_csv(DEST, index=False)
print("saved", DEST, out.shape)

# ---------- prevalence vs MIMIC ----------
mfx = pd.read_csv(
    os.path.join(FIG, "mimic_imp_features_v2.csv"))
print("")
print("PREVALENCE: INSPECT vs MIMIC")
for c in XC:
    if c in ("txt_len", "n_sent"):
        continue
    a = float(pe[c].mean())
    b = float(mfx[c].mean()) if c in mfx else np.nan
    fl = "  (differs > 0.10)" if abs(a - b) > 0.10 else ""
    print("  %-16s INS %.3f  MIM %.3f%s"
          % (c, a, b, fl))

# ---------- validation ----------
REF = {"native_1m_mortality": 0.6976,
       "harm_death_30d": 0.6926,
       "harm_composite_30d": 0.6331,
       "harm_cv_first": 0.5703}
JOBS = [("native_1m_mortality",
         "1_month_mortality"),
        ("harm_death_30d", "death_30d"),
        ("harm_composite_30d", "composite_30d"),
        ("harm_cv_first", "cv_first")]

print("")
print("IN-DISTRIBUTION CV vs recorded")
res = []
for nm, col in JOBS:
    dd = pe.copy()
    if col == "1_month_mortality":
        s = dd[col].astype(str).str.lower()
        dd = dd[s.isin(["true", "false"])]
        y = (dd[col].astype(str).str.lower()
             == "true").astype(int).values
    else:
        if col == "cv_first":
            dd = dd[dd["death_first"] == 0]
        y = pd.to_numeric(
            dd[col], errors="coerce"
        ).fillna(0).astype(int).values
    if y.sum() < 20:
        continue
    X = dd[XC].values.astype(float)
    a, s, ap = evalcv(X, y, dd["person_id"].values)
    r = REF[nm]
    print("  %-22s n=%d ev=%d  AUC %.4f"
          "  (ref %.4f, diff %+.4f)  AP %.4f"
          % (nm, len(y), int(y.sum()), a, r,
             a - r, ap))
    res.append({"model": nm, "n": len(y),
                "ev": int(y.sum()), "auc": a,
                "ref": r, "diff": a - r, "ap": ap})

pd.DataFrame(res).to_csv(
    os.path.join(PROC, "inspect_ctpa_check.csv"),
    index=False)
print("")
print(pd.DataFrame(res).round(4).to_string(
    index=False))