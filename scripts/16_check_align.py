"""Check how many MIMIC admissions with CTPA
features have an embedded impression, comparing
the note index against the feature matrix to show
where admissions are lost.
Run in venv (analysis).
"""
import os
import pandas as pd

# folder holding the dissertation's derived files
DIS = os.environ.get(
    "PE_DISS_DIR", os.path.join("..", "Dissertation"))
FIG = os.path.join(DIS, "figwork", "data")
EMB = os.path.join("data", "embeddings")

idx = pd.read_csv(os.path.join(
    FIG, "ctpa_notes_index.csv"))
feat = pd.read_csv(os.path.join(
    FIG, "mimic_imp_features_v2.csv"),
    usecols=["hadm_id"])
got = pd.read_csv(os.path.join(
    EMB, "ids_mimic.csv"))

print("note index rows:", len(idx))
print("note index admissions:",
      idx["hadm_id"].nunique())
print("feature matrix admissions:",
      feat["hadm_id"].nunique())
print("embedded admissions:", len(got))

f = set(feat["hadm_id"])
n = set(idx["hadm_id"].dropna())
g = set(got["hadm_id"])

print("")
print("in features, not in note index:",
      len(f - n))
print("in note index, not in features:",
      len(n - f))
print("in features, not embedded:", len(f - g))

miss = f - g
sub = idx[idx["hadm_id"].isin(miss)]
print("")
print("of the %d missing, %d have note rows"
      % (len(miss), sub["hadm_id"].nunique()))
if len(sub):
    print("null charttime among those: %d of %d"
          % (int(sub["charttime"].isna().sum()),
             len(sub)))
    print("null text among those: %d"
          % int(sub["text"].isna().sum()))

print("")
print("charttime nulls overall: %d of %d"
      % (int(idx["charttime"].isna().sum()),
         len(idx)))
print("text nulls overall: %d"
      % int(idx["text"].isna().sum()))

kept = idx[idx["hadm_id"].isin(f)]
print("")
print("rows for feature admissions:", len(kept))
print("unique admissions there:",
      kept["hadm_id"].nunique())