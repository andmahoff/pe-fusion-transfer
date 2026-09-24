"""Test whether ctpa_notes_index should be joined
on idx_hadm rather than hadm_id.
Run in venv (analysis).
"""
import os
import pandas as pd

# folder holding the dissertation's derived files
DIS = os.environ.get(
    "PE_DISS_DIR", os.path.join("..", "Dissertation"))
FIG = os.path.join(DIS, "figwork", "data")

idx = pd.read_csv(os.path.join(
    FIG, "ctpa_notes_index.csv"))
feat = pd.read_csv(os.path.join(
    FIG, "mimic_imp_features_v2.csv"),
    usecols=["hadm_id"])
f = set(feat["hadm_id"])

print("index rows:", len(idx))
for c in ("hadm_id", "idx_hadm"):
    v = idx[c].dropna()
    print("")
    print("join on %s:" % c)
    print("  unique:", v.nunique())
    print("  covers %d of %d features"
          % (len(f & set(v)), len(f)))
    print("  extra not in features:",
          len(set(v) - f))

print("")
print("h_before describe:")
print(idx["h_before"].describe().round(2)
      .to_string())
print("")
print("rows with h_before > 0 (pre-admission):",
      int((idx["h_before"] > 0).sum()))
print("hadm_id == idx_hadm on:",
      int((idx["hadm_id"]
           == idx["idx_hadm"]).sum()),
      "of", len(idx))