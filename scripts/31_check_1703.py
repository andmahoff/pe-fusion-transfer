"""Check whether the dissertation's 1,703-admission
cohort and 46-feature CTPA set can be reconstructed.
Run in venv (analysis).
"""
import os
import sys
import pandas as pd

sys.path.insert(0, "scripts")
import fusion_lib as f

FIG = f.FIG
cb = pd.read_csv(os.path.join(
    FIG, "ctpa_comorb_features.csv"))
v2 = pd.read_csv(os.path.join(
    FIG, "mimic_imp_features_v2.csv"))
lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))

print("ctpa_comorb_features rows:", len(cb))
print("  unique hadm_id:",
      cb["hadm_id"].nunique())
print("mimic_imp_features_v2 rows:", len(v2))
print("  unique hadm_id:",
      v2["hadm_id"].nunique())

a = set(cb["hadm_id"])
b = set(v2["hadm_id"])
print("")
print("in comorb not in v2:", len(a - b))
print("in v2 not in comorb:", len(b - a))
print("union:", len(a | b))

dv = [c for c in cb.columns
      if c.startswith("dv_")]
cm = [c for c in cb.columns
      if c.startswith("cm_")]
print("")
print("device flags:", len(dv), dv)
print("comorb flags:", len(cm))

base = [c for c in f.BASE_C
        if c in cb.columns]
print("base features present:", len(base),
      "of", len(f.BASE_C))
tot = len(base) + len(cm) + len(dv)
print("total reconstructable:", tot,
      "(dissertation used 46)")

print("")
print("EVENT COUNTS ON EACH COHORT")
for nm, s in (("v2 (1649)", b),
              ("comorb", a),
              ("union", a | b)):
    m = lab[lab["hadm_id"].isin(s)]
    print("  %-12s n=%d  death %d"
          "  composite %d"
          % (nm, len(m),
             int(m["death_30d"].sum()),
             int(m["composite_30d"].sum())))

ecg = pd.read_csv(os.path.join(
    FIG, "p_ecg_harm_death_30d.csv"))
e = set(ecg["hadm_id"])
print("")
print("WITH ECG COVERAGE")
for nm, s in (("v2", b), ("comorb", a),
              ("union", a | b)):
    k = s & e
    m = lab[lab["hadm_id"].isin(k)]
    print("  %-12s n=%d  death %d"
          % (nm, len(m),
             int(m["death_30d"].sum())))

print("")
print("TARGET: dissertation used n=1703,"
      " 157 death events")