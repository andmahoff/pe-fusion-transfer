"""Check which stored ECG predictions survived,
and whether they align with the fusion cohort.
Run in venv (analysis).
"""
import os
import glob
import sys
import pandas as pd

sys.path.insert(0, "scripts")
import fusion_lib as f

FIG = f.FIG
pats = ["p_ecg_*.csv", "*ecg*predictions*.csv",
        "fusion_cohort_keys.csv",
        "p_cxr_harm_*.csv"]

print("ECG AND CXR PREDICTION FILES")
seen = set()
for p in pats:
    for fp in glob.glob(os.path.join(FIG, p)):
        if fp in seen:
            continue
        seen.add(fp)
        d = pd.read_csv(fp)
        print("")
        print("###", os.path.basename(fp))
        print("  rows %d  cols %s"
              % (len(d), list(d.columns)))
        print(d.head(2).to_string(index=False))

# folder holding the dissertation's derived files
DIS = os.environ.get(
    "PE_DISS_DIR", os.path.join("..", "Dissertation"))
HPC = os.path.join(DIS, "HPC", "Model Results", "ptbxl on mimic")
if os.path.isdir(HPC):
    print("")
    print("HPC ptbxl folder")
    for fp in glob.glob(
            os.path.join(HPC, "*.csv")):
        d = pd.read_csv(fp, nrows=3)
        print("  %s  cols %s"
              % (os.path.basename(fp),
                 list(d.columns)))

print("")
print("OVERLAP WITH THE FUSION COHORT")
mim = f.load_mimic()
print("  CTPA+EHR cohort:", len(mim),
      "admissions")
for fp in sorted(seen):
    d = pd.read_csv(fp)
    key = None
    for k in ("hadm_id", "subject_id"):
        if k in d.columns:
            key = k
            break
    if key is None:
        continue
    ov = len(set(mim[key]) & set(d[key]))
    print("  %-34s %s overlap %d of %d"
          % (os.path.basename(fp), key, ov,
             len(mim)))