"""Verify the CTPA modality uses exactly the 46
features listed in Table 9 of the dissertation.

The table names every feature. This checks them
one by one against f.CTPA46 and against the
columns actually present in the fitted cohort,
and reports prevalence so a feature that exists
but is always zero cannot pass silently.
Run in venv (analysis).

OUTPUT FILE
  data\\processed\\ctpa_feature_check.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
CT_LO, CT_HI = -48.0, 24.0
DEST = os.path.join(
    PROC, "ctpa_feature_check.csv")

# Table 9, transcribed exactly
TABLE9 = {
    "PE descriptors": [
        "pe_pos", "pe_neg", "saddle",
        "central", "lobar", "segmental",
        "subseg", "bilateral", "rv_strain",
        "septal_bow", "reflux",
        "mpa_enlarge", "infarct"],
    "Incidental thoracic": [
        "effusion", "malignancy", "consolid",
        "atelect", "edema", "cardiomeg",
        "adenopathy"],
    "Comorbidity mentions": [
        "cm_emphysema", "cm_fibrosis",
        "cm_bronchiect", "cm_mets",
        "cm_lymphangitic", "cm_cirrhosis",
        "cm_ascites", "cm_pericard_eff",
        "cm_aortic_ath", "cm_aneurysm",
        "cm_valve", "cm_cachexia",
        "cm_obesity", "cm_renal",
        "cm_pleural_thick", "cm_vert_fx"],
    "Support devices": [
        "dv_ett", "dv_trach", "dv_cvc",
        "dv_pacer", "dv_sternotomy",
        "dv_chesttube", "dv_ngtube",
        "dv_ivcfilter"],
    "Report structure": [
        "txt_len", "n_sent"]}

ALL9 = [c for b in TABLE9.values() for c in b]
print("TABLE 9")
for b, cs in TABLE9.items():
    print("  %-22s %2d" % (b, len(cs)))
print("  %-22s %2d  (dissertation states 46)"
      % ("TOTAL", len(ALL9)))

CODE = list(f.CTPA46)
print("")
print("f.CTPA46 declares:", len(CODE))

miss = [c for c in ALL9 if c not in CODE]
extra = [c for c in CODE if c not in ALL9]
print("")
print("TABLE 9 vs f.CTPA46")
print("  in Table 9, absent from code:",
      len(miss))
for c in miss:
    print("     ", c)
print("  in code, absent from Table 9:",
      len(extra))
for c in extra:
    print("     ", c)
if not miss and not extra:
    print("  -> EXACT MATCH")

# the cohort actually fitted
mm = f.load_mimic_ctpa46()
cidx = pd.read_csv(os.path.join(
    FIG, "ctpa_notes_index.csv"))
w = cidx[(cidx["h_before"] >= CT_LO)
         & (cidx["h_before"] <= CT_HI)]
hs = set(w["idx_hadm"].dropna().astype(int))
mm = mm[mm["hadm_id"].isin(hs)]
v2 = pd.read_csv(os.path.join(
    FIG, "mimic_imp_features_v2.csv"),
    usecols=["hadm_id"])
mm = mm[mm["hadm_id"].isin(set(v2["hadm_id"]))]
print("")
print("fitted cohort:", len(mm),
      " (dissertation used 1,703)")

absent = [c for c in ALL9
          if c not in mm.columns]
print("")
print("TABLE 9 vs THE COHORT")
print("  missing columns:", len(absent))
for c in absent:
    print("     ", c)

rows = []
print("")
print("PREVALENCE BY BLOCK")
print("  a binary feature at 0.000 exists but"
      " never fires, so it cannot contribute")
nzero, nconst = 0, 0
for b, cs in TABLE9.items():
    print("")
    print("  " + b)
    for c in cs:
        if c not in mm.columns:
            print("    %-18s MISSING" % c)
            rows.append({"block": b,
                         "feature": c,
                         "present": 0,
                         "mean": np.nan,
                         "sd": np.nan})
            continue
        v = pd.to_numeric(
            mm[c], errors="coerce").dropna()
        mu = float(v.mean()) if len(v) else \
            np.nan
        sd = float(v.std()) if len(v) else \
            np.nan
        binary = set(
            np.unique(v.dropna())) <= {0, 1} \
            if len(v) else False
        flag = ""
        if binary and mu == 0:
            flag = "  (never fires)"
            nzero += 1
        elif sd is not None and sd == 0:
            flag = "  (constant)"
            nconst += 1
        if binary:
            print("    %-18s %.3f%s"
                  % (c, mu, flag))
        else:
            print("    %-18s mean %.1f"
                  "  sd %.1f%s"
                  % (c, mu, sd, flag))
        rows.append({"block": b,
                     "feature": c,
                     "present": 1,
                     "mean": mu, "sd": sd})

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("=" * 62)
print("VERDICT")
print("=" * 62)
ok = (not miss and not extra
      and not absent)
print("  Table 9 features:      %d" % len(ALL9))
print("  declared in f.CTPA46:  %d" % len(CODE))
print("  present in the cohort: %d"
      % (len(ALL9) - len(absent)))
print("  never fire:            %d" % nzero)
print("  constant:              %d" % nconst)
print("")
if ok and nzero == 0:
    print("  the fitted model uses exactly the"
          " 46 features in Table 9")
elif ok:
    print("  all 46 are present, but %d never"
          " fire in this cohort, so the"
          " effective set is %d"
          % (nzero, len(ALL9) - nzero))
else:
    print("  MISMATCH: see the lists above")
print("")
print("saved", DEST, r.shape)