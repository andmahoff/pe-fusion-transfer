"""Compare candidate derivations of INSPECT age (at
pe_date or at the index visit, from year or full date
of birth) against the dissertation's recorded median
of 70.
Run in venv (analysis).
"""
import os
import numpy as np
import pandas as pd

# folder holding the dissertation's derived files
DIS = os.environ.get(
    "PE_DISS_DIR", os.path.join("..", "Dissertation"))
FIG = os.path.join(DIS, "figwork", "data")
INS = os.path.join(DIS, "Datasets", "INSPECT_EHR")
TARGET = 70.0

per = pd.read_csv(
    os.path.join(INS, "person.csv"),
    usecols=lambda c: c in (
        "person_id", "year_of_birth",
        "month_of_birth", "birth_DATETIME"))
print("person cols:", list(per.columns))
print("year_of_birth:")
print(per["year_of_birth"].describe())

hl = pd.read_csv(
    os.path.join(FIG, "inspect_labels_final.csv"))
hl["pdate"] = pd.to_datetime(hl["pe_date"],
                             errors="coerce")
print("")
print("labels cols:", list(hl.columns))
print("pe_date range:", hl["pdate"].min(),
      "to", hl["pdate"].max())

# ---- pull index visit start dates ----
want = set(hl["ivisit"].dropna().astype("int64"))
print("")
print("index visits wanted:", len(want))

vpath = os.path.join(INS, "visit_occurrence.csv")
hdr = pd.read_csv(vpath, nrows=2)
print("visit cols:", list(hdr.columns))

dcol = None
for c in ("visit_start_DATETIME", "visit_start_date",
          "visit_start_datetime"):
    if c in hdr.columns:
        dcol = c
        break
print("using date column:", dcol)

chunks = []
rd = pd.read_csv(vpath,
                 usecols=["visit_occurrence_id", dcol],
                 chunksize=500000)
for i, ch in enumerate(rd):
    ch = ch[ch["visit_occurrence_id"].isin(want)]
    if len(ch):
        chunks.append(ch)
    if (i + 1) % 5 == 0:
        print("  chunk", i + 1)
vis = pd.concat(chunks) if chunks else pd.DataFrame()
print("matched visits:", len(vis))

j = hl.merge(per, on="person_id", how="left")
if len(vis):
    vis = vis.rename(
        columns={"visit_occurrence_id": "ivisit"})
    vis["vdt"] = pd.to_datetime(vis[dcol],
                                errors="coerce")
    j = j.merge(vis[["ivisit", "vdt"]],
                on="ivisit", how="left")
else:
    j["vdt"] = pd.NaT

bd = pd.to_datetime(
    j["birth_DATETIME"], errors="coerce") \
    if "birth_DATETIME" in j.columns else pd.NaT

cands = {}
cands["pe_year_minus_yob"] = (
    j["pdate"].dt.year - j["year_of_birth"])
cands["visit_year_minus_yob"] = (
    j["vdt"].dt.year - j["year_of_birth"])
if not isinstance(bd, pd.Timestamp):
    cands["pe_minus_birthdt"] = (
        (j["pdate"] - bd).dt.days / 365.25)
    cands["visit_minus_birthdt"] = (
        (j["vdt"] - bd).dt.days / 365.25)

print("")
print("CANDIDATE DERIVATIONS (target median %.0f)"
      % TARGET)
for k, v in cands.items():
    v = pd.to_numeric(v, errors="coerce")
    n = int(v.notna().sum())
    if n == 0:
        print("  %-24s no values" % k)
        continue
    md = float(v.median())
    print("  %-24s median %.1f  (diff %+.1f)"
          "  mean %.1f  n=%d  <18 %d  >100 %d"
          % (k, md, md - TARGET, float(v.mean()),
             n, int((v < 18).sum()),
             int((v > 100).sum())))