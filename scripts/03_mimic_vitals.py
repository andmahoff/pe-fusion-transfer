"""Rebuild the five MIMIC vitals that lived in
mimic_pe_ehr_baseline_v7.csv. Three-layer fallback:
ICU chartevents, ward omr blood pressure, ED triage.
Bounded per admission to charttime <= index dischtime.
Itemids and clip bounds match those used to build
the dissertation's v7 feature matrix.
Run in venv (analysis).
"""
import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

PROJ = os.environ.get("GCP_PROJECT_ID")
# folder holding the dissertation's derived files
DIS = os.environ.get(
    "PE_DISS_DIR", os.path.join("..", "Dissertation"))
FIG = os.path.join(DIS, "figwork", "data")
OUT = os.path.join("data", "processed")
os.makedirs(OUT, exist_ok=True)
DEST = os.path.join(OUT, "mimic_vitals_rebuilt.csv")

CHART = {
    "hr": [220045],
    "sbp": [220050, 220179],
    "dbp": [220051, 220180],
    "rr": [220210],
    "temp_f": [223761],
    "temp_c": [223762],
}

CLIP = {
    "hr": (20, 220), "sbp": (40, 260),
    "dbp": (20, 180), "rr": (4, 60),
    "temp": (28, 43),
}

cl = bigquery.Client(project=PROJ)
lab = pd.read_csv(
    os.path.join(FIG, "mimic_labels_harmonised.csv"))
adm = pd.read_csv(
    os.path.join(FIG, "index_admission_times.csv"))
adm["dischtime"] = pd.to_datetime(adm["dischtime"])

ix = lab[["subject_id", "hadm_id"]].merge(
    adm[["hadm_id", "dischtime"]],
    on="hadm_id", how="left")
ix = ix.dropna(subset=["dischtime"])
print("index admissions:", len(ix))

parts = []
for h, s, d in zip(ix["hadm_id"], ix["subject_id"],
                   ix["dischtime"]):
    parts.append(
        "STRUCT(%d AS hid, %d AS sid,"
        " DATETIME '%s' AS dt)"
        % (h, s, d.strftime("%Y-%m-%d %H:%M:%S")))
IDX = "SELECT * FROM UNNEST([\n%s\n])" % (
    ",\n".join(parts))

inv = {}
for k, v in CHART.items():
    for i in v:
        inv[i] = k
cs = ",".join(str(i) for i in inv)
ss = ",".join(str(s) for s in
              sorted(ix["subject_id"].unique()))

# ---------- layer 1: ICU chartevents ----------
q1 = """
WITH idx AS (%s)
SELECT i.hid AS hadm_id, c.itemid,
       AVG(c.valuenum) AS v
FROM `physionet-data.mimiciv_3_1_icu.chartevents` c
JOIN idx i ON c.subject_id = i.sid
WHERE c.subject_id IN (%s)
  AND c.itemid IN (%s)
  AND c.valuenum IS NOT NULL
  AND DATETIME(c.charttime) <= i.dt
GROUP BY i.hid, c.itemid
""" % (IDX, ss, cs)

print("querying chartevents ...")
r1 = cl.query(q1).to_dataframe()
print("  rows:", len(r1))
r1["name"] = r1["itemid"].map(inv)
icu = r1.groupby(["hadm_id", "name"])[
    "v"].mean().unstack().reset_index()
print("  admissions:", len(icu))

# ---------- layer 2: ward omr BP ----------
q2 = """
SELECT subject_id, chartdate, result_value
FROM `physionet-data.mimiciv_3_1_hosp.omr`
WHERE subject_id IN (%s)
  AND result_name = 'Blood Pressure'
""" % ss

print("querying omr ...")
try:
    r2 = cl.query(q2).to_dataframe()
    print("  rows:", len(r2))
except Exception as e:
    print("  omr failed:", e)
    r2 = pd.DataFrame(columns=[
        "subject_id", "chartdate", "result_value"])

if len(r2):
    r2["chartdate"] = pd.to_datetime(
        r2["chartdate"], errors="coerce")
    sp = r2["result_value"].astype(str).str.split(
        "/", n=1, expand=True)
    r2["sbp"] = pd.to_numeric(sp[0],
                              errors="coerce")
    r2["dbp"] = pd.to_numeric(sp[1],
                              errors="coerce")
    r2 = r2.dropna(subset=["sbp", "dbp"])
    j = r2.merge(ix, on="subject_id", how="inner")
    j = j[j["chartdate"] <= j["dischtime"]]
    ward = j.groupby("hadm_id")[
        ["sbp", "dbp"]].mean().reset_index()
    print("  ward BP admissions:", len(ward))
else:
    ward = pd.DataFrame(columns=["hadm_id"])

# ---------- layer 3: ED triage ----------
q3 = """
WITH idx AS (%s)
SELECT i.hid AS hadm_id,
       AVG(t.heartrate) AS hr,
       AVG(t.sbp) AS sbp,
       AVG(t.dbp) AS dbp,
       AVG(t.resprate) AS rr,
       AVG(t.temperature) AS temp_f
FROM `physionet-data.mimiciv_ed.triage` t
JOIN `physionet-data.mimiciv_ed.edstays` e
  ON t.stay_id = e.stay_id
JOIN idx i ON e.subject_id = i.sid
WHERE DATETIME(e.intime) <= i.dt
GROUP BY i.hid
""" % IDX

print("querying ED triage ...")
try:
    ed = cl.query(q3).to_dataframe()
    print("  rows:", len(ed))
except Exception as e:
    print("  ED failed:", e)
    ed = pd.DataFrame(columns=["hadm_id"])


def temp_fix(s):
    s = pd.to_numeric(s, errors="coerce")
    hot = s >= 50
    return s.where(~hot, (s - 32.0) * 5.0 / 9.0)


for d in (icu, ed):
    has_f = "temp_f" in d.columns
    has_c = "temp_c" in d.columns
    if has_f:
        d["temp"] = temp_fix(d["temp_f"])
        if has_c:
            d["temp"] = d["temp"].fillna(
                d["temp_c"])
    elif has_c:
        d["temp"] = d["temp_c"]

# ---------- coalesce ----------
out = lab[["subject_id", "hadm_id"]].copy()
VITALS = ["hr", "sbp", "dbp", "rr", "temp"]


def take(src, cols, tag):
    if not len(src):
        return pd.DataFrame(columns=["hadm_id"])
    keep = ["hadm_id"] + [c for c in cols
                          if c in src.columns]
    d = src[keep].copy()
    d.columns = ["hadm_id"] + [
        tag + "_" + c for c in d.columns[1:]]
    return d


out = out.merge(take(icu, VITALS, "a"),
                on="hadm_id", how="left")
out = out.merge(take(ward, ["sbp", "dbp"], "b"),
                on="hadm_id", how="left")
out = out.merge(take(ed, VITALS, "c"),
                on="hadm_id", how="left")

src_counts = {}
for v in VITALS:
    cols = [t + "_" + v for t in ("a", "b", "c")
            if t + "_" + v in out.columns]
    if not cols:
        out["mean_" + v] = np.nan
        src_counts[v] = {}
        continue
    s = out[cols[0]].copy()
    used = {cols[0]: int(s.notna().sum())}
    for c in cols[1:]:
        add = s.isna() & out[c].notna()
        used[c] = int(add.sum())
        s = s.fillna(out[c])
    lo, hi = CLIP[v]
    s = s.clip(lower=lo, upper=hi)
    out["mean_" + v] = s
    src_counts[v] = used

fin = out[["subject_id", "hadm_id"]
          + ["mean_" + v for v in VITALS]]
fin.to_csv(DEST, index=False)
print("")
print("saved", DEST, fin.shape)

REF = {"hr": 0.817, "sbp": 0.893, "dbp": 0.893,
       "rr": 0.815, "temp": 0.811}
print("")
print("COVERAGE vs the dissertation v7 matrix")
for v in VITALS:
    c = "mean_" + v
    n = fin[c].notna().sum()
    md = float(fin[c].median()) if n else float("nan")
    cov = fin[c].notna().mean()
    print("  %-6s %.3f  (v7 %.3f, diff %+.3f)"
          "  median %.1f"
          % (v, cov, REF[v], cov - REF[v], md))

print("")
print("SOURCE CONTRIBUTION (a=ICU b=ward c=ED)")
for v, d in src_counts.items():
    print("  %-8s %s" % (v, d))