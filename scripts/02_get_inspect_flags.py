"""Pull the four INSPECT condition flags from the
Redivis shahlab INSPECT EHR dataset, expanded through
concept_ancestor so descendant codes count. Writes one
row per person with afib, cancer, copd, heart_failure.
Run in venv_dl only.
"""
import os
import redivis
import pandas as pd

OUT = os.path.join("data", "raw", "inspect")
os.makedirs(OUT, exist_ok=True)
DEST = os.path.join(OUT, "inspect_flags.csv")

FLAGS = {
    "cancer": 443392,
    "heart_failure": 316139,
    "copd": 255573,
    "afib": 313217,
}

if not os.environ.get("REDIVIS_API_TOKEN"):
    raise SystemExit("REDIVIS_API_TOKEN not set")

ds = redivis.user("shahlab").dataset(
    "inspect_ehr:dzc6:v1_2")
print("dataset resolved")

frames = []
for name, anc in FLAGS.items():
    sql = """
    SELECT DISTINCT c.person_id
    FROM condition_occurrence c
    JOIN concept_ancestor a
      ON c.condition_concept_id =
         a.descendant_concept_id
    WHERE a.ancestor_concept_id = %d
    """ % anc
    print("querying", name)
    d = ds.query(sql).to_pandas_dataframe()
    d[name] = 1
    print("  persons:", len(d))
    frames.append(d.set_index("person_id"))

out = pd.concat(frames, axis=1)
out = out.fillna(0).astype(int)
out = out.reset_index()
out.to_csv(DEST, index=False)

print("")
print("saved", DEST, out.shape)
for name in FLAGS:
    print("  %-15s %d persons"
          % (name, int(out[name].sum())))