"""Download the four INSPECT metadata TSVs from the
Redivis aimi dataset 2n96, table full:q80g. These are
the CTPA impressions plus their labels, official splits
and study mapping. Run in venv_dl only.
"""
import os
import redivis

OUT = os.path.join("data", "raw", "inspect")
os.makedirs(OUT, exist_ok=True)

FILES = [
    "impressions_20250611.tsv",
    "labels_20250611.tsv",
    "splits_20250611.tsv",
    "study_mapping_20250611.tsv",
]

if not os.environ.get("REDIVIS_API_TOKEN"):
    raise SystemExit("REDIVIS_API_TOKEN not set")

ds = redivis.user("aimi").dataset("inspect:2n96")
tb = ds.table("full:q80g")
print("dataset resolved")

for name in FILES:
    dest = os.path.join(OUT, name)
    if os.path.exists(dest):
        print("skip (exists):", name)
        continue
    print("downloading", name)
    tb.file(name).download(path=OUT)
    mb = os.path.getsize(dest) / 1e6
    print("  saved %.2f MB" % mb)

print("")
for name in FILES:
    p = os.path.join(OUT, name)
    ok = "OK" if os.path.exists(p) else "MISSING"
    print("%-32s %s" % (name, ok))
print("done")