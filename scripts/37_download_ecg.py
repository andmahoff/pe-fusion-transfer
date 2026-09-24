"""Download only the MIMIC-IV-ECG records for the
PE cohort, using the ecg_path column already in
mimic_pe_mace_cohort.csv. Roughly 6,000 records
at about 120 KB each, so under 1 GB.

Writes a single directory tree, so records are
not split between the files/ and physionet.org/
layouts.
Resumable: rerun to continue after interruption.
Run in venv_ecg only.
"""
import os
import sys
import time
import posixpath
import pandas as pd
import wfdb

# folder holding the dissertation's derived files
DIS = os.environ.get(
    "PE_DISS_DIR", os.path.join("..", "Dissertation"))
DSET = os.path.join(DIS, "Datasets")
OUT = os.path.join("data", "raw", "ecg")
DB = "mimic-iv-ecg"
os.makedirs(OUT, exist_ok=True)


def normalize(p):
    """Strip the files/ or physionet.org/ prefix
    so every path is relative to the database
    root, e.g. pXXXX/pXXXXXXXX/sXXXXXXXX/XXXXXXXX
    """
    s = str(p).replace("\\", "/").strip()
    for pre in ("physionet.org/files/"
                "mimic-iv-ecg/1.0/",
                "physionet.org/files/"
                "mimic-iv-ecg/",
                "physionet.org/", "files/"):
        if s.startswith(pre):
            s = s[len(pre):]
    for ext in (".hea", ".dat"):
        if s.endswith(ext):
            s = s[:-len(ext)]
    return s.strip("/")


coh = pd.read_csv(
    os.path.join(DSET, "MIMICIV",
                 "mimic_pe_mace_cohort.csv"))
print("cohort rows:", len(coh))
print("columns:", list(coh.columns))

if "ecg_path" not in coh.columns:
    raise SystemExit("no ecg_path column")

c = coh.dropna(subset=["ecg_path"]).copy()
c["rec"] = c["ecg_path"].apply(normalize)
recs = sorted(c["rec"].unique())
print("")
print("records with a path:", len(c))
print("unique records:", len(recs))
print("sample paths:")
for r_ in recs[:3]:
    print("   ", r_)

est = len(recs) * 0.125
print("")
print("estimated download: %.0f MB" % est)

todo = []
for r_ in recs:
    d = os.path.join(OUT, *r_.split("/")[:-1])
    b = r_.split("/")[-1]
    if not (os.path.exists(
            os.path.join(d, b + ".hea"))
            and os.path.exists(
                os.path.join(d, b + ".dat"))):
        todo.append(r_)

print("already present:", len(recs) - len(todo))
print("to download:", len(todo))
if not todo:
    print("nothing to do")
    sys.exit(0)

t0 = time.time()
ok, bad = 0, []
for i, r_ in enumerate(todo):
    sub = posixpath.dirname(r_)
    base = posixpath.basename(r_)
    dest = os.path.join(OUT, *sub.split("/"))
    os.makedirs(dest, exist_ok=True)
    try:
        wfdb.dl_files(
            DB, dest,
            [sub + "/" + base + ".hea",
             sub + "/" + base + ".dat"],
            keep_subdirs=False)
        ok += 1
    except Exception as exc:
        bad.append((r_, repr(exc)[:120]))
    if (i + 1) % 100 == 0:
        el = time.time() - t0
        rate = (i + 1) / max(el, 1e-9)
        print("  %d/%d  %.1f/s  eta %.0f min"
              % (i + 1, len(todo), rate,
                 (len(todo) - i - 1)
                 / max(rate, 1e-9) / 60))

print("")
print("downloaded:", ok)
print("failed:", len(bad))
for r_, e in bad[:10]:
    print("  ", r_, "->", e)

tot = 0
for root, _, fs in os.walk(OUT):
    for fn in fs:
        tot += os.path.getsize(
            os.path.join(root, fn))
print("")
print("on disk: %.0f MB" % (tot / 1e6))

idx = c[["subject_id", "hadm_id",
         "ecg_study_id", "ecg_charttime",
         "rec"]].drop_duplicates()
idx.to_csv(os.path.join(
    OUT, "ecg_record_index.csv"), index=False)
print("saved", os.path.join(
    OUT, "ecg_record_index.csv"), idx.shape)