"""Find the correct MIMIC-IV-ECG record path by
testing candidate prefixes against the server.
Run in venv_ecg only.
"""
import os
import pandas as pd
import wfdb

# folder holding the dissertation's derived files
DIS = os.environ.get(
    "PE_DISS_DIR", os.path.join("..", "Dissertation"))
DSET = os.path.join(DIS, "Datasets")
DB = "mimic-iv-ecg"
TMP = os.path.join("data", "raw", "ecg_probe")
os.makedirs(TMP, exist_ok=True)

coh = pd.read_csv(
    os.path.join(DSET, "MIMICIV",
                 "mimic_pe_mace_cohort.csv"))
raw = coh["ecg_path"].dropna().iloc[0]
print("raw ecg_path from cohort:")
print("   ", repr(raw))

s = str(raw).replace("\\", "/").strip()
for e in (".hea", ".dat"):
    if s.endswith(e):
        s = s[:-len(e)]
stem = s.split("/")
print("path parts:", stem)

tail = "/".join(stem[-3:])
print("last three parts:", tail)

CANDS = [
    "files/" + tail,
    tail,
    "1.0/files/" + tail,
    s if not s.startswith("physionet") else None,
    s.replace("physionet.org/files/"
              "mimic-iv-ecg/1.0/", ""),
]
CANDS = [c for c in CANDS if c]
seen, uniq = set(), []
for c in CANDS:
    if c not in seen:
        seen.add(c)
        uniq.append(c)

print("")
print("=" * 58)
print("TESTING CANDIDATE PATHS")
print("=" * 58)
good = None
for c in uniq:
    sub = "/".join(c.split("/")[:-1])
    base = c.split("/")[-1]
    print("")
    print("  trying:", c)
    try:
        wfdb.dl_files(
            DB, TMP,
            [sub + "/" + base + ".hea"],
            keep_subdirs=False)
        fp = os.path.join(TMP, base + ".hea")
        if os.path.exists(fp):
            print("    SUCCESS")
            good = c
            break
        print("    no file written")
    except Exception as exc:
        print("    fail:", repr(exc)[:110])

print("")
print("=" * 58)
if good:
    print("WORKING PREFIX FOUND")
    print("  full path:", good)
    pre = good[:len(good) - len(tail)]
    print("  prefix to prepend:", repr(pre))
    print("")
    print("Now reading the header ...")
    base = good.split("/")[-1]
    h = wfdb.rdheader(
        os.path.join(TMP, base))
    print("  fs:", h.fs, "Hz")
    print("  samples:", h.sig_len)
    print("  leads:", h.n_sig, h.sig_name)
else:
    print("NO CANDIDATE WORKED")
    print("")
    print("Listing the database root instead:")
    try:
        recs = wfdb.get_record_list(DB)
        print("  records found:", len(recs))
        print("  first five:")
        for r_ in recs[:5]:
            print("   ", r_)
    except Exception as exc:
        print("  failed:", repr(exc)[:200])