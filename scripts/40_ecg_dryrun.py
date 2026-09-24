"""Ten-record dry run before the full download.
Checks path uniformity, that .dat files fetch,
that a record reads, and the actual per-record
size and rate.
Run in venv_ecg only.
"""
import os
import time
import posixpath
import numpy as np
import pandas as pd
import wfdb

# folder holding the dissertation's derived files
DIS = os.environ.get(
    "PE_DISS_DIR", os.path.join("..", "Dissertation"))
DSET = os.path.join(DIS, "Datasets")
OUT = os.path.join("data", "raw", "ecg")
DB = "mimic-iv-ecg"
N = 10
PTBXL = ["I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]

coh = pd.read_csv(
    os.path.join(DSET, "MIMICIV",
                 "mimic_pe_mace_cohort.csv"))
c = coh.dropna(subset=["ecg_path"]).copy()


def clean(p):
    s = str(p).replace("\\", "/").strip()
    for e in (".hea", ".dat"):
        if s.endswith(e):
            s = s[:-len(e)]
    return s.strip("/")


c["rec"] = c["ecg_path"].apply(clean)
recs = sorted(c["rec"].unique())

print("PATH UNIFORMITY (%d records)" % len(recs))
depth = pd.Series(
    [len(r_.split("/")) for r_ in recs])
print("  path depths:",
      depth.value_counts().to_dict())
pre = pd.Series(
    [r_.split("/")[0] for r_ in recs])
print("  first segment:",
      pre.value_counts().head(3).to_dict())
p2 = pd.Series(
    [r_.split("/")[1] if len(r_.split("/")) > 1
     else "" for r_ in recs])
print("  second segment: %d distinct, e.g. %s"
      % (p2.nunique(),
         list(p2.unique()[:5])))
odd = [r_ for r_ in recs
       if len(r_.split("/")) != 4
       or not r_.startswith("files/")]
print("  non-conforming:", len(odd))
for r_ in odd[:5]:
    print("     ", r_)

print("")
print("SPREAD OF THE TEST SAMPLE")
step = max(1, len(recs) // N)
test = recs[::step][:N]
for r_ in test:
    print("   ", r_)

print("")
print("DOWNLOADING %d RECORDS" % len(test))
t0 = time.time()
ok, bad = [], []
for r_ in test:
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
        h = os.path.join(dest, base + ".hea")
        d = os.path.join(dest, base + ".dat")
        if os.path.exists(h) and \
                os.path.exists(d):
            ok.append((r_,
                       os.path.getsize(h),
                       os.path.getsize(d)))
        else:
            bad.append((r_, "files not written"))
    except Exception as exc:
        bad.append((r_, repr(exc)[:110]))

el = time.time() - t0
print("")
print("succeeded: %d of %d" % (len(ok),
                               len(test)))
for r_, e in bad:
    print("  FAIL", r_, "->", e)

if ok:
    dat = np.mean([x[2] for x in ok])
    print("")
    print("  mean .hea size: %.0f bytes"
          % np.mean([x[1] for x in ok]))
    print("  mean .dat size: %.0f KB"
          % (dat / 1024))
    print("  rate: %.2f records/s" % (len(ok)
                                      / el))
    print("")
    print("  FULL DOWNLOAD ESTIMATE")
    print("    size:  %.0f MB"
          % (len(recs) * dat / 1e6))
    print("    time:  %.1f hours"
          % (len(recs) / max(len(ok) / el, 1e-9)
             / 3600))

if ok:
    r_ = ok[0][0]
    p = os.path.join(
        OUT, *r_.split("/")[:-1],
        r_.split("/")[-1])
    print("")
    print("READING", r_)
    rec = wfdb.rdrecord(p)
    s = rec.p_signal
    print("  shape:", s.shape, " fs:", rec.fs)
    print("  leads:", rec.sig_name)
    d = [i for i, (a, b) in enumerate(
        zip(rec.sig_name, PTBXL)) if a != b]
    print("  differs from PTB-XL at:", d,
          "(expect [4, 5])")
    print("  NaNs: %d of %d"
          % (int(np.isnan(s).sum()), s.size))
    print("  per-lead SD:",
          np.round(np.nanstd(s, axis=0), 3))

    print("")
    print("  CONSISTENCY ACROSS ALL %d"
          % len(ok))
    fs_, sh, lead = [], [], []
    for r2, _, _ in ok:
        q = os.path.join(
            OUT, *r2.split("/")[:-1],
            r2.split("/")[-1])
        try:
            h = wfdb.rdheader(q)
            fs_.append(h.fs)
            sh.append((h.sig_len, h.n_sig))
            lead.append(tuple(h.sig_name))
        except Exception as exc:
            print("    unreadable:", r2,
                  repr(exc)[:60])
    print("    fs:",
          pd.Series(fs_).value_counts()
          .to_dict())
    print("    shapes:",
          pd.Series(sh).value_counts()
          .to_dict())
    print("    distinct lead orders:",
          len(set(lead)))
    if len(set(lead)) > 1:
        print("    WARNING: lead order varies,"
              " the swap must be per-record")

print("")
if len(ok) == len(test) and not bad:
    print("ALL CHECKS PASSED - run the full"
          " download")
else:
    print("PROBLEMS FOUND - do not run the"
          " full download yet")