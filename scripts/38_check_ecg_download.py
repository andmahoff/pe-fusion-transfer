"""Verify the ECG download before building the
feature extractor. Checks record count, cohort
coverage, and that one record reads with the
expected shape, sampling rate and lead order.
Run in venv_ecg only.
"""
import os
import numpy as np
import pandas as pd
import wfdb

OUT = os.path.join("data", "raw", "ecg")

# PTB-XL training order
PTBXL = ["I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]

heas, dats = [], []
for root, _, fs in os.walk(OUT):
    for fn in fs:
        if fn.endswith(".hea"):
            heas.append(os.path.join(root, fn))
        elif fn.endswith(".dat"):
            dats.append(os.path.join(root, fn))

print("header files:", len(heas))
print("signal files:", len(dats))

tot = sum(os.path.getsize(p)
          for p in heas + dats)
print("on disk: %.0f MB" % (tot / 1e6))

idx_fp = os.path.join(OUT,
                      "ecg_record_index.csv")
if os.path.exists(idx_fp):
    idx = pd.read_csv(idx_fp)
    print("")
    print("index rows:", len(idx))
    print("unique records:",
          idx["rec"].nunique())
    print("unique admissions:",
          idx["hadm_id"].nunique())

    have = set()
    for p in dats:
        have.add(os.path.splitext(
            os.path.basename(p))[0])
    want = set(idx["rec"].astype(str)
               .str.split("/").str[-1])
    print("")
    print("wanted:", len(want))
    print("downloaded:", len(have & want))
    print("missing:", len(want - have))
    miss = sorted(want - have)[:5]
    if miss:
        print("  e.g.", miss)

    got = idx[idx["rec"].astype(str)
              .str.split("/").str[-1]
              .isin(have)]
    print("")
    print("admissions with >=1 record:",
          got["hadm_id"].nunique())
    print("records per admission:")
    print(got.groupby("hadm_id").size()
          .value_counts().head(6).to_string())

print("")
print("=" * 58)
print("READING ONE RECORD")
print("=" * 58)
if not dats:
    raise SystemExit("no signal files found")

p = os.path.splitext(dats[0])[0]
print("record:", p)
r = wfdb.rdrecord(p)
print("  shape:", r.p_signal.shape)
print("  fs:", r.fs, "Hz")
print("  duration: %.1f s"
      % (r.p_signal.shape[0] / r.fs))
print("  leads:", r.sig_name)
print("  units:", r.units[:3])

print("")
print("LEAD ORDER CHECK")
print("  PTB-XL:", PTBXL)
print("  MIMIC :", list(r.sig_name))
if list(r.sig_name) == PTBXL:
    print("  IDENTICAL - no swap needed")
else:
    d = [i for i, (a, b) in enumerate(
        zip(r.sig_name, PTBXL)) if a != b]
    print("  differs at positions:", d)
    if d == [4, 5]:
        print("  CONFIRMED: aVL and aVF are"
              " swapped, as recorded")
    else:
        print("  UNEXPECTED - the reorder needs"
              " rechecking before extraction")

print("")
print("SIGNAL SANITY")
s = r.p_signal
print("  NaNs: %d of %d"
      % (int(np.isnan(s).sum()), s.size))
print("  range: %.3f to %.3f"
      % (float(np.nanmin(s)),
         float(np.nanmax(s))))
print("  per-lead SD:",
      np.round(np.nanstd(s, axis=0), 3))
flat = int((np.nanstd(s, axis=0) < 1e-6).sum())
print("  flat leads:", flat)

print("")
print("SAMPLING RATE ACROSS 20 RECORDS")
fs, shp, bad = [], [], 0
for q in dats[:20]:
    try:
        h = wfdb.rdheader(
            os.path.splitext(q)[0])
        fs.append(h.fs)
        shp.append((h.sig_len, h.n_sig))
    except Exception:
        bad += 1
print("  fs values:",
      pd.Series(fs).value_counts().to_dict())
print("  shapes:",
      pd.Series(shp).value_counts()
      .head(3).to_dict())
print("  unreadable:", bad)