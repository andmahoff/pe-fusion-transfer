"""Extended ECG measurements, v4.

Adds four families the current set lacks:
  1 DANIEL SCORE  the validated PE-specific ECG
                  prognostic instrument: HR>100,
                  S1Q3T3, complete RBBB,
                  T-inversion V1-V4, ST
                  elevation in aVR, and atrial
                  fibrillation. Meta-analysis of
                  3,007 PE patients found all six
                  associated with collapse and
                  30-day mortality.
  2 RV STRAIN     the composite of RBBB, S1Q3T3
                  or negative T in V1-V4,
                  HR 2.58 independent of echo.
  3 REPOLAR       T axis, frontal QRS-T angle,
                  Tp-e interval and Tp-e/QT
                  ratio, ST deviation in every
                  lead including aVR.
  4 POINCARE      SD1, SD2, ratio, area and
                  approximate entropy. Valid at
                  10 s per the ultra-short HRV
                  literature.

Frequency-domain HRV and multiscale entropy are
not computed: both need 2-5 minutes.

AFIB and RBBB come from the SCP logits, which
already detect them, rather than being
re-derived from the waveform.
Run in venv_ecg only.

OUTPUT FILES
  data\\processed\\ecg_derived_v4_record.csv
  data\\processed\\ecg_derived_v4.csv
  results\\ecg_v4_log.txt
"""
import os
import time
import numpy as np
import pandas as pd
import wfdb
from scipy import signal as sg

OUT = os.path.join("data", "raw", "ecg")
PROC = os.path.join("data", "processed")
REC_DEST = os.path.join(
    PROC, "ecg_derived_v4_record.csv")
DEST = os.path.join(PROC, "ecg_derived_v4.csv")

FS = 500
# MIMIC order: I II III aVR aVF aVL V1..V6
L_I, L_II, L_III = 0, 1, 2
L_AVR, L_AVF, L_AVL = 3, 4, 5
V = {i: 6 + i for i in range(6)}


def norm_rec(r):
    s = str(r).replace("\\", "/").strip("/")
    return s if s.startswith("files/") \
        else "files/" + s


def rp(r):
    return os.path.join(
        OUT, *norm_rec(r).split("/"))


def bp(x, lo, hi, fs=FS, order=3):
    b, a = sg.butter(
        order, [lo / (fs / 2), hi / (fs / 2)],
        btype="band")
    return sg.filtfilt(b, a, x)


def detect_r(x, fs=FS):
    f = bp(x, 5.0, 15.0, fs)
    d = np.diff(f, prepend=f[0])
    w = int(0.15 * fs)
    i = np.convolve(d ** 2,
                    np.ones(w) / w,
                    mode="same")
    thr = np.mean(i) + 0.5 * np.std(i)
    pk, _ = sg.find_peaks(
        i, height=thr,
        distance=int(0.25 * fs))
    return pk


def poincare(rr):
    if len(rr) < 4:
        return {}
    a, b = rr[:-1], rr[1:]
    sd1 = float(np.std((a - b) / np.sqrt(2.0),
                       ddof=1))
    sd2 = float(np.std((a + b) / np.sqrt(2.0),
                       ddof=1))
    d = {"pc_sd1": sd1, "pc_sd2": sd2,
         "pc_area": float(np.pi * sd1 * sd2)}
    if sd2 > 1e-9:
        d["pc_ratio"] = sd1 / sd2
    return d


def apen(x, m=2, rf=0.2):
    n = len(x)
    if n < m + 3:
        return np.nan
    r = rf * np.std(x, ddof=1)
    if r <= 0:
        return np.nan

    def phi(mm):
        z = np.array([x[i:i + mm]
                      for i in range(
                          n - mm + 1)])
        c = [np.mean(np.max(
            np.abs(z - z[i]), axis=1) <= r)
            for i in range(len(z))]
        c = np.array(c)
        c = c[c > 0]
        return (float(np.mean(np.log(c)))
                if len(c) else np.nan)
    a, b = phi(m), phi(m + 1)
    return (float(a - b)
            if np.isfinite(a)
            and np.isfinite(b) else np.nan)


def median_beat(sig, pk, pre, post):
    seg = [sig[p - pre:p + post] for p in pk
           if p - pre >= 0
           and p + post < sig.shape[0]]
    if len(seg) < 3:
        return None
    return np.median(np.stack(seg), axis=0)


def axis_of(x, y):
    return float(np.degrees(np.arctan2(y, x)))


def process(sig, fs=FS):
    d = {}
    ii = sig[:, L_II]
    pk = detect_r(ii, fs)
    if len(pk) < 3:
        return d

    rr = np.diff(pk) / fs * 1000.0
    rr = rr[(rr > 300) & (rr < 2000)]
    hr = np.nan
    if len(rr) >= 3:
        hr = 60000.0 / np.mean(rr)
        d["hr_mean"] = hr
        d["rmssd"] = float(np.sqrt(
            np.mean(np.diff(rr) ** 2)))
        d["rr_sd"] = float(np.std(rr, ddof=1))
        d.update(poincare(rr))
        d["apen"] = apen(rr)
        d["n_beats"] = int(len(pk))
    d["dan_hr100"] = (int(hr > 100)
                      if np.isfinite(hr) else 0)

    pre, post = int(0.25 * fs), int(0.45 * fs)
    mb = median_beat(sig, pk, pre, post)
    if mb is None:
        return d
    mb = mb - np.median(
        mb[:int(0.06 * fs)], axis=0)
    r = pre

    # ---- QRS boundaries on lead II ----
    amp = np.abs(mb[:, L_II])
    thr = 0.1 * max(amp[r], 1e-6)
    q = r
    while q > 1 and amp[q] > thr:
        q -= 1
    s_ = r
    while s_ < len(mb) - 2 and amp[s_] > thr:
        s_ += 1
    d["qrs_ms"] = (s_ - q) / fs * 1000.0

    # ---- axes and QRS-T angle ----
    w = int(0.05 * fs)
    qv = mb[max(0, r - w):r + w].sum(axis=0)
    d["qrs_axis"] = axis_of(qv[L_I], qv[L_AVF])
    tl = r + int(0.16 * fs)
    th = min(len(mb), r + int(0.42 * fs))
    tv = None
    if th - tl > 10:
        seg = mb[tl:th]
        tp = int(np.argmax(
            np.abs(seg[:, L_II])))
        tv = seg[tp]
        d["t_axis"] = axis_of(tv[L_I],
                              tv[L_AVF])
        a_ = abs(d["qrs_axis"] - d["t_axis"])
        d["qrst_angle"] = float(
            min(a_, 360.0 - a_))
        # Tp-e: T peak to T end, lead II
        tail = np.abs(seg[tp:, L_II])
        te = tp
        if len(tail) > 3 and tail[0] > 0:
            k = tp
            while (k < len(seg) - 2
                   and np.abs(seg[k, L_II])
                   > 0.1 * tail[0]):
                k += 1
            te = k
        d["tpe_ms"] = ((te - tp) / fs
                       * 1000.0)
        qt = (tl + te - q) / fs * 1000.0
        d["qt_ms"] = qt
        if qt > 0:
            d["tpe_qt"] = d["tpe_ms"] / qt
        if np.isfinite(hr) and hr > 0:
            d["qtc_hodges"] = qt + 1.75 * (
                hr - 60.0)
            d["qtc_over451"] = int(
                d["qtc_hodges"] > 451)
        for k_, ch in V.items():
            d["t_v%d" % (k_ + 1)] = float(
                tv[ch])
        d["t_wave_iii"] = float(tv[L_III])
        d["dan_twi_v14"] = int(all(
            tv[V[k_]] < 0 for k_ in range(4)))
        d["n_twi_v14"] = int(sum(
            1 for k_ in range(4)
            if tv[V[k_]] < 0))

    # ---- ST deviation, all leads incl aVR ----
    j60 = s_ + int(0.06 * fs)
    if j60 < len(mb):
        st = mb[j60]
        for nm, ch in (("i", L_I), ("ii", L_II),
                       ("iii", L_III),
                       ("avr", L_AVR),
                       ("avf", L_AVF),
                       ("avl", L_AVL)):
            d["st_" + nm] = float(st[ch])
        for k_, ch in V.items():
            d["st_v%d" % (k_ + 1)] = float(
                st[ch])
        d["st_max"] = float(np.max(st))
        d["st_min"] = float(np.min(st))
        d["dan_ste_avr"] = int(
            st[L_AVR] > 0.05)

    # ---- S1Q3T3 ----
    wq = int(0.06 * fs)
    s1 = float(np.min(
        mb[max(0, r - 5):min(len(mb),
                             r + wq), L_I]))
    q3 = float(np.min(
        mb[max(0, r - wq):r + 5, L_III]))
    t3 = d.get("t_wave_iii", 0.0)
    d["s_wave_i"] = s1
    d["q_wave_iii"] = q3
    d["dan_s1q3t3"] = int(
        (s1 < -0.15) and (q3 < -0.15)
        and (t3 < 0))

    # ---- RBBB from morphology ----
    rv1 = float(np.max(
        mb[max(0, r - wq):min(len(mb),
                              r + wq), V[0]]))
    sv1 = float(np.min(
        mb[max(0, r - wq):min(len(mb),
                              r + wq), V[0]]))
    d["r_v1"] = rv1
    d["s_v1"] = sv1
    if abs(sv1) > 1e-6:
        d["rs_ratio_v1"] = rv1 / abs(sv1)
    d["dan_rbbb"] = int(
        (d["qrs_ms"] > 120)
        and (d.get("rs_ratio_v1", 0) > 0.5))
    d["rad"] = int(d["qrs_axis"] > 90)

    # ---- composites ----
    dan = ["dan_hr100", "dan_s1q3t3",
           "dan_rbbb", "dan_twi_v14",
           "dan_ste_avr"]
    d["daniel_partial"] = int(sum(
        d.get(k_, 0) for k_ in dan))
    d["rv_strain"] = int(
        d.get("dan_rbbb", 0)
        or d.get("dan_s1q3t3", 0)
        or d.get("dan_twi_v14", 0))

    for i in range(sig.shape[1]):
        d["sd_l%d" % i] = float(
            np.nanstd(sig[:, i]))
    return d


idx = pd.read_csv(os.path.join(
    OUT, "ecg_record_index.csv"))
idx["rec"] = idx["rec"].apply(norm_rec)
recs = sorted(idx["rec"].unique())
print("records:", len(recs), flush=True)

rows, bad = [], 0
t0 = time.time()
for i, r_ in enumerate(recs):
    try:
        rec = wfdb.rdrecord(rp(r_))
        s = np.nan_to_num(rec.p_signal,
                          nan=0.0)
        d = process(s)
        d["rec"] = r_
        rows.append(d)
    except Exception as exc:
        bad += 1
        if bad <= 3:
            print("  FAIL", r_,
                  repr(exc)[:130])
    if (i + 1) % 1000 == 0:
        el = time.time() - t0
        print("  %d/%d  eta %.0f min"
              % (i + 1, len(recs),
                 (len(recs) - i - 1)
                 / max((i + 1) / el, 1e-9)
                 / 60), flush=True)

d = pd.DataFrame(rows)
print("")
print("processed:", len(d), " failed:", bad)
d.to_csv(REC_DEST, index=False)

# add AFIB from the SCP logits for the full
# six-component Daniel score
lg = os.path.join(
    PROC, "exp0_logits_record.csv")
if os.path.exists(lg):
    q = pd.read_csv(lg,
                    usecols=["rec",
                             "scp_AFIB"])
    d = d.merge(q, on="rec", how="left")
    d["dan_afib"] = (
        d["scp_AFIB"] > 0).astype(int)
    dan = ["dan_hr100", "dan_s1q3t3",
           "dan_rbbb", "dan_twi_v14",
           "dan_ste_avr", "dan_afib"]
    d["daniel_score"] = d[dan].sum(axis=1)
    d = d.drop(columns=["scp_AFIB"])
    d.to_csv(REC_DEST, index=False)
    print("Daniel score computed with AFIB")

m = idx[["subject_id", "hadm_id",
         "rec"]].drop_duplicates()
j = m.merge(d, on="rec", how="inner")
num = [c for c in j.columns
       if c not in ("rec", "subject_id",
                    "hadm_id")]
g = j.groupby(["subject_id", "hadm_id"])
agg = g[num].agg(["mean", "max"])
agg.columns = ["%s_%s" % (a, b)
               for a, b in agg.columns]
agg = agg.reset_index()
agg.to_csv(DEST, index=False)
print("saved", DEST, agg.shape)

print("")
print("DANIEL SCORE COMPONENTS")
print("  literature prevalence in acute PE:")
print("    HR>100 ~40%, S1Q3T3 ~10-20%,")
print("    RBBB ~5-15%, TWI V1-V4 ~20-40%,")
print("    ST elev aVR ~10-20%, AF ~5-15%")
print("")
for c in ["dan_hr100", "dan_s1q3t3",
          "dan_rbbb", "dan_twi_v14",
          "dan_ste_avr", "dan_afib",
          "rv_strain"]:
    if c in d.columns:
        print("  %-14s %.3f"
              % (c, d[c].dropna().mean()))
if "daniel_score" in d.columns:
    print("")
    print("  daniel_score distribution:")
    print(d["daniel_score"].value_counts()
          .sort_index().to_string())

print("")
print("NEW CONTINUOUS FEATURES")
for c, lo, hi in [("pc_sd1", 5, 100),
                  ("pc_sd2", 10, 200),
                  ("pc_ratio", 0.1, 1.5),
                  ("qrst_angle", 0, 180),
                  ("t_axis", -90, 180),
                  ("tpe_ms", 40, 140),
                  ("tpe_qt", 0.1, 0.4),
                  ("rs_ratio_v1", 0, 3),
                  ("st_avr", -0.3, 0.3),
                  ("apen", 0, 2.5)]:
    if c not in d.columns:
        continue
    v = d[c].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if not len(v):
        continue
    print("  %-13s n=%4d  median %7.2f"
          "  %4.1f%% in %g-%g"
          % (c, len(v), v.median(),
             100 * float(((v >= lo)
                          & (v <= hi)).mean()),
             lo, hi))