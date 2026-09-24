"""Derived physiological measurements from the
ECG waveforms: rate variability, interval
durations and frontal axis. These are quantities
the 71 SCP statements do not encode.

The record index was written without the files/
prefix while the download kept it, so rec_path
restores it.
Run in venv_ecg only.
"""
import os
import time
import numpy as np
import pandas as pd
import wfdb
from scipy import signal as sg

OUT = os.path.join("data", "raw", "ecg")
PROC = os.path.join("data", "processed")
os.makedirs(PROC, exist_ok=True)
DEST = os.path.join(PROC, "ecg_derived.csv")

FS = 500
LEAD_I, LEAD_II, LEAD_AVF = 0, 1, 4


def norm_rec(r):
    """Canonical record id, always starting
    files/ ."""
    s = str(r).replace("\\", "/").strip("/")
    if not s.startswith("files/"):
        s = "files/" + s
    return s


def rec_path(r):
    return os.path.join(
        OUT, *norm_rec(r).split("/"))


def detect_r(x, fs=FS):
    b, a = sg.butter(3, [5.0 / (fs / 2),
                         15.0 / (fs / 2)],
                     btype="band")
    f = sg.filtfilt(b, a, x)
    d = np.diff(f, prepend=f[0])
    s = d ** 2
    w = max(1, int(0.15 * fs))
    i = np.convolve(s, np.ones(w) / w,
                    mode="same")
    thr = np.mean(i) + 0.5 * np.std(i)
    pk, _ = sg.find_peaks(
        i, height=thr, distance=int(0.25 * fs))
    return pk


def hrv(pk, fs=FS):
    if len(pk) < 4:
        return {}
    rr = np.diff(pk) / fs * 1000.0
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 3:
        return {}
    dr = np.diff(rr)
    return {"hr_mean": 60000.0 / np.mean(rr),
            "rr_mean": float(np.mean(rr)),
            "rr_sd": float(np.std(rr)),
            "rmssd": float(np.sqrt(
                np.mean(dr ** 2))),
            "pnn50": float(
                (np.abs(dr) > 50).mean()),
            "rr_cv": float(
                np.std(rr)
                / max(np.mean(rr), 1e-9)),
            "rr_range": float(rr.max()
                              - rr.min()),
            "n_beats": int(len(pk))}


def intervals(x, pk, fs=FS):
    if len(pk) < 3:
        return {}
    pre, post = int(0.25 * fs), int(0.45 * fs)
    seg = [x[p - pre:p + post] for p in pk
           if p - pre >= 0
           and p + post < len(x)]
    if len(seg) < 3:
        return {}
    m = np.median(np.vstack(seg), axis=0)
    m = m - np.median(m[:int(0.05 * fs)])
    r = pre
    amp = np.abs(m)
    if amp[r] <= 0:
        return {}
    thr = 0.1 * amp[r]
    q = r
    while q > 0 and amp[q] > thr:
        q -= 1
    s_ = r
    while s_ < len(m) - 1 and amp[s_] > thr:
        s_ += 1
    out = {"qrs_ms": float(
        (s_ - q) / fs * 1000.0),
        "r_amp": float(m[r])}
    tail = m[s_:]
    if len(tail) < 20:
        return out
    ta = np.abs(tail)
    t_pk = int(np.argmax(ta))
    if ta[t_pk] <= 0:
        return out
    t_end = t_pk
    while (t_end < len(tail) - 1
           and ta[t_end] > 0.1 * ta[t_pk]):
        t_end += 1
    out["qt_ms"] = float(
        (s_ + t_end - q) / fs * 1000.0)
    out["t_amp"] = float(tail[t_pk])
    return out


def axis(sig, pk, fs=FS):
    if len(pk) < 3:
        return {}
    w = int(0.05 * fs)
    net = {}
    for nm, ch in (("i", LEAD_I),
                   ("avf", LEAD_AVF)):
        v = [np.sum(sig[p - w:p + w, ch])
             for p in pk
             if p - w >= 0
             and p + w < sig.shape[0]]
        if not v:
            return {}
        net[nm] = float(np.median(v))
    return {"qrs_axis": float(np.degrees(
        np.arctan2(net["avf"], net["i"]))),
        "net_i": net["i"],
        "net_avf": net["avf"]}


def amplitudes(sig):
    d = {}
    for i in range(sig.shape[1]):
        v = sig[:, i]
        d["sd_l%d" % i] = float(np.nanstd(v))
        d["ptp_l%d" % i] = float(
            np.nanmax(v) - np.nanmin(v))
    return d


idx = pd.read_csv(os.path.join(
    OUT, "ecg_record_index.csv"))
idx["rec"] = idx["rec"].apply(norm_rec)
idx.to_csv(os.path.join(
    OUT, "ecg_record_index.csv"), index=False)
print("index rewritten with files/ prefix")

recs = sorted(idx["rec"].astype(str).unique())
print("records to process:", len(recs))
print("example path:", rec_path(recs[0]))
print("  .hea exists:",
      os.path.exists(rec_path(recs[0])
                     + ".hea"))
print("  .dat exists:",
      os.path.exists(rec_path(recs[0])
                     + ".dat"))

rows, bad, shown = [], 0, 0
t0 = time.time()
for i, r_ in enumerate(recs):
    try:
        rec = wfdb.rdrecord(rec_path(r_))
        s = rec.p_signal
        if s is None or s.shape[1] < 12:
            raise ValueError("bad shape")
        s = np.nan_to_num(s, nan=0.0)
        ii = s[:, LEAD_II]
        pk = detect_r(ii)
        d = {"rec": r_}
        d.update(hrv(pk))
        d.update(intervals(ii, pk))
        d.update(axis(s, pk))
        d.update(amplitudes(s))
        if "qt_ms" in d and "rr_mean" in d:
            rr_s = d["rr_mean"] / 1000.0
            if rr_s > 0:
                d["qtc_bazett"] = (
                    d["qt_ms"] / np.sqrt(rr_s))
                d["qtc_fridericia"] = (
                    d["qt_ms"]
                    / (rr_s ** (1.0 / 3.0)))
        rows.append(d)
    except Exception as exc:
        bad += 1
        if shown < 5:
            shown += 1
            print("  FAIL", r_)
            print("    path:", rec_path(r_))
            print("    err :", repr(exc)[:200])
    if (i + 1) % 500 == 0:
        el = time.time() - t0
        rate = (i + 1) / max(el, 1e-9)
        print("  %d/%d  ok %d  fail %d"
              "  %.0f/s  eta %.0f min"
              % (i + 1, len(recs), len(rows),
                 bad, rate,
                 (len(recs) - i - 1)
                 / max(rate, 1e-9) / 60),
              flush=True)

print("")
print("processed:", len(rows), " failed:", bad)
if not rows:
    raise SystemExit(
        "nothing processed - see the errors")

d = pd.DataFrame(rows)
d.to_csv(os.path.join(
    PROC, "ecg_derived_record.csv"),
    index=False)

m = idx[["subject_id", "hadm_id",
         "rec"]].drop_duplicates()
j = m.merge(d, on="rec", how="inner")
print("merged to admissions:", j.shape)

num = [c for c in j.columns
       if c not in ("rec", "subject_id",
                    "hadm_id")]
g = j.groupby(["subject_id", "hadm_id"])
agg = g[num].agg(["mean", "max", "min"])
agg.columns = ["%s_%s" % (a, b)
               for a, b in agg.columns]
agg = agg.reset_index()
agg["n_ecg"] = g.size().values
agg.to_csv(DEST, index=False)
print("saved", DEST, agg.shape)

print("")
print("KEY MEASUREMENTS (record level)")
for c in ["hr_mean", "rmssd", "pnn50",
          "qrs_ms", "qt_ms", "qtc_bazett",
          "qrs_axis", "n_beats"]:
    if c in d.columns:
        v = d[c].replace(
            [np.inf, -np.inf], np.nan).dropna()
        if len(v):
            print("  %-16s n=%4d  median %8.1f"
                  "  IQR %.1f to %.1f"
                  % (c, len(v), v.median(),
                     v.quantile(0.25),
                     v.quantile(0.75)))

print("")
print("PLAUSIBILITY")
for c, lo, hi, u in [
        ("hr_mean", 40, 140, "bpm"),
        ("qrs_ms", 60, 140, "ms"),
        ("qtc_bazett", 350, 500, "ms"),
        ("qrs_axis", -30, 90, "deg")]:
    if c in d.columns:
        v = d[c].replace(
            [np.inf, -np.inf], np.nan).dropna()
        if len(v):
            print("  %-14s %.1f%% within"
                  " %d-%d %s"
                  % (c, 100 * float(
                      ((v >= lo) & (v <= hi))
                      .mean()), lo, hi, u))