"""Derived ECG measurements, version 2.

Fixes the delineation that gave QRS 46 ms and a
280-460 ms QTc range in version 1:
  - vector magnitude across all twelve leads,
    so a beat is not missed when one lead is
    isoelectric
  - onset and offset found by return to
    baseline, not by a fraction of the R peak
  - baseline taken from the PR segment
  - physiological bounds, so an implausible
    value is recorded as missing rather than
    as a wrong number
Rate, variability and axis are unchanged, since
those validated at 97.9% and 67.8%.
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
DEST = os.path.join(PROC, "ecg_derived_v2.csv")

FS = 500
LEAD_I, LEAD_II, LEAD_AVF = 0, 1, 4

BOUNDS = {"qrs_ms": (50, 200),
          "qt_ms": (250, 600),
          "pr_ms": (80, 320),
          "hr_mean": (25, 220),
          "qtc_bazett": (300, 650),
          "qtc_fridericia": (300, 650)}


def norm_rec(r):
    s = str(r).replace("\\", "/").strip("/")
    if not s.startswith("files/"):
        s = "files/" + s
    return s


def rec_path(r):
    return os.path.join(
        OUT, *norm_rec(r).split("/"))


def bandpass(x, lo, hi, fs=FS, order=3):
    b, a = sg.butter(
        order, [lo / (fs / 2), hi / (fs / 2)],
        btype="band")
    return sg.filtfilt(b, a, x)


def detect_r(x, fs=FS):
    f = bandpass(x, 5.0, 15.0, fs)
    d = np.diff(f, prepend=f[0])
    s = d ** 2
    w = max(1, int(0.15 * fs))
    i = np.convolve(s, np.ones(w) / w,
                    mode="same")
    thr = np.mean(i) + 0.5 * np.std(i)
    pk, _ = sg.find_peaks(
        i, height=thr, distance=int(0.25 * fs))
    return pk


def vmag(sig, fs=FS):
    """Spatial magnitude across all leads, after
    light filtering. Zero only when every lead is
    at baseline, which is what makes the onset
    and offset reliable."""
    f = np.zeros_like(sig)
    for i in range(sig.shape[1]):
        f[:, i] = bandpass(sig[:, i], 0.5, 40.0,
                           fs)
    return np.sqrt(np.sum(f ** 2, axis=1))


def median_beat(x, pk, pre, post):
    seg = [x[p - pre:p + post] for p in pk
           if p - pre >= 0
           and p + post < len(x)]
    if len(seg) < 3:
        return None
    return np.median(np.vstack(seg), axis=0)


def delineate(m, r, fs=FS):
    """Onset and offset by return to baseline.

    Baseline and its noise level come from the PR
    segment, 200 to 120 ms before R. The complex
    starts and ends where the trace re-enters
    that band and stays there.
    """
    lo = max(0, r - int(0.20 * fs))
    hi = max(lo + 5, r - int(0.12 * fs))
    base = float(np.median(m[lo:hi]))
    noise = float(np.std(m[lo:hi]))
    thr = base + max(3.0 * noise,
                     0.02 * (m[r] - base))

    hold = int(0.012 * fs)
    q = r
    while q > hold:
        if np.all(m[q - hold:q] <= thr):
            break
        q -= 1
    s_ = r
    n = len(m)
    while s_ < n - hold - 1:
        if np.all(m[s_:s_ + hold] <= thr):
            break
        s_ += 1
    return q, s_, base, noise, thr


def t_offset(x, s_, rr_samp, fs=FS):
    """T-wave end by the tangent method: steepest
    descent after the T peak, extrapolated to
    baseline."""
    lim = min(len(x) - 1,
              s_ + int(min(0.55 * rr_samp,
                           0.50 * fs)))
    if lim - s_ < int(0.08 * fs):
        return None
    seg = x[s_:lim]
    base = float(np.median(
        x[max(0, s_ - int(0.02 * fs)):s_ + 1]))
    d = seg - base
    tp = int(np.argmax(np.abs(d)))
    if tp >= len(seg) - 5:
        return None
    tail = d[tp:]
    sl = np.diff(tail)
    k = int(np.argmin(sl)) if d[tp] > 0 \
        else int(np.argmax(sl))
    if abs(sl[k]) < 1e-9:
        return None
    x0 = tp + k
    y0 = tail[k]
    step = -y0 / sl[k]
    end = x0 + step
    if not np.isfinite(end):
        return None
    end = int(min(max(end, tp + 1),
                  len(seg) - 1))
    return s_ + end


def bound(d):
    for k, (lo, hi) in BOUNDS.items():
        if k in d and d[k] is not None:
            if not (lo <= d[k] <= hi):
                d[k] = np.nan
    return d


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


def process(sig, fs=FS):
    ii = sig[:, LEAD_II]
    pk = detect_r(ii)
    d = {}
    d.update(hrv(pk, fs))
    d.update(axis(sig, pk, fs))
    d.update(amplitudes(sig))
    if len(pk) < 3:
        return d

    v = vmag(sig, fs)
    rr = (float(np.median(np.diff(pk)))
          if len(pk) > 1 else fs)
    pre = int(0.30 * fs)
    post = int(min(0.60 * rr, 0.55 * fs))
    mb = median_beat(v, pk, pre, post)
    if mb is None:
        return d
    r = pre
    q, s_, base, noise, thr = delineate(
        mb, r, fs)
    d["qrs_ms"] = (s_ - q) / fs * 1000.0
    d["vm_base"] = base
    d["vm_noise"] = noise
    d["vm_rpeak"] = float(mb[r])

    te = t_offset(mb, s_, rr, fs)
    if te is not None:
        d["qt_ms"] = (te - q) / fs * 1000.0
        d["t_off_ms"] = (te - s_) / fs * 1000.0

    p_hi = max(0, q - int(0.02 * fs))
    p_lo = max(0, q - int(0.25 * fs))
    if p_hi - p_lo > 10:
        seg = mb[p_lo:p_hi] - base
        pp = int(np.argmax(np.abs(seg)))
        if np.abs(seg[pp]) > 3 * noise:
            d["pr_ms"] = ((q - (p_lo + pp))
                          / fs * 1000.0)
            d["p_amp"] = float(seg[pp])

    if "qt_ms" in d and "rr_mean" in d:
        rs = d["rr_mean"] / 1000.0
        if rs > 0:
            d["qtc_bazett"] = (
                d["qt_ms"] / np.sqrt(rs))
            d["qtc_fridericia"] = (
                d["qt_ms"] / (rs ** (1 / 3.0)))
    return bound(d)


idx = pd.read_csv(os.path.join(
    OUT, "ecg_record_index.csv"))
idx["rec"] = idx["rec"].apply(norm_rec)
recs = sorted(idx["rec"].astype(str).unique())
print("records:", len(recs))

rows, bad, shown = [], 0, 0
t0 = time.time()
for i, r_ in enumerate(recs):
    try:
        rec = wfdb.rdrecord(rec_path(r_))
        s = rec.p_signal
        if s is None or s.shape[1] < 12:
            raise ValueError("bad shape")
        s = np.nan_to_num(s, nan=0.0)
        d = process(s)
        d["rec"] = r_
        rows.append(d)
    except Exception as exc:
        bad += 1
        if shown < 5:
            shown += 1
            print("  FAIL", r_, repr(exc)[:140])
    if (i + 1) % 500 == 0:
        el = time.time() - t0
        rate = (i + 1) / max(el, 1e-9)
        print("  %d/%d  ok %d  fail %d  %.0f/s"
              "  eta %.0f min"
              % (i + 1, len(recs), len(rows),
                 bad, rate,
                 (len(recs) - i - 1)
                 / max(rate, 1e-9) / 60),
              flush=True)

print("")
print("processed:", len(rows), " failed:", bad)
if not rows:
    raise SystemExit("nothing processed")

d = pd.DataFrame(rows)
d.to_csv(os.path.join(
    PROC, "ecg_derived_v2_record.csv"),
    index=False)

m = idx[["subject_id", "hadm_id",
         "rec"]].drop_duplicates()
j = m.merge(d, on="rec", how="inner")
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
          "pr_ms", "qrs_ms", "qt_ms",
          "qtc_bazett", "qrs_axis"]:
    if c in d.columns:
        v = d[c].replace(
            [np.inf, -np.inf], np.nan).dropna()
        if len(v):
            print("  %-14s n=%4d  median %7.1f"
                  "  IQR %.1f to %.1f"
                  % (c, len(v), v.median(),
                     v.quantile(0.25),
                     v.quantile(0.75)))

print("")
print("PLAUSIBILITY  (v1 in brackets)")
# typed in from the v1 output
V1 = {"hr_mean": 97.9, "qrs_ms": 21.4,
      "qtc_bazett": 47.3, "qrs_axis": 67.8}
for c, lo, hi, u in [
        ("hr_mean", 40, 140, "bpm"),
        ("pr_ms", 120, 220, "ms"),
        ("qrs_ms", 60, 140, "ms"),
        ("qt_ms", 300, 480, "ms"),
        ("qtc_bazett", 350, 500, "ms"),
        ("qrs_axis", -30, 90, "deg")]:
    if c in d.columns:
        v = d[c].replace(
            [np.inf, -np.inf], np.nan).dropna()
        if len(v):
            pc = 100 * float(
                ((v >= lo) & (v <= hi)).mean())
            was = V1.get(c)
            tag = ("  (v1 %.1f%%)" % was
                   if was else "")
            print("  %-12s %5.1f%% within"
                  " %d-%d %s%s"
                  % (c, pc, lo, hi, u, tag))

print("")
print("COVERAGE (non-missing after bounds)")
for c in ["qrs_ms", "qt_ms", "pr_ms",
          "qtc_bazett"]:
    if c in d.columns:
        print("  %-12s %.1f%%"
              % (c, 100 * float(
                  d[c].notna().mean())))