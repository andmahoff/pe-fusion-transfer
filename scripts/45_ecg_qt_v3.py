"""QT delineation, version 3.

Two problems with v2:
  - the tangent search runs to a fixed window, so
    in slow rhythms it can catch a U wave or the
    following P wave
  - QT correlated with rate at -0.606, steeper
    than physiology alone explains

Fixes:
  - the search window scales with RR and stops at
    the next P wave if one is detectable
  - the T peak must be the dominant deflection in
    its window, so a U wave cannot be mistaken
    for it
  - the tangent is fitted over several samples
    rather than taken from a single difference,
    which is far less noise-sensitive
  - a U-wave notch check rejects beats where the
    trace rises again before baseline

Selection rewards |r| near zero from either
side.
Run in venv_ecg only.
"""
import os
import time
import numpy as np
import pandas as pd
import wfdb
from scipy import signal as sg
from scipy import stats as st

OUT = os.path.join("data", "raw", "ecg")
PROC = os.path.join("data", "processed")
DEST = os.path.join(PROC, "ecg_derived_v3.csv")
REC = os.path.join(
    PROC, "ecg_derived_v3_record.csv")

FS = 500
LEAD_I, LEAD_II, LEAD_AVF = 0, 1, 4
BOUNDS = {"qrs_ms": (50, 200),
          "qt_ms": (250, 600),
          "pr_ms": (80, 320),
          "hr_mean": (25, 220)}


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
    return q, s_, base, noise


def t_end_v3(m, s_, rr, base, noise, fs=FS):
    """T-wave end, guarded against U waves and
    the following P wave.

    Returns (index, flag) where flag records why
    a beat was rejected, for diagnostics.
    """
    # search window scales with RR, and stops
    # well before the next QRS
    lo = s_ + int(0.04 * fs)
    hi = int(min(len(m) - 1,
                 s_ + 0.70 * rr,
                 s_ + 0.60 * fs))
    if hi - lo < int(0.10 * fs):
        return None, "window"

    seg = m[lo:hi] - base
    a = np.abs(seg)
    if a.max() < 4.0 * noise:
        return None, "flat"

    # T peak must dominate its window
    tp = int(np.argmax(a))
    if tp < 3 or tp > len(seg) - 8:
        return None, "edge"

    # reject if a later deflection rivals the T
    # peak, which usually means a U wave or the
    # next P wave is inside the window
    after = a[tp + int(0.06 * fs):]
    if len(after) and after.max() > 0.7 * a[tp]:
        hi2 = tp + int(0.06 * fs) \
            + int(np.argmax(after))
        hi = lo + hi2
        seg = m[lo:hi] - base
        a = np.abs(seg)
        if len(a) < int(0.10 * fs):
            return None, "truncated"
        tp = int(np.argmax(a))
        if tp > len(seg) - 8:
            return None, "edge2"

    # tangent from a multi-sample slope, not one
    # difference, so noise does not dominate
    w = max(3, int(0.012 * fs))
    tail = seg[tp:]
    if len(tail) < w + 3:
        return None, "short"
    sl = np.array([
        np.polyfit(np.arange(w),
                   tail[i:i + w], 1)[0]
        for i in range(len(tail) - w)])
    k = (int(np.argmin(sl)) if seg[tp] > 0
         else int(np.argmax(sl)))
    if abs(sl[k]) < 1e-9:
        return None, "noslope"
    y0 = tail[k + w // 2]
    step = -y0 / sl[k]
    if not np.isfinite(step) or step < 0:
        return None, "backwards"
    end = k + w // 2 + step
    if end > len(tail) - 1:
        end = len(tail) - 1
    return lo + tp + int(end), "ok"


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
    d = {"t_flag": "nopeaks"}
    d.update(hrv(pk, fs))
    d.update(axis(sig, pk, fs))
    d.update(amplitudes(sig))
    if len(pk) < 3:
        return d

    v = vmag(sig, fs)
    rr = float(np.median(np.diff(pk)))
    pre = int(0.30 * fs)
    post = int(min(0.75 * rr, 0.70 * fs))
    mb = median_beat(v, pk, pre, post)
    if mb is None:
        d["t_flag"] = "nobeat"
        return d

    r = pre
    q, s_, base, noise = delineate(mb, r, fs)
    d["qrs_ms"] = (s_ - q) / fs * 1000.0
    d["vm_noise"] = noise

    te, flag = t_end_v3(mb, s_, rr, base,
                        noise, fs)
    d["t_flag"] = flag
    if te is not None:
        d["qt_ms"] = (te - q) / fs * 1000.0

    p_hi = max(0, q - int(0.02 * fs))
    p_lo = max(0, q - int(0.25 * fs))
    if p_hi - p_lo > 10:
        seg = mb[p_lo:p_hi] - base
        pp = int(np.argmax(np.abs(seg)))
        if np.abs(seg[pp]) > 3 * noise:
            d["pr_ms"] = ((q - (p_lo + pp))
                          / fs * 1000.0)

    for k, (lo, hi) in BOUNDS.items():
        if k in d and d[k] is not None:
            if not (lo <= d[k] <= hi):
                d[k] = np.nan
    return d


idx = pd.read_csv(os.path.join(
    OUT, "ecg_record_index.csv"))
idx["rec"] = idx["rec"].apply(norm_rec)
recs = sorted(idx["rec"].astype(str).unique())
print("records:", len(recs))

rows, bad = [], 0
t0 = time.time()
for i, r_ in enumerate(recs):
    try:
        rec = wfdb.rdrecord(rec_path(r_))
        s = np.nan_to_num(rec.p_signal, nan=0.0)
        d = process(s)
        d["rec"] = r_
        rows.append(d)
    except Exception:
        bad += 1
    if (i + 1) % 1000 == 0:
        el = time.time() - t0
        print("  %d/%d  ok %d  eta %.0f min"
              % (i + 1, len(recs), len(rows),
                 (len(recs) - i - 1)
                 / max((i + 1) / el, 1e-9)
                 / 60), flush=True)

d = pd.DataFrame(rows)
print("")
print("processed:", len(d), " failed:", bad)

print("")
print("T-END REJECTION REASONS")
print(d["t_flag"].value_counts()
      .to_string())

rr_s = d["rr_mean"] / 1000.0
qt = d["qt_ms"]
hr = 60.0 / rr_s
d["qtc_bazett"] = qt / np.sqrt(rr_s)
d["qtc_fridericia"] = qt / (rr_s ** (1 / 3.0))
d["qtc_framingham"] = qt + 154.0 * (1.0 - rr_s)
d["qtc_hodges"] = qt + 1.75 * (hr - 60.0)
FORM = ["qtc_bazett", "qtc_fridericia",
        "qtc_framingham", "qtc_hodges"]
for c in FORM:
    v = d[c]
    d[c] = v.where((v >= 300) & (v <= 650))

print("")
print("=" * 58)
print("QT vs HEART RATE  (v2 was r = -0.606)")
m = d["qt_ms"].notna() & d["rr_mean"].notna()
if m.sum() > 100:
    r, _ = st.pearsonr(hr[m], d.loc[m, "qt_ms"])
    print("  r = %+.3f  on %d records"
          % (r, int(m.sum())))

print("")
print("CORRECTION FORMULAE")
print("  %-16s %7s %8s %9s"
      % ("formula", "median", "in range",
         "|r| vs HR"))
best, bestr = None, 9.9
for c in FORM:
    v = d[c].dropna()
    if len(v) < 100:
        continue
    pc = 100 * float(
        ((v >= 350) & (v <= 500)).mean())
    mm = d[c].notna() & np.isfinite(hr)
    r, _ = st.pearsonr(hr[mm], d.loc[mm, c])
    if abs(r) < bestr:
        bestr, best = abs(r), c
    print("  %-16s %7.1f %7.1f%% %+9.3f"
          % (c, v.median(), pc, r))

print("")
print("  least rate-dependent: %s (|r| %.3f)"
      % (best, bestr))
d["qtc_primary"] = d[best]

print("")
print("PLAUSIBILITY BY HEART-RATE BAND")
band = pd.cut(hr, [0, 60, 80, 100, 120, 300],
              labels=["<60", "60-80", "80-100",
                      "100-120", ">120"])
for c in FORM:
    v = d[c]
    g = ((v >= 350) & (v <= 500)).groupby(
        band, observed=False).mean() * 100
    print("  %-16s %s"
          % (c, "  ".join(
              "%s %.0f%%" % (k, x)
              for k, x in g.items()
              if not np.isnan(x))))

d.to_csv(REC, index=False)

m2 = idx[["subject_id", "hadm_id",
          "rec"]].drop_duplicates()
j = m2.merge(d.drop(columns=["t_flag"]),
             on="rec", how="inner")
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
print("")
print("saved", DEST, agg.shape)

print("")
print("FINAL  (v2 in brackets)")
# typed in from the v2 output
V2 = {"qrs_ms": (86.0, 75.9),
      "qt_ms": (404.0, 67.5)}
for c, lo, hi in [("hr_mean", 40, 140),
                  ("pr_ms", 120, 220),
                  ("qrs_ms", 60, 140),
                  ("qt_ms", 300, 480),
                  ("qtc_primary", 350, 500),
                  ("qrs_axis", -30, 90)]:
    if c not in d.columns:
        continue
    v = d[c].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if not len(v):
        continue
    pc = 100 * float(
        ((v >= lo) & (v <= hi)).mean())
    was = V2.get(c)
    tag = ("  (v2 %.1f, %.1f%%)" % was
           if was else "")
    print("  %-14s median %6.1f  %5.1f%%"
          " in range  cov %.1f%%%s"
          % (c, v.median(), pc,
             100 * float(d[c].notna().mean()),
             tag))