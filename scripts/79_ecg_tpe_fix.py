"""Tp-e, QT and QTc by consensus of three T-end
estimates.

At 88 bpm the RR interval is about 680 ms, so the
T wave often has not finished before the next P
wave starts, and a single tangent can land on the
P wave. Three rules guard against this:
  1 RR bound     T end cannot fall later than 60%
                 of RR after QRS onset
  2 TP baseline  the baseline comes from the
                 isoelectric TP segment rather
                 than the end of the search window
  3 consensus    three estimates (tangent, 10%
                 return, maximum curvature) are
                 combined by their median, and a
                 record is rejected when they
                 disagree by more than 40 ms

Only the repolarisation measurements are
recomputed. Everything else is carried over from
the v6 record file.
Run in venv_ecg only.

OUTPUT FILES
  data\\processed\\ecg_derived_v7_record.csv
  data\\processed\\ecg_derived_v7.csv
  results\\ecg_v7_log.txt
"""
import os
import time
import numpy as np
import pandas as pd
import wfdb
from scipy import signal as sg

OUT = os.path.join("data", "raw", "ecg")
PROC = os.path.join("data", "processed")
SRC = os.path.join(
    PROC, "ecg_derived_v6_record.csv")
REC_DEST = os.path.join(
    PROC, "ecg_derived_v7_record.csv")
DEST = os.path.join(PROC, "ecg_derived_v7.csv")

FS = 500
L_II = 1
RR_FRAC = 0.60      # T-end bound
SPREAD_MS = 40.0    # max disagreement allowed

GATE = {"tpe_ms": (40, 130),
        "qt_ms": (280, 520),
        "qtc_hodges": (330, 560),
        "tpe_qt": (0.10, 0.40)}
NEG_BAD = ("s_wave_i", "q_wave_iii", "st_min",
           "t_v1", "t_v2", "t_v3", "t_v4",
           "t_v5", "t_v6", "t_wave_iii")
ANG = ("qrs_axis", "t_axis", "p_axis")


def norm_rec(r):
    s = str(r).replace("\\", "/").strip("/")
    return s if s.startswith("files/") \
        else "files/" + s


def rp(r):
    return os.path.join(
        OUT, *norm_rec(r).split("/"))


def bpf(x, lo, hi, fs=FS):
    b, a = sg.butter(
        3, [lo / (fs / 2), hi / (fs / 2)],
        btype="band")
    return sg.filtfilt(b, a, x)


def lpf(x, hi, fs=FS):
    b, a = sg.butter(3, hi / (fs / 2),
                     btype="low")
    return sg.filtfilt(b, a, x)


def detect_r(x, fs=FS):
    f = bpf(x, 5.0, 15.0, fs)
    dd = np.diff(f, prepend=f[0])
    w = int(0.15 * fs)
    i = np.convolve(dd ** 2, np.ones(w) / w,
                    mode="same")
    thr = np.mean(i) + 0.5 * np.std(i)
    pk, _ = sg.find_peaks(
        i, height=thr,
        distance=int(0.25 * fs))
    return pk


def median_beat(sig, pk, pre, post):
    seg = [sig[p - pre:p + post] for p in pk
           if p - pre >= 0
           and p + post < sig.shape[0]]
    if len(seg) < 3:
        return None
    return np.median(np.stack(seg), axis=0)


def qrs_by_slope(vm, r, fs=FS):
    sl = np.abs(np.gradient(lpf(vm, 40.0, fs)))
    pre = max(0, r - int(0.20 * fs))
    quiet = sl[pre:max(pre + 10,
                       r - int(0.12 * fs))]
    thr = (float(np.median(quiet))
           + 3.0 * float(np.std(quiet)) + 1e-9)
    thr = max(thr, 0.05 * float(sl[r]))
    hold = int(0.010 * fs)
    cap = int(0.09 * fs)
    q = r
    while q > hold and r - q < cap:
        if np.all(sl[q - hold:q] < thr):
            break
        q -= 1
    s_ = r
    n = len(sl)
    while s_ < n - hold - 1 and s_ - r < cap:
        if np.all(sl[s_:s_ + hold] < thr):
            break
        s_ += 1
    return q, s_


def t_end_three(x, tl, hi, base, fs=FS):
    """Three independent estimates of T end.
    Returns (median, spread, tpeak) in samples,
    or (None, None, None)."""
    if hi - tl < int(0.08 * fs):
        return None, None, None
    seg = x[tl:hi] - base
    a = np.abs(seg)
    tp = int(np.argmax(a))
    amp = float(a[tp])
    if amp < 0.02 or tp >= len(seg) - 8:
        return None, None, None
    tail = seg[tp:]
    n = len(tail)
    if n < 12:
        return None, None, None
    est = []

    # 1 tangent: steepest descent after the peak
    w = max(3, int(0.012 * fs))
    if n > w + 3:
        sl = np.array([
            np.polyfit(np.arange(w),
                       tail[i:i + w], 1)[0]
            for i in range(n - w)])
        k = (int(np.argmin(sl)) if seg[tp] > 0
             else int(np.argmax(sl)))
        if abs(sl[k]) > 1e-9:
            y0 = tail[k + w // 2]
            step = -y0 / sl[k]
            if np.isfinite(step) and step >= 0:
                e = k + w // 2 + step
                if 0 < e < n:
                    est.append(float(e))

    # 2 return to 10% of the peak
    at = np.abs(tail)
    k = 0
    while k < n - 2 and at[k] > 0.10 * amp:
        k += 1
    if 0 < k < n:
        est.append(float(k))

    # 3 maximum curvature after the peak
    if n > 10:
        sm = lpf(tail, 20.0, fs) if n > 30 \
            else tail
        d2 = np.gradient(np.gradient(sm))
        lo_ = max(3, int(0.02 * fs))
        if n - lo_ > 3:
            k3 = lo_ + int(np.argmax(
                np.abs(d2[lo_:])))
            if 0 < k3 < n:
                est.append(float(k3))

    if len(est) < 2:
        return None, None, None
    est = np.array(est)
    med = float(np.median(est))
    spread = float(est.max() - est.min())
    return tp, med, spread


def process(sig, fs=FS):
    d = {}
    ii = sig[:, L_II]
    pk = detect_r(ii, fs)
    if len(pk) < 3:
        return d
    rr = np.diff(pk) / fs * 1000.0
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 3:
        return d
    hr = 60000.0 / np.mean(rr)
    rr_samp = float(np.median(np.diff(pk)))

    pre, post = int(0.35 * fs), int(0.50 * fs)
    raw = median_beat(sig, pk, pre, post)
    if raw is None:
        return d
    r = pre

    # TP baseline: genuinely isoelectric
    tp_lo = max(0, r - int(0.34 * fs))
    tp_hi = max(tp_lo + 20, r - int(0.26 * fs))
    mb = raw - np.median(raw[tp_lo:tp_hi],
                         axis=0)
    vm = np.sqrt(np.sum(mb ** 2, axis=1))
    q, s_ = qrs_by_slope(vm, r, fs)

    t2 = lpf(mb[:, L_II], 40.0, fs)
    base = float(np.median(t2[tp_lo:tp_hi]))

    tl = s_ + int(0.02 * fs)
    # rule 1: bound by RR, not a fixed window
    hi = int(min(len(t2) - 1,
                 q + RR_FRAC * rr_samp,
                 s_ + 0.42 * fs))
    tp, te, spread = t_end_three(
        t2, tl, hi, base, fs)
    if tp is None:
        return d

    sp_ms = spread / fs * 1000.0
    d["tpe_spread_ms"] = sp_ms
    d["tpe_ok"] = int(sp_ms <= SPREAD_MS)
    # rule 3: reject on disagreement
    if sp_ms > SPREAD_MS:
        return d

    d["tpe_ms"] = (te - tp) / fs * 1000.0
    qt = (tl + tp + te - q) / fs * 1000.0
    d["qt_ms"] = qt
    if qt > 0:
        d["tpe_qt"] = d["tpe_ms"] / qt
    d["qtc_hodges"] = qt + 1.75 * (hr - 60.0)
    d["qtc_over451"] = int(
        d["qtc_hodges"] > 451)

    for k_, (lo_, hi_) in GATE.items():
        if k_ in d and not (
                lo_ <= d[k_] <= hi_):
            d[k_] = np.nan
    return d


print("reading v6 record file ...")
v6 = pd.read_csv(SRC)
v6["rec"] = v6["rec"].apply(norm_rec)
print("v6 records:", len(v6), flush=True)

idx = pd.read_csv(os.path.join(
    OUT, "ecg_record_index.csv"))
idx["rec"] = idx["rec"].apply(norm_rec)
recs = sorted(idx["rec"].unique())

rows, bad = [], 0
t0 = time.time()
for i, r_ in enumerate(recs):
    try:
        rec = wfdb.rdrecord(rp(r_))
        s = np.nan_to_num(rec.p_signal,
                          nan=0.0)
        dd = process(s)
        dd["rec"] = r_
        rows.append(dd)
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

nw = pd.DataFrame(rows)
print("")
print("recomputed:", len(nw), " failed:", bad)

# replace only the repolarisation columns
REPL = ["tpe_ms", "qt_ms", "qtc_hodges",
        "tpe_qt", "qtc_over451",
        "tpe_spread_ms", "tpe_ok"]
keep = [c for c in v6.columns
        if c not in REPL]
d = v6[keep].merge(
    nw[["rec"] + [c for c in REPL
                  if c in nw.columns]],
    on="rec", how="left")
d.to_csv(REC_DEST, index=False)
print("saved", REC_DEST, d.shape)

m = idx[["subject_id", "hadm_id",
         "rec"]].drop_duplicates()
j = m.merge(d, on="rec", how="inner")
num = [c for c in j.columns
       if c not in ("rec", "subject_id",
                    "hadm_id")]
g = j.groupby(["subject_id", "hadm_id"])
parts = [g[num].mean().add_suffix("_mean")]
pos = [c for c in num
       if not c.startswith(NEG_BAD)
       and c not in ANG]
neg = [c for c in num
       if c.startswith(NEG_BAD)]
if pos:
    parts.append(
        g[pos].max().add_suffix("_worst"))
if neg:
    parts.append(
        g[neg].min().add_suffix("_worst"))
agg = pd.concat(parts, axis=1).reset_index()
for c in ANG:
    if c not in j.columns:
        continue
    rad = np.radians(j[c])
    tmp = j[["subject_id", "hadm_id"]].copy()
    tmp["s"], tmp["c"] = np.sin(rad), np.cos(rad)
    gg = tmp.groupby(["subject_id",
                      "hadm_id"]).mean()
    cm = np.degrees(np.arctan2(
        gg["s"], gg["c"])).rename(
        c + "_circmean").reset_index()
    agg = agg.merge(
        cm, on=["subject_id", "hadm_id"],
        how="left")
    agg = agg.drop(columns=[c + "_mean"],
                   errors="ignore")
agg.to_csv(DEST, index=False)
print("saved", DEST, agg.shape)

print("")
print("REPOLARISATION  (v6 in brackets)")
# typed in from the v6 output
PREV = {"tpe_ms": (114.0, 70.4),
        "qt_ms": (500.0, 18.0),
        "qtc_hodges": (534.5, 22.0),
        "tpe_qt": (0.38, 23.0)}
TGT = {"tpe_ms": (60, 110),
       "qt_ms": (350, 450),
       "qtc_hodges": (400, 470),
       "tpe_qt": (0.15, 0.30)}
for c in ["tpe_ms", "qt_ms", "qtc_hodges",
          "tpe_qt"]:
    if c not in d.columns:
        continue
    v = d[c].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if not len(v):
        print("  %-12s ALL MISSING" % c)
        continue
    pm, pr_ = PREV.get(c, (np.nan, np.nan))
    lo_, hi_ = TGT[c]
    print("  %-12s median %7.2f (was %.2f)"
          "   cover %.0f%% (was %.0f%%)"
          "   target %g-%g"
          % (c, v.median(), pm,
             100.0 * len(v) / len(d),
             100.0 - pr_, lo_, hi_))

print("")
print("CONSENSUS QUALITY")
if "tpe_spread_ms" in d.columns:
    v = d["tpe_spread_ms"].dropna()
    print("  disagreement between the three"
          " methods:")
    print("    median %.1f ms  IQR %.1f-%.1f"
          % (v.median(), v.quantile(0.25),
             v.quantile(0.75)))
    print("    within %g ms: %.1f%%"
          % (SPREAD_MS,
             100.0 * float(
                 (v <= SPREAD_MS).mean())))
    print("")
    print("  a low spread means the three"
          " methods agree, so the estimate is")
    print("  trustworthy. This is the check the"
          " single-method versions lacked.")