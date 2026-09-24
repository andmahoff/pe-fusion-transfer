"""ECG measurements v8. Five corrections, with
the Daniel score now coded from the published
score sheet.

DANIEL SCORE, verified against the original
table (Vereckei Figure 3 reproduction):
  tachycardia >100            2
  incomplete RBBB             2   } maximum,
  complete RBBB               3   } not summed
  TWI in all of V1-V4         4
  TWI in V1   <1mm 0, 1-2mm 1, >2mm 2
  TWI in V2   <1mm 1, 1-2mm 2, >2mm 3
  TWI in V3   <1mm 1, 1-2mm 2, >2mm 3
  S wave in lead I            0
  Q wave in lead III          1
  inverted T in lead III      1
  full S1Q3T3 complex         2
  MAXIMUM 2+3+12+0+1+1+2 = 21

V4 has no individual row; it contributes only
through the all-four bonus. A shallow inversion
in V1 scores zero, which is how the instrument
handles a small negative T in V1 being a normal
finding.

OTHER FIXES
  Tp-e formula. t_end_three returned T end as a
    distance from the T peak; the v7 caller
    computed (te - tp), subtracting the peak
    twice. Most values went negative and were
    gated away, which is why Tp-e held 7% while
    QT, from the same landmarks with the correct
    formula, held 38%. Both offsets are now
    returned relative to tl.
  TP baseline drifted. The window was fixed at
    340-260 ms before R, which above about
    100 bpm sits inside the previous T wave. Now
    placed as a fraction of the record's own RR.
  Sinus tachycardia. The v7 code required the SR
    logit, but PTB-XL labels fast sinus as
    STACH, which was loaded and never used.
  aVR threshold. 0.02 mV was chosen to reproduce
    a pooled 36% prevalence, which is circular
    and sits inside baseline noise. Restored to
    the conventional 0.05 mV; st_avr carries the
    finer detail.
  Spatial QRS-T mixed definitions, summing QRS
    over the complex but taking T at its peak.
    Both are now summed; qrst_spatial_pk keeps
    the peak version.

Run in venv_ecg only.

OUTPUT FILES
  data\\processed\\ecg_derived_v8_record.csv
  data\\processed\\ecg_derived_v8.csv
  results\\ecg_v8_log.txt
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
    PROC, "ecg_derived_v8_record.csv")
DEST = os.path.join(PROC, "ecg_derived_v8.csv")

FS = 500
L_I, L_II, L_III = 0, 1, 2
L_AVR, L_AVF, L_AVL = 3, 4, 5
V = {i: 6 + i for i in range(6)}
RR_FRAC = 0.60
SPREAD_MS = 40.0
STE_AVR = 0.05
MM1, MM2 = 0.10, 0.20      # 1 mm, 2 mm in mV

KORS_X = {L_I: 0.38, L_II: -0.07,
          V[0]: -0.13, V[1]: 0.05,
          V[2]: -0.01, V[3]: 0.14,
          V[4]: 0.06, V[5]: 0.54}
KORS_Y = {L_I: -0.07, L_II: 0.93,
          V[0]: 0.06, V[1]: -0.02,
          V[2]: -0.05, V[3]: 0.06,
          V[4]: -0.17, V[5]: 0.13}
KORS_Z = {L_I: 0.11, L_II: -0.23,
          V[0]: -0.43, V[1]: -0.06,
          V[2]: -0.14, V[3]: -0.20,
          V[4]: -0.11, V[5]: 0.31}

GATE = {"hr_mean": (25, 220),
        "rr_sd": (0, 500), "rmssd": (0, 400),
        "pc_sd1": (0, 300), "pc_sd2": (0, 400),
        "pc_ratio": (0.05, 3.0),
        "qrs_ms": (50, 200),
        "qt_ms": (250, 600),
        "qtc_hodges": (300, 650),
        "tpe_ms": (30, 160),
        "tpe_qt": (0.05, 0.50),
        "pr_ms": (80, 320),
        "p_dur_ms": (40, 160),
        "p_amp_ii": (-0.5, 0.6),
        "pr_seg_ms": (0, 200),
        "qrst_spatial": (0, 180),
        "qrst_spatial_pk": (0, 180),
        "qrst_frontal": (0, 180),
        "rs_ratio_v1": (0, 20),
        "sokolow_lyon": (0.1, 6.0),
        "cornell": (0.1, 5.0)}

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


def kors(mb):
    x = sum(w * mb[:, c]
            for c, w in KORS_X.items())
    y = sum(w * mb[:, c]
            for c, w in KORS_Y.items())
    z = sum(w * mb[:, c]
            for c, w in KORS_Z.items())
    return np.column_stack([x, y, z])


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
    """Returns (tpeak, tend, spread), tpeak and
    tend BOTH measured from tl. The v7 bug was
    returning tend relative to the PEAK while
    the caller treated it as relative to tl."""
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
    at = np.abs(tail)
    k = 0
    while k < n - 2 and at[k] > 0.10 * amp:
        k += 1
    if 0 < k < n:
        est.append(float(k))
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
    return (tp, tp + float(np.median(est)),
            float(est.max() - est.min()))


def axis_of(x, y):
    return float(np.degrees(np.arctan2(y, x)))


def twi_band(amp, v1=False):
    """Daniel T-inversion points for one lead.
    amp is the T amplitude in mV, negative when
    inverted. V1 scores one band lower."""
    if amp >= 0:
        return 0
    dep = abs(amp)
    if dep < MM1:
        band = 0 if v1 else 1
    elif dep <= MM2:
        band = 1 if v1 else 2
    else:
        band = 2 if v1 else 3
    return band


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
    rr_s = float(np.median(np.diff(pk)))

    d["hr_mean"] = hr
    d["rr_sd"] = float(np.std(rr, ddof=1))
    d["rmssd"] = float(np.sqrt(
        np.mean(np.diff(rr) ** 2)))
    d["n_beats"] = int(len(pk))
    if len(rr) >= 4:
        a, b = rr[:-1], rr[1:]
        s1 = float(np.std(
            (a - b) / np.sqrt(2.0), ddof=1))
        s2 = float(np.std(
            (a + b) / np.sqrt(2.0), ddof=1))
        d["pc_sd1"], d["pc_sd2"] = s1, s2
        d["pc_area"] = float(np.pi * s1 * s2)
        if s2 > 1e-9:
            d["pc_ratio"] = s1 / s2
    d["shopp_hr100"] = int(hr > 100)

    pre, post = int(0.35 * fs), int(0.50 * fs)
    raw = median_beat(sig, pk, pre, post)
    if raw is None:
        return d
    r = pre

    tp_hi = max(20, r - int(0.24 * rr_s))
    tp_lo = max(0, r - int(0.50 * rr_s))
    if tp_hi - tp_lo < 15:
        tp_lo = max(0, tp_hi - 20)
    mb_tp = raw - np.median(raw[tp_lo:tp_hi],
                            axis=0)
    pr_lo = max(0, r - int(0.10 * fs))
    pr_hi = max(pr_lo + 10, r - int(0.05 * fs))
    mb = raw - np.median(raw[pr_lo:pr_hi],
                         axis=0)
    vm = np.sqrt(np.sum(mb ** 2, axis=1))
    vm_tp = np.sqrt(np.sum(mb_tp ** 2, axis=1))
    base_tp = float(np.median(
        vm_tp[tp_lo:tp_hi]))
    noise_tp = float(np.std(
        vm_tp[tp_lo:tp_hi])) + 1e-9
    d["tp_win_ms"] = ((r - tp_lo) / fs
                      * 1000.0)

    q, s_ = qrs_by_slope(vm, r, fs)
    d["qrs_ms"] = (s_ - q) / fs * 1000.0
    w = int(0.05 * fs)
    qv = mb[max(0, r - w):r + w].sum(axis=0)
    d["qrs_axis"] = axis_of(qv[L_I], qv[L_AVF])
    d["rad"] = int(d["qrs_axis"] > 90)

    tl = s_ + int(0.02 * fs)
    th = min(len(mb), s_ + int(0.45 * fs))
    tv = None
    if th - tl > 10:
        seg = mb[tl:th]
        tv = np.array([
            seg[int(np.argmax(np.abs(
                seg[:, c]))), c]
            for c in range(mb.shape[1])])
        d["t_axis"] = axis_of(tv[L_I],
                              tv[L_AVF])
        a_ = abs(d["qrs_axis"] - d["t_axis"])
        d["qrst_frontal"] = float(
            min(a_, 360.0 - a_))
        for k, c in V.items():
            d["t_v%d" % (k + 1)] = float(tv[c])
        d["t_wave_iii"] = float(tv[L_III])
        d["n_twi_v14"] = int(sum(
            1 for k in range(4)
            if tv[V[k]] < 0))
        d["n_twi_v14_deep"] = int(sum(
            1 for k in range(4)
            if tv[V[k]] < -MM1))
        d["n_twi_v14_2mm"] = int(sum(
            1 for k in range(4)
            if tv[V[k]] < -MM2))
        d["shopp_twi_v14"] = int(
            d["n_twi_v14"] == 4)

        t2 = lpf(mb[:, L_II], 40.0, fs)
        base2 = float(np.median(
            t2[tp_lo:tp_hi]))
        hi_ = int(min(len(t2) - 1,
                      q + RR_FRAC * rr_s, th))
        tpo, teo, spread = t_end_three(
            t2, tl, hi_, base2, fs)
        if tpo is not None:
            sp = spread / fs * 1000.0
            d["tpe_spread_ms"] = sp
            d["tpe_ok"] = int(sp <= SPREAD_MS)
            d["tpe_ms"] = ((teo - tpo) / fs
                           * 1000.0)
            qt = (tl + teo - q) / fs * 1000.0
            d["qt_ms"] = qt
            if qt > 0:
                d["tpe_qt"] = (d["tpe_ms"]
                               / qt)
            d["qtc_hodges"] = qt + 1.75 * (
                hr - 60.0)
            d["qtc_over451"] = int(
                d["qtc_hodges"] > 451)

        # ---- DANIEL T-INVERSION, from the
        # published score sheet ----
        pts = 0
        pts += twi_band(tv[V[0]], v1=True)
        pts += twi_band(tv[V[1]])
        pts += twi_band(tv[V[2]])
        if all(tv[V[k]] < 0 for k in range(4)):
            pts += 4
        d["daniel_twi"] = int(min(pts, 12))

    xyz = kors(mb)
    qvec = xyz[q:s_].sum(axis=0) \
        if s_ > q else xyz[r]
    if tv is not None and th - tl > 10:
        tseg = xyz[tl:th]
        tmean = tseg.sum(axis=0)
        tpk = int(np.argmax(
            np.sum(tseg ** 2, axis=1)))
        tpeak = tseg[tpk]
        nq = np.linalg.norm(qvec)
        for nm, tvv in (("qrst_spatial",
                         tmean),
                        ("qrst_spatial_pk",
                         tpeak)):
            nt = np.linalg.norm(tvv)
            if nq > 1e-9 and nt > 1e-9:
                d[nm] = float(np.degrees(
                    np.arccos(np.clip(
                        np.dot(qvec, tvv)
                        / (nq * nt),
                        -1.0, 1.0))))
        d["qrs_vmag"] = float(nq)
        d["t_vmag"] = float(
            np.linalg.norm(tpeak))

    p_hi = max(0, q - int(0.01 * fs))
    p_lo = max(0, q - int(0.30 * fs))
    if p_hi - p_lo > 20:
        pv = vm_tp[p_lo:p_hi] - base_tp
        pi = int(np.argmax(pv))
        if pv[pi] > 2.0 * noise_tp:
            p2 = lpf(mb_tp[:, L_II], 40.0,
                     fs)[p_lo:p_hi]
            pa = float(p2[pi])
            d["p_amp_ii"] = pa
            d["p_pulmonale"] = int(pa > 0.25)
            d["p_axis"] = axis_of(
                mb_tp[p_lo + pi, L_I],
                mb_tp[p_lo + pi, L_AVF])
            a2 = np.abs(p2)
            h = max(0.10 * abs(pa), 0.005)
            k = pi
            while k > 0 and a2[k] > h:
                k -= 1
            on = p_lo + k
            k2 = pi
            while k2 < len(a2) - 1 and \
                    a2[k2] > h:
                k2 += 1
            off = p_lo + k2
            d["pr_ms"] = ((q - on) / fs
                          * 1000.0)
            d["p_dur_ms"] = ((off - on) / fs
                             * 1000.0)
            d["pr_seg_ms"] = ((q - off) / fs
                              * 1000.0)

    j60 = s_ + int(0.06 * fs)
    if j60 < len(mb):
        st = mb[j60]
        for nm, c in (("i", L_I), ("ii", L_II),
                      ("iii", L_III),
                      ("avr", L_AVR),
                      ("avf", L_AVF),
                      ("avl", L_AVL)):
            d["st_" + nm] = float(st[c])
        for k, c in V.items():
            d["st_v%d" % (k + 1)] = float(st[c])
        d["st_max"] = float(np.max(st))
        d["st_min"] = float(np.min(st))
        d["shopp_ste_avr"] = int(
            st[L_AVR] > STE_AVR)

    wq = int(0.06 * fs)
    s1 = float(np.min(
        mb[max(0, r - 5):min(len(mb), r + wq),
           L_I]))
    q3 = float(np.min(
        mb[max(0, r - wq):r + 5, L_III]))
    t3 = d.get("t_wave_iii", 0.0)
    d["s_wave_i"], d["q_wave_iii"] = s1, q3
    d["shopp_s1q3t3"] = int(
        (s1 < -0.15) and (q3 < -0.15)
        and (t3 < 0))
    d["q3_present"] = int(q3 < -0.15)
    d["t3_inverted"] = int(t3 < 0)

    def mx(c):
        return float(np.max(
            mb[max(0, r - wq):min(len(mb),
                                  r + wq), c]))

    def mn(c):
        return float(np.min(
            mb[max(0, r - wq):min(len(mb),
                                  r + wq), c]))
    d["r_v1"], d["s_v1"] = mx(V[0]), mn(V[0])
    if abs(d["s_v1"]) > 1e-6:
        d["rs_ratio_v1"] = (d["r_v1"]
                            / abs(d["s_v1"]))
    d["sokolow_lyon"] = (abs(d["s_v1"])
                         + max(mx(V[4]),
                               mx(V[5])))
    d["cornell"] = mx(L_AVL) + abs(mn(V[2]))

    nfrag = 0
    for c in range(mb.shape[1]):
        seg2 = mb[q:s_, c]
        if len(seg2) < 10:
            continue
        pa2 = float(np.max(np.abs(seg2)))
        if pa2 < 0.1:
            continue
        pks, _ = sg.find_peaks(
            np.abs(seg2), height=0.15 * pa2,
            distance=int(0.02 * fs))
        if len(pks) > 2 and len(
                set(np.sign(seg2[pks])
                    .tolist())) > 1:
            nfrag += 1
    d["qrs_frag_leads"] = nfrag

    for i in range(sig.shape[1]):
        d["sd_l%d" % i] = float(
            np.nanstd(sig[:, i]))

    for k_, (lo_, hi_) in GATE.items():
        if k_ in d and d[k_] is not None:
            if not (lo_ <= d[k_] <= hi_):
                d[k_] = np.nan
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

d = pd.DataFrame(rows)
print("")
print("processed:", len(d), " failed:", bad)

lg = os.path.join(
    PROC, "exp0_logits_record.csv")
if os.path.exists(lg):
    q = pd.read_csv(lg)
    cols = [c for c in ("rec", "scp_AFIB",
                        "scp_CRBBB",
                        "scp_IRBBB", "scp_SR",
                        "scp_STACH")
            if c in q.columns]
    d = d.merge(q[cols], on="rec", how="left")
    d["shopp_afib"] = (
        d["scp_AFIB"] > 0).astype(int)
    d["shopp_rbbb"] = (
        d.get("scp_CRBBB", 0) > 0).astype(int)
    d["irbbb"] = (
        d.get("scp_IRBBB", 0) > 0).astype(int)
    stach = (d.get("scp_STACH", 0) > 0) \
        .astype(int)
    sinus = (d.get("scp_SR", 0) > 0).astype(int)
    d["sinus_tach"] = (
        (stach == 1)
        | ((sinus == 1)
           & (d["shopp_hr100"] == 1))
    ).astype(int)

    # RBBB is a MAXIMUM, not a sum
    rbbb_pts = np.maximum(2 * d["irbbb"],
                          3 * d["shopp_rbbb"])
    d["daniel_score"] = (
        2.0 * d["sinus_tach"]
        + rbbb_pts
        + d["daniel_twi"].fillna(0)
        + 0.0                       # S in I
        + 1.0 * d["q3_present"].fillna(0)
        + 1.0 * d["t3_inverted"].fillna(0)
        + 2.0 * d["shopp_s1q3t3"].fillna(0))
    d["daniel_ge10"] = (
        d["daniel_score"] >= 10).astype(int)
    d["daniel_ge3"] = (
        d["daniel_score"] >= 3).astype(int)

    SH = [c for c in
          ["shopp_hr100", "shopp_s1q3t3",
           "shopp_rbbb", "shopp_twi_v14",
           "shopp_ste_avr", "shopp_afib"]
          if c in d.columns]
    d["shopp_count"] = d[SH].sum(axis=1)
    d["rv_strain"] = d[[c for c in
                        ("shopp_rbbb",
                         "shopp_s1q3t3",
                         "shopp_twi_v14")
                        if c in d.columns]] \
        .max(axis=1)
    d = d.drop(columns=[c for c in
                        ("scp_AFIB",
                         "scp_CRBBB",
                         "scp_IRBBB",
                         "scp_SR", "scp_STACH")
                        if c in d.columns])
    print("Daniel score built from the"
          " published table")
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
print("DANIEL SCORE, from the published table")
if "daniel_score" in d.columns:
    v = d["daniel_score"].dropna()
    print("  median %.1f  IQR %.1f-%.1f"
          "  max %.0f  (theoretical max 21)"
          % (v.median(), v.quantile(0.25),
             v.quantile(0.75), v.max()))
    print("")
    print("  distribution:")
    print(v.value_counts().sort_index()
          .head(15).to_string())
    print("")
    print("  >=10 (severe PH cutoff): %.1f%%"
          % (100 * d["daniel_ge10"].mean()))
    print("  >=3 (complicated course): %.1f%%"
          % (100 * d["daniel_ge3"].mean()))
    print("")
    print("  T-inversion component (0-12):"
          " median %.1f  max %.0f"
          % (d["daniel_twi"].dropna().median(),
             d["daniel_twi"].dropna().max()))
    print("  correlation with shopp_count:"
          " %.3f"
          % d[["daniel_score",
               "shopp_count"]].corr()
          .iloc[0, 1])

print("")
print("FIX: Tp-e formula")
for c in ("tpe_ms", "tpe_qt", "qt_ms",
          "qtc_hodges"):
    if c in d.columns:
        v = d[c].dropna()
        print("  %-12s n=%4d (%2.0f%%)"
              "  median %7.2f"
              % (c, len(v),
                 100.0 * len(v) / len(d),
                 v.median() if len(v)
                 else np.nan))
print("  v7 gave tpe_ms 7%, tpe_qt 5%"
      " on the double-subtraction bug")

print("")
print("FIX: TP window scaled by RR")
if "tp_win_ms" in d.columns:
    v = d["tp_win_ms"].dropna()
    print("  start before R: median %.0f ms"
          "  range %.0f-%.0f"
          % (v.median(), v.min(), v.max()))

print("")
print("FIX: aVR at 0.05 mV")
if "shopp_ste_avr" in d.columns:
    print("  prevalence %.3f"
          % d["shopp_ste_avr"].mean())

print("")
print("SHOPP FLAGS")
for c in ["shopp_hr100", "shopp_s1q3t3",
          "shopp_rbbb", "shopp_twi_v14",
          "shopp_ste_avr", "shopp_afib",
          "sinus_tach", "irbbb",
          "rv_strain", "p_pulmonale"]:
    if c in d.columns:
        print("  %-16s %.3f"
              % (c, d[c].dropna().mean()))

print("")
print("T INVERSION DEPTH BANDS")
for c in ("n_twi_v14", "n_twi_v14_deep",
          "n_twi_v14_2mm"):
    if c in d.columns:
        print("  %-16s mean %.2f leads"
              % (c, d[c].dropna().mean()))
print("  Witting: 2 mm inversions in III, aVF,"
      " V1 and V2 gave LR+ 16, specificity 100%")

print("")
print("KEY MEASUREMENTS")
for c in sorted([c for c in d.columns
                 if c in GATE]):
    v = d[c].dropna()
    if len(v):
        print("  %-14s n=%4d (%2.0f%%)"
              "  median %8.2f"
              % (c, len(v),
                 100.0 * len(v) / len(d),
                 v.median()))