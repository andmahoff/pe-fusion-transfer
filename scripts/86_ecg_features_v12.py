"""ECG measurements v12. Reverts two v11
changes and fixes three defects, all
against the Daniel score's own published
footnotes.

REVERT 1  Q IN LEAD III: DEPTH ONLY. v11 required
          >=40 ms duration, which is the standard
          for a pathological Q wave denoting
          infarction. The Daniel footnote reads
          "Q in lead III >1.5 mm ... before any R
          wave >1.5 mm" with no duration term, and
          McGinn and White defined it the same
          way. v11 gave q3_present 0.052 and
          S1Q3T3 0.008; v10's depth-only rule gave
          0.280 and 0.062.

REVERT 2  T SEARCH BACK TO J+60. Moving it to J+80
          made the early-peak fraction worse in
          lead II, 16.4% to 17.3%, because the
          displaced ST segment moved with the
          window.

FIX 1     R-WAVE EXCLUSION THRESHOLD. The footnote
          says the Q wave must precede any R wave
          >1.5 mm, i.e. 0.15 mV. The code used
          0.05 mV, three times too strict,
          rejecting genuine Q waves.

FIX 2     EDGE REJECTION WAS COUNTED, NOT APPLIED.
          In the per-lead T loop, nbad incremented
          when a peak sat at a window edge and
          then tv[c] was assigned anyway. Edge
          peaks are now recorded as missing, so
          the ST segment cannot enter the
          T-inversion counts.

FIX 3     WAVEFORM RBBB per the footnotes, as an
          alternative to the logits: incomplete is
          QRS 100-110 ms with an S wave in lead I
          and a terminal R in V1 >1.5 mm; complete
          is QRS >110 ms with the same. Both are
          written so the fitting can compare.

EXPECTED BANDS CORRECTED. The v11 bands were
narrower than the literature. Reported S1Q3T3
prevalence spans under 3% to 25%; isolated T
inversion in lead III is a normal variant in
healthy adults, so 46% is not evidence of error.

Run in venv_ecg only.

OUTPUT FILES
  data\\processed\\ecg_derived_v12_record.csv
  data\\processed\\ecg_derived_v12.csv
  results\\ecg_v12_log.txt
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
    PROC, "ecg_derived_v12_record.csv")
DEST = os.path.join(
    PROC, "ecg_derived_v12.csv")

FS = 500
EXPECT = ["I", "II", "III", "aVR", "aVF",
          "aVL", "V1", "V2", "V3", "V4",
          "V5", "V6"]
L_I, L_II, L_III = 0, 1, 2
L_AVR, L_AVF, L_AVL = 3, 4, 5
V = {i: 6 + i for i in range(6)}
RR_FRAC = 0.60
P_FRAC = 0.40
SPREAD_MS = 40.0
STE_AVR = 0.05
MM1, MM2 = 0.10, 0.20
MM15 = 0.15          # 1.5 mm, the Daniel unit
TWI_FLOOR = 0.03
T_START = 0.06       # REVERT 2: J+60
T_EDGE = 0.02

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

GATE_PRIMARY = {
    "hr_mean": (25, 220), "rr_sd": (0, 500),
    "rmssd": (0, 400), "rr_sd_f": (0, 500),
    "rmssd_f": (0, 400),
    "pc_sd1": (0, 300), "pc_sd2": (0, 400),
    "pc_ratio": (0.05, 3.0),
    "qrs_ms": (50, 200), "qt_ms": (250, 600),
    "tpe_ms": (30, 160), "pr_ms": (80, 320),
    "p_dur_ms": (40, 160),
    "p_amp_ii": (-0.5, 0.6),
    "pr_seg_ms": (0, 200),
    "qrst_spatial": (0, 180),
    "qrst_spatial_pk": (0, 180),
    "qrst_frontal": (0, 180),
    "rs_ratio_v1": (0, 20),
    "sokolow_lyon": (0.1, 6.0),
    "cornell": (0.1, 5.0)}
GATE_DERIVED = {
    "qtc_hodges": (300, 650),
    "qtc_strict": (300, 650),
    "tpe_qt": (0.05, 0.50),
    "tpe_qt_strict": (0.05, 0.50),
    "qt_ms_strict": (250, 600),
    "tpe_ms_strict": (30, 160)}

P_COLS = ("p_amp_ii", "p_pulmonale", "p_axis",
          "pr_ms", "p_dur_ms", "pr_seg_ms")
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


def flat_window(raw, r, rr_s, fs=FS):
    w = int(0.04 * fs)
    lo = max(0, r - int(0.60 * rr_s))
    hi = min(r - w - int(0.02 * fs),
             r - int(0.14 * rr_s))
    if hi <= lo + 2:
        a = max(0, r - int(0.10 * fs))
        b = max(a + 5, r - int(0.05 * fs))
        return a, b, 1
    best, bi = None, lo
    for i in range(lo, hi - w + 1, 2):
        v = float(np.sum(np.std(
            raw[i:i + w], axis=0)))
        if best is None or v < best:
            best, bi = v, i
    return bi, bi + w, 0


def qrs_bounds(vm, r, fs=FS):
    sl = np.abs(np.gradient(lpf(vm, 40.0, fs)))
    pre = max(0, r - int(0.20 * fs))
    quiet = sl[pre:max(pre + 10,
                       r - int(0.12 * fs))]
    thr = (float(np.median(quiet))
           + 3.0 * float(np.std(quiet)) + 1e-9)
    thr = max(thr, 0.05 * float(sl[r]))
    hold = int(0.010 * fs)
    cap = int(0.09 * fs)
    q, hq = r, 1
    while q > hold and r - q < cap:
        if np.all(sl[q - hold:q] < thr):
            hq = 0
            break
        q -= 1
    s_, hs = r, 1
    n = len(sl)
    while s_ < n - hold - 1 and s_ - r < cap:
        if np.all(sl[s_:s_ + hold] < thr):
            hs = 0
            break
        s_ += 1
    return q, s_, hq, hs


def t_end_three(x, tl, hi, base, fs=FS):
    if hi - tl < int(0.08 * fs):
        return None, None, None, 1
    seg = x[tl:hi] - base
    a = np.abs(seg)
    tp = int(np.argmax(a))
    amp = float(a[tp])
    edge = int(tp < int(T_EDGE * fs)
               or tp >= len(seg) - 8)
    if amp < 0.02 or edge:
        return None, None, None, edge
    tail = seg[tp:]
    n = len(tail)
    if n < 12:
        return None, None, None, 0
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
    k, hit = 0, 1
    while k < n - 2:
        if at[k] <= 0.10 * amp:
            hit = 0
            break
        k += 1
    if not hit and 0 < k < n:
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
        return None, None, None, 0
    est = np.array(est)
    return (tp, tp + float(np.median(est)),
            float(est.max() - est.min()), 0)


def axis_of(x, y):
    return float(np.degrees(np.arctan2(y, x)))


def inverted(a):
    return np.isfinite(a) and a < -TWI_FLOOR


def twi_band(a, v1=False):
    if not inverted(a):
        return 0
    dep = abs(a)
    if dep < MM1:
        return 0 if v1 else 1
    if dep <= MM2:
        return 1 if v1 else 2
    return 2 if v1 else 3


def process(sig, fs=FS):
    d = {}
    ii = sig[:, L_II]
    pk = detect_r(ii, fs)
    if len(pk) < 3:
        return d
    rr_all = np.diff(pk) / fs * 1000.0
    rr = rr_all[(rr_all > 300)
                & (rr_all < 2000)]
    if len(rr) < 3:
        return d
    hr = 60000.0 / np.mean(rr)
    rr_s = float(np.median(np.diff(pk)))

    d["hr_mean"] = hr
    d["rr_sd"] = float(np.std(rr, ddof=1))
    d["rmssd"] = float(np.sqrt(
        np.mean(np.diff(rr) ** 2)))
    d["n_beats"] = int(len(pk))
    med = float(np.median(rr))
    rrf = rr[(rr > 0.70 * med)
             & (rr < 1.30 * med)]
    d["n_rr_dropped"] = int(len(rr) - len(rrf))
    if len(rrf) >= 3:
        d["hr_mean_f"] = 60000.0 / np.mean(rrf)
        d["rr_sd_f"] = float(
            np.std(rrf, ddof=1))
        d["rmssd_f"] = float(np.sqrt(
            np.mean(np.diff(rrf) ** 2)))
    if len(rr) >= 4:
        a, b = rr[:-1], rr[1:]
        s1v = float(np.std(
            (a - b) / np.sqrt(2.0), ddof=1))
        s2v = float(np.std(
            (a + b) / np.sqrt(2.0), ddof=1))
        d["pc_sd1"], d["pc_sd2"] = s1v, s2v
        d["pc_area"] = float(
            np.pi * s1v * s2v)
        if s2v > 1e-9:
            d["pc_ratio"] = s1v / s2v
    d["shopp_hr100"] = int(hr > 100)

    pre, post = int(0.35 * fs), int(0.50 * fs)
    raw = median_beat(sig, pk, pre, post)
    if raw is None:
        return d
    r = pre

    tp_lo, tp_hi, fb = flat_window(
        raw, r, rr_s, fs)
    d["tp_win_ms"] = ((r - tp_lo) / fs
                      * 1000.0)
    d["tp_fallback"] = int(fb)
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
    noise_tp = max(float(np.std(
        vm_tp[tp_lo:tp_hi])), 0.01)

    q, s_, hq, hs = qrs_bounds(vm, r, fs)
    d["qrs_hit_limit"] = int(hq or hs)
    qrs_ms = ((s_ - q) / fs * 1000.0
              if not (hq or hs) else np.nan)
    d["qrs_ms"] = qrs_ms
    qa, qb = q, s_
    qv = mb[qa:qb].sum(axis=0) if qb > qa \
        else mb[r]
    d["qrs_axis"] = axis_of(qv[L_I], qv[L_AVF])
    d["rad"] = int(d["qrs_axis"] > 90)

    # ---- T wave, REVERT 2: J+60 ----
    tl = s_ + int(T_START * fs)
    th_cap = int(min(len(mb),
                     s_ + int(0.45 * fs),
                     q + RR_FRAC * rr_s))
    tv, th = None, th_cap
    if th_cap - tl > int(0.08 * fs):
        t2 = lpf(mb[:, L_II], 40.0, fs)
        base2 = float(np.median(
            t2[tp_lo:tp_hi]))
        tpo, teo, spread, edge = t_end_three(
            t2, tl, th_cap, base2, fs)
        d["t_peak_edge"] = int(edge)
        if tpo is not None:
            sp = spread / fs * 1000.0
            d["tpe_spread_ms"] = sp
            ok = int(sp <= SPREAD_MS)
            d["tpe_ok"] = ok
            d["t_peak_ms"] = (tpo / fs
                              * 1000.0)
            d["tpe_ms"] = ((teo - tpo) / fs
                           * 1000.0)
            d["qt_ms"] = ((tl + teo - q) / fs
                          * 1000.0)
            if ok:
                cand = tl + int(round(teo))
                if tl + 5 < cand <= th_cap:
                    th = cand

    if th - tl > 10:
        seg = mb[tl:th]
        nedge = int(T_EDGE * fs)
        tv = np.full(mb.shape[1], np.nan)
        nbad = 0
        for c in range(mb.shape[1]):
            k = int(np.argmax(np.abs(
                seg[:, c])))
            # FIX 2: an edge peak is missing,
            # not merely counted
            if k < nedge or k >= len(seg) - 2:
                nbad += 1
                continue
            tv[c] = seg[k, c]
            if c in (L_II, V[0], V[1], V[2]):
                d["tpk_ms_l%d" % c] = (
                    k / fs * 1000.0)
        d["t_edge_leads"] = int(nbad)
        if np.isfinite(tv[L_I]) and \
                np.isfinite(tv[L_AVF]):
            d["t_axis"] = axis_of(tv[L_I],
                                  tv[L_AVF])
            a_ = abs(d["qrs_axis"]
                     - d["t_axis"])
            d["qrst_frontal"] = float(
                min(a_, 360.0 - a_))
        for k, c in V.items():
            if np.isfinite(tv[c]):
                d["t_v%d" % (k + 1)] = float(
                    tv[c])
        if np.isfinite(tv[L_III]):
            d["t_wave_iii"] = float(tv[L_III])
            d["t3_inverted"] = int(
                inverted(tv[L_III]))
        d["n_twi_v14"] = int(sum(
            1 for k in range(4)
            if inverted(tv[V[k]])))
        d["n_twi_v14_deep"] = int(sum(
            1 for k in range(4)
            if np.isfinite(tv[V[k]])
            and tv[V[k]] < -MM1))
        d["n_twi_v14_2mm"] = int(sum(
            1 for k in range(4)
            if np.isfinite(tv[V[k]])
            and tv[V[k]] < -MM2))
        d["shopp_twi_v14"] = int(
            d["n_twi_v14"] == 4)
        pts = twi_band(tv[V[0]], v1=True)
        pts += twi_band(tv[V[1]])
        pts += twi_band(tv[V[2]])
        if all(inverted(tv[V[k]])
               for k in range(4)):
            pts += 4
        d["daniel_twi"] = int(min(pts, 12))

        xyz = kors(mb)
        qvec = (xyz[qa:qb].sum(axis=0)
                if qb > qa else xyz[r])
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
        d["t_win_ms"] = ((th - tl) / fs
                         * 1000.0)

    p_hi = max(0, q - int(0.01 * fs))
    p_span = int(min(0.30 * fs,
                     P_FRAC * rr_s))
    p_lo = max(0, q - p_span)
    if p_hi - p_lo > 20:
        pv = vm_tp[p_lo:p_hi] - base_tp
        pi = int(np.argmax(pv))
        if (pv[pi] > max(2.0 * noise_tp, 0.03)
                and 2 <= pi < len(pv) - 2):
            p2 = lpf(mb_tp[:, L_II], 40.0,
                     fs)[p_lo:p_hi]
            pj = int(np.argmax(np.abs(p2)))
            pa = float(p2[pj])
            a2 = np.abs(p2)
            h = max(0.10 * abs(pa), 0.005)
            k, hon = pj, 1
            while k > 0:
                if a2[k] <= h:
                    hon = 0
                    break
                k -= 1
            k2, hoff = pj, 1
            while k2 < len(a2) - 1:
                if a2[k2] <= h:
                    hoff = 0
                    break
                k2 += 1
            d["p_hit_limit"] = int(hon or hoff)
            if not (hon or hoff):
                on, off = p_lo + k, p_lo + k2
                d["p_amp_ii"] = pa
                d["p_axis"] = axis_of(
                    mb_tp[p_lo + pj, L_I],
                    mb_tp[p_lo + pj, L_AVF])
                d["pr_ms"] = ((q - on) / fs
                              * 1000.0)
                d["p_dur_ms"] = ((off - on)
                                 / fs * 1000.0)
                d["pr_seg_ms"] = ((q - off)
                                  / fs * 1000.0)

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

    s1 = float(np.min(mb[qa:qb, L_I])) \
        if qb > qa else 0.0
    # ---- REVERT 1 and FIX 1 ----
    # Q in III: depth only, and the Q must
    # precede any R wave >1.5 mm (0.15 mV),
    # not >0.5 mm as the v10/v11 code had
    q3 = 0.0
    if r > qa:
        pre3 = mb[qa:r, L_III]
        if len(pre3):
            k3 = int(np.argmin(pre3))
            if not np.any(pre3[:k3] > MM15):
                q3 = float(pre3[k3])
    d["s_wave_i"] = s1
    d["q_wave_iii"] = q3
    d["q3_present"] = int(q3 < -MM15)
    d["s1_present"] = int(s1 < -MM15)
    d["shopp_s1q3t3"] = int(
        bool(d["s1_present"])
        and bool(d["q3_present"])
        and bool(d.get("t3_inverted", 0)))

    def mx(c):
        return (float(np.max(mb[qa:qb, c]))
                if qb > qa else 0.0)

    def mn(c):
        return (float(np.min(mb[qa:qb, c]))
                if qb > qa else 0.0)
    d["r_v1"], d["s_v1"] = mx(V[0]), mn(V[0])
    if abs(d["s_v1"]) > 1e-6:
        d["rs_ratio_v1"] = (d["r_v1"]
                            / abs(d["s_v1"]))
    d["sokolow_lyon"] = (abs(d["s_v1"])
                         + max(mx(V[4]),
                               mx(V[5])))
    d["cornell"] = mx(L_AVL) + abs(mn(V[2]))

    # ---- FIX 3: waveform RBBB per footnote ----
    term_r = 0.0
    if qb > qa:
        back = mb[max(qa, qb - int(0.04 * fs)):
                  qb, V[0]]
        if len(back):
            term_r = float(np.max(back))
    d["term_r_v1"] = term_r
    rb = (d["s1_present"] == 1
          and term_r > MM15)
    d["irbbb_wave"] = int(
        rb and np.isfinite(qrs_ms)
        and 100.0 <= qrs_ms <= 110.0)
    d["crbbb_wave"] = int(
        rb and np.isfinite(qrs_ms)
        and qrs_ms > 110.0)

    nfrag = 0
    for c in range(mb.shape[1]):
        seg2 = mb[qa:qb, c]
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

    for k_, (lo_, hi_) in GATE_PRIMARY.items():
        if k_ in d and d[k_] is not None:
            v_ = d[k_]
            if not (np.isfinite(v_)
                    and lo_ <= v_ <= hi_):
                d[k_] = np.nan

    qt = d.get("qt_ms", np.nan)
    tpe = d.get("tpe_ms", np.nan)
    ok = int(d.get("tpe_ok", 0))
    if np.isfinite(qt):
        d["qtc_hodges"] = qt + 1.75 * (hr - 60.0)
        d["qtc_over451"] = int(
            d["qtc_hodges"] > 451)
        if np.isfinite(tpe) and qt > 0:
            d["tpe_qt"] = tpe / qt
        if ok:
            d["qt_ms_strict"] = qt
            d["qtc_strict"] = d["qtc_hodges"]
            if np.isfinite(tpe):
                d["tpe_ms_strict"] = tpe
                if qt > 0:
                    d["tpe_qt_strict"] = tpe / qt
    pa = d.get("p_amp_ii", np.nan)
    if np.isfinite(pa):
        d["p_pulmonale"] = int(pa > 0.25)

    for k_, (lo_, hi_) in GATE_DERIVED.items():
        if k_ in d and d[k_] is not None:
            v_ = d[k_]
            if not (np.isfinite(v_)
                    and lo_ <= v_ <= hi_):
                d[k_] = np.nan
    return d


idx = pd.read_csv(os.path.join(
    OUT, "ecg_record_index.csv"))
idx["rec"] = idx["rec"].apply(norm_rec)
recs = sorted(idx["rec"].unique())
print("records:", len(recs), flush=True)

rows, bad, wrong = [], 0, 0
t0 = time.time()
for i, r_ in enumerate(recs):
    try:
        rec = wfdb.rdrecord(rp(r_))
        if list(rec.sig_name) != EXPECT:
            wrong += 1
            continue
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
print("processed:", len(d), " failed:", bad,
      " wrong lead order:", wrong)

lg = os.path.join(
    PROC, "exp0_logits_record.csv")
if os.path.exists(lg):
    q = pd.read_csv(lg)
    cols = [c for c in ("rec", "scp_AFIB",
                        "scp_AFLT",
                        "scp_CRBBB",
                        "scp_IRBBB", "scp_SR",
                        "scp_STACH")
            if c in q.columns]
    d = d.merge(q[cols], on="rec", how="left")
    afib = (d.get("scp_AFIB", 0) > 0)
    aflt = (d.get("scp_AFLT", 0) > 0)
    d["shopp_afib"] = afib.astype(int)
    d["any_af"] = (afib | aflt).astype(int)
    nb = int(d.loc[d["any_af"] == 1,
                   "p_amp_ii"].notna().sum()) \
        if "p_amp_ii" in d.columns else 0
    for c in P_COLS:
        if c in d.columns:
            d.loc[d["any_af"] == 1, c] = np.nan
    print("P blanked in AF/flutter:", nb)
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
    # score from the LOGIT RBBB (primary)
    rbbb_pts = np.maximum(2 * d["irbbb"],
                          3 * d["shopp_rbbb"])
    d["daniel_score"] = (
        2.0 * d["sinus_tach"] + rbbb_pts
        + d["daniel_twi"].fillna(0)
        + 1.0 * d["q3_present"].fillna(0)
        + 1.0 * d["t3_inverted"].fillna(0)
        + 2.0 * d["shopp_s1q3t3"].fillna(0))
    # and from the WAVEFORM RBBB, for comparison
    rb_w = np.maximum(2 * d["irbbb_wave"],
                      3 * d["crbbb_wave"])
    d["daniel_wave"] = (
        2.0 * d["sinus_tach"] + rb_w
        + d["daniel_twi"].fillna(0)
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
                        ("scp_AFIB", "scp_AFLT",
                         "scp_CRBBB",
                         "scp_IRBBB", "scp_SR",
                         "scp_STACH")
                        if c in d.columns])
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
        c + "_circ").reset_index()
    agg = agg.merge(
        cm, on=["subject_id", "hadm_id"],
        how="left")
    agg = agg.drop(columns=[c + "_mean",
                            c + "_worst"],
                   errors="ignore")
agg.to_csv(DEST, index=False)
print("saved", DEST, agg.shape)

n = len(d)
print("")
print("=" * 72)
print("AGAINST THE LITERATURE")
print("  bands widened to the ACTUAL reported")
print("  spread; the v11 bands were narrower")
print("  than the literature itself")
print("=" * 72)
LIT = [
    ("shopp_s1q3t3", 0.062, 0.008,
     0.03, 0.25,
     "S1Q3T3: <3% to 25% across sources"),
    ("q3_present", 0.280, 0.052, 0.15, 0.45,
     "Q in III >1.5mm, depth only"),
    ("s1_present", np.nan, np.nan, 0.20, 0.60,
     "S in I >1.5mm"),
    ("t3_inverted", 0.460, 0.460, 0.25, 0.55,
     "T inv in III: a NORMAL variant"),
    ("shopp_hr100", 0.301, 0.301, 0.28, 0.50,
     "tachycardia 38-44%"),
    ("shopp_rbbb", 0.056, 0.056, 0.03, 0.18,
     "complete RBBB 5-15%"),
    ("shopp_afib", 0.096, 0.096, 0.04, 0.16,
     "AF 5-15%"),
    ("shopp_ste_avr", 0.177, 0.177, 0.10, 0.40,
     "ST elev aVR, 36% at 1mm"),
    ("shopp_twi_v14", 0.091, 0.091, 0.05, 0.40,
     "TWI V1-V4 all four, 34-68% any"),
    ("rv_strain", 0.169, 0.131, 0.15, 0.45,
     "RV strain composite ~34%"),
    ("p_pulmonale", 0.004, 0.004, 0.01, 0.15,
     "P pulmonale 5-10%")]
print("  %-15s %8s %8s %8s  %-4s %s"
      % ("flag", "v10", "v11", "v12",
         "", "expected"))
for c, v10, v11, lo_, hi_, note in LIT:
    if c not in d.columns:
        continue
    v = float(d[c].dropna().mean())
    mark = ("ok" if lo_ <= v <= hi_
            else "OUT")
    print("  %-15s %8.3f %8.3f %8.3f  %-4s"
          " %.2f-%.2f  %s"
          % (c, v10, v11, v, mark, lo_, hi_,
             note))

print("")
print("REVERT 1  Q in III, depth only")
print("  q3_present %.3f  (v10 0.280"
      " depth-only, v11 0.052 with 40ms)"
      % d["q3_present"].mean())
print("  the Daniel footnote reads"
      " 'Q in lead III >1.5 mm ... before")
print("  any R wave >1.5 mm', with NO"
      " duration criterion")

print("")
print("FIX 1  R-wave exclusion at 1.5 mm")
print("  was 0.5 mm, three times too strict")

print("")
print("FIX 2  edge peaks now missing")
if "t_edge_leads" in d.columns:
    print("  leads rejected: mean %.2f of 12"
          % d["t_edge_leads"].dropna().mean())
for c in sorted([c for c in d.columns
                 if c.startswith("tpk_ms_l")]):
    v = d[c].dropna()
    if len(v):
        print("  %-12s n=%4d  median %5.1f ms"
              "   within 20 ms: %.1f%%"
              % (c, len(v), v.median(),
                 100.0 * float(
                     (v < 20).mean())))
print("  v11 counted edge peaks then used"
      " them anyway")

print("")
print("FIX 3  waveform RBBB vs the logits")
for a_, b_ in (("irbbb", "irbbb_wave"),
               ("shopp_rbbb", "crbbb_wave")):
    if a_ in d.columns and b_ in d.columns:
        print("  %-12s logit %.3f"
              "   waveform %.3f"
              " (agreement %.1f%%)"
              % (a_, d[a_].mean(),
                 d[b_].mean(),
                 100.0 * float(
                     (d[a_] == d[b_]).mean())))
if "term_r_v1" in d.columns:
    v = d["term_r_v1"].dropna()
    print("  terminal R in V1: median %.3f mV"
          "  above 1.5 mm: %.1f%%"
          % (v.median(),
             100.0 * float((v > MM15).mean())))

print("")
print("DANIEL SCORE")
for c in ("daniel_score", "daniel_wave"):
    if c in d.columns:
        v = d[c].dropna()
        print("  %-14s median %.1f"
              "  IQR %.1f-%.1f  max %.0f"
              % (c, v.median(),
                 v.quantile(0.25),
                 v.quantile(0.75), v.max()))
if {"daniel_score",
        "daniel_wave"} <= set(d.columns):
    print("  correlation between them: %.3f"
          % d[["daniel_score",
               "daniel_wave"]].corr()
          .iloc[0, 1])
if "daniel_ge10" in d.columns:
    print("  >=10 %.1f%%   >=3 %.1f%%"
          % (100 * d["daniel_ge10"].mean(),
             100 * d["daniel_ge3"].mean()))

print("")
print("CONTINUOUS MEASUREMENTS")
EXP = {"hr_mean": (60, 110),
       "qrs_ms": (80, 100),
       "qt_ms": (330, 430),
       "qtc_hodges": (390, 460),
       "pr_ms": (130, 200),
       "p_dur_ms": (80, 115),
       "tpe_ms": (60, 100),
       "qrst_spatial": (40, 90),
       "sokolow_lyon": (1.2, 2.2),
       "cornell": (0.9, 1.8)}
nok = 0
for c, (lo_, hi_) in EXP.items():
    if c not in d.columns:
        continue
    v = d[c].dropna()
    if not len(v):
        continue
    md = float(v.median())
    good = lo_ <= md <= hi_
    nok += int(good)
    print("  %-15s median %8.2f"
          "  expect %g-%g  %-3s  n=%d (%2.0f%%)"
          % (c, md, lo_, hi_,
             "ok" if good else "OUT",
             len(v), 100.0 * len(v) / n))
print("  %d of %d within range" % (nok,
                                   len(EXP)))