"""ECG measurements v6: three landmark fixes
plus the weighted Daniel score.

FIXES
  qrs_ms    was 124 ms with the 100 ms cap
            binding. Offset now found by the
            slope returning to near zero rather
            than amplitude crossing a threshold.
            The QRS ends where deflection stops
            changing rapidly; the ST segment
            continues at a non-zero level but
            with a shallow slope, which an
            amplitude rule cannot distinguish.
  tpe_ms    40.7% rejected. Now the tangent
            method: fit the steepest descent
            after the T peak over a 12 ms window
            and extrapolate to baseline. Far less
            sensitive to where the tail flattens
            than a fixed-fraction rule.
  p_dur_ms  28.2% rejected. Threshold lowered to
            10% on a 40 Hz low-passed lead II,
            since the P wave is low-amplitude and
            a 15% cut on a noisy trace ends
            early.

ADDED: the weighted Daniel score, 21 points.
  sinus tachycardia          2
  incomplete RBBB            2
  complete RBBB              3
  T inversion V1-V4       0-12  (graded by
                                 depth: 1 point
                                 <1mm, 2 for
                                 1-2mm, 3 for
                                 >2mm, per lead)
  S wave in lead I           0  (scored zero)
  Q wave in lead III         1
  inverted T in lead III     1
  full S1Q3T3 complex        2

The graded T component carries 12 of the 21
points, which is why the unweighted Shopp count
added nothing while its components did.

Run in venv_ecg only.

OUTPUT FILES
  data\\processed\\ecg_derived_v6_record.csv
  data\\processed\\ecg_derived_v6.csv
  results\\ecg_v6_log.txt
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
    PROC, "ecg_derived_v6_record.csv")
DEST = os.path.join(PROC, "ecg_derived_v6.csv")

FS = 500
L_I, L_II, L_III = 0, 1, 2
L_AVR, L_AVF, L_AVL = 3, 4, 5
V = {i: 6 + i for i in range(6)}

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


def bpf(x, lo, hi, fs=FS, order=3):
    b, a = sg.butter(
        order, [lo / (fs / 2), hi / (fs / 2)],
        btype="band")
    return sg.filtfilt(b, a, x)


def lpf(x, hi, fs=FS, order=3):
    b, a = sg.butter(order, hi / (fs / 2),
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


def axis_of(x, y):
    return float(np.degrees(np.arctan2(y, x)))


def qrs_by_slope(vm, r, fs=FS):
    """QRS bounds by SLOPE, not amplitude.

    The QRS ends where deflection stops changing
    rapidly. The ST segment continues at a
    non-zero level but with a shallow slope, so
    an amplitude threshold cannot separate them
    and runs on into the T wave.
    """
    sl = np.abs(np.gradient(
        lpf(vm, 40.0, fs)))
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


def t_end_tangent(x, start, fs=FS):
    """T end by the tangent method: steepest
    descent after the T peak, extrapolated to
    baseline. Standard, and far more robust than
    a fixed-fraction rule."""
    if len(x) - start < int(0.08 * fs):
        return None, None
    seg = x[start:]
    base = float(np.median(seg[-int(0.03 * fs):]
                           )) if len(seg) > \
        int(0.03 * fs) else 0.0
    dv = seg - base
    tp = int(np.argmax(np.abs(dv)))
    if tp >= len(seg) - 8:
        return None, None
    w = max(3, int(0.012 * fs))
    tail = dv[tp:]
    if len(tail) < w + 3:
        return None, None
    sl = np.array([
        np.polyfit(np.arange(w),
                   tail[i:i + w], 1)[0]
        for i in range(len(tail) - w)])
    k = (int(np.argmin(sl)) if dv[tp] > 0
         else int(np.argmax(sl)))
    if abs(sl[k]) < 1e-9:
        return None, None
    y0 = tail[k + w // 2]
    step = -y0 / sl[k]
    if not np.isfinite(step) or step < 0:
        return None, None
    end = k + w // 2 + step
    end = int(min(max(end, tp + 1),
                  len(tail) - 1))
    return tp, tp + end - (k + w // 2) + \
        int(step) if False else tp + int(end)


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
    d["shopp_hr100"] = (int(hr > 100)
                        if np.isfinite(hr)
                        else 0)

    pre, post = int(0.35 * fs), int(0.50 * fs)
    raw = median_beat(sig, pk, pre, post)
    if raw is None:
        return d
    r = pre

    tp_lo = max(0, r - int(0.34 * fs))
    tp_hi = max(tp_lo + 20, r - int(0.26 * fs))
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

    # ---- FIX 1: QRS by slope ----
    q, s_ = qrs_by_slope(vm, r, fs)
    d["qrs_ms"] = (s_ - q) / fs * 1000.0

    w = int(0.05 * fs)
    qv = mb[max(0, r - w):r + w].sum(axis=0)
    d["qrs_axis"] = axis_of(qv[L_I], qv[L_AVF])
    d["rad"] = int(d["qrs_axis"] > 90)

    # ---- T wave ----
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
        d["shopp_twi_v14"] = int(
            d["n_twi_v14"] == 4)

        # ---- FIX 2: Tp-e by tangent ----
        t2 = lpf(mb[:, L_II], 40.0, fs)
        tp, te = t_end_tangent(t2, tl, fs)
        if tp is not None and te is not None:
            d["tpe_ms"] = ((te - tp) / fs
                           * 1000.0)
            qt = (tl + te - q) / fs * 1000.0
            d["qt_ms"] = qt
            if qt > 0:
                d["tpe_qt"] = (d["tpe_ms"]
                               / qt)
            if np.isfinite(hr) and hr > 0:
                d["qtc_hodges"] = qt + 1.75 * (
                    hr - 60.0)
                d["qtc_over451"] = int(
                    d["qtc_hodges"] > 451)

    xyz = kors(mb)
    qvec = xyz[max(0, r - w):r + w].sum(axis=0)
    if tv is not None and th - tl > 10:
        ts = xyz[tl:th]
        tpk = int(np.argmax(
            np.sum(ts ** 2, axis=1)))
        tvec = ts[tpk]
        nq = np.linalg.norm(qvec)
        nt = np.linalg.norm(tvec)
        if nq > 1e-9 and nt > 1e-9:
            d["qrst_spatial"] = float(
                np.degrees(np.arccos(np.clip(
                    np.dot(qvec, tvec)
                    / (nq * nt), -1.0, 1.0))))
            d["qrs_vmag"] = float(nq)
            d["t_vmag"] = float(nt)

    # ---- FIX 3: P duration, smoothed, 10% ----
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
            st[L_AVR] > 0.02)

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

    # ---- GRADED T-inversion for Daniel ----
    # 1 point <1mm, 2 for 1-2mm, 3 for >2mm,
    # summed across V1-V4, capped at 12
    if tv is not None:
        pts = 0
        for k in range(4):
            a_ = tv[V[k]]
            if a_ >= 0:
                continue
            dep = abs(a_)
            pts += (1 if dep < 0.1
                    else 2 if dep < 0.2 else 3)
        d["dan_twi_points"] = int(min(pts, 12))

    nfrag = 0
    for c in range(mb.shape[1]):
        seg = mb[q:s_, c]
        if len(seg) < 10:
            continue
        pa = float(np.max(np.abs(seg)))
        if pa < 0.1:
            continue
        pks, _ = sg.find_peaks(
            np.abs(seg), height=0.15 * pa,
            distance=int(0.02 * fs))
        if len(pks) > 2 and len(
                set(np.sign(seg[pks])
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

# ---- WEIGHTED DANIEL SCORE ----
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
    sinus = (d.get("scp_SR", 0) > 0).astype(int)

    dan = np.zeros(len(d))
    # sinus tachycardia, 2
    dan += 2 * ((d["shopp_hr100"] == 1)
                & (sinus == 1)).astype(int)
    # incomplete RBBB 2, complete RBBB 3
    dan += 2 * d["irbbb"]
    dan += 3 * d["shopp_rbbb"]
    # graded T inversion, 0-12
    dan += d["dan_twi_points"].fillna(0)
    # S in lead I scores 0 points
    # Q in lead III, 1
    dan += 1 * d["q3_present"].fillna(0)
    # inverted T in lead III, 1
    dan += 1 * d["t3_inverted"].fillna(0)
    # full S1Q3T3 complex, 2
    dan += 2 * d["shopp_s1q3t3"].fillna(0)
    d["daniel_weighted"] = dan

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
    print("weighted Daniel score built")
d.to_csv(REC_DEST, index=False)

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
print("THE THREE FIXES  (v5 in brackets)")
PREV = {"qrs_ms": (124.0, 2.7),
        "tpe_ms": (80.0, 40.7),
        "p_dur_ms": (78.0, 28.2)}
for c, lo, hi in [("qrs_ms", 80, 120),
                  ("tpe_ms", 60, 110),
                  ("p_dur_ms", 80, 110)]:
    if c not in d.columns:
        continue
    v = d[c].replace(
        [np.inf, -np.inf], np.nan).dropna()
    pm, pr_ = PREV[c]
    print("  %-11s median %6.1f (was %.1f)"
          "   rejected %.1f%% (was %.1f%%)"
          "   target %d-%d"
          % (c, v.median(), pm,
             100.0 * float(d[c].isna().mean()),
             pr_, lo, hi))

print("")
print("WEIGHTED DANIEL SCORE (0-21)")
if "daniel_weighted" in d.columns:
    v = d["daniel_weighted"].dropna()
    print("  median %.1f  IQR %.1f-%.1f"
          "  max %.0f"
          % (v.median(), v.quantile(0.25),
             v.quantile(0.75), v.max()))
    print("  distribution:")
    print(v.value_counts().sort_index()
          .head(12).to_string())
    print("")
    print("  correlation with the unweighted"
          " Shopp count: %.3f"
          % d[["daniel_weighted",
               "shopp_count"]].corr()
          .iloc[0, 1])
    print("  graded T points median: %.1f"
          % d["dan_twi_points"].dropna()
          .median())

print("")
print("ALL MEASUREMENTS")
for c in sorted([c for c in d.columns
                 if c in GATE]):
    v = d[c].dropna()
    if not len(v):
        continue
    print("  %-14s n=%4d (%2.0f%%)"
          "  median %8.2f"
          % (c, len(v),
             100.0 * len(v) / len(d),
             v.median()))