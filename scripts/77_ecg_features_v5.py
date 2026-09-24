"""ECG measurements v5.

Record-level waveform measurements at the native
500 Hz:
  QRS bounds    12 ms hold at 3x noise, capped at
                200 ms from the R peak
  baselines     TP for the P wave, PR for ST and
                QRS amplitudes, since ST is
                conventionally measured against
                the PR segment
  P wave        duration measured on lead II at
                15%; PR taken from P onset
  QRS-T angle   spatial, via the Kors transform
  voltages      Sokolow-Lyon and Cornell
  RBBB          from the CRBBB logit
  Poincare      SD1/SD2 ratio

Every measurement passes a physiological gate
before being written; a value outside its range
is set to missing rather than passed to the
model.
Run in venv_ecg only.

OUTPUT FILES
  data\\processed\\ecg_derived_v5_record.csv
  data\\processed\\ecg_derived_v5.csv
  results\\ecg_v5_log.txt
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
    PROC, "ecg_derived_v5_record.csv")
DEST = os.path.join(PROC, "ecg_derived_v5.csv")

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

# physiological gates: outside range -> missing
GATE = {"hr_mean": (25, 220),
        "rr_sd": (0, 500),
        "rmssd": (0, 400),
        "pc_sd1": (0, 300),
        "pc_sd2": (0, 400),
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

NEG_BAD = ("s_wave_i", "q_wave_iii",
           "st_min", "t_v1", "t_v2", "t_v3",
           "t_v4", "t_v5", "t_v6",
           "t_wave_iii")
ANG = ("qrs_axis", "t_axis", "p_axis")


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
    i = np.convolve(d ** 2, np.ones(w) / w,
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
            d["pc_area"] = float(
                np.pi * s1 * s2)
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

    # TWO BASELINES
    # TP: quietest region well before the P wave
    tp_lo = max(0, r - int(0.34 * fs))
    tp_hi = max(tp_lo + 20, r - int(0.26 * fs))
    mb_tp = raw - np.median(raw[tp_lo:tp_hi],
                            axis=0)
    # PR: the segment just before QRS onset,
    # which is the clinical convention for ST
    pr_lo = max(0, r - int(0.10 * fs))
    pr_hi = max(pr_lo + 10, r - int(0.05 * fs))
    mb = raw - np.median(raw[pr_lo:pr_hi],
                         axis=0)

    vm = np.sqrt(np.sum(mb ** 2, axis=1))
    base = float(np.median(vm[pr_lo:pr_hi]))
    noise = float(np.std(vm[pr_lo:pr_hi])) \
        + 1e-9
    vm_tp = np.sqrt(np.sum(mb_tp ** 2, axis=1))
    base_tp = float(np.median(
        vm_tp[tp_lo:tp_hi]))
    noise_tp = float(np.std(
        vm_tp[tp_lo:tp_hi])) + 1e-9

    # ---- QRS bounds: REVERTED settings ----
    thr = base + max(3.0 * noise,
                     0.02 * (vm[r] - base))
    hold = int(0.012 * fs)
    cap = int(0.10 * fs)          # 100 ms each
    q = r                          # side, so
    while q > hold and r - q < cap:   # 200 ms
        if np.all(vm[q - hold:q] <= thr):
            break
        q -= 1
    s_ = r
    while (s_ < len(vm) - hold - 1
           and s_ - r < cap):
        if np.all(vm[s_:s_ + hold] <= thr):
            break
        s_ += 1
    d["qrs_ms"] = (s_ - q) / fs * 1000.0

    w = int(0.05 * fs)
    qv = mb[max(0, r - w):r + w].sum(axis=0)
    d["qrs_axis"] = axis_of(qv[L_I], qv[L_AVF])
    d["rad"] = int(d["qrs_axis"] > 90)

    # ---- T wave, per lead ----
    tl = s_ + int(0.02 * fs)
    th = min(len(mb), s_ + int(0.42 * fs))
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

        # Tp-e on lead II, reverted threshold
        t2 = mb[tl:th, L_II]
        tp = int(np.argmax(np.abs(t2)))
        amp = abs(t2[tp])
        te = tp
        if amp > 0.02:
            k = tp
            while (k < len(t2) - 2
                   and abs(t2[k]) > 0.10 * amp):
                k += 1
            te = k
        d["tpe_ms"] = (te - tp) / fs * 1000.0
        qt = (tl + te - q) / fs * 1000.0
        d["qt_ms"] = qt
        if qt > 0:
            d["tpe_qt"] = d["tpe_ms"] / qt
        if np.isfinite(hr) and hr > 0:
            d["qtc_hodges"] = qt + 1.75 * (
                hr - 60.0)
            d["qtc_over451"] = int(
                d["qtc_hodges"] > 451)

    # ---- spatial QRS-T ----
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

    # ---- P wave: TP baseline, lead II ----
    p_hi = max(0, q - int(0.01 * fs))
    p_lo = max(0, q - int(0.30 * fs))
    if p_hi - p_lo > 20:
        pv = vm_tp[p_lo:p_hi] - base_tp
        pi = int(np.argmax(pv))
        if pv[pi] > 2.0 * noise_tp:
            p2 = mb_tp[p_lo:p_hi, L_II]
            pa = float(p2[pi])
            d["p_amp_ii"] = pa
            d["p_pulmonale"] = int(pa > 0.25)
            d["p_axis"] = axis_of(
                mb_tp[p_lo + pi, L_I],
                mb_tp[p_lo + pi, L_AVF])
            # duration on lead II at 15%
            a2 = np.abs(p2)
            h = max(0.15 * abs(pa), 0.01)
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

    # ---- ST at J+60, PR baseline ----
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

    # ---- S1Q3T3, PR baseline ----
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

    # ---- PHYSIOLOGICAL GATE ----
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
                        "scp_CRBBB")
            if c in q.columns]
    d = d.merge(q[cols], on="rec", how="left")
    d["shopp_afib"] = (
        d["scp_AFIB"] > 0).astype(int)
    if "scp_CRBBB" in d.columns:
        d["shopp_rbbb"] = (
            d["scp_CRBBB"] > 0).astype(int)
    d = d.drop(columns=[c for c in
                        ("scp_AFIB",
                         "scp_CRBBB")
                        if c in d.columns])
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
print("SHOPP FLAG PREVALENCE")
print("  literature: tachycardia 38%,"
      " T-inv V1 38%, ST elev aVR 36%,"
      " RBBB 5-15%, AF 5-15%")
for c in ["shopp_hr100", "shopp_s1q3t3",
          "shopp_rbbb", "shopp_twi_v14",
          "shopp_ste_avr", "shopp_afib",
          "rv_strain", "p_pulmonale"]:
    if c in d.columns:
        print("  %-16s %.3f"
              % (c, d[c].dropna().mean()))

print("")
print("MEASUREMENTS  (run 2 in brackets)")
PREV = {"pr_ms": 114.0, "qrs_ms": 434.0,
        "tpe_ms": 30.0, "p_dur_ms": 34.0,
        "qrs_frag_leads": 5.0,
        "p_amp_ii": 0.01}
for c in ["hr_mean", "qrs_ms", "qt_ms",
          "qtc_hodges", "tpe_ms", "tpe_qt",
          "pr_ms", "p_dur_ms", "p_amp_ii",
          "pr_seg_ms", "p_axis",
          "qrst_spatial", "qrst_frontal",
          "pc_ratio", "rs_ratio_v1",
          "sokolow_lyon", "cornell",
          "qrs_frag_leads"]:
    if c not in d.columns:
        continue
    v = d[c].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if not len(v):
        print("  %-15s ALL MISSING" % c)
        continue
    lo_, hi_ = GATE.get(c, (-np.inf, np.inf))
    was = ("  (was %.1f)" % PREV[c]
           if c in PREV else "")
    print("  %-15s n=%4d (%.0f%%)"
          "  median %7.2f  IQR %.1f-%.1f%s"
          % (c, len(v),
             100.0 * len(v) / len(d),
             v.median(), v.quantile(0.25),
             v.quantile(0.75), was))

print("")
print("GATE REJECTIONS")
for c in ["qrs_ms", "qt_ms", "tpe_ms",
          "pr_ms", "p_dur_ms"]:
    if c in d.columns:
        print("  %-13s %.1f%% set to missing"
              % (c, 100.0 * float(
                  d[c].isna().mean())))

if {"qrst_spatial",
        "qrst_frontal"} <= set(d.columns):
    q2 = d[["qrst_spatial",
            "qrst_frontal"]].dropna()
    print("")
    print("spatial vs frontal QRS-T: %.3f"
          % q2.corr().iloc[0, 1])