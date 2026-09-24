"""Rebuild the multimodal model using the
source-CV tuned EHR and CTPA modalities. Each
combination is evaluated on its own complete-case
cohort, so combinations that do not need CXR are
not restricted to the 32% of admissions that have
one. The ECG and CXR predictions are reused as
stored, so those modalities keep their original
regularisation while EHR and CTPA are source-CV
tuned.
Run in venv (analysis).
"""
import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC = f.PROC
FIG = f.FIG
SCALE = "std"

OUTS = ["death_30d", "composite_30d", "cv_first"]
STORED = {"ecg": "p_ecg_harm_%s.csv",
          "cxr": "p_cxr_harm_%s.csv"}
MIN_N = 200
MIN_EV = 30


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def load_stored(tag, oc, keys):
    fp = os.path.join(FIG, STORED[tag] % oc)
    if not os.path.exists(fp):
        return None
    d = pd.read_csv(fp)
    if "hadm_id" not in d.columns:
        return None
    col = "p_" + tag
    if col not in d.columns:
        num = [c for c in d.columns
               if c not in ("hadm_id",
                            "subject_id")
               and pd.api.types
               .is_numeric_dtype(d[c])]
        if not num:
            return None
        col = num[0]
    d = d[["hadm_id", col]].dropna()
    d = d.drop_duplicates("hadm_id")
    d.columns = ["hadm_id", tag]
    m = keys.merge(d, on="hadm_id", how="left")
    return m[tag].values


def zs_ehr_ctpa(sdf, ys, tdf):
    se, sc_ = f.blocks(sdf)
    te, tc = f.blocks(tdf)
    a, b = f.prep(se, te, SCALE)
    c, d = f.prep(sc_, tc, SCALE)
    me = f.fit_lr(a, ys, "ehr")
    mc = f.fit_lr(c, ys, "ctpa")
    return (me.predict_proba(b)[:, 1],
            mc.predict_proba(d)[:, 1],
            me.predict_proba(a)[:, 1],
            mc.predict_proba(c)[:, 1])


def tune_w(sp, ys, step=0.05):
    """Grid over weights, fitted on SOURCE
    predictions only."""
    k = len(sp)
    best, bw = -1.0, tuple([1.0 / k] * k)
    if k == 2:
        for w in np.arange(0, 1.001, step):
            a = roc_auc_score(
                ys, w * sp[0] + (1 - w) * sp[1])
            if a > best:
                best, bw = a, (w, 1 - w)
    elif k == 3:
        for w1 in np.arange(0, 1.001, step):
            for w2 in np.arange(
                    0, 1.001 - w1, step):
                w3 = 1 - w1 - w2
                a = roc_auc_score(
                    ys, w1 * sp[0] + w2 * sp[1]
                    + w3 * sp[2])
                if a > best:
                    best, bw = a, (w1, w2, w3)
    return bw


ins = f.load_inspect()
mim = f.load_mimic()

rows = []
for oc in OUTS:
    sdf, ys = f.labels(ins, oc)
    tdf, yt = f.labels(mim, oc)
    keys = tdf[["subject_id",
                "hadm_id"]].reset_index(drop=True)
    grp = tdf["gid"].values

    pe, pc, se_, sc_ = zs_ehr_ctpa(sdf, ys, tdf)
    P = {"ehr": pe, "ctpa": pc}
    S = {"ehr": se_, "ctpa": sc_}
    for tag in STORED:
        v = load_stored(tag, oc, keys)
        if v is not None and \
                np.isfinite(v).sum() > 100:
            P[tag] = v

    print("")
    print("=" * 62)
    print("%s  n=%d ev=%d" % (oc, len(yt),
                              int(yt.sum())))
    for t in sorted(P):
        print("  %-5s coverage %.3f"
              % (t, float(np.isfinite(
                  P[t]).mean())))
    print("")
    print("  %-18s %5s %4s %7s %7s %s"
          % ("model", "n", "ev", "AUC",
             "vsEHR", "95% CI"))

    names = sorted(P)
    combos = []
    for r_ in range(1, len(names) + 1):
        combos += list(combinations(names, r_))

    for cb in combos:
        # complete cases for this combination
        ok = np.ones(len(yt), dtype=bool)
        for t in cb:
            ok &= np.isfinite(P[t])
        if "ehr" not in cb:
            ok &= np.isfinite(P["ehr"])
        n = int(ok.sum())
        ev = int(yt[ok].sum())
        if n < MIN_N or ev < MIN_EV:
            print("  %-18s %5d %4d  skipped"
                  % ("+".join(cb), n, ev))
            continue

        y = yt[ok]
        g = grp[ok]
        ref = _rank(P["ehr"][ok])
        ra = roc_auc_score(y, ref)

        if len(cb) == 1:
            p = _rank(P[cb[0]][ok])
            w = None
        else:
            if all(t in S for t in cb):
                sp = [_rank(S[t]) for t in cb]
                w = tune_w(sp, ys)
            else:
                w = tuple([1.0 / len(cb)]
                          * len(cb))
            p = sum(wi * _rank(P[t][ok])
                    for wi, t in zip(w, cb))

        auc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        if tuple(cb) == ("ehr",):
            gn = lo = hi = 0.0
        else:
            gn, lo, hi, _ = f.boot_diff(
                y, p, ref, g)
        star = ("*" if (lo > 0 or hi < 0)
                else " ")
        print("  %-18s %5d %4d  %.4f  %+.4f"
              " [%+.4f,%+.4f] %s"
              % ("+".join(cb), n, ev, auc, gn,
                 lo, hi, star))
        rows.append({
            "outcome": oc, "model": "+".join(cb),
            "n_mod": len(cb), "n": n, "ev": ev,
            "auc": auc, "ap": ap, "ehr_ref": ra,
            "gain": gn, "lo": lo, "hi": hi,
            "weights": (str([round(x, 2)
                             for x in w])
                        if w else ""),
            "sig": int(lo > 0 or hi < 0)})

r = pd.DataFrame(rows)
r.to_csv(os.path.join(PROC,
                      "three_modality.csv"),
         index=False)

print("")
print("=" * 62)
print("BEST PER OUTCOME (any cohort size)")
for oc, sub in r.groupby("outcome"):
    b = sub.loc[sub["auc"].idxmax()]
    print("  %-14s %-18s %.4f  %+.4f"
          "  n=%d ev=%d  w=%s"
          % (oc, b["model"], b["auc"],
             b["gain"], b["n"], b["ev"],
             b["weights"]))

print("")
print("BEST ON THE FULL COHORT (no CXR)")
nx = r[~r["model"].str.contains("cxr")]
for oc, sub in nx.groupby("outcome"):
    b = sub.loc[sub["auc"].idxmax()]
    print("  %-14s %-18s %.4f  %+.4f"
          "  n=%d ev=%d  w=%s"
          % (oc, b["model"], b["auc"],
             b["gain"], b["n"], b["ev"],
             b["weights"]))

print("")
print("DEATH_30D, full cohort, sorted by AUC")
d = nx[nx["outcome"] == "death_30d"]
print(d.sort_values("auc", ascending=False)
      .round(4).to_string(index=False))

print("")
print("DISSERTATION COMPARISON")
print("  WMEAN3-CTPA: 0.8715 on n=1703,"
      " 157 events, death_30d")
b = d.loc[d["auc"].idxmax()]
print("  this rebuild: %.4f on n=%d, %d events"
      % (b["auc"], b["n"], b["ev"]))
print("  NOTE: different cohort, so this is a"
      " reference point rather than a")
print("  like-for-like comparison.")

print("")
print("saved three_modality.csv", r.shape)