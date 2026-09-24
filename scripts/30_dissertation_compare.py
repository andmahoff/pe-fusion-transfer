"""Direct comparison with the dissertation's
WMEAN3-CTPA model. That model tuned its fusion
weights by cross-validation on the MIMIC target,
so this reproduces that protocol and reports it
alongside the strict source-tuned version.
The two differ only in where the weights come
from, so the gap between them is the value of
target-label access.
Run in venv (analysis).
"""
import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC = f.PROC
FIG = f.FIG
SCALE = "std"
SEED = 42
NFOLD = 5
STEP = 0.05

OUTS = ["death_30d", "composite_30d", "cv_first"]
STORED = {"ecg": "p_ecg_harm_%s.csv",
          "cxr": "p_cxr_harm_%s.csv"}

# dissertation reference figures
REF = {"death_30d": 0.8715}


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
    return keys.merge(
        d, on="hadm_id", how="left")[tag].values


def grid_w(ps, y, step=STEP):
    """Weights maximising AUC on the supplied
    predictions and labels."""
    k = len(ps)
    best, bw = -1.0, tuple([1.0 / k] * k)
    if k == 1:
        return (1.0,)
    if k == 2:
        for w in np.arange(0, 1.001, step):
            a = roc_auc_score(
                y, w * ps[0] + (1 - w) * ps[1])
            if a > best:
                best, bw = a, (w, 1 - w)
    elif k == 3:
        for w1 in np.arange(0, 1.001, step):
            for w2 in np.arange(
                    0, 1.001 - w1, step):
                w3 = 1 - w1 - w2
                a = roc_auc_score(
                    y, w1 * ps[0] + w2 * ps[1]
                    + w3 * ps[2])
                if a > best:
                    best, bw = a, (w1, w2, w3)
    else:
        for w1 in np.arange(0, 1.001, 0.1):
            for w2 in np.arange(
                    0, 1.001 - w1, 0.1):
                for w3 in np.arange(
                        0, 1.001 - w1 - w2, 0.1):
                    w4 = 1 - w1 - w2 - w3
                    a = roc_auc_score(
                        y, w1 * ps[0]
                        + w2 * ps[1]
                        + w3 * ps[2]
                        + w4 * ps[3])
                    if a > best:
                        best, bw = a, (w1, w2,
                                       w3, w4)
    return bw


def target_cv_fuse(R, y, grp, cb):
    """Dissertation protocol: weights fitted on
    the target by grouped CV, applied to the
    held-out fold. Uses target labels, so this is
    not zero-shot."""
    ps = [R[t] for t in cb]
    out = np.zeros(len(y))
    ws = []
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    X = np.column_stack(ps)
    for tr, te in cv.split(X, y, grp):
        w = grid_w([p[tr] for p in ps], y[tr])
        ws.append(w)
        out[te] = sum(wi * p[te]
                      for wi, p in zip(w, ps))
    return out, np.mean(ws, axis=0)


def src_fuse(R, S, cb, ys):
    """Strict protocol: weights fitted on SOURCE
    predictions only. Falls back to equal weight
    if any modality has no source score."""
    if all(t in S for t in cb):
        w = grid_w([_rank(S[t]) for t in cb], ys)
    else:
        w = tuple([1.0 / len(cb)] * len(cb))
    return (sum(wi * R[t]
                for wi, t in zip(w, cb)), w)


ins = f.load_inspect()
mim = f.load_mimic()

rows = []
for oc in OUTS:
    sdf, ys = f.labels(ins, oc)
    tdf, yt = f.labels(mim, oc)
    keys = tdf[["subject_id",
                "hadm_id"]].reset_index(drop=True)
    grp = tdf["gid"].values

    se, sc_ = f.blocks(sdf)
    te, tc = f.blocks(tdf)
    a, b = f.prep(se, te, SCALE)
    c, d = f.prep(sc_, tc, SCALE)
    me = f.fit_lr(a, ys, "ehr")
    mc = f.fit_lr(c, ys, "ctpa")

    P = {"ehr": me.predict_proba(b)[:, 1],
         "ctpa": mc.predict_proba(d)[:, 1]}
    S = {"ehr": me.predict_proba(a)[:, 1],
         "ctpa": mc.predict_proba(c)[:, 1]}
    for tag in STORED:
        v = load_stored(tag, oc, keys)
        if v is not None and \
                np.isfinite(v).sum() > 100:
            P[tag] = v

    print("")
    print("=" * 66)
    print("%s   n=%d  ev=%d"
          % (oc, len(yt), int(yt.sum())))
    print("=" * 66)

    names = sorted(P)
    combos = []
    for r_ in range(2, len(names) + 1):
        combos += [c_ for c_ in
                   combinations(names, r_)
                   if "ehr" in c_]

    print("  %-18s %5s %4s %8s %8s %8s"
          % ("model", "n", "ev", "srcW",
             "tgtCV", "diff"))

    for cb in combos:
        ok = np.ones(len(yt), dtype=bool)
        for t in cb:
            ok &= np.isfinite(P[t])
        n = int(ok.sum())
        ev = int(yt[ok].sum())
        if n < 200 or ev < 30:
            continue

        y = yt[ok]
        g = grp[ok]
        R = {t: _rank(P[t][ok]) for t in cb}
        ref = R["ehr"]
        ra = roc_auc_score(y, ref)

        p1, w1 = src_fuse(R, S, cb, ys)
        a1 = roc_auc_score(y, p1)
        g1, l1, h1, _ = f.boot_diff(
            y, p1, ref, g)

        p2, w2 = target_cv_fuse(R, y, g, cb)
        a2 = roc_auc_score(y, p2)
        g2, l2, h2, _ = f.boot_diff(
            y, p2, ref, g)

        gd, ld, hd, _ = f.boot_diff(
            y, p2, p1, g)
        star = ("*" if (ld > 0 or hd < 0)
                else " ")

        print("  %-18s %5d %4d  %.4f  %.4f"
              "  %+.4f [%+.4f,%+.4f] %s"
              % ("+".join(cb), n, ev, a1, a2,
                 gd, ld, hd, star))
        print("      srcW %-22s gain %+.4f"
              " [%+.4f,%+.4f]"
              % (str([round(x, 2) for x in w1]),
                 g1, l1, h1))
        print("      tgtW %-22s gain %+.4f"
              " [%+.4f,%+.4f]"
              % (str([round(x, 2) for x in w2]),
                 g2, l2, h2))

        for prot, au, gn, lo, hi, w in (
                ("source", a1, g1, l1, h1, w1),
                ("target_cv", a2, g2, l2, h2,
                 w2)):
            rows.append({
                "outcome": oc,
                "model": "+".join(cb),
                "protocol": prot, "n_mod": len(cb),
                "n": n, "ev": ev, "auc": au,
                "ehr_ref": ra, "gain": gn,
                "lo": lo, "hi": hi,
                "weights": str(
                    [round(x, 2) for x in w]),
                "sig": int(lo > 0 or hi < 0)})

        if len(cb) == len(names):
            continue

    ehr_auc = roc_auc_score(
        yt, _rank(P["ehr"]))
    rows.append({
        "outcome": oc, "model": "ehr",
        "protocol": "unimodal", "n_mod": 1,
        "n": len(yt), "ev": int(yt.sum()),
        "auc": ehr_auc, "ehr_ref": ehr_auc,
        "gain": 0.0, "lo": 0.0, "hi": 0.0,
        "weights": "", "sig": 0})
    print("  %-18s %5d %4d  %.4f"
          % ("ehr (unimodal)", len(yt),
             int(yt.sum()), ehr_auc))

r = pd.DataFrame(rows)
r.to_csv(os.path.join(
    PROC, "dissertation_compare.csv"),
    index=False)

print("")
print("=" * 66)
print("HEAD TO HEAD WITH THE DISSERTATION")
print("=" * 66)
d = r[(r["outcome"] == "death_30d")]
d = d.sort_values("auc", ascending=False)
print(d[["model", "protocol", "n", "ev",
         "auc", "gain", "lo", "hi",
         "weights"]].round(4)
      .to_string(index=False))

print("")
print("  dissertation WMEAN3-CTPA: %.4f"
      " on n=1703, 157 events"
      % REF["death_30d"])
t3 = d[(d["model"] == "ctpa+ecg+ehr")
       & (d["protocol"] == "target_cv")]
if len(t3):
    v = float(t3["auc"].iloc[0])
    print("  this rebuild, same protocol:"
          " %.4f on n=%d, %d events"
          % (v, int(t3["n"].iloc[0]),
             int(t3["ev"].iloc[0])))
    print("  difference: %+.4f"
          % (v - REF["death_30d"]))

print("")
print("PROTOCOL EFFECT: what target labels buy")
pv = r[r["protocol"] != "unimodal"].pivot_table(
    index=["outcome", "model"],
    columns="protocol", values="auc")
pv["diff"] = pv["target_cv"] - pv["source"]
print(pv.round(4).to_string())
print("")
print("  mean advantage of target tuning:"
      " %+.4f" % float(pv["diff"].mean()))

print("")
print("BEST MODEL PER OUTCOME AND PROTOCOL")
for (oc, prot), sub in r[
        r["protocol"] != "unimodal"].groupby(
        ["outcome", "protocol"]):
    b = sub.loc[sub["auc"].idxmax()]
    print("  %-14s %-10s %-18s %.4f"
          "  %+.4f  n=%d"
          % (oc, prot, b["model"], b["auc"],
             b["gain"], b["n"]))

print("")
print("saved dissertation_compare.csv", r.shape)