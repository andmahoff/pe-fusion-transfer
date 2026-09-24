"""The fusion grid rerun, with symmetric
source-CV regularisation for both modalities.
Scaling is set by SCALE at the top so the same
script produces the std and blockstd versions.
Run in venv (analysis).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC = f.PROC
SPLIT_SEED = 42
LEARNER = "lr"
SCALE = "std"
TAG = "tuned_" + SCALE

DIRS = [("I2M", "inspect", "mimic"),
        ("M2I", "mimic", "inspect"),
        ("M2M", "mimic", "mimic"),
        ("I2I", "inspect", "inspect")]

ARCHS = (
    [("early", v, False)
     for v in ["plain", "blockstd"]]
    + [("inter", v, True)
       for v in f.INTER_VARIANTS]
    + [("late", v, False)
       for v in f.LATE_VARIANTS]
    + [("uni", "ehr_only", False),
       ("uni", "ctpa_only", False)])

print("learner mode:", f.LEARNER_MODE)
print("scaling:", SCALE)
print("architectures:", len(ARCHS))


def call(fam, var, src, tgt, ys, seed):
    if fam == "early":
        return f.early(src, tgt, ys, variant=var,
                       learner=LEARNER,
                       scale=SCALE)
    if fam in ("late", "uni"):
        return f.late(src, tgt, ys, variant=var,
                      learner=LEARNER,
                      scale=SCALE)
    if fam == "inter":
        return f.intermediate(src, tgt, ys,
                              variant=var,
                              seed=seed,
                              scale=SCALE)
    raise ValueError(fam)


def oof(tdf, yt, fam, var, seed):
    xe, xc = f.blocks(tdf)
    grp = tdf["gid"].values
    p = np.zeros(len(yt))
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SPLIT_SEED)
    for tr, te in cv.split(xe, yt, grp):
        p[te] = call(fam, var,
                     (xe[tr], xc[tr]),
                     (xe[te], xc[te]),
                     yt[tr], seed)
    return p


ins = f.load_inspect()
mim = f.load_mimic()
DATA = {"inspect": ins, "mimic": mim}

rows = []
t0 = time.time()

for tag, sname, tname in DIRS:
    same = sname == tname
    for oc in f.OUTCOMES:
        sdf, ys = f.labels(DATA[sname], oc)
        tdf, yt = f.labels(DATA[tname], oc)
        grp = tdf["gid"].values
        print("")
        print("=" * 62)
        print("%s  %s  n=%d ev=%d"
              % (tag, oc, len(yt), int(yt.sum())))

        preds = {}
        for fam, var, stoch in ARCHS:
            name = fam + ":" + var
            seeds = f.SEEDS if stoch else [SPLIT_SEED]
            acc = []
            for sd in seeds:
                if same:
                    acc.append(oof(tdf, yt, fam,
                                   var, sd))
                else:
                    acc.append(call(
                        fam, var, f.blocks(sdf),
                        f.blocks(tdf), ys, sd))
            preds[name] = np.mean(acc, axis=0)

        # report the C each modality selected
        if not same:
            se, sc_ = f.blocks(sdf)
            te, tc = f.blocks(tdf)
            a, _ = f.prep(se, te, SCALE)
            b, _ = f.prep(sc_, tc, SCALE)
            ce = f.fit_lr(a, ys, "ehr").chosen_c_
            cc = f.fit_lr(b, ys, "ctpa").chosen_c_
            print("  chosen C: ehr %.3f,"
                  " ctpa %.3f" % (ce, cc))
        else:
            ce = cc = np.nan

        ref = preds["uni:ehr_only"]
        ra = roc_auc_score(yt, ref)
        ca = roc_auc_score(yt,
                           preds["uni:ctpa_only"])
        print("  EHR %.4f  CTPA %.4f  gap %.4f"
              % (ra, ca, ra - ca))

        for name, p in preds.items():
            auc = roc_auc_score(yt, p)
            if name == "uni:ehr_only":
                g = lo = hi = 0.0
            else:
                g, lo, hi, _ = f.boot_diff(
                    yt, p, ref, grp)
            star = ("*" if (lo > 0 or hi < 0)
                    else " ")
            print("  %-16s %.4f  %+.4f"
                  " [%+.4f,%+.4f] %s"
                  % (name, auc, g, lo, hi, star))
            rows.append({
                "direction": tag, "outcome": oc,
                "arch": name, "n": len(yt),
                "ev": int(yt.sum()), "auc": auc,
                "ap": average_precision_score(
                    yt, p),
                "ehr_ref": ra, "ctpa_ref": ca,
                "gap": ra - ca, "c_ehr": ce,
                "c_ctpa": cc, "gain": g,
                "lo": lo, "hi": hi,
                "sig": int(lo > 0 or hi < 0)})

r = pd.DataFrame(rows)
dest = os.path.join(PROC,
                    "grid_%s.csv" % TAG)
r.to_csv(dest, index=False)
print("")
print("elapsed %.1f min" % ((time.time() - t0) / 60))
print("saved", dest, r.shape)

print("")
print("GAIN OVER EHR ALONE, MEAN OVER 12 CELLS")
print(r.groupby("arch")[["gain", "sig"]]
      .agg(["mean", "sum"]).round(4).to_string())

old = os.path.join(PROC, "grid_summary.csv")
if os.path.exists(old):
    o = pd.read_csv(old).groupby(
        "arch")["gain"].mean()
    n = r.groupby("arch")["gain"].mean()
    c = pd.DataFrame({"fixedC": o,
                      "tunedC": n})
    c["change"] = c["tunedC"] - c["fixedC"]
    print("")
    print("FIXED C VS TUNED C")
    print(c.dropna().round(4).to_string())

print("")
print("=" * 62)
print("PRIMARY: I2M death_30d")
print("registered prediction: late gain +0.0110")
print("=" * 62)
p = r[(r["direction"] == "I2M")
      & (r["outcome"] == "death_30d")]
print(p.round(4).to_string(index=False))