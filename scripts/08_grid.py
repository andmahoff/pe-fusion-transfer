"""The fusion grid. Four transfer directions, three
outcomes, twelve architectures: three early, seven
intermediate, two late, plus two unimodal
references. Intermediate runs across three seeds,
reported per seed and averaged.
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

DIRS = [("I2M", "inspect", "mimic"),
        ("M2I", "mimic", "inspect"),
        ("M2M", "mimic", "mimic"),
        ("I2I", "inspect", "inspect")]

ARCHS = (
    [("early", v, False) for v in f.EARLY_VARIANTS]
    + [("inter", v, True)
       for v in f.INTER_VARIANTS]
    + [("late", v, False) for v in f.LATE_VARIANTS]
    + [("uni", "ehr_only", False),
       ("uni", "ctpa_only", False)])

print("architectures:", len(ARCHS))
print("")
print("INTERMEDIATE PARAMETER COUNTS")
for v in f.INTER_VARIANTS:
    print("  %-10s %d" % (v, f.n_params(v)))


def call(fam, var, src, tgt, ys, seed):
    if fam == "early":
        return f.early(src, tgt, ys,
                       variant=var,
                       learner=LEARNER)
    if fam in ("late", "uni"):
        return f.late(src, tgt, ys,
                      variant=var,
                      learner=LEARNER)
    if fam == "inter":
        return f.intermediate(src, tgt, ys,
                              variant=var,
                              seed=seed)
    raise ValueError(fam)


def cross(sdf, tdf, ys, fam, var, seed):
    return call(fam, var, f.blocks(sdf),
                f.blocks(tdf), ys, seed)


def oof(tdf, yt, fam, var, seed):
    xe, xc = f.blocks(tdf)
    grp = tdf["gid"].values
    p = np.zeros(len(yt))
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SPLIT_SEED)
    for tr, te in cv.split(xe, yt, grp):
        s = (xe[tr], xc[tr])
        t = (xe[te], xc[te])
        p[te] = call(fam, var, s, t, yt[tr], seed)
    return p


ins = f.load_inspect()
mim = f.load_mimic()
DATA = {"inspect": ins, "mimic": mim}

per_rows = []
sum_rows = []
t0 = time.time()

for tag, sname, tname in DIRS:
    same = sname == tname
    for oc in f.OUTCOMES:
        sdf, ys = f.labels(DATA[sname], oc)
        tdf, yt = f.labels(DATA[tname], oc)
        grp = tdf["gid"].values
        print("")
        print("=" * 64)
        print("%s  %s   tgt n=%d ev=%d"
              % (tag, oc, len(yt), int(yt.sum())))

        preds = {}
        for fam, var, stoch in ARCHS:
            name = fam + ":" + var
            seeds = f.SEEDS if stoch else [SPLIT_SEED]
            acc = []
            for sd in seeds:
                if same:
                    p = oof(tdf, yt, fam, var, sd)
                else:
                    p = cross(sdf, tdf, ys, fam,
                              var, sd)
                acc.append(p)
                if stoch:
                    per_rows.append({
                        "direction": tag,
                        "outcome": oc,
                        "arch": name, "seed": sd,
                        "auc": roc_auc_score(yt, p),
                        "ap": average_precision_score(
                            yt, p)})
            pm = np.mean(acc, axis=0)
            preds[name] = pm
            aucs = [roc_auc_score(yt, a)
                    for a in acc]
            sd_auc = (float(np.std(aucs))
                      if len(aucs) > 1 else 0.0)
            print("  %-16s AUC %.4f  AP %.4f"
                  "  seedSD %.4f"
                  % (name, roc_auc_score(yt, pm),
                     average_precision_score(yt, pm),
                     sd_auc))

        ref = preds["uni:ehr_only"]
        ref_auc = roc_auc_score(yt, ref)
        block = []
        for name, p in preds.items():
            auc = roc_auc_score(yt, p)
            row = {"direction": tag, "outcome": oc,
                   "arch": name, "n": len(yt),
                   "ev": int(yt.sum()),
                   "auc": auc,
                   "ap": average_precision_score(
                       yt, p),
                   "ehr_ref": ref_auc}
            if name == "uni:ehr_only":
                row.update({"gain": 0.0, "lo": 0.0,
                            "hi": 0.0, "pgt": 0.5})
            else:
                g, lo, hi, pg = f.boot_diff(
                    yt, p, ref, grp)
                row.update({"gain": g, "lo": lo,
                            "hi": hi, "pgt": pg})
            block.append(row)
            sum_rows.append(row)

        print("  -- gain vs EHR alone (%.4f) --"
              % ref_auc)
        for x in block:
            if x["arch"] == "uni:ehr_only":
                continue
            star = "*" if (x["lo"] > 0
                           or x["hi"] < 0) else " "
            print("  %-16s %+.4f"
                  " [%+.4f,%+.4f] %s"
                  % (x["arch"], x["gain"],
                     x["lo"], x["hi"], star))

pr = pd.DataFrame(per_rows)
sr = pd.DataFrame(sum_rows)
pr.to_csv(os.path.join(PROC, "grid_per_seed.csv"),
          index=False)
sr.to_csv(os.path.join(PROC, "grid_summary.csv"),
          index=False)
print("")
print("elapsed %.1f min" % ((time.time() - t0) / 60))
print("saved grid_per_seed.csv", pr.shape)
print("saved grid_summary.csv", sr.shape)

print("")
print("=" * 64)
print("PRIMARY CONTRAST: I2M, death_30d")
print("=" * 64)
p = sr[(sr["direction"] == "I2M")
       & (sr["outcome"] == "death_30d")]
print(p.round(4).to_string(index=False))

print("")
print("PER-SEED SPREAD (intermediate only)")
print(pr.round(4).to_string(index=False))