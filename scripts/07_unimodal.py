"""Unimodal baselines across all four transfer
directions. Each modality alone, both learners,
three outcomes. These are the reference points that
every fusion increment is measured against.
Run in venv (analysis).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC = f.PROC
DEST = os.path.join(PROC, "unimodal.csv")

DIRS = [("I2M", "inspect", "mimic"),
        ("M2I", "mimic", "inspect"),
        ("M2M", "mimic", "mimic"),
        ("I2I", "inspect", "inspect")]

MODS = {"ehr": f.EHR_COLS, "ctpa": f.CTPA_COLS}


def boot_ci(y, p, grp, nboot=f.NBOOT, seed=42):
    us = np.unique(grp)
    ix = {u: np.where(grp == u)[0] for u in us}
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(nboot):
        pk = rng.choice(us, len(us), replace=True)
        ii = np.concatenate([ix[u] for u in pk])
        if len(np.unique(y[ii])) < 2:
            continue
        b.append(roc_auc_score(y[ii], p[ii]))
    b = np.array(b)
    return (float(np.percentile(b, 2.5)),
            float(np.percentile(b, 97.5)))


def transfer(src, tgt, cols, ys, learner):
    Xs = src[cols].values.astype(float)
    Xt = tgt[cols].values.astype(float)
    a, b = f.prep(Xs, Xt)
    m = f.LEARNERS[learner]()
    m.fit(a, ys)
    return m.predict_proba(b)[:, 1]


def internal(df, cols, y, learner):
    """Out-of-fold, grouped, averaged over seeds."""
    X = df[cols].values.astype(float)
    grp = df["gid"].values
    acc = np.zeros((len(f.SEEDS), len(y)))
    for k, s in enumerate(f.SEEDS):
        cv = StratifiedGroupKFold(
            n_splits=5, shuffle=True,
            random_state=s)
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y, grp):
            a, b = f.prep(X[tr], X[te])
            m = f.LEARNERS[learner]()
            m.fit(a, y[tr])
            oof[te] = m.predict_proba(b)[:, 1]
        acc[k] = oof
    return acc.mean(axis=0)


ins = f.load_inspect()
mim = f.load_mimic()
DATA = {"inspect": ins, "mimic": mim}

rows = []
for tag, sname, tname in DIRS:
    same = sname == tname
    for oc in f.OUTCOMES:
        sdf, ys = f.labels(DATA[sname], oc)
        tdf, yt = f.labels(DATA[tname], oc)
        print("")
        print("=" * 60)
        print("%s  %s  src n=%d ev=%d |"
              " tgt n=%d ev=%d"
              % (tag, oc, len(ys), int(ys.sum()),
                 len(yt), int(yt.sum())))

        for mod, cols in MODS.items():
            for lr in ("lr", "gb"):
                if same:
                    p = internal(tdf, cols, yt, lr)
                else:
                    p = transfer(sdf, tdf, cols,
                                 ys, lr)
                auc = roc_auc_score(yt, p)
                ap = average_precision_score(yt, p)
                lo, hi = boot_ci(
                    yt, p, tdf["gid"].values)
                print("  %-5s %-3s AUC %.4f"
                      " [%.4f,%.4f]  AP %.4f"
                      % (mod, lr, auc, lo, hi, ap))
                rows.append({
                    "direction": tag,
                    "outcome": oc,
                    "modality": mod,
                    "learner": lr,
                    "n": len(yt),
                    "ev": int(yt.sum()),
                    "auc": auc, "lo": lo,
                    "hi": hi, "ap": ap})

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)
print("")
print("saved", DEST, r.shape)
print("")
print(r.round(4).to_string(index=False))