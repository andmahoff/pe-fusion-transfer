"""Calibration and decision-curve comparison for the
representative architectures. AUC measures ranking
only, so this checks whether the probabilities
themselves are usable at clinical thresholds.
Run in venv (analysis).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC = f.PROC
SPLIT_SEED = 42
LEARNER = "lr"

DIRS = [("I2M", "inspect", "mimic"),
        ("M2I", "mimic", "inspect"),
        ("M2M", "mimic", "mimic"),
        ("I2I", "inspect", "inspect")]

REPS = [("uni", "ehr_only"),
        ("early", "plain"),
        ("early", "blockstd"),
        ("inter", "plain"),
        ("late", "mean"),
        ("late", "wsrc")]

THRESH = np.arange(0.01, 0.51, 0.01)


def call(fam, var, src, tgt, ys, seed):
    if fam == "early":
        return f.early(src, tgt, ys, variant=var,
                       learner=LEARNER)
    if fam in ("late", "uni"):
        return f.late(src, tgt, ys, variant=var,
                      learner=LEARNER)
    if fam == "inter":
        return f.intermediate(src, tgt, ys,
                              variant=var,
                              seed=seed)
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


def cal_slope(y, p):
    """Slope and intercept of the calibration
    line. Slope 1 and intercept 0 are ideal.
    Slope below 1 means over-extreme risks."""
    e = np.clip(p, 1e-6, 1 - 1e-6)
    lo = np.log(e / (1 - e)).reshape(-1, 1)
    m = LogisticRegression(
        penalty=None, max_iter=5000)
    m.fit(lo, y)
    return (float(m.coef_[0][0]),
            float(m.intercept_[0]))


def deciles(y, p, k=10):
    d = pd.DataFrame({"y": y, "p": p})
    try:
        d["b"] = pd.qcut(d["p"], k,
                         labels=False,
                         duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    g = d.groupby("b").agg(
        pred=("p", "mean"),
        obs=("y", "mean"),
        n=("y", "size")).reset_index()
    return g


def net_benefit(y, p, t):
    """Standard decision-curve net benefit."""
    n = len(y)
    pos = p >= t
    tp = float(((pos) & (y == 1)).sum())
    fp = float(((pos) & (y == 0)).sum())
    return tp / n - (fp / n) * (t / (1 - t))


ins = f.load_inspect()
mim = f.load_mimic()
DATA = {"inspect": ins, "mimic": mim}

sum_rows = []
dec_rows = []
dca_rows = []

for tag, sname, tname in DIRS:
    same = sname == tname
    for oc in f.OUTCOMES:
        sdf, ys = f.labels(DATA[sname], oc)
        tdf, yt = f.labels(DATA[tname], oc)
        prev = float(yt.mean())
        print("")
        print("=" * 62)
        print("%s  %s  n=%d ev=%d prev=%.3f"
              % (tag, oc, len(yt),
                 int(yt.sum()), prev))

        for fam, var in REPS:
            name = fam + ":" + var
            if same:
                p = oof(tdf, yt, fam, var,
                        SPLIT_SEED)
            else:
                p = call(fam, var, f.blocks(sdf),
                         f.blocks(tdf), ys,
                         SPLIT_SEED)
            if p.max() > 1.0 or p.min() < 0.0:
                p = np.clip(p, 0.0, 1.0)

            auc = roc_auc_score(yt, p)
            br = brier_score_loss(yt, p)
            sl, ic = cal_slope(yt, p)
            print("  %-16s AUC %.4f  Brier %.4f"
                  "  slope %.3f  intercept %+.3f"
                  % (name, auc, br, sl, ic))
            sum_rows.append({
                "direction": tag, "outcome": oc,
                "arch": name, "n": len(yt),
                "ev": int(yt.sum()), "prev": prev,
                "auc": auc, "brier": br,
                "slope": sl, "intercept": ic})

            g = deciles(yt, p)
            for _, r in g.iterrows():
                dec_rows.append({
                    "direction": tag,
                    "outcome": oc, "arch": name,
                    "decile": int(r["b"]),
                    "pred": r["pred"],
                    "obs": r["obs"],
                    "n": int(r["n"])})

            for t in THRESH:
                dca_rows.append({
                    "direction": tag,
                    "outcome": oc, "arch": name,
                    "thresh": float(t),
                    "nb": net_benefit(yt, p, t)})

        for t in THRESH:
            dca_rows.append({
                "direction": tag, "outcome": oc,
                "arch": "treat_all",
                "thresh": float(t),
                "nb": prev - (1 - prev)
                * (t / (1 - t))})
            dca_rows.append({
                "direction": tag, "outcome": oc,
                "arch": "treat_none",
                "thresh": float(t), "nb": 0.0})

sr = pd.DataFrame(sum_rows)
sr.to_csv(os.path.join(PROC, "calibration.csv"),
          index=False)
pd.DataFrame(dec_rows).to_csv(
    os.path.join(PROC, "calib_deciles.csv"),
    index=False)
pd.DataFrame(dca_rows).to_csv(
    os.path.join(PROC, "decision_curves.csv"),
    index=False)

print("")
print("saved calibration.csv, calib_deciles.csv,"
      " decision_curves.csv")
print("")
print("PRIMARY: I2M death_30d")
print(sr[(sr["direction"] == "I2M")
         & (sr["outcome"] == "death_30d")]
      .round(4).to_string(index=False))

print("")
print("CALIBRATION SLOPE BY ARCHITECTURE"
      " (mean over 12 cells)")
print(sr.groupby("arch")[
    ["auc", "brier", "slope", "intercept"]]
    .mean().round(4).to_string())