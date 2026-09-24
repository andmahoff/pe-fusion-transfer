"""Controlled test of the gap mechanism. Weakens
the EHR modality in steps while holding the CTPA
modality fixed, then measures fusion gain at each
step. This moves the gap from the opposite end to
the dissertation's script 140, and separates 'the gap matters' from
'the weaker modality's strength matters'.
Run in venv (analysis).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC = f.PROC
SEED = 42
SCALE = "std"

# (label, feature subset, training fraction)
RUNGS = [
    ("full", None, 1.0),
    ("half_rows", None, 0.50),
    ("quarter_rows", None, 0.25),
    ("eighth_rows", None, 0.125),
    ("no_vitals", "novit", 1.0),
    ("labs_only", "labs", 1.0),
    ("age_flags", "agefl", 1.0),
    ("age_only", "age", 1.0),
]

VIT = ["temp", "hr", "sbp", "dbp", "rr"]


def subset(cols, kind):
    if kind is None:
        return cols
    if kind == "novit":
        return [c for c in cols if c not in VIT]
    if kind == "labs":
        return [c for c in cols
                if c not in VIT
                and c not in f.FLAGS
                and c != "age"]
    if kind == "agefl":
        return f.FLAGS + ["age"]
    if kind == "age":
        return ["age"]
    raise ValueError(kind)


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def run_cell(sdf, ys, tdf, cols, frac, seed):
    """Returns EHR, CTPA and fused predictions.
    Only the EHR modality is affected by cols and
    frac; CTPA sees the full source every time."""
    rng = np.random.default_rng(seed)
    n = len(ys)
    if frac < 1.0:
        k = max(50, int(round(n * frac)))
        pos = np.where(ys == 1)[0]
        neg = np.where(ys == 0)[0]
        npos = max(5, int(round(len(pos) * frac)))
        nneg = max(20, k - npos)
        take = np.concatenate([
            rng.choice(pos, min(npos, len(pos)),
                       replace=False),
            rng.choice(neg, min(nneg, len(neg)),
                       replace=False)])
    else:
        take = np.arange(n)

    xe_s = sdf[cols].values.astype(float)
    xe_t = tdf[cols].values.astype(float)
    a, b = f.prep(xe_s[take], xe_t, SCALE)
    me = f.fit_lr(a, ys[take], "ehr")
    pe = _rank(me.predict_proba(b)[:, 1])
    pe_s = _rank(me.predict_proba(a)[:, 1])

    xc_s = sdf[f.CTPA_COLS].values.astype(float)
    xc_t = tdf[f.CTPA_COLS].values.astype(float)
    c, d = f.prep(xc_s, xc_t, SCALE)
    mc = f.fit_lr(c, ys, "ctpa")
    pc = _rank(mc.predict_proba(d)[:, 1])
    pc_s = _rank(mc.predict_proba(c)[:, 1])

    best, bw = -1.0, 0.5
    ysub = ys[take]
    for w in np.arange(0.0, 1.01, 0.05):
        sc = roc_auc_score(
            ysub, w * pe_s + (1 - w)
            * pc_s[take])
        if sc > best:
            best, bw = sc, w
    return pe, pc, bw * pe + (1 - bw) * pc, bw


ins = f.load_inspect()
mim = f.load_mimic()
DIRS = [("I2M", ins, mim), ("M2I", mim, ins)]

rows = []
t0 = time.time()

for tag, sdf0, tdf0 in DIRS:
    for oc in ["death_30d", "composite_30d"]:
        sdf, ys = f.labels(sdf0, oc)
        tdf, yt = f.labels(tdf0, oc)
        grp = tdf["gid"].values
        print("")
        print("=" * 62)
        print("%s  %s  n=%d ev=%d"
              % (tag, oc, len(yt), int(yt.sum())))
        print("  %-14s %-6s %-6s %-6s %-7s %s"
              % ("rung", "ehr", "ctpa", "gap",
                 "fused", "gain [95% CI]"))

        for lab, kind, frac in RUNGS:
            cols = subset(f.EHR_COLS, kind)
            accs = []
            for sd in f.SEEDS:
                pe, pc, pf, w = run_cell(
                    sdf, ys, tdf, cols, frac, sd)
                accs.append((pe, pc, pf, w))
                if frac == 1.0:
                    break
            pe = np.mean([a[0] for a in accs],
                         axis=0)
            pc = np.mean([a[1] for a in accs],
                         axis=0)
            pf = np.mean([a[2] for a in accs],
                         axis=0)
            wm = float(np.mean(
                [a[3] for a in accs]))

            ae = roc_auc_score(yt, pe)
            ac = roc_auc_score(yt, pc)
            af = roc_auc_score(yt, pf)
            g, lo, hi, _ = f.boot_diff(
                yt, pf, pe, grp)
            star = ("*" if (lo > 0 or hi < 0)
                    else " ")
            print("  %-14s %.4f %.4f %.4f"
                  " %.4f  %+.4f"
                  " [%+.4f,%+.4f] %s"
                  % (lab, ae, ac, ae - ac, af,
                     g, lo, hi, star))
            rows.append({
                "direction": tag, "outcome": oc,
                "rung": lab, "nfeat": len(cols),
                "frac": frac, "ehr": ae,
                "ctpa": ac, "gap": ae - ac,
                "fused": af, "gain": g,
                "lo": lo, "hi": hi,
                "weight_ehr": wm,
                "sig": int(lo > 0 or hi < 0)})

r = pd.DataFrame(rows)
r.to_csv(os.path.join(PROC, "ladder.csv"),
         index=False)
print("")
print("elapsed %.1f min" % ((time.time() - t0) / 60))

print("")
print("=" * 62)
print("CTPA HELD FIXED? (should be constant"
      " within each cell)")
print(r.groupby(["direction", "outcome"])
      ["ctpa"].agg(["min", "max", "std"])
      .round(5).to_string())

print("")
print("REGRESSION OF GAIN ON GAP (ladder only)")
for nm, sub in [("all", r),
                ("I2M", r[r["direction"] == "I2M"]),
                ("M2I", r[r["direction"] == "M2I"])]:
    x = sub["gap"].values
    y = sub["gain"].values
    if len(x) < 4:
        continue
    sl, ic, rr, p, se = stats.linregress(x, y)
    rho, _ = stats.spearmanr(x, y)
    print("  %-5s slope %+.4f (se %.4f)"
          "  r=%+.3f  R2=%.3f  p=%.4f"
          "  rho=%+.3f  n=%d"
          % (nm, sl, se, rr, rr ** 2, p,
             rho, len(x)))

print("")
print("GAIN VS CTPA (should be FLAT: CTPA is"
      " fixed within each cell)")
x = r["ctpa"].values
y = r["gain"].values
sl, ic, rr, p, se = stats.linregress(x, y)
print("  slope %+.4f (se %.4f)  R2=%.3f"
      "  p=%.4f" % (sl, se, rr ** 2, p))

gp = os.path.join(PROC,
                  "gap_regression_tuned.csv")
if os.path.exists(gp):
    g = pd.read_csv(gp)
    o = g[g["set"] == "late_best"]
    if len(o):
        o = o.iloc[0]
        print("")
        print("OBSERVED VS PREDICTED BY THE"
              " GRID-FITTED LINE")
        print("  line: %+.4f %+.4f * gap"
              % (o["intercept"], o["slope"]))
        pr = (o["intercept"]
              + o["slope"] * r["gap"])
        cmp = r[["direction", "outcome", "rung",
                 "gap", "gain"]].copy()
        cmp["predicted"] = pr.values
        cmp["resid"] = cmp["gain"] - cmp["predicted"]
        print(cmp.round(4).to_string(index=False))
        print("")
        print("  mean abs residual: %.4f"
              % float(cmp["resid"].abs().mean()))
        print("  correlation obs vs pred: %.3f"
              % float(np.corrcoef(
                  cmp["gain"],
                  cmp["predicted"])[0, 1]))

print("")
print("saved ladder.csv", r.shape)