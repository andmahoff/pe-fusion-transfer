"""What governs fusion gain beyond the gap.
Regenerates the ladder rungs and the grid cells,
keeping per-patient predictions so that
prediction correlation can be measured, then
fits gain against gap, mean AUC, correlation,
event rate and transfer status.
Run in venv (analysis).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC = f.PROC
SEED = 42
SCALE = "std"

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
        return [c for c in cols if c not in VIT
                and c not in f.FLAGS
                and c != "age"]
    if kind == "agefl":
        return f.FLAGS + ["age"]
    if kind == "age":
        return ["age"]
    raise ValueError(kind)


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def transfer(sdf, ys, tdf, cols, frac, seed):
    rng = np.random.default_rng(seed)
    n = len(ys)
    if frac < 1.0:
        pos = np.where(ys == 1)[0]
        neg = np.where(ys == 0)[0]
        npos = max(5, int(round(len(pos) * frac)))
        nneg = max(20, int(round(len(neg) * frac)))
        take = np.concatenate([
            rng.choice(pos, min(npos, len(pos)),
                       replace=False),
            rng.choice(neg, min(nneg, len(neg)),
                       replace=False)])
    else:
        take = np.arange(n)

    a, b = f.prep(
        sdf[cols].values.astype(float)[take],
        tdf[cols].values.astype(float), SCALE)
    me = f.fit_lr(a, ys[take], "ehr")
    pe = _rank(me.predict_proba(b)[:, 1])
    pe_s = _rank(me.predict_proba(a)[:, 1])

    c, d = f.prep(
        sdf[f.CTPA_COLS].values.astype(float),
        tdf[f.CTPA_COLS].values.astype(float),
        SCALE)
    mc = f.fit_lr(c, ys, "ctpa")
    pc = _rank(mc.predict_proba(d)[:, 1])
    pc_s = _rank(mc.predict_proba(c)[:, 1])

    ysub = ys[take]
    best, bw = -1.0, 0.5
    for w in np.arange(0.0, 1.01, 0.05):
        sc = roc_auc_score(
            ysub, w * pe_s + (1 - w) * pc_s[take])
        if sc > best:
            best, bw = sc, w
    return pe, pc, bw * pe + (1 - bw) * pc, bw


def internal(df, y, cols, seed):
    xe = df[cols].values.astype(float)
    xc = df[f.CTPA_COLS].values.astype(float)
    grp = df["gid"].values
    pe = np.zeros(len(y))
    pc = np.zeros(len(y))
    pf = np.zeros(len(y))
    ws = []
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(xe, y, grp):
        a, b = f.prep(xe[tr], xe[te], SCALE)
        c, d = f.prep(xc[tr], xc[te], SCALE)
        me = f.fit_lr(a, y[tr], "ehr")
        mc = f.fit_lr(c, y[tr], "ctpa")
        qe = _rank(me.predict_proba(b)[:, 1])
        qc = _rank(mc.predict_proba(d)[:, 1])
        se = _rank(me.predict_proba(a)[:, 1])
        sc = _rank(mc.predict_proba(c)[:, 1])
        best, bw = -1.0, 0.5
        for w in np.arange(0.0, 1.01, 0.05):
            v = roc_auc_score(
                y[tr], w * se + (1 - w) * sc)
            if v > best:
                best, bw = v, w
        pe[te], pc[te] = qe, qc
        pf[te] = bw * qe + (1 - bw) * qc
        ws.append(bw)
    return pe, pc, pf, float(np.mean(ws))


def record(tag, oc, rung, cols, frac,
           yt, pe, pc, pf, w, grp, cross):
    ae = roc_auc_score(yt, pe)
    ac = roc_auc_score(yt, pc)
    af = roc_auc_score(yt, pf)
    g, lo, hi, _ = f.boot_diff(yt, pf, pe, grp)
    r_p, _ = stats.pearsonr(pe, pc)
    r_s, _ = stats.spearmanr(pe, pc)
    ev = int(yt.sum())
    return {"source": tag, "outcome": oc,
            "rung": rung, "nfeat": len(cols),
            "frac": frac, "n": len(yt),
            "ev": ev, "rate": ev / len(yt),
            "ehr": ae, "ctpa": ac,
            "gap": ae - ac,
            "mean_auc": 0.5 * (ae + ac),
            "min_auc": min(ae, ac),
            "corr_p": r_p, "corr_s": r_s,
            "fused": af, "gain": g,
            "lo": lo, "hi": hi,
            "weight_ehr": w, "cross": cross,
            "sig": int(lo > 0 or hi < 0)}


ins = f.load_inspect()
mim = f.load_mimic()
rows = []
t0 = time.time()

print("LADDER RUNGS")
for tag, s0, t0d in [("I2M", ins, mim),
                     ("M2I", mim, ins)]:
    for oc in ["death_30d", "composite_30d"]:
        sdf, ys = f.labels(s0, oc)
        tdf, yt = f.labels(t0d, oc)
        grp = tdf["gid"].values
        for lab, kind, frac in RUNGS:
            cols = subset(f.EHR_COLS, kind)
            acc = []
            for sd in (f.SEEDS if frac < 1.0
                       else [SEED]):
                acc.append(transfer(
                    sdf, ys, tdf, cols, frac, sd))
            pe = np.mean([a[0] for a in acc], 0)
            pc = np.mean([a[1] for a in acc], 0)
            pf = np.mean([a[2] for a in acc], 0)
            w = float(np.mean(
                [a[3] for a in acc]))
            r = record(tag, oc, lab, cols, frac,
                       yt, pe, pc, pf, w, grp, 1)
            rows.append(r)
            print("  %-4s %-14s %-13s gap %.4f"
                  "  corr %.3f  gain %+.4f"
                  % (tag, oc, lab, r["gap"],
                     r["corr_s"], r["gain"]))

print("")
print("GRID CELLS (full features)")
for tag, s0, t0d in [("I2M", ins, mim),
                     ("M2I", mim, ins),
                     ("M2M", mim, mim),
                     ("I2I", ins, ins)]:
    same = tag in ("M2M", "I2I")
    for oc in f.OUTCOMES:
        sdf, ys = f.labels(s0, oc)
        tdf, yt = f.labels(t0d, oc)
        grp = tdf["gid"].values
        if same:
            pe, pc, pf, w = internal(
                tdf, yt, f.EHR_COLS, SEED)
        else:
            pe, pc, pf, w = transfer(
                sdf, ys, tdf, f.EHR_COLS,
                1.0, SEED)
        r = record(tag, oc, "grid", f.EHR_COLS,
                   1.0, yt, pe, pc, pf, w, grp,
                   int(not same))
        rows.append(r)
        print("  %-4s %-14s gap %.4f  corr %.3f"
              "  meanAUC %.4f  gain %+.4f"
              % (tag, oc, r["gap"], r["corr_s"],
                 r["mean_auc"], r["gain"]))

d = pd.DataFrame(rows)
d.to_csv(os.path.join(PROC, "moderators.csv"),
         index=False)
print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))


def ols(df, ys, xs, lab):
    sub = df[[ys] + xs].dropna()
    if len(sub) < len(xs) + 3:
        print("  %-28s too few" % lab)
        return
    X = np.column_stack(
        [np.ones(len(sub))]
        + [sub[c].values for c in xs])
    y = sub[ys].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    dof = len(sub) - X.shape[1]
    s2 = float(resid @ resid) / max(dof, 1)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / max(ss, 1e-12)
    n = len(xs)
    adj = 1 - (1 - r2) * (len(sub) - 1) / max(dof, 1)
    print("  %s  (n=%d, R2=%.3f, adjR2=%.3f)"
          % (lab, len(sub), r2, adj))
    names = ["intercept"] + xs
    for i, nm in enumerate(names):
        t = beta[i] / max(se[i], 1e-12)
        p = 2 * (1 - stats.t.cdf(abs(t), dof))
        star = "*" if p < 0.05 else " "
        print("    %-12s %+.4f (se %.4f)"
              "  t=%+.2f  p=%.4f %s"
              % (nm, beta[i], se[i], t, p, star))


LAD = d[d["rung"] != "grid"]
GRID = d[d["rung"] == "grid"]
INRANGE = LAD[LAD["gap"] > 0]

print("")
print("=" * 62)
print("DOES AVERAGE STRENGTH MATTER?")
print("=" * 62)
print("")
print("LADDER, gap only (32 rungs)")
ols(LAD, "gain", ["gap"], "gain ~ gap")
print("")
print("LADDER, gap + mean AUC")
ols(LAD, "gain", ["gap", "mean_auc"],
    "gain ~ gap + mean_auc")
print("")
print("LADDER, positive gaps only (in range)")
ols(INRANGE, "gain", ["gap"],
    "gain ~ gap [in range]")
ols(INRANGE, "gain", ["gap", "mean_auc"],
    "gain ~ gap + mean_auc [in range]")
print("")
print("GRID, gap + mean AUC (12 cells)")
ols(GRID, "gain", ["gap"], "gain ~ gap")
ols(GRID, "gain", ["gap", "mean_auc"],
    "gain ~ gap + mean_auc")

print("")
print("  confounding check: corr(gap, mean_auc)")
for nm, sub in [("ladder", LAD),
                ("in-range", INRANGE),
                ("grid", GRID)]:
    c = float(np.corrcoef(
        sub["gap"], sub["mean_auc"])[0, 1])
    print("    %-9s %+.3f" % (nm, c))

print("")
print("=" * 62)
print("OTHER MODERATORS")
print("=" * 62)
print("")
print("Univariate correlation with gain")
for v in ["gap", "mean_auc", "min_auc",
          "corr_s", "corr_p", "rate", "ev",
          "weight_ehr", "cross"]:
    for nm, sub in [("ladder", LAD),
                    ("grid", GRID)]:
        s = sub[[v, "gain"]].dropna()
        if len(s) < 5 or s[v].std() == 0:
            continue
        r, p = stats.pearsonr(s[v], s["gain"])
        rho, _ = stats.spearmanr(s[v], s["gain"])
        print("  %-11s %-7s r=%+.3f  rho=%+.3f"
              "  p=%.4f  n=%d"
              % (v, nm, r, rho, p, len(s)))

print("")
print("MULTIVARIATE, ladder")
ols(LAD, "gain", ["gap", "corr_s"],
    "gain ~ gap + corr")
ols(LAD, "gain", ["gap", "mean_auc", "corr_s"],
    "gain ~ gap + mean_auc + corr")
print("")
print("MULTIVARIATE, grid")
ols(GRID, "gain", ["gap", "corr_s"],
    "gain ~ gap + corr")
ols(GRID, "gain", ["gap", "rate"],
    "gain ~ gap + event rate")

print("")
print("saved moderators.csv", d.shape)