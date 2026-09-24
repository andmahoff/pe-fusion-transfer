"""Six further candidate moderators of fusion
gain: error complementarity on the positives,
discordance yield, calibration mismatch, feature
overlap, class-conditional strength ratio, and
outcome type. Regenerates the same 44 rows as
script 22 so results are directly comparable.
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
from sklearn.metrics import brier_score_loss

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC = f.PROC
SEED = 42
SCALE = "std"
DISC_Q = 0.20

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

# CTPA features that encode comorbidity, so
# overlap the EHR flags conceptually
OVERLAP_C = ["malignancy", "cm_mets",
             "cm_lymphangitic", "cm_emphysema",
             "cardiomeg", "edema",
             "cm_cirrhosis", "cm_renal"]


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
    if frac < 1.0:
        pos = np.where(ys == 1)[0]
        neg = np.where(ys == 0)[0]
        np_ = max(5, int(round(len(pos) * frac)))
        nn_ = max(20, int(round(len(neg) * frac)))
        take = np.concatenate([
            rng.choice(pos, min(np_, len(pos)),
                       replace=False),
            rng.choice(neg, min(nn_, len(neg)),
                       replace=False)])
    else:
        take = np.arange(len(ys))

    a, b = f.prep(
        sdf[cols].values.astype(float)[take],
        tdf[cols].values.astype(float), SCALE)
    me = f.fit_lr(a, ys[take], "ehr")
    pe = me.predict_proba(b)[:, 1]
    pe_s = _rank(me.predict_proba(a)[:, 1])

    c, d = f.prep(
        sdf[f.CTPA_COLS].values.astype(float),
        tdf[f.CTPA_COLS].values.astype(float),
        SCALE)
    mc = f.fit_lr(c, ys, "ctpa")
    pc = mc.predict_proba(d)[:, 1]
    pc_s = _rank(mc.predict_proba(c)[:, 1])

    ysub = ys[take]
    best, bw = -1.0, 0.5
    for w in np.arange(0.0, 1.01, 0.05):
        v = roc_auc_score(
            ysub, w * pe_s + (1 - w) * pc_s[take])
        if v > best:
            best, bw = v, w
    pf = bw * _rank(pe) + (1 - bw) * _rank(pc)
    return pe, pc, pf, bw


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
        qe = me.predict_proba(b)[:, 1]
        qc = mc.predict_proba(d)[:, 1]
        se = _rank(me.predict_proba(a)[:, 1])
        sc = _rank(mc.predict_proba(c)[:, 1])
        best, bw = -1.0, 0.5
        for w in np.arange(0.0, 1.01, 0.05):
            v = roc_auc_score(
                y[tr], w * se + (1 - w) * sc)
            if v > best:
                best, bw = v, w
        pe[te], pc[te] = qe, qc
        pf[te] = (bw * _rank(qe)
                  + (1 - bw) * _rank(qc))
        ws.append(bw)
    return pe, pc, pf, float(np.mean(ws))


# ---------- the six moderators ----------
def m1_error_complementarity(y, pe, pc):
    """Do the two modalities fail on the same
    positives? Rank correlation computed on the
    positive cases only. High means they miss
    the same patients, so fusion cannot help."""
    m = y == 1
    if m.sum() < 10:
        return np.nan, np.nan
    re = _rank(pe)[m]
    rc = _rank(pc)[m]
    rho, _ = stats.spearmanr(re, rc)
    # fraction of positives both rank in the
    # bottom half, i.e. jointly missed
    both = float(((re < 0.5) & (rc < 0.5)).mean())
    return float(rho), both


def m2_discordance_yield(y, pe, pc, q=DISC_Q):
    """Where the two modalities disagree most,
    is either right? Compares event rate in the
    top discordance quintile against the rest,
    and asks which modality wins there."""
    re, rc = _rank(pe), _rank(pc)
    d = np.abs(re - rc)
    thr = np.quantile(d, 1.0 - q)
    hi = d >= thr
    if hi.sum() < 20 or len(np.unique(
            y[hi])) < 2:
        return np.nan, np.nan, np.nan
    rate_hi = float(y[hi].mean())
    rate_lo = float(y[~hi].mean())
    ae = roc_auc_score(y[hi], re[hi])
    ac = roc_auc_score(y[hi], rc[hi])
    return (rate_hi - rate_lo,
            float(max(ae, ac)),
            float(abs(ae - ac)))


def m3_calibration_mismatch(y, pe, pc):
    """How differently calibrated are the two
    modalities? Brier difference and the gap in
    mean predicted risk."""
    e = np.clip(pe, 1e-6, 1 - 1e-6)
    c = np.clip(pc, 1e-6, 1 - 1e-6)
    be = brier_score_loss(y, e)
    bc = brier_score_loss(y, c)
    return (float(abs(be - bc)),
            float(abs(e.mean() - c.mean())))


def m4_feature_overlap(sdf, tdf, ys, cols):
    """How much of the CTPA signal sits in
    features that duplicate EHR comorbidity?
    Fits CTPA with and without the overlapping
    block and reports the drop."""
    keep = [c for c in f.CTPA_COLS
            if c not in OVERLAP_C]
    out = []
    for cc in (f.CTPA_COLS, keep):
        a, b = f.prep(
            sdf[cc].values.astype(float),
            tdf[cc].values.astype(float), SCALE)
        m = f.fit_lr(a, ys, "ctpa")
        out.append(m.predict_proba(b)[:, 1])
    return out[0], out[1]


def m5_class_conditional(y, pe, pc):
    """Does the strength ratio differ between
    classes? Spread of each modality's scores
    within positives against within negatives."""
    re, rc = _rank(pe), _rank(pc)
    if (y == 1).sum() < 10:
        return np.nan
    de = re[y == 1].mean() - re[y == 0].mean()
    dc = rc[y == 1].mean() - rc[y == 0].mean()
    return float(de - dc)


ins = f.load_inspect()
mim = f.load_mimic()
rows = []
t0 = time.time()

JOBS = ([("I2M", ins, mim, r) for r in RUNGS]
        + [("M2I", mim, ins, r) for r in RUNGS])

print("LADDER RUNGS")
for tag, s0, t0d, (lab, kind, frac) in JOBS:
    for oc in ["death_30d", "composite_30d"]:
        sdf, ys = f.labels(s0, oc)
        tdf, yt = f.labels(t0d, oc)
        grp = tdf["gid"].values
        cols = subset(f.EHR_COLS, kind)
        acc = []
        for sd in (f.SEEDS if frac < 1.0
                   else [SEED]):
            acc.append(transfer(
                sdf, ys, tdf, cols, frac, sd))
        pe = np.mean([a[0] for a in acc], 0)
        pc = np.mean([a[1] for a in acc], 0)
        pf = np.mean([a[2] for a in acc], 0)
        w = float(np.mean([a[3] for a in acc]))

        ae = roc_auc_score(yt, pe)
        ac = roc_auc_score(yt, pc)
        af = roc_auc_score(yt, pf)
        g, lo, hi, _ = f.boot_diff(
            yt, pf, _rank(pe), grp)
        rho_pos, both_miss = \
            m1_error_complementarity(yt, pe, pc)
        d_rate, d_auc, d_diff = \
            m2_discordance_yield(yt, pe, pc)
        b_gap, r_gap = \
            m3_calibration_mismatch(yt, pe, pc)
        pc_full, pc_red = m4_feature_overlap(
            sdf, tdf, ys, cols)
        ov = (roc_auc_score(yt, pc_full)
              - roc_auc_score(yt, pc_red))
        cc = m5_class_conditional(yt, pe, pc)

        rows.append({
            "set": "ladder", "direction": tag,
            "outcome": oc, "rung": lab,
            "n": len(yt), "ev": int(yt.sum()),
            "rate": float(yt.mean()),
            "ehr": ae, "ctpa": ac,
            "gap": ae - ac,
            "mean_auc": 0.5 * (ae + ac),
            "gain": g, "lo": lo, "hi": hi,
            "m1_rho_pos": rho_pos,
            "m1_both_miss": both_miss,
            "m2_disc_rate": d_rate,
            "m2_disc_auc": d_auc,
            "m2_disc_diff": d_diff,
            "m3_brier_gap": b_gap,
            "m3_mean_gap": r_gap,
            "m4_overlap": ov,
            "m5_classcond": cc,
            "weight_ehr": w,
            "cross": 1})
        print("  %-4s %-14s %-13s gap %+.4f"
              "  m1 %.3f  m2 %+.4f  m4 %+.4f"
              % (tag, oc, lab, ae - ac,
                 rho_pos, d_rate, ov))

print("")
print("GRID CELLS")
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
        ae = roc_auc_score(yt, pe)
        ac = roc_auc_score(yt, pc)
        g, lo, hi, _ = f.boot_diff(
            yt, pf, _rank(pe), grp)
        rho_pos, both_miss = \
            m1_error_complementarity(yt, pe, pc)
        d_rate, d_auc, d_diff = \
            m2_discordance_yield(yt, pe, pc)
        b_gap, r_gap = \
            m3_calibration_mismatch(yt, pe, pc)
        if same:
            ov = np.nan
        else:
            a1, a2 = m4_feature_overlap(
                sdf, tdf, ys, f.EHR_COLS)
            ov = (roc_auc_score(yt, a1)
                  - roc_auc_score(yt, a2))
        cc = m5_class_conditional(yt, pe, pc)

        rows.append({
            "set": "grid", "direction": tag,
            "outcome": oc, "rung": "grid",
            "n": len(yt), "ev": int(yt.sum()),
            "rate": float(yt.mean()),
            "ehr": ae, "ctpa": ac,
            "gap": ae - ac,
            "mean_auc": 0.5 * (ae + ac),
            "gain": g, "lo": lo, "hi": hi,
            "m1_rho_pos": rho_pos,
            "m1_both_miss": both_miss,
            "m2_disc_rate": d_rate,
            "m2_disc_auc": d_auc,
            "m2_disc_diff": d_diff,
            "m3_brier_gap": b_gap,
            "m3_mean_gap": r_gap,
            "m4_overlap": ov,
            "m5_classcond": cc,
            "weight_ehr": w,
            "cross": int(not same)})
        print("  %-4s %-14s gap %+.4f  m1 %.3f"
              "  m2 %+.4f  gain %+.4f"
              % (tag, oc, ae - ac, rho_pos,
                 d_rate, g))

d = pd.DataFrame(rows)
d.to_csv(os.path.join(PROC,
                      "moderators2.csv"),
         index=False)
print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))


def ols(df, ys, xs, lab):
    sub = df[[ys] + xs].dropna()
    if len(sub) < len(xs) + 3:
        print("  %-34s too few" % lab)
        return
    X = np.column_stack(
        [np.ones(len(sub))]
        + [sub[c].values for c in xs])
    y = sub[ys].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = len(sub) - X.shape[1]
    s2 = float(res @ res) / max(dof, 1)
    se = np.sqrt(np.diag(
        s2 * np.linalg.pinv(X.T @ X)))
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(res @ res) / max(ss, 1e-12)
    adj = 1 - (1 - r2) * (len(sub) - 1) \
        / max(dof, 1)
    print("  %s  (n=%d, adjR2=%.3f)"
          % (lab, len(sub), adj))
    for i, nm in enumerate(["intercept"] + xs):
        t = beta[i] / max(se[i], 1e-12)
        p = 2 * (1 - stats.t.cdf(abs(t), dof))
        print("    %-14s %+.5f (se %.5f)"
              "  p=%.4f %s"
              % (nm, beta[i], se[i], p,
                 "*" if p < 0.05 else " "))


LAD = d[d["set"] == "ladder"]
GRID = d[d["set"] == "grid"]
MODS = ["m1_rho_pos", "m1_both_miss",
        "m2_disc_rate", "m2_disc_auc",
        "m2_disc_diff", "m3_brier_gap",
        "m3_mean_gap", "m4_overlap",
        "m5_classcond"]

print("")
print("=" * 62)
print("UNIVARIATE ASSOCIATION WITH GAIN")
print("=" * 62)
for v in MODS:
    for nm, sub in [("ladder", LAD),
                    ("grid", GRID)]:
        s = sub[[v, "gain"]].dropna()
        if len(s) < 5 or s[v].std() == 0:
            continue
        r, p = stats.pearsonr(s[v], s["gain"])
        rho, _ = stats.spearmanr(s[v], s["gain"])
        print("  %-14s %-7s r=%+.3f"
              "  rho=%+.3f  p=%.4f  n=%d"
              % (v, nm, r, rho, p, len(s)))

print("")
print("=" * 62)
print("CONTROLLING FOR GAP")
print("=" * 62)
for v in MODS:
    print("")
    for nm, sub in [("ladder", LAD),
                    ("grid", GRID)]:
        ols(sub, "gain", ["gap", v],
            "%s [%s]" % (v, nm))

print("")
print("=" * 62)
print("OUTCOME TYPE (m6)")
print("=" * 62)
g = GRID.copy()
for oc in f.OUTCOMES:
    g["is_" + oc] = (g["outcome"]
                     == oc).astype(int)
ols(g, "gain", ["gap", "is_death_30d"],
    "outcome: death vs rest [grid]")
ols(g, "gain", ["gap", "is_cv_first"],
    "outcome: cv_first vs rest [grid]")
print("")
print("  mean gain by outcome (grid)")
print(g.groupby("outcome")[
    ["gap", "gain"]].mean().round(4)
    .to_string())

print("")
print("collinearity with gap")
for v in MODS:
    for nm, sub in [("ladder", LAD),
                    ("grid", GRID)]:
        s = sub[[v, "gap"]].dropna()
        if len(s) < 5 or s[v].std() == 0:
            continue
        c = float(np.corrcoef(
            s[v], s["gap"])[0, 1])
        print("  %-14s %-7s %+.3f"
              % (v, nm, c))

print("")
print("saved moderators2.csv", d.shape)