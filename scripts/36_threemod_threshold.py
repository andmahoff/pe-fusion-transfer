"""Tests when a third modality is worth adding,
comparing three candidate predictors of
three-modality gain.
Run in venv (analysis).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats as st
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
SEED, NFOLD = 42, 5
SCALE = "std"
# late_best gap line from 20_gap_tuned.py, typed in
INT2, SL2 = 0.0360, -0.2425

OUTS = ["death_30d", "death_30d_inhosp",
        "composite_30d", "cv_first"]
DEGRADE = [("full", 1.00), ("d80", 0.80),
           ("d60", 0.60), ("d40", 0.40),
           ("d20", 0.20)]


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def blur(p, frac, seed=SEED):
    if frac >= 1.0:
        return p
    rng = np.random.default_rng(seed)
    return (frac * _rank(p)
            + (1.0 - frac) * rng.random(len(p)))


def gw(ps, y, step=0.05):
    k = len(ps)
    best, bw = -1.0, tuple([1.0 / k] * k)
    if k == 2:
        for w in np.arange(0, 1.001, step):
            a = roc_auc_score(
                y, w * ps[0] + (1 - w) * ps[1])
            if a > best:
                best, bw = a, (w, 1 - w)
    else:
        for w1 in np.arange(0, 1.001, step):
            for w2 in np.arange(
                    0, 1.001 - w1, step):
                a = roc_auc_score(
                    y, w1 * ps[0] + w2 * ps[1]
                    + (1 - w1 - w2) * ps[2])
                if a > best:
                    best, bw = a, (
                        w1, w2, 1 - w1 - w2)
    return bw


def wcv(ps, y, grp):
    out = np.zeros(len(y))
    ws = []
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    X = np.column_stack(ps)
    for tr, te in cv.split(X, y, grp):
        w = gw([p[tr] for p in ps], y[tr])
        ws.append(w)
        out[te] = sum(wi * p[te]
                      for wi, p in zip(w, ps))
    return out, np.mean(ws, axis=0)


def labels_any(df, oc):
    if oc in f.OUTCOMES:
        return f.labels(df, oc)
    y = pd.to_numeric(df[oc], errors="coerce")
    return df, y.fillna(0).astype(int).values


keys = pd.read_csv(os.path.join(
    FIG, "fusion_cohort_keys.csv"))
keys = keys[["hadm_id", "p_ecg", "p_ecg_max",
             "p_cxr"]].drop_duplicates("hadm_id")

mm = f.load_mimic()
ins = f.load_inspect()
rows = []
t0 = time.time()

for oc in OUTS:
    if oc not in mm.columns:
        continue
    d, y = labels_any(mm, oc)
    grp = d["subject_id"].values
    src_oc = ("death_30d"
              if oc == "death_30d_inhosp"
              else oc)
    se, ysrc = f.labels(ins, src_oc)

    a, b = f.prep(
        se[f.EHR_COLS].values.astype(float),
        d[f.EHR_COLS].values.astype(float),
        SCALE)
    P = {"ehr": f.fit_lr(
        a, ysrc, "ehr").predict_proba(b)[:, 1]}
    c, e = f.prep(
        se[f.CTPA_COLS].values.astype(float),
        d[f.CTPA_COLS].values.astype(float),
        SCALE)
    P["ctpa"] = f.fit_lr(
        c, ysrc, "ctpa").predict_proba(e)[:, 1]

    k = d[["hadm_id"]].merge(
        keys, on="hadm_id", how="left")
    P["ecg"] = 0.5 * (
        _rank(k["p_ecg"].values)
        + _rank(k["p_ecg_max"].values))
    P["cxr"] = k["p_cxr"].values

    print("")
    print("=" * 62)
    print("%s  n=%d ev=%d"
          % (oc, len(y), int(y.sum())))

    names = [t for t in P
             if np.isfinite(P[t]).sum() > 300]
    for pair in combinations(names, 2):
        for t3 in [t for t in names
                   if t not in pair]:
            ok = np.ones(len(y), dtype=bool)
            for t in list(pair) + [t3]:
                ok &= np.isfinite(P[t])
            if ok.sum() < 300 or y[ok].sum() < 30:
                continue
            yy, gg = y[ok], grp[ok]
            R = {t: _rank(P[t][ok])
                 for t in list(pair) + [t3]}
            p2, _ = wcv(
                [R[pair[0]], R[pair[1]]],
                yy, gg)
            a2 = roc_auc_score(yy, p2)
            best1 = max(
                roc_auc_score(yy, R[pair[0]]),
                roc_auc_score(yy, R[pair[1]]))

            for lab, frac in DEGRADE:
                v = _rank(blur(R[t3], frac))
                a3rd = roc_auc_score(yy, v)
                p3, w3 = wcv(
                    [R[pair[0]], R[pair[1]], v],
                    yy, gg)
                a3 = roc_auc_score(yy, p3)
                rho, _ = st.spearmanr(v, p2)
                rows.append({
                    "outcome": oc,
                    "pair": "+".join(pair),
                    "third": t3, "degrade": lab,
                    "n": int(ok.sum()),
                    "ev": int(yy.sum()),
                    "auc_third": a3rd,
                    "auc_2mod": a2,
                    "auc_best1": best1,
                    "auc_3mod": a3,
                    "gain": a3 - a2,
                    "gap_single": best1 - a3rd,
                    "gap_fused": a2 - a3rd,
                    "corr_fused": float(rho),
                    "w_third": float(w3[2]),
                    "pred_2modline":
                        INT2 + SL2
                        * (best1 - a3rd)})
            print("  %-14s + %-5s  2mod %.4f"
                  "  3mod %.4f  gain %+.4f"
                  % ("+".join(pair), t3, a2,
                     rows[-len(DEGRADE)]
                     ["auc_3mod"],
                     rows[-len(DEGRADE)]["gain"]))

r = pd.DataFrame(rows)
r["resid"] = r["gain"] - r["pred_2modline"]
r.to_csv(os.path.join(PROC, "threemod.csv"),
         index=False)
full = r[r["degrade"] == "full"].copy()
print("")
print("points generated:", len(r))
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))


def fit(x, y_, lab, out):
    m = ~(np.isnan(x) | np.isnan(y_))
    x, y_ = x[m], y_[m]
    if len(x) < 6:
        print("  %-18s too few" % lab)
        return None
    sl, ic, rr, p, se_ = st.linregress(x, y_)
    rho, _ = st.spearmanr(x, y_)
    print("  %-18s slope %+.4f (se %.4f)"
          "  r=%+.3f  R2=%.3f  p=%.4f"
          "  rho=%+.3f  n=%d"
          % (lab, sl, se_, rr, rr ** 2, p,
             rho, len(x)))
    out.append({"pred": lab, "slope": sl,
                "se": se_, "intercept": ic,
                "r2": rr ** 2, "p": p,
                "n": len(x)})
    return ic, sl


print("")
print("=" * 62)
print("WHICH PREDICTOR EXPLAINS 3-MOD GAIN?")
print("=" * 62)
res = []
for nm, sub in (("raw", full), ("all", r)):
    print("")
    print(nm.upper(), "(n=%d)" % len(sub))
    for v in ["gap_single", "gap_fused",
              "corr_fused", "auc_third"]:
        fit(sub[v].values, sub["gain"].values,
            "%s [%s]" % (v, nm), res)

print("")
print("TWO-MODALITY LINE OVER-PREDICTS BY")
print("  all points: %+.4f"
      % float(r["resid"].mean()))
print("  raw cells:  %+.4f"
      % float(full["resid"].mean()))

print("")
print("=" * 62)
print("TWO VS THREE MODALITY SLOPES")
print("=" * 62)
print("  two-modality (script 20):"
      "  %+.4f, zero at gap 0.148"
      % SL2)
for tag in ("gap_single [raw]",
            "gap_single [all]"):
    o = [x for x in res if x["pred"] == tag]
    if not o:
        continue
    o = o[0]
    z = (-o["intercept"] / o["slope"]
         if o["slope"] < 0 else np.nan)
    print("  three-modality %-16s"
          " %+.4f, zero at gap %.3f"
          % (tag, o["slope"], z))
    print("    gain = %+.4f %+.4f * gap"
          "   (R2 %.3f, n %d)"
          % (o["intercept"], o["slope"],
             o["r2"], o["n"]))
    for g in (0.00, 0.05, 0.10, 0.15, 0.20):
        print("      gap %.2f -> %+.4f"
              % (g, o["intercept"]
                 + o["slope"] * g))

print("")
print("APPLIED TO THE OBSERVED CELLS")
o = [x for x in res
     if x["pred"] == "gap_single [raw]"]
if o:
    ic, sl = o[0]["intercept"], o[0]["slope"]
    v = full[["outcome", "pair", "third",
              "auc_third", "auc_2mod",
              "gap_single", "gain"]].copy()
    v["predicted"] = ic + sl * v["gap_single"]
    v["call"] = np.where(
        v["predicted"] > 0, "add", "skip")
    v["correct"] = np.where(
        (v["predicted"] > 0) == (v["gain"] > 0),
        "yes", "no")
    print(v.sort_values("gap_single")
          .round(4).to_string(index=False))
    print("")
    print("  rule accuracy: %.1f%% of %d cells"
          % (100 * float(
              (v["correct"] == "yes").mean()),
             len(v)))

pd.DataFrame(res).to_csv(
    os.path.join(PROC,
                 "threemod_regression.csv"),
    index=False)
print("")
print("saved threemod.csv,"
      " threemod_regression.csv")