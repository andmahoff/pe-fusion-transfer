"""Paired bootstrap comparison of quantile,
block-standardised and robust scaling against
source-fitted standardisation, with an interval for
each difference.
Run in venv (analysis).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC = f.PROC
SPLIT_SEED = 42
NB = f.NBOOT

DIRS = [("I2M", "inspect", "mimic"),
        ("M2I", "mimic", "inspect"),
        ("M2M", "mimic", "mimic"),
        ("I2I", "inspect", "inspect")]

SCALERS = ["std", "quantile", "blockstd",
           "robust"]

REPS = [("uni", "ehr_only"),
        ("early", "plain"),
        ("late", "wsrc")]


def scale(src_X, tgt_X, mode, nblk=None):
    """Impute on source, then apply the chosen
    scaling. quantile is fitted per dataset, so
    each is mapped to its own percentiles."""
    im = SimpleImputer(strategy="median")
    a = im.fit_transform(src_X)
    b = im.transform(tgt_X)

    if mode == "quantile":
        n = min(1000, a.shape[0])
        qa = QuantileTransformer(
            n_quantiles=n,
            output_distribution="normal",
            random_state=SPLIT_SEED)
        m = min(1000, b.shape[0])
        qb = QuantileTransformer(
            n_quantiles=m,
            output_distribution="normal",
            random_state=SPLIT_SEED)
        return qa.fit_transform(a), \
            qb.fit_transform(b)

    if mode == "robust":
        sc = RobustScaler()
    else:
        sc = StandardScaler()
    a2 = sc.fit_transform(a)
    b2 = sc.transform(b)
    if mode == "blockstd" and nblk:
        a2 = a2 / np.sqrt(nblk)
        b2 = b2 / np.sqrt(nblk)
    return a2, b2


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def run(src, tgt, ys, fam, var, mode):
    se, sc_ = src
    te, tc = tgt
    se, te = scale(se, te, mode, se.shape[1])
    sc_, tc = scale(sc_, tc, mode, sc_.shape[1])

    if fam == "early":
        m = f.mk_lr()
        m.fit(np.hstack([se, sc_]), ys)
        return m.predict_proba(
            np.hstack([te, tc]))[:, 1]

    me = f.mk_lr()
    me.fit(se, ys)
    mc = f.mk_lr()
    mc.fit(sc_, ys)
    pe = _rank(me.predict_proba(te)[:, 1])
    pc = _rank(mc.predict_proba(tc)[:, 1])
    if var == "ehr_only":
        return pe
    if var == "mean":
        return 0.5 * pe + 0.5 * pc
    ps = _rank(me.predict_proba(se)[:, 1])
    qs = _rank(mc.predict_proba(sc_)[:, 1])
    best, bw = -1.0, 0.5
    for w in np.arange(0.0, 1.01, 0.05):
        a = roc_auc_score(
            ys, w * ps + (1 - w) * qs)
        if a > best:
            best, bw = a, w
    return bw * pe + (1 - bw) * pc


def oof(tdf, yt, fam, var, mode):
    xe, xc = f.blocks(tdf)
    grp = tdf["gid"].values
    p = np.zeros(len(yt))
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SPLIT_SEED)
    for tr, te in cv.split(xe, yt, grp):
        p[te] = run((xe[tr], xc[tr]),
                    (xe[te], xc[te]),
                    yt[tr], fam, var, mode)
    return p


ins = f.load_inspect()
mim = f.load_mimic()
DATA = {"inspect": ins, "mimic": mim}

rows = []
for tag, sname, tname in DIRS:
    same = sname == tname
    for oc in f.OUTCOMES:
        sdf, ys = f.labels(DATA[sname], oc)
        tdf, yt = f.labels(DATA[tname], oc)
        grp = tdf["gid"].values
        print("")
        print("=" * 60)
        print("%s  %s  n=%d ev=%d"
              % (tag, oc, len(yt), int(yt.sum())))

        preds = {}
        for fam, var in REPS:
            for mode in SCALERS:
                k = (fam + ":" + var, mode)
                if same:
                    preds[k] = oof(tdf, yt, fam,
                                   var, mode)
                else:
                    preds[k] = run(
                        f.blocks(sdf),
                        f.blocks(tdf), ys,
                        fam, var, mode)

        for fam, var in REPS:
            name = fam + ":" + var
            base = preds[(name, "std")]
            ab = roc_auc_score(yt, base)
            for mode in SCALERS:
                if mode == "std":
                    continue
                p = preds[(name, mode)]
                g, lo, hi, pg = f.boot_diff(
                    yt, p, base, grp, nboot=NB)
                star = ("*" if (lo > 0 or hi < 0)
                        else " ")
                print("  %-14s %-9s %.4f vs"
                      " %.4f  %+.4f"
                      " [%+.4f,%+.4f] %s"
                      % (name, mode,
                         roc_auc_score(yt, p),
                         ab, g, lo, hi, star))
                rows.append({
                    "direction": tag,
                    "outcome": oc, "arch": name,
                    "scaler": mode,
                    "auc": roc_auc_score(yt, p),
                    "auc_std": ab, "diff": g,
                    "lo": lo, "hi": hi,
                    "pgt": pg,
                    "sig": int(lo > 0 or hi < 0)})

r = pd.DataFrame(rows)
r.to_csv(os.path.join(PROC,
                      "scaling_test.csv"),
         index=False)

print("")
print("=" * 60)
print("SUMMARY: difference vs std, by scaler")
print(r.groupby("scaler")[
    ["diff", "sig"]].agg(
    ["mean", "count"]).round(4).to_string())

print("")
print("SIGNIFICANT CELLS (of %d per scaler)"
      % (len(DIRS) * len(f.OUTCOMES)
         * len(REPS)))
for m in SCALERS:
    if m == "std":
        continue
    s = r[r["scaler"] == m]
    up = int(((s["lo"] > 0)).sum())
    dn = int(((s["hi"] < 0)).sum())
    print("  %-9s better %d, worse %d, of %d"
          % (m, up, dn, len(s)))

print("")
print("PRIMARY: I2M death_30d")
print(r[(r["direction"] == "I2M")
        & (r["outcome"] == "death_30d")]
      .round(4).to_string(index=False))