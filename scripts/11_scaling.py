"""Six feature-scaling conditions crossed with the
representative fusion architectures, across all four
transfer directions and three outcomes. None of the
conditions uses target-domain labels.
Run in venv (analysis).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC = f.PROC
SPLIT_SEED = 42
EPS = 1e-6

SCALERS = ["std", "blockstd", "robust",
           "quantile", "tgtfit", "coral"]
ARCHS = ["ehr_only", "early", "late_mean",
         "late_wsrc"]

DIRS = [("I2M", "inspect", "mimic"),
        ("M2I", "mimic", "inspect"),
        ("M2M", "mimic", "mimic"),
        ("I2I", "inspect", "inspect")]


def impute(a, b):
    im = SimpleImputer(strategy="median")
    return im.fit_transform(a), im.transform(b)


def coral(a, b):
    """Align source covariance to target."""
    ca = np.cov(a, rowvar=False) + np.eye(
        a.shape[1]) * EPS
    cb = np.cov(b, rowvar=False) + np.eye(
        b.shape[1]) * EPS
    wa = linalg.fractional_matrix_power(ca, -0.5)
    wb = linalg.fractional_matrix_power(cb, 0.5)
    out = a @ np.real(wa) @ np.real(wb)
    return np.asarray(out, dtype=float)


def scale(a, b, how, width=None):
    """a is source, b is target. Returns both.
    No target labels are used anywhere here."""
    a, b = impute(a, b)
    if how in ("std", "blockstd"):
        s = StandardScaler().fit(a)
        a, b = s.transform(a), s.transform(b)
        if how == "blockstd":
            k = np.sqrt(float(width))
            a, b = a / k, b / k
    elif how == "robust":
        s = RobustScaler().fit(a)
        a, b = s.transform(a), s.transform(b)
    elif how == "quantile":
        n = max(10, min(1000, len(a)))
        qa = QuantileTransformer(
            n_quantiles=n, output_distribution="uniform",
            random_state=42).fit(a)
        m = max(10, min(1000, len(b)))
        qb = QuantileTransformer(
            n_quantiles=m, output_distribution="uniform",
            random_state=42).fit(b)
        a, b = qa.transform(a), qb.transform(b)
    elif how == "tgtfit":
        s = StandardScaler().fit(b)
        a, b = s.transform(a), s.transform(b)
    elif how == "coral":
        s = StandardScaler().fit(a)
        a, b = s.transform(a), s.transform(b)
        a = coral(a, b)
    else:
        raise ValueError(how)
    return a, b


def mk():
    return LogisticRegression(C=0.1, max_iter=5000)


def rank(p):
    return pd.Series(p).rank(pct=True).values


def run(src, tgt, ys, how, arch):
    se, sc = src
    te, tc = tgt
    se, te = scale(se, te, how, se.shape[1])
    sc, tc = scale(sc, tc, how, sc.shape[1])

    if arch == "early":
        m = mk()
        m.fit(np.hstack([se, sc]), ys)
        return m.predict_proba(
            np.hstack([te, tc]))[:, 1]

    me = mk()
    me.fit(se, ys)
    pe = rank(me.predict_proba(te)[:, 1])
    if arch == "ehr_only":
        return pe
    mc = mk()
    mc.fit(sc, ys)
    pc = rank(mc.predict_proba(tc)[:, 1])
    if arch == "late_mean":
        return 0.5 * pe + 0.5 * pc
    if arch == "late_wsrc":
        qe = rank(me.predict_proba(se)[:, 1])
        qc = rank(mc.predict_proba(sc)[:, 1])
        best, bw = -1.0, 0.5
        for w in np.arange(0.0, 1.01, 0.05):
            a = roc_auc_score(
                ys, w * qe + (1 - w) * qc)
            if a > best:
                best, bw = a, w
        return bw * pe + (1 - bw) * pc
    raise ValueError(arch)


def oof(tdf, yt, how, arch):
    xe, xc = f.blocks(tdf)
    grp = tdf["gid"].values
    p = np.zeros(len(yt))
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SPLIT_SEED)
    for tr, te in cv.split(xe, yt, grp):
        p[te] = run((xe[tr], xc[tr]),
                    (xe[te], xc[te]),
                    yt[tr], how, arch)
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

        cell = {}
        for how in SCALERS:
            line = "  %-9s" % how
            for arch in ARCHS:
                if same:
                    p = oof(tdf, yt, how, arch)
                else:
                    p = run(f.blocks(sdf),
                            f.blocks(tdf), ys,
                            how, arch)
                a = roc_auc_score(yt, p)
                cell[(how, arch)] = p
                line += "  %s %.4f" % (arch[:4], a)
                rows.append({
                    "direction": tag,
                    "outcome": oc, "scaler": how,
                    "arch": arch, "n": len(yt),
                    "ev": int(yt.sum()), "auc": a,
                    "cross": not same})
            print(line)

        base = cell[("std", "ehr_only")]
        for how in SCALERS:
            for arch in ARCHS:
                if how == "std" and \
                        arch == "ehr_only":
                    continue
                g, lo, hi, _ = f.boot_diff(
                    yt, cell[(how, arch)],
                    base, grp, nboot=1000)
                for r in rows:
                    if (r["direction"] == tag
                            and r["outcome"] == oc
                            and r["scaler"] == how
                            and r["arch"] == arch):
                        r["gain"] = g
                        r["lo"] = lo
                        r["hi"] = hi

r = pd.DataFrame(rows)
r.to_csv(os.path.join(PROC, "scaling.csv"),
         index=False)
print("")
print("elapsed %.1f min" % ((time.time() - t0) / 60))

print("")
print("MEAN AUC BY SCALER AND ARCH")
print(r.pivot_table(index="scaler",
                    columns="arch",
                    values="auc")
      .round(4).to_string())

print("")
print("REGISTERED TEST: cross-dataset vs internal")
print("(quantile should help only where shift"
      " exists)")
piv = r.pivot_table(index="scaler",
                    columns="cross",
                    values="auc")
piv.columns = ["internal", "cross"]
piv["difference"] = piv["cross"] - piv["internal"]
print(piv.round(4).to_string())

print("")
print("PRIMARY: I2M death_30d")
p = r[(r["direction"] == "I2M")
      & (r["outcome"] == "death_30d")]
print(p.pivot_table(index="scaler",
                    columns="arch",
                    values="auc")
      .round(4).to_string())