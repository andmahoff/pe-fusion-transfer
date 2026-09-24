"""The fusion grid rerun under quantile scaling.
Tests whether the architecture conclusions from
script 08 survive a better preprocessing baseline,
or were an artefact of source-fitted
standardisation.
Run in venv (analysis).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC = f.PROC
SPLIT_SEED = 42

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


def qscale(src_X, tgt_X):
    """Quantile mapping, fitted per dataset."""
    im = SimpleImputer(strategy="median")
    a = im.fit_transform(src_X)
    b = im.transform(tgt_X)
    qa = QuantileTransformer(
        n_quantiles=min(1000, a.shape[0]),
        output_distribution="normal",
        random_state=SPLIT_SEED)
    qb = QuantileTransformer(
        n_quantiles=min(1000, b.shape[0]),
        output_distribution="normal",
        random_state=SPLIT_SEED)
    return qa.fit_transform(a), qb.fit_transform(b)


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def inter_q(src, tgt, ys, variant, seed):
    """Intermediate fusion with quantile scaling
    in place of the library's standardisation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    se, sc_ = src
    te, tc = tgt
    se, te = qscale(se, te)
    sc_, tc = qscale(sc_, tc)

    moddrop = 0.2 if variant == "moddrop" else 0.0
    xe = torch.tensor(se, dtype=torch.float32)
    xc = torch.tensor(sc_, dtype=torch.float32)
    yy = torch.tensor(ys, dtype=torch.float32)
    qe = torch.tensor(te, dtype=torch.float32)
    qc = torch.tensor(tc, dtype=torch.float32)

    net = f.JointNet(xe.shape[1], xc.shape[1],
                     variant=variant, lat=f.LAT)
    pw = torch.tensor(
        float((ys == 0).sum()) /
        max(1.0, float((ys == 1).sum())))
    crit = torch.nn.BCEWithLogitsLoss(
        pos_weight=pw)
    opt = torch.optim.Adam(
        net.parameters(), lr=1e-3,
        weight_decay=1e-4)

    n = len(ys)
    rng = np.random.default_rng(seed)
    net.train()
    for _ in range(150):
        idx = rng.permutation(n)
        for i in range(0, n, 256):
            j = idx[i:i + 256]
            if len(j) < 8:
                continue
            be, bc = xe[j], xc[j]
            if moddrop > 0:
                u = rng.random()
                if u < moddrop:
                    be = torch.zeros_like(be)
                elif u < 2 * moddrop:
                    bc = torch.zeros_like(bc)
            opt.zero_grad()
            crit(net(be, bc), yy[j]).backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        return torch.sigmoid(
            net(qe, qc)).numpy()


def call(fam, var, src, tgt, ys, seed):
    se, sc_ = src
    te, tc = tgt
    if fam == "inter":
        return inter_q(src, tgt, ys, var, seed)

    a_e, b_e = qscale(se, te)
    a_c, b_c = qscale(sc_, tc)

    if fam == "early":
        if var == "blockstd":
            a_e = a_e / np.sqrt(a_e.shape[1])
            b_e = b_e / np.sqrt(b_e.shape[1])
            a_c = a_c / np.sqrt(a_c.shape[1])
            b_c = b_c / np.sqrt(b_c.shape[1])
        m = f.mk_lr()
        m.fit(np.hstack([a_e, a_c]), ys)
        return m.predict_proba(
            np.hstack([b_e, b_c]))[:, 1]

    me = f.mk_lr()
    me.fit(a_e, ys)
    mc = f.mk_lr()
    mc.fit(a_c, ys)
    pe = _rank(me.predict_proba(b_e)[:, 1])
    pc = _rank(mc.predict_proba(b_c)[:, 1])
    if var == "ehr_only":
        return pe
    if var == "ctpa_only":
        return pc
    if var == "mean":
        return 0.5 * pe + 0.5 * pc
    ps = _rank(me.predict_proba(a_e)[:, 1])
    qs = _rank(mc.predict_proba(a_c)[:, 1])
    best, bw = -1.0, 0.5
    for w in np.arange(0.0, 1.01, 0.05):
        a = roc_auc_score(
            ys, w * ps + (1 - w) * qs)
        if a > best:
            best, bw = a, w
    return bw * pe + (1 - bw) * pc


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

        ref = preds["uni:ehr_only"]
        ra = roc_auc_score(yt, ref)
        print("  EHR alone: %.4f" % ra)
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
                "ehr_ref": ra, "gain": g,
                "lo": lo, "hi": hi,
                "sig": int(lo > 0 or hi < 0)})

r = pd.DataFrame(rows)
r.to_csv(os.path.join(PROC,
                      "grid_quantile.csv"),
         index=False)
print("")
print("elapsed %.1f min" % ((time.time() - t0) / 60))

print("")
print("GAIN OVER EHR ALONE, MEAN OVER 12 CELLS")
print(r.groupby("arch")[["gain", "sig"]]
      .agg(["mean", "sum"]).round(4).to_string())

old = os.path.join(PROC, "grid_summary.csv")
if os.path.exists(old):
    o = pd.read_csv(old)
    o = o.groupby("arch")["gain"].mean()
    n = r.groupby("arch")["gain"].mean()
    c = pd.DataFrame({"std": o, "quantile": n})
    c["change"] = c["quantile"] - c["std"]
    print("")
    print("GAIN UNDER std VS quantile")
    print(c.dropna().round(4).to_string())

print("")
print("PRIMARY: I2M death_30d")
print(r[(r["direction"] == "I2M")
        & (r["outcome"] == "death_30d")]
      .round(4).to_string(index=False))