"""Fit the CTPA modality on frozen transformer
embeddings and compare against the 38-feature
regex baseline. Reports in-distribution and
transferred performance for four encoders, with
full-dimension and PCA-reduced variants.
Run in venv (analysis).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC = f.PROC
EMB = os.path.join("data", "embeddings")
SPLIT_SEED = 42
NPC = 50

# encoder name -> whether it was pretrained on MIMIC
ENC = {"radbert": False,
       "pubmedbert": False,
       "clinicalbert": True,
       "bioclinmodern": True}

CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def mk(c):
    return LogisticRegression(
        C=c, max_iter=5000)


def prep_emb(a, b, npc=None):
    sc = StandardScaler()
    x = sc.fit_transform(a)
    y = sc.transform(b)
    if npc:
        p = PCA(n_components=min(npc,
                                 x.shape[1],
                                 x.shape[0] - 1),
                random_state=SPLIT_SEED)
        x = p.fit_transform(x)
        y = p.transform(y)
    return x, y


def pick_c(X, y, grp):
    """Choose C on SOURCE data only, so the
    transfer stays label-free on the target."""
    best, bc = -1.0, 0.1
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True,
        random_state=SPLIT_SEED)
    for c in CS:
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y, grp):
            m = mk(c)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(
                X[te])[:, 1]
        a = roc_auc_score(y, oof)
        if a > best:
            best, bc = a, c
    return bc, best


ins = f.load_inspect()
mim = f.load_mimic()

iid = pd.read_csv(os.path.join(
    EMB, "ids_inspect.csv"))
mid = pd.read_csv(os.path.join(
    EMB, "ids_mimic.csv"))

# align embedding rows to the loader order
ipos = {v: i for i, v in
        enumerate(iid["person_id"])}
mpos = {v: i for i, v in
        enumerate(mid["hadm_id"])}
iord = np.array([ipos[v] for v
                 in ins["person_id"]])
mord = np.array([mpos[v] for v
                 in mim["hadm_id"]])
print("aligned INSPECT %d, MIMIC %d"
      % (len(iord), len(mord)))

E = {}
for nm in ENC:
    a = np.load(os.path.join(
        EMB, "%s_inspect.npy" % nm))[iord]
    b = np.load(os.path.join(
        EMB, "%s_mimic.npy" % nm))[mord]
    E[nm] = (a, b)
    print("  %-14s %s %s" % (nm, a.shape,
                             b.shape))

rows = []
for oc in f.OUTCOMES:
    sdf, ys = f.labels(ins, oc)
    tdf, yt = f.labels(mim, oc)
    si = sdf.index.values
    ti = tdf.index.values
    sg = sdf["gid"].values
    tg = tdf["gid"].values
    smask = np.isin(ins.index.values, si)
    tmask = np.isin(mim.index.values, ti)

    print("")
    print("=" * 62)
    print("%s  INSPECT n=%d ev=%d |"
          " MIMIC n=%d ev=%d"
          % (oc, len(ys), int(ys.sum()),
             len(yt), int(yt.sum())))

    # regex baseline on the same rows
    Xs = sdf[f.CTPA_COLS].values.astype(float)
    Xt = tdf[f.CTPA_COLS].values.astype(float)
    a, b = f.prep(Xs, Xt)
    bc, cv_auc = pick_c(a, ys, sg)
    m = mk(bc)
    m.fit(a, ys)
    p = m.predict_proba(b)[:, 1]
    zs = roc_auc_score(yt, p)
    print("  %-22s dim %4d  I2I %.4f"
          "  I2M %.4f  (C=%.3f)"
          % ("regex", a.shape[1], cv_auc,
             zs, bc))
    rows.append({"outcome": oc,
                 "encoder": "regex",
                 "variant": "features",
                 "dim": a.shape[1], "C": bc,
                 "ins_cv": cv_auc, "i2m": zs,
                 "i2m_ap":
                 average_precision_score(yt, p),
                 "contaminated": 0})

    for nm, contam in ENC.items():
        ea, eb = E[nm]
        ea = ea[smask]
        eb = eb[tmask]
        for tag, npc in (("full", None),
                         ("pca%d" % NPC, NPC)):
            a, b = prep_emb(ea, eb, npc)
            bc, cv_auc = pick_c(a, ys, sg)
            m = mk(bc)
            m.fit(a, ys)
            p = m.predict_proba(b)[:, 1]
            zs = roc_auc_score(yt, p)
            flag = " [contam]" if contam else ""
            print("  %-14s %-7s dim %4d"
                  "  I2I %.4f  I2M %.4f"
                  "  (C=%.3f)%s"
                  % (nm, tag, a.shape[1],
                     cv_auc, zs, bc, flag))
            rows.append({
                "outcome": oc, "encoder": nm,
                "variant": tag,
                "dim": a.shape[1], "C": bc,
                "ins_cv": cv_auc, "i2m": zs,
                "i2m_ap":
                average_precision_score(yt, p),
                "contaminated": int(contam)})

r = pd.DataFrame(rows)
r.to_csv(os.path.join(PROC, "bert_arm.csv"),
         index=False)
print("")
print("saved bert_arm.csv", r.shape)

print("")
print("=" * 62)
print("DEATH_30D, sorted by transferred AUC")
d = r[r["outcome"] == "death_30d"]
d = d.sort_values("i2m", ascending=False)
print(d.round(4).to_string(index=False))

print("")
print("IN-DISTRIBUTION VS TRANSFER COST")
d = d.copy()
d["cost"] = d["ins_cv"] - d["i2m"]
print(d[["encoder", "variant", "ins_cv",
         "i2m", "cost", "contaminated"]]
      .round(4).to_string(index=False))

print("")
print("PREDICTED FUSION GAIN FOR EACH ENCODER")
print("  fitted line: gain = a + b * gap,"
       " b = -0.2490")
# typed in from the earlier gap regression
EHR = 0.8594
IC, SL = 0.03871, -0.2490
best = d[(d["outcome"] == "death_30d")
         & (d["contaminated"] == 0)]
for _, x in best.iterrows():
    gp = EHR - x["i2m"]
    print("  %-14s %-7s ctpa %.4f"
          " -> gap %.4f -> predicted %+.4f"
          % (x["encoder"], x["variant"],
             x["i2m"], gp, IC + SL * gp))