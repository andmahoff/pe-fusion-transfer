"""Encoder depth sweep for intermediate fusion.

Depth varies on its own: 0, 1, 2 and 3 hidden
layers per encoder, built explicitly, with width
fixed at 32 and latent size at 16. In script 23,
depths 1 and 2 built the same network, and the one
deeper configuration also changed width, so its
depth result could not be separated from width.

Each depth runs at dropout 0.5 and 0.8, since
dropout sits in every hidden layer and compounds
with depth. The encoder-free control and
late:wsrc are kept as reference points.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\depth_sweep.csv
  results\\depth_sweep_log.txt
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC = f.PROC
SEEDS = [42, 7, 13]
DEPTHS = [0, 1, 2, 3]
DROPS = [0.5, 0.8]
WIDTH, LATENT = 32, 16
EPOCHS, BATCH, LR = 200, 64, 1e-3
WD = 1e-4
NFOLD = 5
OUTS = ["death_30d"]
DIRS = ["I2M", "M2I", "M2M", "I2I"]
DEST = os.path.join(PROC, "depth_sweep.csv")

torch.set_num_threads(max(1, os.cpu_count() - 1))


def _rank(p):
    return pd.Series(p).rank(pct=True).values


class Enc(nn.Module):
    """depth = number of hidden layers.
    depth 0 is a bare linear projection, so the
    four settings genuinely differ, unlike
    script 23 where 1 and 2 collapsed."""

    def __init__(self, nin, width, latent,
                 depth, drop):
        super().__init__()
        layers, d = [], nin
        for _ in range(depth):
            layers += [nn.Linear(d, width),
                       nn.ReLU(),
                       nn.Dropout(drop)]
            d = width
        layers.append(nn.Linear(d, latent))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class JointNet(nn.Module):
    def __init__(self, n_a, n_b, depth, drop,
                 width=WIDTH, latent=LATENT):
        super().__init__()
        self.ea = Enc(n_a, width, latent,
                      depth, drop)
        self.eb = Enc(n_b, width, latent,
                      depth, drop)
        self.head = nn.Sequential(
            nn.Linear(2 * latent, width),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(width, 1))

    def forward(self, xa, xb):
        return self.head(torch.cat(
            [self.ea(xa), self.eb(xb)],
            dim=1)).squeeze(1)


class NoEnc(nn.Module):
    """The encoder-free control: raw features
    through dropout straight to a small head.
    1,089 parameters on 66 inputs."""

    def __init__(self, nin, drop, latent=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, latent), nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(latent, 1))

    def forward(self, xa, xb):
        return self.net(
            torch.cat([xa, xb], dim=1)).squeeze(1)


def nparams(m):
    return int(sum(p.numel() for p in
                   m.parameters()
                   if p.requires_grad))


def train_net(model, Xa, Xb, y, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(
        model.parameters(), lr=LR,
        weight_decay=WD)
    lossf = nn.BCEWithLogitsLoss()
    ta = torch.tensor(Xa, dtype=torch.float32)
    tb = torch.tensor(Xb, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    n = len(y)
    model.train()
    for ep in range(EPOCHS):
        idx = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = idx[i:i + BATCH]
            if len(b) < 4:
                continue
            opt.zero_grad()
            out = model(ta[b], tb[b])
            loss = lossf(out, ty[b])
            loss.backward()
            opt.step()
    return model


def predict(model, Xa, Xb):
    model.eval()
    with torch.no_grad():
        o = model(
            torch.tensor(Xa,
                         dtype=torch.float32),
            torch.tensor(Xb,
                         dtype=torch.float32))
    return torch.sigmoid(o).numpy()


def get_cell(direction, oc, ins, mim):
    """Returns source and target matrices for a
    direction, with grouping for the
    within-dataset cells."""
    if direction == "I2M":
        ds, ys = f.labels(ins, oc)
        dt, yt = f.labels(mim, oc)
        return ds, ys, dt, yt, None
    if direction == "M2I":
        ds, ys = f.labels(mim, oc)
        dt, yt = f.labels(ins, oc)
        return ds, ys, dt, yt, None
    if direction == "M2M":
        d, y = f.labels(mim, oc)
        return d, y, d, y, d["subject_id"].values
    d, y = f.labels(ins, oc)
    gcol = ("subject_id" if "subject_id"
            in d.columns else "person_id")
    return d, y, d, y, d[gcol].values


def run_cfg(kind, depth, drop, cell, seed):
    """kind is 'joint' or 'noenc'. Returns the
    target AUROC."""
    ds, ys, dt, yt, grp = cell
    A = f.EHR_COLS
    B = [c for c in f.CTPA_COLS
         if c in ds.columns and c in dt.columns]
    Xs_a, Xt_a = f.prep(
        ds[A].values.astype(float),
        dt[A].values.astype(float))
    Xs_b, Xt_b = f.prep(
        ds[B].values.astype(float),
        dt[B].values.astype(float))

    def build():
        if kind == "noenc":
            return NoEnc(len(A) + len(B), drop)
        return JointNet(len(A), len(B), depth,
                        drop)

    if grp is None:
        m = build()
        np_ = nparams(m)
        m = train_net(m, Xs_a, Xs_b, ys, seed)
        p = predict(m, Xt_a, Xt_b)
        return roc_auc_score(yt, p), np_

    # within-dataset: grouped OOF
    p = np.zeros(len(yt))
    np_ = None
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    X = np.column_stack([Xs_a, Xs_b])
    for tr, te in cv.split(X, ys, grp):
        m = build()
        if np_ is None:
            np_ = nparams(m)
        m = train_net(m, Xs_a[tr], Xs_b[tr],
                      ys[tr], seed)
        p[te] = predict(m, Xt_a[te], Xt_b[te])
    return roc_auc_score(yt, p), np_


def late_wsrc(cell, seed):
    """Weighted rank average, weights chosen on
    the source, as the reference point."""
    ds, ys, dt, yt, grp = cell
    A = f.EHR_COLS
    B = [c for c in f.CTPA_COLS
         if c in ds.columns and c in dt.columns]
    a1, b1 = f.prep(
        ds[A].values.astype(float),
        dt[A].values.astype(float))
    a2, b2 = f.prep(
        ds[B].values.astype(float),
        dt[B].values.astype(float))
    if grp is None:
        pa = _rank(f.fit_lr(a1, ys, "ehr")
                   .predict_proba(b1)[:, 1])
        pb = _rank(f.fit_lr(a2, ys, "ctpa")
                   .predict_proba(b2)[:, 1])
        # source-side weight
        sa = _rank(f.fit_lr(a1, ys, "ehr")
                   .predict_proba(a1)[:, 1])
        sb = _rank(f.fit_lr(a2, ys, "ctpa")
                   .predict_proba(a2)[:, 1])
        best, bw = -1.0, 0.5
        for w in np.arange(0, 1.001, 0.05):
            s = roc_auc_score(
                ys, w * sb + (1 - w) * sa)
            if s > best:
                best, bw = s, w
        return roc_auc_score(
            yt, bw * pb + (1 - bw) * pa)
    pa = np.zeros(len(yt))
    pb = np.zeros(len(yt))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(a1, ys, grp):
        pa[te] = f.fit_lr(
            a1[tr], ys[tr], "ehr"
        ).predict_proba(a1[te])[:, 1]
        pb[te] = f.fit_lr(
            a2[tr], ys[tr], "ctpa"
        ).predict_proba(a2[te])[:, 1]
    pa, pb = _rank(pa), _rank(pb)
    best, bw = -1.0, 0.5
    for w in np.arange(0, 1.001, 0.05):
        s = roc_auc_score(yt, w * pb
                          + (1 - w) * pa)
        if s > best:
            best, bw = s, w
    return roc_auc_score(
        yt, bw * pb + (1 - bw) * pa)


def ehr_only(cell, seed):
    ds, ys, dt, yt, grp = cell
    A = f.EHR_COLS
    a1, b1 = f.prep(
        ds[A].values.astype(float),
        dt[A].values.astype(float))
    if grp is None:
        return roc_auc_score(
            yt, f.fit_lr(a1, ys, "ehr")
            .predict_proba(b1)[:, 1])
    p = np.zeros(len(yt))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(a1, ys, grp):
        p[te] = f.fit_lr(
            a1[tr], ys[tr], "ehr"
        ).predict_proba(a1[te])[:, 1]
    return roc_auc_score(yt, p)


ins = f.load_inspect()
mim = f.load_mimic()
print("INSPECT", ins.shape, " MIMIC",
      mim.shape, flush=True)

# show the parameter counts, to confirm the depths differ
print("")
print("PARAMETER COUNTS BY DEPTH")
print("  (script 23 had shallow and base"
      " identical at 3,905)")
for dp in DEPTHS:
    m = JointNet(28, 38, dp, 0.5)
    print("  depth %d  %6d parameters"
          % (dp, nparams(m)))
m = NoEnc(66, 0.5)
print("  noenc     %6d parameters"
      % nparams(m), flush=True)

rows = []
t0 = time.time()

for oc in OUTS:
    for dr in DIRS:
        try:
            cell = get_cell(dr, oc, ins, mim)
        except Exception as exc:
            print("skip", dr, repr(exc)[:80])
            continue
        ds, ys, dt, yt, grp = cell
        print("")
        print("=" * 72)
        print("%s  %s   source n=%d ev=%d"
              "   target n=%d ev=%d"
              % (dr, oc, len(ys),
                 int(ys.sum()), len(yt),
                 int(yt.sum())), flush=True)

        eh = np.mean([ehr_only(cell, s)
                      for s in SEEDS])
        lw = np.mean([late_wsrc(cell, s)
                      for s in SEEDS])
        print("  reference: ehr %.4f"
              "   late:wsrc %.4f (%+.4f)"
              % (eh, lw, lw - eh), flush=True)
        for nm, v in (("ehr", eh),
                      ("late_wsrc", lw)):
            rows.append({
                "outcome": oc, "direction": dr,
                "config": nm, "depth": np.nan,
                "drop": np.nan,
                "nparams": np.nan, "auc": v,
                "sd": np.nan,
                "gain_vs_ehr": v - eh})

        print("")
        print("  %-12s %6s %5s %8s %8s %s"
              % ("config", "params", "drop",
                 "AUC", "SD", "vs ehr"))
        for drop in DROPS:
            for dp in DEPTHS:
                aa, npar = [], None
                for s in SEEDS:
                    a_, n_ = run_cfg(
                        "joint", dp, drop,
                        cell, s)
                    aa.append(a_)
                    npar = n_
                v = np.array(aa)
                print("  d%-11d %6d %5.1f"
                      " %8.4f %8.4f  %+.4f"
                      % (dp, npar, drop,
                         v.mean(),
                         v.std(ddof=1),
                         v.mean() - eh),
                      flush=True)
                rows.append({
                    "outcome": oc,
                    "direction": dr,
                    "config": "depth%d" % dp,
                    "depth": dp, "drop": drop,
                    "nparams": npar,
                    "auc": v.mean(),
                    "sd": v.std(ddof=1),
                    "gain_vs_ehr":
                        v.mean() - eh})
            aa, npar = [], None
            for s in SEEDS:
                a_, n_ = run_cfg(
                    "noenc", 0, drop, cell, s)
                aa.append(a_)
                npar = n_
            v = np.array(aa)
            print("  %-12s %6d %5.1f %8.4f"
                  " %8.4f  %+.4f"
                  % ("noenc", npar, drop,
                     v.mean(), v.std(ddof=1),
                     v.mean() - eh), flush=True)
            rows.append({
                "outcome": oc, "direction": dr,
                "config": "noenc",
                "depth": -1, "drop": drop,
                "nparams": npar,
                "auc": v.mean(),
                "sd": v.std(ddof=1),
                "gain_vs_ehr": v.mean() - eh})

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 72)
print("GAIN OVER EHR ALONE, BY DEPTH")
q = r[r["depth"].notna() & (r["depth"] >= 0)]
for drop in DROPS:
    s = q[q["drop"] == drop]
    if not len(s):
        continue
    print("")
    print("  dropout %.1f" % drop)
    print(s.pivot_table(index="config",
                        columns="direction",
                        values="gain_vs_ehr")
          .round(4).to_string())

print("")
print("DOES DEPTH TRACK GAIN?")
from scipy import stats as st
for drop in DROPS:
    s = q[(q["drop"] == drop)
          & (q["depth"] >= 0)]
    if len(s) > 3:
        rr, pp = st.pearsonr(
            s["depth"].values,
            s["gain_vs_ehr"].values)
        sr, sp = st.spearmanr(
            s["depth"].values,
            s["gain_vs_ehr"].values)
        print("  dropout %.1f   Pearson"
              " r=%+.3f p=%.3f   Spearman"
              " r=%+.3f p=%.3f  on %d points"
              % (drop, rr, pp, sr, sp, len(s)))
print("  script 23 reported r = -0.176 for"
      " depth, but that rested on one")
print("  configuration which also changed"
      " width from 32 to 48")

print("")
print("PARAMETERS vs GAIN")
if len(q) > 3:
    rr, pp = st.pearsonr(
        q["nparams"].values,
        q["gain_vs_ehr"].values)
    print("  Pearson r = %+.3f  p = %.3f"
          % (rr, pp))

print("")
print("ENCODER-FREE CONTROL vs THE BEST"
      " NETWORK")
for drop in DROPS:
    s = r[(r["drop"] == drop)]
    if not len(s):
        continue
    for dr in DIRS:
        ss = s[s["direction"] == dr]
        if not len(ss):
            continue
        ne = ss[ss["config"] == "noenc"]
        bn = ss[ss["config"].str.startswith(
            "depth")]
        if len(ne) and len(bn):
            b = bn.loc[bn["auc"].idxmax()]
            print("  drop %.1f  %-4s  noenc"
                  " %.4f   best %s %.4f"
                  "   %+.4f"
                  % (drop, dr,
                     ne["auc"].iloc[0],
                     b["config"], b["auc"],
                     b["auc"]
                     - ne["auc"].iloc[0]))

print("")
print("BEST PER CELL")
for dr in DIRS:
    s = r[r["direction"] == dr]
    if not len(s):
        continue
    b = s.loc[s["auc"].idxmax()]
    print("  %-4s  %-12s %.4f  (%+.4f over"
          " ehr)"
          % (dr, b["config"], b["auc"],
             b["gain_vs_ehr"]))
print("")
print("saved", DEST, r.shape)