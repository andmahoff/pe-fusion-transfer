"""Sweeps dropout for intermediate fusion from 0.0
to 0.9 with the architecture otherwise fixed, and
adds weight decay and label smoothing as
alternative regularisers. Tests whether the monotone trend
in script 23 continues or turns.
Run in venv (analysis).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as st
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC = f.PROC
SCALE = "std"
LAT = 16
HID = 32

# label, dropout, weight decay, label smoothing
CONFIGS = [
    ("drop00", 0.0, 1e-4, 0.0),
    ("drop20", 0.2, 1e-4, 0.0),
    ("drop40", 0.4, 1e-4, 0.0),
    ("drop50", 0.5, 1e-4, 0.0),
    ("drop60", 0.6, 1e-4, 0.0),
    ("drop70", 0.7, 1e-4, 0.0),
    ("drop80", 0.8, 1e-4, 0.0),
    ("drop90", 0.9, 1e-4, 0.0),
    ("d50_wd1e2", 0.5, 1e-2, 0.0),
    ("d50_wd1e1", 0.5, 1e-1, 0.0),
    ("d50_ls10", 0.5, 1e-4, 0.10),
    ("d80_wd1e2", 0.8, 1e-2, 0.0),
]


def enc(n, lat, hid, p):
    return nn.Sequential(
        nn.Linear(n, hid), nn.BatchNorm1d(hid),
        nn.ReLU(), nn.Dropout(p),
        nn.Linear(hid, lat), nn.ReLU())


class Net(nn.Module):
    def __init__(self, ne, nc, p):
        super().__init__()
        self.ee = enc(ne, LAT, HID, p)
        self.ec = enc(nc, LAT, HID, p)
        self.head = nn.Sequential(
            nn.Linear(2 * LAT, 16), nn.ReLU(),
            nn.Dropout(p), nn.Linear(16, 1))

    def forward(self, xe, xc):
        a = self.ee(xe)
        b = self.ec(xc)
        return self.head(
            torch.cat([a, b], dim=1)).squeeze(1)


def run(src, tgt, ys, cfg, seed, epochs=150):
    _, p, wd, ls = cfg
    torch.manual_seed(seed)
    np.random.seed(seed)
    se, sc_ = src
    te, tc = tgt
    se, te = f.prep(se, te, SCALE)
    sc_, tc = f.prep(sc_, tc, SCALE)

    xe = torch.tensor(se, dtype=torch.float32)
    xc = torch.tensor(sc_, dtype=torch.float32)
    yv = ys.astype(float)
    if ls > 0:
        yv = yv * (1 - ls) + 0.5 * ls
    yy = torch.tensor(yv, dtype=torch.float32)
    qe = torch.tensor(te, dtype=torch.float32)
    qc = torch.tensor(tc, dtype=torch.float32)

    net = Net(xe.shape[1], xc.shape[1], p)
    pw = torch.tensor(
        float((ys == 0).sum())
        / max(1.0, float((ys == 1).sum())))
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(
        net.parameters(), lr=1e-3,
        weight_decay=wd)

    n = len(ys)
    rng = np.random.default_rng(seed)
    net.train()
    for _ in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, 256):
            j = idx[i:i + 256]
            if len(j) < 8:
                continue
            opt.zero_grad()
            crit(net(xe[j], xc[j]),
                 yy[j]).backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        pt = torch.sigmoid(net(qe, qc)).numpy()
        ps = torch.sigmoid(net(xe, xc)).numpy()
    return pt, ps


torch.set_num_threads(max(1, os.cpu_count() - 1))
ins = f.load_inspect()
mim = f.load_mimic()
DIRS = [("I2M", ins, mim), ("M2I", mim, ins)]

rows = []
t0 = time.time()
total = len(DIRS) * 2 * len(CONFIGS)
done = 0

for tag, s0, t0d in DIRS:
    for oc in ["death_30d", "composite_30d"]:
        sdf, ys = f.labels(s0, oc)
        tdf, yt = f.labels(t0d, oc)
        grp = tdf["gid"].values
        src = f.blocks(sdf)
        tgt = f.blocks(tdf)

        ref = f.late(src, tgt, ys, "ehr_only")
        ra = roc_auc_score(yt, ref)
        la = roc_auc_score(
            yt, f.late(src, tgt, ys, "wsrc"))

        print("")
        print("=" * 62)
        print("%s  %s  n=%d ev=%d"
              % (tag, oc, len(yt), int(yt.sum())))
        print("  EHR alone %.4f | late:wsrc %.4f"
              % (ra, la))
        print("  %-11s %6s %7s %8s %s"
              % ("config", "AUC", "srcAUC",
                 "overfit", "gain vs EHR"))

        for cfg in CONFIGS:
            acc, accs = [], []
            for sd in f.SEEDS:
                pt, ps = run(src, tgt, ys,
                             cfg, sd)
                acc.append(pt)
                accs.append(ps)
            pm = np.mean(acc, axis=0)
            sm = np.mean(accs, axis=0)
            auc = roc_auc_score(yt, pm)
            sauc = roc_auc_score(ys, sm)
            g, lo, hi, _ = f.boot_diff(
                yt, pm, ref, grp)
            star = ("*" if (lo > 0 or hi < 0)
                    else " ")
            done += 1
            el = (time.time() - t0) / 60.0
            print("  %-11s %.4f %.4f  %+.4f"
                  "  %+.4f [%+.4f,%+.4f] %s"
                  "  [%d/%d %.0fm eta %.0fm]"
                  % (cfg[0], auc, sauc,
                     sauc - auc, g, lo, hi, star,
                     done, total, el,
                     el / done * (total - done)))
            rows.append({
                "direction": tag, "outcome": oc,
                "config": cfg[0], "drop": cfg[1],
                "wd": cfg[2], "ls": cfg[3],
                "auc": auc, "src_auc": sauc,
                "overfit": sauc - auc,
                "ehr_ref": ra, "late_ref": la,
                "gain": g, "lo": lo, "hi": hi,
                "vs_late": auc - la,
                "sig": int(lo > 0 or hi < 0)})

r = pd.DataFrame(rows)
r.to_csv(os.path.join(PROC, "dropout.csv"),
         index=False)
print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))

print("")
print("=" * 62)
print("MEAN OVER 4 CELLS")
g1 = r.groupby("config", sort=False)
s = pd.DataFrame({
    "drop": g1["drop"].first(),
    "auc": g1["auc"].mean(),
    "src_auc": g1["src_auc"].mean(),
    "overfit": g1["overfit"].mean(),
    "gain": g1["gain"].mean(),
    "vs_late": g1["vs_late"].mean(),
    "wins": g1.apply(
        lambda d: int(((d["gain"] > 0)
                       & (d["sig"] == 1)).sum())),
    "losses": g1.apply(
        lambda d: int(((d["gain"] < 0)
                       & (d["sig"] == 1)).sum())),
})
print(s.round(4).to_string())

print("")
print("DROPOUT SWEEP ONLY, sorted by gain")
d = s[s.index.str.startswith("drop")]
print(d.sort_values("gain", ascending=False)
      .round(4).to_string())

print("")
print("IS THE TREND MONOTONE?")
x = d["drop"].values
for v in ["gain", "overfit", "auc"]:
    rr, p = st.pearsonr(x, d[v].values)
    rho, _ = st.spearmanr(x, d[v].values)
    print("  dropout vs %-8s r=%+.3f"
          "  rho=%+.3f  p=%.4f"
          % (v, rr, rho, p))
best = d["gain"].idxmax()
print("  best dropout: %s (gain %+.4f)"
      % (best, d.loc[best, "gain"]))
if best in ("drop80", "drop90"):
    print("  NOTE: optimum at the edge of the"
          " range; the curve has not turned")
else:
    print("  optimum is interior, so the curve"
          " has turned")

print("")
print("BEST OVERALL PER CELL")
for (dd, o), sub in r.groupby(
        ["direction", "outcome"]):
    b = sub.loc[sub["auc"].idxmax()]
    print("  %-4s %-14s %-11s %.4f"
          "  vs EHR %+.4f  vs late %+.4f"
          % (dd, o, b["config"], b["auc"],
             b["gain"], b["vs_late"]))

print("")
print("ANY CELL WHERE INTERMEDIATE BEATS"
      " late:wsrc?")
w = r[r["vs_late"] > 0]
print(w[["direction", "outcome", "config",
         "auc", "late_ref", "vs_late"]]
      .round(4).to_string(index=False)
      if len(w) else "  none")

print("")
print("saved dropout.csv", r.shape)