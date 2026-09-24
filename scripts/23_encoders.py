"""Tests whether encoder capacity changes the
intermediate-fusion result. Varies latent width,
depth, dropout, shared versus separate encoders
and training length, holding the fusion mechanism
fixed at plain concatenation.
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

# label, latent, hidden, depth, dropout,
# shared, epochs, lr
CONFIGS = [
    ("base",       16, 32, 2, 0.2, 0, 150, 1e-3),
    ("lat4",        4, 32, 2, 0.2, 0, 150, 1e-3),
    ("lat8",        8, 32, 2, 0.2, 0, 150, 1e-3),
    ("lat32",      32, 32, 2, 0.2, 0, 150, 1e-3),
    ("shallow",    16, 32, 1, 0.2, 0, 150, 1e-3),
    ("deep",       16, 48, 3, 0.2, 0, 150, 1e-3),
    ("drop05",     16, 32, 2, 0.5, 0, 150, 1e-3),
    ("drop00",     16, 32, 2, 0.0, 0, 150, 1e-3),
    ("shared_enc", 16, 32, 2, 0.2, 1, 150, 1e-3),
    ("ep400",       8, 16, 2, 0.2, 0, 400, 1e-3),
]


def make_enc(n, lat, hid, depth, p):
    layers = []
    d = n
    for _ in range(max(1, depth - 1)):
        layers += [nn.Linear(d, hid),
                   nn.BatchNorm1d(hid),
                   nn.ReLU(), nn.Dropout(p)]
        d = hid
    layers += [nn.Linear(d, lat), nn.ReLU()]
    return nn.Sequential(*layers)


class Net(nn.Module):
    """Plain concatenation, so any difference is
    encoder capacity rather than the fusion
    mechanism."""

    def __init__(self, ne, nc, lat, hid, depth,
                 p, shared):
        super().__init__()
        self.shared = shared
        if shared:
            self.w = max(ne, nc)
            self.enc = make_enc(
                self.w, lat, hid, depth, p)
        else:
            self.ee = make_enc(ne, lat, hid,
                               depth, p)
            self.ec = make_enc(nc, lat, hid,
                               depth, p)
        self.head = nn.Sequential(
            nn.Linear(2 * lat, 16), nn.ReLU(),
            nn.Dropout(p), nn.Linear(16, 1))

    def _pad(self, x):
        k = self.w - x.shape[1]
        if k <= 0:
            return x
        z = torch.zeros(x.shape[0], k,
                        device=x.device)
        return torch.cat([x, z], dim=1)

    def forward(self, xe, xc):
        if self.shared:
            a = self.enc(self._pad(xe))
            b = self.enc(self._pad(xc))
        else:
            a = self.ee(xe)
            b = self.ec(xc)
        return self.head(
            torch.cat([a, b], dim=1)).squeeze(1)


def run(src, tgt, ys, cfg, seed):
    (_, lat, hid, depth, p, shared,
     epochs, lr) = cfg
    torch.manual_seed(seed)
    np.random.seed(seed)
    se, sc_ = src
    te, tc = tgt
    se, te = f.prep(se, te, SCALE)
    sc_, tc = f.prep(sc_, tc, SCALE)

    xe = torch.tensor(se, dtype=torch.float32)
    xc = torch.tensor(sc_, dtype=torch.float32)
    yy = torch.tensor(ys, dtype=torch.float32)
    qe = torch.tensor(te, dtype=torch.float32)
    qc = torch.tensor(tc, dtype=torch.float32)

    net = Net(xe.shape[1], xc.shape[1], lat,
              hid, depth, p, shared)
    pw = torch.tensor(
        float((ys == 0).sum())
        / max(1.0, float((ys == 1).sum())))
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(
        net.parameters(), lr=lr,
        weight_decay=1e-4)

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
        p_t = torch.sigmoid(net(qe, qc)).numpy()
        p_s = torch.sigmoid(net(xe, xc)).numpy()
    npar = sum(q.numel()
               for q in net.parameters())
    return p_t, p_s, npar


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
        lw = f.late(src, tgt, ys, "wsrc")
        la = roc_auc_score(yt, lw)

        print("")
        print("=" * 62)
        print("%s  %s  n=%d ev=%d"
              % (tag, oc, len(yt), int(yt.sum())))
        print("  EHR alone %.4f | late:wsrc %.4f"
              % (ra, la))
        print("  %-11s %6s %6s %7s %8s %s"
              % ("config", "par", "AUC", "srcAUC",
                 "overfit", "gain vs EHR"))

        for cfg in CONFIGS:
            acc, accs, npar = [], [], 0
            for sd in f.SEEDS:
                pt, ps, npar = run(
                    src, tgt, ys, cfg, sd)
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
            print("  %-11s %6d %.4f %.4f  %+.4f"
                  "  %+.4f [%+.4f,%+.4f] %s"
                  "  [%d/%d %.0fm eta %.0fm]"
                  % (cfg[0], npar, auc, sauc,
                     sauc - auc, g, lo, hi, star,
                     done, total, el,
                     el / done * (total - done)))
            rows.append({
                "direction": tag, "outcome": oc,
                "config": cfg[0], "lat": cfg[1],
                "hid": cfg[2], "depth": cfg[3],
                "drop": cfg[4], "shared": cfg[5],
                "epochs": cfg[6], "lr": cfg[7],
                "params": npar, "auc": auc,
                "src_auc": sauc,
                "overfit": sauc - auc,
                "ehr_ref": ra, "late_ref": la,
                "gain": g, "lo": lo, "hi": hi,
                "sig": int(lo > 0 or hi < 0)})

r = pd.DataFrame(rows)
r.to_csv(os.path.join(PROC, "encoders.csv"),
         index=False)
print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))

print("")
print("=" * 62)
print("MEAN OVER 4 CELLS, sorted by gain")
g1 = r.groupby("config")
s = pd.DataFrame({
    "params": g1["params"].mean(),
    "auc": g1["auc"].mean(),
    "overfit": g1["overfit"].mean(),
    "gain": g1["gain"].mean(),
    "wins": g1.apply(
        lambda d: int(((d["gain"] > 0)
                       & (d["sig"] == 1)).sum())),
    "losses": g1.apply(
        lambda d: int(((d["gain"] < 0)
                       & (d["sig"] == 1)).sum())),
})
print(s.sort_values("gain", ascending=False)
      .round(4).to_string())

print("")
print("BEST CONFIG PER CELL vs late:wsrc")
for (d, o), sub in r.groupby(
        ["direction", "outcome"]):
    b = sub.loc[sub["auc"].idxmax()]
    print("  %-4s %-14s best %-11s %.4f"
          "  vs late %.4f  (%+.4f)"
          % (d, o, b["config"], b["auc"],
             b["late_ref"],
             b["auc"] - b["late_ref"]))

print("")
print("DOES CAPACITY HELP OR HURT?")
for v in ["params", "lat", "depth"]:
    x = np.log(r[v].values.astype(float) + 1)
    rr, p = st.pearsonr(x, r["gain"].values)
    print("  log(%-6s) vs gain:    r=%+.3f"
          "  p=%.4f" % (v, rr, p))
x = np.log(r["params"].values.astype(float) + 1)
rr, p = st.pearsonr(x, r["overfit"].values)
print("  log(params) vs overfit:  r=%+.3f"
      "  p=%.4f" % (rr, p))

print("")
print("saved encoders.csv", r.shape)