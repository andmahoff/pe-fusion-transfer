"""Extends the dropout sweep past 0.9 to find where
the curve turns, and adds controls that remove the
encoder entirely, to test whether the limit of
intermediate fusion is simply the linear model.
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

# label, dropout, weight decay, mode
# mode: mlp | identity | linear
CONFIGS = [
    ("drop80", 0.80, 1e-4, "mlp"),
    ("drop90", 0.90, 1e-4, "mlp"),
    ("drop93", 0.93, 1e-4, "mlp"),
    ("drop95", 0.95, 1e-4, "mlp"),
    ("drop97", 0.97, 1e-4, "mlp"),
    ("wd1e0", 0.50, 1.0, "mlp"),
    ("wd3e1", 0.50, 0.3, "mlp"),
    ("identity", 0.00, 1e-4, "identity"),
    ("id_drop50", 0.50, 1e-4, "identity"),
    ("linear", 0.00, 1e-4, "linear"),
    ("linear_wd", 0.00, 1e-1, "linear"),
]


class Net(nn.Module):
    """mlp      - two encoders then a joint head
    identity - no encoder, features pass
               straight to the joint head
    linear   - no encoder and no hidden layer,
               so a plain logistic regression on
               the concatenated blocks
    """

    def __init__(self, ne, nc, p, mode):
        super().__init__()
        self.mode = mode
        if mode == "mlp":
            def enc(n):
                return nn.Sequential(
                    nn.Linear(n, HID),
                    nn.BatchNorm1d(HID),
                    nn.ReLU(), nn.Dropout(p),
                    nn.Linear(HID, LAT),
                    nn.ReLU())
            self.ee = enc(ne)
            self.ec = enc(nc)
            d = 2 * LAT
        else:
            d = ne + nc
        if mode == "linear":
            self.head = nn.Linear(d, 1)
        else:
            self.head = nn.Sequential(
                nn.Dropout(p),
                nn.Linear(d, 16), nn.ReLU(),
                nn.Dropout(p), nn.Linear(16, 1))

    def forward(self, xe, xc):
        if self.mode == "mlp":
            z = torch.cat(
                [self.ee(xe), self.ec(xc)], 1)
        else:
            z = torch.cat([xe, xc], 1)
        return self.head(z).squeeze(1)


def run(src, tgt, ys, cfg, seed, epochs=150):
    _, p, wd, mode = cfg
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

    net = Net(xe.shape[1], xc.shape[1], p, mode)
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
    npar = sum(q.numel()
               for q in net.parameters())
    return pt, ps, npar


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
        ea = roc_auc_score(
            yt, f.early(src, tgt, ys, "plain"))

        print("")
        print("=" * 62)
        print("%s  %s  n=%d ev=%d"
              % (tag, oc, len(yt), int(yt.sum())))
        print("  EHR %.4f | late:wsrc %.4f"
              " | early:plain %.4f"
              % (ra, la, ea))
        print("  %-11s %6s %7s %8s %s"
              % ("config", "AUC", "srcAUC",
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
                "wd": cfg[2], "mode": cfg[3],
                "params": npar, "auc": auc,
                "src_auc": sauc,
                "overfit": sauc - auc,
                "ehr_ref": ra, "late_ref": la,
                "early_ref": ea, "gain": g,
                "lo": lo, "hi": hi,
                "vs_late": auc - la,
                "vs_early": auc - ea,
                "sig": int(lo > 0 or hi < 0)})

r = pd.DataFrame(rows)
r.to_csv(os.path.join(PROC,
                      "dropout_ext.csv"),
         index=False)
print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))

g1 = r.groupby("config", sort=False)
s = pd.DataFrame({
    "drop": g1["drop"].first(),
    "mode": g1["mode"].first(),
    "params": g1["params"].first(),
    "auc": g1["auc"].mean(),
    "overfit": g1["overfit"].mean(),
    "gain": g1["gain"].mean(),
    "vs_late": g1["vs_late"].mean(),
    "vs_early": g1["vs_early"].mean(),
    "wins": g1.apply(
        lambda d: int(((d["gain"] > 0)
                       & (d["sig"] == 1)).sum())),
    "losses": g1.apply(
        lambda d: int(((d["gain"] < 0)
                       & (d["sig"] == 1)).sum())),
})
print("")
print("=" * 62)
print("MEAN OVER 4 CELLS")
print(s.round(4).to_string())

print("")
print("HAS THE CURVE TURNED?")
d = s[s["mode"] == "mlp"]
d = d[d["drop"] >= 0.8].sort_values("drop")
print(d[["drop", "auc", "gain",
         "overfit"]].round(4).to_string())
best = d["gain"].idxmax()
print("  best: %s at dropout %.2f"
      % (best, d.loc[best, "drop"]))
if d.loc[best, "drop"] >= 0.97:
    print("  optimum still at the edge of the range")
else:
    print("  TURNED: an interior optimum exists")
if len(d) > 3:
    rr, p = st.pearsonr(d["drop"].values,
                        d["gain"].values)
    print("  dropout vs gain beyond 0.8:"
          "  r=%+.3f  p=%.4f" % (rr, p))

print("")
print("DOES THE ENCODER EARN ITS PLACE?")
print("  best MLP vs the encoder-free controls")
b = s.loc[s[s["mode"] == "mlp"]["gain"].idxmax()]
for nm in ["identity", "id_drop50", "linear",
           "linear_wd"]:
    if nm in s.index:
        print("  %-11s gain %+.4f   (best MLP"
              " %s %+.4f, diff %+.4f)"
              % (nm, s.loc[nm, "gain"], b.name,
                 b["gain"],
                 b["gain"] - s.loc[nm, "gain"]))

print("")
print("PER CELL: best config and how it compares")
for (dd, o), sub in r.groupby(
        ["direction", "outcome"]):
    x = sub.loc[sub["auc"].idxmax()]
    print("  %-4s %-14s %-11s %.4f"
          "  vs EHR %+.4f  vs late %+.4f"
          "  vs early %+.4f"
          % (dd, o, x["config"], x["auc"],
             x["gain"], x["vs_late"],
             x["vs_early"]))

print("")
print("saved dropout_ext.csv", r.shape)