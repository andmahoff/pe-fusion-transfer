"""Re-runs the seven intermediate fusion
mechanisms across dropout levels. All were
originally tested at 0.2, which scripts 25 and 26
showed is the overfitting regime where no
mechanism can help. 0.8 is the stable optimum;
0.9 and above collapse and are excluded.
Also compares against the encoder-free control,
which script 26 found competitive.
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
DROPS = [0.2, 0.5, 0.8]


class NoEnc(nn.Module):
    """Encoder-free control: raw features go
    straight to a dropout-regularised head."""

    def __init__(self, ne, nc, p):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(ne + nc, 16), nn.ReLU(),
            nn.Dropout(p), nn.Linear(16, 1))

    def forward(self, xe, xc):
        return self.head(
            torch.cat([xe, xc], 1)).squeeze(1)


def build(variant, ne, nc, p):
    if variant == "noenc":
        return NoEnc(ne, nc, p)
    return f.JointNet(ne, nc, variant=variant,
                      lat=f.LAT, p=p)


def run(src, tgt, ys, variant, p, seed,
        epochs=150):
    torch.manual_seed(seed)
    np.random.seed(seed)
    se, sc_ = src
    te, tc = tgt
    se, te = f.prep(se, te, SCALE)
    sc_, tc = f.prep(sc_, tc, SCALE)

    moddrop = 0.2 if variant == "moddrop" else 0.0
    xe = torch.tensor(se, dtype=torch.float32)
    xc = torch.tensor(sc_, dtype=torch.float32)
    yy = torch.tensor(ys, dtype=torch.float32)
    qe = torch.tensor(te, dtype=torch.float32)
    qc = torch.tensor(tc, dtype=torch.float32)

    net = build(variant, xe.shape[1],
                xc.shape[1], p)
    pw = torch.tensor(
        float((ys == 0).sum())
        / max(1.0, float((ys == 1).sum())))
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(
        net.parameters(), lr=1e-3,
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
        pt = torch.sigmoid(net(qe, qc)).numpy()
        ps = torch.sigmoid(net(xe, xc)).numpy()
    npar = sum(q.numel()
               for q in net.parameters())
    return pt, ps, npar


VARIANTS = list(f.INTER_VARIANTS) + ["noenc"]

torch.set_num_threads(max(1, os.cpu_count() - 1))
ins = f.load_inspect()
mim = f.load_mimic()
DIRS = [("I2M", ins, mim), ("M2I", mim, ins)]

rows = []
t0 = time.time()
total = (len(DIRS) * 2 * len(DROPS)
         * len(VARIANTS))
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
        print("  EHR %.4f | late %.4f"
              " | early %.4f" % (ra, la, ea))

        for p in DROPS:
            print("  -- dropout %.1f --" % p)
            for v in VARIANTS:
                acc, accs, npar = [], [], 0
                for sd in f.SEEDS:
                    pt, ps, npar = run(
                        src, tgt, ys, v, p, sd)
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
                print("    %-10s %5d %.4f %.4f"
                      "  %+.4f  %+.4f"
                      " [%+.4f,%+.4f] %s"
                      "  [%d/%d eta %.0fm]"
                      % (v, npar, auc, sauc,
                         sauc - auc, g, lo, hi,
                         star, done, total,
                         el / done
                         * (total - done)))
                rows.append({
                    "direction": tag,
                    "outcome": oc, "variant": v,
                    "drop": p, "params": npar,
                    "auc": auc, "src_auc": sauc,
                    "overfit": sauc - auc,
                    "ehr_ref": ra,
                    "late_ref": la,
                    "early_ref": ea, "gain": g,
                    "lo": lo, "hi": hi,
                    "vs_late": auc - la,
                    "vs_early": auc - ea,
                    "sig": int(lo > 0
                               or hi < 0)})

r = pd.DataFrame(rows)
r.to_csv(os.path.join(PROC,
                      "mechanisms_hd.csv"),
         index=False)
print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))

print("")
print("=" * 62)
print("MEAN GAIN BY MECHANISM AND DROPOUT")
print(r.pivot_table(index="variant",
                    columns="drop",
                    values="gain")
      .round(4).to_string())

print("")
print("MEAN OVERFIT BY MECHANISM AND DROPOUT")
print(r.pivot_table(index="variant",
                    columns="drop",
                    values="overfit")
      .round(4).to_string())

print("")
print("SIGNIFICANT WINS (of 4 cells each)")
w = r[(r["gain"] > 0) & (r["sig"] == 1)]
if len(w):
    print(w.pivot_table(index="variant",
                        columns="drop",
                        values="gain",
                        aggfunc="count")
          .fillna(0).astype(int).to_string())
else:
    print("  none")

print("")
print("SIGNIFICANT LOSSES")
l = r[(r["gain"] < 0) & (r["sig"] == 1)]
if len(l):
    print(l.pivot_table(index="variant",
                        columns="drop",
                        values="gain",
                        aggfunc="count")
          .fillna(0).astype(int).to_string())
else:
    print("  none")

BEST = max(DROPS)
print("")
print("=" * 62)
print("AT DROPOUT %.1f, sorted by gain" % BEST)
hd = r[r["drop"] == BEST]
g1 = hd.groupby("variant")
s = pd.DataFrame({
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
print(s.sort_values("gain", ascending=False)
      .round(4).to_string())

print("")
print("DOES ANY MECHANISM BEAT plain"
      " AT DROPOUT %.1f?" % BEST)
base = float(s.loc["plain", "gain"])
for v in s.sort_values(
        "gain", ascending=False).index:
    if v == "plain":
        continue
    print("  %-10s %+.4f  (plain %+.4f,"
          " diff %+.4f)"
          % (v, s.loc[v, "gain"], base,
             s.loc[v, "gain"] - base))

print("")
print("DO THE ENCODERS EARN THEIR PLACE?")
ne = float(s.loc["noenc", "gain"])
better = [v for v in s.index
          if v != "noenc"
          and s.loc[v, "gain"] > ne]
print("  encoder-free control: %+.4f"
      " (%d params)"
      % (ne, int(s.loc["noenc", "params"])))
print("  mechanisms beating it: %s"
      % (", ".join(better) if better
         else "NONE"))

print("")
print("INTERACTION: does dropout change the"
      " ORDERING of mechanisms?")
for p in DROPS:
    sub = r[r["drop"] == p].groupby(
        "variant")["gain"].mean()
    print("  dropout %.1f  best %-10s %+.4f"
          "   worst %-10s %+.4f"
          % (p, sub.idxmax(), sub.max(),
             sub.idxmin(), sub.min()))
a = r[r["drop"] == DROPS[0]].groupby(
    "variant")["gain"].mean()
b = r[r["drop"] == BEST].groupby(
    "variant")["gain"].mean()
rho, pv = st.spearmanr(a.rank(), b.rank())
print("  rank correlation %.1f vs %.1f:"
      "  rho=%+.3f  p=%.4f"
      % (DROPS[0], BEST, rho, pv))
if pv > 0.05:
    print("  ORDERING NOT STABLE: the original"
          " mechanism ranking does not hold at"
          " the correct operating point")
else:
    print("  ordering stable across dropout")

print("")
print("PER CELL: best mechanism at dropout %.1f"
      % BEST)
for (d, o), sub in hd.groupby(
        ["direction", "outcome"]):
    x = sub.loc[sub["auc"].idxmax()]
    print("  %-4s %-14s %-10s %.4f"
          "  vs EHR %+.4f  vs late %+.4f"
          "  vs early %+.4f"
          % (d, o, x["variant"], x["auc"],
             x["gain"], x["vs_late"],
             x["vs_early"]))

print("")
print("saved mechanisms_hd.csv", r.shape)