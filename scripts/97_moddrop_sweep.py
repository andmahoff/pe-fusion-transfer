"""Modality dropout rate sweep for intermediate
fusion.

Script 27 tested modality dropout at one setting:
rate 0.2 per modality, applied to 40% of batches.
This sweeps
  rate            0.0 to 0.7 per modality
  batch fraction  0.4 and 1.0
  unit dropout    0.8, the working setting, and
                  0.2, the setting of script 27

and adds a missing-modality stress test, with
CTPA withheld at prediction time for a fraction
of patients. Modality dropout is designed for a
modality missing at prediction time, and every
patient in these cohorts has both EHR and CTPA,
so the stress test is the condition under which
it would be expected to help.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\moddrop_sweep.csv
  results\\moddrop_sweep_log.txt
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
MD_RATES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
BATCH_FRAC = [0.4, 1.0]
UNIT_DROPS = [0.8, 0.2]
MISS_FRAC = [0.0, 0.25, 0.50, 0.75]
WIDTH, LATENT, DEPTH = 32, 16, 1
EPOCHS, BATCH, LR, WD = 200, 64, 1e-3, 1e-4
NFOLD = 5
OUTS = ["death_30d"]
DIRS = ["I2M", "M2I", "M2M", "I2I"]
DEST = os.path.join(PROC,
                    "moddrop_sweep.csv")

torch.set_num_threads(max(1, os.cpu_count() - 1))


def _rank(p):
    return pd.Series(p).rank(pct=True).values


class Enc(nn.Module):
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


class ModDropNet(nn.Module):
    """Intermediate fusion with modality
    dropout: whole modalities are zeroed during
    training, not individual units."""

    def __init__(self, n_a, n_b, drop,
                 md_rate, batch_frac):
        super().__init__()
        self.ea = Enc(n_a, WIDTH, LATENT,
                      DEPTH, drop)
        self.eb = Enc(n_b, WIDTH, LATENT,
                      DEPTH, drop)
        self.head = nn.Sequential(
            nn.Linear(2 * LATENT, WIDTH),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(WIDTH, 1))
        self.md = md_rate
        self.bf = batch_frac

    def forward(self, xa, xb):
        za, zb = self.ea(xa), self.eb(xb)
        if self.training and self.md > 0:
            # apply to a fraction of batches
            if torch.rand(1).item() < self.bf:
                n = za.shape[0]
                ma = (torch.rand(n, 1)
                      > self.md).float()
                mb = (torch.rand(n, 1)
                      > self.md).float()
                # never drop both, or the row
                # carries no information at all
                both = ((ma == 0) & (mb == 0)) \
                    .squeeze(1)
                ma[both] = 1.0
                za, zb = za * ma, zb * mb
        return self.head(torch.cat(
            [za, zb], dim=1)).squeeze(1)


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
    for _ in range(EPOCHS):
        idx = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = idx[i:i + BATCH]
            if len(b) < 4:
                continue
            opt.zero_grad()
            loss = lossf(model(ta[b], tb[b]),
                         ty[b])
            loss.backward()
            opt.step()
    return model


def predict(model, Xa, Xb, miss=0.0,
            seed=0):
    """miss is the fraction of test patients for
    whom CTPA is withheld, replaced by zeros.
    This is the scenario modality dropout was
    designed for."""
    model.eval()
    Xb2 = Xb.copy()
    if miss > 0:
        rng = np.random.default_rng(seed)
        k = rng.random(len(Xb2)) < miss
        Xb2[k] = 0.0
    with torch.no_grad():
        o = model(
            torch.tensor(Xa,
                         dtype=torch.float32),
            torch.tensor(Xb2,
                         dtype=torch.float32))
    return torch.sigmoid(o).numpy()


def get_cell(direction, oc, ins, mim):
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


def prep_cell(cell):
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
    return (Xs_a, Xs_b, ys, Xt_a, Xt_b, yt,
            grp, len(A), len(B))


def run_cfg(P, drop, md, bf, seed,
            miss=0.0):
    (Xs_a, Xs_b, ys, Xt_a, Xt_b, yt, grp,
     n_a, n_b) = P
    if grp is None:
        m = ModDropNet(n_a, n_b, drop, md, bf)
        m = train_net(m, Xs_a, Xs_b, ys, seed)
        p = predict(m, Xt_a, Xt_b, miss, seed)
        return roc_auc_score(yt, p)
    p = np.zeros(len(yt))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    X = np.column_stack([Xs_a, Xs_b])
    for tr, te in cv.split(X, ys, grp):
        m = ModDropNet(n_a, n_b, drop, md, bf)
        m = train_net(m, Xs_a[tr], Xs_b[tr],
                      ys[tr], seed)
        p[te] = predict(m, Xt_a[te], Xt_b[te],
                        miss, seed)
    return roc_auc_score(yt, p)


def ehr_only(P, seed):
    (Xs_a, _, ys, Xt_a, _, yt, grp,
     _, _) = P
    if grp is None:
        return roc_auc_score(
            yt, f.fit_lr(Xs_a, ys, "ehr")
            .predict_proba(Xt_a)[:, 1])
    p = np.zeros(len(yt))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(Xs_a, ys, grp):
        p[te] = f.fit_lr(
            Xs_a[tr], ys[tr], "ehr"
        ).predict_proba(Xt_a[te])[:, 1]
    return roc_auc_score(yt, p)


ins = f.load_inspect()
mim = f.load_mimic()
print("INSPECT", ins.shape, " MIMIC",
      mim.shape)
print("")
print("modality dropout rates:", MD_RATES)
print("batch fractions:", BATCH_FRAC)
print("unit dropout:", UNIT_DROPS)
print("  script 27 tested one point:"
      " rate 0.2 at batch fraction 0.4",
      flush=True)

rows = []
t0 = time.time()

for oc in OUTS:
    for dr in DIRS:
        try:
            cell = get_cell(dr, oc, ins, mim)
            P = prep_cell(cell)
        except Exception as exc:
            print("skip", dr, repr(exc)[:80])
            continue
        eh = np.mean([ehr_only(P, s)
                      for s in SEEDS])
        print("")
        print("=" * 74)
        print("%s  %s   ehr alone %.4f"
              % (dr, oc, eh), flush=True)

        for drop in UNIT_DROPS:
            print("")
            print("  unit dropout %.1f" % drop)
            print("  %-6s %-6s %8s %8s %s"
                  % ("md", "bfrac", "AUC",
                     "SD", "vs ehr"))
            for bf in BATCH_FRAC:
                for md in MD_RATES:
                    aa = [run_cfg(P, drop, md,
                                  bf, s)
                          for s in SEEDS]
                    v = np.array(aa)
                    star = (" (script 27 setting)"
                            if (md == 0.2
                                and bf == 0.4)
                            else "")
                    print("  %-6.1f %-6.1f"
                          " %8.4f %8.4f"
                          "  %+.4f%s"
                          % (md, bf, v.mean(),
                             v.std(ddof=1),
                             v.mean() - eh,
                             star), flush=True)
                    rows.append({
                        "outcome": oc,
                        "direction": dr,
                        "unit_drop": drop,
                        "md_rate": md,
                        "batch_frac": bf,
                        "miss": 0.0,
                        "auc": v.mean(),
                        "sd": v.std(ddof=1),
                        "ehr": eh,
                        "gain_vs_ehr":
                            v.mean() - eh})

        # ---- the scenario it was built for ----
        print("")
        print("  MISSING-MODALITY STRESS TEST"
              "  (unit dropout 0.8)")
        print("  CTPA withheld at prediction"
              " time for a fraction of patients")
        print("  %-6s %s"
              % ("md", "  ".join(
                  "%.0f%%" % (100 * m)
                  for m in MISS_FRAC)))
        for md in (0.0, 0.2, 0.5):
            line = "  %-6.1f" % md
            for ms in MISS_FRAC:
                aa = [run_cfg(P, 0.8, md, 1.0,
                              s, miss=ms)
                      for s in SEEDS]
                v = float(np.mean(aa))
                line += "  %.4f" % v
                rows.append({
                    "outcome": oc,
                    "direction": dr,
                    "unit_drop": 0.8,
                    "md_rate": md,
                    "batch_frac": 1.0,
                    "miss": ms, "auc": v,
                    "sd": float(np.std(
                        aa, ddof=1)),
                    "ehr": eh,
                    "gain_vs_ehr": v - eh})
            print(line, flush=True)

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("GAIN OVER EHR, ALL PATIENTS COMPLETE")
q = r[r["miss"] == 0.0]
for drop in UNIT_DROPS:
    s = q[q["unit_drop"] == drop]
    if not len(s):
        continue
    print("")
    print("  unit dropout %.1f" % drop)
    print(s.pivot_table(
        index=["batch_frac", "md_rate"],
        columns="direction",
        values="gain_vs_ehr")
        .round(4).to_string())

print("")
print("DOES THE RATE MATTER?")
from scipy import stats as st
for drop in UNIT_DROPS:
    for bf in BATCH_FRAC:
        s = q[(q["unit_drop"] == drop)
              & (q["batch_frac"] == bf)]
        if len(s) > 3:
            rr, pp = st.pearsonr(
                s["md_rate"].values,
                s["gain_vs_ehr"].values)
            print("  drop %.1f  bfrac %.1f"
                  "   r = %+.3f  p = %.3f"
                  % (drop, bf, rr, pp))
print("  a flat relationship means the rate"
      " was never the limiting factor")

print("")
print("BEST RATE PER CELL")
for dr in DIRS:
    s = q[q["direction"] == dr]
    if not len(s):
        continue
    b = s.loc[s["auc"].idxmax()]
    z = s[(s["md_rate"] == 0.2)
          & (s["batch_frac"] == 0.4)]
    old = (z["auc"].iloc[0] if len(z)
           else np.nan)
    print("  %-4s  best md=%.1f bfrac=%.1f"
          " drop=%.1f  %.4f"
          "   script 27 setting %.4f"
          "   %+.4f"
          % (dr, b["md_rate"],
             b["batch_frac"],
             b["unit_drop"], b["auc"], old,
             b["auc"] - old))

print("")
print("=" * 74)
print("MISSING-MODALITY STRESS TEST")
print("  this is the only condition under"
      " which modality dropout should win")
m = r[r["miss"] > 0]
if len(m):
    print("")
    print(m.pivot_table(
        index=["direction", "md_rate"],
        columns="miss", values="auc")
        .round(4).to_string())
    print("")
    print("  AUROC LOST AS CTPA IS WITHHELD")
    for dr in DIRS:
        s = r[(r["direction"] == dr)
              & (r["unit_drop"] == 0.8)
              & (r["batch_frac"] == 1.0)]
        if not len(s):
            continue
        for md in (0.0, 0.2, 0.5):
            z = s[s["md_rate"] == md]
            b = z[z["miss"] == 0.0]
            w = z[z["miss"] == 0.75]
            if len(b) and len(w):
                print("    %-4s md=%.1f"
                      "   complete %.4f"
                      "   75%% missing %.4f"
                      "   lost %.4f"
                      % (dr, md,
                         b["auc"].iloc[0],
                         w["auc"].iloc[0],
                         b["auc"].iloc[0]
                         - w["auc"].iloc[0]))
    print("")
    print("  if modality dropout helps"
          " anywhere, it is here: a smaller")
    print("  loss at high missingness for"
          " md > 0 than for md = 0")
print("")
print("saved", DEST, r.shape)