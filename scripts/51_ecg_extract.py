"""Regenerate the 71 SCP logits from the PTB-XL
checkpoint, so the head can be refitted.

Architecture read from the checkpoint: fastai
XResNet1D101, numbered sequential stages, stem
12->32->32->64, blocks [3,4,23,3], expansion 4,
head 1024->71 after concat pooling.

Preprocessing, matching the dissertation's ECG
pipeline:
  - lead swap at positions 4 and 5
  - 500 Hz to 100 Hz via resample_poly
  - standardiser mean -0.00077717, scale
    0.23895642
  - 250-sample windows, 125 stride, elementwise
    MAX aggregation
Run in venv_ecg only.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from scipy.signal import resample_poly

# folder holding the dissertation's derived files
DIS = os.environ.get(
    "PE_DISS_DIR", os.path.join("..", "Dissertation"))
HPC = os.path.join(DIS, "HPC")
CKPT = os.path.join(
    HPC, "xresnet1d101_ptbxl_weights.pt")
OUT = os.path.join("data", "raw", "ecg")
PROC = os.path.join("data", "processed")
DEST = os.path.join(PROC, "ecg_logits.csv")

MEAN, SCALE = -0.00077717, 0.23895642
WIN, STRIDE = 250, 125
FS_IN, FS_OUT = 500, 100
NCLS = 71


def conv(ni, nf, ks=3, stride=1):
    return nn.Conv1d(ni, nf, ks, stride=stride,
                     padding=ks // 2,
                     bias=False)


def cbr(ni, nf, ks=3, stride=1, act=True,
        zero_bn=False):
    bn = nn.BatchNorm1d(nf)
    nn.init.constant_(bn.weight,
                      0.0 if zero_bn else 1.0)
    layers = [conv(ni, nf, ks, stride), bn]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Block(nn.Module):
    """fastai XResNet bottleneck, expansion 4."""

    def __init__(self, ni, nh, stride=1,
                 expansion=4):
        super().__init__()
        nf = nh * expansion
        self.convs = nn.Sequential(
            cbr(ni, nh, 1),
            cbr(nh, nh, 3, stride=stride),
            cbr(nh, nf, 1, act=False,
                zero_bn=True))
        self.idconv = (None if ni == nf
                       else cbr(ni, nf, 1,
                                act=False))
        self.pool = (None if stride == 1
                     else nn.AvgPool1d(
                         2, stride=2,
                         ceil_mode=True))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        idt = x
        if self.pool is not None:
            idt = self.pool(idt)
        if self.idconv is not None:
            idt = self.idconv(idt)
        return self.act(self.convs(x) + idt)


def build(nin=12, ncls=NCLS,
          layers=(3, 4, 23, 3)):
    stem = [cbr(nin, 32, 5, stride=2),
            cbr(32, 32, 5),
            cbr(32, 64, 5)]
    mods = list(stem)
    mods.append(nn.MaxPool1d(3, stride=2,
                             padding=1))
    ni = 64
    blocks = []
    for i, n in enumerate(layers):
        nh = 64 * (2 ** i)
        for j in range(n):
            stride = 2 if (i > 0 and j == 0) \
                else 1
            blocks.append(
                nn.Sequential(
                    Block(ni, nh, stride)))
            ni = nh * 4
    mods += blocks
    mods += [nn.AdaptiveAvgPool1d(1),
             nn.Flatten(),
             nn.Linear(ni * 2, ncls)]
    return nn.Sequential(*mods), ni


print("loading checkpoint ...")
obj = torch.load(CKPT, map_location="cpu",
                 weights_only=False)
sd = obj
for k in ("state_dict", "model",
          "model_state_dict"):
    if isinstance(obj, dict) and k in obj:
        sd = obj[k]
        break
sd = {k: v for k, v in sd.items()
      if hasattr(v, "shape")}
print("tensors in checkpoint:", len(sd))

net, nfeat = build()
own = net.state_dict()
print("tensors in model:", len(own))

ck = list(sd.items())
mk = list(own.items())
if len(ck) != len(mk):
    print("")
    print("COUNT MISMATCH - mapping by ORDER")
    print("  checkpoint %d vs model %d"
          % (len(ck), len(mk)))

new, bad = {}, 0
for (ka, va), (kb, vb) in zip(ck, mk):
    if tuple(va.shape) == tuple(vb.shape):
        new[kb] = va
    else:
        bad += 1
        if bad <= 8:
            print("  SHAPE MISMATCH")
            print("    ckpt %-46s %s"
                  % (ka, tuple(va.shape)))
            print("    model %-45s %s"
                  % (kb, tuple(vb.shape)))

print("")
print("matched by order: %d of %d  (%d bad)"
      % (len(new), len(mk), bad))
if bad:
    raise SystemExit(
        "architecture does not match the"
        " checkpoint; see the mismatches above")

net.load_state_dict(new, strict=True)
net.eval()
print("STRICT LOAD SUCCEEDED")

# take logits from the full model, and the
# pre-head features as well
feat_net = nn.Sequential(*list(net)[:-1])


def preprocess(sig):
    """Lead swap, resample, standardise."""
    s = sig.copy()
    s[:, [4, 5]] = s[:, [5, 4]]
    s = resample_poly(s, FS_OUT, FS_IN, axis=0)
    s = (s - MEAN) / SCALE
    return s.astype(np.float32)


def windows(s):
    n = s.shape[0]
    out = []
    i = 0
    while i + WIN <= n:
        out.append(s[i:i + WIN])
        i += STRIDE
    if not out:
        pad = np.zeros((WIN, s.shape[1]),
                       dtype=np.float32)
        pad[:n] = s
        out.append(pad)
    return np.stack(out)


def norm_rec(r):
    s = str(r).replace("\\", "/").strip("/")
    if not s.startswith("files/"):
        s = "files/" + s
    return s


idx = pd.read_csv(os.path.join(
    OUT, "ecg_record_index.csv"))
idx["rec"] = idx["rec"].apply(norm_rec)
recs = sorted(idx["rec"].unique())
print("")
print("records:", len(recs))

torch.set_num_threads(
    max(1, os.cpu_count() - 1))
rows, bad2 = [], 0
t0 = time.time()

for i, r_ in enumerate(recs):
    try:
        p = os.path.join(
            OUT, *r_.split("/"))
        rec = wfdb.rdrecord(p)
        s = np.nan_to_num(rec.p_signal, nan=0.0)
        w = windows(preprocess(s))
        x = torch.tensor(
            w).permute(0, 2, 1)
        with torch.no_grad():
            lg = net(x).numpy()
        agg = lg.max(axis=0)
        d = {"rec": r_}
        for j in range(NCLS):
            d["scp_%02d" % j] = float(agg[j])
        rows.append(d)
    except Exception as exc:
        bad2 += 1
        if bad2 <= 3:
            print("  FAIL", r_, repr(exc)[:150])
    if (i + 1) % 250 == 0:
        el = time.time() - t0
        rate = (i + 1) / max(el, 1e-9)
        print("  %d/%d  %.1f/s  eta %.0f min"
              % (i + 1, len(recs), rate,
                 (len(recs) - i - 1)
                 / max(rate, 1e-9) / 60),
              flush=True)

d = pd.DataFrame(rows)
print("")
print("extracted:", len(d), " failed:", bad2)
if not len(d):
    raise SystemExit("nothing extracted")

d.to_csv(os.path.join(
    PROC, "ecg_logits_record.csv"),
    index=False)

m = idx[["subject_id", "hadm_id",
         "rec"]].drop_duplicates()
j = m.merge(d, on="rec", how="inner")
cols = [c for c in j.columns
        if c.startswith("scp_")]
g = j.groupby(["subject_id", "hadm_id"])
agg = g[cols].agg(["mean", "max"])
agg.columns = ["%s_%s" % (a, b)
               for a, b in agg.columns]
agg = agg.reset_index()
agg["n_ecg"] = g.size().values
agg.to_csv(DEST, index=False)
print("saved", DEST, agg.shape)

print("")
print("LOGIT SANITY")
v = d[cols].values
print("  range: %.2f to %.2f"
      % (v.min(), v.max()))
print("  mean: %.3f  sd: %.3f"
      % (v.mean(), v.std()))
print("  per-statement sd, first 8:",
      np.round(v.std(axis=0)[:8], 3))
print("  constant statements:",
      int((v.std(axis=0) < 1e-6).sum()),
      "of", len(cols))