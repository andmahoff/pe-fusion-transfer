"""Regenerate the 71 SCP logits from the exp0
benchmark checkpoint.

Architecture read from the state dict:
  0,1,2   stem 12->32->32->64, kernel 5
  4       3 bottleneck blocks, hidden 64,
          out 256, with an idpath projection
  5,6,7   4, 23 and 3 blocks, hidden 64,
          out 256, stride 2 on the first block,
          no idpath since widths already match
  8       concat pool 512, BN,
          Linear 512->128, ReLU, BN,
          Linear 128->71

Every stage keeps a hidden width of 64 and
expands to 256, which is why the network is
only 3.70 M parameters.

fastai registers the same module as both
`convs` and `convpath.0`, so the checkpoint
carries 594 duplicates. The loader drops them
after verifying they are identical.

Preprocessing, confirmed from the checkpoint's
own scaler pickle:
  lead swap at positions 4 and 5
  500 Hz -> 100 Hz via resample_poly
  standardise by mean -0.00077717,
  scale 0.23895642
  250-sample windows, 125 stride,
  elementwise MAX aggregation
Run in venv_ecg only.

OUTPUT FILES
  data\\processed\\exp0_logits_record.csv
  data\\processed\\exp0_logits.csv
  results\\extract_exp0_log.txt
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from scipy.signal import resample_poly

CKPT = os.path.join(
    os.path.expanduser("~"),
    "ecg_ptbxl_benchmarking", "output", "exp0",
    "models", "fastai_xresnet1d101", "models",
    "fastai_xresnet1d101.pth")
OUT = os.path.join("data", "raw", "ecg")
PROC = os.path.join("data", "processed")
os.makedirs(PROC, exist_ok=True)
REC_DEST = os.path.join(
    PROC, "exp0_logits_record.csv")
DEST = os.path.join(PROC, "exp0_logits.csv")

MEAN, SCALE = -0.00077717, 0.23895642
WIN, STRIDE = 250, 125
FS_IN, FS_OUT = 500, 100
NCLS = 71
NH, EXP = 64, 4
# tag, n_blocks, stride on first block
LAYERS = [(4, 3, 1), (5, 4, 2),
          (6, 23, 2), (7, 3, 2)]


def cbr(ni, nf, ks=3, stride=1, act=True):
    layers = [nn.Conv1d(ni, nf, ks,
                        stride=stride,
                        padding=ks // 2,
                        bias=False),
              nn.BatchNorm1d(nf)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, ni, nh=NH, stride=1,
                 exp=EXP, ks=5):
        super().__init__()
        nf = nh * exp
        self.convs = nn.Sequential(
            cbr(ni, nh, 1),
            cbr(nh, nh, ks, stride=stride),
            cbr(nh, nf, 1, act=False))
        self.idpath = None
        if ni != nf:
            self.idpath = nn.Sequential(
                cbr(ni, nf, 1, act=False))
        self.pool = (nn.AvgPool1d(
            2, stride=2, ceil_mode=True)
            if stride != 1 else None)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        idt = x
        if self.pool is not None:
            idt = self.pool(idt)
        if self.idpath is not None:
            idt = self.idpath(idt)
        o = self.convs(x)
        if o.shape[-1] != idt.shape[-1]:
            n = min(o.shape[-1], idt.shape[-1])
            o, idt = o[..., :n], idt[..., :n]
        return self.act(o + idt)


class Net(nn.Module):
    def __init__(self, nin=12, ncls=NCLS):
        super().__init__()
        self.m0 = cbr(nin, 32, 5, stride=2)
        self.m1 = cbr(32, 32, 5)
        self.m2 = cbr(32, 64, 5)
        self.m3 = nn.MaxPool1d(3, stride=2,
                               padding=1)
        ni = 64
        for tag, n, st in LAYERS:
            blocks = []
            for j in range(n):
                blocks.append(ResBlock(
                    ni, NH,
                    stride=(st if j == 0
                            else 1)))
                ni = NH * EXP
            setattr(self, "m%d" % tag,
                    nn.Sequential(*blocks))
        self.nf = ni
        self.bn1 = nn.BatchNorm1d(ni * 2)
        self.fc1 = nn.Linear(ni * 2, 128)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, ncls)

    def embed(self, x):
        x = self.m3(self.m2(
            self.m1(self.m0(x))))
        for tag, _, _ in LAYERS:
            x = getattr(self, "m%d" % tag)(x)
        z = torch.cat(
            [nn.functional.adaptive_avg_pool1d(
                x, 1),
             nn.functional.adaptive_max_pool1d(
                 x, 1)], dim=1).flatten(1)
        return self.relu(self.fc1(self.bn1(z)))

    def forward(self, x):
        e = self.embed(x)
        return self.fc2(self.bn2(e)), e


def remap(sd):
    out = {}
    for k, v in sd.items():
        if ".convpath." in k:
            continue
        p = k.split(".")
        top = p[0]
        if top in ("0", "1", "2", "4", "5",
                   "6", "7"):
            out["m%s.%s" % (
                top, ".".join(p[1:]))] = v
        elif top == "8":
            sub, rest = p[1], ".".join(p[2:])
            key = {"2": "bn1", "4": "fc1",
                   "6": "bn2",
                   "8": "fc2"}.get(sub)
            if key:
                out[key + "." + rest] = v
    return out


print("loading checkpoint ...", flush=True)
raw = torch.load(CKPT, map_location="cpu",
                 weights_only=False)["model"]
raw = {k: v for k, v in raw.items()
       if hasattr(v, "shape")}
print("checkpoint tensors:", len(raw))

dup = 0
for k in raw:
    if ".convpath.0." in k:
        tw = k.replace(".convpath.0.",
                       ".convs.")
        if tw in raw and torch.equal(
                raw[k].float(),
                raw[tw].float()):
            dup += 1
print("convpath duplicates confirmed:", dup)

sd = remap(raw)
print("after remap:", len(sd))

net = Net()
own = net.state_dict()
print("model tensors:", len(own))

miss = [k for k in own if k not in sd]
extra = [k for k in sd if k not in own]
bad = [k for k in own if k in sd
       and tuple(own[k].shape)
       != tuple(sd[k].shape)]
print("in model not ckpt:", len(miss))
for k in miss[:12]:
    print("   ", k, tuple(own[k].shape))
print("in ckpt not model:", len(extra))
for k in extra[:12]:
    print("   ", k, tuple(sd[k].shape))
print("shape mismatches:", len(bad))
for k in bad[:12]:
    print("    %-42s model %s ckpt %s"
          % (k, tuple(own[k].shape),
             tuple(sd[k].shape)))

if miss or extra or bad:
    raise SystemExit(
        "architecture mismatch; see the"
        " lists above")

net.load_state_dict(sd, strict=True)
net.eval()
print("STRICT LOAD SUCCEEDED", flush=True)

lab = pd.read_csv(os.path.join(
    PROC, "scp_labels.csv"))
NAMES = list(lab["scp"])
print("statement names:", len(NAMES))


def preprocess(sig):
    s = sig.copy()
    s[:, [4, 5]] = s[:, [5, 4]]
    s = resample_poly(s, FS_OUT, FS_IN, axis=0)
    return ((s - MEAN) / SCALE).astype(
        np.float32)


def windows(s):
    n, out, i = s.shape[0], [], 0
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
    return s if s.startswith("files/") \
        else "files/" + s


idx = pd.read_csv(os.path.join(
    OUT, "ecg_record_index.csv"))
idx["rec"] = idx["rec"].apply(norm_rec)
recs = sorted(idx["rec"].unique())
print("records:", len(recs), flush=True)

torch.set_num_threads(
    max(1, os.cpu_count() - 1))
rows, nbad = [], 0
t0 = time.time()

for i, r_ in enumerate(recs):
    try:
        p = os.path.join(OUT, *r_.split("/"))
        rec = wfdb.rdrecord(p)
        s = np.nan_to_num(rec.p_signal,
                          nan=0.0)
        w = windows(preprocess(s))
        x = torch.tensor(w).permute(0, 2, 1)
        with torch.no_grad():
            lg, _ = net(x)
        a = lg.numpy().max(axis=0)
        d = {"rec": r_}
        for j, nm in enumerate(NAMES):
            d["scp_" + nm] = float(a[j])
        rows.append(d)
    except Exception as exc:
        nbad += 1
        if nbad <= 3:
            print("  FAIL", r_,
                  repr(exc)[:150])
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
print("extracted:", len(d), " failed:", nbad)
if not len(d):
    raise SystemExit("nothing extracted")
d.to_csv(REC_DEST, index=False)
print("saved", REC_DEST, d.shape)

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
print("SANITY  (want logits within about +-10)")
v = d[cols].values
print("  range: %.2f to %.2f"
      % (v.min(), v.max()))
print("  mean %.3f  sd %.3f"
      % (v.mean(), v.std()))
print("  constant statements:",
      int((v.std(axis=0) < 1e-6).sum()),
      "of", len(cols))
pr = 1.0 / (1.0 + np.exp(-v))
print("")
print("  highest mean probability:")
for k in np.argsort(-pr.mean(axis=0))[:8]:
    print("    %-10s %.3f"
          % (NAMES[k], pr[:, k].mean()))
if "AFIB" in NAMES:
    print("  AFIB mean probability: %.3f"
          % pr[:, NAMES.index("AFIB")].mean())