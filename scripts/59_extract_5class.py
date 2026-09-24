"""Extract features from the local 5-class PTB-XL
checkpoint.

Architecture read from the state dict:
  prep    3 convs, 12->32->32->64, kernel 3
  layer1  3 BasicBlocks, 64,  kernel 7
  layer2  4 BasicBlocks, 128, first stride 2
  layer3  23 BasicBlocks, 256, first stride 2
  layer4  3 BasicBlocks, 512, first stride 2
  fc      512 -> 5
434 tensors, which matches exactly.

Saves both the 5 superdiagnostic logits and the
512-dimensional penultimate embedding per record,
then aggregates to admission level.

This is a 5-class model, not the 71-class
exp0 model behind the stored p_ecg. Its training
preprocessing is not documented, so the
benchmark standardiser and windowing are applied
for consistency and the
result should be read as a learned
representation rather than a faithful refit.
Run in venv_ecg only.

OUTPUT FILES
  data\\processed\\ecg5_record.csv
  data\\processed\\ecg5.csv
  results\\extract5_log.txt
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
CKPT = os.path.join(
    DIS, "HPC", "Model Results", "ptbxl",
    "Full DL", "xresnet1d101_ptbxl_weights.pt")
OUT = os.path.join("data", "raw", "ecg")
PROC = os.path.join("data", "processed")
os.makedirs(PROC, exist_ok=True)
REC_DEST = os.path.join(
    PROC, "ecg5_record.csv")
DEST = os.path.join(PROC, "ecg5.csv")

MEAN, SCALE = -0.00077717, 0.23895642
WIN, STRIDE = 250, 125
FS_IN, FS_OUT = 500, 100
NEMB, NCLS = 512, 5


class BasicBlock(nn.Module):
    """conv1, bn1, relu, conv2, bn2, add,
    relu. Names match the checkpoint."""

    def __init__(self, ni, nf, stride=1,
                 ks=7):
        super().__init__()
        self.conv1 = nn.Conv1d(
            ni, nf, ks, stride=stride,
            padding=ks // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(nf)
        self.conv2 = nn.Conv1d(
            nf, nf, ks, stride=1,
            padding=ks // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(nf)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or ni != nf:
            self.downsample = nn.Sequential(
                nn.Conv1d(ni, nf, 1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm1d(nf))

    def forward(self, x):
        idt = x
        o = self.relu(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o))
        if self.downsample is not None:
            idt = self.downsample(x)
        if o.shape[-1] != idt.shape[-1]:
            n = min(o.shape[-1],
                    idt.shape[-1])
            o = o[..., :n]
            idt = idt[..., :n]
        return self.relu(o + idt)


class Net(nn.Module):
    def __init__(self, nin=12, ncls=NCLS,
                 layers=(3, 4, 23, 3)):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv1d(nin, 32, 3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        self.pool = nn.MaxPool1d(
            3, stride=2, padding=1)
        w = [64, 128, 256, 512]
        ni = 64
        for i, n in enumerate(layers):
            blocks = []
            for j in range(n):
                st = 2 if (i > 0 and j == 0) \
                    else 1
                blocks.append(
                    BasicBlock(ni, w[i], st))
                ni = w[i]
            setattr(self, "layer%d" % (i + 1),
                    nn.Sequential(*blocks))
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(NEMB, ncls)

    def features(self, x):
        x = self.pool(self.prep(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avg(x).flatten(1)

    def forward(self, x):
        e = self.features(x)
        return self.fc(e), e


print("loading checkpoint ...", flush=True)
obj = torch.load(CKPT, map_location="cpu",
                 weights_only=False)
sd = obj
if isinstance(obj, dict):
    for k in ("state_dict", "model",
              "model_state_dict"):
        if k in obj and isinstance(obj[k],
                                   dict):
            sd = obj[k]
            break
sd = {k: v for k, v in sd.items()
      if hasattr(v, "shape")}
print("checkpoint tensors:", len(sd))

net = Net()
own = net.state_dict()
print("model tensors:", len(own))

missing = [k for k in own if k not in sd]
extra = [k for k in sd if k not in own]
print("in model not ckpt:", len(missing))
for k in missing[:8]:
    print("   ", k, tuple(own[k].shape))
print("in ckpt not model:", len(extra))
for k in extra[:8]:
    print("   ", k, tuple(sd[k].shape))

bad = [k for k in own if k in sd
       and tuple(own[k].shape)
       != tuple(sd[k].shape)]
print("shape mismatches:", len(bad))
for k in bad[:8]:
    print("    %-46s model %s ckpt %s"
          % (k, tuple(own[k].shape),
             tuple(sd[k].shape)))

if missing or extra or bad:
    raise SystemExit(
        "architecture does not match the"
        " checkpoint; see the lists above")

net.load_state_dict(sd, strict=True)
net.eval()
print("STRICT LOAD SUCCEEDED", flush=True)


def preprocess(sig):
    s = sig.copy()
    s[:, [4, 5]] = s[:, [5, 4]]
    s = resample_poly(s, FS_OUT, FS_IN, axis=0)
    return ((s - MEAN) / SCALE).astype(
        np.float32)


def windows(s):
    n = s.shape[0]
    out, i = [], 0
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
print("records:", len(recs), flush=True)

torch.set_num_threads(
    max(1, os.cpu_count() - 1))
rows, bad2 = [], 0
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
            lg, em = net(x)
        d = {"rec": r_}
        a = lg.numpy().max(axis=0)
        for j in range(NCLS):
            d["cls_%d" % j] = float(a[j])
        b = em.numpy().max(axis=0)
        for j in range(NEMB):
            d["e_%03d" % j] = float(b[j])
        rows.append(d)
    except Exception as exc:
        bad2 += 1
        if bad2 <= 3:
            print("  FAIL", r_,
                  repr(exc)[:140])
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
d.to_csv(REC_DEST, index=False)
print("saved", REC_DEST, d.shape)

m = idx[["subject_id", "hadm_id",
         "rec"]].drop_duplicates()
j = m.merge(d, on="rec", how="inner")
cols = [c for c in j.columns
        if c.startswith(("cls_", "e_"))]
g = j.groupby(["subject_id", "hadm_id"])
agg = g[cols].agg(["mean", "max"])
agg.columns = ["%s_%s" % (a, b)
               for a, b in agg.columns]
agg = agg.reset_index()
agg["n_ecg"] = g.size().values
agg.to_csv(DEST, index=False)
print("saved", DEST, agg.shape)

print("")
print("SANITY")
cl = [c for c in d.columns
      if c.startswith("cls_")]
v = d[cl].values
print("  logit range: %.2f to %.2f"
      % (v.min(), v.max()))
print("  per-class sd:",
      np.round(v.std(axis=0), 3))
ec = [c for c in d.columns
      if c.startswith("e_")]
e = d[ec].values
print("  embedding range: %.2f to %.2f"
      % (e.min(), e.max()))
print("  dead embedding dims:",
      int((e.std(axis=0) < 1e-6).sum()),
      "of", len(ec))