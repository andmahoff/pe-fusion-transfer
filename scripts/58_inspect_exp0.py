"""Read the exp0 checkpoint, its standard scaler
and its label binariser. Confirms the 71-class
head and recovers the SCP statement names.
Run in venv_ecg only.

OUTPUT FILES
  results\\exp0_inspect_log.txt
  data\\processed\\scp_labels.csv
"""
import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
from collections import Counter, OrderedDict

EXP = os.path.join(
    os.path.expanduser("~"), "ecg_ptbxl_benchmarking", "output", "exp0")
CKPT = os.path.join(
    EXP, "models", "fastai_xresnet1d101",
    "models", "fastai_xresnet1d101.pth")
SCALER = os.path.join(
    EXP, "data", "standard_scaler.pkl")
MLB = os.path.join(EXP, "data", "mlb.pkl")
PROC = os.path.join("data", "processed")
os.makedirs(PROC, exist_ok=True)

print("CHECKPOINT")
print(" ", CKPT)
print("  exists:", os.path.exists(CKPT))
if not os.path.exists(CKPT):
    raise SystemExit("not found")
print("  %.1f MB"
      % (os.path.getsize(CKPT) / 1e6))

obj = torch.load(CKPT, map_location="cpu",
                 weights_only=False)
sd = obj
if isinstance(obj, dict):
    print("  top-level keys:",
          list(obj.keys())[:8])
    for k in ("model", "state_dict",
              "model_state_dict"):
        if k in obj and isinstance(
                obj[k], (dict, OrderedDict)):
            sd = obj[k]
            print("  nested under:", k)
            break
sd = {k: v for k, v in sd.items()
      if hasattr(v, "shape")}

print("")
print("  tensor entries:", len(sd),
      " (expected 446)")
print("  parameters: %.2f M"
      % (sum(int(v.numel())
             for v in sd.values()) / 1e6))
print("  dtypes:",
      dict(Counter(str(v.dtype)
                   for v in sd.values())))

print("")
print("FIRST 12 KEYS")
for k in list(sd)[:12]:
    print("  %-52s %s %s"
          % (k, tuple(sd[k].shape),
             sd[k].dtype))
print("")
print("LAST 12 KEYS")
for k in list(sd)[-12:]:
    print("  %-52s %s %s"
          % (k, tuple(sd[k].shape),
             sd[k].dtype))

print("")
print("TOP-LEVEL PREFIXES")
pre = {}
for k in sd:
    p = k.split(".")[0]
    pre[p] = pre.get(p, 0) + 1
for p, n in sorted(pre.items(),
                   key=lambda x: -x[1])[:14]:
    print("  %-18s %d" % (p, n))

print("")
print("BLOCK STRUCTURE")
seen = {}
for k in sd:
    m = re.match(r"^([\w]+)\.(\d+)\.", k)
    if m:
        seen.setdefault(
            m.group(1), set()).add(
            int(m.group(2)))
for a in sorted(seen):
    ix = sorted(seen[a])
    print("  %-14s %d entries  %s"
          % (a, len(ix), ix[:10]))

print("")
print("SECOND-LEVEL under the largest prefix")
big = max(pre, key=pre.get)
lvl = {}
for k in sd:
    if k.startswith(big + "."):
        parts = k.split(".")
        if len(parts) > 2:
            lvl.setdefault(
                parts[1], set()).add(parts[2])
for a in sorted(lvl, key=lambda x: (
        len(x), x))[:12]:
    print("  %s.%-4s -> %s"
          % (big, a, sorted(lvl[a])[:6]))

print("")
print("LINEAR LAYERS  (want a 71-output head)")
for k, v in sd.items():
    if len(v.shape) == 2:
        print("  %-52s %d -> %d"
              % (k, v.shape[1], v.shape[0]))

print("")
print("CONV LAYERS (first 20)")
n = 0
tot = sum(1 for v in sd.values()
          if len(v.shape) == 3)
for k, v in sd.items():
    if len(v.shape) == 3:
        print("  %-50s in %4d out %4d k %d"
              % (k, v.shape[1], v.shape[0],
                 v.shape[2]))
        n += 1
        if n >= 20:
            print("  ... %d conv layers total"
                  % tot)
            break

print("")
print("KERNEL SIZES")
print("  ", dict(Counter(
    v.shape[2] for v in sd.values()
    if len(v.shape) == 3)))

print("")
print("=" * 64)
print("STANDARD SCALER")
print("=" * 64)
try:
    with open(SCALER, "rb") as fh:
        sc = pickle.load(fh)
    print("  type:", type(sc).__name__)
    for at in ("mean_", "scale_", "var_",
               "n_features_in_",
               "n_samples_seen_"):
        if hasattr(sc, at):
            print("  %-18s %s"
                  % (at, getattr(sc, at)))
    print("")
    print("  expected mean -0.00077717,"
          " scale 0.23895642")
except Exception as exc:
    print("  FAILED:", repr(exc)[:220])
    print("  a scikit-learn version mismatch is"
          " the usual cause; the two constants"
          " are already known so this is not"
          " fatal")

print("")
print("=" * 64)
print("LABEL BINARISER (SCP statement names)")
print("=" * 64)
try:
    with open(MLB, "rb") as fh:
        mlb = pickle.load(fh)
    cls = list(getattr(mlb, "classes_", []))
    print("  classes:", len(cls), "(want 71)")
    print("  first 24:", cls[:24])
    pd.DataFrame({"idx": range(len(cls)),
                  "scp": cls}).to_csv(
        os.path.join(PROC, "scp_labels.csv"),
        index=False)
    print("  saved data/processed/"
          "scp_labels.csv")
except Exception as exc:
    print("  FAILED:", repr(exc)[:220])