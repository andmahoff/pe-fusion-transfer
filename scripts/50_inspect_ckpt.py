"""Read both PTB-XL checkpoints and describe
their architecture from the state dict.
Run in venv_ecg only.
"""
import os
import re
import torch
from collections import Counter, OrderedDict

# folder holding the dissertation's derived files
DIS = os.environ.get(
    "PE_DISS_DIR", os.path.join("..", "Dissertation"))
BASE = os.path.join(DIS, "HPC", "Model Results", "ptbxl")
CANDS = [
    os.path.join(
        BASE, "Full DL fastai attempt",
        "xresnet1d101_fastai_enhanced"
        "_weights.pt"),
    os.path.join(
        BASE, "Full DL",
        "xresnet1d101_ptbxl_weights.pt"),
]

for fp in CANDS:
    if not os.path.exists(fp):
        print("MISSING:", fp)
        continue
    print("")
    print("=" * 66)
    print(os.path.basename(fp))
    print("  %.1f MB" % (os.path.getsize(fp)
                         / 1e6))
    print("=" * 66)

    obj = torch.load(fp, map_location="cpu",
                     weights_only=False)
    sd = obj
    if isinstance(obj, dict):
        print("top-level keys:",
              list(obj.keys())[:6])
        for k in ("state_dict", "model",
                  "model_state_dict"):
            if k in obj and isinstance(
                    obj[k], (dict, OrderedDict)):
                sd = obj[k]
                print("  nested under:", k)
                break
    sd = {k: v for k, v in sd.items()
          if hasattr(v, "shape")}
    print("tensor entries:", len(sd))
    print("parameters: %.2f M"
          % (sum(int(v.numel())
                 for v in sd.values()) / 1e6))

    print("")
    print("FIRST 10 KEYS")
    for k in list(sd)[:10]:
        print("  %-54s %s"
              % (k, tuple(sd[k].shape)))
    print("")
    print("LAST 10 KEYS")
    for k in list(sd)[-10:]:
        print("  %-54s %s"
              % (k, tuple(sd[k].shape)))

    print("")
    print("TOP-LEVEL PREFIXES")
    pre = {}
    for k in sd:
        p = k.split(".")[0]
        pre[p] = pre.get(p, 0) + 1
    for p, n in sorted(pre.items(),
                       key=lambda x: -x[1])[:12]:
        print("  %-16s %d" % (p, n))

    print("")
    print("BLOCK STRUCTURE")
    seen = {}
    for k in sd:
        m = re.match(r"^(\w+)\.(\d+)\.", k)
        if m:
            seen.setdefault(
                m.group(1), set()).add(
                int(m.group(2)))
    for a in sorted(seen):
        ix = sorted(seen[a])
        print("  %-12s %d entries  %s"
              % (a, len(ix), ix[:8]))

    print("")
    print("CONV LAYERS (first 24)")
    n = 0
    tot = sum(1 for v in sd.values()
              if len(v.shape) == 3)
    for k, v in sd.items():
        if len(v.shape) == 3:
            print("  %-52s in %4d out %4d k %d"
                  % (k, v.shape[1], v.shape[0],
                     v.shape[2]))
            n += 1
            if n >= 24:
                print("  ... %d conv layers"
                      " total" % tot)
                break

    print("")
    print("LINEAR LAYERS")
    for k, v in sd.items():
        if len(v.shape) == 2:
            print("  %-52s %d -> %d"
                  % (k, v.shape[1], v.shape[0]))

    print("")
    print("KERNEL SIZES")
    ks = [v.shape[2] for v in sd.values()
          if len(v.shape) == 3]
    print("  ", dict(Counter(ks)))
    print("  mostly 3s with some 1s means"
          " bottleneck blocks;")
    print("  almost all 3s means BasicBlocks")