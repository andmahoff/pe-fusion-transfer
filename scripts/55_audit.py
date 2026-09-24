"""Reads every saved result file and prints the
key numbers from each.
Run in venv (analysis).
"""
import os
import sys
import pandas as pd

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC = f.PROC
pd.set_option("display.width", 200)


def load(name):
    p = os.path.join(PROC, name)
    if not os.path.exists(p):
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def head(t):
    print("")
    print("=" * 64)
    print(t)
    print("=" * 64)


head("FILES PRESENT")
for fn in sorted(os.listdir(PROC)):
    if fn.endswith(".csv"):
        p = os.path.join(PROC, fn)
        try:
            n = len(pd.read_csv(p))
        except Exception:
            n = -1
        print("  %-34s %6d rows  %6.0f KB"
              % (fn, n,
                 os.path.getsize(p) / 1024))

head("1. UNIMODAL BASELINES (lr)")
d = load("unimodal.csv")
if d is not None:
    q = d[d["learner"] == "lr"]
    print(q.pivot_table(
        index=["direction", "outcome"],
        columns="modality",
        values="auc").round(4).to_string())

head("2. FUSION GRID, tuned C")
d = load("grid_tuned_std.csv")
if d is not None:
    print("MEAN GAIN OVER EHR ALONE")
    g = d.groupby("arch").agg(
        mean_gain=("gain", "mean"),
        sig_wins=("sig", lambda s: int(
            ((d.loc[s.index, "gain"] > 0)
             & (s == 1)).sum())),
        sig_loss=("sig", lambda s: int(
            ((d.loc[s.index, "gain"] < 0)
             & (s == 1)).sum())))
    print(g.round(4).sort_values(
        "mean_gain", ascending=False)
        .to_string())
    print("")
    print("PRIMARY CONTRAST: I2M death_30d")
    p = d[(d["direction"] == "I2M")
          & (d["outcome"] == "death_30d")]
    print(p[["arch", "auc", "gain", "lo",
             "hi", "sig"]].round(4)
          .to_string(index=False))

head("3. GAP RELATIONSHIP")
d = load("gap_table_tuned.csv")
if d is not None:
    print(d[["direction", "outcome", "ehr",
             "ctpa", "gap", "late_best",
             "early_best", "inter_best"]]
          .round(4).to_string(index=False))
d = load("gap_regression_tuned.csv")
if d is not None:
    print("")
    print(d[["set", "slope", "se", "r2",
             "p", "n"]].round(4)
          .to_string(index=False))

head("4. LADDER (controlled gap test)")
d = load("ladder.csv")
if d is not None:
    print(d[["direction", "outcome", "rung",
             "ehr", "ctpa", "gap", "gain",
             "lo", "hi"]].round(4)
          .to_string(index=False))

head("5. MODERATORS")
for fn in ("moderators.csv",
           "moderators2.csv"):
    d = load(fn)
    if d is None:
        continue
    print("")
    print(fn, d.shape)
    print("  columns:", list(d.columns)[:14])

head("6. INTERMEDIATE FUSION: dropout")
for fn in ("dropout.csv", "dropout_ext.csv",
           "mechanisms_hd.csv",
           "encoders.csv"):
    d = load(fn)
    if d is None:
        continue
    print("")
    print(fn)
    key = ("config" if "config" in d.columns
           else "variant")
    if key in d.columns:
        cols = [c for c in
                ["auc", "gain", "vs_late",
                 "overfit", "sig"]
                if c in d.columns]
        print(d.groupby(key)[cols].mean()
              .round(4).sort_values(
                  cols[0], ascending=False)
              .to_string())

head("7. THREE-MODALITY MODELS")
for fn in ("three_modality.csv",
           "dissertation_compare.csv",
           "diss_exact.csv"):
    d = load(fn)
    if d is None:
        continue
    print("")
    print(fn)
    cols = [c for c in
            ["outcome", "model", "protocol",
             "n", "ev", "auc", "gain", "lo",
             "hi", "weights"]
            if c in d.columns]
    x = d[cols]
    if "outcome" in x.columns:
        x = x[x["outcome"].isin(
            ["death_30d"])]
    print(x.sort_values(
        "auc", ascending=False).round(4)
        .to_string(index=False))

head("8. ECG WORK")
for fn in ("ecg_fit.csv",
           "ecg_trajectory_fit.csv",
           "ecg_traj_equal.csv",
           "ecg_traj_fusion.csv",
           "ecg_hybrid.csv",
           "ecg_hybrid_fusion.csv",
           "ecg_stratified.csv",
           "ecg_scalefix.csv",
           "ecg_head.csv"):
    d = load(fn)
    if d is None:
        print("")
        print(fn, "- not present")
        continue
    print("")
    print("-" * 64)
    print(fn, d.shape)
    cols = [c for c in
            ["outcome", "model", "ecg",
             "version", "n", "ev", "auc",
             "auc_2mod", "auc_3mod", "gain",
             "vs_stored", "lo", "hi", "sig"]
            if c in d.columns]
    print(d[cols].round(4)
          .to_string(index=False))

head("9. THREE-MODALITY GAIN, FULL COHORT")
print("Does a third modality beat ctpa+ehr on")
print("the FULL cohort? Every gain vs the")
print("two-modality model, with its interval.")
for fn in ("ecg_hybrid_fusion.csv",
           "ecg_stratified.csv",
           "ecg_scalefix.csv"):
    d = load(fn)
    if d is None:
        continue
    g = ("gain" if "gain" in d.columns
         else None)
    if g is None:
        continue
    print("")
    print(fn)
    x = d.copy()
    lab = ("ecg" if "ecg" in x.columns
           else "model")
    x = x[~x[lab].astype(str)
          .str.contains("_vs_")]
    print(x[[c for c in
             ["outcome", lab, "n", "ev",
              g, "lo", "hi", "sig"]
             if c in x.columns]]
          .round(4).to_string(index=False))
    ns = int(x["sig"].sum()) \
        if "sig" in x.columns else 0
    print("  significant cells: %d of %d"
          % (ns, len(x)))