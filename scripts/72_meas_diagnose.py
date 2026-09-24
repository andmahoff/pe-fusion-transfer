"""Tests whether the derived waveform measurements
are noise or under-regularised.

Trajectory scored 0.5604 standalone and failed a
permutation test, so it was noise. The
measurements score 0.6822 standalone, which is
well above chance, so the diagnosis is likely
different.

Applies the same four diagnostics plus an
extended C sweep:
  1 STANDALONE   measurements alone, swept over
                 C down to 1e-6
  2 TRAIN vs OOF the overfitting signature
  3 PERMUTATION  shuffled against real
  4 COMBINED     logits plus measurements over
                 the full C grid, with and
                 without block scaling, so the
                 130-column block cannot
                 dominate the 71 logits by count

Measurements are rebuilt from the record-level
file on the same windowed cohort as the logits,
so the comparison is like for like.
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\meas_diagnose.csv
  results\\meas_diagnose_log.txt
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
RAW = os.path.join("data", "raw", "ecg")
SEEDS = [42, 7, 13, 1, 2]
NFOLD = 5
WIN_LO, WIN_HI = -12.0, 48.0
CGRID = [1e-6, 1e-5, 1e-4, 3e-4, 1e-3,
         3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
OUTS = ["cv_first", "death_30d_inhosp",
        "composite_30d", "death_30d"]
DEST = os.path.join(PROC,
                    "meas_diagnose.csv")
DROP = ("rec", "t_flag", "vm_noise",
        "vm_base", "vm_rpeak", "net_i",
        "net_avf")


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def run(X, y, grp, seed, C, blocks=None,
        scale=False, want_train=False):
    p = np.zeros(len(y))
    tr = []
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for a_, b_ in cv.split(X, y, grp):
        im = SimpleImputer(strategy="median")
        A = im.fit_transform(X[a_])
        B = im.transform(X[b_])
        sc = StandardScaler()
        A, B = sc.fit_transform(A), sc.transform(B)
        if scale and blocks:
            i = 0
            for n in blocks:
                k = np.sqrt(n)
                A[:, i:i + n] /= k
                B[:, i:i + n] /= k
                i += n
        m = LogisticRegression(
            C=C, max_iter=8000,
            class_weight="balanced")
        m.fit(A, y[a_])
        p[b_] = m.predict_proba(B)[:, 1]
        if want_train:
            tr.append(roc_auc_score(
                y[a_],
                m.predict_proba(A)[:, 1]))
    return (roc_auc_score(y, p),
            float(np.mean(tr)) if tr
            else np.nan)


def sweep(X, y, grp, blocks=None,
          scale=False, want_train=False):
    best, curve = None, []
    for C in CGRID:
        oo, tt = [], []
        for s in SEEDS:
            o, t = run(X, y, grp, s, C,
                       blocks, scale,
                       want_train)
            oo.append(o)
            tt.append(t)
        m = float(np.mean(oo))
        sd = float(np.std(oo, ddof=1))
        t = (float(np.nanmean(tt))
             if want_train else np.nan)
        curve.append((C, m))
        if best is None or m > best[1]:
            best = (C, m, sd, t)
    return best, curve


# ---------- windowed logits ----------
rec = pd.read_csv(os.path.join(
    PROC, "exp0_logits_record.csv"))
idx = pd.read_csv(os.path.join(
    RAW, "ecg_record_index.csv"))
idx["rec"] = idx["rec"].astype(str).apply(
    lambda s: s if s.startswith("files/")
    else "files/" + s.replace("\\", "/")
    .strip("/"))
LG = [c for c in rec.columns
      if c.startswith("scp_")]
adm = pd.read_csv(os.path.join(
    FIG, "index_admission_times.csv"))
adm["admittime"] = pd.to_datetime(
    adm["admittime"], errors="coerce")

R = idx.merge(rec, on="rec", how="inner")
R["t"] = pd.to_datetime(
    R["ecg_charttime"], errors="coerce")
R = R.merge(adm[["hadm_id", "admittime"]],
            on="hadm_id", how="left")
R["h"] = ((R["t"] - R["admittime"])
          .dt.total_seconds() / 3600.0)
R = R[(R["h"] >= WIN_LO)
      & (R["h"] <= WIN_HI)].copy()
A = R.groupby(["subject_id", "hadm_id"],
              as_index=False)[LG].mean()
keep_rec = set(R["rec"])
print("windowed admissions:", len(A))

# ---------- measurements, same window ----------
mv = pd.read_csv(os.path.join(
    PROC, "ecg_derived_v3_record.csv"))
mv["rec"] = mv["rec"].astype(str).apply(
    lambda s: s if s.startswith("files/")
    else "files/" + s.replace("\\", "/")
    .strip("/"))
mv = mv[mv["rec"].isin(keep_rec)]
MC = [c for c in mv.columns
      if c not in DROP
      and pd.api.types.is_numeric_dtype(
          mv[c])]
MV = idx[["subject_id", "hadm_id",
          "rec"]].merge(
    mv[["rec"] + MC], on="rec", how="inner")
MA = MV.groupby(["subject_id", "hadm_id"],
                as_index=False)[MC].agg(
    ["mean", "max"])
MA.columns = ["%s_%s" % (a, b) if b else a
              for a, b in MA.columns]
MA = MA.reset_index()
MEAS = [c for c in MA.columns
        if c not in ("subject_id",
                     "hadm_id")]
cov = MA[MEAS].notna().mean()
MEAS = [c for c in MEAS if cov[c] >= 0.50]
print("measurement columns:", len(MEAS),
      " logits:", len(LG), flush=True)

lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
D = A.merge(lab, on=["subject_id",
                     "hadm_id"],
            how="inner").merge(
    MA[["hadm_id"] + MEAS], on="hadm_id",
    how="left")
D = D[D[MEAS].notna().sum(axis=1) >= 5]
print("cohort:", len(D), flush=True)

res = []
t0 = time.time()

for oc in OUTS:
    d = D
    if oc == "cv_first":
        d = D[D["death_first"] == 0]
    y = pd.to_numeric(
        d[oc], errors="coerce").fillna(
        0).astype(int).values
    grp = d["subject_id"].values
    XL = d[LG].values.astype(float)
    XM = d[MEAS].values.astype(float)
    XB = np.column_stack([XL, XM])

    print("")
    print("=" * 76)
    print("%s   n=%d ev=%d"
          % (oc, len(y), int(y.sum())),
          flush=True)

    print("")
    print("  1. STANDALONE MEASUREMENTS")
    (c, m, s, t), cv1 = sweep(
        XM, y, grp, want_train=True)
    edge = ("  (at the grid edge)"
            if c in (CGRID[0], CGRID[-1])
            else "")
    print("     peak C=%.0e  OOF %.4f"
          " (SD %.4f)  train %.4f"
          "  gap %+.4f%s"
          % (c, m, s, t, t - m, edge))
    print("     curve: " + " ".join(
        "%.4f" % x[1] for x in cv1))
    res.append({"outcome": oc,
                "test": "standalone_meas",
                "auc": m, "sd": s,
                "train": t, "gap": t - m,
                "C": c,
                "nfeat": XM.shape[1]})

    print("")
    print("  2. LOGITS ALONE, for reference")
    (cl, ml, sl, tl), _ = sweep(
        XL, y, grp, want_train=True)
    print("     peak C=%.0e  OOF %.4f"
          "  train %.4f  gap %+.4f"
          % (cl, ml, tl, tl - ml))
    res.append({"outcome": oc,
                "test": "logits",
                "auc": ml, "sd": sl,
                "train": tl, "gap": tl - ml,
                "C": cl, "nfeat": len(LG)})

    print("")
    print("  3. COMBINED, extended C sweep")
    (cb, mb, sb, tb), cv3 = sweep(
        XB, y, grp, want_train=True)
    print("     unscaled  peak C=%.0e"
          "  OOF %.4f  train %.4f"
          "  gap %+.4f"
          % (cb, mb, tb, tb - mb))
    print("     curve: " + " ".join(
        "%.4f" % x[1] for x in cv3))
    (cs_, ms, ss, ts), _ = sweep(
        XB, y, grp,
        blocks=[len(LG), len(MEAS)],
        scale=True, want_train=True)
    print("     scaled    peak C=%.0e"
          "  OOF %.4f" % (cs_, ms))
    best_comb = max(mb, ms)
    print("     measurements contribute"
          " %+.4f" % (best_comb - ml))
    for nm, cc, mm2, ss2, tt2 in (
            ("combined", cb, mb, sb, tb),
            ("combined_scaled", cs_, ms,
             ss, ts)):
        res.append({"outcome": oc,
                    "test": nm, "auc": mm2,
                    "sd": ss2, "train": tt2,
                    "gap": tt2 - mm2,
                    "C": cc,
                    "nfeat": XB.shape[1]})

    print("")
    print("  4. PERMUTATION, 3 draws")
    perm = []
    for k in range(3):
        rng = np.random.default_rng(200 + k)
        XP = np.column_stack(
            [XL, XM[rng.permutation(len(XM))]])
        (_, mp, _, _), _ = sweep(XP, y, grp)
        perm.append(mp)
    pm = float(np.mean(perm))
    print("     shuffled %.4f   real %.4f"
          "   difference %+.4f"
          % (pm, mb, mb - pm))
    print("     ->", "real beats shuffled"
          if mb - pm > 0.005
          else "no better than shuffled")
    res.append({"outcome": oc,
                "test": "permutation",
                "auc": pm, "sd": np.nan,
                "train": np.nan,
                "gap": mb - pm, "C": np.nan,
                "nfeat": XB.shape[1]})

r = pd.DataFrame(res)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 76)
print("PEAK AUROC BY TEST")
q = r[r["test"] != "permutation"]
print(q.pivot_table(index="test",
                    columns="outcome",
                    values="auc")
      .round(4).to_string())
print("")
print("C AT THE PEAK")
print(q.pivot_table(index="test",
                    columns="outcome",
                    values="C")
      .to_string())
print("")
print("OVERFITTING GAP (train minus OOF)")
print(q.pivot_table(index="test",
                    columns="outcome",
                    values="gap")
      .round(4).to_string())
print("")
print("REAL MINUS SHUFFLED")
p = r[r["test"] == "permutation"]
print(p[["outcome", "auc", "gap"]]
      .rename(columns={
          "auc": "shuffled",
          "gap": "real_minus_shuf"})
      .round(4).to_string(index=False))
print("")
print("DO THE MEASUREMENTS CONTRIBUTE?")
for oc in OUTS:
    s = r[r["outcome"] == oc]
    if not len(s):
        continue
    ml = float(s[s["test"] == "logits"]
               ["auc"].iloc[0])
    mb = s[s["test"].str.startswith(
        "combined")]["auc"].max()
    st = float(s[s["test"]
                 == "standalone_meas"]
               ["auc"].iloc[0])
    print("  %-18s standalone %.4f"
          "   logits %.4f   combined %.4f"
          "   delta %+.4f"
          % (oc, st, ml, mb, mb - ml))
print("")
print("VERDICT")
sa = r[r["test"] == "standalone_meas"][
    "auc"].mean()
pv = r[r["test"] == "permutation"][
    "gap"].mean()
print("  mean standalone AUROC: %.4f"
      "  (trajectory was 0.5604)" % sa)
print("  mean real minus shuffled: %+.4f"
      "  (trajectory was +0.0003)" % pv)
print("")
print("saved", DEST, r.shape)