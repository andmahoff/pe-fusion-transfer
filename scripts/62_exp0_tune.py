"""The four experiments on the 71 SCP logits.

  0 validation: refit the original head on the
    freshly extracted logits. If it lands near
    the stored 0.7367 on death_30d, the whole
    extraction pipeline is validated end to end.
  1 C tuning
  2 class weighting (the original head used
    class_weight="balanced")
  3 gradient boosting over named statements
  4 trajectory on the logits: change in statements
    such as AFIB or ASMI between serial recordings

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\exp0_tune.csv
  data\\processed\\exp0_trajectory.csv
  data\\processed\\exp0_coefs.csv
  results\\exp0_tune_log.txt
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
RAW = os.path.join("data", "raw", "ecg")
SEED, NFOLD = 42, 5
CS = [0.0001, 0.001, 0.003, 0.01, 0.03,
      0.1, 0.3, 1.0, 3.0]
CLIP = 40.0
OUTS = ["death_30d", "death_30d_inhosp",
        "composite_30d"]
DEST = os.path.join(PROC, "exp0_tune.csv")
# statements worth tracking over time
TRACK = ["AFIB", "AFLT", "STACH", "SBRAD",
         "ASMI", "IMI", "ALMI", "NDT", "NST_",
         "STD_", "STE_", "ABQRS", "NORM",
         "SR", "PVC", "LVH", "RVH", "CRBBB",
         "CLBBB", "1AVB"]


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def fold_lr(Xa, ya, ga, Xb, cw=None,
            fixed_c=None):
    im = SimpleImputer(strategy="median")
    a = im.fit_transform(Xa)
    b = im.transform(Xb)
    sc = StandardScaler()
    a, b = sc.fit_transform(a), sc.transform(b)
    if fixed_c is not None:
        bc = fixed_c
    else:
        best, bc = -1.0, 0.01
        k = min(3, max(2, int(ya.sum()) // 8))
        try:
            icv = StratifiedGroupKFold(
                n_splits=k, shuffle=True,
                random_state=SEED)
            for c in CS:
                q = np.zeros(len(ya))
                for t2, v2 in icv.split(
                        a, ya, ga):
                    m = LogisticRegression(
                        C=c, max_iter=6000,
                        class_weight=cw)
                    m.fit(a[t2], ya[t2])
                    q[v2] = m.predict_proba(
                        a[v2])[:, 1]
                s = roc_auc_score(ya, q)
                if s > best:
                    best, bc = s, c
        except Exception:
            pass
    m = LogisticRegression(
        C=bc, max_iter=6000, class_weight=cw)
    m.fit(a, ya)
    return m.predict_proba(b)[:, 1], bc


def fold_gb(Xa, ya, Xb, **kw):
    p = dict(random_state=SEED, max_depth=3,
             learning_rate=0.05, max_iter=300,
             l2_regularization=1.0,
             early_stopping=True,
             validation_fraction=0.15)
    p.update(kw)
    m = HistGradientBoostingClassifier(**p)
    m.fit(Xa, ya)
    return m.predict_proba(Xb)[:, 1]


def oof(X, y, grp, folds, kind="lr", **kw):
    p = np.zeros(len(y))
    cs = []
    for tr, te in folds:
        if kind == "gb":
            p[te] = fold_gb(X[tr], y[tr],
                            X[te], **kw)
        else:
            p[te], c = fold_lr(
                X[tr], y[tr], grp[tr], X[te],
                **kw)
            cs.append(c)
    return p, cs


def build_traj(names):
    """Trajectory on the statement logits."""
    keep = [n for n in TRACK if n in names]
    cols = ["scp_" + n for n in keep]
    rec = pd.read_csv(
        os.path.join(
            PROC, "exp0_logits_record.csv"),
        usecols=["rec"] + cols)
    idx = pd.read_csv(os.path.join(
        RAW, "ecg_record_index.csv"))
    idx["rec"] = idx["rec"].astype(str).apply(
        lambda s: s if s.startswith("files/")
        else "files/" + s.replace("\\", "/")
        .strip("/"))
    j = idx[["hadm_id", "rec",
             "ecg_charttime"]].merge(
        rec, on="rec", how="inner")
    j["t"] = pd.to_datetime(
        j["ecg_charttime"], errors="coerce")
    j = j.dropna(subset=["t"]).sort_values(
        ["hadm_id", "t"])
    for c in cols:
        j[c] = np.clip(
            pd.to_numeric(j[c],
                          errors="coerce"),
            -CLIP, CLIP)
    rows = []
    for h, g in j.groupby("hadm_id"):
        if len(g) < 2:
            continue
        d = {"hadm_id": h, "n_rec": len(g)}
        hrs = (g["t"].iloc[-1]
               - g["t"].iloc[0]).total_seconds()
        hrs /= 3600.0
        d["span_h"] = hrs
        for c in cols:
            v = g[c].values
            if np.isnan(v).all():
                continue
            d[c + "_first"] = float(v[0])
            d[c + "_last"] = float(v[-1])
            d[c + "_delta"] = float(
                v[-1] - v[0])
            d[c + "_sd"] = float(np.nanstd(v))
            d[c + "_rng"] = float(
                np.nanmax(v) - np.nanmin(v))
            if hrs > 0.5:
                d[c + "_slope"] = float(
                    (v[-1] - v[0]) / hrs)
        rows.append(d)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(
        PROC, "exp0_trajectory.csv"),
        index=False)
    print("logit trajectory:", t.shape,
          " statements tracked:", len(keep),
          flush=True)
    return t


e0 = pd.read_csv(os.path.join(
    PROC, "exp0_logits.csv"))
lab = pd.read_csv(os.path.join(
    PROC, "scp_labels.csv"))
NAMES = list(lab["scp"])
print("exp0 logits:", e0.shape, flush=True)

LG = [c for c in e0.columns
      if c.startswith("scp_")]
for c in LG:
    e0[c] = np.clip(e0[c], -CLIP, CLIP)
print("logit cols:", len(LG),
      "clipped at +-%.0f" % CLIP)

traj = build_traj(NAMES)
TR = [c for c in traj.columns
      if c != "hadm_id"]

mim = f.load_mimic()
ins = f.load_inspect()
rows, coefs = [], []

for oc in OUTS:
    if oc not in mim.columns:
        continue
    d = mim
    y = pd.to_numeric(d[oc], errors="coerce")
    y = y.fillna(0).astype(int).values
    grp = d["subject_id"].values

    fp = os.path.join(
        FIG, "p_ecg_harm_%s.csv" % oc)
    if not os.path.exists(fp):
        continue
    stq = pd.read_csv(fp)[
        ["hadm_id", "p_ecg"]].drop_duplicates(
        "hadm_id")

    a = d[["hadm_id"]].merge(
        e0[["hadm_id"] + LG],
        on="hadm_id", how="left")
    a = a.merge(traj, on="hadm_id", how="left")
    a = a.merge(stq, on="hadm_id", how="left")

    ok = a[LG].notna().all(axis=1).values
    ok &= np.isfinite(a["p_ecg"]).values
    yy, gg = y[ok], grp[ok]
    if yy.sum() < 30:
        continue

    Xl = a.loc[ok, LG].values.astype(float)
    Xt = a.loc[ok, TR].values.astype(float)
    ps = a.loc[ok, "p_ecg"].values
    multi = (a.loc[ok, TR].notna()
             .sum(axis=1) >= 10).values

    print("")
    print("=" * 72)
    print("%s  n=%d ev=%d  multi=%d"
          % (oc, int(ok.sum()), int(yy.sum()),
             int(multi.sum())), flush=True)

    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    folds = list(cv.split(Xl, yy, gg))

    R = {"stored": (_rank(ps), None)}
    JOBS = [
        ("orig_bal_C1", Xl, "lr",
         {"cw": "balanced", "fixed_c": 1.0}),
        ("logit_C1", Xl, "lr",
         {"fixed_c": 1.0}),
        ("logit_Ctuned", Xl, "lr", {}),
        ("logit_bal_tuned", Xl, "lr",
         {"cw": "balanced"}),
        ("logit_gb", Xl, "gb", {}),
        ("logit_gb_deep", Xl, "gb",
         {"max_depth": 5,
          "max_leaf_nodes": 31}),
        ("traj_only", Xt, "lr", {}),
        ("logit_traj_lr",
         np.column_stack([Xl, Xt]), "lr", {}),
        ("logit_traj_gb",
         np.column_stack([Xl, Xt]), "gb", {}),
    ]
    for nm, X, kind, kw in JOBS:
        p, cs = oof(X, yy, gg, folds,
                    kind=kind, **kw)
        R[nm] = (_rank(p), cs)

    bn0 = max([j[0] for j in JOBS],
              key=lambda k: roc_auc_score(
                  yy, R[k][0]))
    R["blend"] = (0.5 * (R[bn0][0]
                         + R["stored"][0]),
                  None)

    base = roc_auc_score(yy, R["stored"][0])
    print("  %-16s %7s %7s %8s %8s %s"
          % ("model", "AUC", "AP", "single",
             "multi", "C"))
    for nm in ["stored"] + [j[0] for j in JOBS] \
            + ["blend"]:
        v, cs = R[nm]
        au = roc_auc_score(yy, v)
        a1 = (roc_auc_score(yy[~multi],
                            v[~multi])
              if yy[~multi].sum() > 5
              else np.nan)
        a2 = (roc_auc_score(yy[multi],
                            v[multi])
              if yy[multi].sum() > 5
              else np.nan)
        mk = "*" if au > base else " "
        print("  %-16s %.4f  %.4f  %.4f"
              "  %.4f %s %s"
              % (nm, au,
                 average_precision_score(yy, v),
                 a1, a2, mk,
                 str(sorted(set(cs)))
                 if cs else ""), flush=True)
        rows.append({
            "outcome": oc, "model": nm,
            "n": int(ok.sum()),
            "ev": int(yy.sum()), "auc": au,
            "auc_single": a1, "auc_multi": a2,
            "vs_stored": au - base})

    bn = max([k for k in R if k != "stored"],
             key=lambda k: roc_auc_score(
                 yy, R[k][0]))
    g, lo, hi, _ = f.boot_diff(
        yy, R[bn][0], R["stored"][0], gg)
    print("")
    print("  best (%s) vs stored: %+.4f"
          " [%+.4f,%+.4f] %s"
          % (bn, g, lo, hi,
             "*" if (lo > 0 or hi < 0) else ""),
          flush=True)
    rows.append({
        "outcome": oc,
        "model": "BEST:" + bn,
        "n": int(ok.sum()),
        "ev": int(yy.sum()),
        "auc": roc_auc_score(yy, R[bn][0]),
        "auc_single": lo, "auc_multi": hi,
        "vs_stored": g})

    # named coefficients from the tuned head
    im = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Z = sc.fit_transform(
        im.fit_transform(Xl))
    _, cbest = fold_lr(Xl, yy, gg, Xl)
    mm = LogisticRegression(C=cbest,
                            max_iter=6000)
    mm.fit(Z, yy)
    co = mm.coef_[0]
    o = np.argsort(-np.abs(co))[:12]
    print("")
    print("  TOP STATEMENTS (C=%.3f)" % cbest)
    for k in o:
        print("    %-14s %+.4f"
              % (LG[k].replace("scp_", ""),
                 co[k]))
        coefs.append({
            "outcome": oc,
            "statement":
                LG[k].replace("scp_", ""),
            "coef": float(co[k]), "C": cbest})

    src = ("death_30d"
           if oc == "death_30d_inhosp" else oc)
    se, ysrc = f.labels(ins, src)
    p1, q1 = f.prep(
        se[f.EHR_COLS].values.astype(float),
        d.loc[ok, f.EHR_COLS].values
        .astype(float))
    a_ehr = roc_auc_score(
        yy, _rank(f.fit_lr(
            p1, ysrc, "ehr").predict_proba(
            q1)[:, 1]))
    ab = roc_auc_score(yy, R[bn][0])
    gp = a_ehr - ab
    # predicted gain uses the late_best gap line
    # from 20_gap_tuned.py, typed in
    print("")
    print("  EHR %.4f | best ECG %.4f"
          " | gap %.4f -> predicted 3mod"
          " gain %+.4f"
          % (a_ehr, ab, gp,
             0.0360 - 0.2425 * gp), flush=True)
    rows.append({
        "outcome": oc, "model": "GAP_CHECK",
        "n": int(ok.sum()), "ev": int(yy.sum()),
        "auc": ab, "auc_single": a_ehr,
        "auc_multi": gp,
        "vs_stored": 0.0360 - 0.2425 * gp})

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)
pd.DataFrame(coefs).to_csv(
    os.path.join(PROC, "exp0_coefs.csv"),
    index=False)

print("")
print("=" * 72)
print("VALIDATION: does orig_bal_C1 reproduce"
      " the stored modality?")
q = r[r["model"].isin(
    ["stored", "orig_bal_C1"])]
print(q.pivot_table(index="model",
                    columns="outcome",
                    values="auc")
      .round(4).to_string())

print("")
print("GAIN OVER THE STORED MODALITY")
q = r[~r["model"].str.startswith(
    ("BEST:", "GAP_"))]
print(q.pivot_table(index="model",
                    columns="outcome",
                    values="vs_stored")
      .round(4).to_string())

print("")
print("BEST PER OUTCOME")
for oc, s in q.groupby("outcome"):
    b = s.loc[s["auc"].idxmax()]
    print("  %-18s %-16s %.4f  (%+.4f)"
          % (oc, b["model"], b["auc"],
             b["vs_stored"]))

print("")
print("GAP CHECK")
g = r[r["model"] == "GAP_CHECK"]
print(g[["outcome", "auc", "auc_single",
         "auc_multi", "vs_stored"]]
      .rename(columns={
          "auc": "best_ecg",
          "auc_single": "ehr",
          "auc_multi": "gap",
          "vs_stored": "pred_3mod"})
      .round(4).to_string(index=False))

print("")
print("saved", DEST, r.shape)