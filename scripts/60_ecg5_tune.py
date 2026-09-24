"""Four experiments on a learned ECG
representation rather than handcrafted
measurements.

  1 C tuning
  2 class weighting
  3 gradient boosting over the representation
  4 trajectory on the learned features

Item 4 combines trajectory features with the
learned representation.

Feature blocks:
  cls    10 cols, 5 logits x mean/max
  pca30  512-dim embedding reduced to 30 PCs
  traj   first/last/delta/slope of the logits
         across serial recordings
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\ecg5_tune.csv
  results\\ecg5_tune_log.txt
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
      0.1, 0.3, 1.0]
NPC = 30
OUTS = ["death_30d", "death_30d_inhosp",
        "composite_30d"]
DEST = os.path.join(PROC, "ecg5_tune.csv")


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def fold_lr(Xa, ya, ga, Xb, cw=None,
            fixed_c=None, npc=None):
    im = SimpleImputer(strategy="median")
    a = im.fit_transform(Xa)
    b = im.transform(Xb)
    sc = StandardScaler()
    a, b = sc.fit_transform(a), sc.transform(b)
    if npc:
        k = min(npc, a.shape[1],
                a.shape[0] - 1)
        p = PCA(n_components=k,
                random_state=SEED)
        a, b = p.fit_transform(a), p.transform(b)
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


def fold_gb(Xa, ya, Xb):
    m = HistGradientBoostingClassifier(
        random_state=SEED, max_depth=3,
        learning_rate=0.05, max_iter=300,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15)
    m.fit(Xa, ya)
    return m.predict_proba(Xb)[:, 1]


def oof(X, y, grp, folds, kind="lr", **kw):
    p = np.zeros(len(y))
    cs = []
    for tr, te in folds:
        if kind == "gb":
            p[te] = fold_gb(X[tr], y[tr], X[te])
        else:
            p[te], c = fold_lr(
                X[tr], y[tr], grp[tr], X[te],
                **kw)
            cs.append(c)
    return p, cs


# ---------- build trajectory on the logits ----
def build_traj():
    rec = pd.read_csv(os.path.join(
        PROC, "ecg5_record.csv"),
        usecols=lambda c: (c == "rec"
                           or c.startswith(
                               "cls_")))
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
    cls = [c for c in j.columns
           if c.startswith("cls_")]
    rows = []
    for h, g in j.groupby("hadm_id"):
        if len(g) < 2:
            continue
        d = {"hadm_id": h, "n_rec": len(g)}
        hrs = (g["t"].iloc[-1]
               - g["t"].iloc[0]).total_seconds()
        hrs = hrs / 3600.0
        d["span_h"] = hrs
        for c in cls:
            v = pd.to_numeric(
                g[c], errors="coerce").values
            if np.isnan(v).all():
                continue
            d[c + "_first"] = float(v[0])
            d[c + "_last"] = float(v[-1])
            d[c + "_delta"] = float(v[-1] - v[0])
            d[c + "_absd"] = float(
                abs(v[-1] - v[0]))
            d[c + "_sd"] = float(np.nanstd(v))
            d[c + "_rng"] = float(
                np.nanmax(v) - np.nanmin(v))
            if hrs > 0.5:
                d[c + "_slope"] = float(
                    (v[-1] - v[0]) / hrs)
        rows.append(d)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(
        PROC, "ecg5_trajectory.csv"),
        index=False)
    print("logit trajectory:", t.shape,
          flush=True)
    return t


e5 = pd.read_csv(os.path.join(PROC, "ecg5.csv"))
print("ecg5:", e5.shape, flush=True)
traj5 = build_traj()

CLS = [c for c in e5.columns
       if c.startswith("cls_")]
EMB = [c for c in e5.columns
       if c.startswith("e_")]
TR = [c for c in traj5.columns
      if c not in ("hadm_id",)]
print("cls %d  emb %d  traj %d"
      % (len(CLS), len(EMB), len(TR)),
      flush=True)

mim = f.load_mimic()
ins = f.load_inspect()
rows = []

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
        e5[["hadm_id"] + CLS + EMB],
        on="hadm_id", how="left")
    a = a.merge(traj5, on="hadm_id",
                how="left")
    a = a.merge(stq, on="hadm_id", how="left")

    ok = a[CLS].notna().all(axis=1).values
    ok &= np.isfinite(a["p_ecg"]).values
    yy, gg = y[ok], grp[ok]
    if yy.sum() < 30:
        continue

    Xc = a.loc[ok, CLS].values.astype(float)
    Xe = a.loc[ok, EMB].values.astype(float)
    Xt = a.loc[ok, TR].values.astype(float)
    ps = a.loc[ok, "p_ecg"].values
    multi = (a.loc[ok, TR].notna()
             .sum(axis=1) >= 10).values

    print("")
    print("=" * 70)
    print("%s  n=%d ev=%d  multi=%d"
          % (oc, int(ok.sum()), int(yy.sum()),
             int(multi.sum())), flush=True)

    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    folds = list(cv.split(Xc, yy, gg))

    R = {"stored": (_rank(ps), None)}
    JOBS = [
        ("cls_C1", Xc, "lr",
         {"fixed_c": 1.0}),
        ("cls_Ctuned", Xc, "lr", {}),
        ("cls_bal", Xc, "lr",
         {"cw": "balanced"}),
        ("cls_gb", Xc, "gb", {}),
        ("emb_pca", Xe, "lr", {"npc": NPC}),
        ("emb_pca_bal", Xe, "lr",
         {"npc": NPC, "cw": "balanced"}),
        ("cls_traj", np.column_stack([Xc, Xt]),
         "gb", {}),
        ("cls_emb", np.column_stack(
            [Xc, Xe]), "lr", {"npc": NPC}),
        ("all_gb", np.column_stack(
            [Xc, Xt]), "gb", {}),
    ]
    for nm, X, kind, kw in JOBS:
        p, cs = oof(X, yy, gg, folds,
                    kind=kind, **kw)
        R[nm] = (_rank(p), cs)

    best_new = max(
        [k for k in R if k != "stored"],
        key=lambda k: roc_auc_score(yy, R[k][0]))
    R["blend"] = (0.5 * (R[best_new][0]
                         + R["stored"][0]), None)

    base = roc_auc_score(yy, R["stored"][0])
    print("  %-14s %7s %7s %8s %8s %s"
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
        print("  %-14s %.4f  %.4f  %.4f"
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

print("")
print("=" * 70)
print("GAIN OVER THE STORED 71-SCP MODALITY")
q = r[r["model"] != "GAP_CHECK"]
print(q.pivot_table(index="model",
                    columns="outcome",
                    values="vs_stored")
      .round(4).to_string())

print("")
print("BEST PER OUTCOME")
for oc, s in q.groupby("outcome"):
    b = s.loc[s["auc"].idxmax()]
    print("  %-18s %-14s %.4f  (%+.4f)"
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