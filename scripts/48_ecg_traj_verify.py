"""Verify the trajectory result.

Two problems with script 47:
  1. 'stored' came from a model fitted
     elsewhere, while 'static' and 'traj' were
     fitted by CV on MIMIC. That advantage is
     not trajectory information.
  2. Whether the three-modality model improves
     was not tested on this subset.

This refits every representation the same way on
the same 972 admissions, then runs the full
three-modality fusion with a paired bootstrap.
Run in venv (analysis).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
SEED, NFOLD = 42, 5
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
OUTS = ["death_30d", "death_30d_inhosp",
        "composite_30d"]
# late_best gap line from 20_gap_tuned.py, typed in
INT2, SL2 = 0.0360, -0.2425


def _rank(p):
    return pd.Series(p).rank(pct=True).values


def oof(X, y, grp):
    """Grouped OOF with C chosen inside each
    training fold."""
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    for tr, te in cv.split(X, y, grp):
        im = SimpleImputer(strategy="median")
        a = im.fit_transform(X[tr])
        b = im.transform(X[te])
        sc = StandardScaler()
        a, b = sc.fit_transform(a), sc.transform(b)
        best, bc = -1.0, 0.1
        icv = StratifiedGroupKFold(
            n_splits=3, shuffle=True,
            random_state=SEED)
        for c in CS:
            q = np.zeros(len(tr))
            for t2, v2 in icv.split(
                    a, y[tr], grp[tr]):
                m = LogisticRegression(
                    C=c, max_iter=5000)
                m.fit(a[t2], y[tr][t2])
                q[v2] = m.predict_proba(
                    a[v2])[:, 1]
            s = roc_auc_score(y[tr], q)
            if s > best:
                best, bc = s, c
        m = LogisticRegression(C=bc,
                               max_iter=5000)
        m.fit(a, y[tr])
        p[te] = m.predict_proba(b)[:, 1]
    return p


def gw(ps, y, step=0.05):
    k = len(ps)
    best, bw = -1.0, tuple([1.0 / k] * k)
    if k == 2:
        for w in np.arange(0, 1.001, step):
            a = roc_auc_score(
                y, w * ps[0] + (1 - w) * ps[1])
            if a > best:
                best, bw = a, (w, 1 - w)
    else:
        for w1 in np.arange(0, 1.001, step):
            for w2 in np.arange(
                    0, 1.001 - w1, step):
                a = roc_auc_score(
                    y, w1 * ps[0] + w2 * ps[1]
                    + (1 - w1 - w2) * ps[2])
                if a > best:
                    best, bw = a, (
                        w1, w2, 1 - w1 - w2)
    return bw


def wcv(ps, y, grp):
    out = np.zeros(len(y))
    ws = []
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=SEED)
    X = np.column_stack(ps)
    for tr, te in cv.split(X, y, grp):
        w = gw([p[tr] for p in ps], y[tr])
        ws.append(w)
        out[te] = sum(wi * p[te]
                      for wi, p in zip(w, ps))
    return out, np.mean(ws, axis=0)


traj = pd.read_csv(os.path.join(
    PROC, "ecg_trajectory.csv"))
stat = pd.read_csv(os.path.join(
    PROC, "ecg_derived_v3.csv"))
mim = f.load_mimic()
ins = f.load_inspect()

tcov = traj.notna().mean()
tcols = [c for c in traj.columns
         if c not in ("hadm_id", "subject_id")
         and tcov[c] >= 0.60]
scov = stat.notna().mean()
scols = [c for c in stat.columns
         if c not in ("subject_id", "hadm_id")
         and pd.api.types.is_numeric_dtype(
             stat[c])
         and scov[c] >= 0.60]
print("trajectory cols:", len(tcols))
print("static cols:", len(scols))

rows, fus = [], []

for oc in OUTS:
    if oc not in mim.columns:
        continue
    d = mim
    y = pd.to_numeric(d[oc], errors="coerce")
    y = y.fillna(0).astype(int).values
    grp = d["subject_id"].values

    fp = os.path.join(
        FIG, "p_ecg_harm_%s.csv" % oc)
    stq = pd.read_csv(fp)[
        ["hadm_id", "p_ecg"]].drop_duplicates(
        "hadm_id")

    a = d[["hadm_id"]].merge(
        traj[["hadm_id"] + tcols],
        on="hadm_id", how="left")
    a = a.merge(stat[["hadm_id"] + scols],
                on="hadm_id", how="left")
    a = a.merge(stq, on="hadm_id", how="left")

    ok = (a[tcols].notna().sum(axis=1)
          >= 10).values
    ok &= np.isfinite(a["p_ecg"]).values
    yy, gg = y[ok], grp[ok]
    if yy.sum() < 25:
        continue

    print("")
    print("=" * 64)
    print("%s   n=%d  ev=%d  (multi-ECG subset)"
          % (oc, int(ok.sum()), int(yy.sum())))

    ps = a.loc[ok, "p_ecg"].values
    Xt = a.loc[ok, tcols].values.astype(float)
    Xs = a.loc[ok, scols].values.astype(float)

    # PART 1: equal footing. Every
    # representation refitted the same way.
    E = {}
    E["stored_raw"] = _rank(ps)
    E["stored_refit"] = _rank(
        oof(ps.reshape(-1, 1), yy, gg))
    E["static"] = _rank(oof(Xs, yy, gg))
    E["traj"] = _rank(oof(Xt, yy, gg))
    E["static+traj"] = _rank(oof(
        np.column_stack([Xs, Xt]), yy, gg))
    E["all"] = _rank(oof(
        np.column_stack(
            [Xs, Xt, ps.reshape(-1, 1)]),
        yy, gg))

    print("")
    print("  PART 1: EQUAL FOOTING")
    print("  %-14s %7s %7s %8s"
          % ("model", "AUC", "AP", "n feat"))
    nf = {"stored_raw": 1, "stored_refit": 1,
          "static": len(scols),
          "traj": len(tcols),
          "static+traj": len(scols) + len(tcols),
          "all": len(scols) + len(tcols) + 1}
    for nm in ["stored_raw", "stored_refit",
               "static", "traj",
               "static+traj", "all"]:
        au = roc_auc_score(yy, E[nm])
        print("  %-14s %.4f  %.4f  %6d"
              % (nm, au,
                 average_precision_score(
                     yy, E[nm]), nf[nm]))
        rows.append({
            "outcome": oc, "model": nm,
            "n": int(ok.sum()),
            "ev": int(yy.sum()), "auc": au,
            "nfeat": nf[nm]})

    br = roc_auc_score(yy, E["stored_refit"])
    bs = roc_auc_score(yy, E["static"])
    bt = roc_auc_score(yy, E["traj"])
    print("")
    print("  traj vs static (both refitted):"
          " %+.4f" % (bt - bs))
    g, lo, hi, _ = f.boot_diff(
        yy, E["traj"], E["static"], gg)
    print("    bootstrap %+.4f [%+.4f,%+.4f]%s"
          % (g, lo, hi,
             " *" if (lo > 0 or hi < 0) else ""))
    print("  best vs stored_refit: %+.4f"
          % (max(bs, bt,
                 roc_auc_score(
                     yy, E["static+traj"]))
             - br))

    # PART 2: does it change the three-modality
    # model on this cohort?
    src_oc = ("death_30d"
              if oc == "death_30d_inhosp"
              else oc)
    se, ysrc = f.labels(ins, src_oc)
    p1, q1 = f.prep(
        se[f.EHR_COLS].values.astype(float),
        d.loc[ok, f.EHR_COLS].values
        .astype(float))
    p_ehr = _rank(f.fit_lr(
        p1, ysrc, "ehr").predict_proba(
        q1)[:, 1])
    p2, q2 = f.prep(
        se[f.CTPA_COLS].values.astype(float),
        d.loc[ok, f.CTPA_COLS].values
        .astype(float))
    p_ct = _rank(f.fit_lr(
        p2, ysrc, "ctpa").predict_proba(
        q2)[:, 1])

    a_ehr = roc_auc_score(yy, p_ehr)
    a_ct = roc_auc_score(yy, p_ct)
    two, w2 = wcv([p_ct, p_ehr], yy, gg)
    a2 = roc_auc_score(yy, two)

    print("")
    print("  PART 2: THREE-MODALITY FUSION")
    print("  EHR %.4f | CTPA %.4f"
          " | ctpa+ehr %.4f"
          % (a_ehr, a_ct, a2))
    print("  %-14s %7s %8s %9s %8s %s"
          % ("ecg version", "ecgAUC", "gap",
             "predicted", "3mod", "observed"))

    for nm in ["stored_raw", "traj",
               "static+traj", "all"]:
        v = E[nm]
        ae = roc_auc_score(yy, v)
        gp = a_ehr - ae
        pr = INT2 + SL2 * gp
        p3, w3 = wcv([p_ct, v, p_ehr], yy, gg)
        a3 = roc_auc_score(yy, p3)
        gn, lo, hi, _ = f.boot_diff(
            yy, p3, two, gg)
        star = ("*" if (lo > 0 or hi < 0)
                else " ")
        print("  %-14s %.4f  %+.4f  %+.4f"
              "  %.4f  %+.4f"
              " [%+.4f,%+.4f] %s"
              % (nm, ae, gp, pr, a3, gn,
                 lo, hi, star))
        fus.append({
            "outcome": oc, "ecg": nm,
            "n": int(ok.sum()),
            "ev": int(yy.sum()),
            "ecg_auc": ae, "gap": gp,
            "predicted": pr, "auc_2mod": a2,
            "auc_3mod": a3, "gain": gn,
            "lo": lo, "hi": hi,
            "w_ecg": float(w3[1]),
            "sig": int(lo > 0 or hi < 0)})

r1 = pd.DataFrame(rows)
r2 = pd.DataFrame(fus)
r1.to_csv(os.path.join(
    PROC, "ecg_traj_equal.csv"), index=False)
r2.to_csv(os.path.join(
    PROC, "ecg_traj_fusion.csv"), index=False)

print("")
print("=" * 64)
print("HOW MUCH OF THE GAIN WAS TARGET"
      " FITTING?")
print("=" * 64)
for oc, s in r1.groupby("outcome"):
    g = s.set_index("model")["auc"]
    if not {"stored_raw", "stored_refit"} \
            <= set(g.index):
        continue
    print("  %-18s raw %.4f -> refit %.4f"
          "  (fitting worth %+.4f)"
          % (oc, g["stored_raw"],
             g["stored_refit"],
             g["stored_refit"] - g["stored_raw"]))

print("")
print("TRAJECTORY VALUE, LIKE FOR LIKE")
for oc, s in r1.groupby("outcome"):
    g = s.set_index("model")["auc"]
    print("  %-18s static %.4f  traj %.4f"
          "  (%+.4f)  both %.4f"
          % (oc, g["static"], g["traj"],
             g["traj"] - g["static"],
             g["static+traj"]))

print("")
print("=" * 64)
print("DOES THE THIRD MODALITY NOW PAY?")
print("=" * 64)
for oc, s in r2.groupby("outcome"):
    b = s.loc[s["auc_3mod"].idxmax()]
    print("  %-18s best %s: 2mod %.4f ->"
          " 3mod %.4f  %+.4f"
          " [%+.4f,%+.4f] %s"
          % (oc, b["ecg"], b["auc_2mod"],
             b["auc_3mod"], b["gain"],
             b["lo"], b["hi"],
             "SIGNIFICANT" if b["sig"]
             else "not significant"))

print("")
print("PREDICTED VS OBSERVED (gap mechanism)")
v = r2[["outcome", "ecg", "gap", "predicted",
        "gain"]].copy()
v["resid"] = v["gain"] - v["predicted"]
print(v.round(4).to_string(index=False))
print("  mean abs residual: %.4f"
      % float(v["resid"].abs().mean()))

print("")
print("saved ecg_traj_equal.csv,"
      " ecg_traj_fusion.csv")