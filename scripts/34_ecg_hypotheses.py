"""Two hypotheses about the ECG modality.

H1 OUTCOME MISMATCH. An ECG measures physiology.
The outcome decomposition found in-hospital death
is physiology-driven while 30-day mortality is
comorbidity-driven, so the ECG modality may be
answering the wrong question.

H2 CONDITIONAL VALUE. Right ventricular strain is
the classic ECG sign in pulmonary embolism. The
modality may be informative where CTPA reports RV
strain and useless elsewhere, which would make
conditional fusion a different architecture from
global weighting.

Uses the mean+max ECG aggregation, which script 33
found best.
Run in venv (analysis).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
SEED, NFOLD = 42, 5
# late_best gap line from 20_gap_tuned.py, typed in
INT, SLOPE = 0.0360, -0.2425

OUTS = ["death_30d", "death_30d_inhosp",
        "composite_30d", "cv_first"]


def _rank(p):
    return pd.Series(p).rank(pct=True).values


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


def wcv(R, y, grp, cb):
    ps = [R[t] for t in cb]
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


def labels_any(df, oc):
    """f.labels only knows three outcomes, so
    handle death_30d_inhosp here."""
    if oc in f.OUTCOMES:
        return f.labels(df, oc)
    d = df
    y = pd.to_numeric(d[oc], errors="coerce")
    return d, y.fillna(0).astype(int).values


keys = pd.read_csv(os.path.join(
    FIG, "fusion_cohort_keys.csv"))
keys = keys[["hadm_id", "p_ecg",
             "p_ecg_max"]].drop_duplicates(
    "hadm_id")

mm = f.load_mimic()
ins = f.load_inspect()
have = [c for c in OUTS if c in mm.columns]
print("outcomes available:", have)

rows = []
sub_rows = []

for oc in have:
    st_fp = os.path.join(
        FIG, "p_ecg_harm_%s.csv" % oc)
    if not os.path.exists(st_fp):
        print("")
        print("skip %s: no stored ECG file" % oc)
        continue

    d, y = labels_any(mm, oc)
    grp = d["subject_id"].values

    src_oc = oc
    if oc == "death_30d_inhosp":
        src_oc = "death_30d"
        print("")
        print("NOTE: INSPECT has no in-hospital"
              " label, so EHR and CTPA are")
        print("      trained on death_30d and"
              " evaluated on death_30d_inhosp")
    se, ysrc = f.labels(ins, src_oc)

    a, b = f.prep(
        se[f.EHR_COLS].values.astype(float),
        d[f.EHR_COLS].values.astype(float))
    p_ehr = f.fit_lr(
        a, ysrc, "ehr").predict_proba(b)[:, 1]
    c, e = f.prep(
        se[f.CTPA_COLS].values.astype(float),
        d[f.CTPA_COLS].values.astype(float))
    p_ct = f.fit_lr(
        c, ysrc, "ctpa").predict_proba(e)[:, 1]

    k = d[["hadm_id"]].merge(
        keys, on="hadm_id", how="left")
    p_ecg = 0.5 * (_rank(k["p_ecg"].values)
                   + _rank(k["p_ecg_max"].values))

    ok = (np.isfinite(p_ecg)
          & np.isfinite(p_ehr)
          & np.isfinite(p_ct))
    yy = y[ok]
    gg = grp[ok]
    if yy.sum() < 30:
        print("")
        print("skip %s: only %d events"
              % (oc, int(yy.sum())))
        continue

    R = {"ehr": _rank(p_ehr[ok]),
         "ctpa": _rank(p_ct[ok]),
         "ecg": _rank(p_ecg[ok])}

    print("")
    print("=" * 60)
    print("%s   n=%d  ev=%d  rate %.3f"
          % (oc, int(ok.sum()), int(yy.sum()),
             float(yy.mean())))

    ae = roc_auc_score(yy, R["ehr"])
    ac = roc_auc_score(yy, R["ctpa"])
    ag = roc_auc_score(yy, R["ecg"])
    p2, w2 = wcv(R, yy, gg, ("ctpa", "ehr"))
    p3, w3 = wcv(R, yy, gg,
                 ("ctpa", "ecg", "ehr"))
    a2 = roc_auc_score(yy, p2)
    a3 = roc_auc_score(yy, p3)
    gap = ae - ag
    pred = INT + SLOPE * gap

    print("  ehr %.4f | ctpa %.4f | ecg %.4f"
          % (ae, ac, ag))
    print("  ecg gap to ehr: %.4f"
          "   predicted 3mod gain %+.4f"
          % (gap, pred))
    print("  ctpa+ehr      %.4f  w=%s"
          % (a2, [round(x, 2) for x in w2]))
    print("  ctpa+ecg+ehr  %.4f  w=%s"
          "   observed %+.4f"
          % (a3, [round(x, 2) for x in w3],
             a3 - a2))

    rows.append({
        "test": "H1_outcome", "outcome": oc,
        "n": int(ok.sum()), "ev": int(yy.sum()),
        "ehr": ae, "ctpa": ac, "ecg": ag,
        "gap": gap, "auc_2mod": a2,
        "auc_3mod": a3,
        "gain_3v2": a3 - a2,
        "predicted": pred,
        "w_ecg": float(w3[1])})

    # ---- H2: conditional on RV strain ----
    if "rv_strain" not in d.columns:
        continue
    rv = d["rv_strain"].values[ok].astype(int)
    print("")
    print("  RV STRAIN SUBGROUPS"
          "  (flagged %d of %d, %.1f%%)"
          % (int(rv.sum()), len(rv),
             100.0 * rv.mean()))
    for nm, m in (("rv_strain=1", rv == 1),
                  ("rv_strain=0", rv == 0)):
        if m.sum() < 100 or yy[m].sum() < 15:
            print("    %-12s too few"
                  " (n=%d ev=%d)"
                  % (nm, int(m.sum()),
                     int(yy[m].sum())))
            continue
        se_ = roc_auc_score(yy[m], R["ehr"][m])
        sg = roc_auc_score(yy[m], R["ecg"][m])
        sc = roc_auc_score(yy[m], R["ctpa"][m])
        print("    %-12s n=%4d ev=%3d"
              "  ehr %.4f  ctpa %.4f"
              "  ecg %.4f  gap %.4f"
              % (nm, int(m.sum()),
                 int(yy[m].sum()), se_, sc, sg,
                 se_ - sg))
        sub_rows.append({
            "test": "H2_subgroup",
            "outcome": oc, "subgroup": nm,
            "n": int(m.sum()),
            "ev": int(yy[m].sum()),
            "ehr": se_, "ctpa": sc, "ecg": sg,
            "gap": se_ - sg})

    # conditional fusion: ECG only where rv=1
    p_cond = p2.copy()
    m = rv == 1
    if m.sum() >= 100 and yy[m].sum() >= 15:
        pc3, _ = wcv(
            {t: v[m] for t, v in R.items()},
            yy[m], gg[m],
            ("ctpa", "ecg", "ehr"))
        p_cond[m] = pc3
        ac_ = roc_auc_score(yy, p_cond)
        gd, lo, hi, _ = f.boot_diff(
            yy, p_cond, p2, gg)
        star = ("*" if (lo > 0 or hi < 0)
                else " ")
        print("    conditional fusion %.4f"
              "  vs ctpa+ehr %+.4f"
              " [%+.4f,%+.4f] %s"
              % (ac_, gd, lo, hi, star))
        rows.append({
            "test": "H2_conditional",
            "outcome": oc, "n": int(ok.sum()),
            "ev": int(yy.sum()), "ehr": ae,
            "ctpa": ac, "ecg": ag, "gap": gap,
            "auc_2mod": a2, "auc_3mod": ac_,
            "gain_3v2": ac_ - a2,
            "predicted": np.nan,
            "w_ecg": np.nan})

r = pd.DataFrame(rows)
s = pd.DataFrame(sub_rows)
r.to_csv(os.path.join(PROC,
                      "ecg_hypotheses.csv"),
         index=False)
if len(s):
    s.to_csv(os.path.join(
        PROC, "ecg_subgroups.csv"), index=False)

print("")
print("=" * 60)
print("H1: IS THE ECG MODALITY STRONGER ON"
      " IN-HOSPITAL DEATH?")
h1 = r[r["test"] == "H1_outcome"]
print(h1[["outcome", "n", "ev", "ecg", "ehr",
          "gap", "auc_2mod", "auc_3mod",
          "gain_3v2", "predicted"]]
      .round(4).to_string(index=False))
if {"death_30d",
        "death_30d_inhosp"} <= set(h1["outcome"]):
    a = float(h1[h1["outcome"] == "death_30d"]
              ["ecg"].iloc[0])
    b_ = float(h1[h1["outcome"]
                  == "death_30d_inhosp"]
               ["ecg"].iloc[0])
    print("")
    print("  ecg on death_30d          %.4f"
          % a)
    print("  ecg on death_30d_inhosp   %.4f"
          % b_)
    print("  difference                %+.4f"
          % (b_ - a))
    if b_ - a > 0.03:
        print("  SUPPORTED: the ECG modality is"
              " stronger on in-hospital death")
    elif b_ - a > 0.01:
        print("  WEAK SUPPORT")
    else:
        print("  NOT SUPPORTED: outcome"
              " mismatch is not the issue")

if len(s):
    print("")
    print("H2: IS THE ECG MODALITY BETTER WHERE"
          " CTPA REPORTS RV STRAIN?")
    print(s.round(4).to_string(index=False))
    pv = s.pivot_table(index="outcome",
                       columns="subgroup",
                       values="ecg")
    if pv.shape[1] == 2:
        pv["diff"] = (pv["rv_strain=1"]
                      - pv["rv_strain=0"])
        print("")
        print("  ECG AUC by subgroup")
        print(pv.round(4).to_string())
        md = float(pv["diff"].mean())
        print("  mean difference %+.4f" % md)
        if md > 0.05:
            print("  SUPPORTED: the ECG modality is"
                  " selectively informative")
        else:
            print("  NOT SUPPORTED: the ECG"
                  " modality is not selectively"
                  " informative")

    cf = r[r["test"] == "H2_conditional"]
    if len(cf):
        print("")
        print("  CONDITIONAL FUSION vs ctpa+ehr")
        print(cf[["outcome", "auc_2mod",
                  "auc_3mod", "gain_3v2"]]
              .round(4).to_string(index=False))

print("")
print("saved ecg_hypotheses.csv", r.shape)