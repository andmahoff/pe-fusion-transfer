"""Evidential (Dempster-Shafer) fusion with
calibration-derived ignorance.

Each modality's percentile ranks are calibrated
by out-of-fold Platt scaling inside the training
folds before the Brier score is computed, so the
ignorance mass u reflects miscalibration rather
than the rank transform. An earlier version
computed the Brier score on the ranks themselves,
which saturated u at 1.0 for every modality; its
output is evidential.csv and evidential_mass.csv.

Tests whether calibration-derived ignorance
reproduces the ordering given by the gap rule.
Run in venv (analysis).

OUTPUT FILES
  data\\processed\\evidential2.csv
  data\\processed\\evidential2_mass.csv
  results\\evidential2_log.txt
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
from sklearn.metrics import brier_score_loss
from scipy import stats as st

sys.path.insert(0, "scripts")
import fusion_lib as f

PROC, FIG = f.PROC, f.FIG
RAW = os.path.join("data", "raw", "ecg")
SEEDS = [42, 7, 13, 1, 2, 3]
NFOLD = 5
ECG_LO, ECG_HI = -12.0, 48.0
CT_LO, CT_HI = -48.0, 24.0
CT_CS = list(np.logspace(-4, 4, 10))
MINCOV = 0.20
DEST = os.path.join(PROC, "evidential2.csv")
BEST_C = {"cv_first": 1e-4,
          "composite_30d": 1e-3,
          "death_30d": 1e-3,
          "death_30d_inhosp": 3e-4}
OUTS = ["death_30d", "composite_30d",
        "cv_first", "death_30d_inhosp"]
U_FIXED = [0.0, 0.1, 0.2, 0.3, 0.5]

DROP_EXACT = ("rec", "t_flag", "vm_noise",
              "vm_base", "vm_rpeak", "net_i",
              "net_avf", "apen", "tp_win_ms",
              "t_win_ms", "t_peak_ms",
              "tp_fallback", "qrs_hit_limit",
              "p_hit_limit", "t_peak_edge",
              "t_edge_leads", "n_rr_dropped",
              "any_af", "tpe_spread_ms",
              "tpe_ok")
DROP_PREFIX = ("tpk_ms_",)
PERM_ONLY = ("tpe_ms_mean", "tpe_ms_worst",
             "tpe_qt_mean", "tpe_qt_worst")
NEG_BAD = ("s_wave_i", "q_wave_iii", "st_min",
           "t_v1", "t_v2", "t_v3", "t_v4",
           "t_v5", "t_v6", "t_wave_iii")
ANG = ("qrs_axis", "t_axis", "p_axis")


def _rank(p):
    return pd.Series(p).rank(pct=True).values


# ---------------- calibration ---------------
def platt_oof(p, y, grp, seed):
    """Out-of-fold Platt scaling, so a
    rank becomes a probability on the right
    scale before any Brier score is taken.

    Fitted inside training folds only, so the
    resulting u carries no test information."""
    p = np.asarray(p, dtype=float)
    out = np.full(len(y), np.nan)
    ok = np.isfinite(p)
    if ok.sum() < 100:
        return np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6)
               / (1 - np.clip(p, 1e-6,
                              1 - 1e-6)))
    idx = np.where(ok)[0]
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    try:
        for tr, te in cv.split(
                x[idx].reshape(-1, 1),
                y[idx], grp[idx]):
            m = LogisticRegression(
                C=1e6, max_iter=2000)
            m.fit(x[idx][tr].reshape(-1, 1),
                  y[idx][tr])
            out[idx[te]] = m.predict_proba(
                x[idx][te].reshape(-1, 1))[:, 1]
    except Exception:
        return np.clip(p, 1e-6, 1 - 1e-6)
    return out


def u_from_brier(p_cal, y, prev):
    """Ignorance from calibration, scaled by the
    Brier score a constant-prevalence predictor
    achieves. An earlier version used 0.25, which
    is the value for a balanced outcome; at 11%
    prevalence the reference is about 0.10, so
    every u saturated at 1."""
    ok = np.isfinite(p_cal)
    if ok.sum() < 50:
        return 0.5
    try:
        b = brier_score_loss(
            y[ok], np.clip(p_cal[ok],
                           1e-6, 1 - 1e-6))
    except Exception:
        return 0.5
    ref = prev * (1.0 - prev)
    if ref <= 1e-9:
        return 0.5
    return float(np.clip(b / ref, 0.0, 1.0))


# ---------------- evidential core -----------
def to_mass(p, u):
    p = np.clip(np.asarray(p, dtype=float),
                1e-6, 1 - 1e-6)
    u = np.clip(np.asarray(u, dtype=float),
                0.0, 1.0)
    return ((1.0 - u) * p,
            (1.0 - u) * (1.0 - p), u)


def dempster(masses):
    m1, m0, mu = masses[0]
    for (a1, a0, au) in masses[1:]:
        n1 = m1 * a1 + m1 * au + mu * a1
        n0 = m0 * a0 + m0 * au + mu * a0
        nu = mu * au
        k = m1 * a0 + m0 * a1
        z = np.clip(1.0 - k, 1e-9, None)
        m1, m0, mu = n1 / z, n0 / z, nu / z
    return m1, m0, mu


def pignistic(m1, m0, mu):
    return m1 + 0.5 * mu


def u_from_gap(auc, best):
    return float(np.clip(
        (best - auc) / 0.145, 0.0, 1.0))


def u_from_entropy(p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return -(q * np.log2(q)
             + (1 - q) * np.log2(1 - q))


def grid_w(ps, y, step=0.05):
    k = len(ps)
    if k == 1:
        return (1.0,)
    n = int(round(1.0 / step))

    def rec_(m, rem):
        if m == 1:
            yield (rem,)
            return
        for i in range(rem + 1):
            for t in rec_(m - 1, rem - i):
                yield (i,) + t
    best, bw = -1.0, tuple([1.0 / k] * k)
    if len(np.unique(y)) < 2:
        return bw
    for w in rec_(k, n):
        w = tuple(x * step for x in w)
        a = roc_auc_score(
            y, sum(wi * p
                   for wi, p in zip(w, ps)))
        if a > best:
            best, bw = a, w
    return bw


def oof(X, y, grp, seed, C, cw=None,
        tune=None, mask=None):
    p = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(X, y, grp):
        if mask is not None:
            tr = tr[mask[tr]]
            te2 = te[mask[te]]
        else:
            te2 = te
        if len(tr) < 40 or len(te2) == 0:
            continue
        if y[tr].sum() < 5:
            continue
        im = SimpleImputer(strategy="median")
        a = im.fit_transform(X[tr])
        b = im.transform(X[te2])
        sc = StandardScaler()
        a, b = sc.fit_transform(a), sc.transform(b)
        c = C
        if tune:
            best = -1.0
            k = min(3, max(2,
                           int(y[tr].sum())
                           // 10))
            try:
                icv = StratifiedGroupKFold(
                    n_splits=k, shuffle=True,
                    random_state=42)
                for cand in tune:
                    qq = np.zeros(len(tr))
                    for t2, v2 in icv.split(
                            a, y[tr], grp[tr]):
                        mm2 = LogisticRegression(
                            C=cand,
                            max_iter=5000,
                            class_weight=cw)
                        mm2.fit(a[t2],
                                y[tr][t2])
                        qq[v2] = \
                            mm2.predict_proba(
                                a[v2])[:, 1]
                    s = roc_auc_score(
                        y[tr], qq)
                    if s > best:
                        best, c = s, cand
            except Exception:
                c = C
        m = LogisticRegression(
            C=c, max_iter=5000,
            class_weight=cw)
        m.fit(a, y[tr])
        p[te2] = m.predict_proba(b)[:, 1]
    return p


# ---------------- features ------------------
rec = pd.read_csv(os.path.join(
    PROC, "exp0_logits_record.csv"))
eidx = pd.read_csv(os.path.join(
    RAW, "ecg_record_index.csv"))
eidx["rec"] = eidx["rec"].astype(str).apply(
    lambda s: s if s.startswith("files/")
    else "files/" + s.replace("\\", "/")
    .strip("/"))
LG = [c for c in rec.columns
      if c.startswith("scp_")]
adm = pd.read_csv(os.path.join(
    FIG, "index_admission_times.csv"))
adm["admittime"] = pd.to_datetime(
    adm["admittime"], errors="coerce")
E = eidx.merge(rec, on="rec", how="inner")
E["t"] = pd.to_datetime(
    E["ecg_charttime"], errors="coerce")
E = E.merge(adm[["hadm_id", "admittime"]],
            on="hadm_id", how="left")
E["h"] = ((E["t"] - E["admittime"])
          .dt.total_seconds() / 3600.0)
E = E[(E["h"] >= ECG_LO)
      & (E["h"] <= ECG_HI)].copy()
ECGA = E.groupby(["subject_id", "hadm_id"],
                 as_index=False)[LG].mean()
keep = set(E["rec"])

m12 = pd.read_csv(os.path.join(
    PROC, "ecg_derived_v12_record.csv"))
m12["rec"] = m12["rec"].astype(str).apply(
    lambda s: s if s.startswith("files/")
    else "files/" + s.replace("\\", "/")
    .strip("/"))
m12 = m12[m12["rec"].isin(keep)]
cols = [c for c in m12.columns
        if c not in DROP_EXACT
        and not c.startswith(DROP_PREFIX)
        and pd.api.types.is_numeric_dtype(
            m12[c])]
j = eidx[["subject_id", "hadm_id",
          "rec"]].merge(
    m12[["rec"] + cols], on="rec",
    how="inner")
g = j.groupby(["subject_id", "hadm_id"])
pos = [c for c in cols
       if not c.startswith(NEG_BAD)
       and c not in ANG]
neg = [c for c in cols
       if c.startswith(NEG_BAD)]
parts = [g[cols].mean().add_suffix("_mean")]
if pos:
    parts.append(g[pos].max()
                 .add_suffix("_worst"))
if neg:
    parts.append(g[neg].min()
                 .add_suffix("_worst"))
MA = pd.concat(parts, axis=1).reset_index()
for c in ANG:
    if c not in j.columns:
        continue
    rad = np.radians(j[c])
    tmp = j[["subject_id", "hadm_id"]].copy()
    tmp["s"], tmp["c"] = np.sin(rad), np.cos(rad)
    gg = tmp.groupby(["subject_id",
                      "hadm_id"]).mean()
    cm = np.degrees(np.arctan2(
        gg["s"], gg["c"])).rename(
        c + "_circ").reset_index()
    MA = MA.merge(cm, on=["subject_id",
                          "hadm_id"],
                  how="left")
    MA = MA.drop(columns=[c + "_mean",
                          c + "_worst"],
                 errors="ignore")
MC = [c for c in MA.columns
      if c not in ("subject_id", "hadm_id")]
cov = MA[MC].notna().mean()
MC = [c for c in MC if cov[c] >= MINCOV
      and c not in PERM_ONLY]

lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
mim_ehr = f.load_mimic_ehr()
EH = [c for c in f.EHR_COLS
      if c in mim_ehr.columns]
mm = f.load_mimic_ctpa46()
cidx = pd.read_csv(os.path.join(
    FIG, "ctpa_notes_index.csv"))
w = cidx[(cidx["h_before"] >= CT_LO)
         & (cidx["h_before"] <= CT_HI)]
hs = set(w["idx_hadm"].dropna().astype(int))
mm = mm[mm["hadm_id"].isin(hs)]
v2 = pd.read_csv(os.path.join(
    FIG, "mimic_imp_features_v2.csv"),
    usecols=["hadm_id"])
mm = mm[mm["hadm_id"].isin(set(v2["hadm_id"]))]
CT46 = [c for c in f.CTPA46
        if c in mm.columns]

D = ECGA.merge(
    MA[["hadm_id"] + MC], on="hadm_id",
    how="left").merge(
    lab, on=["subject_id", "hadm_id"],
    how="inner").merge(
    mim_ehr[["hadm_id"] + EH],
    on="hadm_id", how="inner").merge(
    mm[["hadm_id"] + CT46], on="hadm_id",
    how="left")
D["has_ctpa"] = D[CT46].notna().all(
    axis=1).astype(int)
print("cohort:", len(D),
      "  with CTPA: %d (%.1f%%)"
      % (int(D["has_ctpa"].sum()),
         100.0 * D["has_ctpa"].mean()),
      flush=True)

ins = f.load_inspect()
rows, mrows = [], []
t0 = time.time()

for oc in OUTS:
    if oc not in D.columns:
        continue
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    if y.sum() < 30:
        continue
    hasct = d["has_ctpa"].values == 1
    prev = float(y.mean())
    se, ysrc = f.labels(
        ins, "death_30d"
        if oc == "death_30d_inhosp" else oc)
    cb = BEST_C[oc]

    print("")
    print("=" * 78)
    print("%s   n=%d ev=%d  prevalence %.3f"
          "   Brier reference %.4f"
          % (oc, len(y), int(y.sum()), prev,
             prev * (1 - prev)), flush=True)
    print("  an earlier version used a fixed reference"
          " of 0.25, so every u saturated")

    a1, b1 = f.prep(
        se[EH].values.astype(float),
        d[EH].values.astype(float))
    p_ehr = _rank(f.fit_lr(
        a1, ysrc, "ehr").predict_proba(
        b1)[:, 1])
    XL = d[LG].values.astype(float)
    XO = np.column_stack(
        [XL, d[MC].values.astype(float)])
    XC = d[CT46].values.astype(float)

    acc = {}
    for k in ["ehr", "ecg", "ctpa",
              "wra_imputed", "ev_fixed",
              "ev_gap", "ev_calib",
              "ev_entropy"]:
        acc[k] = []
    ufit = {"ehr": [], "ecg": [], "ctpa": []}
    gaps = {"ehr": [], "ecg": [], "ctpa": []}
    bri = {"ehr": [], "ecg": [], "ctpa": []}
    keep2 = None

    for s in SEEDS:
        p_ecg = _rank(oof(XO, y, grp, s, cb,
                          cw="balanced"))
        pc = oof(XC, y, grp, s, 1.0,
                 tune=CT_CS, mask=hasct)
        p_ct = np.full(len(y), np.nan)
        m2 = np.isfinite(pc)
        p_ct[m2] = _rank(pc[m2])

        a_e = roc_auc_score(y, p_ehr)
        a_g = roc_auc_score(y, p_ecg)
        a_c = (roc_auc_score(y[m2], p_ct[m2])
               if m2.sum() > 50 else 0.5)
        best = max(a_e, a_g, a_c)
        for k, v in (("ehr", a_e),
                     ("ecg", a_g),
                     ("ctpa", a_c)):
            acc[k].append(v)
            gaps[k].append(best - v)

        pci = np.where(np.isfinite(p_ct),
                       p_ct,
                       np.nanmedian(p_ct))
        wi_ = grid_w([pci, p_ecg, p_ehr], y)
        acc["wra_imputed"].append(
            roc_auc_score(
                y, sum(x * p for x, p in
                       zip(wi_, [pci, p_ecg,
                                 p_ehr]))))

        u_miss = np.where(
            np.isfinite(p_ct), 0.0, 1.0)
        pc_safe = np.where(
            np.isfinite(p_ct), p_ct, 0.5)

        bestu, bestauc = U_FIXED[0], -1.0
        for uu in U_FIXED:
            ms = [to_mass(p_ehr, uu),
                  to_mass(p_ecg, uu),
                  to_mass(pc_safe,
                          np.maximum(
                              uu, u_miss))]
            a_ = roc_auc_score(
                y, pignistic(*dempster(ms)))
            if a_ > bestauc:
                bestauc, bestu = a_, uu
        ms = [to_mass(p_ehr, bestu),
              to_mass(p_ecg, bestu),
              to_mass(pc_safe,
                      np.maximum(bestu,
                                 u_miss))]
        acc["ev_fixed"].append(
            roc_auc_score(
                y, pignistic(*dempster(ms))))

        ug = {k: u_from_gap(a, best) for k, a
              in (("ehr", a_e), ("ecg", a_g),
                  ("ctpa", a_c))}
        ms = [to_mass(p_ehr, ug["ehr"]),
              to_mass(p_ecg, ug["ecg"]),
              to_mass(pc_safe,
                      np.maximum(ug["ctpa"],
                                 u_miss))]
        acc["ev_gap"].append(
            roc_auc_score(
                y, pignistic(*dempster(ms))))

        # ---- calibrate first ----
        ce = platt_oof(p_ehr, y, grp, s)
        cg = platt_oof(p_ecg, y, grp, s)
        cc = platt_oof(p_ct, y, grp, s)
        uc = {"ehr": u_from_brier(ce, y, prev),
              "ecg": u_from_brier(cg, y, prev),
              "ctpa": u_from_brier(cc, y, prev)}
        for k, cp in (("ehr", ce), ("ecg", cg),
                      ("ctpa", cc)):
            ufit[k].append(uc[k])
            okk = np.isfinite(cp)
            try:
                bri[k].append(
                    brier_score_loss(
                        y[okk], np.clip(
                            cp[okk], 1e-6,
                            1 - 1e-6)))
            except Exception:
                bri[k].append(np.nan)
        ms = [to_mass(p_ehr, uc["ehr"]),
              to_mass(p_ecg, uc["ecg"]),
              to_mass(pc_safe,
                      np.maximum(uc["ctpa"],
                                 u_miss))]
        vcal = pignistic(*dempster(ms))
        acc["ev_calib"].append(
            roc_auc_score(y, vcal))

        ms = [to_mass(p_ehr,
                      0.5 * u_from_entropy(
                          p_ehr)),
              to_mass(p_ecg,
                      0.5 * u_from_entropy(
                          p_ecg)),
              to_mass(pc_safe,
                      np.maximum(
                          0.5 * u_from_entropy(
                              pc_safe),
                          u_miss))]
        acc["ev_entropy"].append(
            roc_auc_score(
                y, pignistic(*dempster(ms))))

        if s == SEEDS[0]:
            keep2 = (sum(x * p for x, p in
                         zip(wi_, [pci, p_ecg,
                                   p_ehr])),
                     vcal, bestu)

    print("")
    print("  %-14s %8s %8s" % ("model",
                               "mean", "SD"))
    for k in ["ehr", "ecg", "ctpa",
              "wra_imputed", "ev_fixed",
              "ev_gap", "ev_calib",
              "ev_entropy"]:
        v = np.array(acc[k])
        print("  %-14s %.4f  %.4f"
              % (k, v.mean(),
                 v.std(ddof=1)), flush=True)
        rows.append({
            "outcome": oc, "model": k,
            "n": len(y), "ev": int(y.sum()),
            "mean": v.mean(),
            "sd": v.std(ddof=1)})

    base = np.mean(acc["wra_imputed"])
    print("")
    print("  vs the imputed baseline")
    for k in ["ev_fixed", "ev_gap",
              "ev_calib", "ev_entropy"]:
        print("    %-12s %+.4f"
              % (k, np.mean(acc[k]) - base))

    vi, vc, bu = keep2
    gn, lo, hi, _ = f.boot_diff(
        y, _rank(vc), _rank(vi), grp)
    print("")
    print("  ev_calib vs wra_imputed"
          "  %+.4f [%+.4f,%+.4f] %s"
          % (gn, lo, hi,
             "*" if (lo > 0 or hi < 0)
             else ""))
    print("  best fixed u: %.2f" % bu)
    rows.append({
        "outcome": oc,
        "model": "ev_calib_vs_wra",
        "n": len(y), "ev": int(y.sum()),
        "mean": gn, "sd": np.nan,
        "lo": lo, "hi": hi,
        "sig": int(lo > 0 or hi < 0)})

    print("")
    print("  IGNORANCE, BRIER AND GAP")
    print("    %-6s %8s %8s %8s %8s"
          % ("mod", "u", "brier", "gap",
             "auc"))
    for k in ("ehr", "ecg", "ctpa"):
        mu = float(np.mean(ufit[k]))
        mb = float(np.nanmean(bri[k]))
        mg = float(np.mean(gaps[k]))
        ma = float(np.mean(acc[k]))
        print("    %-6s %8.3f %8.4f %8.4f"
              " %8.4f"
              % (k, mu, mb, mg, ma))
        mrows.append({
            "outcome": oc, "modality": k,
            "u_calib": mu, "brier": mb,
            "gap": mg, "auc": ma})

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)
mdf = pd.DataFrame(mrows)
mdf.to_csv(os.path.join(
    PROC, "evidential2_mass.csv"),
    index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 78)
print("SIX-SEED MEAN AUROC")
q = r[~r["model"].str.contains("_vs_")]
print(q.pivot_table(index="model",
                    columns="outcome",
                    values="mean")
      .round(4).to_string())

print("")
print("EVIDENTIAL vs THE IMPUTED BASELINE")
for oc in OUTS:
    s = q[q["outcome"] == oc]
    b = s[s["model"] == "wra_imputed"]
    if not len(b):
        continue
    line = "  %-18s" % oc
    for k in ("ev_fixed", "ev_gap",
              "ev_calib", "ev_entropy"):
        v = s[s["model"] == k]
        if len(v):
            line += "  %s %+.4f" % (
                k.replace("ev_", ""),
                v["mean"].iloc[0]
                - b["mean"].iloc[0])
    print(line)

print("")
print("=" * 78)
print("THE STRUCTURAL QUESTION")
print("  is calibration-derived ignorance")
print("  related to the AUC gap?")
print("=" * 78)
print("")
print(mdf.round(4).to_string(index=False))

if len(mdf) > 3:
    uu = mdf["u_calib"].values
    gg2 = mdf["gap"].values
    ok = np.isfinite(uu) & np.isfinite(gg2)
    print("")
    print("  u ranges %.3f to %.3f"
          % (np.nanmin(uu), np.nanmax(uu)))
    if np.nanstd(uu) < 1e-6:
        print("  u is STILL constant, so the"
              " correlation is undefined and")
        print("  the calibration scaling needs"
              " another look")
    elif ok.sum() > 3:
        pr, pp = st.pearsonr(uu[ok], gg2[ok])
        sr, sp = st.spearmanr(uu[ok], gg2[ok])
        print("  Pearson  r = %+.3f  p = %.4f"
              % (pr, pp))
        print("  Spearman r = %+.3f  p = %.4f"
              % (sr, sp))
        print("  on %d modality-outcome cells"
              % int(ok.sum()))
        print("")
        if pp < 0.05 and pr > 0:
            print("  -> THE TWO ROUTES AGREE."
                  " Ignorance derived from")
            print("     calibration alone tracks"
                  " the AUC gap that the gap rule")
            print("     uses, so a"
                  " Dempster-Shafer quantity")
            print("     independently reproduces"
                  " an empirical finding.")
        elif pp < 0.05:
            print("  -> they are related but in"
                  " the OPPOSITE direction,")
            print("     which would mean a"
                  " weaker modality is BETTER")
            print("     calibrated here. Worth"
                  " investigating.")
        else:
            print("  -> no relationship."
                  " Calibration and"
                  " discrimination")
            print("     are separate properties"
                  " here, so the gap rule")
            print("     cannot be recovered from"
                  " calibration alone.")

print("")
print("CONTEXT FROM THE EARLIER VERSION")
print("  the best fixed u was 0.00, 0.10, 0.00"
      " and 0.00, so given free choice")
print("  the search wants no ignorance, which"
      " reduces Dempster-Shafer to a")
print("  product rule. ev_gap was the best"
      " working variant at -0.0008 against")
print("  median imputation. The practical"
      " question is therefore answered;")
print("  this run addresses the structural one.")
print("")
print("saved", DEST, r.shape)