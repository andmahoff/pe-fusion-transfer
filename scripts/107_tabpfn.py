"""TabPFN on the raw74 and pca8 EHR+CTPA
representations.

Run separately from script 106 because each
prediction call takes about a minute on CPU, so
two representations across four outcomes take
roughly 3.5 hours. raw74 is where the tree
ensembles did best in script 105, and pca8 is
where the L2 head did best.

TabPFN is pretrained and fits nothing at
prediction time: the training rows are passed as
context and predictions come from one forward
pass.

REQUIRES, in the same terminal that runs this:
  $env:TABPFN_TOKEN="your-key"
  $env:TABPFN_NO_BROWSER="1"
TABPFN_NO_BROWSER skips the interactive login,
which fails on Windows (WinError 10038).

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\tabpfn_results.csv
  results\\tabpfn_log.txt
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import fusion_lib as f


def keep_awake(on=True):
    try:
        import ctypes
        flag = (0x80000000 | 0x00000001
                if on else 0x80000000)
        ctypes.windll.kernel32 \
            .SetThreadExecutionState(flag)
        return True
    except Exception:
        return False


import atexit
if keep_awake(True):
    print("sleep suppressed for this process")
atexit.register(keep_awake, False)

if not os.environ.get("TABPFN_TOKEN"):
    print("")
    print("TABPFN_TOKEN is not set in this"
          " terminal. Set it and rerun:")
    print('  $env:TABPFN_TOKEN="your-key"')
    print('  $env:TABPFN_NO_BROWSER="1"')
    sys.exit(1)
os.environ.setdefault("TABPFN_NO_BROWSER", "1")

try:
    from tabpfn import TabPFNClassifier
except Exception as exc:
    print("tabpfn unavailable:",
          repr(exc)[:80])
    sys.exit(1)

PROC, FIG = f.PROC, f.FIG
SEEDS = [42, 7, 13, 1, 2]
NFOLD = 5
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
PCA_K = 8
CT_LO, CT_HI = -48.0, 24.0
OUTS = ["death_30d_inhosp", "death_30d",
        "composite_30d", "cv_first"]
DEST = os.path.join(PROC,
                    "tabpfn_results.csv")
# script 105, five seeds, for comparison
REF = {"death_30d_inhosp":
       {"late": 0.8834, "rf_raw": 0.9066,
        "l2_pca": 0.9015},
       "death_30d":
       {"late": 0.8823, "rf_raw": 0.8875,
        "l2_pca": 0.8796},
       "composite_30d":
       {"late": 0.8423, "rf_raw": 0.8468,
        "l2_pca": 0.8331},
       "cv_first":
       {"late": 0.8074, "rf_raw": 0.7979,
        "l2_pca": 0.7503}}


def clean_fit(A):
    A = np.asarray(A, dtype=float).copy()
    med = np.nanmedian(A, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(A)
    if bad.any():
        ii = np.where(bad)
        A[ii] = np.take(med, ii[1])
    return A, med


def clean_apply(A, med):
    A = np.asarray(A, dtype=float).copy()
    bad = ~np.isfinite(A)
    if bad.any():
        ii = np.where(bad)
        A[ii] = np.take(med, ii[1])
    return A


def prep_fold(Xtr, Xte):
    Xtr, md = clean_fit(Xtr)
    Xte = clean_apply(Xte, md)
    sc = StandardScaler()
    return sc.fit_transform(Xtr), \
        sc.transform(Xte)


def oof_pfn(Xa, Xb, y, grp, seed, use_pca,
            log_every=True):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for i, (tr, te) in enumerate(cv.split(
            np.zeros((len(y), 1)), y, grp)):
        tf = time.time()
        At, Ae = prep_fold(Xa[tr], Xa[te])
        Bt, Be = prep_fold(Xb[tr], Xb[te])
        Zt = np.column_stack([At, Bt])
        Ze = np.column_stack([Ae, Be])
        if use_pca:
            k = int(min(PCA_K,
                        Zt.shape[1] - 1,
                        len(tr) - 1))
            pc = PCA(n_components=k,
                     random_state=42)
            Zt2 = pc.fit_transform(Zt)
            Ze2 = pc.transform(Ze)
            Zt, Ze = Zt2, Ze2
        m = TabPFNClassifier(device="cpu")
        m.fit(Zt, y[tr])
        p[te] = m.predict_proba(Ze)[:, 1]
        if log_every:
            print("      fold %d/%d  %.0f s"
                  % (i + 1, NFOLD,
                     time.time() - tf),
                  flush=True)
    return p


print("")
print("TabPFN alone. Measured at 61.5 s per"
      " call on CPU, so expect")
print("about 5 minutes per seed and 25 minutes"
      " per representation-outcome cell.")

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
CT = [c for c in f.CTPA46 if c in mm.columns]
lab = pd.read_csv(os.path.join(
    FIG, "mimic_labels_harmonised.csv"))
D = mm[["subject_id", "hadm_id"] + CT].merge(
    mim_ehr[["hadm_id"] + EH], on="hadm_id",
    how="inner").merge(
    lab, on=["subject_id", "hadm_id"],
    how="inner")
print("")
print("cohort:", len(D))
print("  EHR %d   CTPA %d   total %d"
      % (len(EH), len(CT), len(EH) + len(CT)),
      flush=True)

rows = []
t0 = time.time()

for oc in OUTS:
    if oc not in D.columns:
        continue
    d, y = f.labels(D, oc)
    grp = d["subject_id"].values
    if y.sum() < 25:
        continue
    Xa = d[EH].values.astype(float)
    Xb = d[CT].values.astype(float)
    tb = time.time()
    rf_ = REF.get(oc, {})
    print("")
    print("#" * 74)
    print("%s   n=%d ev=%d" % (oc, len(y),
                               int(y.sum())))
    if rf_:
        print("  to beat: late %.4f   rf/raw"
              " %.4f   l2/pca %.4f"
              % (rf_["late"], rf_["rf_raw"],
                 rf_["l2_pca"]), flush=True)

    for use_pca, rname in ((False, "raw74"),
                           (True,
                            "pca%d" % PCA_K)):
        print("")
        print("    %s" % rname, flush=True)
        aa = []
        for s in SEEDS:
            print("     seed %d" % s,
                  flush=True)
            try:
                p = oof_pfn(Xa, Xb, y, grp,
                            s, use_pca)
                aa.append(
                    roc_auc_score(y, p))
                print("      seed AUC %.4f"
                      % aa[-1], flush=True)
            except Exception as exc:
                print("      FAILED %s"
                      % repr(exc)[:60],
                      flush=True)
        if not aa:
            continue
        v = np.array(aa)
        line = ("    %-8s tabpfn  %.4f"
                " (SD %.4f)"
                % (rname, v.mean(),
                   v.std(ddof=1)))
        if rf_:
            line += ("   vs late %+.4f"
                     % (v.mean()
                        - rf_["late"]))
        print(line, flush=True)
        rows.append({
            "outcome": oc, "repr": rname,
            "learner": "tabpfn",
            "mean": v.mean(),
            "sd": v.std(ddof=1),
            "n_seeds": len(v),
            "late": rf_.get("late", np.nan),
            "rf_raw": rf_.get("rf_raw",
                              np.nan),
            "l2_pca": rf_.get("l2_pca",
                              np.nan),
            "vs_late": (v.mean()
                        - rf_["late"]
                        if rf_ else np.nan)})
        pd.DataFrame(rows).to_csv(DEST,
                                  index=False)

    print("")
    print("  %s done in %.1f min  (saved)"
          % (oc, (time.time() - tb) / 60),
          flush=True)

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("TabPFN AGAINST THE BEST OF EACH"
      " FAMILY")
print("  %-18s %-7s %8s %8s %8s %8s"
      % ("outcome", "repr", "tabpfn", "late",
         "rf/raw", "l2/pca"))
for _, x in r.iterrows():
    print("  %-18s %-7s %8.4f %8.4f %8.4f"
          " %8.4f"
          % (x["outcome"], x["repr"],
             x["mean"], x["late"],
             x["rf_raw"], x["l2_pca"]))

print("")
print("DOES TabPFN BEAT THEM?")
for _, x in r.iterrows():
    best_other = max(x["late"], x["rf_raw"],
                     x["l2_pca"])
    print("  %-18s %-7s %+.4f against the"
          " best of the other three (%.4f)"
          % (x["outcome"], x["repr"],
             x["mean"] - best_other,
             best_other))

print("")
print("RAW vs PCA FOR TabPFN")
print("  script 105 found PCA helping L2 and"
      " hurting every tree. A model that")
print("  handles correlated features natively"
      " should gain nothing from it.")
for oc in OUTS:
    s = r[r["outcome"] == oc]
    a = s[s["repr"] == "raw74"]
    b = s[s["repr"] == "pca%d" % PCA_K]
    if len(a) and len(b):
        print("  %-18s raw %.4f   pca %.4f"
              "   %+.4f"
              % (oc, a["mean"].iloc[0],
                 b["mean"].iloc[0],
                 b["mean"].iloc[0]
                 - a["mean"].iloc[0]))
print("")
print("saved", DEST, r.shape)
keep_awake(False)