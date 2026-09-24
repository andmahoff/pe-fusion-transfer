"""Rotation forest parameters: tree count crossed
with leaf size, then the parameters of each
rotation.

In the partition form of a rotation forest, the
subset size s fixes the number of rotations (p/s)
and coverage (100%), so the subset grid in script
111 confounded rotation count with rotation size.

STAGE A  tree count crossed with leaf size.
         Script 111 measured every leaf size at
         100 trees; small leaves make
         high-variance trees, which more trees
         can average out.
STAGE B  the unlocked form, where each tree draws
         a set number of subsets of a set size
         and undrawn features pass through
         unrotated:
           n_rot  rotations per tree
           size   features per rotation; a
                  rotation must span both
                  modality blocks to form a
                  cross-modal direction
           cov    fraction of features rotated
           boot   bootstrap fraction for the PCA
                  fit, 0.75 in script 111
         n_rot and size are crossed; cov and boot
         are swept at the winner.

Tuned on death_30d_inhosp and validated on
death_30d, to show whether the setting transfers.
Ten seeds throughout.

Run in venv (analysis).

OUTPUT FILES
  data\\processed\\rot_params.csv
  results\\rot_params_log.txt
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as sla
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss

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

PROC, FIG = f.PROC, f.FIG
SEEDS = [42, 7, 13, 1, 2, 3, 5, 8, 21, 99]
NFOLD = 5
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CCA_REG = 0.1
CT_LO, CT_HI = -48.0, 24.0

TUNE_ON = "death_30d_inhosp"
VALIDATE_ON = ["death_30d"]

# stage A: the interaction with a mechanism
A_LEAF = [1, 2, 5, 10, 25]
A_NTREE = [100, 300, 600, 1200]
# stage B1: CROSSED, the pair that may interact
B_NROT = [3, 5, 10, 20]
B_SIZE = [2, 4, 8, 16]
# stage B2 and B3: at the B1 winner
B_COV = [0.25, 0.50, 0.75, 1.00]
B_BOOT = [0.40, 0.60, 0.75, 1.00]
B_SUBSET = [2, 3, 4, 8]
B_BASE = dict(cov=1.00, boot=0.75)
NTREE_B = 300

# depth and k from script 111 stage 2
DEPTH_BY = {"death_30d_inhosp": 6,
            "death_30d": 12,
            "composite_30d": 12,
            "cv_first": 12}
K_BY_OUT = {"death_30d_inhosp": 15,
            "death_30d": 10,
            "composite_30d": 3,
            "cv_first": 10}
DEST = os.path.join(PROC, "rot_params.csv")
# scripts 110 and 111, typed in
PREV = {"death_30d_inhosp":
        {"late": 0.8661, "s110": 0.9182,
         "s111": 0.9046},
        "death_30d":
        {"late": 0.8839, "s110": 0.9048,
         "s111": 0.9036}}


def _rank(p):
    return pd.Series(p).rank(pct=True).values


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


def rcca(X, Y, reg, k):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    n = len(X)
    p, q = X.shape[1], Y.shape[1]
    Cxx = X.T @ X / n + reg * np.eye(p)
    Cyy = Y.T @ Y / n + reg * np.eye(q)
    try:
        Kx = sla.fractional_matrix_power(
            Cxx, -0.5).real
        Ky = sla.fractional_matrix_power(
            Cyy, -0.5).real
        U, s, Vt = np.linalg.svd(
            Kx @ (X.T @ Y / n) @ Ky,
            full_matrices=False)
    except Exception:
        return None, None
    k = int(min(k, len(s)))
    return Kx @ U[:, :k], Ky @ Vt[:k].T


def build(At, Ae, Bt, Be, k):
    Xt = np.column_stack([At, Bt])
    Xe = np.column_stack([Ae, Be])
    if k < 1:
        return Xt, Xe
    A_, B_ = rcca(At, Bt, CCA_REG, k)
    if A_ is None:
        return Xt, Xe
    ma, mb = At.mean(0), Bt.mean(0)
    return (np.column_stack([
        Xt, (At - ma) @ A_, (Bt - mb) @ B_]),
        np.column_stack([
            Xe, (Ae - ma) @ A_,
            (Be - mb) @ B_]))


# ---------- rotation forest, fast -----------
def _pca_block(S):
    """Eigenvectors of the small covariance,
    largest first. Far cheaper than sklearn's PCA
    for blocks of 2 to 32 columns."""
    S = S - S.mean(0)
    n = max(len(S) - 1, 1)
    try:
        w, V = np.linalg.eigh(S.T @ S / n)
        return V[:, ::-1]
    except Exception:
        return np.eye(S.shape[1])


def _make_blocks(X, rng, n_rot, size, cov,
                 boot):
    """n_rot None is the partition form: every
    covered feature belongs to exactly one block
    and the rotation replaces the originals.
    n_rot set gives SAMPLED blocks, which may
    overlap, so rotations are APPENDED and the raw
    features survive."""
    n, p = X.shape
    ncov = int(max(2, round(cov * p)))
    cols = rng.permutation(p)[:ncov]
    if n_rot is None:
        groups = [cols[i:i + size]
                  for i in range(0, len(cols),
                                 size)]
    else:
        groups = []
        for _ in range(n_rot):
            sz = int(min(size, len(cols)))
            groups.append(rng.choice(
                cols, sz, replace=False))
    nb = max(10, int(round(boot * n)))
    out = []
    for idx in groups:
        if len(idx) < 2:
            continue
        rows = rng.choice(n, nb, replace=True)
        out.append((idx, _pca_block(
            X[np.ix_(rows, idx)])))
    return out


def _apply(X, blocks, partition):
    """Blockwise, so the cost is the sum of block
    sizes squared rather than p squared. The old
    version built a dense 94x94 matrix and did a
    full X @ R for every tree."""
    if partition:
        Z = X.copy()
        for idx, C in blocks:
            Z[:, idx] = X[:, idx] @ C
        return Z
    parts = [X]
    for idx, C in blocks:
        parts.append(X[:, idx] @ C)
    return np.column_stack(parts)


class RotationForest:
    def __init__(self, n_estimators=300,
                 n_rot=None, size=8, cov=1.0,
                 boot=0.75, leaf=5, depth=12,
                 seed=42):
        self.n = n_estimators
        self.n_rot = n_rot
        self.size = size
        self.cov = cov
        self.boot = boot
        self.leaf = leaf
        self.depth = depth
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.part_ = self.n_rot is None
        self.bl_, self.tr_ = [], []
        for _ in range(self.n):
            bl = _make_blocks(
                X, rng, self.n_rot, self.size,
                self.cov, self.boot)
            Z = _apply(X, bl, self.part_)
            rows = rng.choice(n, n,
                              replace=True)
            t = DecisionTreeClassifier(
                max_depth=self.depth,
                min_samples_leaf=self.leaf,
                class_weight="balanced",
                random_state=int(
                    rng.integers(1e6)))
            t.fit(Z[rows], y[rows])
            self.bl_.append(bl)
            self.tr_.append(t)
        return self

    def predict_proba(self, X):
        out = np.zeros(len(X))
        for bl, t in zip(self.bl_, self.tr_):
            out += t.predict_proba(
                _apply(X, bl, self.part_))[:, 1]
        out /= len(self.tr_)
        return np.column_stack([1 - out, out])


def oof_rot(Xa, Xb, y, grp, seed, k, cfg):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        At, Ae = prep_fold(Xa[tr], Xa[te])
        Bt, Be = prep_fold(Xb[tr], Xb[te])
        Zt, Ze = build(At, Ae, Bt, Be, k)
        m = RotationForest(seed=seed, **cfg)
        m.fit(Zt, y[tr])
        p[te] = m.predict_proba(Ze)[:, 1]
    return p


def evaluate(Xa, Xb, y, grp, k, cfg,
             seeds=SEEDS):
    return np.array([roc_auc_score(
        y, oof_rot(Xa, Xb, y, grp, s, k, cfg))
        for s in seeds])


def oof_single(X, y, grp, seed):
    p = np.zeros(len(y))
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(
            np.zeros((len(y), 1)), y, grp):
        Xt, Xe = prep_fold(X[tr], X[te])
        best, bc = -1.0, 1.0
        for c in CS:
            try:
                m = LogisticRegression(
                    C=c, max_iter=3000)
                m.fit(Xt, y[tr])
                s = roc_auc_score(
                    y[tr],
                    m.predict_proba(Xt)[:, 1])
                if s > best:
                    best, bc = s, c
            except Exception:
                continue
        m = LogisticRegression(C=bc,
                               max_iter=3000)
        m.fit(Xt, y[tr])
        p[te] = m.predict_proba(Xe)[:, 1]
    return p


def late_from(pa, pb, y):
    pa, pb = _rank(pa), _rank(pb)
    bs, bv = -1.0, pa
    for w in np.arange(0, 1.001, 0.05):
        v = w * pb + (1 - w) * pa
        a = roc_auc_score(y, v)
        if a > bs:
            bs, bv = a, v
    return bv


def platt_oof(p, y, grp, seed=42):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(q / (1 - q)).reshape(-1, 1)
    out = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(
        n_splits=NFOLD, shuffle=True,
        random_state=seed)
    for tr, te in cv.split(x, y, grp):
        try:
            m = LogisticRegression(
                C=1e6, max_iter=2000)
            m.fit(x[tr], y[tr])
            out[te] = m.predict_proba(
                x[te])[:, 1]
        except Exception:
            out[te] = p[te]
    return out


def calib(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    try:
        b = brier_score_loss(y, p)
        x = np.log(p / (1 - p)).reshape(-1, 1)
        m = LogisticRegression(C=1e6,
                               max_iter=2000)
        m.fit(x, y)
        return b, float(m.coef_[0][0])
    except Exception:
        return np.nan, np.nan


# ---------------- cohort --------------------
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
print("cohort:", len(D), "  EHR %d   CTPA %d"
      % (len(EH), len(CT)))
print("  tuning on %s, validating on %s"
      % (TUNE_ON, ", ".join(VALIDATE_ON)),
      flush=True)

# ---------- TIMING PROBE ----------
print("")
print("TIMING PROBE")
print("  measured before the run, to estimate its length")
Xp = np.random.rand(1300, 94)
yp = (np.random.rand(1300) > 0.94).astype(int)
tt = []
for tag, cfg in (
        ("partition size2",
         dict(n_estimators=100, n_rot=None,
              size=2, leaf=1)),
        ("partition size8",
         dict(n_estimators=100, n_rot=None,
              size=8, leaf=5)),
        ("sampled 10x8",
         dict(n_estimators=100, n_rot=10,
              size=8, leaf=5)),
        ("sampled 20x16",
         dict(n_estimators=100, n_rot=20,
              size=16, leaf=25))):
    t = time.time()
    RotationForest(seed=42, **cfg).fit(Xp, yp)
    el = time.time() - t
    tt.append(el)
    print("  %-18s %6.2f s per 100 trees"
          % (tag, el), flush=True)
per_tree = float(np.mean(tt)) / 100.0
nA = len(A_LEAF) * sum(A_NTREE)
nB1 = len(B_NROT) * len(B_SIZE) * NTREE_B
nB2 = ((len(B_COV) + len(B_BOOT)
        + len(B_SUBSET)) * NTREE_B)
tune_trees = nA + nB1 + nB2
val_trees = NTREE_B * (1 + len(VALIDATE_ON))
tot = tune_trees + val_trees
est = (tot * per_tree * NFOLD
       * len(SEEDS) / 60.0)
print("")
print("  %d tree-fits per fold-seed"
      "   %.4f s each" % (tot, per_tree))
print("  PROJECTED TOTAL: %.0f minutes"
      " (%.1f hours)" % (est, est / 60))
sc_ = per_tree * NFOLD * len(SEEDS) / 60.0
print("")
print("  stage A %.0f min   B1 %.0f min"
      "   B2+B3 %.0f min   final %.0f min"
      % (nA * sc_, nB1 * sc_, nB2 * sc_,
         val_trees * sc_))
print("")
print("  Ctrl+C now if that is too long."
      " Results save after every stage, so")
print("  stopping later keeps what has run.")
time.sleep(5)

rows = []
t0 = time.time()

# ================= TUNING ===================
oc = TUNE_ON
d, y = f.labels(D, oc)
grp = d["subject_id"].values
Xa = d[EH].values.astype(float)
Xb = d[CT].values.astype(float)
dp = DEPTH_BY.get(oc, 12)
kk = K_BY_OUT.get(oc, 10)
pv = PREV.get(oc, {})
print("")
print("#" * 74)
print("TUNING ON %s   n=%d ev=%d"
      % (oc, len(y), int(y.sum())))
print("  depth=%s k=%d, from script 111"
      % (dp, kk))
pa = _rank(oof_single(Xa, y, grp, 42))
pb = _rank(oof_single(Xb, y, grp, 42))
ref = roc_auc_score(y, late_from(pa, pb, y))
print("  late %.4f   script 110 %.4f"
      "   script 111 %.4f"
      % (ref, pv.get("s110", np.nan),
         pv.get("s111", np.nan)), flush=True)
rows.append({"outcome": oc, "stage": "ref",
             "param": "late", "mean": ref,
             "sd": np.nan, "vs_late": 0.0})

# ---- STAGE A: tree count x leaf ----
tb = time.time()
print("")
print("  STAGE A: TREE COUNT x LEAF")
print("  script 111 found leaf 1 and 2 worst,"
      " but every value ran at 100 TREES.")
print("  Small leaves make high-variance trees,"
      " and averaging more is how a forest")
print("  absorbs that. If leaf 1 improves"
      " across the row, the earlier claim was")
print("  conditional on the tree count.")
print("  %-6s" % "leaf", end="")
for nt in A_NTREE:
    print(" %9s" % ("n=%d" % nt), end="")
print("")
bestA = None
for lf in A_LEAF:
    line = "  %-6d" % lf
    for nt in A_NTREE:
        cfg = dict(n_estimators=nt,
                   n_rot=None, size=3,
                   cov=1.0, boot=0.75,
                   leaf=lf, depth=dp)
        try:
            aa = evaluate(Xa, Xb, y, grp, kk,
                          cfg)
        except Exception:
            line += " %9s" % "fail"
            continue
        line += " %+9.4f" % (aa.mean() - ref)
        rows.append({
            "outcome": oc, "stage": "A",
            "param": "leaf", "value": lf,
            "ntree": nt, "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_late": aa.mean() - ref})
        if bestA is None or \
                aa.mean() > bestA[0]:
            bestA = (aa.mean(), lf, nt)
    print(line, flush=True)
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)
blf = bestA[1] if bestA else 5
bnt = bestA[2] if bestA else NTREE_B
if bestA:
    print("  best leaf=%d ntree=%d  %.4f"
          % (blf, bnt, bestA[0]))
print("  stage A: %.1f min"
      % ((time.time() - tb) / 60), flush=True)

# ---- STAGE B1: n_rot x size, CROSSED ----
tb = time.time()
print("")
print("  STAGE B1: n_rot x size, CROSSED"
      "  (leaf=%d)" % blf)
print("  small rotations may work best when"
      " there are many, large ones when few.")
print("  A one-at-a-time sweep sees neither.")
print("  %-7s" % "n_rot", end="")
for sz in B_SIZE:
    print(" %9s" % ("sz=%d" % sz), end="")
print("")
bestB = None
for nr in B_NROT:
    line = "  %-7d" % nr
    for sz in B_SIZE:
        cfg = dict(n_estimators=NTREE_B,
                   n_rot=nr, size=sz,
                   cov=B_BASE["cov"],
                   boot=B_BASE["boot"],
                   leaf=blf, depth=dp)
        try:
            aa = evaluate(Xa, Xb, y, grp, kk,
                          cfg)
        except Exception:
            line += " %9s" % "fail"
            continue
        line += " %+9.4f" % (aa.mean() - ref)
        rows.append({
            "outcome": oc, "stage": "B1",
            "param": "n_rot_x_size",
            "n_rot": nr, "size": sz,
            "ntree": NTREE_B,
            "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_late": aa.mean() - ref})
        if bestB is None or \
                aa.mean() > bestB[0]:
            bestB = (aa.mean(), nr, sz)
    print(line, flush=True)
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)
bnr = bestB[1] if bestB else 10
bsz = bestB[2] if bestB else 8
if bestB:
    print("  best n_rot=%d size=%d  %.4f"
          % (bnr, bsz, bestB[0]))
print("  stage B1: %.1f min"
      % ((time.time() - tb) / 60), flush=True)

# ---- STAGE B2: cov and boot ----
tb = time.time()
print("")
print("  STAGE B2: cov and boot at n_rot=%d"
      " size=%d leaf=%d" % (bnr, bsz, blf))
best2 = dict(B_BASE)
for pname, grid in (("cov", B_COV),
                    ("boot", B_BOOT)):
    print("")
    print("    %-6s %10s %8s %9s"
          % (pname, "AUC", "SD", "vs late"))
    bv, bm = B_BASE[pname], -1.0
    for v in grid:
        cfg = dict(n_estimators=NTREE_B,
                   n_rot=bnr, size=bsz,
                   cov=B_BASE["cov"],
                   boot=B_BASE["boot"],
                   leaf=blf, depth=dp)
        cfg[pname] = v
        try:
            aa = evaluate(Xa, Xb, y, grp, kk,
                          cfg)
        except Exception as exc:
            print("    %-6s FAILED %s"
                  % (str(v), repr(exc)[:35]))
            continue
        mark = ("  (base)"
                if v == B_BASE[pname] else "")
        print("    %-6s %10.4f %8.4f %+9.4f%s"
              % (str(v), aa.mean(),
                 aa.std(ddof=1),
                 aa.mean() - ref, mark),
              flush=True)
        rows.append({
            "outcome": oc, "stage": "B2",
            "param": pname, "value": v,
            "n_rot": bnr, "size": bsz,
            "ntree": NTREE_B,
            "mean": aa.mean(),
            "sd": aa.std(ddof=1),
            "vs_late": aa.mean() - ref})
        if aa.mean() > bm:
            bm, bv = aa.mean(), v
    best2[pname] = bv
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)
print("  stage B2: %.1f min"
      % ((time.time() - tb) / 60), flush=True)

# ---- STAGE B3: the partition form ----
tb = time.time()
print("")
print("  STAGE B3: SUBSET, the partition form")
print("  script 111 suggested small values"
      " without resolving between them")
print("    %-8s %10s %8s %9s"
      % ("subset", "AUC", "SD", "vs late"))
bpart, bpm = None, -1.0
for sb in B_SUBSET:
    cfg = dict(n_estimators=NTREE_B,
               n_rot=None, size=sb, cov=1.0,
               boot=best2["boot"], leaf=blf,
               depth=dp)
    try:
        aa = evaluate(Xa, Xb, y, grp, kk, cfg)
    except Exception:
        continue
    print("    %-8d %10.4f %8.4f %+9.4f"
          % (sb, aa.mean(), aa.std(ddof=1),
             aa.mean() - ref), flush=True)
    rows.append({
        "outcome": oc, "stage": "B3",
        "param": "subset", "value": sb,
        "ntree": NTREE_B, "mean": aa.mean(),
        "sd": aa.std(ddof=1),
        "vs_late": aa.mean() - ref})
    if aa.mean() > bpm:
        bpm, bpart = aa.mean(), sb
pd.DataFrame(rows).to_csv(DEST, index=False)
print("  stage B3: %.1f min"
      % ((time.time() - tb) / 60), flush=True)

# pick the better of sampled and partition
use_part = (bpm > (bestB[0] if bestB
                   else -1.0))
if use_part:
    FINAL = dict(n_estimators=bnt,
                 n_rot=None, size=int(bpart),
                 cov=1.0,
                 boot=float(best2["boot"]))
    desc = ("partition subset=%d boot=%.2f"
            % (bpart, best2["boot"]))
else:
    FINAL = dict(n_estimators=bnt,
                 n_rot=int(bnr),
                 size=int(bsz),
                 cov=float(best2["cov"]),
                 boot=float(best2["boot"]))
    desc = ("sampled n_rot=%d size=%d"
            " cov=%.2f boot=%.2f"
            % (bnr, bsz, best2["cov"],
               best2["boot"]))
print("")
print("  TUNED: %s, leaf=%d, ntree=%d"
      % (desc, blf, bnt), flush=True)

# ================ VALIDATION ================
print("")
print("#" * 74)
print("VALIDATION: the tuned setting on an"
      " outcome it was not tuned on")
print("  a setting that only works where it was"
      " chosen was fitted to that cell")
print("#" * 74, flush=True)

for voc in [TUNE_ON] + VALIDATE_ON:
    if voc not in D.columns:
        continue
    dv, yv = f.labels(D, voc)
    gv = dv["subject_id"].values
    if yv.sum() < 25:
        continue
    Av = dv[EH].values.astype(float)
    Bv = dv[CT].values.astype(float)
    kv = K_BY_OUT.get(voc, 10)
    pvv = PREV.get(voc, {})
    if voc == TUNE_ON:
        rv = ref
    else:
        pav = _rank(oof_single(Av, yv, gv, 42))
        pbv = _rank(oof_single(Bv, yv, gv, 42))
        rv = roc_auc_score(
            yv, late_from(pav, pbv, yv))
    cfg = dict(FINAL)
    cfg.update(leaf=blf,
               depth=DEPTH_BY.get(voc, 12))
    aa, sl, pl = [], [], []
    for s in SEEDS:
        p = oof_rot(Av, Bv, yv, gv, s, kv, cfg)
        aa.append(roc_auc_score(yv, p))
        _, s_ = calib(p, yv)
        sl.append(s_)
        pc = platt_oof(p, yv, gv, s)
        ok = np.isfinite(pc)
        pl.append(calib(pc[ok], yv[ok])[1])
    aa = np.array(aa)
    tag = ("TUNED HERE" if voc == TUNE_ON
           else "held out")
    print("")
    print("  %-18s %s" % (voc, tag))
    print("    %.4f (SD %.4f, range"
          " %.4f-%.4f)   late %.4f   %+.4f"
          % (aa.mean(), aa.std(ddof=1),
             aa.min(), aa.max(), rv,
             aa.mean() - rv))
    print("    slope %.3f -> %.3f after Platt"
          "   script 110 %.4f   111 %.4f"
          % (np.nanmean(sl), np.nanmean(pl),
             pvv.get("s110", np.nan),
             pvv.get("s111", np.nan)),
          flush=True)
    rows.append({
        "outcome": voc, "stage": "final",
        "param": ("tuned" if voc == TUNE_ON
                  else "heldout"),
        "value": blf, "ntree": bnt,
        "n_rot": FINAL.get("n_rot"),
        "size": FINAL.get("size"),
        "cov": FINAL.get("cov"),
        "boot": FINAL.get("boot"),
        "mean": aa.mean(),
        "sd": aa.std(ddof=1), "lo": aa.min(),
        "hi": aa.max(),
        "slope": np.nanmean(sl),
        "slope_platt": np.nanmean(pl),
        "vs_late": aa.mean() - rv})
    pd.DataFrame(rows).to_csv(DEST,
                              index=False)

r = pd.DataFrame(rows)
r.to_csv(DEST, index=False)

print("")
print("elapsed %.1f min"
      % ((time.time() - t0) / 60))
print("")
print("=" * 74)
print("STAGE A: TREE COUNT x LEAF")
a = r[r["stage"] == "A"]
if len(a):
    print(a.pivot_table(index="value",
                        columns="ntree",
                        values="vs_late")
          .round(4).to_string())
    print("")
    print("  marginal by tree count")
    print(a.groupby("ntree")["vs_late"].mean()
          .round(4).to_string())
    print("  marginal by leaf")
    print(a.groupby("value")["vs_late"].mean()
          .round(4).to_string())
    print("")
    print("  DOES MORE TREES RESCUE SMALL"
          " LEAVES?")
    piv = a.pivot_table(index="value",
                        columns="ntree",
                        values="mean")
    for lf_ in piv.index:
        row = piv.loc[lf_]
        print("    leaf=%-3d  %.4f at %d ->"
              " %.4f at %d   %+.4f"
              % (lf_, row.iloc[0],
                 int(piv.columns[0]),
                 row.iloc[-1],
                 int(piv.columns[-1]),
                 row.iloc[-1] - row.iloc[0]))
    print("")
    print("    a larger rise for small leaves"
          " means the earlier 'leaf 1 and 2")
    print("    are worse' was conditional on"
          " running at 100 trees")
    rg = a["mean"].max() - a["mean"].min()
    print("")
    print("  range %.4f   mean seed SD %.4f"
          "   ratio %.1f"
          % (rg, a["sd"].mean(),
             rg / max(a["sd"].mean(), 1e-9)))
    print("  a ratio below about 2 means the"
          " whole grid is inside seed noise")

print("")
print("STAGE B1: n_rot x size")
b = r[r["stage"] == "B1"]
if len(b):
    print(b.pivot_table(index="n_rot",
                        columns="size",
                        values="vs_late")
          .round(4).to_string())
    print("")
    print("  marginal by n_rot")
    print(b.groupby("n_rot")["vs_late"].mean()
          .round(4).to_string())
    print("  marginal by size")
    print(b.groupby("size")["vs_late"].mean()
          .round(4).to_string())
    rg = b["mean"].max() - b["mean"].min()
    print("")
    print("  range %.4f   mean seed SD %.4f"
          "   ratio %.1f"
          % (rg, b["sd"].mean(),
             rg / max(b["sd"].mean(), 1e-9)))
    print("")
    print("  IS THERE AN INTERACTION?")
    print("  the same best size at every n_rot"
          " means the two are separable")
    piv = b.pivot_table(index="n_rot",
                        columns="size",
                        values="mean")
    for nr in piv.index:
        print("    n_rot=%-3d best size %d"
              % (nr, piv.loc[nr].idxmax()))

print("")
print("STAGE B2 and B3")
for stg, names in (("B2", ("cov", "boot")),
                   ("B3", ("subset",))):
    z0 = r[r["stage"] == stg]
    for pname in names:
        z = z0[z0["param"] == pname]
        if not len(z):
            continue
        rg = z["mean"].max() - z["mean"].min()
        print("  %-7s range %.4f   mean seed SD"
              " %.4f   ratio %.1f   best %s"
              % (pname, rg, z["sd"].mean(),
                 rg / max(z["sd"].mean(), 1e-9),
                 str(z.loc[z["mean"].idxmax(),
                           "value"])))

print("")
print("=" * 74)
print("DOES THE TUNED SETTING TRANSFER?")
fz = r[r["stage"] == "final"]
if len(fz):
    print(fz[["outcome", "param", "ntree",
              "n_rot", "size", "cov", "boot",
              "mean", "sd", "vs_late",
              "slope_platt"]].round(4)
          .to_string(index=False))
    t_ = fz[fz["param"] == "tuned"]
    h_ = fz[fz["param"] == "heldout"]
    if len(t_) and len(h_):
        print("")
        print("  tuned cell %+.4f over late;"
              " held out %+.4f"
              % (t_["vs_late"].iloc[0],
                 h_["vs_late"].mean()))
        if h_["vs_late"].mean() > 0:
            print("  -> it transfers, so the"
                  " setting is not fitted to")
            print("     the cell it was chosen"
                  " on")
        else:
            print("  -> it does not transfer,"
                  " so the tuning found")
            print("     something specific to"
                  " one cell")

print("")
print("AGAINST THE EARLIER RUNS")
print("  script 110 used the default subset of"
      " 3 regardless of its printed mtry;")
print("  script 111 selected from 35 cells at"
      " two seeds. Everything here uses ten")
print("  seeds.")
print("")
print("saved", DEST, r.shape)
keep_awake(False)