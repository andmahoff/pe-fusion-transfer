"""Shared module for the bidirectional INSPECT/MIMIC
fusion project. Loads and canonicalises the four
feature matrices, builds the cohort for each transfer
direction, and defines the early, intermediate and
late fusion architectures.

REGULARISATION. The dissertation fixed C=0.1 for the
EHR modality and cross-validated C for CTPA. Applying
one fixed C to both would handicap one modality and
contaminate the gap measure, which is the project's
headline quantity. LEARNER_MODE controls this:
  "tuned"  - both modalities tune C on SOURCE cross-
             validation (primary; no target labels)
  "fixed"  - EHR C=0.1, CTPA cross-validated
             (dissertation configuration, sensitivity)

COHORTS. load_mimic() inner-joins the 38-feature CTPA
matrix, giving the 1,649-admission fusion cohort used
throughout the grid. load_mimic_ehr() returns the EHR
modality WITHOUT that join, for reproductions that use
a different CTPA cohort.
Imported by later scripts; not run directly.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# folder holding the dissertation's derived files
DIS = os.environ.get(
    "PE_DISS_DIR", os.path.join("..", "Dissertation"))
FIG = os.path.join(DIS, "figwork", "data")
DSET = os.path.join(DIS, "Datasets")
RAW = os.path.join("data", "raw", "inspect")
PROC = os.path.join("data", "processed")

SEEDS = [42, 7, 13]
NBOOT = 2000
PCA_VAR = 0.95

LAT = 16
RANK = 4
TUCKER_CORE = 8

LEARNER_MODE = "tuned"
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1,
          0.3, 1.0, 3.0, 10.0]
C_FIXED_EHR = 0.1
TUNE_FOLDS = 3
TUNE_SEED = 42

TA = ["temp", "hr", "sbp", "dbp", "rr",
      "creatinine", "sodium", "potassium",
      "bun", "glucose", "calcium", "bicarb",
      "chloride", "hct", "plt", "wbc", "hgb",
      "aniongap", "rbc", "mchc", "mch", "mcv",
      "rdw"]
FLAGS = ["afib", "cancer", "copd",
         "heart_failure"]
EHR_COLS = TA + FLAGS + ["age"]

BASE_C = ["pe_pos", "saddle", "central", "lobar",
          "segmental", "subseg", "bilateral",
          "rv_strain", "septal_bow", "reflux",
          "mpa_enlarge", "infarct", "effusion",
          "malignancy", "consolid", "atelect",
          "edema", "cardiomeg", "adenopathy",
          "pe_neg", "txt_len", "n_sent"]
COMORB_C = ["cm_emphysema", "cm_fibrosis",
            "cm_bronchiect", "cm_mets",
            "cm_lymphangitic", "cm_cirrhosis",
            "cm_ascites", "cm_pericard_eff",
            "cm_aortic_ath", "cm_aneurysm",
            "cm_valve", "cm_cachexia",
            "cm_obesity", "cm_renal",
            "cm_pleural_thick", "cm_vert_fx"]
CTPA_COLS = BASE_C + COMORB_C
DEVICE_C = ["dv_ett", "dv_trach", "dv_cvc",
            "dv_pacer", "dv_sternotomy",
            "dv_chesttube", "dv_ngtube",
            "dv_ivcfilter"]
CTPA46 = CTPA_COLS + DEVICE_C

OUTCOMES = ["death_30d", "composite_30d",
            "cv_first"]

EARLY_VARIANTS = ["plain", "blockstd", "pca"]
INTER_VARIANTS = ["plain", "gated", "moddrop",
                  "lowrank", "bilinear", "tucker",
                  "crossattn"]
LATE_VARIANTS = ["mean", "wsrc"]


# ---------- loaders ----------
def load_inspect():
    """INSPECT: person-level, 3300 PE-cohort
    persons with both modalities."""
    e = pd.read_csv(os.path.join(
        FIG, "inspect_expanded_feats.csv"))
    lab = pd.read_csv(os.path.join(
        FIG, "inspect_labels_final.csv"))
    fl = pd.read_csv(os.path.join(
        RAW, "inspect_flags.csv"))
    per = pd.read_csv(
        os.path.join(DSET, "INSPECT_EHR",
                     "person.csv"),
        usecols=["person_id", "year_of_birth"])
    ct = pd.read_csv(os.path.join(
        PROC, "inspect_imp_features_v2.csv"))

    ren = {"m7_" + a: a for a in TA}
    e = e.rename(columns=ren)
    e = e[["person_id"] + [a for a in TA
                           if a in e.columns]]

    d = e.merge(lab[["person_id", "pe_date",
                     "death_30d",
                     "composite_30d", "cv_first",
                     "death_first"]],
                on="person_id", how="left")
    d = d.merge(fl, on="person_id", how="left")
    for f_ in FLAGS:
        d[f_] = d[f_].fillna(0).astype(int)
    d = d.merge(per, on="person_id", how="left")
    yr = pd.to_datetime(d["pe_date"],
                        errors="coerce").dt.year
    d["age"] = yr - d["year_of_birth"]
    d.loc[(d["age"] < 18) | (d["age"] > 100),
          "age"] = np.nan

    keep = ["person_id"] + CTPA_COLS
    d = d.merge(ct[keep], on="person_id",
                how="inner")
    d["gid"] = d["person_id"]
    return d


def load_mimic_ehr():
    """MIMIC EHR modality with labels, WITHOUT
    any CTPA join. Use this when the CTPA cohort
    is defined elsewhere."""
    e = pd.read_csv(os.path.join(
        FIG, "mimic_expanded_feats.csv"))
    v = pd.read_csv(os.path.join(
        PROC, "mimic_vitals_rebuilt.csv"))
    v = v.drop(columns=["subject_id"])
    v.columns = ["hadm_id"] + [
        c.replace("mean_", "")
        for c in v.columns[1:]]
    e = e.merge(v, on="hadm_id", how="left")

    ren = {"mi_" + a: a for a in TA}
    e = e.rename(columns=ren)

    fl = pd.read_csv(os.path.join(
        FIG, "mimic_condition_flags_timed.csv"))
    fl = fl[["hadm_id"] + [f_ + "_prior_idx"
                           for f_ in FLAGS]]
    fl = fl.drop_duplicates("hadm_id")
    fl.columns = ["hadm_id"] + FLAGS
    e = e.merge(fl, on="hadm_id", how="left")
    for f_ in FLAGS:
        e[f_] = e[f_].fillna(0).astype(int)

    coh = pd.read_csv(
        os.path.join(DSET, "MIMICIV",
                     "mimic_pe_mace_cohort.csv"),
        usecols=["hadm_id", "age_at_admit"])
    coh = coh.dropna().drop_duplicates("hadm_id")
    coh = coh.rename(
        columns={"age_at_admit": "age"})
    e = e.merge(coh, on="hadm_id", how="left")

    lab = pd.read_csv(os.path.join(
        FIG, "mimic_labels_harmonised.csv"))
    d = e.merge(lab, on=["subject_id", "hadm_id"],
                how="inner")
    d["gid"] = d["subject_id"]
    return d


def load_mimic():
    """MIMIC restricted to the 38-feature CTPA
    fusion cohort (1,649 admissions)."""
    d = load_mimic_ehr()
    ct = pd.read_csv(os.path.join(
        FIG, "mimic_imp_features_v2.csv"))
    ct = ct[["hadm_id"] + CTPA_COLS]
    return d.merge(ct, on="hadm_id", how="inner")


def load_mimic_ctpa46():
    """MIMIC EHR joined to the 46-feature CTPA
    set, including the eight device flags. Used
    for the dissertation reproduction."""
    d = load_mimic_ehr()
    cb = pd.read_csv(os.path.join(
        FIG, "ctpa_comorb_features.csv"))
    keep = ["hadm_id"] + [c for c in CTPA46
                          if c in cb.columns]
    cb = cb[keep].drop_duplicates("hadm_id")
    return d.merge(cb, on="hadm_id", how="inner")


def blocks(df):
    """Return the two feature blocks as arrays."""
    xe = df[EHR_COLS].values.astype(float)
    xc = df[CTPA_COLS].values.astype(float)
    return xe, xc


def labels(df, outcome):
    d = df
    if outcome == "cv_first":
        d = df[df["death_first"] == 0]
    y = pd.to_numeric(d[outcome],
                      errors="coerce")
    y = y.fillna(0).astype(int).values
    return d, y


# ---------- preprocessing ----------
def prep(src_X, tgt_X, mode="std"):
    """Impute on source, then scale.
    mode: std | blockstd | quantile"""
    im = SimpleImputer(strategy="median")
    a = im.fit_transform(src_X)
    b = im.transform(tgt_X)

    if mode == "quantile":
        qa = QuantileTransformer(
            n_quantiles=min(1000, a.shape[0]),
            output_distribution="normal",
            random_state=TUNE_SEED)
        qb = QuantileTransformer(
            n_quantiles=min(1000, b.shape[0]),
            output_distribution="normal",
            random_state=TUNE_SEED)
        return qa.fit_transform(a), \
            qb.fit_transform(b)

    sc = StandardScaler()
    a2 = sc.fit_transform(a)
    b2 = sc.transform(b)
    if mode == "blockstd":
        k = np.sqrt(a2.shape[1])
        a2, b2 = a2 / k, b2 / k
    return a2, b2


# ---------- learners ----------
def tune_c(X, y, grid=None, folds=TUNE_FOLDS):
    """Pick C by stratified CV on SOURCE data."""
    grid = grid or C_GRID
    if int(y.sum()) < folds * 2:
        return C_FIXED_EHR
    cv = StratifiedKFold(
        n_splits=folds, shuffle=True,
        random_state=TUNE_SEED)
    best, bc = -1.0, C_FIXED_EHR
    for c in grid:
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y):
            m = LogisticRegression(
                C=c, max_iter=5000)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(
                X[te])[:, 1]
        a = roc_auc_score(y, oof)
        if a > best:
            best, bc = a, c
    return bc


def fit_lr(X, y, block="ehr", mode=None):
    """Fit with the C policy set by
    LEARNER_MODE."""
    mode = mode or LEARNER_MODE
    if mode == "fixed" and block == "ehr":
        c = C_FIXED_EHR
    else:
        c = tune_c(X, y)
    m = LogisticRegression(C=c, max_iter=5000)
    m.fit(X, y)
    m.chosen_c_ = c
    return m


def mk_lr():
    return LogisticRegression(
        C=C_FIXED_EHR, max_iter=5000)


def mk_gb():
    return HistGradientBoostingClassifier(
        random_state=42, max_depth=3,
        learning_rate=0.05, max_iter=300,
        l2_regularization=1.0)


LEARNERS = {"lr": mk_lr, "gb": mk_gb}


def _fit(X, y, block, learner):
    if learner == "gb":
        m = mk_gb()
        m.fit(X, y)
        return m
    return fit_lr(X, y, block=block)


# ---------- EARLY fusion ----------
def early(src, tgt, ys, variant="plain",
          learner="lr", scale="std"):
    se, sc_ = src
    te, tc = tgt
    sm = "blockstd" if variant == "blockstd" \
        else scale
    se, te = prep(se, te, sm)
    sc_, tc = prep(sc_, tc, sm)

    if variant == "pca":
        pe = PCA(n_components=PCA_VAR,
                 svd_solver="full",
                 random_state=42)
        pc = PCA(n_components=PCA_VAR,
                 svd_solver="full",
                 random_state=42)
        se = pe.fit_transform(se)
        te = pe.transform(te)
        sc_ = pc.fit_transform(sc_)
        tc = pc.transform(tc)
    elif variant not in ("plain", "blockstd"):
        raise ValueError(variant)

    m = _fit(np.hstack([se, sc_]), ys,
             "both", learner)
    return m.predict_proba(
        np.hstack([te, tc]))[:, 1]


# ---------- LATE fusion ----------
def _rank(p):
    return pd.Series(p).rank(pct=True).values


def late(src, tgt, ys, variant="mean",
         learner="lr", scale="std"):
    se, sc_ = src
    te, tc = tgt
    se2, te2 = prep(se, te, scale)
    sc2, tc2 = prep(sc_, tc, scale)

    me = _fit(se2, ys, "ehr", learner)
    mc = _fit(sc2, ys, "ctpa", learner)
    pe_t = _rank(me.predict_proba(te2)[:, 1])
    pc_t = _rank(mc.predict_proba(tc2)[:, 1])

    if variant == "ehr_only":
        return pe_t
    if variant == "ctpa_only":
        return pc_t
    if variant == "mean":
        return 0.5 * pe_t + 0.5 * pc_t
    if variant == "wsrc":
        pe_s = _rank(me.predict_proba(se2)[:, 1])
        pc_s = _rank(mc.predict_proba(sc2)[:, 1])
        best, bw = -1.0, 0.5
        for ww in np.arange(0.0, 1.01, 0.05):
            a = roc_auc_score(
                ys, ww * pe_s + (1 - ww) * pc_s)
            if a > best:
                best, bw = a, ww
        return bw * pe_t + (1 - bw) * pc_t
    raise ValueError(variant)


# ---------- INTERMEDIATE fusion ----------
def _enc(n, lat, p):
    return nn.Sequential(
        nn.Linear(n, 32), nn.BatchNorm1d(32),
        nn.ReLU(), nn.Dropout(p),
        nn.Linear(32, lat), nn.ReLU())


class JointNet(nn.Module):
    """One encoder per modality, a fusion block
    whose form depends on the variant, then a
    shared head."""

    def __init__(self, ne, nc, variant="plain",
                 lat=LAT, p=0.2):
        super().__init__()
        self.v = variant
        self.lat = lat
        self.ee = _enc(ne, lat, p)
        self.ec = _enc(nc, lat, p)
        d = 2 * lat

        if variant == "gated":
            self.g = nn.Sequential(
                nn.Linear(d, 2), nn.Sigmoid())
        elif variant == "lowrank":
            self.fe = nn.Linear(lat + 1,
                                RANK * lat)
            self.fc = nn.Linear(lat + 1,
                                RANK * lat)
            d = lat
        elif variant == "bilinear":
            self.pe = nn.Linear(lat, lat)
            self.pc = nn.Linear(lat, lat)
            d = lat
        elif variant == "tucker":
            self.pe = nn.Linear(lat, TUCKER_CORE)
            self.pc = nn.Linear(lat, TUCKER_CORE)
            self.core = nn.Parameter(
                torch.randn(TUCKER_CORE,
                            TUCKER_CORE, lat)
                * 0.05)
            d = lat
        elif variant == "crossattn":
            self.qe = nn.Linear(lat, lat)
            self.ke = nn.Linear(lat, lat)
            self.ve = nn.Linear(lat, lat)
            self.qc = nn.Linear(lat, lat)
            self.kc = nn.Linear(lat, lat)
            self.vc = nn.Linear(lat, lat)

        self.head = nn.Sequential(
            nn.Linear(d, 16), nn.ReLU(),
            nn.Dropout(p), nn.Linear(16, 1))

    def fuse(self, a, b):
        v = self.v
        if v in ("plain", "moddrop"):
            return torch.cat([a, b], dim=1)
        if v == "gated":
            z = torch.cat([a, b], dim=1)
            g = self.g(z)
            return torch.cat(
                [a * g[:, 0:1], b * g[:, 1:2]],
                dim=1)
        if v == "lowrank":
            one = torch.ones(a.shape[0], 1,
                             device=a.device)
            fa = self.fe(torch.cat([a, one], 1))
            fb = self.fc(torch.cat([b, one], 1))
            fa = fa.view(-1, RANK, self.lat)
            fb = fb.view(-1, RANK, self.lat)
            return (fa * fb).sum(dim=1)
        if v == "bilinear":
            z = self.pe(a) * self.pc(b)
            z = torch.sign(z) * torch.sqrt(
                torch.abs(z) + 1e-8)
            return nn.functional.normalize(z, dim=1)
        if v == "tucker":
            return torch.einsum(
                "bi,bj,ijk->bk", self.pe(a),
                self.pc(b), self.core)
        if v == "crossattn":
            s = float(self.lat) ** 0.5
            wa = torch.softmax(
                (self.qe(a) * self.kc(b)).sum(
                    1, keepdim=True) / s, dim=0)
            wb = torch.softmax(
                (self.qc(b) * self.ke(a)).sum(
                    1, keepdim=True) / s, dim=0)
            return torch.cat(
                [a + wa * self.vc(b),
                 b + wb * self.ve(a)], dim=1)
        raise ValueError(v)

    def forward(self, xe, xc):
        a = self.ee(xe)
        b = self.ec(xc)
        return self.head(self.fuse(a, b)).squeeze(1)


def intermediate(src, tgt, ys, variant="plain",
                 seed=42, epochs=150, lat=LAT,
                 scale="std", p=0.2):
    if variant not in INTER_VARIANTS:
        raise ValueError(variant)
    torch.manual_seed(seed)
    np.random.seed(seed)
    se, sc_ = src
    te, tc = tgt
    se, te = prep(se, te, scale)
    sc_, tc = prep(sc_, tc, scale)

    moddrop = 0.2 if variant == "moddrop" else 0.0
    xe = torch.tensor(se, dtype=torch.float32)
    xc = torch.tensor(sc_, dtype=torch.float32)
    yy = torch.tensor(ys, dtype=torch.float32)
    qe = torch.tensor(te, dtype=torch.float32)
    qc = torch.tensor(tc, dtype=torch.float32)

    net = JointNet(xe.shape[1], xc.shape[1],
                   variant=variant, lat=lat, p=p)
    pw = torch.tensor(
        float((ys == 0).sum()) /
        max(1.0, float((ys == 1).sum())))
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(net.parameters(),
                           lr=1e-3,
                           weight_decay=1e-4)

    n = len(ys)
    rng = np.random.default_rng(seed)
    net.train()
    for _ in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, 256):
            j = idx[i:i + 256]
            if len(j) < 8:
                continue
            be, bc = xe[j], xc[j]
            if moddrop > 0:
                u = rng.random()
                if u < moddrop:
                    be = torch.zeros_like(be)
                elif u < 2 * moddrop:
                    bc = torch.zeros_like(bc)
            opt.zero_grad()
            crit(net(be, bc), yy[j]).backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(qe, qc)).numpy()


def n_params(variant, ne=28, nc=38, lat=LAT):
    net = JointNet(ne, nc, variant=variant,
                   lat=lat)
    return sum(q.numel() for q in net.parameters())


# ---------- evaluation ----------
def boot_diff(y, pa, pb, grp, nboot=NBOOT,
              seed=42):
    """Paired subject-level bootstrap of
    AUC(pa) - AUC(pb)."""
    us = np.unique(grp)
    ix = {u: np.where(grp == u)[0] for u in us}
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(nboot):
        pk = rng.choice(us, len(us), replace=True)
        ii = np.concatenate([ix[u] for u in pk])
        if len(np.unique(y[ii])) < 2:
            continue
        d.append(roc_auc_score(y[ii], pa[ii])
                 - roc_auc_score(y[ii], pb[ii]))
    d = np.array(d)
    return (float(np.mean(d)),
            float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)),
            float((d > 0).mean()))