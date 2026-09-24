"""Extract frozen sentence embeddings from CTPA
impressions using several clinical encoders.
MIMIC notes join on idx_hadm, not hadm_id, and are
restricted to the presentation window of -48h to
+24h relative to admission, taking the scan
nearest the admission.
Run in venv_nlp only.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# folder holding the dissertation's derived files
DIS = os.environ.get(
    "PE_DISS_DIR", os.path.join("..", "Dissertation"))
FIG = os.path.join(DIS, "figwork", "data")
RAW = os.path.join("data", "raw", "inspect")
PROC = os.path.join("data", "processed")
EMB = os.path.join("data", "embeddings")
os.makedirs(EMB, exist_ok=True)
D = "_20250611.tsv"

MAXLEN = 256
BATCH = 16
WIN_LO = -48.0
WIN_HI = 24.0

MODELS = {
    "radbert": ("zzxslp/RadBERT-RoBERTa-4m",
                False),
    "pubmedbert": (
        "microsoft/BiomedNLP-BiomedBERT-base"
        "-uncased-abstract-fulltext", False),
    "clinicalbert": (
        "emilyalsentzer/Bio_ClinicalBERT", True),
    "bioclinmodern": (
        "thomas-sounack/BioClinical-ModernBERT"
        "-base", True),
}

torch.set_num_threads(max(1, os.cpu_count() - 1))


def load_inspect_texts():
    files = {}
    for nm in ("impressions", "labels", "splits",
               "study_mapping"):
        p = os.path.join(RAW, nm + D)
        if os.path.exists(p):
            files[nm] = pd.read_csv(p, sep="\t")

    imp = files["impressions"]
    src = None
    for nm, df in files.items():
        if nm == "impressions":
            continue
        if ("person_id" in df.columns
                and "impression_id" in df.columns):
            src = nm
            break
    if src is None:
        raise SystemExit("no person_id mapping")
    print("  person_id from:", src)

    m = files[src]
    cols = ["impression_id", "person_id"]
    if "procedure_DATETIME" in m.columns:
        cols.append("procedure_DATETIME")
    g = imp.merge(m[cols].drop_duplicates(
        "impression_id"),
        on="impression_id", how="left")

    if "procedure_DATETIME" not in g.columns:
        sm = files.get("study_mapping")
        if sm is not None:
            g = g.merge(
                sm[["impression_id",
                    "procedure_DATETIME"]]
                .drop_duplicates("impression_id"),
                on="impression_id", how="left")

    sel = pd.read_csv(os.path.join(
        PROC, "inspect_imp_features_v2.csv"),
        usecols=["person_id"])
    hl = pd.read_csv(os.path.join(
        FIG, "inspect_labels_final.csv"))
    hl["pdate"] = pd.to_datetime(
        hl["pe_date"], errors="coerce")

    g = g[g["person_id"].isin(
        set(sel["person_id"]))]
    g["pdt"] = pd.to_datetime(
        g["procedure_DATETIME"], errors="coerce")
    g = g.merge(hl[["person_id", "pdate"]],
                on="person_id", how="left")
    g["gap"] = (g["pdt"] - g["pdate"]).abs()
    g = g[g["gap"] <= pd.Timedelta(days=30)]
    g = g.sort_values(["person_id", "gap"])
    g = g.groupby("person_id",
                  as_index=False).first()
    g["txt"] = g["impressions"].astype(str)
    return g[["person_id", "txt"]]


def load_mimic_texts():
    """Join on idx_hadm and keep the presentation
    window, taking the scan nearest admission."""
    idx = pd.read_csv(os.path.join(
        FIG, "ctpa_notes_index.csv"))
    keep = pd.read_csv(os.path.join(
        FIG, "mimic_imp_features_v2.csv"),
        usecols=["hadm_id"])
    f = set(keep["hadm_id"])

    idx = idx[idx["idx_hadm"].isin(f)].copy()
    print("  rows for feature admissions:",
          len(idx))
    print("  admissions covered:",
          idx["idx_hadm"].nunique())

    w = idx[(idx["h_before"] >= WIN_LO)
            & (idx["h_before"] <= WIN_HI)]
    print("  inside window %.0f to %.0f h: %d rows,"
          " %d admissions"
          % (WIN_LO, WIN_HI, len(w),
             w["idx_hadm"].nunique()))

    lost = f - set(w["idx_hadm"])
    if lost:
        print("  outside window, nearest kept:",
              len(lost))
        extra = idx[idx["idx_hadm"].isin(lost)]
        w = pd.concat([w, extra])

    w = w.copy()
    w["absh"] = w["h_before"].abs()
    w = w.sort_values(["idx_hadm", "absh"])
    w = w.groupby("idx_hadm",
                  as_index=False).first()

    def imp_only(t):
        t = str(t)
        low = t.lower()
        k = low.rfind("impression")
        if k >= 0:
            t = t[k:]
            c = t.find(":")
            if 0 <= c < 20:
                t = t[c + 1:]
        return " ".join(t.split())

    w["txt"] = w["text"].apply(imp_only)
    w = w.rename(columns={"idx_hadm": "hadm_id"})
    return w[["hadm_id", "txt"]]


def embed(texts, hf_id):
    tok = AutoTokenizer.from_pretrained(hf_id)
    mod = AutoModel.from_pretrained(hf_id)
    mod.eval()
    out = []
    t0 = time.time()
    n = len(texts)
    for i in range(0, n, BATCH):
        b = texts[i:i + BATCH]
        enc = tok(b, padding=True,
                  truncation=True,
                  max_length=MAXLEN,
                  return_tensors="pt")
        with torch.no_grad():
            h = mod(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1)
        m = m.type_as(h)
        pooled = (h * m).sum(1) / m.sum(1).clamp(
            min=1e-9)
        out.append(pooled.numpy())
        if (i // BATCH) % 50 == 0:
            el = time.time() - t0
            done = min(i + BATCH, n)
            rate = done / max(el, 1e-9)
            print("    %d/%d  %.1f/s  eta %.1f min"
                  % (done, n, rate,
                     (n - done) / max(rate, 1e-9)
                     / 60.0))
    return np.vstack(out)


print("loading texts ...")
ins = load_inspect_texts()
mim = load_mimic_texts()
print("INSPECT impressions:", len(ins),
      "(expect 3300)")
print("MIMIC impressions:", len(mim),
      "(expect 1649)")
print("median chars: INSPECT %d, MIMIC %d"
      % (int(ins["txt"].str.len().median()),
         int(mim["txt"].str.len().median())))

ins[["person_id"]].to_csv(
    os.path.join(EMB, "ids_inspect.csv"),
    index=False)
mim[["hadm_id"]].to_csv(
    os.path.join(EMB, "ids_mimic.csv"),
    index=False)

ok, bad = [], []
for name, (hf_id, contam) in MODELS.items():
    print("")
    print("=" * 58)
    print("%s  (%s)%s" % (name, hf_id,
                          "  [MIMIC-pretrained]"
                          if contam else ""))
    try:
        for tag, df in (("inspect", ins),
                        ("mimic", mim)):
            p = os.path.join(
                EMB, "%s_%s.npy" % (name, tag))
            if os.path.exists(p):
                e = np.load(p, mmap_mode="r")
                if e.shape[0] == len(df):
                    print("  skip %s (exists,"
                          " rows match)" % tag)
                    continue
                print("  redoing %s (had %d,"
                      " need %d)"
                      % (tag, e.shape[0], len(df)))
            print("  embedding", tag, len(df))
            e = embed(df["txt"].tolist(), hf_id)
            np.save(p, e)
            print("  saved %s  shape %s"
                  % (p, e.shape))
        ok.append(name)
    except Exception as exc:
        print("  FAILED:", repr(exc)[:300])
        bad.append((name, repr(exc)[:200]))

print("")
print("=" * 58)
print("succeeded:", ok)
for n, e in bad:
    print("failed:", n, "->", e)
