"""Faster MIMIC-IV-ECG download using a
persistent session and a thread pool. Same URLs
and credentials as the wfdb version, so it is a
transport change rather than a logic change.

Run with --test to try 20 records first.
Resumable and safe to interrupt.
Run in venv_ecg only.
"""
import os
import sys
import time
import threading
import pandas as pd
import requests
from getpass import getpass
from concurrent.futures import ThreadPoolExecutor

# folder holding the dissertation's derived files
DIS = os.environ.get(
    "PE_DISS_DIR", os.path.join("..", "Dissertation"))
DSET = os.path.join(DIS, "Datasets")
OUT = os.path.join("data", "raw", "ecg")
BASE = ("https://physionet.org/files"
        "/mimic-iv-ecg/1.0/")
THREADS = 8
TIMEOUT = 60
RETRIES = 3

TEST = "--test" in sys.argv
os.makedirs(OUT, exist_ok=True)

user = input("PhysioNet username: ").strip()
pw = getpass("PhysioNet password: ")

local = threading.local()


def sess():
    s = getattr(local, "s", None)
    if s is None:
        s = requests.Session()
        s.auth = (user, pw)
        a = requests.adapters.HTTPAdapter(
            pool_connections=THREADS,
            pool_maxsize=THREADS,
            max_retries=0)
        s.mount("https://", a)
        local.s = s
    return s


def clean(p):
    s = str(p).replace("\\", "/").strip()
    for e in (".hea", ".dat"):
        if s.endswith(e):
            s = s[:-len(e)]
    return s.strip("/")


coh = pd.read_csv(
    os.path.join(DSET, "MIMICIV",
                 "mimic_pe_mace_cohort.csv"))
c = coh.dropna(subset=["ecg_path"]).copy()
c["rec"] = c["ecg_path"].apply(clean)
recs = sorted(c["rec"].unique())

todo = []
for r_ in recs:
    parts = r_.split("/")
    d = os.path.join(OUT, *parts[:-1])
    b = parts[-1]
    if not (os.path.exists(
            os.path.join(d, b + ".hea"))
            and os.path.exists(
                os.path.join(d, b + ".dat"))):
        todo.append(r_)

print("")
print("total records:", len(recs))
print("already present:", len(recs) - len(todo))
print("to download:", len(todo))

if TEST:
    step = max(1, len(todo) // 20)
    todo = todo[::step][:20]
    print("TEST MODE: %d records" % len(todo))

if not todo:
    print("nothing to do")
    sys.exit(0)

lock = threading.Lock()
state = {"ok": 0, "fail": 0, "bytes": 0}
bad = []
t0 = time.time()


def fetch(rec):
    parts = rec.split("/")
    d = os.path.join(OUT, *parts[:-1])
    b = parts[-1]
    os.makedirs(d, exist_ok=True)
    got = 0
    for ext in (".hea", ".dat"):
        dest = os.path.join(d, b + ext)
        if os.path.exists(dest) and \
                os.path.getsize(dest) > 0:
            got += os.path.getsize(dest)
            continue
        url = BASE + rec + ext
        last = None
        for attempt in range(RETRIES):
            try:
                r = sess().get(
                    url, timeout=TIMEOUT,
                    stream=True)
                if r.status_code == 401:
                    raise RuntimeError(
                        "401 unauthorised -"
                        " check credentials")
                if r.status_code != 200:
                    raise RuntimeError(
                        "%d for %s"
                        % (r.status_code, ext))
                tmp = dest + ".part"
                n = 0
                with open(tmp, "wb") as fh:
                    for ch in r.iter_content(
                            65536):
                        if ch:
                            fh.write(ch)
                            n += len(ch)
                os.replace(tmp, dest)
                got += n
                last = None
                break
            except Exception as exc:
                last = exc
                time.sleep(1.0 + attempt)
        if last is not None:
            raise last
    return got


def work(rec):
    try:
        n = fetch(rec)
        with lock:
            state["ok"] += 1
            state["bytes"] += n
    except Exception as exc:
        with lock:
            state["fail"] += 1
            bad.append((rec, repr(exc)[:120]))
    with lock:
        done = state["ok"] + state["fail"]
        if done % 200 == 0 or done == len(todo):
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            print("  %d/%d  ok %d  fail %d"
                  "  %.1f/s  %.0f MB"
                  "  eta %.0f min"
                  % (done, len(todo),
                     state["ok"], state["fail"],
                     rate,
                     state["bytes"] / 1e6,
                     (len(todo) - done)
                     / max(rate, 1e-9) / 60),
                  flush=True)


print("")
print("downloading with %d threads ..."
      % THREADS)
try:
    with ThreadPoolExecutor(
            max_workers=THREADS) as ex:
        list(ex.map(work, todo))
except KeyboardInterrupt:
    print("")
    print("interrupted - rerun to resume")

el = time.time() - t0
print("")
print("succeeded:", state["ok"])
print("failed:", state["fail"])
print("downloaded: %.0f MB"
      % (state["bytes"] / 1e6))
print("elapsed: %.1f min" % (el / 60))
if state["ok"]:
    rate = state["ok"] / max(el, 1e-9)
    print("rate: %.2f records/s" % rate)
    rem = len(recs) - (len(recs) - len(todo)) \
        - state["ok"]
    if rem > 0:
        print("remaining %d -> %.1f hours"
              % (rem, rem / max(rate, 1e-9)
                 / 3600))

for r_, e in bad[:10]:
    print("  FAIL", r_, "->", e)

if TEST:
    print("")
    if state["fail"] == 0:
        print("TEST PASSED - rerun without"
              " --test for the full download")
    else:
        print("TEST FAILED - use"
              " scripts\\37_download_ecg.py"
              " instead")