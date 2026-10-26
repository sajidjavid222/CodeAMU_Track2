#!/usr/bin/env python3

"""IndoML 2026 Track 2, B200 training script.

Usage:
  T2_DATA=/data/indoml T2_BUDGET=smoke python track2_b200_train.py     # 15 minute check
  T2_DATA=/data/indoml T2_BUDGET=big   python track2_b200_train.py     # full run

T2_DATA must hold the official Track 2 validation folder (and the Track 1 predictions.jsonl).
HF_TOKEN must be in the environment. Output: $T2_WORK/export, upload that to Kaggle.
"""

# ============================== CONFIG (B200 training box) ==============================
import os, sys, gc, io, re, json, math, time, random, shutil, zipfile, hashlib, tempfile, subprocess, warnings
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
warnings.filterwarnings("ignore")

# Scale knobs. "smoke" first, then "big".
BUDGET = os.environ.get("T2_BUDGET", "big")     # "smoke" | "t4" | "mid" | "big"
SEED = 2026

VAANI_REPO = "ARTPARK-IISc/Vaani"
GOLD_REPO = "ARTPARK-IISc/Vaani-Noise-Event-Dataset"
ASR_REPO = "ARTPARK-IISc/SraVaani-1.0"
TRANSCRIPT_NAME, ID_KEY, TEXT_KEY, WAV_SUBTYPE = "transcripts.jsonl", "clip_id", "text", "PCM_16"

# Where the three official folders live on this box (copy them from Kaggle).
DATA_ROOT = Path(os.environ.get("T2_DATA", "/data/indoml"))
VAL_ROOT_OVERRIDE = None
TEST_ROOT_OVERRIDE = None       # not needed for training, only if you also infer here
T1_PRED_OVERRIDE = None

WORK = Path(os.environ.get("T2_WORK", "./t2work"))
SCRATCH = WORK / "scratch"
POOL_DIR = WORK / "pools"
CKPT_DIR = WORK / "ckpt"
EXPORT_DIR = WORK / "export"          # upload this folder to Kaggle as a dataset
SUB_DIR = SCRATCH / "submission_wavs"
HF_HOME = Path(os.environ.get("HF_HOME", str(WORK / "hf_home")))
for d in (WORK, SCRATCH, POOL_DIR, CKPT_DIR, EXPORT_DIR, SUB_DIR, HF_HOME):
    d.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_DATASETS_CACHE"] = str(HF_HOME / "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BUDGETS = {
    "smoke": dict(gold_max_min=5, hop=160, seg_sec=2.0, disk_gb=14.0, min_free_gb=1.0, gold_files=2, gold_rows=400, gold_rg=2, n_dist_in=2, n_dist_rand=1, files_per_dist=1,
                  rg_per_file=2, clips_per_dist=40, speech_hours_cap=0.3,
                  bs=4, workers=2, eval_every_min=2, n_models=1, model_min=3, polish_min=1,
                  arch=dict(ch=(16, 24, 32, 48), n_dp=2, hid=32, attn=True)),
    "t4":    dict(gold_max_min=100, hop=160, seg_sec=4.0, disk_gb=14.0, min_free_gb=2.0,
                  gold_files=60, gold_rows=8000, gold_rg=3, n_dist_in=20, n_dist_rand=6, files_per_dist=2,
                  rg_per_file=3, clips_per_dist=300, speech_hours_cap=12.0,
                  bs=8, workers=4, eval_every_min=15, n_models=1, model_min=200, polish_min=25,
                  arch=dict(ch=(32, 48, 64, 96), n_dp=4, hid=96, attn=True)),
    "mid":   dict(gold_max_min=180, disk_gb=400.0, gold_files=60, gold_rows=20000, gold_rg=4, n_dist_in=25, n_dist_rand=10, files_per_dist=3,
                  rg_per_file=5, clips_per_dist=600, speech_hours_cap=40.0,
                  bs=24, workers=12, eval_every_min=15, n_models=2, model_min=240, polish_min=30,
                  arch=dict(ch=(48, 64, 96, 128), n_dp=5, hid=128, attn=True)),
    # Sized for this cluster's 6 h short queue on one 3g.71gb MIG slice (10 CPU cap).
    # n_models=1 so a single session finishes and exports; chain more with T2_RESUME.
    "omni":  dict(gold_max_min=45, disk_gb=120.0, min_free_gb=4.0, gold_files=60, gold_rows=20000, gold_rg=4,
                  n_dist_in=25, n_dist_rand=10, files_per_dist=3,
                  rg_per_file=5, clips_per_dist=600, speech_hours_cap=40.0,
                  bs=24, workers=8, eval_every_min=15, n_models=1, model_min=185, polish_min=25,
                  arch=dict(ch=(48, 64, 96, 128), n_dp=5, hid=128, attn=True)),
    "big":   dict(gold_max_min=300, disk_gb=400.0, gold_files=182, gold_rows=90000, gold_rg=8, n_dist_in=45, n_dist_rand=20, files_per_dist=6,
                  rg_per_file=8, clips_per_dist=1200, speech_hours_cap=140.0,
                  bs=32, workers=16, eval_every_min=20, n_models=3, model_min=420, polish_min=45,
                  arch=dict(ch=(64, 96, 128, 192), n_dp=6, hid=160, attn=True)),
}
B = BUDGETS[BUDGET]
SR = 16000
N_FFT = 512
HOP = B.get("hop", 128)
SEG_SEC = B.get("seg_sec", 6.0)
ARCH = dict(B["arch"], n_fft=N_FFT, hop=HOP)
DISK_BUDGET_GB = float(os.environ.get("T2_DISK_GB", B.get("disk_gb", 400)))
MIN_FREE_GB = B.get("min_free_gb", 10.0)
RESUME_FROM = os.environ.get("T2_RESUME")      # path to a .pt to continue from (multi session T4 runs)

# Loss weights
W_MRSTFT = 0.1
W_MEL = 0.3          # ASR facing mel loss, this is the dWER lever
HARVEST_NOISE = True  # mine real noise from natural train fold clips after the first model

SEL = dict(min_dur=2.0, max_dur=16.0, snr_tagless=18.0, snr_untagged=26.0, max_clip_frac=1e-3,
           min_vad=0.35, per_group_cap=4)
EVT = dict(min_dur=0.12, max_dur=8.0, margin=0.03, max_speech_ratio=0.55, min_clean_gap=2.0)
GOLD_MAX_MIN = float(os.environ.get("T2_GOLD_MIN", B.get("gold_max_min", 240)))
STRICT_GROUP_EXCLUDE = True
MASK_AUG = dict(p_zero=0.35, p_drop=0.25, jitter=0.15, p_false=0.20, false_dur=(0.2, 1.5))
PRED_PAD = 0.10

random.seed(SEED); np.random.seed(SEED)
print("BUDGET", BUDGET, "| arch", ARCH)


# ============================== INSTALLS + TOKEN ==============================
def _pip(*pk):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pk], check=False)

for mod, pkg in [("soundfile", "soundfile"), ("librosa", "librosa"), ("jiwer", "jiwer"),
                 ("webrtcvad", "webrtcvad-wheels"), ("pyarrow", "pyarrow"), ("huggingface_hub", "huggingface_hub")]:
    try:
        __import__(mod)
    except Exception:
        _pip(pkg)

import soundfile as sf
import scipy.signal as sps
import pyarrow.parquet as pq
import torch, torch.nn as nn, torch.nn.functional as F

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        print("set HF_TOKEN in the environment before running")
assert HF_TOKEN, "Add HF_TOKEN in Kaggle Secrets (and accept the gated terms of both Vaani repos)."
os.environ["HF_TOKEN"] = HF_TOKEN
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
print("device", device, torch.__version__)

# ============================== DISK GUARD ==============================
class DiskBudgetExceeded(RuntimeError):
    pass

def dir_gb(p):
    p = Path(p)
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9

class DiskGuard:
    def __init__(self, every_sec=20):
        self.every, self.last, self.used = every_sec, 0.0, 0.0
    def check(self, tag="", force=False):
        if not force and time.time() - self.last < self.every:
            return self.used
        self.last = time.time()
        self.used = sum(dir_gb(d) for d in (POOL_DIR, CKPT_DIR, SCRATCH))
        free = shutil.disk_usage(WORK).free / 1e9
        if self.used > DISK_BUDGET_GB or free < MIN_FREE_GB:
            raise DiskBudgetExceeded(f"[{tag}] used {self.used:.2f} GB, free {free:.2f} GB")
        return self.used

DISK = DiskGuard()
print("disk used by notebook: %.2f GB, free %.1f GB" % (DISK.check(force=True), shutil.disk_usage(WORK).free / 1e9))

# ============================== INPUT DISCOVERY ==============================
AUD = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
INPUT = DATA_ROOT

def _has(d, exts, limit=1):
    n = 0
    for p in d.rglob("*"):
        if p.suffix.lower() in exts:
            n += 1
            if n >= limit:
                return True
    return False

def _candidate_dirs(max_depth=4):
    out = []
    if not INPUT.exists():
        return out
    for p in INPUT.rglob("*"):
        if p.is_dir() and len(p.relative_to(INPUT).parts) <= max_depth:
            out.append(p)
    return out

def _count_audio(d):
    return sum(1 for p in d.rglob("*") if p.suffix.lower() in AUD)

def discover_inputs():
    dirs = _candidate_dirs()
    val_root = Path(VAL_ROOT_OVERRIDE) if VAL_ROOT_OVERRIDE else None
    if val_root is None:
        c = [d for d in dirs if "valid" in d.name.lower() and _has(d, AUD + (".parquet",))]
        val_root = min(c, key=lambda d: len(d.parts)) if c else None
    test_root = Path(TEST_ROOT_OVERRIDE) if TEST_ROOT_OVERRIDE else None
    if test_root is None:
        c = [d for d in dirs if "test" in d.name.lower() and _has(d, AUD + (".parquet",))
             and (val_root is None or val_root not in d.parents and d != val_root)]
        test_root = min(c, key=lambda d: len(d.parts)) if c else None
    if test_root is None:
        best = None
        for d in (INPUT.iterdir() if INPUT.exists() else []):
            if val_root is not None and (d == val_root or d in val_root.parents):
                continue
            n = _count_audio(d)
            if n and (best is None or n > best[1]):
                best = (d, n)
        test_root = best[0] if best else None
    t1 = Path(T1_PRED_OVERRIDE) if T1_PRED_OVERRIDE else None
    if t1 is None and INPUT.exists():
        c = sorted(p for p in INPUT.rglob("*.jsonl") if "pred" in p.name.lower())
        t1 = c[0] if c else None
    return val_root, test_root, t1

VAL_ROOT, TEST_ROOT, T1_PATH = discover_inputs()
print("VAL_ROOT :", VAL_ROOT)
print("TEST_ROOT:", TEST_ROOT)
print("T1 preds :", T1_PATH)

def load_t1_predictions(path):
    """clip_id -> list of (onset, offset) seconds. Accepts onset/offset or start/end keys."""
    out = {}
    if path is None or not Path(path).exists():
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        cid = str(r.get("clip_id") or r.get("id") or r.get("filename") or "")
        cid = re.sub(r"\.(wav|flac|mp3|ogg|m4a)$", "", Path(cid).name, flags=re.I)
        ev = []
        for e in (r.get("events") or []):
            a = e.get("onset", e.get("start")); b = e.get("offset", e.get("end"))
            try:
                a, b = float(a), float(b)
            except Exception:
                continue
            if b > a:
                ev.append((a, b))
        out[cid] = sorted(ev)
    return out

T1 = load_t1_predictions(T1_PATH)
if PRED_PAD > 0:
    T1 = {k: [(max(0.0, a - PRED_PAD), b + PRED_PAD) for a, b in v] for k, v in T1.items()}
print("T1 clips:", len(T1), "| clips with events:", sum(1 for v in T1.values() if v))

# ============================== OFFICIAL SCORER (verbatim logic) ==============================
TAG_RE = re.compile(r"</?[^<>]*>|\[[^\[\]]*\]")
import unicodedata

def si_sdr(reference, enhanced):
    ref = np.asarray(reference, dtype=np.float64); enh = np.asarray(enhanced, dtype=np.float64)
    n = min(len(ref), len(enh)); ref, enh = ref[:n], enh[:n]
    scale = np.dot(enh, ref) / np.dot(ref, ref)
    s_t = scale * ref; e_n = enh - s_t
    return float(np.clip(10.0 * np.log10(np.dot(s_t, s_t) / np.dot(e_n, e_n)), -100.0, 100.0))

def normalize_text(text):
    s = TAG_RE.sub(" ", text or "")
    s = "".join(" " if unicodedata.category(c).startswith("P") else c for c in s)
    return " ".join(s.lower().split())

def delta_wer(gt, noisy_asr, submitted, clip_ids):
    from jiwer import wer as jwer
    refs, noisy, enh = [], [], []
    for cid in clip_ids:
        g = normalize_text(gt[cid])
        if not g:
            continue
        refs.append(g)
        noisy.append(normalize_text(noisy_asr.get(cid, "")) or "@")
        enh.append(normalize_text(submitted.get(cid, "")) or "@")
    if not refs:
        return float("nan")
    return jwer(refs, noisy) - jwer(refs, enh)

# ============================== AUDIO UTILS + VAD ==============================
def to_mono(w):
    w = np.asarray(w, dtype=np.float32)
    if w.ndim > 1:
        w = w.mean(axis=0) if w.shape[0] < w.shape[1] else w.mean(axis=1)
    return w

def resample(w, sr):
    if sr == SR:
        return w.astype(np.float32)
    g = math.gcd(int(sr), SR)
    return sps.resample_poly(w, SR // g, int(sr) // g).astype(np.float32)

def decode_bytes(b):
    try:
        w, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    except Exception:
        import librosa
        with tempfile.NamedTemporaryFile(suffix=".bin", dir=SCRATCH, delete=True) as t:
            t.write(b); t.flush()
            w, sr = librosa.load(t.name, sr=None, mono=True)
    return np.ascontiguousarray(resample(to_mono(w), sr))

def read_audio(p):
    try:
        w, sr = sf.read(str(p), dtype="float32", always_2d=False)
    except Exception:
        import librosa
        w, sr = librosa.load(str(p), sr=None, mono=True)
    return np.ascontiguousarray(resample(to_mono(w), sr))

def frame_power(w, fr=320, hop=160):
    if len(w) < fr:
        return np.array([np.mean(w ** 2) + 1e-12])
    n = 1 + (len(w) - fr) // hop
    idx = np.arange(fr)[None, :] + hop * np.arange(n)[:, None]
    return (w[idx] ** 2).mean(1) + 1e-12

def est_snr_db(w):
    p = frame_power(w)
    return float(10 * np.log10(np.percentile(p, 90) / np.percentile(p, 10)))

try:
    import webrtcvad
    _VAD = webrtcvad.Vad(2)
except Exception:
    _VAD = None

VAD_FR = 480  # 30 ms

def vad_mask(w):
    """Per 30 ms frame speech flag."""
    n = len(w) // VAD_FR
    if n == 0:
        return np.zeros(0, bool)
    if _VAD is not None:
        pcm = (np.clip(w[: n * VAD_FR], -1, 1) * 32767).astype(np.int16)
        return np.array([_VAD.is_speech(pcm[i * VAD_FR:(i + 1) * VAD_FR].tobytes(), SR) for i in range(n)])
    fp = (w[: n * VAD_FR].reshape(n, VAD_FR) ** 2).mean(1) + 1e-12
    return 10 * np.log10(fp) > 10 * np.log10(np.percentile(fp, 20)) + 8

def speech_ratio(w):
    m = vad_mask(w)
    return float(m.mean()) if len(m) else 1.0

def xfade_concat(chunks, xf=160):
    out = None
    for c in chunks:
        c = np.asarray(c, np.float32)
        if out is None:
            out = c.copy(); continue
        k = min(xf, len(out), len(c))
        if k > 0:
            r = np.linspace(0, 1, k, dtype=np.float32)
            out[-k:] = out[-k:] * (1 - r) + c[:k] * r
            out = np.concatenate([out, c[k:]])
        else:
            out = np.concatenate([out, c])
    return out if out is not None else np.zeros(0, np.float32)

def extract_floor(w, avoid=None, min_run=0.25, max_len=8.0):
    """Concatenate non speech, low energy regions (optionally avoiding event windows) into a floor bed."""
    m = vad_mask(w)
    if len(m) == 0:
        return None
    fp = (w[: len(m) * VAD_FR].reshape(len(m), VAD_FR) ** 2).mean(1) + 1e-12
    ok = (~m) & (fp <= np.percentile(fp, 40))
    if avoid:
        for a, b in avoid:
            ok[max(0, int(a * SR) // VAD_FR - 1): int(b * SR) // VAD_FR + 2] = False
    runs, s = [], None
    for i, v in enumerate(list(ok) + [False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if (i - s) * VAD_FR >= min_run * SR:
                runs.append((s * VAD_FR, i * VAD_FR))
            s = None
    if not runs:
        return None
    bed = xfade_concat([w[a:b] for a, b in runs])[: int(max_len * SR)]
    if len(bed) < 0.5 * SR or np.sqrt(np.mean(bed ** 2)) < 1e-5:
        return None
    return bed

def fit_len(w, L, rng, xf=160):
    if len(w) >= L:
        s = int(rng.integers(0, len(w) - L + 1))
        return w[s:s + L].copy()
    reps = []
    tot = 0
    while tot < L + xf:
        reps.append(w); tot += len(w) - xf
    return xfade_concat(reps, xf)[:L].copy()

def windows_from_noise(n, rel_db=6.0, min_dur=0.1, gap=0.15):
    """Burst windows (sec) where the noise envelope rises rel_db above its floor."""
    fr, hop = 800, 400
    p = frame_power(n, fr, hop)
    db = 10 * np.log10(p)
    floor = np.percentile(db, 30)
    on = db > floor + rel_db
    wins, s = [], None
    for i, v in enumerate(list(on) + [False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            wins.append([s * hop / SR, (i * hop + fr) / SR]); s = None
    merged = []
    for a, b in wins:
        if merged and a - merged[-1][1] < gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, min(b, len(n) / SR)) for a, b in merged if b - a >= min_dur], float(floor)

def sample_mask_from_windows(wins, n, ramp=0.05):
    m = np.zeros(n, np.float32)
    for a, b in wins:
        m[max(0, int(a * SR)):min(n, int(b * SR))] = 1.0
    k = int(ramp * SR)
    if k > 1 and m.any():
        ker = np.ones(k, np.float32) / k
        m = np.clip(np.convolve(m, ker, mode="same") * 1.0, 0, 1)
    return m

def frames_from_windows(wins, n):
    T = n // HOP + 1
    f = np.zeros(T, np.float32)
    for a, b in wins:
        f[max(0, int(a * SR) // HOP):min(T, int(b * SR) // HOP + 1)] = 1.0
    return f

# ============================== VALIDATION LOADER ==============================
def _strip_stem(s):
    s = Path(str(s)).name
    s = re.sub(r"\.(wav|flac|mp3|ogg|m4a)$", "", s, flags=re.I)
    return s

def _base_stem(s):
    return re.sub(r"(_?(noisy|noise|mix|mixed|clean|cleanref|ref|synthetic\w*))+$", "", _strip_stem(s), flags=re.I)

def _role(path, root):
    name = path.parent.name.lower()
    rel = "/".join(path.relative_to(root).parts).lower()
    if "clean" in name or "clean" in path.stem.lower():
        role = "clean"
    elif "noisy" in name or "mix" in name:
        role = "noisy"
    elif "noiseaudio" in name:
        # IndoML's `syntheticNoiseAudio` holds the noisy MIXTURE, not the noise
        # signal -- the same files the identity submission shipped as input.
        # The generic "noise" branch below classified it as noise, so the loader
        # synthesised x = c + n = 2*clean + noise. That inflated held-out input
        # SI-SDR to 6.50 dB against a true -0.33 dB, and trained the model on a
        # task (2c+n -> c) that does not exist at test time: ~10 dB of the
        # local-vs-leaderboard gap.
        role = "noisy"
    elif "noise" in name or re.search(r"noise(?!s)(?!y)", path.stem.lower()):
        role = "noise"
    else:
        role = "noisy"
    return role, ("synth" in rel)

def _materialize_parquet(root):
    """If the validation set is parquet, write its audio columns to wav files in SCRATCH."""
    out = SCRATCH / "val_from_parquet"
    files = sorted(root.rglob("*.parquet"))
    if not files:
        return None, []
    rows = []
    for fp in files:
        t = pq.read_table(fp)
        cols = t.column_names
        audio_cols = [c for c in cols if str(t.schema.field(c).type).startswith("struct") and "bytes" in str(t.schema.field(c).type)]
        d = t.to_pylist()
        for i, r in enumerate(d):
            cid = str(r.get("clip_id") or r.get("id") or r.get("filename") or f"{fp.stem}_{i}")
            cid = _base_stem(cid)
            meta = {k: v for k, v in r.items() if k not in audio_cols}
            for c in audio_cols:
                cell = r.get(c)
                if not cell or not cell.get("bytes"):
                    continue
                role = "clean" if "clean" in c.lower() else ("noise" if ("noise" in c.lower() and "noisy" not in c.lower()) else "noisy")
                prefix = "synthetic_" if "synth" in (c + fp.as_posix()).lower() else ""
                sub = out / (prefix + role)
                sub.mkdir(parents=True, exist_ok=True)
                sf.write(sub / f"{cid}.wav", decode_bytes(cell["bytes"]), SR, subtype="PCM_16")
            rows.append(dict(meta, clip_id=cid))
    return out, rows

def _load_meta(root):
    meta = {}
    def add(r):
        if not isinstance(r, dict):
            return
        cid = r.get("clip_id") or r.get("id") or r.get("filename") or r.get("file") or r.get("audio")
        if cid is None or isinstance(cid, (dict, list)):
            return
        meta.setdefault(_base_stem(cid), {}).update(r)
    for p in list(root.rglob("*.json")) + list(root.rglob("*.jsonl")) + list(root.rglob("*.csv")):
        try:
            if p.suffix == ".jsonl":
                for line in open(p, encoding="utf-8"):
                    if line.strip():
                        add(json.loads(line))
            elif p.suffix == ".csv":
                import csv
                for r in csv.DictReader(open(p, encoding="utf-8")):
                    add(r)
            else:
                obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                if isinstance(obj, dict) and obj and all(isinstance(v, dict) for v in obj.values()):
                    for k, v in obj.items():
                        add(dict(v, clip_id=k))
                else:
                    rows = obj if isinstance(obj, list) else [obj]
                    for key in ("clips", "data", "items", "records", "validation"):
                        if isinstance(obj, dict) and isinstance(obj.get(key), list):
                            rows = obj[key]; break
                    for r in rows:
                        add(r)
        except Exception as e:
            print("meta read failed", p, e)
    return meta

def _pick(r, *names):
    for n in names:
        v = r.get(n)
        if v not in (None, "", [], {}):
            return v
    return None

def load_validation(root):
    if root is None:
        print("No validation set found. Held out scoring will be skipped.")
        return []
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in AUD]
    if not files:
        proot, _ = _materialize_parquet(root)
        if proot is not None:
            files = [p for p in proot.rglob("*") if p.suffix.lower() in AUD]
            root_for_roles = proot
        else:
            root_for_roles = root
    else:
        root_for_roles = root
    meta = _load_meta(root)
    groups = defaultdict(dict)
    synth_flag = {}
    for p in files:
        role, syn = _role(p, root_for_roles)
        b = _base_stem(p.stem)
        groups[b].setdefault(role, p)
        synth_flag[b] = synth_flag.get(b, False) or syn or role in ("clean", "noise")
    # metadata may point to the clean / noise files explicitly
    by_name = {p.name.lower(): p for p in files}
    for b, r in meta.items():
        for k, v in r.items():
            if isinstance(v, str) and v.lower().split("/")[-1] in by_name:
                kl = k.lower()
                role = "clean" if "clean" in kl else ("noise" if "noise" in kl and "noisy" not in kl else None)
                if role:
                    groups[b][role] = by_name[v.lower().split("/")[-1]]; synth_flag[b] = True
    items = []
    for b, g in sorted(groups.items()):
        r = meta.get(b, {})
        items.append(dict(
            id=_strip_stem(g.get("noisy", g.get("clean", g.get("noise"))).stem) if "noisy" in g else b,
            noisy=g.get("noisy"), clean=g.get("clean"), noise=g.get("noise"),
            syn=bool(synth_flag.get(b)) and ("clean" in g),
            gt=_pick(r, "transcript", "text", "ground_truth", "reference_text", "sentence", "groundTruth"),
            group=str(_pick(r, "speakerId", "speaker_id", "speaker", "district") or ""),
            meta=r))
    n_syn = sum(i["syn"] for i in items)
    print(f"validation: {len(items)} clips | synthetic with clean ref {n_syn} | "
          f"with noise track {sum(1 for i in items if i['noise'] is not None)} | with transcript {sum(1 for i in items if i['gt'])}")
    return items

VAL_ITEMS = load_validation(VAL_ROOT)
for it in VAL_ITEMS[:3]:
    print({k: (str(v)[:80] if k != "meta" else list(v)[:8]) for k, v in it.items()})

# ============================== FOLDS (group split, held out fold is untouchable) ==============================
def group_key(stem):
    toks = re.split(r"[_\-]", _strip_stem(stem).lower())
    return "_".join(toks[:-1]) if len(toks) > 2 else _strip_stem(stem).lower()

def fold_of(key, k=5):
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % k

for it in VAL_ITEMS:
    it["fold"] = fold_of(it["group"] or group_key(it["id"]))
    it["holdout"] = it["fold"] == 0

TRAIN_PAIRS = [i for i in VAL_ITEMS if i["syn"] and not i["holdout"]]
HOLD_SYN = [i for i in VAL_ITEMS if i["syn"] and i["holdout"]]
NATURAL = [i for i in VAL_ITEMS if not i["syn"]]
print(f"official synthetic pairs: train fold {len(TRAIN_PAIRS)} | held out {len(HOLD_SYN)} | natural {len(NATURAL)}")

def load_pair(it):
    """Returns noisy, clean, noise arrays aligned to the same length."""
    c = read_audio(it["clean"]) if it["clean"] is not None else None
    n = read_audio(it["noise"]) if it["noise"] is not None else None
    x = read_audio(it["noisy"]) if it["noisy"] is not None else None
    if c is not None:
        L = min(len(v) for v in (c, n, x) if v is not None)
        c = c[:L]; n = n[:L] if n is not None else None; x = x[:L] if x is not None else None
        if x is None and n is not None:
            x = c + n
        if n is None and x is not None:
            n = x - c
    return x, c, n

# ============================== CALIBRATION FROM TRAIN FOLD ONLY ==============================
PRIORS = dict(floor_snr=(8.0, 35.0), event_snr=(-6.0, 15.0), event_dur=(0.3, 4.0),
              mix_rms_db=(-36.0, -18.0), in_sisdr=(0.0, 18.0))

class Calib:
    def __init__(self):
        self.v = defaultdict(list)
    def add(self, k, x):
        if x is not None and np.isfinite(x):
            self.v[k].append(float(x))
    def sample(self, k, rng, widen=0.0, jitter=1.5):
        arr = self.v.get(k)
        lo, hi = PRIORS[k]
        if not arr or rng.random() < widen:
            if k == "event_dur":
                return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            return float(rng.uniform(lo, hi))
        x = float(arr[int(rng.integers(0, len(arr)))])
        if k == "event_dur":
            return float(np.clip(x * np.exp(rng.normal(0, 0.25)), 0.1, 10.0))
        return float(x + rng.normal(0, jitter))
    def n_events(self, rng, lo, hi, dur=None):
        rate = self.v.get("ev_rate")
        if dur is not None and rate:                     # events per second, scaled to the crop
            r = rate[int(rng.integers(0, len(rate)))]
            return int(np.clip(rng.poisson(max(0.05, r * dur)), lo, hi))
        arr = self.v.get("n_events")
        if arr and rng.random() > 0.3:
            return int(np.clip(arr[int(rng.integers(0, len(arr)))], lo, hi))
        return int(rng.integers(lo, hi + 1))
    def summary(self):
        for k, a in self.v.items():
            a = np.array(a)
            print(f"  {k:12s} n={len(a):5d} p10={np.percentile(a,10):7.2f} p50={np.percentile(a,50):7.2f} p90={np.percentile(a,90):7.2f}")

CAL = Calib()
OFFICIAL_INPUT_SISDR = []
for it in TRAIN_PAIRS:
    x, c, n = load_pair(it)
    if c is None or n is None:
        continue
    wins, _ = windows_from_noise(n)
    pc = np.mean(c ** 2) + 1e-10
    ev = np.zeros(len(n), bool)
    for a, b in wins:
        ev[int(a * SR):int(b * SR)] = True
        seg = n[int(a * SR):int(b * SR)]
        if len(seg) > 160:
            CAL.add("event_snr", 10 * np.log10(pc / (np.mean(seg ** 2) + 1e-10)))
            CAL.add("event_dur", b - a)
    if (~ev).sum() > SR * 0.2:
        CAL.add("floor_snr", 10 * np.log10(pc / (np.mean(n[~ev] ** 2) + 1e-10)))
    CAL.add("n_events", len(wins))
    CAL.add("ev_rate", len(wins) / max(0.5, len(n) / SR))
    CAL.add("mix_rms_db", 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9))
    CAL.add("in_sisdr", si_sdr(c, x))
    OFFICIAL_INPUT_SISDR.append(si_sdr(c, x))
print("calibration from", len(TRAIN_PAIRS), "train fold pairs")
CAL.summary()
if OFFICIAL_INPUT_SISDR:
    print("official input SI-SDR (train fold): mean %.2f dB" % np.mean(OFFICIAL_INPUT_SISDR))

# ============================== LEAKAGE GUARD ==============================
def _all_stems(root):
    if root is None:
        return set()
    return {_strip_stem(p.name).lower() for p in Path(root).rglob("*") if p.suffix.lower() in AUD + (".parquet",)}

BLOCK_STEMS = _all_stems(VAL_ROOT) | _all_stems(TEST_ROOT) | {k.lower() for k in T1.keys()}
BLOCK_STEMS |= {_base_stem(s).lower() for s in list(BLOCK_STEMS)}
_gc = Counter(group_key(s) for s in BLOCK_STEMS)
BLOCK_GROUPS = {g for g, c in _gc.items() if c <= 50 and g not in BLOCK_STEMS}   # coarse keys would block whole districts
print(f"blocklist: {len(BLOCK_STEMS)} stems, {len(BLOCK_GROUPS)} groups (dropped {sum(1 for c in _gc.values() if c > 50)} coarse groups)")
print("  example stems:", sorted(BLOCK_STEMS)[:3], "| example groups:", sorted(BLOCK_GROUPS)[:3])
LEAK_SKIPPED = Counter()

def is_blocked(name, src):
    if not name:
        return False
    st = _strip_stem(name).lower()
    if st in BLOCK_STEMS or _base_stem(st).lower() in BLOCK_STEMS:
        LEAK_SKIPPED[src + ":stem"] += 1
        return True
    if STRICT_GROUP_EXCLUDE and group_key(st) in BLOCK_GROUPS:
        LEAK_SKIPPED[src + ":group"] += 1
        return True
    return False

# ============================== POOLS (int16 on disk, memmap for workers) ==============================
class PoolWriter:
    def __init__(self, name):
        self.name = name
        self.bin = POOL_DIR / f"{name}.bin"
        self.idx_path = POOL_DIR / f"{name}.json"
        self.index = json.load(open(self.idx_path)) if self.idx_path.exists() else []
        self.off = self.bin.stat().st_size // 2 if self.bin.exists() else 0
        self.f = open(self.bin, "ab")
        self.sec = Counter()
        for r in self.index:
            self.sec[r.get("src")] += r["n"] / SR
    def add(self, w, **meta):
        x = (np.clip(np.nan_to_num(w), -1, 1) * 32767).astype(np.int16)
        self.f.write(x.tobytes())
        self.index.append(dict(o=int(self.off), n=int(len(x)), **meta))
        self.off += len(x)
        self.sec[meta.get("src")] += len(x) / SR
    def seconds(self, src=None):
        return sum(self.sec.values()) if src is None else self.sec.get(src, 0.0)
    def close(self):
        self.f.close()
        json.dump(self.index, open(self.idx_path, "w"))

class Pool:
    def __init__(self, name):
        self.name = name
        p = POOL_DIR / f"{name}.json"
        self.index = json.load(open(p)) if p.exists() else []
        self._mm = None
        self.by_src = defaultdict(list)
        for i, r in enumerate(self.index):
            self.by_src[r.get("src", "")].append(i)
    def _m(self):
        if self._mm is None and self.index:
            self._mm = np.memmap(POOL_DIR / f"{self.name}.bin", dtype=np.int16, mode="r")
        return self._mm
    def __len__(self):
        return len(self.index)
    def get(self, i):
        r = self.index[i]
        return np.asarray(self._m()[r["o"]:r["o"] + r["n"]], dtype=np.float32) / 32768.0
    def ids(self, srcs):
        out = []
        for s in srcs:
            out += self.by_src.get(s, [])
        return out

def reset_pools(names):
    for n in names:
        for ext in (".bin", ".json"):
            p = POOL_DIR / f"{n}{ext}"
            if p.exists():
                p.unlink()

# ============================== OFFICIAL TRAIN FOLD -> pools ==============================
# This cluster's COMPUTE NODES HAVE NO NETWORK (verified: huggingface.co http=000),
# so the gold and Vaani pool builds -- which stream parquet row-groups from
# the Hub -- can only run on the login node. Set T2_POOLS_READY=1 to reuse pools
# that were built there, instead of wiping and rebuilding them.
POOLS_READY = os.environ.get("T2_POOLS_READY", "0") == "1"
if POOLS_READY:
    _need = [POOL_DIR / f"{n}.json" for n in ("event", "floor", "offc", "offn")]
    _missing = [p.name for p in _need if not p.exists()]
    if _missing:
        raise SystemExit(f"T2_POOLS_READY=1 but pools are missing: {_missing}")
    print("reusing pre-built pools from", POOL_DIR)
else:
    reset_pools(["offc", "offn", "floor", "event", "speech", "gclean"])
W_OFFC, W_OFFN = PoolWriter("offc"), PoolWriter("offn")
W_FLOOR, W_EVENT = PoolWriter("floor"), PoolWriter("event")   # shared by official, gold and Vaani sources
for it in ([] if POOLS_READY else TRAIN_PAIRS):
    x, c, n = load_pair(it)
    if c is None or n is None:
        continue
    W_OFFC.add(c, src="official", id=it["id"])
    W_OFFN.add(n, src="official", id=it["id"])
    wins, _ = windows_from_noise(n)
    ev = np.zeros(len(n), bool)
    for a, b in wins:
        ev[int(a * SR):int(b * SR)] = True
        seg = n[max(0, int((a - 0.02) * SR)):int((b + 0.02) * SR)]
        if len(seg) > 0.15 * SR:
            W_EVENT.add(seg, src="official", cat="official")
    quiet = n[~ev]
    if len(quiet) > 0.5 * SR:
        W_FLOOR.add(quiet[: 8 * SR], src="official")
W_OFFC.close(); W_OFFN.close()
print(f"official train fold -> pairs {len(W_OFFC.index)}, floors {W_FLOOR.seconds('official'):.0f}s, bursts {W_EVENT.seconds('official'):.0f}s")

# ============================== HF SELECTIVE READER ==============================
from huggingface_hub import HfApi, HfFileSystem
HF_API = HfApi(token=HF_TOKEN)
HF_FS = HfFileSystem(token=HF_TOKEN)

def hf_ls(repo, path="", recursive=False):
    from huggingface_hub.hf_api import RepoFolder
    dirs, files = [], []
    try:
        for e in HF_API.list_repo_tree(repo, path_in_repo=path or None, recursive=recursive, repo_type="dataset"):
            (dirs if isinstance(e, RepoFolder) else files).append(e.path)
    except Exception as ex:
        print("list failed", repo, path, ex)
    return dirs, files

def iter_parquet_rowgroups(repo, path, meta_cols, rng, max_groups):
    """Yield (meta_rows, audio_cells) for randomly chosen row groups. Only those bytes are fetched."""
    with HF_FS.open(f"datasets/{repo}/{path}", "rb", block_size=16 * 2 ** 20) as fh:
        pf = pq.ParquetFile(fh)
        names = set(pf.schema_arrow.names)
        mcols = [c for c in meta_cols if c in names]
        acol = "audio" if "audio" in names else next((c for c in names if "audio" in c.lower()), None)
        groups = list(range(pf.num_row_groups))
        rng.shuffle(groups)
        for g in groups[:max_groups]:
            meta = pf.read_row_group(g, columns=mcols).to_pylist() if mcols else [{}] * pf.metadata.row_group(g).num_rows
            yield g, meta, (lambda gg=g: pf.read_row_group(gg, columns=[acol]).column(acol).to_pylist())

def cell_name(cell):
    if isinstance(cell, dict):
        return cell.get("path") or ""
    return ""

def cell_bytes(cell):
    if isinstance(cell, dict):
        return cell.get("bytes")
    return None

def _truthy(v):
    return str(v).strip().lower() in ("true", "yes", "1", "y")

# ============================== GOLD POOL (real events, clean gaps, in-domain floors) ==============================
def _spans(ts):
    out = []
    for s in (ts or []):
        try:
            a, b = float(s.get("start")), float(s.get("end"))
        except Exception:
            continue
        if b > a:
            out.append((a, b, str(s.get("category") or s.get("tag") or "")))
    return sorted(out)

def _gaps(spans, dur, pad=0.05):
    free, cur = [], 0.0
    for a, b, _ in spans:
        a, b = max(0.0, a - pad), min(dur, b + pad)
        if a > cur:
            free.append((cur, a))
        cur = max(cur, b)
    if cur < dur:
        free.append((cur, dur))
    return free

def build_gold_pool():
    rng = np.random.default_rng(SEED + 1)
    _, files = hf_ls(GOLD_REPO, "", recursive=True)
    files = [f for f in files if f.endswith(".parquet") and "test" not in f.lower()]
    print("gold parquet files:", len(files))
    rng.shuffle(files)
    W_GC = PoolWriter("gclean")
    W_SP = PoolWriter("speech")          # tag free gold rows are clean speech, same corpus as the eval
    district_counts, n_rows, n_sp, t0 = Counter(), 0, 0, time.time()

    def _accept_speech(w):
        d = len(w) / SR
        if not (SEL["min_dur"] <= d <= SEL["max_dur"] * 3):
            return False
        if np.mean(np.abs(w) > 0.99) > SEL["max_clip_frac"] or est_snr_db(w) < SEL["snr_tagless"]:
            return False
        return speech_ratio(w) >= SEL["min_vad"]

    cols = ["annotationQuality", "NoiseSubCategoryTimeStamp", "state", "district", "transcript"]
    for fi, f in enumerate(files[: B["gold_files"]]):
        if n_rows >= B["gold_rows"]:
            break
        try:
            for g, meta, get_audio in iter_parquet_rowgroups(GOLD_REPO, f, cols, rng, max_groups=B.get("gold_rg", 6)):
                for r in meta:
                    district_counts[(str(r.get("state", "")), str(r.get("district", "")))] += 1
                sel = [i for i, r in enumerate(meta)
                       if r.get("annotationQuality") == "verified_timestamps"
                       or (str(r.get("transcript") or "").strip()
                           and not TAG_RE.search(str(r.get("transcript") or "")))]
                if not sel:
                    continue
                cells = get_audio()
                for i in sel:
                    if n_rows >= B["gold_rows"]:
                        break
                    if is_blocked(cell_name(cells[i]), "gold"):
                        continue
                    try:
                        w = decode_bytes(cell_bytes(cells[i]))
                    except Exception:
                        continue
                    dur = len(w) / SR
                    if meta[i].get("annotationQuality") != "verified_timestamps":
                        gk = group_key(cell_name(cells[i]) or f"g{n_sp}")
                        if _accept_speech(w):
                            W_SP.add(w[: int(SEL["max_dur"] * SR)], src="goldspeech",
                                     district=str(meta[i].get("district", "")), snr=round(est_snr_db(w), 1))
                            n_sp += 1
                        continue
                    n_rows += 1
                    sp = _spans(meta[i].get("NoiseSubCategoryTimeStamp"))
                    for a, b, cat in sp:
                        a2, b2 = max(0.0, a - EVT["margin"]), min(dur, b + EVT["margin"])
                        if not (EVT["min_dur"] <= b2 - a2 <= EVT["max_dur"]):
                            continue
                        seg = w[int(a2 * SR):int(b2 * SR)]
                        if np.abs(seg).max() < 1e-3 or speech_ratio(seg) > EVT["max_speech_ratio"]:
                            continue
                        W_EVENT.add(seg, src="gold", cat=cat)
                    for a, b in _gaps(sp, dur):
                        if b - a >= EVT["min_clean_gap"]:
                            seg = w[int(a * SR):int(b * SR)]
                            if np.abs(seg).max() > 1e-3:
                                W_GC.add(seg, src="gclean")
                    fl = extract_floor(w, avoid=[(a, b) for a, b, _ in sp])
                    if fl is not None:
                        W_FLOOR.add(fl, src="gold")
                DISK.check("gold")
                if (time.time() - t0) / 60 > GOLD_MAX_MIN:
                    break
        except DiskBudgetExceeded:
            raise
        except Exception as ex:
            print("gold file failed", f, ex)
        if fi % 10 == 0:
            print(f"  gold file {fi}: verified {n_rows} | speech {n_sp} | events {W_EVENT.seconds('gold'):.0f}s | "
                  f"gclean {W_GC.seconds():.0f}s | goldspeech {W_SP.seconds()/3600:.2f} h | "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
        if (time.time() - t0) / 60 > GOLD_MAX_MIN:
            print("  gold time budget reached"); break
    W_GC.close(); W_SP.close()
    print(f"gold done: {n_rows} verified rows, {n_sp} tag free speech rows, "
          f"events {W_EVENT.seconds('gold')/60:.1f} min, clean gaps {W_GC.seconds()/60:.1f} min, "
          f"gold speech {W_SP.seconds()/3600:.2f} h, floors {W_FLOOR.seconds('gold')/60:.1f} min")
    return district_counts

GOLD_DISTRICTS = Counter() if POOLS_READY else build_gold_pool()
print("top gold districts:", GOLD_DISTRICTS.most_common(8))

# ============================== VAANI SPEECH + BACKGROUND POOL ==============================
def _norm_name(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def list_vaani_districts():
    """Walks audio/<State>/<District> one level at a time (cheap)."""
    top, _ = hf_ls(VAANI_REPO, "")
    base = next((d for d in top if d.lower().endswith("audio")), None)
    if base is None:
        print("Vaani top level:", top[:20])
        return {}
    out = {}
    states, _ = hf_ls(VAANI_REPO, base)
    for st in states:
        dists, _ = hf_ls(VAANI_REPO, st)
        for d in dists:
            out[(_norm_name(Path(st).name), _norm_name(Path(d).name))] = d
    return out

def choose_districts(avail):
    rng = np.random.default_rng(SEED + 2)
    chosen = []
    for (st, di), _ in GOLD_DISTRICTS.most_common():
        key = (_norm_name(st), _norm_name(di))
        if key in avail and avail[key] not in chosen:
            chosen.append(avail[key])
        if len(chosen) >= B["n_dist_in"]:
            break
    rest = [v for v in avail.values() if v not in chosen]
    rng.shuffle(rest)
    chosen += rest[: B["n_dist_rand"] + max(0, B["n_dist_in"] - len(chosen))]
    return chosen

FLOOR_HOURS_CAP = max(0.2, B["speech_hours_cap"] / 4)

class SpeechSelector:
    def __init__(self):
        self.W = PoolWriter("speech")
        self.group_count = Counter()
        self.stats = Counter()
    def process(self, w, name, meta, district):
        s = self.stats
        s["seen"] += 1
        dur = len(w) / SR
        tr = str(meta.get("transcript") or "")
        has_tr = _truthy(meta.get("isTranscriptionAvailable")) and bool(tr.strip())
        tagged = bool(TAG_RE.search(tr)) if has_tr else False
        if self.W_floor.seconds("vaani") < FLOOR_HOURS_CAP * 3600 and s["seen"] % 2 == 0:
            fl = extract_floor(w)
            if fl is not None:
                self.W_floor.add(fl, src="vaani", district=district)
        if tagged:
            s["tagged"] += 1; return
        if not (SEL["min_dur"] <= dur <= SEL["max_dur"] * 3):
            s["dur"] += 1; return
        if np.mean(np.abs(w) > 0.99) > SEL["max_clip_frac"]:
            s["clip"] += 1; return
        snr = est_snr_db(w)
        if snr < (SEL["snr_tagless"] if has_tr else SEL["snr_untagged"]):
            s["snr"] += 1; return
        if speech_ratio(w) < SEL["min_vad"]:
            s["vad"] += 1; return
        gk = district + "|" + group_key(name or f"x_{s['seen']}")
        if self.group_count[gk] >= SEL["per_group_cap"]:
            s["cap"] += 1; return
        self.group_count[gk] += 1
        self.W.add(w[: int(SEL["max_dur"] * SR)], src="vaani", district=district, tagless=has_tr, snr=round(snr, 1))
        s["kept"] += 1

def stream_fallback(sel):
    """Fallback when the repo is not laid out as audio/<State>/<District>/*.parquet."""
    try:
        from datasets import load_dataset, get_dataset_config_names, Audio
        cfgs = [c for c in get_dataset_config_names(VAANI_REPO, token=HF_TOKEN) if c.lower().startswith("audio")]
    except Exception as ex:
        print("config listing failed:", ex); return
    avail = {(_norm_name(c.split("/")[-2] if c.count("/") >= 2 else ""), _norm_name(c.split("/")[-1])): c for c in cfgs}
    for cfg in choose_districts(avail):
        if sel.W.seconds() / 3600 >= B["speech_hours_cap"]:
            break
        try:
            ds = load_dataset(VAANI_REPO, cfg, split="train", streaming=True, token=HF_TOKEN)
            ds = ds.cast_column("audio", Audio(decode=False)).shuffle(seed=SEED, buffer_size=64)
            before = sel.stats["kept"]
            for k, ex in enumerate(ds):
                if sel.stats["kept"] - before >= B["clips_per_dist"] or k >= 4 * B["clips_per_dist"]:
                    break
                nm = cell_name(ex["audio"])
                if is_blocked(nm, "vaani"):
                    continue
                try:
                    w = decode_bytes(cell_bytes(ex["audio"]))
                except Exception:
                    continue
                sel.process(w, nm, ex, cfg.split("/")[-1])
            DISK.check("vaani-stream")
            shutil.rmtree(HF_HOME / "datasets", ignore_errors=True)
            print(f"  {cfg}: kept {sel.stats['kept']-before} | speech {sel.W.seconds()/3600:.2f} h")
        except DiskBudgetExceeded:
            raise
        except Exception as ex:
            print("  stream failed", cfg, ex)

def build_vaani_pool():
    avail = list_vaani_districts()
    print("Vaani districts available:", len(avail))
    sel = SpeechSelector(); sel.W_floor = W_FLOOR
    if not avail:
        print("Could not list Vaani parquet tree. Trying datasets streaming by config.")
        stream_fallback(sel)
        sel.W.close(); return sel
    chosen = choose_districts(avail)
    print("chosen districts:", len(chosen), chosen[:6], "...")
    rng = np.random.default_rng(SEED + 3)
    cols = ["language", "gender", "state", "district", "isTranscriptionAvailable", "transcript"]
    t0 = time.time()
    for di, dpath in enumerate(chosen):
        if sel.W.seconds() / 3600 >= B["speech_hours_cap"]:
            break
        _, files = hf_ls(VAANI_REPO, dpath, recursive=True)
        pqs = [f for f in files if f.endswith(".parquet")]
        if not pqs:
            print("  no parquet under", dpath, "->", files[:3]); continue
        rng.shuffle(pqs)
        kept_before = sel.stats["kept"]
        quota = B["clips_per_dist"]
        per_group = max(4, 2 * quota // (B["files_per_dist"] * B["rg_per_file"]))
        for f in pqs[: B["files_per_dist"]]:
            try:
                for g, meta, get_audio in iter_parquet_rowgroups(VAANI_REPO, f, cols, rng, B["rg_per_file"]):
                    idx = list(range(len(meta)))
                    rng.shuffle(idx)
                    # prefer transcribed tag free rows, then untranscribed rows
                    idx.sort(key=lambda i: 0 if (_truthy(meta[i].get("isTranscriptionAvailable"))
                                                 and not TAG_RE.search(str(meta[i].get("transcript") or ""))) else 1)
                    idx = idx[:per_group]
                    cells = get_audio()
                    for i in idx:
                        nm = cell_name(cells[i])
                        if is_blocked(nm, "vaani"):
                            continue
                        try:
                            w = decode_bytes(cell_bytes(cells[i]))
                        except Exception:
                            continue
                        sel.process(w, nm, meta[i], Path(dpath).name)
                    del cells
                    DISK.check("vaani")
                    if sel.stats["kept"] - kept_before >= quota:
                        break
            except DiskBudgetExceeded:
                raise
            except Exception as ex:
                if "403" in str(ex) or "restricted" in str(ex).lower():
                    sel.stats["gated"] += 1
                    if sel.stats["gated"] >= 3 and sel.stats["kept"] == 0:
                        print("\n  STOP: your token cannot read", VAANI_REPO,
                              "\n  Ask for access at https://huggingface.co/datasets/ARTPARK-IISc/Vaani"
                              "\n  Training continues with gold corpus speech only.\n")
                        return sel
                else:
                    print("  vaani file failed", f, ex)
            if sel.stats["kept"] - kept_before >= quota:
                break
        print(f"  [{di+1}/{len(chosen)}] {dpath}: kept {sel.stats['kept']-kept_before} | "
              f"speech {sel.W.seconds()/3600:.2f} h | floors {W_FLOOR.seconds('vaani')/60:.0f} min | "
              f"{(time.time()-t0)/60:.1f} min | disk {DISK.check(force=True):.2f} GB")
        gc.collect()
    sel.W.close()
    print("selection stats:", dict(sel.stats))
    return sel

SEL_RESULT = None if POOLS_READY else build_vaani_pool()
if not POOLS_READY:
    W_FLOOR.close(); W_EVENT.close()
print("leakage skips:", dict(LEAK_SKIPPED))

# ============================== POOL REPORT (diversity) ==============================
POOLS = {n: Pool(n) for n in ["speech", "gclean", "floor", "event", "offc", "offn"]}
for n, p in POOLS.items():
    secs = sum(r["n"] for r in p.index) / SR
    print(f"{n:7s} items {len(p):6d}  hours {secs/3600:6.2f}  sources {dict(Counter(r.get('src') for r in p.index))}")
if os.environ.get("T2_POOLS_ONLY", "0") == "1":
    # Pool building is the only stage that needs the Hub, and only the LOGIN node
    # has outbound internet here (compute nodes resolve DNS but time out on :443,
    # including to 1.1.1.1 -- verified). So build pools here, then run training on
    # GPU with T2_POOLS_READY=1.
    print("T2_POOLS_ONLY: pools built, exiting before training")
    raise SystemExit(0)

sp = POOLS["speech"].index
if sp:
    print("speech districts:", len(set(r['district'] for r in sp)),
          "| tag free transcribed share %.2f" % np.mean([r.get("tagless", False) for r in sp]),
          "| median est SNR %.1f dB" % np.median([r["snr"] for r in sp]))
print("event categories:", Counter(r.get("cat") for r in POOLS["event"].index).most_common(10))
print("disk used: %.2f GB" % DISK.check(force=True))

# ============================== MIXING ENGINE ==============================
BASE_MODE = dict(p_off=0.0, remix=0.5, speech=["vaani"], floor=["vaani"], event=["gold"], n_ev=(1, 2),
                 floor_p=1.0, snr_mode="calib", widen=0.0, reverb_p=0.0, eq_p=0.0, mod_p=0.0,
                 noevent_p=0.0, overlap=False, loud="calib")
MODES = {
    # A: current style synthetic data (gold clean gaps + pasted gold events, no floor)
    "A": dict(speech=["gclean"], floor=[], event=["gold"], floor_p=0.0, snr_mode="local", loud="peak"),
    # B: official train fold pairs only (random crop, gain, remix within train fold)
    "B": dict(p_off=1.0),
    # C: official + Vaani speech and Vaani/official floors, events from official bursts only
    "C": dict(p_off=0.3, speech=["vaani", "goldspeech"], floor=["vaani", "official"], event=["official"], widen=0.1),
    # D: C + real gold event snippets + gold clean gaps + gold floors
    "D": dict(p_off=0.3, speech=["vaani", "goldspeech", "gclean"], floor=["vaani", "official", "gold"],
              event=["official", "gold"], n_ev=(1, 3), widen=0.15),
    # E: D + wide SNRs, 0..4 events with overlap, no event clips, floor modulation, reverb, channel EQ
    # F: E, but the official organiser pairs dominate. The test set is entirely
    # their generator; at p_off=0.25 three quarters of training came from ours,
    # which is the remaining local-vs-leaderboard gap (held-out 21.97 dB ->
    # test 16.41 dB) now that the loader bug is fixed.
    "F": dict(p_off=0.65, speech=["vaani", "goldspeech", "gclean"],
              floor=["vaani", "official", "gold"], event=["official", "gold"],
              n_ev=(0, 4), widen=0.35, reverb_p=0.35, eq_p=0.5, mod_p=0.5,
              noevent_p=0.12, overlap=True),
    "E": dict(p_off=0.25, speech=["vaani", "goldspeech", "gclean"], floor=["vaani", "official", "gold"],
              event=["official", "gold"], n_ev=(0, 4), widen=0.35, reverb_p=0.35, eq_p=0.5, mod_p=0.5,
              noevent_p=0.12, overlap=True),
}
MODES["P"] = dict(MODES["E"], p_off=0.6)   # polish stage: closer to the official distribution

def rand_rir(rng):
    rt = rng.uniform(0.15, 0.7)
    n = int(rt * SR)
    t = np.arange(n) / SR
    h = rng.standard_normal(n) * np.exp(-6.9 * t / rt)
    h[0] = np.abs(h).max() * rng.uniform(1.5, 4.0)
    return (h / np.sqrt((h ** 2).sum())).astype(np.float32)

def apply_rir(x, h):
    y = sps.fftconvolve(x, h)[: len(x)]
    return (y * np.sqrt((np.mean(x ** 2) + 1e-12) / (np.mean(y ** 2) + 1e-12))).astype(np.float32)

def rand_eq(rng):
    sos = []
    if rng.random() < 0.6:
        sos.append(sps.butter(2, rng.uniform(50, 250), "highpass", fs=SR, output="sos"))
    if rng.random() < 0.6:
        sos.append(sps.butter(4, rng.uniform(3800, 7800), "lowpass", fs=SR, output="sos"))
    return np.concatenate(sos) if sos else None

def slow_env(L, rng, depth_db=3.0):
    k = max(2, int(L / SR * 2) + 2)
    pts = rng.uniform(-depth_db, depth_db, k)
    return (10 ** (np.interp(np.arange(L), np.linspace(0, L - 1, k), pts) / 20)).astype(np.float32)

class Mixer:
    def __init__(self, mode, L):
        self.m = dict(BASE_MODE, **MODES[mode])
        self.L = L
        self.p = {n: Pool(n) for n in ["speech", "gclean", "floor", "event", "offc", "offn"]}
        spk = []
        for s in self.m["speech"]:
            pn = "gclean" if s == "gclean" else "speech"
            spk += [(pn, i) for i in self.p[pn].ids([s])]
        if not spk:
            spk = [("gclean", i) for i in range(len(self.p["gclean"]))] or [("offc", i) for i in range(len(self.p["offc"]))]
        long_ = [(pn, i) for pn, i in spk if self.p[pn].index[i]["n"] >= L]
        self.spk = long_ if len(long_) >= 50 else spk
        self.floor_ids = self.p["floor"].ids(self.m["floor"])
        self.event_ids = self.p["event"].ids(self.m["event"]) or list(range(len(self.p["event"])))
        self.n_off = len(self.p["offc"])
        if self.m["p_off"] > 0 and self.n_off == 0:
            self.m["p_off"] = 0.0
        if not self.spk:
            self.m["p_off"] = 1.0
        assert self.spk or self.n_off, "no speech source at all"

    def _speech(self, rng):
        L = self.L
        pn, i = self.spk[int(rng.integers(len(self.spk)))]
        w = self.p[pn].get(i)
        cp = float(np.mean(w ** 2)) + 1e-10
        if len(w) >= L:
            best = None
            for _ in range(3):
                s = int(rng.integers(0, len(w) - L + 1))
                seg = w[s:s + L]
                e = float(np.mean(seg ** 2))
                if best is None or e > best[0]:
                    best = (e, seg)
                if e > 0.3 * cp:
                    break
            seg = best[1].copy()
        else:
            seg = np.zeros(L, np.float32)
            off = int(rng.integers(0, L - len(w) + 1))
            seg[off:off + len(w)] = w
        return seg, cp

    def _official(self, rng):
        L = self.L
        i = int(rng.integers(self.n_off))
        c, nz = self.p["offc"].get(i), self.p["offn"].get(i)
        n0 = min(len(c), len(nz))
        c, nz = c[:n0], nz[:n0]
        if n0 >= L:
            s = int(rng.integers(0, n0 - L + 1))
            c, nz = c[s:s + L].copy(), nz[s:s + L].copy()
        else:
            c, nz = np.pad(c, (0, L - n0)), np.pad(nz, (0, L - n0))
        if rng.random() < self.m["remix"]:
            nz = nz * 10 ** (rng.uniform(-4, 4) / 20)
            r = rng.random()
            if r < 0.3 and self.n_off > 1:
                nz = fit_len(self.p["offn"].get(int(rng.integers(self.n_off))), L, rng)
            elif r < 0.6:
                nz = np.roll(nz, int(rng.integers(L)))
        g = 10 ** (rng.uniform(-6, 6) / 20)
        return (c * g).astype(np.float32), (nz * g).astype(np.float32)

    def make(self, rng):
        L, m = self.L, self.m
        if rng.random() < m["p_off"]:
            clean, noise = self._official(rng)
            wins = windows_from_noise(noise)[0] if np.any(noise) else []
        else:
            clean, cp = self._speech(rng)
            pref = max(float(np.mean(clean ** 2)), 0.3 * cp, 1e-8)
            noise = np.zeros(L, np.float32)
            if self.floor_ids and rng.random() < m["floor_p"]:
                fw = fit_len(self.p["floor"].get(self.floor_ids[int(rng.integers(len(self.floor_ids)))]), L, rng)
                if rng.random() < m["mod_p"]:
                    fw = fw * slow_env(L, rng)
                snr = CAL.sample("floor_snr", rng, m["widen"])
                noise += fw * math.sqrt(pref / ((np.mean(fw ** 2) + 1e-10) * 10 ** (snr / 10)))
            lo, hi = m["n_ev"]
            k = (CAL.n_events(rng, lo, hi, dur=L / SR) if m["snr_mode"] == "calib"
                 else int(rng.integers(lo, hi + 1)))
            if rng.random() < m["noevent_p"]:
                k = 0
            wins, occ = [], []
            for _ in range(k if self.event_ids else 0):
                ew = self.p["event"].get(self.event_ids[int(rng.integers(len(self.event_ids)))])
                if m["snr_mode"] == "local":
                    dur = min(len(ew) / SR, 3.0)
                else:
                    dur = CAL.sample("event_dur", rng, m["widen"])
                    if len(ew) / SR < dur and rng.random() < 0.5:
                        dur = len(ew) / SR
                n = int(np.clip(dur, 0.15, L / SR) * SR)
                seg = fit_len(ew, n, rng)
                if rng.random() < m["reverb_p"]:
                    seg = apply_rir(seg, rand_rir(rng))
                fi = int(rng.integers(16, max(17, min(int(0.08 * SR), n // 2))))
                seg[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)
                seg[-fi:] *= np.linspace(1, 0, fi, dtype=np.float32)
                start = None
                for _t in range(20):
                    cand = int(rng.integers(-int(0.3 * n), max(-int(0.3 * n) + 1, L - int(0.7 * n))))
                    a, b = max(0, cand), min(L, cand + n)
                    if m["overlap"] or all(b <= x or a >= y for x, y in occ):
                        start = cand; break
                if start is None or b - a < 0.1 * SR:
                    continue
                if m["snr_mode"] == "local":
                    snr = rng.uniform(-5, 12)
                    pc = max(float(np.mean(clean[a:b] ** 2)), 0.1 * pref, 1e-10)
                else:
                    snr = CAL.sample("event_snr", rng, m["widen"])
                    pc = pref
                piece = seg[a - start:b - start]
                noise[a:b] += piece * math.sqrt(pc / ((np.mean(piece ** 2) + 1e-10) * 10 ** (snr / 10)))
                occ.append((a, b)); wins.append((a / SR, b / SR))
            wins = sorted(wins)
            # Match the overall mixture SNR to the official distribution. The floor and the
            # bursts keep their relative balance, only the total noise level is set.
            if m["snr_mode"] == "calib" and np.any(noise):
                cur = 10 * np.log10((np.mean(clean ** 2) + 1e-12) / (np.mean(noise ** 2) + 1e-12))
                tgt = CAL.sample("in_sisdr", rng, m["widen"], jitter=1.0)
                noise = noise * (10 ** ((cur - tgt) / 20.0))
            if m["loud"] == "calib":
                mix = clean + noise
                tgt = 10 ** (CAL.sample("mix_rms_db", rng, m["widen"]) / 20)
                g = tgt / (np.sqrt(np.mean(mix ** 2)) + 1e-9)
            else:
                g = rng.uniform(0.3, 0.85) / (np.abs(clean).max() + 1e-9)
            clean, noise = clean * g, noise * g
        if rng.random() < m["eq_p"]:
            sos = rand_eq(rng)
            if sos is not None:
                clean = sps.sosfilt(sos, clean).astype(np.float32)
                noise = sps.sosfilt(sos, noise).astype(np.float32)
        mix = clean + noise
        pk = float(np.abs(mix).max())
        if pk > 0.98:
            clean, noise, mix = clean * 0.98 / pk, noise * 0.98 / pk, mix * 0.98 / pk
        return mix.astype(np.float32), clean.astype(np.float32), noise.astype(np.float32), wins

# self check: consistency, levels and match to official input SI-SDR
L_TRAIN = int(SEG_SEC * SR)
for mode in ["A", "B", "C", "D", "E"]:
    try:
        mx = Mixer(mode, L_TRAIN)
    except AssertionError as ex:
        print(mode, "unavailable:", ex); continue
    rng = np.random.default_rng(7)
    sis, rms, nev = [], [], []
    for _ in range(40):
        x, c, nz, w = mx.make(rng)
        assert np.isfinite(x).all() and np.abs(x).max() <= 0.99 and np.allclose(x, c + nz, atol=1e-5)
        if np.any(c) and np.any(nz):
            sis.append(si_sdr(c, x))
        rms.append(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)); nev.append(len(w))
    print(f"mode {mode}: input SI-SDR mean {np.mean(sis):6.2f} dB | mix rms {np.mean(rms):6.1f} dBFS | events/clip {np.mean(nev):.2f}")
if OFFICIAL_INPUT_SISDR:
    print(f"official reference: input SI-SDR mean {np.mean(OFFICIAL_INPUT_SISDR):.2f} dB, mix rms {np.mean(CAL.v['mix_rms_db']):.1f} dBFS")

# ============================== DATASET ==============================
def degrade_windows(wins, rng, dur, aug=None):
    """Simulates Track 1 errors so the model does not trust the event mask blindly.
    The same function is used for the held out set, so the local number stays honest."""
    a = aug or MASK_AUG
    if rng.random() < a["p_zero"]:
        return []
    out = []
    for s0, e0 in wins:
        if rng.random() < a["p_drop"]:
            continue
        s1, e1 = s0 + rng.normal(0, a["jitter"]), e0 + rng.normal(0, a["jitter"])
        if e1 - s1 > 0.05:
            out.append((max(0.0, s1), min(dur, e1)))
    if rng.random() < a["p_false"] and dur > 0.5:
        lo, hi = a["false_dur"]
        st = rng.uniform(0, max(0.01, dur - 0.3))
        out.append((st, min(dur, st + rng.uniform(lo, hi))))
    return sorted(out)

class MixStream(torch.utils.data.IterableDataset):
    def __init__(self, mode, L):
        self.mode, self.L = mode, L
    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng((wi.seed if wi else SEED + int(time.time())) % 2 ** 32)
        mixer = Mixer(self.mode, self.L)
        while True:
            x, c, _, w = mixer.make(rng)
            ev = frames_from_windows(degrade_windows(w, rng, self.L / SR), self.L)
            yield torch.from_numpy(x), torch.from_numpy(c), torch.from_numpy(ev)

# ============================== MODEL: TF-DPCRN v2 (wider, deeper, time attention) ==============================
class DPBlock(nn.Module):
    def __init__(self, C, H, attn=True, heads=4):
        super().__init__()
        self.intra = nn.LSTM(C, H, batch_first=True, bidirectional=True)
        self.intra_fc, self.intra_ln = nn.Linear(2 * H, C), nn.LayerNorm(C)
        self.inter = nn.LSTM(C, H, batch_first=True, bidirectional=True)
        self.inter_fc, self.inter_ln = nn.Linear(2 * H, C), nn.LayerNorm(C)
        self.attn = None
        if attn:
            heads = next((h for h in (heads, 4, 2, 1) if C % h == 0), 1)
            self.att_ln = nn.LayerNorm(C)
            self.attn = nn.MultiheadAttention(C, heads, batch_first=True)

    def forward(self, x):                                   # B,C,T,F
        B_, C, T, Fq = x.shape
        y = x.permute(0, 2, 3, 1).reshape(B_ * T, Fq, C)
        y = self.intra_ln(self.intra_fc(self.intra(y)[0])).reshape(B_, T, Fq, C)
        x2 = x.permute(0, 2, 3, 1) + y                      # B,T,F,C
        z = x2.permute(0, 2, 1, 3).reshape(B_ * Fq, T, C)
        z = self.inter_ln(self.inter_fc(self.inter(z)[0])).reshape(B_, Fq, T, C).permute(0, 2, 1, 3)
        out = x2 + z                                        # B,T,F,C
        if self.attn is not None:
            s = self.att_ln(out.mean(2))                    # B,T,C  frequency pooled summary
            a, _ = self.attn(s, s, s, need_weights=False)
            out = out + a[:, :, None, :]
        return out.permute(0, 3, 1, 2)

class TFDPCRN(nn.Module):
    def __init__(self, ch=(64, 96, 128, 192), n_dp=6, hid=160, attn=True, n_fft=N_FFT, hop=HOP):
        super().__init__()
        self.cfg = dict(ch=tuple(ch), n_dp=n_dp, hid=hid, attn=attn, n_fft=n_fft, hop=hop)
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("win", torch.hann_window(n_fft), persistent=False)
        self.enc = nn.ModuleList()
        cin = 4
        for c in ch:
            self.enc.append(nn.Sequential(nn.Conv2d(cin, c, (3, 3), (1, 2), (1, 1)), nn.BatchNorm2d(c), nn.PReLU(c)))
            cin = c
        self.dp = nn.Sequential(*[DPBlock(ch[-1], hid, attn) for _ in range(n_dp)])
        self.dec = nn.ModuleList()
        for i in reversed(range(len(ch))):
            cout = ch[i - 1] if i > 0 else 2
            layers = [nn.ConvTranspose2d(ch[i] * 2, cout, (3, 3), (1, 2), (1, 1))]
            if i > 0:
                layers += [nn.BatchNorm2d(cout), nn.PReLU(cout)]
            self.dec.append(nn.Sequential(*layers))

    def forward(self, wav, ev):
        n = wav.shape[-1]
        with torch.autocast(device_type=wav.device.type, enabled=False):
            X = torch.stft(wav.float(), self.n_fft, self.hop, window=self.win, return_complex=True)
            mag = X.abs().clamp_min(1e-8)
            magc = mag ** 0.3
            Xc = X / mag * magc
        T, Fq = X.shape[-1], X.shape[1]
        ev = ev[:, :T].float()
        if ev.shape[1] < T:
            ev = F.pad(ev, (0, T - ev.shape[1]))
        feat = torch.stack([Xc.real, Xc.imag, magc], 1).permute(0, 1, 3, 2)
        h = torch.cat([feat, ev[:, None, :, None].expand(-1, 1, T, Fq)], 1)
        skips = []
        for m in self.enc:
            h = m(h); skips.append(h)
        h = self.dp(h)
        for k, m in enumerate(self.dec):
            h = m(torch.cat([h, skips[-1 - k]], 1))
        with torch.autocast(device_type=wav.device.type, enabled=False):
            h = h.float().permute(0, 1, 3, 2)
            M = torch.complex(h[:, 0], h[:, 1])
            mm = M.abs()
            M = M / (mm + 1e-8) * torch.tanh(mm)
            return torch.istft(M * X, self.n_fft, self.hop, window=self.win, length=n)

def build_model(cfg=None):
    return TFDPCRN(**(cfg or ARCH))

print("model params: %.2f M" % (sum(p.numel() for p in build_model().parameters()) / 1e6))


# ============================== LOSSES (SI-SDR + multi resolution STFT + ASR facing mel) ==============================
def _mel_fb(sr=SR, n_fft=512, n_mels=128, fmin=20.0, fmax=7600.0):
    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for m in range(n_mels):
        l, c, r = bins[m], bins[m + 1], bins[m + 2]
        for k in range(l, min(c, fb.shape[1])):
            fb[m, k] = (k - l) / max(1, c - l)
        for k in range(c, min(r, fb.shape[1])):
            fb[m, k] = (r - k) / max(1, r - c)
    return torch.from_numpy(fb)

_MELFB = _mel_fb().to(device)
_MELWIN = torch.hann_window(512).to(device)

def MEL(x):
    S = torch.stft(x, 512, 160, window=_MELWIN, return_complex=True).abs() ** 2
    return torch.matmul(_MELFB, S)

def si_sdr_t(est, ref, eps=1e-8):
    est = est - est.mean(-1, keepdim=True); ref = ref - ref.mean(-1, keepdim=True)
    s = (est * ref).sum(-1, keepdim=True) / ((ref ** 2).sum(-1, keepdim=True) + eps) * ref
    return 10 * torch.log10((s ** 2).sum(-1) / (((est - s) ** 2).sum(-1) + eps) + eps)

def mrstft_loss(est, ref):
    tot = 0.0
    for nf, hp in [(512, 128), (1024, 256), (256, 64)]:
        w = torch.hann_window(nf, device=est.device)
        E = torch.stft(est, nf, hp, window=w, return_complex=True).abs()
        R = torch.stft(ref, nf, hp, window=w, return_complex=True).abs()
        tot = tot + torch.norm(R - E, p="fro") / (torch.norm(R, p="fro") + 1e-7) \
                  + F.l1_loss(torch.log(E + 1e-5), torch.log(R + 1e-5))
    return tot / 3

def mel_loss(est, ref):
    """Log mel distance. The mandated ASR listens to roughly these features, so this term
    protects phonemes and is the part that moves dWER rather than SI-SDR."""
    return F.l1_loss(torch.log(MEL(est) + 1e-5), torch.log(MEL(ref) + 1e-5))

def loss_fn(est, ref):
    valid = ((ref ** 2).mean(-1) > 1e-9).float()
    si = -(si_sdr_t(est, ref) * valid).sum() / valid.sum().clamp_min(1)
    return si + W_MRSTFT * mrstft_loss(est, ref) + W_MEL * mel_loss(est, ref), si


# ============================== INFERENCE HELPERS ==============================
@torch.no_grad()
def _run_chunk(model, seg, wins):
    x = torch.from_numpy(np.ascontiguousarray(seg, dtype=np.float32))[None].to(device)
    ev = torch.from_numpy(frames_from_windows(wins, len(seg)))[None].to(device)
    with torch.autocast(device_type=device, enabled=(device == "cuda")):
        y = model(x, ev)
    return np.nan_to_num(y[0].float().cpu().numpy())[: len(seg)]

@torch.no_grad()
def enhance_raw(model, wav, wins, chunk_sec=20.0, ov_sec=1.0):
    model.eval()
    wav = np.asarray(wav, np.float32)
    n = len(wav)
    if n < N_FFT:
        return _run_chunk(model, np.pad(wav, (0, N_FFT - n)), wins)[:n]
    win, ov = int(chunk_sec * SR), int(ov_sec * SR)
    if n <= win:
        return _run_chunk(model, wav, wins)
    out, acc = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for s0 in range(0, n, win - ov):
        seg = wav[s0:s0 + win]
        if len(seg) < N_FFT:
            seg = np.pad(seg, (0, N_FFT - len(seg)))
        sh = [(a - s0 / SR, b - s0 / SR) for a, b in wins if b > s0 / SR and a < (s0 + win) / SR]
        y = _run_chunk(model, seg, sh)[: min(win, n - s0)]
        fade = np.ones(len(y), np.float32)
        k = min(ov, len(y) // 2)
        if k and s0 > 0:
            fade[:k] = np.linspace(0, 1, k)
        if k and s0 + win < n:
            fade[-k:] = np.linspace(1, 0, k)
        out[s0:s0 + len(y)] += y * fade; acc[s0:s0 + len(y)] += fade
        if s0 + win >= n:
            break
    return out / np.maximum(acc, 1e-6)

def post(mix, est, wins, a_in, a_out, match_rms=True):
    """Event aware blend + level restore (free under SI-SDR, important for ASR) + peak guard."""
    m = sample_mask_from_windows(wins, len(mix))
    alpha = a_out + (a_in - a_out) * m
    y = np.nan_to_num(alpha * est + (1 - alpha) * mix)
    if match_rms:
        ri, ro = np.sqrt(np.mean(mix ** 2)), np.sqrt(np.mean(y ** 2))
        if ri > 1e-8 and ro > 1e-8:
            y = y * (ri / ro)
    pk = np.abs(y).max()
    if pk > 0.99:
        y = y / pk * 0.99
    return y.astype(np.float32)

# held out cache (never used for training)
HOLD = []
_vr = np.random.default_rng(99)
for it in HOLD_SYN:
    x, c, n = load_pair(it)
    if c is None:
        continue
    if it["id"] in T1:
        w, src = T1[it["id"]], "t1"
    else:
        w, src = degrade_windows(windows_from_noise(n)[0], _vr, len(x) / SR) if n is not None else [], "sim_t1"
    HOLD.append(dict(id=it["id"], x=x, c=c, wins=w, gt=it["gt"], wsrc=src))
if not HOLD:
    print("No held out official pairs; using a fixed synthetic mode E set for quick validation.")
    _mx = Mixer("E", int(6 * SR)); _r = np.random.default_rng(123)
    for k in range(150):
        x, c, _, w = _mx.make(_r)
        HOLD.append(dict(id=f"synval{k}", x=x, c=c, wins=degrade_windows(w, _r, 6.0), gt=None, wsrc="sim"))
print("held out quick val clips:", len(HOLD), Counter(h["wsrc"] for h in HOLD))

def quick_val(model):
    return float(np.mean([si_sdr(h["c"], enhance_raw(model, h["x"], h["wins"])) for h in HOLD]))

print("held out input SI-SDR (no processing): %.2f dB" % np.mean([si_sdr(h["c"], h["x"]) for h in HOLD]))

# ============================== TRAINING (bf16, EMA, time budget) ==============================
import copy

AMP_DTYPE = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
print("autocast dtype:", AMP_DTYPE)

def train_mode(mode, minutes, init_state=None, lr=1e-3, tag=None, seed=None, cfg=None):
    tag = tag or f"mode{mode}"
    if seed is not None:
        torch.manual_seed(seed)
    model = build_model(cfg).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.98))
    use_scaler = (AMP_DTYPE == torch.float16 and device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    nw = B["workers"]
    dl = torch.utils.data.DataLoader(MixStream(mode, L_TRAIN), batch_size=B["bs"], num_workers=nw,
                                     pin_memory=(device == "cuda"), prefetch_factor=6 if nw else None,
                                     persistent_workers=bool(nw))
    t0 = time.time(); step = 0; hist = []; n_oom = 0; run_si = []
    best = (quick_val(ema) if init_state is not None else -1e9, copy.deepcopy(ema.state_dict()))
    next_eval = t0 + B["eval_every_min"] * 60
    for x, c, ev in dl:
        frac = (time.time() - t0) / (minutes * 60)
        if frac >= 1:
            break
        cur = lr * min(1.0, (step + 1) / 1000) * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = cur
        model.train()
        x, c, ev = x.to(device, non_blocking=True), c.to(device, non_blocking=True), ev.to(device, non_blocking=True)
        try:
            with torch.autocast(device_type=device, dtype=AMP_DTYPE, enabled=(device == "cuda")):
                est = model(x, ev)
            loss, si = loss_fn(est.float(), c)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
        except torch.cuda.OutOfMemoryError:
            n_oom += 1
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            if n_oom == 1:
                print("  CUDA OOM, skipping batch. Lower B['bs'] if this repeats.")
            continue
        d = min(0.9995, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        run_si.append(-si.item()); step += 1
        if step % 200 == 0:
            print(f"  [{tag}] step {step} | {(time.time()-t0)/60:6.1f} min | lr {cur:.2e} | "
                  f"train SI-SDR {np.mean(run_si[-200:]):.2f}", flush=True)
        if time.time() >= next_eval:
            v = quick_val(ema); hist.append((step, v))
            print(f"  [{tag}] held out SI-SDR {v:.3f} dB", flush=True)
            if v > best[0]:
                best = (v, copy.deepcopy(ema.state_dict()))
            next_eval += B["eval_every_min"] * 60
    del dl
    v = quick_val(ema); hist.append((step, v))
    if v > best[0]:
        best = (v, copy.deepcopy(ema.state_dict()))
    torch.save(dict(state=best[1], cfg=(cfg or ARCH), score=best[0]), CKPT_DIR / f"{tag}.pt")
    print(f"[{tag}] done: {step} steps, best held out SI-SDR {best[0]:.3f} dB"
          + (f" | OOM skips {n_oom}" if n_oom else ""))
    return best[1], best[0], hist

def load_model(state, cfg=None):
    m = build_model(cfg).to(device)
    m.load_state_dict(state)
    return m.eval()


# ============================== GPU FIT CHECK (auto batch size) ==============================
def fit_batch_size(bs):
    if device != "cuda":
        return min(bs, 4)
    m = build_model().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(AMP_DTYPE == torch.float16))
    while bs >= 2:
        try:
            torch.cuda.reset_peak_memory_stats()
            x = (torch.randn(bs, L_TRAIN, device=device) * 0.05)
            ev = torch.zeros(bs, L_TRAIN // HOP + 1, device=device)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                est = m(x, ev)
            loss, _ = loss_fn(est.float(), x)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            print(f"batch {bs} fits, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
            break
        except torch.cuda.OutOfMemoryError:
            print("batch", bs, "does not fit, halving")
            bs //= 2
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    del m, opt
    torch.cuda.empty_cache(); gc.collect()
    return max(2, bs)

B["bs"] = fit_batch_size(B["bs"])
print("training batch size:", B["bs"])

# ============================== MANDATED ASR ==============================
class Transcriber:
    """Same loading path as the official baseline. The audit re-runs this model on the submitted WAVs."""
    def __init__(self):
        from huggingface_hub import snapshot_download
        from transformers import AutoModel
        path = snapshot_download(ASR_REPO, token=HF_TOKEN)
        self.asr = AutoModel.from_pretrained(path, trust_remote_code=True).to(device).eval()
        print("mandated ASR loaded:", path)

    @torch.no_grad()
    def transcribe_paths(self, paths, bs=32):
        out = []
        for i in range(0, len(paths), bs):
            chunk = [str(p) for p in paths[i:i + bs]]
            try:
                hyps = self.asr.transcribe(chunk, return_hypotheses=True)
                out += [(h.text if hasattr(h, "text") else str(h)).strip() for h in hyps]
            except Exception:
                for p in chunk:
                    try:
                        h = self.asr.transcribe([p], return_hypotheses=True)[0]
                        out.append((h.text if hasattr(h, "text") else str(h)).strip())
                    except Exception:
                        out.append("")
        return out

    def transcribe_many(self, wavs):
        d = Path(tempfile.mkdtemp(dir=SCRATCH))
        ps = []
        for k, w in enumerate(wavs):
            p = d / f"{k:06d}.wav"
            sf.write(p, np.asarray(w, np.float32), SR, subtype=WAV_SUBTYPE)
            ps.append(p)
        try:
            return self.transcribe_paths(ps)
        finally:
            shutil.rmtree(d, ignore_errors=True)

ASR = None
def get_asr():
    global ASR
    if ASR is None:
        ASR = Transcriber()
    return ASR

# ============================== FULL VALIDATION (SI-SDR held out + pooled dWER) ==============================
EVAL_CACHE = {}

def prepare_eval():
    if EVAL_CACHE:
        return EVAL_CACHE
    asr = get_asr()
    items = [dict(h) for h in HOLD if h["id"].startswith("synval") is False]
    for it in NATURAL:
        if it["noisy"] is None:
            continue
        items.append(dict(id=it["id"], x=read_audio(it["noisy"]), c=None, wins=T1.get(it["id"], []),
                          gt=it["gt"], wsrc="t1" if it["id"] in T1 else "none"))
    need_gt = [h for h in items if not h["gt"] and h["c"] is not None]
    if need_gt:
        for h, t in zip(need_gt, asr.transcribe_many([h["c"] for h in need_gt])):
            h["gt"] = t
    noisy = asr.transcribe_many([h["x"] for h in items])
    EVAL_CACHE.update(items=items, gt={h["id"]: h["gt"] for h in items if h["gt"]},
                      noisy={h["id"]: t for h, t in zip(items, noisy)})
    print(f"eval set: {len(items)} clips, {len(EVAL_CACHE['gt'])} with transcripts")
    return EVAL_CACHE

def full_eval(state, grid_in=(1.0, 0.9, 0.8), grid_out=(1.0, 0.9, 0.75, 0.6), n_asr=3, extra=((0.85, 0.85),)):
    E_ = prepare_eval()
    model = load_model(state)
    ests = {h["id"]: enhance_raw(model, h["x"], h["wins"]) for h in E_["items"]}
    syn = [h for h in E_["items"] if h["c"] is not None]
    ids = [h["id"] for h in E_["items"] if h["id"] in E_["gt"]]
    rows = []
    for ai in grid_in:
        for ao in grid_out:
            s = np.mean([si_sdr(h["c"], post(h["x"], ests[h["id"]], h["wins"], ai, ao)) for h in syn]) if syn else 0.0
            rows.append([ai, ao, float(s), float("nan"), float("nan")])
    rows.sort(key=lambda r: -r[2])
    todo = [tuple(r[:2]) for r in rows[:n_asr]] + [e for e in extra]
    seen = set()
    for ai, ao in todo:
        if (ai, ao) in seen:
            continue
        seen.add((ai, ao))
        outs = [post(h["x"], ests[h["id"]], h["wins"], ai, ao) for h in E_["items"]]
        hyp = dict(zip([h["id"] for h in E_["items"]], get_asr().transcribe_many(outs)))
        dw = delta_wer(E_["gt"], E_["noisy"], hyp, ids)
        s = np.mean([si_sdr(h["c"], o) for h, o in zip(E_["items"], outs) if h["c"] is not None]) if syn else 0.0
        r = next((r for r in rows if r[0] == ai and r[1] == ao), None)
        if r is None:
            r = [ai, ao, float(s), 0, 0]; rows.append(r)
        r[3] = float(dw); r[4] = float(s + 100 * (dw if np.isfinite(dw) else 0))
    scored = [r for r in rows if np.isfinite(r[4])]
    best = max(scored, key=lambda r: r[4])
    print(" a_in  a_out   SI-SDR    dWER%   Combined")
    for r in sorted(rows, key=lambda r: -(r[4] if np.isfinite(r[4]) else -1e9))[:10]:
        print(f" {r[0]:.2f}  {r[1]:.2f}  {r[2]:7.3f}  {100*r[3]:7.2f}  {r[4]:8.3f}")
    del model
    return dict(a_in=best[0], a_out=best[1], si=best[2], dwer=best[3], combined=best[4])

# ============================== HARVEST REAL NOISE FROM NATURAL TRAIN FOLD CLIPS ==============================
def harvest_noise(state, max_clips=400):
    """A trained model gives an estimate of the noise itself: noisy minus enhanced.
    Those residuals are real, in domain noise with real floors and real bursts, and they
    come only from natural clips in the TRAIN fold, never from held out or test audio."""
    src = [it for it in NATURAL if not it["holdout"] and it["noisy"] is not None][:max_clips]
    if not src:
        print("no natural train fold clips to harvest from"); return 0, 0
    model = load_model(state)
    Wf, We = PoolWriter("floor"), PoolWriter("event")
    n_f = n_e = 0
    for it in src:
        x = read_audio(it["noisy"])
        r = x - enhance_raw(model, x, T1.get(it["id"], []))
        if not np.isfinite(r).all() or np.sqrt(np.mean(r ** 2)) < 1e-5:
            continue
        wins, _ = windows_from_noise(r)
        ev = np.zeros(len(r), bool)
        for a, b in wins:
            ev[int(a * SR):int(b * SR)] = True
            seg = r[int(a * SR):int(b * SR)]
            if len(seg) > 0.15 * SR and speech_ratio(seg) <= 0.5:
                We.add(seg, src="harvest", cat="harvest"); n_e += 1
        quiet = r[~ev]
        if len(quiet) > 0.5 * SR:
            Wf.add(quiet[: 8 * SR], src="harvest"); n_f += 1
    Wf.close(); We.close()
    del model; torch.cuda.empty_cache()
    print(f"harvested {n_f} floors and {n_e} bursts from {len(src)} natural train fold clips")
    return n_f, n_e


# ============================== ENSEMBLE EVALUATION ==============================
def ens_enhance(models, wav, wins):
    ys = [enhance_raw(m, wav, wins) for m in models]
    n = min(len(y) for y in ys)
    # SI-SDR is blind to polarity, so a model can learn an INVERTED output and still score
    # 22 dB alone; averaging it with a non-inverted one cancels (measured: 22.8 + 22.7 -> 4.1
    # dB). Align every estimate to the mixture, which always correlates positively with speech.
    x = np.asarray(wav, np.float32)[:n]
    ys = [y[:n] if np.dot(y[:n], x) >= 0 else -y[:n] for y in ys]
    return np.mean(ys, axis=0).astype(np.float32)

def full_eval_ens(states, cfgs=None, grid_in=(1.0, 0.95, 0.9, 0.8), grid_out=(1.0, 0.9, 0.8, 0.7, 0.6),
                  n_asr=4, extra=((0.9, 0.9),)):
    E_ = prepare_eval()
    models = [load_model(s, (cfgs[i] if cfgs else None)) for i, s in enumerate(states)]
    ests = {h["id"]: ens_enhance(models, h["x"], h["wins"]) for h in E_["items"]}
    syn = [h for h in E_["items"] if h["c"] is not None]
    ids = [h["id"] for h in E_["items"] if h["id"] in E_["gt"]]
    rows = []
    for ai in grid_in:
        for ao in grid_out:
            s = np.mean([si_sdr(h["c"], post(h["x"], ests[h["id"]], h["wins"], ai, ao)) for h in syn]) if syn else 0.0
            rows.append([ai, ao, float(s), float("nan"), float("nan")])
    rows.sort(key=lambda r: -r[2])
    todo = [tuple(r[:2]) for r in rows[:n_asr]] + list(extra)
    seen = set()
    for ai, ao in todo:
        if (ai, ao) in seen:
            continue
        seen.add((ai, ao))
        outs = [post(h["x"], ests[h["id"]], h["wins"], ai, ao) for h in E_["items"]]
        hyp = dict(zip([h["id"] for h in E_["items"]], get_asr().transcribe_many(outs)))
        dw = delta_wer(E_["gt"], E_["noisy"], hyp, ids)
        s = np.mean([si_sdr(h["c"], o) for h, o in zip(E_["items"], outs) if h["c"] is not None]) if syn else 0.0
        r = next((r for r in rows if r[0] == ai and r[1] == ao), None)
        if r is None:
            r = [ai, ao, float(s), 0.0, 0.0]; rows.append(r)
        r[3] = float(dw); r[4] = float(s + 100 * (dw if np.isfinite(dw) else 0))
    print(" a_in  a_out   SI-SDR    dWER%   Combined")
    for r in sorted(rows, key=lambda r: -(r[4] if np.isfinite(r[4]) else -1e9))[:10]:
        print(f" {r[0]:.2f}  {r[1]:.2f}  {r[2]:7.3f}  {100*r[3]:7.2f}  {r[4]:8.3f}")
    best = max([r for r in rows if np.isfinite(r[4])], key=lambda r: r[4])
    for m in models:
        del m
    torch.cuda.empty_cache()
    return dict(a_in=best[0], a_out=best[1], si=best[2], dwer=best[3], combined=best[4])


# ============================== TRAIN THE ENSEMBLE AND EXPORT ==============================
STATES, SCORES = [], []
INIT0 = None
if RESUME_FROM and Path(RESUME_FROM).exists():
    _d = torch.load(RESUME_FROM, map_location="cpu", weights_only=False)
    INIT0 = _d.get("state", _d)
    print("resuming model 0 from", RESUME_FROM, "| arch in file:", _d.get("cfg"))
KEEP = os.environ.get("T2_KEEP")                 # exported model reused as model 0 (not retrained)
N_MODELS = int(os.environ.get("T2_N_MODELS", B["n_models"]))
# T2_EVAL_ONLY="a.pt,b.pt" + T2_EVAL_SCORES="22.8,22.7": no training, just evaluate
# single-vs-ensemble on the held-out set and export the winner
EVAL_ONLY = [p for p in os.environ.get("T2_EVAL_ONLY", "").split(",") if p]
if EVAL_ONLY:
    _sc = [float(v) for v in os.environ.get("T2_EVAL_SCORES", "").split(",") if v]
    for i, p in enumerate(EVAL_ONLY):
        _d = torch.load(p, map_location="cpu", weights_only=False)
        STATES.append(_d.get("state", _d)); SCORES.append(_sc[i] if i < len(_sc) else 0.0)
        print(f"eval-only model {i}: {p} (held out {SCORES[-1]:.3f} dB)")
    # T2_TRAIN_MORE=n: after preloading, train n NEW models; k continues from the preloaded
    # count so each new model gets a fresh seed (SEED + 17k) and its own checkpoint tag
    N_MODELS = int(os.environ.get("T2_TRAIN_MORE", "0"))
K0 = len(STATES)
for k in range(K0, K0 + N_MODELS):
    if k == 0 and KEEP:
        _d = torch.load(KEEP, map_location="cpu", weights_only=False)
        STATES.append(_d.get("state", _d)); SCORES.append(float(os.environ.get("T2_KEEP_SCORE", "0")))
        print(f"model 0: reused {KEEP} | held out SI-SDR {SCORES[-1]:.3f} dB (from its export)")
        continue
    mode = os.environ.get("T2_MODE") or ("E" if k % 2 == 0 else "D")
    st, sc, _ = train_mode(mode, B["model_min"], init_state=(INIT0 if k == 0 else None),
                           lr=(4e-4 if (k == 0 and INIT0 is not None) else 1e-3),
                           tag=f"m{k}_{mode}", seed=SEED + 17 * k)
    st, sc, _ = train_mode("P", B["polish_min"], init_state=st, lr=2e-4, tag=f"m{k}_polish", seed=SEED + 17 * k)
    STATES.append(st); SCORES.append(sc)
    print(f"model {k}: held out SI-SDR {sc:.3f} dB")
    if k == 0 and HARVEST_NOISE:
        harvest_noise(st)          # models 1.. also see real harvested noise
    torch.cuda.empty_cache(); gc.collect()

single = full_eval_ens([STATES[int(np.argmax(SCORES))]])
print("best single model:", single)
ENS = full_eval_ens(STATES) if len(STATES) > 1 else single
print("ensemble:", ENS)
USE_ENS = len(STATES) > 1 and ENS["combined"] > single["combined"]
FINAL_STATES = STATES if USE_ENS else [STATES[int(np.argmax(SCORES))]]
FINAL_EVAL = ENS if USE_ENS else single
print("using", len(FINAL_STATES), "model(s)")

shutil.rmtree(EXPORT_DIR, ignore_errors=True); EXPORT_DIR.mkdir(parents=True)
for i, s in enumerate(FINAL_STATES):
    torch.save(dict(state=s, cfg=ARCH), EXPORT_DIR / f"t2_model_{i}.pt")
json.dump(dict(arch={k: list(v) if isinstance(v, tuple) else v for k, v in ARCH.items()},
               a_in=FINAL_EVAL["a_in"], a_out=FINAL_EVAL["a_out"], n_fft=N_FFT, hop=HOP, sr=SR,
               heldout_si=FINAL_EVAL.get("si"), heldout_dwer=FINAL_EVAL.get("dwer"),
               heldout_combined=FINAL_EVAL.get("combined"), scores=SCORES,
               leakage_skips=dict(LEAK_SKIPPED)),
          open(EXPORT_DIR / "t2_export.json", "w"), indent=2, default=str)
print("export folder ready:", EXPORT_DIR, [p.name for p in EXPORT_DIR.iterdir()])
print("upload this folder to Kaggle as a private dataset, then run the inference notebook.")
