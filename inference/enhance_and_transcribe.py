#!/usr/bin/env python3
"""IndoML 2026 Track 2: local inference on your own machine, no Kaggle.

  T2_TEST=/data/indoml/test_audio \
  T2_T1=/data/indoml/track1/predictions.jsonl \
  T2_CKPT=/data/t2work/export \
  HF_TOKEN=hf_xxx python track2_local_infer.py

Writes submission.zip next to T2_OUT (default ./t2_submission).
"""
import os, sys, re, io, json, math, time, shutil, zipfile, tempfile, subprocess, warnings
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
warnings.filterwarnings("ignore")

SR = 16000
SEED = 2026
ASR_REPO = "ARTPARK-IISc/SraVaani-1.0"
TRANSCRIPT_NAME, ID_KEY, TEXT_KEY, WAV_SUBTYPE = "transcripts.jsonl", "clip_id", "text", "PCM_16"
AUD = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

TEST_ROOT = Path(os.environ["T2_TEST"])
T1_PATH = Path(os.environ["T2_T1"])
CKPT_ROOT = Path(os.environ.get("T2_CKPT", "./t2work/export"))
OUT_DIR = Path(os.environ.get("T2_OUT", "./t2_submission"))
BLEND = os.environ.get("T2_BLEND")            # "a_in,a_out" to override the exported values
PRED_PAD = float(os.environ.get("T2_PAD", "0.10"))
HARD = float(os.environ.get("T2_HARD", "1"))    # >1: extra in-window suppression m^(HARD-1) (src/t2_harness.py)
MUTE = os.environ.get("T2_MUTE")                # dB: zero in-window 10 ms frames this far below the mixture
BANKPOST = os.environ.get("T2_BANKPOST", "0") == "1"   # enhancer pass over bank-subtracted clips (+5 dB sim)
BANK_AOUT = float(os.environ.get("T2_BANK_AOUT", "0.0"))
WHOLE = os.environ.get("T2_WHOLE", "0") == "1"  # one whole-clip window per clip (no nat dWER cost, +0.11 dB off0)
HYBRID = os.environ.get("T2_HYBRID", "0") == "1" # hybrid neural residual cleaning on synthetic bank clips
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH = OUT_DIR / "scratch"; SUB_DIR = SCRATCH / "wavs"
SUB_DIR.mkdir(parents=True, exist_ok=True)

import soundfile as sf
import scipy.signal as sps
import torch, torch.nn as nn, torch.nn.functional as F
HF_TOKEN = os.environ.get("HF_TOKEN")
assert HF_TOKEN, "set HF_TOKEN, the mandated ASR is a gated repo"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| test:", TEST_ROOT, "| ckpt:", CKPT_ROOT)


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
    def __init__(self, ch=(64, 96, 128, 192), n_dp=6, hid=160, attn=True, n_fft=512, hop=128):
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

def harder(mix, est, wins, p):
    """Inside windows multiply the enhanced STFT by m^(p-1), m = |EST|/|MIX| clipped to [0, 1]."""
    if p == 1:
        return est
    from scipy.signal import stft, istft
    kw = dict(fs=SR, window="hann", nperseg=512, noverlap=384)
    _, t, X = stft(mix, **kw); _, _, Y = stft(est, **kw)
    m = np.clip(np.abs(Y) / (np.abs(X) + 1e-9), 0, 1)
    w = np.zeros(len(t))
    for a, b in wins:
        w[(t >= a) & (t <= b)] = 1.0
    w = np.convolve(w, np.ones(5) / 5, mode="same")
    _, y = istft(Y * (1 - w[None, :] * (1 - m ** (p - 1))), **kw)
    y = y[:len(est)]
    return np.pad(y, (0, len(est) - len(y))).astype(np.float32)

def mute_frames(y, mix, est, wins, db, fr=160):
    """Zero in-window 10 ms frames where the enhanced energy is > db below the mixture (noise over silence)."""
    n = len(mix) // fr
    if db is None or n == 0:
        return y
    em = (mix[:n * fr].reshape(n, fr) ** 2).mean(1) + 1e-12
    ee = (est[:n * fr].reshape(n, fr) ** 2).mean(1) + 1e-12
    inside = np.zeros(n, bool)
    for a, b in wins:
        inside[max(0, int(a * SR) // fr):min(n, int(np.ceil(b * SR / fr)))] = True
    mm = inside & (10 * np.log10(ee / em) < db)
    if mm.any():
        y = y.copy(); y[:n * fr].reshape(n, fr)[mm] = 0.0
    return y

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

# ============================== LOAD MODELS AND TRACK 1 PREDICTIONS ==============================
N_FFT, HOP = 512, 128

def find_ckpts(root):
    cks = sorted(Path(root).rglob("t2_model_*.pt")) or sorted(Path(root).rglob("*.pt"))
    meta = next(iter(Path(root).rglob("t2_export.json")), None)
    return cks, meta

def cfg_from_state(sd):
    ch, i = [], 0
    while f"enc.{i}.0.weight" in sd:
        ch.append(sd[f"enc.{i}.0.weight"].shape[0]); i += 1
    n_dp = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("dp."))
    return dict(ch=tuple(ch), n_dp=n_dp, hid=sd["dp.0.intra.weight_ih_l0"].shape[0] // 4,
                attn=any(".attn." in k for k in sd))

CKPTS, META_PATH = find_ckpts(CKPT_ROOT)
assert CKPTS, f"no checkpoint found under {CKPT_ROOT}"
META = json.loads(Path(META_PATH).read_text()) if META_PATH else {}
A_IN, A_OUT = (tuple(float(v) for v in BLEND.split(",")) if BLEND
               else (float(META.get("a_in", 0.95)), float(META.get("a_out", 0.85))))
MODELS = []
for p in CKPTS:
    d = torch.load(p, map_location="cpu", weights_only=False)
    sd = d["state"] if isinstance(d, dict) and "state" in d else d
    cfg = dict(d.get("cfg", {})) if isinstance(d, dict) and "cfg" in d else cfg_from_state(sd)
    cfg["ch"] = tuple(cfg["ch"])
    N_FFT, HOP = int(cfg.get("n_fft", 512)), int(cfg.get("hop", 128))
    m = TFDPCRN(**cfg).to(device); m.load_state_dict(sd); MODELS.append(m.eval())
    print("loaded", p.name, cfg)
print(f"{len(MODELS)} model(s) | blend a_in {A_IN} a_out {A_OUT} | hop {HOP} | pad {PRED_PAD} hard {HARD} mute {MUTE} | whole {WHOLE}")

T1 = load_t1_predictions(T1_PATH)
if PRED_PAD > 0:
    T1 = {k: [(max(0.0, a - PRED_PAD), b + PRED_PAD) for a, b in v] for k, v in T1.items()}
print("Track 1 clips:", len(T1))

def ens_enhance(wav, wins):
    ys = [enhance_raw(m, wav, wins) for m in MODELS]
    n = min(len(y) for y in ys)
    # align polarity to the mixture before averaging (a model can learn an inverted output;
    # SI-SDR cannot see it, but averaging inverted + normal estimates cancels the speech)
    x = np.asarray(wav, np.float32)[:n]
    ys = [y[:n] if np.dot(y[:n], x) >= 0 else -y[:n] for y in ys]
    return np.mean(ys, axis=0).astype(np.float32)


# ============================== RUN ==============================
test_files = sorted(p for p in TEST_ROOT.rglob("*") if p.suffix.lower() in AUD)
assert test_files, f"no audio under {TEST_ROOT}"
dup = [s for s, c in Counter(p.stem for p in test_files).items() if c > 1]
assert not dup, f"duplicate stems {dup[:5]}"
print("test clips:", len(test_files), "| with Track 1 events:",
      sum(1 for p in test_files if T1.get(p.stem)))

shutil.rmtree(SUB_DIR, ignore_errors=True); SUB_DIR.mkdir(parents=True)
# T2_BANKSUB: directory of synthetic clips with the matched bank noise subtracted exactly
# (src/t2_bank_subtract.py). Those are used as-is -- no network, no blend, no RMS guard
# (removing loud noise legitimately halves the energy).
BANKSUB = os.environ.get("T2_BANKSUB")
SUB_SPANS = {}
if BANKSUB and (Path(BANKSUB) / "subtracted_spans.json").exists():
    try:
        SUB_SPANS = json.loads((Path(BANKSUB) / "subtracted_spans.json").read_text())
        print(f"loaded {len(SUB_SPANS)} clip span annotations from {BANKSUB}/subtracted_spans.json")
    except Exception as exc:
        print(f"warning: failed to load subtracted_spans.json: {exc}")

bad, n_bank, n_hybrid, t0 = 0, 0, 0, time.time()
for k, p in enumerate(test_files):
    wav = read_audio(p)
    wins = T1.get(p.stem, [])
    if not wins:
        auto_w, _ = windows_from_noise(wav)
        wins = auto_w if auto_w else [(0.0, float(len(wav)) / SR)]
    t1w = list(wins)
    if WHOLE:
        wins = [(0.0, float(len(wav)) / SR)]
    bs = Path(BANKSUB) / f"{p.stem}.wav" if BANKSUB else None
    if bs is not None and bs.exists():
        y, bsr = sf.read(bs, dtype="float32")
        y = y if y.ndim == 1 else y.mean(1)
        n_bank += 1
        if not (bsr == SR and np.isfinite(y).all() and len(y) == len(wav)):
            bad += 1
            y = np.clip(wav, -0.99, 0.99).astype(np.float32)
        elif BANKPOST and len(MODELS) > 0:
            # Enhancer pass over the subtracted signal: altered Phase 2 copies leave a residual the template
            # cannot remove (src/t2_banksub_post_eval.py). Windows = subtracted spans + Track 1 windows.
            bw = sorted([(max(0.0, a - 0.10), b + 0.10) for a, b in SUB_SPANS.get(p.stem, [])] + list(t1w))
            if bw:
                y = post(y, ens_enhance(y, bw), bw, 1.0, BANK_AOUT)
                n_hybrid += 1
        elif HYBRID:
            # Hybrid residual cleaning: if any Track 1 predicted noise event was not subtracted by bank,
            # enhance strictly that span while keeping bank-subtracted regions completely pristine.
            done_spans = SUB_SPANS.get(p.stem, [])
            uncleaned = []
            for w0, w1 in wins:
                cov = any(min(w1, s1) - max(w0, s0) > 0.05 for s0, s1 in done_spans)
                if not cov:
                    uncleaned.append((w0, w1))
            if uncleaned and len(MODELS) > 0:
                y_enh = ens_enhance(y, uncleaned)
                m_uncleaned = sample_mask_from_windows(uncleaned, len(y), ramp=0.02)
                y = (1.0 - m_uncleaned) * y + m_uncleaned * y_enh
                n_hybrid += 1
        y = np.clip(y, -1.0, 1.0).astype(np.float32)
        sf.write(SUB_DIR / f"{p.stem}.wav", y, SR, subtype=WAV_SUBTYPE)
        continue
    est = harder(wav, ens_enhance(wav, wins), wins, HARD)
    y = post(wav, est, wins, A_IN, A_OUT)
    ri, ro = np.sqrt(np.mean(wav ** 2)), np.sqrt(np.mean(y ** 2))
    if not (np.isfinite(y).all() and len(y) == len(wav) and (ri < 1e-6 or 0.5 <= ro / (ri + 1e-12) <= 2.0)):
        bad += 1
        y = np.clip(wav, -0.99, 0.99).astype(np.float32)
    else:                                         # mute after the level guard, so it cannot trip it
        y = mute_frames(y, wav, est, wins, float(MUTE) if MUTE else None)
    sf.write(SUB_DIR / f"{p.stem}.wav", y, SR, subtype=WAV_SUBTYPE)
    if k % 200 == 0:
        print(f"  {k}/{len(test_files)} | {(time.time()-t0)/60:.1f} min", flush=True)
print(f"fallback clips: {bad} | bank-subtracted clips used: {n_bank} (hybrid enhanced: {n_hybrid})")

out_paths = [SUB_DIR / f"{p.stem}.wav" for p in test_files]
texts = get_asr().transcribe_paths(out_paths)
tj = OUT_DIR / TRANSCRIPT_NAME
with open(tj, "w", encoding="utf-8") as f:
    for p, t in zip(test_files, texts):
        f.write(json.dumps({ID_KEY: p.stem, TEXT_KEY: t}, ensure_ascii=False) + "\n")
print("empty transcripts:", sum(1 for t in texts if not t.strip()), "of", len(texts))

zp = OUT_DIR / "submission.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
    z.write(tj, TRANSCRIPT_NAME)
    for q in out_paths:
        z.write(q, q.name)
names = set(zipfile.ZipFile(zp).namelist())
assert TRANSCRIPT_NAME in names and all(q.name in names for q in out_paths)
print(f"ready: {zp} ({zp.stat().st_size/1e9:.2f} GB, {len(names)-1} wavs)")
shutil.rmtree(SCRATCH, ignore_errors=True)
