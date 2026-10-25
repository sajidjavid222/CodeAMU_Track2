"""Fine-tune the three t2_ens3 enhancers on Phase 2-like mixtures (model-only).

Why: the same t2_ens3 model scored SI-SDR 17.46 on Phase 1 and 12.13 on Phase 2. Measured on 100
confident-synthetic clips per phase, the Phase 2 generator has fewer noise-over-silence events
(25.5% -> 10.1%), louder noise under speech (local SNR median -7.6 -> -9.7 dB), more events
(3.7 -> 6.0 per clip, ~0.47 events/s) and slightly altered noise copies. The enhancers were trained
on Phase 1-style mixtures, so this fine-tunes them on mixtures drawn from the Phase 2 profile.

Data (training fold only; the 138 held-out synthetic clips of track2_train.py are never used):
  clean beds : official train-fold clean refs (pool offc) 50%, Vaani speech 25%, gold clean 25%
  noise      : validation noise recordings (models/noise_bank.npz) 70%, noise-event pool 30%,
               each with mild random alteration (FIR tilt, +-0.4% resample, -35 dB hiss)
Same loss (-SI-SDR + 0.1 MR-STFT + 0.3 log-mel), AdamW, EMA and window augmentation as track2_train.py.

Eval (old vs new ensemble, same code path as the submissions):
  off0 : the 138 held-out official synthetic pairs, v67 validation windows
  p2h  : 120 Phase 2-like mixtures (~12.7 s) built from held-out clean refs, with transcripts
  nat  : 300 natural validation clips (dWER guard)

    python src/t2_finetune_p2.py            # GPU; T2FT_MIN minutes per model (default 40)
"""
import copy
import hashlib
import json
import math
import os
import re
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)                          # repository root
sys.path.insert(0, HERE)
import enhance as E                                   # noqa: E402  TFDPCRN + inference, as submitted
from asr_wer import wer_counts                        # noqa: E402

SR = 16000
V = os.environ.get("T2_VAL_DIR", f"{PROJ}/data/validation")               # official validation folder
POOLS = os.environ.get("T2_POOLS", f"{PROJ}/work/t2work/pools")           # written by train/track2_train.py
OLD = os.environ.get("T2FT_INIT", f"{PROJ}/work/t2work/export")   # models to start from / compare against
OUT = os.environ.get("T2FT_OUT", f"{PROJ}/work/t2_p2ft")
EXPORT = f"{OUT}/export"
MINUTES = float(os.environ.get("T2FT_MIN", "40"))
SMOKE = int(os.environ.get("T2FT_SMOKE", "0"))
L = int(8.0 * SR)
BS = int(os.environ.get("T2FT_BS", "16"))
LR = float(os.environ.get("T2FT_LR", "2e-4"))
W_MEL = float(os.environ.get("T2FT_WMEL", "0.3"))   # ASR-facing log-mel term; 0.3 = the original recipe
SCRATCH = int(os.environ.get("T2FT_SCRATCH", "0"))  # 1: random init, trained only on the Phase 2 profile
BIG = int(os.environ.get("T2FT_BIG", "0"))          # 1: wider/deeper TF-DPCRN (the class default)
ONLY = os.environ.get("T2FT_ONLY")                  # train just this model index (parallel jobs)
# Leaderboard fit over the 6 Phase 2-adapted submissions: LB ~ -24.68 + 1.53*off0 + 0.27*p2h (max resid 0.12).
# The real organiser clips (off0) dominate, so: mix REAL training-fold organiser pairs into training
# (T2FT_REAL = probability per example) and select checkpoints by the fitted score, not by p2h alone.
P_REAL = float(os.environ.get("T2FT_REAL", "0"))
W_OFF0, W_P2H = 1.53, 0.27
# Checkpoint selection: "fit" (default, used for the final p2mix stage) = W_OFF0*SI-SDR(off0[:70]) + W_P2H*SI-SDR(p2h[:40]);
# "p2h" = SI-SDR(p2h[:40]) only, as used by the earlier stages p2ft / p2ft_cont / p2ft_long.
SELECT = os.environ.get("T2FT_SELECT", "fit")
# ASR-guided term: our synthetic dWER is 6.7 vs 11.5-17.4 for the top teams, and on off0 we reach only half
# of the clean-reference dWER ceiling, so the enhancer's distortions (not residual noise) cost the words.
# T2FT_WASR > 0 adds W * 10*log10(|E(est)-E(ref)|^2 / |E(ref)|^2), E = encoder of the frozen mandated ASR
# (SraVaani-1.0) -- in dB like the -SI-SDR term, so W is "dB of SI-SDR traded per dB of encoder distance".
W_ASR = float(os.environ.get("T2FT_WASR", "0"))
# T2FT_WKD > 0: frame-level distillation of the ASR's TDT joint posteriors (tokens + durations, blank context),
# clean-reference teacher -> W * 10*log10(mean KL). Targets only what drives decoding, not all encoder dims.
W_KD = float(os.environ.get("T2FT_WKD", "0"))
SEL = os.environ.get("T2FT_SEL", "fit")             # fit: best fitted SI-SDR score | last: final EMA
TICK_WER = int(os.environ.get("T2FT_TICKWER", "1" if W_ASR > 0 or W_KD > 0 else "0"))   # off0[:70] WER at every tick
EVAL1 = int(os.environ.get("T2FT_EVAL1", "0"))      # with T2FT_ONLY: evaluate that single model old vs new
P2 = dict(rate=0.47, p_silence=0.10, snr_mu=-9.7, snr_sd=4.0, mix_db=-22.8, max_ev=4.0)
MASK = dict(p_zero=0.10, p_drop=0.10, jitter=0.08, p_false=0.20)
PAD = 0.10
dev = E.device
torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------------- data
def group_key(stem):                                  # identical to track2_train.py folds
    toks = re.split(r"[_\-]", stem.lower())
    return "_".join(toks[:-1]) if len(toks) > 2 else stem.lower()


def fold(stem):
    return int(hashlib.md5(group_key(stem).encode()).hexdigest(), 16) % 5


class Pool:
    def __init__(self, name):
        self.index = json.load(open(f"{POOLS}/{name}.json")); self.name = name; self._mm = None

    def get(self, i):
        if self._mm is None:
            self._mm = np.memmap(f"{POOLS}/{self.name}.bin", dtype=np.int16, mode="r")
        r = self.index[i]
        return np.asarray(self._mm[r["o"]:r["o"] + r["n"]], dtype=np.float32) / 32768.0


class Bank:
    def __init__(self):
        z = np.load(os.environ.get("T2_NOISE_BANK", f"{PROJ}/checkpoints/noise_bank.npz"))
        o, t = z["o16"], z["t16"]
        self.t = [t[o[i]:o[i + 1]] for i in range(len(o) - 1)]


def alter(t, rng):
    """Small random change to a noise copy (Phase 2 copies are ~-26 dB off the validation originals)."""
    t = np.convolve(t, [1.0, rng.uniform(-0.12, 0.12), rng.uniform(-0.06, 0.06)], mode="same")
    if rng.random() < 0.5:
        r = 1 + rng.uniform(-0.004, 0.004); n = max(2, int(round(len(t) * r)))
        t = np.interp(np.linspace(0, len(t) - 1, n), np.arange(len(t)), t)
    t = t + rng.normal(0, 1, len(t)) * np.sqrt(np.mean(t ** 2) + 1e-12) * 10 ** (-35 / 20)
    return t.astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


class Mixer:
    def __init__(self, beds):
        self.beds = beds                               # list of (sampler weight, callable(rng) -> wav)
        self.bank, self.ev = Bank(), Pool("event")

    def bed(self, rng, n):
        parts, tot = [], 0
        w = np.array([b[0] for b in self.beds]); w = w / w.sum()
        while tot < n + SR:
            s = self.beds[rng.choice(len(self.beds), p=w)][1](rng)
            if len(s) < 0.5 * SR or rms(s) < 1e-4:
                continue
            s = s / rms(s) * 10 ** (rng.uniform(-29, -23) / 20)
            gap = np.zeros(int(rng.uniform(0, 0.3) * SR), np.float32)
            parts += [s, gap]; tot += len(s) + len(gap)
        c = np.concatenate(parts); o = int(rng.integers(0, len(c) - n + 1))
        return c[o:o + n].astype(np.float32)

    def noise(self, rng):
        if rng.random() < 0.7:
            t = self.bank.t[int(rng.integers(len(self.bank.t)))]
        else:
            t = self.ev.get(int(rng.integers(len(self.ev.index))))
        t = np.asarray(t, np.float32)
        if len(t) > P2["max_ev"] * SR:
            a = int(rng.integers(0, len(t) - int(P2["max_ev"] * SR))); t = t[a:a + int(rng.uniform(0.3, P2["max_ev"]) * SR)]
        return alter(t, rng)

    def make(self, rng, n=L, clean=None):
        c = self.bed(rng, n) if clean is None else clean.copy()
        n = len(c)
        noise, spans = np.zeros(n, np.float32), []
        for _ in range(max(1, rng.poisson(P2["rate"] * n / SR))):
            t = self.noise(rng)
            if len(t) >= n - 1 or len(t) < 0.05 * SR:
                continue
            for _try in range(20):
                a = int(rng.integers(0, n - len(t))); b = a + len(t)
                if all(b <= s or a >= e for s, e in spans):
                    break
            else:
                continue
            sp = rms(c) ** 2
            if rng.random() < P2["p_silence"]:          # noise over digital silence
                c[a:b] = 0.0
                snr = rng.normal(-6.0, 4.0); ref = sp
            else:
                snr = float(np.clip(rng.normal(P2["snr_mu"], P2["snr_sd"]), -22, 4))
                es = float(np.mean(c[a:b] ** 2)); ref = es if es > 1e-3 * sp else sp
            noise[a:b] += t / rms(t) * np.sqrt(ref * 10 ** (-snr / 10))
            spans.append((a, b))
        x = c + noise
        g = 10 ** (rng.normal(P2["mix_db"], 2.0) / 20) / rms(x)
        g = min(g, 0.99 / (np.abs(x).max() * 1.0 + 1e-9) * 1.0) if np.abs(x * g).max() > 0.99 else g
        return (x * g).astype(np.float32), (c * g).astype(np.float32), [(a / SR, b / SR) for a, b in sorted(spans)]


def degrade(wins, rng, dur):
    if rng.random() < MASK["p_zero"]:
        return []
    out = []
    for s0, e0 in wins:
        if rng.random() < MASK["p_drop"]:
            continue
        s1, e1 = s0 - PAD + rng.normal(0, MASK["jitter"]), e0 + PAD + rng.normal(0, MASK["jitter"])
        if e1 - s1 > 0.05:
            out.append((max(0.0, s1), min(dur, e1)))
    if rng.random() < MASK["p_false"] and dur > 0.5:
        st = rng.uniform(0, dur - 0.3); out.append((st, min(dur, st + rng.uniform(0.2, 1.5))))
    return sorted(out)


def train_beds():
    offc, sp, gc = Pool("offc"), Pool("speech"), Pool("gclean")
    held = {i for i, r in enumerate(offc.index) if fold(r["id"]) == 0}
    assert not held, f"{len(held)} held-out clips in the offc pool"
    pick = lambda p: (lambda rng: p.get(int(rng.integers(len(p.index)))))
    return [(0.5, pick(offc)), (0.25, pick(sp)), (0.25, pick(gc))]


def real_pair(rng, offc, offn):
    """A real organiser training-fold mixture (clean + noise from the pools, exact to 1 LSB), filled to L
    by concatenating clips; windows = the true noise support."""
    cs, ns, tot = [], [], 0
    while tot < L:
        i = int(rng.integers(len(offc.index)))
        c, n = offc.get(i), offn.get(i)
        g = np.zeros(int(rng.uniform(0, 0.2) * SR), np.float32)
        cs += [c, g]; ns += [n, g]; tot += len(c) + len(g)
    c, n = np.concatenate(cs), np.concatenate(ns)
    o = int(rng.integers(0, len(c) - L + 1)); c, n = c[o:o + L], n[o:o + L]
    act = np.convolve((np.abs(n) > 1e-4).astype(float), np.ones(800) / 800, mode="same") > 0
    edges = np.flatnonzero(np.diff(np.r_[0, act.astype(int), 0]))
    wins = [(a / SR, b / SR) for a, b in zip(edges[::2], edges[1::2]) if b - a > 0.05 * SR]
    x = c + n
    pk = float(np.abs(x).max())
    if pk > 0.99:
        x, c = x * 0.99 / pk, c * 0.99 / pk
    return x.astype(np.float32), c.astype(np.float32), wins


class Stream(torch.utils.data.IterableDataset):
    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng((wi.seed if wi else 1234) % 2 ** 32)
        mx = Mixer(train_beds())
        offc, offn = Pool("offc"), Pool("offn")
        while True:
            x, c, w = real_pair(rng, offc, offn) if rng.random() < P_REAL else mx.make(rng)
            ev = E.frames_from_windows(degrade(w, rng, L / SR), L)
            yield torch.from_numpy(x), torch.from_numpy(c), torch.from_numpy(ev)


# ---------------------------------------------------------------- loss (as track2_train.py)
def _mel_fb(n_fft=512, n_mels=128, fmin=20.0, fmax=7600.0):
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0); mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)) / SR).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for m in range(n_mels):
        l, c, r = bins[m], bins[m + 1], bins[m + 2]
        for k in range(l, min(c, fb.shape[1])):
            fb[m, k] = (k - l) / max(1, c - l)
        for k in range(c, min(r, fb.shape[1])):
            fb[m, k] = (r - k) / max(1, r - c)
    return torch.from_numpy(fb)


MELFB, MELWIN = _mel_fb().to(dev), torch.hann_window(512).to(dev)


def si_sdr_t(est, ref, eps=1e-8):
    est = est - est.mean(-1, keepdim=True); ref = ref - ref.mean(-1, keepdim=True)
    s = (est * ref).sum(-1, keepdim=True) / ((ref ** 2).sum(-1, keepdim=True) + eps) * ref
    return 10 * torch.log10((s ** 2).sum(-1) / (((est - s) ** 2).sum(-1) + eps) + eps)


def loss_fn(est, ref):
    valid = ((ref ** 2).mean(-1) > 1e-9).float()
    si = -(si_sdr_t(est, ref) * valid).sum() / valid.sum().clamp_min(1)
    mr = 0.0
    for nf, hp in [(512, 128), (1024, 256), (256, 64)]:
        w = torch.hann_window(nf, device=est.device)
        Eh = torch.stft(est, nf, hp, window=w, return_complex=True).abs()
        R = torch.stft(ref, nf, hp, window=w, return_complex=True).abs()
        mr = mr + torch.norm(R - Eh, p="fro") / (torch.norm(R, p="fro") + 1e-7) + F.l1_loss(torch.log(Eh + 1e-5), torch.log(R + 1e-5))
    mel = lambda x: torch.matmul(MELFB, torch.stft(x, 512, 160, window=MELWIN, return_complex=True).abs() ** 2)
    loss = si + 0.1 * mr / 3 + W_MEL * F.l1_loss(torch.log(mel(est) + 1e-5), torch.log(mel(ref) + 1e-5))
    if W_ASR > 0 or W_KD > 0:
        e_db, kd_db = Asr.get().losses(est, ref)
        a = (e_db * valid).sum() / valid.sum().clamp_min(1)
        k = (kd_db * valid).sum() / valid.sum().clamp_min(1)
        loss_fn.asr = float(a.detach()) if W_KD == 0 else float(k.detach())
        loss = loss + W_ASR * a + W_KD * k
    return loss


class Asr:
    """The frozen mandated ASR (SraVaani-1.0): transcription for eval, and a differentiable encoder for the
    loss. encode() replicates SraVaaniTDTModel.extract_features for full-length batches (max diff 2e-6)."""
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        from huggingface_hub import snapshot_download
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(snapshot_download("ARTPARK-IISc/SraVaani-1.0", token=os.environ.get("HF_TOKEN")),
                                           trust_remote_code=True).to(dev).eval()
        self.m._ensure_loaded()
        for p in self.m.encoder.parameters():
            p.requires_grad_(False)
        for p in self.m.decoder_joint.parameters():
            p.requires_grad_(False)
        self.P, self.win, self.fb = self.m._p, self.m._window.float(), self.m._fb.float()

    def transcribe(self, wavs):
        out = []
        with torch.no_grad():
            for i in range(0, len(wavs), 32):
                hs = self.m.transcribe(list(wavs[i:i + 32]), return_hypotheses=True)
                out += [(h.text if hasattr(h, "text") else str(h)).strip() for h in hs]
        return out

    def encode(self, wav):
        P = self.P; B, N = wav.shape; T = N // P["hop_length"]
        x = torch.cat((wav[:, :1], wav[:, 1:] - P["preemph"] * wav[:, :-1]), 1)
        S = torch.stft(x, P["n_fft"], P["hop_length"], P["win_length"], window=self.win, center=True,
                       pad_mode="constant", return_complex=True)
        m = torch.log(torch.matmul(self.fb, S.real ** 2 + S.imag ** 2) + P["log_zero_guard_value"])
        v = m[..., :T]; mu = v.mean(-1, keepdim=True)
        sd = ((v - mu) ** 2).sum(-1, keepdim=True).div(T - 1).sqrt() + P["CONSTANT"]
        f = torch.cat([(v - mu) / sd, torch.full_like(m[..., T:], P["pad_value"])], -1)
        return self.m.encoder(f, torch.full((B,), T, dtype=torch.long, device=wav.device))[0]

    def joint0(self, e):
        """TDT joint logits per encoder frame with a blank (start) prediction context -> (B, T, 5001 + 5)."""
        B = e.shape[0]; cfg = self.m.config
        h = torch.zeros(cfg.pred_rnn_layers, B, cfg.pred_hidden, device=e.device)
        tgt = torch.full((B, 1), cfg.blank_id, dtype=torch.int32, device=e.device)
        return self.m.decoder_joint(e, tgt, torch.ones(B, dtype=torch.int32, device=e.device), h, h.clone())[0][:, :, 0]

    def losses(self, est, ref):
        """(enc_db, kd_db) per item: encoder-distance dB and joint-posterior KL dB (teacher = clean ref)."""
        d = 3e-4 * torch.randn_like(ref)
        with torch.no_grad():
            er = self.encode(ref + d)
            lt = self.joint0(er) if W_KD > 0 else None
        ee = self.encode(est + d)
        e_db = 10 * torch.log10(((ee - er) ** 2).mean((1, 2)) / ((er ** 2).mean((1, 2)) + 1e-8) + 1e-4)
        if W_KD == 0:
            return e_db, torch.zeros_like(e_db)
        ls, V = self.joint0(ee), self.m.config.blank_id + 1
        kl = lambda a, b: (F.softmax(a, -1) * (F.log_softmax(a, -1) - F.log_softmax(b, -1))).sum(-1)
        k = (kl(lt[..., :V], ls[..., :V]) + kl(lt[..., V:], ls[..., V:])).mean(-1)
        return e_db, 10 * torch.log10(k + 1e-4)

    def enc_db(self, est, ref):
        """Per item 10*log10(|E(est)-E(ref)|^2 / |E(ref)|^2). A shared -70 dBFS dither on both inputs, so a
        residual far below the speech (e.g. over digital silence) is not a target."""
        d = 3e-4 * torch.randn_like(ref)
        with torch.no_grad():
            er = self.encode(ref + d)
        ee = self.encode(est + d)
        return 10 * torch.log10(((ee - er) ** 2).mean((1, 2)) / ((er ** 2).mean((1, 2)) + 1e-8) + 1e-4)


# ---------------------------------------------------------------- eval sets
def sisdr(e, r):
    e = e - e.mean(); r = r - r.mean(); a = np.dot(e, r) / (np.dot(r, r) + 1e-12); d = e - a * r
    return 10 * np.log10((np.dot(a * r, a * r) + 1e-12) / (np.dot(d, d) + 1e-12))


def q16(y):
    return np.round(np.clip(y, -1, 32767 / 32768) * 32768).astype(np.float32) / 32768


def eval_sets():
    md = json.load(open(f"{V}/validationMetadata.json"))
    t1 = {json.loads(l)["clip_id"]: [(max(0.0, e["onset"] - PAD), e["offset"] + PAD) for e in json.loads(l)["events"]]
          for l in open(f"{HERE}/val_t1_predictions.jsonl")}
    hold = [r for r in md if r.get("syntheticData") and fold(r["segmentFileName"][:-4]) == 0]
    off0, beds = [], []
    for r in hold:
        f = r["segmentFileName"]
        x, _ = sf.read(f"{V}/syntheticNoiseAudio/{f}", dtype="float32"); c, _ = sf.read(f"{V}/syntheticCleanRefAudio/{f}", dtype="float32")
        n = min(len(x), len(c)); off0.append(dict(mix=x[:n], clean=c[:n], wins=t1.get(f[:-4], []), ref=r.get("transcript") or ""))
        beds.append((c[:n], r.get("transcript") or ""))
    rng = np.random.default_rng(2026); mx = Mixer([(1.0, lambda g: np.zeros(SR, np.float32))])
    p2h = []
    for _ in range(12 if SMOKE else 120):                # ~12.7 s, like Phase 2 synthetic clips
        idx = rng.permutation(len(beds))[:3]
        c = np.concatenate([beds[i][0] / rms(beds[i][0]) * 10 ** (-26 / 20) for i in idx]).astype(np.float32)
        x, cl, w = mx.make(rng, clean=c)
        wins = [(max(0.0, a - PAD + rng.normal(0, 0.03)), b + PAD + rng.normal(0, 0.03)) for a, b in w]
        p2h.append(dict(mix=x, clean=cl, wins=wins, ref=" ".join(beds[i][1] for i in idx)))
    nat = sorted((r for r in md if not r.get("syntheticData")), key=lambda r: hashlib.md5(r["segmentFileName"].encode()).hexdigest())[:(6 if SMOKE else 300)]
    natl = []
    for r in nat:
        x, _ = sf.read(f"{V}/naturalNoisyAudio/{r['segmentFileName']}", dtype="float32")
        natl.append(dict(mix=x, clean=None, wins=t1.get(r["segmentFileName"][:-4], []), ref=r.get("transcript") or ""))
    return {"off0": off0[:(6 if SMOKE else None)], "p2h": p2h, "nat": natl}


def enhance_set(models, S):
    E.MODELS[:] = models
    return [q16(E.post(s["mix"], E.ens_enhance(s["mix"], s["wins"]), s["wins"], 1.0, 1.0)) for s in S]


# ---------------------------------------------------------------- train
def load(k):
    d = torch.load(f"{OLD}/t2_model_{k}.pt", map_location="cpu", weights_only=False)
    cfg = dict(d["cfg"]); cfg["ch"] = tuple(cfg["ch"])
    m = E.TFDPCRN(**cfg).to(dev); m.load_state_dict(d["state"]); return m.eval(), cfg


def fresh(k):
    """Random init, Phase 2 profile only. BIG uses the wider/deeper default arch."""
    cfg = dict(ch=(64, 96, 128, 192), n_dp=6, hid=160, attn=True, n_fft=512, hop=128) if BIG else \
        dict(ch=(48, 64, 96, 128), n_dp=5, hid=128, attn=True, n_fft=512, hop=128)
    torch.manual_seed(1000 + k)
    return E.TFDPCRN(**cfg).to(dev).eval(), cfg


def quick(model, S):
    """S = (off0 subset, p2h subset) -> fitted leaderboard score W_OFF0*off0 + W_P2H*p2h (SI-SDR means).
    With TICK_WER also the mandated-ASR WER on the off0 subset (quick.extra; not used for selection)."""
    model.eval()
    with torch.no_grad():
        outs = [[E.enhance_raw(model, s["mix"], s["wins"]) for s in part] for part in S]
    m = [float(np.mean([sisdr(o, s["clean"]) for o, s in zip(op, part)])) for op, part in zip(outs, S)]
    quick.last, quick.extra = m, ""
    if TICK_WER:
        hyp = Asr.get().transcribe([q16(E.post(s["mix"], o, s["wins"], 1.0, 1.0)) for o, s in zip(outs[0], S[0])])
        e = n = 0
        for h, s in zip(hyp, S[0]):
            a, b = wer_counts(s["ref"], h); e += a; n += b
        quick.extra = f" wer_off0 {100 * e / max(n, 1):.2f}"
    return m[1] if SELECT == "p2h" else W_OFF0 * m[0] + W_P2H * m[1]


def finetune(k, val):
    model, cfg = fresh(k) if SCRATCH else load(k)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, betas=(0.9, 0.98))
    amp = torch.bfloat16 if (dev == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    dl = torch.utils.data.DataLoader(Stream(), batch_size=BS, num_workers=0 if SMOKE else 8,
                                     pin_memory=(dev == "cuda"), persistent_workers=not SMOKE)
    best = (quick(ema, val), copy.deepcopy(ema.state_dict())); print(f"model {k}: start fit-score {best[0]:.3f} (off0 {quick.last[0]:.2f}, p2h {quick.last[1]:.2f}){quick.extra}", flush=True)
    t0, step, run, arun, nxt = time.time(), 0, [], [], time.time() + (60 if SMOKE else 600)
    for x, c, ev in dl:
        frac = (time.time() - t0) / (MINUTES * 60)
        if frac >= 1 or (SMOKE and step >= 3):
            break
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, (step + 1) / 300) * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * frac)))
        model.train()
        x, c, ev = x.to(dev), c.to(dev), ev.to(dev)
        try:
            with torch.autocast(device_type=dev, dtype=amp, enabled=(dev == "cuda")):
                est = model(x, ev)
            loss = loss_fn(est.float(), c)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); print("  OOM, batch skipped", flush=True); continue
        d = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        run.append(float(loss)); step += 1
        if W_ASR > 0 or W_KD > 0:
            arun.append(loss_fn.asr)
        if time.time() >= nxt:
            q = quick(ema, val); nxt = time.time() + (60 if SMOKE else 600)
            if SEL == "fit" and q > best[0]:
                best = (q, copy.deepcopy(ema.state_dict()))
            if W_ASR > 0 or W_KD > 0:                  # snapshots: SEL=last keeps only the final EMA
                os.makedirs(f"{OUT}/snaps", exist_ok=True)
                torch.save({"state": ema.state_dict(), "cfg": cfg}, f"{OUT}/snaps/m{k}_{(time.time() - t0) / 60:.0f}min.pt")
            at = f" ({'kd' if W_KD > 0 else 'asr'} {np.mean(arun[-200:]):.2f} dB)" if arun else ""
            print(f"  model {k} step {step} loss {np.mean(run[-200:]):.3f}{at} | fit {q:.3f} (off0 {quick.last[0]:.2f} p2h {quick.last[1]:.2f}; best {best[0]:.3f}){quick.extra} | {(time.time() - t0) / 60:.0f} min", flush=True)
    q = quick(ema, val)
    print(f"model {k}: final EMA fit {q:.3f} (off0 {quick.last[0]:.2f} p2h {quick.last[1]:.2f}){quick.extra}", flush=True)
    if SEL == "last" or q > best[0]:
        best = (q, copy.deepcopy(ema.state_dict()))
    os.makedirs(EXPORT, exist_ok=True)
    torch.save({"state": best[1], "cfg": cfg}, f"{EXPORT}/t2_model_{k}.pt")
    print(f"model {k}: {step} steps, best fit-score {best[0]:.3f} -> {EXPORT}/t2_model_{k}.pt", flush=True)


def main():
    S = eval_sets()
    print(f"{'SCRATCH' if SCRATCH else 'init ' + OLD} | real {P_REAL} | big {BIG} | w_mel {W_MEL} | lr {LR} | "
          f"{MINUTES} min/model | only {ONLY} | w_asr {W_ASR} | w_kd {W_KD} | sel {SEL}", flush=True)
    print({k: len(v) for k, v in S.items()}, "| off0 input SI-SDR %.2f | p2h input SI-SDR %.2f" % (
        np.mean([sisdr(s["mix"], s["clean"]) for s in S["off0"]]), np.mean([sisdr(s["mix"], s["clean"]) for s in S["p2h"]])), flush=True)
    val = (S["off0"][:70], S["p2h"][:40])
    if TICK_WER:
        e = n = 0
        for h, s_ in zip(Asr.get().transcribe([q16(s_["mix"]) for s_ in val[0]]), val[0]):
            a, b = wer_counts(s_["ref"], h); e += a; n += b
        print(f"noisy wer_off0 (first 70) {100 * e / max(n, 1):.2f}", flush=True)
    for k in ([int(ONLY)] if ONLY else range(3)):
        if not os.path.exists(f"{EXPORT}/t2_model_{k}.pt"):
            finetune(k, val)
    if ONLY and not EVAL1:                            # parallel job: train only, compare later
        print(f"trained model {ONLY} -> {EXPORT}; skipping the ensemble eval", flush=True)
        return
    ks = [int(ONLY)] if ONLY else list(range(3))      # EVAL1: the single model, old vs new
    old = [load(k)[0] for k in ks]
    new = []
    for k in ks:
        d = torch.load(f"{EXPORT}/t2_model_{k}.pt", map_location="cpu", weights_only=False)
        m = E.TFDPCRN(**d["cfg"]).to(dev); m.load_state_dict(d["state"]); new.append(m.eval())
    if not ONLY:
        meta = json.load(open(f"{OLD}/t2_export.json"))
        json.dump({**meta, "finetune": "phase2-profile", "p2": P2, "init": OLD, "w_mel": W_MEL, "lr": LR,
                   "w_asr": W_ASR, "real": P_REAL}, open(f"{EXPORT}/t2_export.json", "w"), indent=1)

    A = Asr.get()

    def counts(S_, outs):                             # per-clip (errors, ref words)
        return [wer_counts(s["ref"], h) for s, h in zip(S_, A.transcribe(outs))]

    def wer(c, idx):
        return sum(c[i][0] for i in idx) / max(1, sum(c[i][1] for i in idx))

    rep = {}
    for name, S_ in S.items():
        c0 = counts(S_, [q16(s["mix"]) for s in S_])
        subs = {name: range(len(S_))}
        if name == "off0":
            subs["off0b"] = range(70, len(S_))        # never used for checkpoint selection
        if name != "nat":                             # clean-reference ceiling
            cc = counts(S_, [q16(s["clean"]) for s in S_])
            for sn, idx in subs.items():
                rep[f"{sn}_clean"] = {"dwer": 100 * (wer(c0, idx) - wer(cc, idx)), "wer": 100 * wer(cc, idx)}
                print(f"{sn:5s} clean: noisy wer {100 * wer(c0, idx):.2f} | ceiling dwer {rep[sn + '_clean']['dwer']:.2f} wer {100 * wer(cc, idx):.2f}", flush=True)
        for tag, ms in (("old", old), ("new", new)):
            outs = enhance_set(ms, S_)
            c = counts(S_, outs)
            for sn, idx in subs.items():
                r = {"dwer": 100 * (wer(c0, idx) - wer(c, idx)), "wer": 100 * wer(c, idx)}
                if name != "nat":
                    r["sisdr"] = float(np.mean([sisdr(outs[i], S_[i]["clean"]) for i in idx]))
                rep[f"{sn}_{tag}"] = r
                print(f"{sn:5s} {tag}: noisy wer {100 * wer(c0, idx):.2f} | " + " ".join(f"{k} {v:.2f}" for k, v in r.items()), flush=True)
    json.dump(rep, open(f"{OUT}/eval{'_' + ONLY if ONLY else ''}.json", "w"), indent=1)
    for tag in ("old", "new"):
        p = rep[f"p2h_{tag}"]["sisdr"] + 0.32 * rep[f"p2h_{tag}"]["dwer"] + 0.68 * rep[f"nat_{tag}"]["dwer"]
        print(f"PROXY Phase-2-like ({tag}) = p2h SI-SDR + 0.32*p2h dWER + 0.68*nat dWER = {p:.2f}", flush=True)
    d = lambda key, f: rep[f"{key}_new"][f] - rep[f"{key}_old"][f]
    est = W_OFF0 * d("off0", "sisdr") + W_P2H * d("p2h", "sisdr") + 0.32 * d("off0", "dwer") + 0.68 * d("nat", "dwer")
    print(f"EST leaderboard change (new - old) = 1.53*dSI(off0) + 0.27*dSI(p2h) + 0.32*d dWER(off0) + 0.68*d dWER(nat) "
          f"= {W_OFF0 * d('off0', 'sisdr'):+.2f} {W_P2H * d('p2h', 'sisdr'):+.2f} {0.32 * d('off0', 'dwer'):+.2f} "
          f"{0.68 * d('nat', 'dwer'):+.2f} = {est:+.2f}", flush=True)


if __name__ == "__main__":
    main()
