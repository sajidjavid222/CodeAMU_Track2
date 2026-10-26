"""Track 2 neural enhancer (TF-DPCRN, 3-model ensemble), used for clips routed as NATURAL and
for synthetic clips where no refnoise recording was found. Conditioned on Track 1 event windows.
Code copied verbatim from the inference script that produced the leaderboard submission.
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

SR = 16000
N_FFT, HOP = 512, 128
device = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = []


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


def ens_enhance(wav, wins):
    ys = [enhance_raw(m, wav, wins) for m in MODELS]
    n = min(len(y) for y in ys)
    # align polarity to the mixture before averaging (a model can learn an inverted output;
    # SI-SDR cannot see it, but averaging inverted + normal estimates cancels the speech)
    x = np.asarray(wav, np.float32)[:n]
    ys = [y[:n] if np.dot(y[:n], x) >= 0 else -y[:n] for y in ys]
    return np.mean(ys, axis=0).astype(np.float32)


def load_models(paths):
    global N_FFT, HOP
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        sd = d["state"] if isinstance(d, dict) and "state" in d else d
        cfg = dict(d["cfg"]); cfg["ch"] = tuple(cfg["ch"])
        N_FFT, HOP = int(cfg.get("n_fft", 512)), int(cfg.get("hop", 128))
        m = TFDPCRN(**cfg).to(device); m.load_state_dict(sd); MODELS.append(m.eval())
    return MODELS
