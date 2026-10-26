"""Validation reference noise + FFT template matcher (Track 2, Phase 2).

The organisers' synthetic clips are clean speech plus noise segments pasted WHOLE from a finite pool. In the
official validation set every synthetic clip ships its clean reference, so noise = mixture - clean, cut at the
labelled event timestamps, gives that pool ("the refnoise"): 2,213 labelled events -> 1,181 distinct recordings.

Matching (8 kHz, pre-emphasis whitening 0.97, unit-norm templates): batched FFT cross-correlation of a clip
against every template, normalised by the local signal energy; up to 3 peaks per template above TH_KEEP; then a
greedy pick by explained energy, rejecting candidates overlapping an earlier pick by > 30 %.

The refnoise can be built from the official validation folder (build_templates_from_validation) or loaded from the
exported checkpoints/reference_noise.npz (load_templates_npz); both give identical templates.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import scipy.signal as ss
import soundfile as sf
import torch

SR8 = 8000
TH_KEEP = 0.20


def white(x):
    """Pre-emphasis whitening: exact copies stay exact (linear filter), chance alignments of tonal sounds drop."""
    return np.append(x[:1], x[1:] - 0.97 * x[:-1])


def read8k(path):
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(1)
    return white(ss.resample_poly(x, SR8, sr) if sr != SR8 else x)


def build_templates_from_validation(val_dir):
    """Templates exactly as used for the submissions (src: refnoise_match.build_templates with PE=True, NB=False)."""
    md = [r for r in json.load(open(f"{val_dir}/validationMetadata.json")) if r.get("syntheticData")]
    raw = []
    for r in md:
        f = r["segmentFileName"]
        x, sr = sf.read(f"{val_dir}/syntheticNoiseAudio/{f}", dtype="float64")
        c, _ = sf.read(f"{val_dir}/syntheticCleanRefAudio/{f}", dtype="float64")
        x = x.mean(1) if x.ndim > 1 else x
        c = c.mean(1) if c.ndim > 1 else c
        k = min(len(x), len(c))
        n = white(ss.resample_poly(x[:k] - c[:k], SR8, sr))
        for e in r["NoiseSubCategoryTimeStamp"]:
            a, b = int(round(float(e["start"]) * SR8)), int(round(float(e["end"]) * SR8))
            t = n[a:b]
            if len(t) >= int(0.05 * SR8) and np.linalg.norm(t) > 1e-6:
                raw.append((t / np.linalg.norm(t), f[:-4], float(e["start"]), float(e["end"])))
    # dedupe identical recordings (same length +-8 samples, normalised dot >= 0.995)
    raw.sort(key=lambda z: len(z[0]))
    refnoise, meta = [], []
    for t, cid, s0, s1 in raw:
        hit = None
        for j in range(len(refnoise) - 1, -1, -1):
            if len(refnoise[j]) < len(t) - 8:
                break
            m = min(len(t), len(refnoise[j]), 4000)
            if abs(np.dot(t[:m], refnoise[j][:m])) / (np.linalg.norm(t[:m]) * np.linalg.norm(refnoise[j][:m]) + 1e-12) >= 0.995:
                hit = j
                break
        if hit is None:
            refnoise.append(t); meta.append((cid, s0, s1))
    print(f"refnoise: {len(raw)} labelled events -> {len(refnoise)} distinct recordings", flush=True)
    return refnoise, meta


def template16_from_validation(val_dir, meta_k):
    """Raw 16 kHz noise of template k (mixture - clean at its labelled span), float64."""
    cid, s0, s1 = meta_k
    x, _ = sf.read(f"{val_dir}/syntheticNoiseAudio/{cid}.wav", dtype="float64")
    c, _ = sf.read(f"{val_dir}/syntheticCleanRefAudio/{cid}.wav", dtype="float64")
    x = x.mean(1) if x.ndim > 1 else x
    c = c.mean(1) if c.ndim > 1 else c
    n = min(len(x), len(c))
    return (x[:n] - c[:n])[int(round(s0 * 16000)):int(round(s1 * 16000))]


def load_templates_npz(path):
    """checkpoints/reference_noise.npz: t8/o8 = 8 kHz whitened unit-norm templates, t16/o16 = raw 16 kHz templates."""
    z = dict(np.load(path))                      # read every array once (NpzFile re-reads on each key access)
    t8, o8, t16a, o16 = z["t8"], z["o8"], z["t16"], z["o16"]
    refnoise = [t8[o8[i]:o8[i + 1]] for i in range(len(o8) - 1)]
    t16 = [t16a[o16[i]:o16[i + 1]].astype(np.float64) for i in range(len(o16) - 1)]
    meta = [(str(c), float(a), float(b)) for c, a, b in zip(z["src_clip"], z["src_start"], z["src_end"])]
    return refnoise, meta, t16


def export_templates_npz(val_dir, path):
    """Rebuild checkpoints/reference_noise.npz from the official validation set."""
    refnoise, meta = build_templates_from_validation(val_dir)
    t16 = [template16_from_validation(val_dir, m).astype(np.float32) for m in meta]
    o8 = np.cumsum([0] + [len(t) for t in refnoise]); o16 = np.cumsum([0] + [len(t) for t in t16])
    np.savez(path, t8=np.concatenate(refnoise), o8=o8, t16=np.concatenate(t16), o16=o16,
             src_clip=np.array([m[0] for m in meta]), src_start=np.array([m[1] for m in meta]),
             src_end=np.array([m[2] for m in meta]))


class GPUMatcher:
    def __init__(self, refnoise, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.refnoise = refnoise
        self.lens = np.array([len(t) for t in refnoise])
        self.cache = {}
        print(f"matcher on {self.device} with {len(refnoise)} templates", flush=True)

    def tf(self, N):
        if N not in self.cache:
            self.cache.clear()
            T = torch.zeros(len(self.refnoise), N, dtype=torch.float32, device=self.device)
            for i, t in enumerate(self.refnoise):
                T[i, :len(t)] = torch.from_numpy(t.astype(np.float32)).to(self.device)
            self.cache[N] = torch.fft.rfft(T, n=N)
        return self.cache[N]

    def match(self, x, chunk=512):
        Lx = len(x)
        N = 1 << int(np.ceil(np.log2(Lx + self.lens.max())))
        Tf = self.tf(N)
        X = torch.fft.rfft(torch.from_numpy(x.astype(np.float32)).to(self.device), n=N)
        cs = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
        cands = []
        for s in range(0, len(self.refnoise), chunk):
            idx = torch.arange(s, min(s + chunk, len(self.refnoise)), device=self.device)
            xc = torch.fft.irfft(X[None, :] * torch.conj(Tf[idx]), n=N)[:, :Lx].cpu().numpy()
            for r, k in enumerate(np.arange(s, min(s + chunk, len(self.refnoise)))):
                n = self.lens[k]
                if n > Lx:
                    continue
                v = xc[r, :Lx - n + 1]
                we = np.sqrt(np.maximum(cs[n:Lx + 1] - cs[:Lx - n + 1], 1e-12))
                cor = np.abs(v) / we
                for _ in range(3):                      # the same noise may be pasted more than once
                    j = int(np.argmax(cor))
                    if cor[j] < TH_KEEP:
                        break
                    cands.append((float(v[j] ** 2), float(cor[j]), j, int(k)))
                    cor[max(0, j - n // 2):j + n // 2 + 1] = 0
        cands.sort(reverse=True)                         # by explained energy
        picked = []
        for ee, c, j, k in cands:
            a, b = j, j + self.lens[k]
            if any(min(b, B) - max(a, A) > 0.3 * min(b - a, B - A) for A, B, *_ in picked):
                continue
            picked.append((a, b, c, ee, k))
        return [dict(onset=a / SR8, offset=b / SR8, cor=round(c, 4), ee=ee, k=k) for a, b, c, ee, k in picked]


def match_clips(audio_dir, clip_ids, refnoise, out_json):
    if os.path.exists(out_json):
        print(f"{out_json} exists, reusing", flush=True)
        return json.load(open(out_json))
    M = GPUMatcher(refnoise)
    res, t0 = {}, time.time()
    for i, cid in enumerate(clip_ids):
        p = os.path.join(audio_dir, f"{cid}.wav")
        if os.path.exists(p):
            res[cid] = M.match(read8k(p))
        if (i + 1) % 200 == 0 or i + 1 == len(clip_ids):
            print(f"  matched {i + 1}/{len(clip_ids)} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    json.dump(res, open(out_json, "w"))
    return res
