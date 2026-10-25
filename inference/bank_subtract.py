"""Exact subtraction of matched bank copies (Track 2, Phase 2).

For every bank match with correlation >= TH (0.45), strongest first: refine the placement by +-32 samples on the
double pre-emphasised signal, fit a non-negative scalar gain by least squares on the whitened signals, and
subtract gain * template (raw 16 kHz noise of that recording). Phase 2 copies are slightly altered (~-26 dB off
the validation originals), so a residual remains; inference/enhance_and_transcribe.py (T2_BANKPOST=1) cleans it.

Writes <out_dir>/<clip>.wav (float32) and <out_dir>/subtracted_spans.json {clip: [[onset, offset], ...]}.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import soundfile as sf

SR = 16000
LAG = 32
TH = 0.45


def _pe(x):
    return np.append(x[:1], x[1:] - 0.97 * x[:-1])


def _w2(x):
    return _pe(_pe(x))


def _gain(seg, t):
    ws, wt = _w2(seg), _w2(t)
    return max(0.0, float(np.dot(ws, wt) / (np.dot(wt, wt) + 1e-12)))


def subtract_clip(y, matches, templates16):
    out = y.astype(np.float64).copy()
    placed = []
    for m in sorted(matches, key=lambda x: -x["cor"]):
        t = templates16[m["k"]]
        L = len(t)
        p = int(round(m["onset"] * SR))
        best = None
        for d in range(-LAG, LAG + 1):
            a = p + d
            if a < 0 or a + L > len(out):
                continue
            v = abs(np.dot(_w2(out[a:a + L]), _w2(t)))
            if best is None or v > best[0]:
                best = (v, a)
        if best is None:
            continue
        a = best[1]
        out[a:a + L] -= _gain(out[a:a + L], t) * t
        placed.append((a / SR, (a + L) / SR))
    return out, placed


def run(audio_dir, matches, templates16, out_dir, th=TH):
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(c, [m for m in ms if m["cor"] >= th]) for c, ms in sorted(matches.items())]
    jobs = [(c, ms) for c, ms in jobs if ms]
    print(f"bank subtraction: {len(jobs)} clips", flush=True)
    spans, t0 = {}, time.time()
    for i, (cid, ms) in enumerate(jobs):
        p = os.path.join(audio_dir, f"{cid}.wav")
        if not os.path.exists(p):
            continue
        x, sr = sf.read(p, dtype="float64")
        x = x.mean(1) if x.ndim > 1 else x
        y, done = subtract_clip(x, ms, templates16)
        sf.write(os.path.join(out_dir, f"{cid}.wav"), y.astype(np.float32), sr, subtype="FLOAT")
        spans[cid] = done
        if (i + 1) % 250 == 0 or i + 1 == len(jobs):
            print(f"  [{i + 1}/{len(jobs)}] {time.time() - t0:.0f}s | spans {sum(map(len, spans.values()))}", flush=True)
    json.dump(spans, open(os.path.join(out_dir, "subtracted_spans.json"), "w"), indent=2)
    return spans
