# CodeAMU — IndoML 2026 Datathon, Track 2: Noise Event Removal

**Team CodeAMU** — Shah Ahmed Shakir Abu Asim Khan (Aligarh Muslim University), Sajid Javid (IIIT Delhi),
Osama Zaheer (Binary Semantics), Ahraz Shamim (Vecmocon Technologies)
**Primary contact:** Sajid Javid — sajidj@iiitd.ac.in

**Final standing: 1st place, Phase 2 (final) leaderboard — Combined 26.4363** (`submission_t2_phase2_bankpost_p2mix.zip`).
Technical report: [`docs/CodeAMU_track_2_1.pdf`](docs/CodeAMU_track_2_1.pdf).
Companion Track 1 system (event windows and routing used as inputs here): <https://github.com/sajidjavid222/CodeAMU_Track1>.

## System overview

```
noisy clip ──► [route: synthetic? (Track 1 router)] ──yes──► bank matching (validation noise bank, FFT, 8 kHz)
                                                              └─► exact subtraction of matched copies (fitted gain)
            ──► TF-DPCRN ensemble (3 members), conditioned on the Track 1 event windows
                  · bank-subtracted clips: enhancer pass over (subtracted spans ± 0.1 s) ∪ Track 1 windows
                  · other clips: enhancer over the whole clip, blended 1.0 inside / 0.9 outside the windows
            ──► RMS restored, peak guard, 16 kHz PCM_16 WAV ──► mandated ASR (SraVaani-1.0) ──► transcripts.jsonl
```

* **Enhancer** — TF-DPCRN (3.9 M parameters per member): power-law compressed complex STFT (n_fft 512, hop 128) plus a
  per-frame noise-event indicator in, a convolutional encoder (48/64/96/128 channels), 5 dual-path blocks (frequency
  BiLSTM, time BiLSTM, frequency-pooled self-attention), a transposed-convolution decoder with skips, and a bounded
  complex ratio mask out. Three members, polarity-aligned waveform averaging. Trained from scratch (no pretrained
  weights) on our own mixtures and the official training-fold pairs, then fine-tuned on mixtures that follow the
  Phase 2 test statistics (see `train/config.yaml`).
* **Noise bank** — the official validation release ships the clean reference of every synthetic clip, so
  noise = mixture − clean at the labelled timestamps gives the organisers' noise pool: 2,213 labelled events →
  1,181 distinct recordings (`checkpoints/noise_bank.npz`, bit-identical to rebuilding it from the validation folder).
  Matched copies in the test audio are subtracted with a least-squares gain; Phase 2 copies are slightly altered,
  and the enhancer pass cleans the residual.
* **ASR** — the mandated `ARTPARK-IISc/SraVaani-1.0`, unchanged, writes `transcripts.jsonl` from the enhanced WAVs.

## Repository layout

```
CodeAMU_Track2/
├── README.md
├── requirements_train.txt / requirements_inference.txt
├── inference/
│   ├── infer.py                    # entry point: bank matching -> subtraction -> enhancement + ASR
│   ├── bank.py                     # noise bank (from the npz or the validation folder) + FFT matcher
│   ├── bank_subtract.py            # exact subtraction of matched copies
│   └── enhance_and_transcribe.py   # TF-DPCRN ensemble, event-window conditioning, post-processing, SraVaani ASR
├── train/
│   ├── track2_train.py             # data pools + base TF-DPCRN members (curriculum modes, polish stage)
│   ├── t2_finetune_p2.py           # Phase 2-profile fine-tuning (stages 2-5)
│   ├── enhance.py, asr_wer.py      # model definition / WER helpers used in training
│   ├── config.yaml                 # every training stage, exactly as run
│   └── val_t1_predictions.jsonl    # Track 1 windows for the validation set (checkpoint selection)
├── checkpoints/
│   ├── t2_model_{0,1,2}.pt         # FINAL enhancer ensemble (15.6 MB each)
│   ├── t2_export.json
│   └── noise_bank.npz              # validation noise bank (75 MB)
├── configs/model_config.yaml       # architecture + inference settings
├── inputs/phase2/
│   ├── track1_predictions.jsonl    # Track 1 event timestamps for the Phase 2 test clips (CodeAMU Track 1 system)
│   └── route.json                  # synthetic/natural routing of the Phase 2 test clips (Track 1 router)
├── scripts/run_inference.sh, scripts/train_all.sh
└── docs/CodeAMU_track_2_1.pdf      # technical report
```

## Environment and hardware

| | |
|---|---|
| Python | 3.10.12 |
| PyTorch / CUDA / cuDNN | 2.11.0+cu128 / CUDA 12.8 / cuDNN 9.19 |
| Training GPU | NVIDIA H200 (141 GB), used as MIG `3g.71gb` slices (71 GB GPU memory) under SLURM; 10 CPU cores, 60 GB RAM per job, 6 h per job |
| Inference GPU | same (one 3g.71gb slice); ≈40–50 min for the 3,178 Phase 2 clips (bank matching ≈18 min, subtraction ≈1 min, enhancement + ASR ≈19 min). Runs on CPU too (slower). |
| OS | Ubuntu 22.04.4 LTS (Linux 5.15), x86-64 |
| System packages | `libsndfile1` (soundfile) |

**Pretrained models / external resources**

| Resource | Version / source | Role |
|---|---|---|
| ARTPARK-IISc/SraVaani-1.0 | Hugging Face, revision `f5dd5358325a5208775b91dad98918e079ea2b27` (gated) | mandated ASR for `transcripts.jsonl` (unchanged) |
| ARTPARK-IISc/Vaani-Noise-Event-Dataset | Hugging Face dataset (gated) | gold noise events, clean gaps and floors for training mixtures |
| ARTPARK-IISc/Vaani | Hugging Face dataset (gated) | speech beds for training mixtures |
| Official Track 2 validation release | competition "Files" page | organiser synthetic pairs (training fold), noise bank, held-out evaluation |

No pretrained weights are used by the enhancer.

## Setup

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
pip install -r requirements_inference.txt          # or requirements_train.txt for training
export HF_TOKEN=hf_xxx                             # access to ARTPARK-IISc/SraVaani-1.0 (gated)
```

## Inference

```bash
bash scripts/run_inference.sh /path/to/phase2/audio /path/to/output
# equivalent:
python inference/infer.py --input_dir /path/to/phase2/audio --output_dir /path/to/output \
    --t1_predictions inputs/phase2/track1_predictions.jsonl --route inputs/phase2/route.json \
    [--val_dir /path/to/validation]      # optional: rebuild the noise bank from the official validation folder
```

**Inputs**
* `--input_dir`: noisy test clips, 16 kHz mono WAV, one file per clip (`<clip_id>.wav`).
* `--t1_predictions`: Track 1 noise-event timestamps, one JSON object per line
  `{"clip_id": ..., "events": [{"onset": s, "offset": s}, ...]}` — produced for the Phase 2 test set by the CodeAMU
  Track 1 system ([CodeAMU_Track1](https://github.com/sajidjavid222/CodeAMU_Track1), file shipped in `inputs/phase2/`).
* `--route`: `{clip_id: true|false}` synthetic routing from the Track 1 synthetic/natural routers (shipped).

Both input files are regenerated from the audio alone by the Track 1 repository:
`python inference/run_pipeline.py --audio_dir <phase2 audio> --run_dir <dir> --val_dir <validation> --stages 1,2 --track2_inputs`
writes `<dir>/output/track2_inputs/{track1_predictions.jsonl, route.json}` (24 detectors, decoding under the combined
audio + detector routing). Checked: run on the cached stage 1 detector outputs of the original Phase 2 run, it
reproduces both shipped files byte for byte (3,178 clips, 1,635 routed synthetic, 13,008 events).

**Outputs** (in `--output_dir`)
* `<clip_id>.wav` for every input clip: 16 kHz, mono, PCM 16-bit, same length as the input.
* `transcripts.jsonl`: `{"clip_id": ..., "text": ...}` per clip, from SraVaani-1.0 on the enhanced audio.
* `submission.zip`: all WAVs + `transcripts.jsonl` at the zip root (Codabench format).

**Reproduction check.** Running `inference/infer.py` (what `scripts/run_inference.sh` calls) on the Phase 2 test audio (one H200
`3g.71gb` slice, 49 min) regenerates the submitted zip exactly: all 3,178 WAVs are bit-identical to the submitted
ones and all 3,178 transcripts are identical.

## Training

```bash
export T2_DATA=/path/to/official/data      # must contain validation/ (official Track 2 validation release)
export HF_TOKEN=hf_xxx                     # Vaani datasets are gated
bash scripts/train_all.sh
```

| Stage | Script | What | Time |
|---|---|---|---|
| 0 | `track2_train.py` (`T2_POOLS_ONLY=1`) | data pools: Vaani speech beds, gold events / clean gaps / floors, official train-fold pairs, harvested floors | ≈2 h |
| 1 | `track2_train.py` (`T2_MODE=F`) × 3 | base members: 185 min curriculum training + 25 min polish each; held-out SI-SDR 22.8 / 22.7 / 22.6 dB | 3 × ≈4 h (one 6 h job each) |
| 2 | `t2_finetune_p2.py` | p2ft: Phase 2-profile mixtures, 40 min / member, lr 2e-4 | ≈2.5 h |
| 3 | `t2_finetune_p2.py` | p2ft_cont: 45 min / member, lr 1e-4 | ≈2.5 h |
| 4 | `t2_finetune_p2.py` × 3 | p2ft_long: 300 min / member, lr 1e-4 | 3 × ≈5.5 h |
| 5 | `t2_finetune_p2.py` × 3 | p2mix (final): 240 min / member, lr 1e-4, real organiser pairs 30 / 50 / 30 % | 3 × ≈4.5 h |

Losses, mixture statistics, window augmentation, EMA, schedule and checkpoint selection are listed in
`train/config.yaml`. Held-out evaluation uses the 138 official synthetic pairs of filename-prefix fold 0, whose
clean speech is never used in training (their noise recordings, like every pool recording, are in the noise bank that
the fine-tuning mixer draws from).

## Notes

* Every clip is processed by the enhancer; the bank subtraction is an additional step for clips in which matched
  copies of pool recordings are found.
* `transcripts.jsonl` is always produced by the mandated SraVaani-1.0 run on the submitted WAVs.
