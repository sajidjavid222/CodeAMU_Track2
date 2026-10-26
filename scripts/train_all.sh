#!/usr/bin/env bash
# CodeAMU - Track 2: full training chain of the enhancer ensembles in checkpoints/enhancer/ (see train/config.yaml).
# Each python call below ran as one job on a 6 h queue (1x NVIDIA H200 MIG 3g.71gb slice, 10 CPU, 60 GB RAM).
#   T2_DATA=<folder holding the official Track 2 validation data> HF_TOKEN=hf_xxx bash scripts/train_all.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
: "${T2_DATA:?folder with the official validation data}" "${HF_TOKEN:?HF token (Vaani datasets are gated)}"
export T2_WORK="$ROOT/work/t2work" T2_BUDGET=omni T2_VAL_DIR="${T2_VAL_DIR:-$T2_DATA/validation}"
export T2_POOLS="$T2_WORK/pools" T2_NOISE_BANK="$ROOT/checkpoints/noise_bank.npz"

# stage 0: data pools (Vaani speech, gold events/floors, official train-fold pairs, harvested floors)
T2_POOLS_ONLY=1 python train/track2_train.py
# stage 1: three base members (mode F + polish P); each is its own 6 h session
T2_POOLS_READY=1 T2_MODE=F python train/track2_train.py
mkdir -p work/members && cp "$T2_WORK/export/t2_model_0.pt" work/members/m0.pt
T2_POOLS_READY=1 T2_MODE=F T2_KEEP=work/members/m0.pt T2_KEEP_SCORE=22.822 T2_N_MODELS=2 python train/track2_train.py
cp "$T2_WORK/export/t2_model_1.pt" work/members/m1.pt
T2_POOLS_READY=1 T2_MODE=F T2_EVAL_ONLY=work/members/m0.pt,work/members/m1.pt T2_EVAL_SCORES=22.822,22.683 \
    T2_TRAIN_MORE=1 python train/track2_train.py          # -> work/t2work/export/t2_model_{0,1,2}.pt

# stages 2-5: Phase 2 profile fine-tuning (train/t2_finetune_p2.py)
T2FT_INIT=work/t2work/export   T2FT_OUT=work/t2_p2ft      T2FT_MIN=40  T2FT_LR=2e-4 T2FT_SELECT=p2h python train/t2_finetune_p2.py
T2FT_INIT=work/t2_p2ft/export  T2FT_OUT=work/t2_p2ft_cont T2FT_MIN=45  T2FT_LR=1e-4 T2FT_SELECT=p2h python train/t2_finetune_p2.py
for K in 0 1 2; do
  T2FT_INIT=work/t2_p2ft_cont/export T2FT_OUT=work/t2_p2ft_long T2FT_ONLY=$K T2FT_MIN=300 T2FT_LR=1e-4 T2FT_SELECT=p2h \
      python train/t2_finetune_p2.py
done
cp work/t2_p2ft_cont/export/t2_export.json work/t2_p2ft_long/export/
REAL=(0.3 0.5 0.3)
for K in 0 1 2; do
  T2FT_INIT=work/t2_p2ft_long/export T2FT_OUT=work/t2_p2mix T2FT_ONLY=$K T2FT_REAL=${REAL[$K]} T2FT_MIN=240 T2FT_LR=1e-4 \
      T2FT_SELECT=fit python train/t2_finetune_p2.py
done
mkdir -p checkpoints/enhancer/p2mix checkpoints/enhancer/asrbt checkpoints/enhancer/sisdr5
cp work/t2_p2mix/export/t2_model_*.pt checkpoints/enhancer/p2mix/

# stage 6: ASR-oriented continuation of the p2mix members (log-mel weight 1.2), all three members in one job
T2FT_INIT=work/t2_p2mix/export T2FT_OUT=work/t2_p2mix_asr T2FT_MIN=40 T2FT_LR=1e-4 T2FT_WMEL=1.2 T2FT_REAL=0.4 \
    T2FT_SELECT=fit python train/t2_finetune_p2.py
cp work/t2_p2mix_asr/export/t2_model_*.pt checkpoints/enhancer/asrbt/

# sisdr5: the five distinct models of p2mix + asrbt (asrbt member 1 was identical to p2mix member 1)
for i in 0 1 2; do cp work/t2_p2mix/export/t2_model_$i.pt checkpoints/enhancer/sisdr5/t2_model_$i.pt; done
cp work/t2_p2mix_asr/export/t2_model_0.pt checkpoints/enhancer/sisdr5/t2_model_3.pt
cp work/t2_p2mix_asr/export/t2_model_2.pt checkpoints/enhancer/sisdr5/t2_model_4.pt
echo "ensembles -> checkpoints/enhancer/{p2mix,asrbt,sisdr5}/"
