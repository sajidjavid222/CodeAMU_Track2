#!/usr/bin/env bash
# CodeAMU - Track 2 Phase 2 inference (reproduces submission_t2_phase2_bankpost_p2mix.zip, Combined 26.4363).
#   HF_TOKEN=hf_xxx bash scripts/run_inference.sh <input_audio_dir> <output_dir>
# Optional: T1=<Track 1 predictions.jsonl> ROUTE=<route.json> VAL_DIR=<official validation folder (rebuilds the bank)>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IN="${1:?input audio dir}"; OUT="${2:?output dir}"
: "${HF_TOKEN:?set HF_TOKEN (the mandated ASR ARTPARK-IISc/SraVaani-1.0 is gated)}"
ARGS=(--input_dir "$IN" --output_dir "$OUT" --work_dir "${WORK:-$OUT/_work}"
      --t1_predictions "${T1:-$ROOT/inputs/phase2/track1_predictions.jsonl}" --route "${ROUTE:-$ROOT/inputs/phase2/route.json}"
      --checkpoints "$ROOT/checkpoints")
[ -n "${ENHANCER:-}" ] && ARGS+=(--enhancer "$ENHANCER")     # p2mix | asrbt | sisdr5 (default: the ranked final)
[ -n "${VAL_DIR:-}" ] && ARGS+=(--val_dir "$VAL_DIR")
python "$ROOT/inference/infer.py" "${ARGS[@]}"
echo "submission: $OUT/submission.zip"
