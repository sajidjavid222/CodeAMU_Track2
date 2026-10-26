#!/usr/bin/env python3
"""CodeAMU - IndoML 2026 Track 2 (noise event removal), Phase 2 inference.

    python inference/infer.py --input_dir <test audio dir> --output_dir <out dir> \
        [--t1_predictions inputs/phase2/track1_predictions.jsonl] [--route inputs/phase2/route.json] \
        [--checkpoints checkpoints] [--enhancer p2mix|asrbt|sisdr5] [--val_dir <official validation dir>] [--work_dir work]

Enhancer ensembles (checkpoints/enhancer/<name>/t2_model_*.pt; each reproduces one Phase 2 submission exactly):
  asrbt  : p2mix after 40 more minutes with log-mel weight 1.2 (stage 6)            -> Codabench 987258, 26.4894 (FINAL)
  p2mix  : the 3 Phase 2-profile members with real organiser pairs (stage 5)        -> Codabench 985438, 26.4363
  sisdr5 : the 5 distinct models of p2mix + asrbt (asrbt member 1 = p2mix member 1) -> Codabench 987250, 26.4582

Pipeline:
  1. refnoise matching    : clips routed synthetic are matched against the validation reference noise (inference/refnoise.py)
  2. refnoise subtraction : every match with cor >= 0.45 is subtracted with a fitted gain (inference/refnoise_subtract.py)
  3. enhancement + ASR: the TF-DPCRN ensemble (checkpoints/enhancer/<name>/) enhances every clip conditioned on
     the Track 1 event windows; refnoise-subtracted clips get an enhancer pass over their subtracted spans + Track 1
     windows (T2_REFNOISE_POST=1); the mandated ASR (ARTPARK-IISc/SraVaani-1.0) transcribes the enhanced audio
     (inference/enhance_and_transcribe.py). Output: one 16 kHz PCM_16 WAV per clip + transcripts.jsonl +
     submission.zip.
HF_TOKEN must be set (SraVaani-1.0 is a gated Hugging Face repo).
"""
import argparse
import json
import os
import subprocess
import sys

ENHANCERS = ("asrbt", "p2mix", "sisdr5")
FINAL_ENHANCER = "asrbt"            # the ranked final submission (Codabench ID 987258, Combined 26.4894)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True, help="folder of noisy 16 kHz WAV clips")
    ap.add_argument("--output_dir", required=True, help="enhanced WAVs, transcripts.jsonl and submission.zip go here")
    ap.add_argument("--t1_predictions", default=f"{ROOT}/inputs/phase2/track1_predictions.jsonl",
                    help="Track 1 noise-event timestamps (predictions.jsonl from the CodeAMU Track 1 system)")
    ap.add_argument("--route", default=f"{ROOT}/inputs/phase2/route.json",
                    help="{clip_id: true|false} synthetic routing from the Track 1 router")
    ap.add_argument("--checkpoints", default=f"{ROOT}/checkpoints", help="reference_noise.npz + enhancer/<name>/")
    ap.add_argument("--enhancer", default=FINAL_ENHANCER, choices=ENHANCERS, help="enhancer ensemble (see above)")
    ap.add_argument("--val_dir", default=None,
                    help="official validation folder; if given the refnoise is rebuilt from it, else checkpoints/reference_noise.npz")
    ap.add_argument("--work_dir", default=f"{ROOT}/work")
    args = ap.parse_args()
    assert os.environ.get("HF_TOKEN"), "set HF_TOKEN (the mandated ASR ARTPARK-IISc/SraVaani-1.0 is gated)"
    os.makedirs(args.work_dir, exist_ok=True)

    import refnoise
    import refnoise_subtract
    if args.val_dir:
        templates8, meta = refnoise.build_templates_from_validation(args.val_dir)
        templates16 = [refnoise.template16_from_validation(args.val_dir, m) for m in meta]
    else:
        templates8, meta, templates16 = refnoise.load_templates_npz(f"{args.checkpoints}/reference_noise.npz")
    route = json.load(open(args.route))
    syn = sorted(c for c, v in route.items() if v)
    print(f"[1/3] refnoise matching: {len(syn)} synthetic-routed clips", flush=True)
    matches = refnoise.match_clips(args.input_dir, syn, templates8, f"{args.work_dir}/refnoise_matches.json")
    print("[2/3] refnoise subtraction", flush=True)
    sub_dir = f"{args.work_dir}/refnoise_sub"
    if not os.path.exists(f"{sub_dir}/subtracted_spans.json"):
        refnoise_subtract.run(args.input_dir, matches, templates16, sub_dir)
    print("[3/3] enhancement + mandated ASR", flush=True)
    env = dict(os.environ, T2_TEST=args.input_dir, T2_T1=args.t1_predictions,
               T2_CKPT=f"{args.checkpoints}/enhancer/{args.enhancer}",
               T2_OUT=args.output_dir, T2_BLEND="1.0,0.90", T2_PAD="0.10", T2_HARD="1.0", T2_WHOLE="0",
               T2_REFNOISE_SUB=sub_dir, T2_REFNOISE_POST="1", T2_REFNOISE_AOUT="0.0")
    env.pop("T2_MUTE", None); env.pop("T2_HYBRID", None)
    subprocess.run([sys.executable, f"{HERE}/enhance_and_transcribe.py"], env=env, check=True)


if __name__ == "__main__":
    main()
