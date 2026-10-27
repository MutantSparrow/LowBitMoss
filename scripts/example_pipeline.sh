#!/bin/bash
# Example end-to-end pipeline: the "mixG" recipe (low-bit attention + 3-bit FFN with block-wise QAT, rank-32 LoRA, text embedder quantized,
# then end-to-end tuning) from the bf16 teacher to evaluated images. The result is a 2.1 GB moss_lowbit_v1 file.
#
#   export COMFYUI_DIR=C:/path/to/ComfyUI          # ComfyUI v0.42.x checkout with its .venv; Windows paths in the C:/... form under Git Bash
#   export LOWBITMOSS_HOME=/path/to/workdir        # optional, default: the repository root (data/ out/ eval/ logs/ are created there, ~15 GB)
#   bash scripts/example_pipeline.sh
#
# Needs in COMFYUI_DIR/models: diffusion_models/moss_image_sft_bf16.safetensors, text_encoders/qwen3.5_9b_bf16.safetensors, vae/flux2-vae.safetensors.
# Everything runs on one GPU (developed on an RTX 4080 16 GB, 128 GB RAM): run one GPU job at a time, they thrash each other when VRAM is shared.
# Every step writes logs/<step>.log. Steps 1, 2 and 7 skip work that already exists, train_blocks.py can resume with --resume out/<name>.safetensors.partial.pt.
set -euo pipefail

: "${COMFYUI_DIR:?set COMFYUI_DIR to your ComfyUI checkout}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export LOWBITMOSS_HOME="${LOWBITMOSS_HOME:-$(dirname "$HERE")}"
if [ -z "${PY:-}" ]; then
  if [ -x "$COMFYUI_DIR/.venv/Scripts/python.exe" ]; then PY="$COMFYUI_DIR/.venv/Scripts/python.exe"; else PY="$COMFYUI_DIR/.venv/bin/python"; fi
fi
DATA="$LOWBITMOSS_HOME/data"; OUT="$LOWBITMOSS_HOME/out"; EVAL="$LOWBITMOSS_HOME/eval"; LOGS="$LOWBITMOSS_HOME/logs"
mkdir -p "$DATA" "$OUT" "$EVAL" "$LOGS"
cd "$HERE"

run() {  # run <log name> <command...>: output goes to logs/<log name>.log
  local log="$1"; shift
  echo "== $log"
  "$@" > "$LOGS/$log.log" 2>&1
}

# 1. Calibration data: the teacher's own sampling trajectories (25 steps, cfg 2.5, er_sde, 512 px) on 160 prompts plus 80 text-rich prompts
#    (files p-t*, which sort first so the held-out files, the last ones, stay ordinary prompts). Both cond and uncond calls of 10 steps per prompt are kept.
run capture_calib   "$PY" -I capture.py --res 512 --n 160 --keep 8
run capture_text    "$PY" -I capture.py --res 512 --n 80 --keep 8 --prompts ../prompts/calib_text_prompts.json --prefix p-t

# 2. Teacher outputs for every captured sample -> data/teacher_out_512.pt (the reference of the output-error proxy and of end-to-end tuning)
[ -f "$DATA/teacher_out_512.pt" ] || run teacher_targets "$PY" -I teacher_targets.py

# 3. Sensitivity of every block linear to ternary rounding -> logs/sens_tern.json (only needed to derive new allocations, see 4)
#    and, optionally, of the non-block linears -> logs/sens_extra.json (the policy in quantize_extras.py comes from it)
# run sensitivity       "$PY" -I sensitivity.py --kind tern
# run sensitivity_extra "$PY" -I sensitivity_extra.py --kinds int4,int2

# 4. Bit allocation: scripts/alloc_mix*.json are the ones used for the reported results (mixG = attention of mixA, FFN int3, int4 in blocks 0,1,40,41).
#    To re-derive them from your own sensitivity map (overwrites the shipped files): "$PY" -I make_alloc.py

# 5. Block-wise distillation QAT of every block with the allocation, rank-32 LoRA (data-aware SVD init) on every quantized linear, ~70 min
run train_mixG "$PY" -I train_blocks.py --kind @alloc_mixG.json --loss std --lora_rank 32 --n_samples 1536 --holdout 48 --epochs 2 \
    --lr_w 1e-4 --lr_s 2e-3 --out "$OUT/mixG_lora32.safetensors" --log "$LOGS/mixG_lora32.json"

# 6. Post-training quantization of the text embedder (no training): 2.5 GB -> 2.1 GB
run extras_mixG "$PY" -I quantize_extras.py --inp "$OUT/mixG_lora32.safetensors" --out "$OUT/mixG_x.safetensors"

# 7. Teacher reference images (24 held-out prompts, 768 px) once, then the student images and the metrics against them
[ -d "$EVAL/teacher_bf16" ] || run gen_teacher "$PY" -I gen_inproc.py --bf16 --unet "$COMFYUI_DIR/models/diffusion_models/moss_image_sft_bf16.safetensors" --tag teacher_bf16
run gen_mixG_x     "$PY" -I gen_inproc.py --unet "$OUT/mixG_x.safetensors" --tag mixG_x
run metrics_mixG_x "$PY" -I metrics.py --teacher "$EVAL/teacher_bf16" --student "$EVAL/mixG_x" --prompts eval

# 8. End-to-end tuning of the group scales and LoRA branches against the teacher outputs (codes frozen), ~1 h
run tune_mixG "$PY" -I e2e_tune.py --inp "$OUT/mixG_x.safetensors" --out "$OUT/mixG_tuned.safetensors" \
    --epochs 0.7 --mb 2 --lr 2e-4 --lora_lr 4e-5 --tw 2 --cfg_w 1

# 9. Evaluate the tuned model and make a contact sheet (rows = prompts, columns = teacher, mixG_x, mixG_tuned)
run gen_mixG_tuned     "$PY" -I gen_inproc.py --unet "$OUT/mixG_tuned.safetensors" --tag mixG_tuned
run metrics_mixG_tuned "$PY" -I metrics.py --teacher "$EVAL/teacher_bf16" --student "$EVAL/mixG_tuned" --prompts eval
run sheet_mixG         "$PY" -I sheet.py --dirs "$EVAL/teacher_bf16" "$EVAL/mixG_x" "$EVAL/mixG_tuned" --thumb 256 --rows 6 --out "$EVAL/sheet_mixG.png"

# 10. Check that the ComfyUI loader path reproduces the training-side output error (always do this for a new file), then install the model:
#     copy out/mixG_tuned.safetensors to COMFYUI_DIR/models/diffusion_models/ and use the "MOSS LowBit UNET Loader" node.
run loader_proxy "$PY" -I test_loader_proxy.py "$OUT/mixG_tuned.safetensors" 24
echo "done: $OUT/mixG_tuned.safetensors"
