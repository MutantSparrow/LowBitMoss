# LowBitMoss

Low-bit (binary / 2 / 3 / 4-bit mixed) quantization of the **MOSS-Image SFT** diffusion transformer with **block-wise distillation QAT**:
training and evaluation code, the file format, and a ComfyUI loader node.

**Status: research in progress.** The code works end to end and produced a 2.14 GB model (teacher: 10.2 GB) whose images are close to the bf16
teacher on most prompts, but it is not a finished or released quantization, no model files are included, and the numbers below come from one
machine, 24 evaluation prompts and one seed per prompt.

## Results so far

Evaluation: 24 held-out prompts (`prompts/eval_prompts.json`), 768 px, 25 steps, cfg 2.5, `er_sde`; teacher CLIP score 36.17. "Judge" is an independent
LLM instance that opened every image and scored 0-5 (F faithfulness, T text, C counts/spatial, Q quality, A would-accept); treat +-0.2 as noise.

| model | size | CLIP | DINOv2 cos | judge F / T / C / Q / A |
|---|---|---|---|---|
| bf16 teacher | 10.2 GB | 36.17 | 1 | |
| naive int4 rounding | 3.1 GB | | | good images, no judge run |
| uniform ternary / binary (block-wise QAT, no LoRA) | 1.9 / 1.3 GB | 22.5 (ternary) | 0.12 | unusable, no structure |
| mixA: bin/int2/int4 by sensitivity + rank-32 LoRA | 2.07 GB | 35.66 | 0.81 | 3.50 / 1.40 / 2.14 / 3.81 / 2.90 |
| mixF_ptq: mixA attention + FFN int4 by plain rounding | 2.42 GB | 36.07 | 0.80 | 4.29 / 4.36 / 2.71 / 4.29 / 4.02 |
| **mixG_x**: low-bit attention + QAT-trained int3 FFN (int4 in blocks 0, 1, 40, 41), text embedder quantized | **2.14 GB** | **36.31** | **0.884** | **4.50 / 4.43 / 3.36 / 4.44 / 4.29** |

mixG_x versus mixF_ptq, image by image: 15 better, 9 same, 0 worse. Remaining defects: exact counts of six objects (a fruit bowl shows 4 fruits
instead of 6), infographic layout logic, and small text glitches (a dropped colon, one extra garbled sign). Further experiments (end-to-end tuning
of mixG, a more aggressive 2-bit-FFN allocation "mixH") were running when this repository was prepared and are not evaluated here.

**The key problem was text and counting.** The first coherent low-bit model (mixA) matched the teacher on CLIP score (98.6%) and had a good
output-error proxy, yet all 6 text prompts and both counting prompts were wrong. Neither CLIP score nor output MSE sees those failures. An ablation
traced them to the low-bit FFN (the middle blocks' FFN at 1 bit), which the per-layer MSE sensitivity had rated harmless; the FFN needs about
3-4 bits, while attention tolerates 1-2 bits. Details in [docs/METHOD.md](docs/METHOD.md).

| ablation (restore groups to bf16) | FFN precision study | mixG vs predecessors |
|---|---|---|
| [![ablation](docs/images/sheet_ablation.jpg)](docs/images/sheet_ablation.jpg) | [![ffn](docs/images/sheet_ffn_study.jpg)](docs/images/sheet_ffn_study.jpg) | [![mixG](docs/images/sheet_mixG.jpg)](docs/images/sheet_mixG.jpg) |

Columns of the last sheet: teacher, tuned mixA, mixF_ptq, mixG_x (9 critical prompts: 6 with text, typewriter keys, 2 with counts).

## Pipeline

Run from `scripts/` with the ComfyUI venv Python (`$PY`, always `-I`). `scripts/example_pipeline.sh` runs all of this for the mixG recipe.
Work files go to `$LOWBITMOSS_HOME/{data,out,eval,logs}` (default: the repository root, git-ignored).

1. Capture teacher trajectories as calibration data (160 prompts + 80 text-rich prompts, 512 px):
   `$PY -I capture.py --res 512 --n 160 --keep 8` and
   `$PY -I capture.py --res 512 --n 80 --keep 8 --prompts ../prompts/calib_text_prompts.json --prefix p-t`
2. Teacher outputs for every captured sample (the reference of the error proxy): `$PY -I teacher_targets.py`
3. Per-layer sensitivity: `$PY -I sensitivity.py --kind tern` (and `$PY -I sensitivity_extra.py --kinds int4,int2` for the non-block linears)
4. Bit allocation (the shipped `scripts/alloc_mix*.json` are the ones used for the results; this re-derives them from your sensitivity map): `$PY -I make_alloc.py`
5. Block-wise distillation QAT with LoRA:
   `$PY -I train_blocks.py --kind @alloc_mixG.json --loss std --lora_rank 32 --n_samples 1536 --holdout 48 --epochs 2 --lr_w 1e-4 --lr_s 2e-3 --out ../out/mixG_lora32.safetensors --log ../logs/mixG_lora32.json`
6. Quantize the text embedder (no training): `$PY -I quantize_extras.py --inp ../out/mixG_lora32.safetensors --out ../out/mixG_x.safetensors`
7. Generate evaluation images (teacher once, then the student) and compare:
   `$PY -I gen_inproc.py --bf16 --unet <ComfyUI>/models/diffusion_models/moss_image_sft_bf16.safetensors --tag teacher_bf16`,
   `$PY -I gen_inproc.py --unet ../out/mixG_x.safetensors --tag mixG_x`,
   `$PY -I metrics.py --teacher ../eval/teacher_bf16 --student ../eval/mixG_x --prompts eval`
8. End-to-end tuning of group scales and LoRA against the teacher outputs:
   `$PY -I e2e_tune.py --inp ../out/mixG_x.safetensors --out ../out/mixG_tuned.safetensors --epochs 0.7 --mb 2 --lr 2e-4 --lora_lr 4e-5 --tw 2 --cfg_w 1`
9. Contact sheet: `$PY -I sheet.py --dirs ../eval/teacher_bf16 ../eval/mixG_x --thumb 256 --rows 6 --out ../eval/sheet.png`
10. Check the ComfyUI loader path against the training-side proxy: `$PY -I test_loader_proxy.py ../out/mixG_tuned.safetensors 24`

Ablation helper (`make_hybrid.py`): restore selected linears of a file to bf16 or requantize them by plain rounding, to find which layers carry which quality.
Server-based generation (`gen_eval.py`, same workflow through a running ComfyUI) and a header dump (`inventory.py <file>`) are also included.

## Hardware used

NVIDIA RTX 4080 (16 GB), 128 GB system RAM, Windows 11. Block-wise QAT of all 42 blocks took about 72 minutes, end-to-end tuning about an hour (700 steps),
a 24-image evaluation (about 30 s per image) roughly 12 minutes. Data and checkpoints need about 15 GB of disk.

## Requirements

* A ComfyUI checkout at v0.42.5 (developed against it; the CPU tests also pass on v0.43.0) with its venv: torch 2.12.1+cu130 and the packages ComfyUI needs
  (including `comfy-kitchen`, `safetensors`, `numpy`, `pillow`). Nothing is pip-installed by this repository.
* `transformers` and `scipy` for `metrics.py` (CLIP and DINOv2 are downloaded from the Hugging Face hub on first use).
* The MOSS-Image files in that ComfyUI: `models/diffusion_models/moss_image_sft_bf16.safetensors` (teacher), `models/text_encoders/qwen3.5_9b_bf16.safetensors`, `models/vae/flux2-vae.safetensors`.
* Environment: `COMFYUI_DIR` (required; the folder containing `main.py`) and optionally `LOWBITMOSS_HOME` (work directory, default the repository root).
  Under Git Bash on Windows give paths in the `C:/...` form.

Scripts never modify ComfyUI; `-I` (isolated mode) is used everywhere, so each script adds its own directory to `sys.path`.

## Using a quantized model in ComfyUI

The stock `UNETLoader` cannot read these files. Copy (or symlink) `custom_nodes/ComfyUI-MOSS-LowBit` into `ComfyUI/custom_nodes/`, put the `.safetensors` file in
`ComfyUI/models/diffusion_models/`, restart ComfyUI and use the node **MOSS LowBit UNET Loader** (category `model/loaders`) in place of the `UNETLoader` of the official
MOSS-Image template (25 steps, cfg 2.5, `er_sde`, `simple`, text encoder type `moss_image`, `flux2-vae`). The loader keeps the packed codes resident and dequantizes each
linear to bf16 inside the call. The node uses ComfyUI's `QuantizedTensor` layout API and may need updating when that API changes.

## File format (moss_lowbit_v1)

One `.safetensors` file with the original tensor names. Every quantized linear `X` has `X.weight` (uint8, packed codes), `X.weight_scale` (fp16, one scale per 128 inputs),
`X.weight_kind` (1 bin, 2 tern, 3 int2, 4 int4, 5 tern5, 6 int3) and optionally a bf16 low-rank correction `X.lora_a` / `X.lora_b` (`W = Q + B A`). Everything else stays bf16.
Kinds can be mixed per layer; mixG_x is 2.14 GB. Full specification: [SPEC.md](SPEC.md).

## Repository layout

```
scripts/            training, evaluation and test code (_paths.py reads LOWBITMOSS_HOME / COMFYUI_DIR), alloc_mix*.json, example_pipeline.sh
custom_nodes/       ComfyUI-MOSS-LowBit loader node
prompts/            calibration prompts (192 + 80 text-rich) and the 24 evaluation prompts
docs/               METHOD.md (method and findings), TESTING.md, images/ (contact sheets)
SPEC.md             file format and loader requirements
```

Tests: see [docs/TESTING.md](docs/TESTING.md) (three CPU-only scripts pass; the rest need a GPU or the teacher).

## Known limitations

* Not a finished quantization: counts of larger numbers of objects, infographic logic and some small text still fail; the evaluation is small (24 prompts, one seed each, LLM judges).
* No fused low-bit kernels: the loader dequantizes per linear call, so the gain is memory, not speed (speed was not benchmarked here).
* Only the attention tolerates 1-2 bits; the FFN needs about 3-4 bits (2-bit FFN collapses with plain rounding; with QAT it is untested here).
* Calibration at 512 px, evaluation at 768 px; other resolutions are not evaluated.
* Developed and run on Windows; shell and path handling use `os.path.join` and should work on Linux, but that has not been tried. `example_pipeline.sh` was assembled from the run scripts and only syntax-checked.
* Depends on ComfyUI internals (`comfy.quant_ops`, `comfy.ldm.moss_image`), developed against v0.42.5.

## Credits

* The **file format** follows the PrismML Bonsai-Image whitepaper (group-wise low-bit weights with fp16 group scales, small sensitive layers in 16 bit). It is cited as a format
  reference only: the whitepaper does not disclose its training recipe, so this project's block-wise distillation QAT is not a reproduction of Bonsai.
* [ComfyUI](https://github.com/Comfy-Org/ComfyUI) (model code, loaders, quantized-tensor layout API) and **MOSS-Image** by OpenMOSS-Team (the model being quantized; Qwen3.5 text encoder and FLUX.2 VAE are used as shipped for it).
* Metrics use OpenAI CLIP ViT-B/16 and Meta DINOv2-small through `transformers`.
* This work was done with assistance from Claude Code (Anthropic).

## License

**NO LICENSE CHOSEN YET.** The loader node builds on ComfyUI internals (GPL-3.0), so pick a compatible license before making the repository public.
