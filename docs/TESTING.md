# Testing

All tests are plain scripts (no pytest). Run them with the ComfyUI venv's Python in isolated mode from the `scripts/` folder, with `COMFYUI_DIR` set:

```
export COMFYUI_DIR=/path/to/ComfyUI
cd scripts
$COMFYUI_DIR/.venv/Scripts/python.exe -I -B test_int3.py
```

## CPU-only (no GPU, no model files)

They hide all CUDA devices (`CUDA_VISIBLE_DEVICES=-1`, `--cpu`), import `comfy` from `COMFYUI_DIR` and exit non-zero on failure. `-B` keeps Python from writing bytecode into the ComfyUI tree.

| script | what it checks | time |
|---|---|---|
| `test_tern5.py` | pack/unpack round trips of all kinds against independent definitions, kind 5 vs kind 2 bit-identity, loader layout and `Linear` loading (with/without bias and LoRA, wrong packed widths rejected), `QLinear.export`, `e2e_tune.PackedLinear` | ~7 s |
| `test_int3.py` | kind 6 (int3): all 2**24 three-byte words round trip, dequantize/quantize references, loader layout, `Linear` loading and shape validation, mixed-kind safetensors round trip, qlayer fake-quant/STE/export. Section 7 reads one tensor of the bf16 teacher if `COMFYUI_DIR/models/diffusion_models/moss_image_sft_bf16.safetensors` exists (skipped otherwise) | ~30 s |
| `test_lora_layout.py` | `F.linear(x, QuantizedTensor)` equals `x (Q + B A)^T` for kinds 1-4 with and without the low-rank branch; params keep `lora_a/lora_b` through `to_device`/`clone` | ~5 s |
| `test_lora_fp8.py` | compact LoRA (float8_e4m3fn factors + scalar fp16 scales): `lora_to_fp8` / `lora_from_stored` (range, no NaN, scale rounded up, per-element e4m3 bound), the loader layout for kinds 1-6 against a dense reference, ComfyUI's `Params` machinery (`to_device`, `clone`, `empty_like` + `copy_`, gathered buffers, `state_dict_tensors`), strict `Linear.load_state_dict` incl. 15 rejected malformed combinations, safetensors round trip, `compact_lora.py` on a small mixed file (other tensors bit-identical, idempotent, bad input refused), the `e2e_tune.PackedLinear` call site. bf16-LoRA files and kinds 1-6 are compared bit for bit with the loader and `lowbit_pack` from before the fp8 format, kept in `scripts/orig/`. Section 7 compares a real `mixH2_x.safetensors` / `mixH2_x_fp8lora.safetensors` pair if both are in `$LOWBITMOSS_HOME/out` (skipped otherwise) | ~30 s |

Last run (CPU only, GPU hidden): all four pass, against a ComfyUI checkout at tag v0.43.0 with torch 2.12.1+cu130 (the code was developed on v0.42.5).
`scripts/orig/` is not a test: it is the frozen pre-fp8 `lowbit_pack.py` and loader `lowbit.py` that `test_lora_fp8.py` uses as its reference; do not edit it.

`python -I -B -c "import py_compile, glob; [py_compile.compile(f, cfile=None, doraise=True) for f in glob.glob('scripts/*.py')]"` is a syntax check of everything
(it also writes `__pycache__`, which is git-ignored).

## Need the GPU and/or large files (not run in CI-style checks)

| script | needs |
|---|---|
| `test_lowbit_loader.py <file> [blocks]` | GPU; a moss_lowbit_v1 file. Runs the packed model and a bf16 model built from the dequantized weights and compares outputs, memory and ms/forward (`blocks` keeps only the first N blocks so the reference fits next to other GPU users) |
| `test_loader_proxy.py <file> [n]` | GPU; `data/calib_512` and `data/teacher_out_512.pt` (steps 1-2 of the pipeline). Output relative MSE vs the teacher through the ComfyUI loader path; compare with `e2e_tune.py --inp <file> --epochs 0` |
| `make_rtn_test.py int4\|bin\|mixed` | GPU and the 10 GB bf16 teacher. Writes a naive round-to-nearest moss_lowbit_v1 file for loader tests (quality irrelevant) |
| `hps3_score.py --tags a,b [--examples]` | GPU, the 16.6 GB HPSv3 checkpoint, the Qwen2-VL-7B-Instruct config/tokenizer files and a clone of the upstream HPSv3 repo under `$LOWBITMOSS_AUX` (default `$LOWBITMOSS_HOME/models_aux`, see the README), 24 images per tag in `eval/<tag>/`. Not run in CI-style checks |
| `hps3_checks.py` | same as `hps3_score.py`. Plausibility checks of the scorer (determinism, wrong prompt, blurred / grey / noise images, sequential vs 3D mrope position ids); there are no reference values to compare with |
| `test_distill_lib.py` | CPU only, but needs the 4-step LoRA in `$LOWBITMOSS_AUX/turbo4` (1.19 GB) and ComfyUI's bf16 teacher. For a few linears of blocks 0, 20 and 41: delta shapes, the size of the delta relative to the weight, how much of the LoRA delta survives the round-to-nearest bf16 merge (projection, cosine, share of changed weights); the quantities quoted in the README's "The LoRA delta is mostly below bf16 rounding". Not run in this repository form |

`hps3_score.py` loads the model with a layer-offload device map: `infer_auto_device_map` with `max_memory={0: "11GiB", "cpu": "90GiB"}`, set by `--gpu_gib` (default 11), keeps about 11 GiB of the model on the GPU and the rest of the decoder layers in pinned CPU RAM,
streamed to the GPU by accelerate during each forward pass (the log line `device_map: ... decoder layers on cpu: N` shows the split). Lower `--gpu_gib` when another process uses the GPU; run one GPU job at a time (`--wait_gpu` waits until under 2000 MiB are in use).
Only `hps3_score.summarize` and `category_winners.py` run on the CPU. Run against the original work-directory files (the saved `eval/<tag>/hps3.json` and `metrics.json`, not part of this repository), `summarize` reproduced `eval/hps3_summary.json` and `.md`
and `category_winners.py` reproduced `eval/category_winners.md` byte for byte, apart from the one note line in `hps3_score.py` that now gives the actual input size (420x420 for a 768 px square image). In this repository form the model loading and the scoring of images of `hps3_score.py` / `hps3_checks.py` have not been run, only syntax-checked.

The training/evaluation scripts (`capture.py`, `teacher_targets.py`, `sensitivity*.py`, `train_blocks.py`, `e2e_tune.py`, `gen_inproc.py`, `make_hybrid.py --ptq`) need the GPU, the
teacher, the text encoder and the VAE; `metrics.py` needs `transformers` and downloads `openai/clip-vit-base-patch16` and `facebook/dinov2-small` from the Hugging Face hub on first use;
`gen_eval.py` needs a running ComfyUI server. `example_pipeline.sh` has only been syntax-checked (`bash -n`) in its consolidated form; its steps are the commands of the runs that produced the reported results.
`make_alloc.py` is CPU-only given `logs/sens_tern.json`; it reproduces all eight shipped `alloc_mix*.json` (mixA-E, G, H, I) byte for byte from the original sensitivity map.
`compact_lora.py` and `size_report.py` are CPU-only too (`compact_lora.py` is exercised by `test_lora_fp8.py`; run it on a real file to see the saving and the per-factor error).
`gen_inproc.py` (supervisor, per-image watchdog, planned worker recycling) needs the GPU and was only syntax-checked in this repository form.
The distilled-model scripts (README, "Distilled (4-step) models"): `make_turbo_teacher.py` and `add_turbo_factors.py` are CPU-only but need the 4-step LoRA in `$LOWBITMOSS_AUX/turbo4` (1.19 GB) and the bf16 teacher (`add_turbo_factors.py` prints a fold check of
a few random linears against `Q + B A + dW`); `check_euler.py` is CPU-only and needs the captures and `data/teacher_out_turbo768.pt`; `capture_turbo.py` needs the GPU and the merged teacher. `distill_lib.py` is their helper.
They are in this repository in the path convention of `_paths.py`; only syntax-checked in this form, the work-directory originals produced the reported results.
The text-encoder scripts (README, "Text encoder"): `quantize_encoder.py` needs the GPU, the bf16 encoder (19.3 GB) and, for `--embed int8`, ComfyUI's int8 encoder; `gen_inproc.py --clip-lowbit` / `--extra-seeds` need the GPU, a DiT file and the VAE; `strip_id.py` is CPU-only (Pillow, images in `eval/<tag>/`). The MOSS LowBit CLIP Loader node (`custom_nodes/ComfyUI-MOSS-LowBit/__init__.py`) needs a running ComfyUI. All of them are only syntax-checked (`py_compile`) in this repository form, as is `test_distill_lib.py`; the work-directory originals produced the reported results.
The scripts of the recommended package (README, "Text encoder" and "VAE"): `prune_encoder.py` (CPU; needs Comfy-Org's `qwen3.5_9b_w4a8_gptq.safetensors`, 7.34 GB, and writes 4.93 GB), `encode_prompts.py` (GPU; an encoder file, no DiT or VAE) and `cond_error.py` (CPU; two or more `.pt` files from `encode_prompts.py`), `make_small_vae.py` (CPU; needs `flux2-vae.safetensors` and the downloaded `flux2-small-decoder-only.safetensors` in `models/vae`), `vae_compare.py` (GPU; saved latents in `eval/<tag>/latents` and the VAE files in `models/vae`), `png_psnr.py` and `check_pngs.py` (CPU; Pillow, safetensors; folders of eval PNGs). `check_euler.py` is listed above. All of them were only syntax-checked (`py_compile`) in this repository form; the work-directory originals produced the numbers in the README. The conditioning errors, the prune check (identical conditioning and latents) and the VAE comparison were therefore not re-run from the repository copies.
The chart scripts (README, "Recommended package"): `likeness_plot.py` (CPU, `transformers` + `scipy`; DINOv2-small on the 24 images of the eval tags it lists in `$LOWBITMOSS_HOME/eval`, `eval/hps3_*.json` and `logs/cond_error.log`) writes `docs/charts/*.csv`, and `likeness_charts.py` (CPU, `matplotlib`) draws the four PNGs from those CSVs.
Both were only syntax-checked (`py_compile`) in this repository form: the committed CSVs and PNGs in `docs/charts/` are the work-directory outputs of the originals, and regenerating them needs the original eval folders.
