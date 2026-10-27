# Testing

All tests are plain scripts (no pytest). Run them with the ComfyUI venv's Python in isolated mode from the `scripts/` folder, with `COMFYUI_DIR` set:

```
export COMFYUI_DIR=C:/path/to/ComfyUI
cd scripts
$COMFYUI_DIR/.venv/Scripts/python.exe -I -B test_int3.py
```

## CPU-only (no GPU, no model files)

They hide all CUDA devices (`CUDA_VISIBLE_DEVICES=-1`, `--cpu`), import `comfy` from `COMFYUI_DIR` and exit non-zero on failure. `-B` keeps Python from writing bytecode into the ComfyUI tree.

| script | what it checks | time |
|---|---|---|
| `test_tern5.py` | pack/unpack round trips of all kinds against independent definitions, kind 5 vs kind 2 bit-identity, loader layout and `Linear` loading (with/without bias and LoRA, wrong packed widths rejected), `QLinear.export`, `e2e_tune.PackedLinear` | ~7 s |
| `test_int3.py` | kind 6 (int3): all 2**24 three-byte words round trip, dequantize/quantize references, loader layout, `Linear` loading and shape validation, mixed-kind safetensors round trip, qlayer fake-quant/STE/export. Section 7 reads one tensor of the bf16 teacher if `COMFYUI_DIR/models/diffusion_models/moss_image_sft_bf16.safetensors` exists (skipped otherwise) | ~30 s |
| `test_lora_layout.py` | `F.linear(x, QuantizedTensor)` equals `x (Q + B A)^T` for kinds 1-4 with and without the low-rank branch; params keep `lora_a/lora_b` through `to_device`/`clone` | ~13 s |

Last run (CPU only): all three pass, against a ComfyUI checkout at tag v0.43.0 with torch 2.12.1+cu130 (the code was developed on v0.42.5).

`python -I -B -c "import py_compile, glob; [py_compile.compile(f, cfile=None, doraise=True) for f in glob.glob('scripts/*.py')]"` is a syntax check of everything
(it also writes `__pycache__`, which is git-ignored).

## Need the GPU and/or large files (not run in CI-style checks)

| script | needs |
|---|---|
| `test_lowbit_loader.py <file> [blocks]` | GPU; a moss_lowbit_v1 file. Runs the packed model and a bf16 model built from the dequantized weights and compares outputs, memory and ms/forward (`blocks` keeps only the first N blocks so the reference fits next to other GPU users) |
| `test_loader_proxy.py <file> [n]` | GPU; `data/calib_512` and `data/teacher_out_512.pt` (steps 1-2 of the pipeline). Output relative MSE vs the teacher through the ComfyUI loader path; compare with `e2e_tune.py --inp <file> --epochs 0` |
| `make_rtn_test.py int4\|bin\|mixed` | GPU and the 10 GB bf16 teacher. Writes a naive round-to-nearest moss_lowbit_v1 file for loader tests (quality irrelevant) |

The training/evaluation scripts (`capture.py`, `teacher_targets.py`, `sensitivity*.py`, `train_blocks.py`, `e2e_tune.py`, `gen_inproc.py`, `make_hybrid.py --ptq`) need the GPU, the
teacher, the text encoder and the VAE; `metrics.py` needs `transformers` and downloads `openai/clip-vit-base-patch16` and `facebook/dinov2-small` from the Hugging Face hub on first use;
`gen_eval.py` needs a running ComfyUI server. `example_pipeline.sh` has only been syntax-checked (`bash -n`) in its consolidated form; its steps are the commands of the runs that produced the reported results.
`make_alloc.py` is CPU-only given `logs/sens_tern.json`; it reproduces all shipped `alloc_mix*.json` byte for byte from the original sensitivity map.
