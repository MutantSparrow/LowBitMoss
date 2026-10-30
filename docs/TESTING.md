# Testing

Tests are standalone Python scripts. Use a Python environment with the dependencies in `requirements.txt` and make the ComfyUI checkout available through `COMFYUI_DIR`, as described in the repository README.

```sh
cd scripts
python -I -B test_tern5.py
python -I -B test_int3.py
python -I -B test_lora_layout.py
python -I -B test_lora_fp8.py
```

## Numerical format tests

These tests run on the CPU and do not require the full model files. Optional real-weight checks are skipped when their inputs are absent.

| Script | Coverage |
|---|---|
| `test_tern5.py` | Packing and unpacking against independent definitions; base-3 ternary identity; loader shapes, bias, low-rank factors, and export |
| `test_int3.py` | Exhaustive three-byte packing round trips; quantization/dequantization references; mixed-format serialization; fake-quant and export behavior |
| `test_lora_layout.py` | Linear outputs against the dense `Q + B A` reference and preservation of factor parameters through device and clone operations |
| `test_lora_fp8.py` | Factor scales, e4m3 rounding bounds, dense-reference reconstruction, malformed-input rejection, and serialization round trips |

`scripts/orig/` holds the frozen pre-fp8 implementation used as an independent compatibility reference by `test_lora_fp8.py`.

## Model and data checks

| Script | Required inputs and purpose |
|---|---|
| `test_lowbit_loader.py <file> [blocks]` | GPU and a `moss_lowbit_v1` file; compares the packed loader with a dense model reconstructed from the same weights |
| `test_loader_proxy.py <file> [n]` | GPU, calibration captures, and their matching teacher outputs; measures held-out output relative MSE through the inference loader |
| `test_distill_lib.py` | Source bf16 teacher and four-step LoRA; checks correction shapes, relative magnitude, and bf16 merge behavior |
| `check_euler.py` | Distilled captures and matching teacher outputs; verifies the sampler update convention |
| `hps3_checks.py` | HPSv3 checkpoint and configuration; checks determinism and response to deliberately degraded inputs |
| `cond_error.py` | Saved conditioning tensors; compares encoder output error against the stated reference |
| `png_psnr.py`, `check_pngs.py` | Saved evaluation PNGs; paired pixel comparisons and image integrity |

Download the calibration and evaluation artifacts from [Hugging Face](https://huggingface.co/MutantSparrow/LowBitMoss). Follow the [data guide](https://huggingface.co/MutantSparrow/LowBitMoss/blob/main/DATA.md) when selecting capture/target pairs. The expanded 240-prompt non-distilled pool belongs with `teacher_out_512_plus.pt`; the legacy `teacher_out_512.pt` belongs to the original 160-prompt pool. Check that a script's configured inputs match the pair being used before running a loader-proxy comparison.

The loader-proxy test, teacher-target generator, end-to-end tuner, and example pipeline default to the expanded pair. `test_loader_proxy.py` accepts `--data` and `--targets` for another matching pool and rejects unequal capture/target counts before loading the GPU model. `LOWBITMOSS_TARGETS` selects the pipeline's target file; use `--out` for the generator and `--targets` for the tuner when selecting another pool:

```sh
python -I -B test_loader_proxy.py MODEL.safetensors 24 --data CAPTURES --targets MATCHING_TARGETS.pt
python -I -B e2e_tune.py --inp MODEL.safetensors --epochs 0 --data CAPTURES --targets MATCHING_TARGETS.pt
```

Generation, training, and encoder checks require the source model components and a GPU. `metrics.py` uses CLIP and DINOv2; `hps3_score.py` additionally needs the HPSv3 checkpoint and its configuration. Numerical parity between this HPS-based scorer and the official implementation remains unvalidated: its plausibility checks do not establish benchmark equivalence.

## Interpreting validation

The paper records the experiment settings and per-configuration evidence. Publishing checks cover Python syntax, file integrity, document links, and metadata; they do not rerun the full GPU training and evaluation suite. The portable code and example pipeline should be validated in the environment used for a new experiment. A reproduced score must identify its prompts, seeds, sampler, resolution, DiT, text encoder, and VAE.

See [METHOD.md](METHOD.md), [SPEC.md](../SPEC.md), and the whitepaper's Section 5 and Appendix C for the experimental protocol and numerical definitions.
