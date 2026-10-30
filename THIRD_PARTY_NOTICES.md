# Third-party provenance and modifications

| Component | Upstream source | License | LowBitMoss modifications |
|---|---|---|---|
| MOSS-Image SFT DiT | https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT | Apache-2.0 | Mixed-precision quantization-aware distillation, low-rank corrections, compact packing and optional modulation quantization |
| ComfyUI-format model packaging | https://huggingface.co/Comfy-Org/MOSS-Image | Apache-2.0 | Reuses upstream packaged components; GPTQ text encoder pruned of unused tensors |
| Qwen3.5-9B | https://huggingface.co/Qwen/Qwen3.5-9B | Apache-2.0 | Released text encoder is the Comfy-Org W4A8 GPTQ representation with unused tensors removed |
| Four-step MOSS adapter | https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps | Apache-2.0 | Applied to the teacher before QAT, or folded as stored low-rank factors into the quantized model |
| FLUX.2 small decoder | https://huggingface.co/black-forest-labs/FLUX.2-small-decoder | Apache-2.0 | Converted/repacked for ComfyUI; decoder-only variant omits encoder tensors; full variant retains encoder support |

The original model weights belong to their respective upstream contributors. This release does not claim ownership of upstream contributions or endorsement by their authors. Model names identify their provenance. The licenses and any available upstream notice files are included under `licenses/`.

The included original LowBitMoss code, paper, prompts, and project-generated research artifacts are released under Apache-2.0. Evaluation images and tensor captures were generated during the experiments; model provenance is listed above.

ComfyUI is a separately installed dependency licensed under GPL-3.0; its checkout is not included or relicensed by this project. Other separately installed libraries retain their upstream licenses.
