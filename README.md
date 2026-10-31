<div align="center">

# LowBitMoss

**Low-bit MOSS-Image for consumer hardware**

[![GitHub code](https://img.shields.io/badge/GitHub-Code-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/MutantSparrow/LowBitMoss)
[![Models and data](https://img.shields.io/badge/🤗%20Hugging%20Face-Models%20%26%20Data-ffc107?style=flat-square)](https://huggingface.co/MutantSparrow/LowBitMoss)
[![Whitepaper](https://img.shields.io/badge/📄%20Paper-Whitepaper-b31b1b?style=flat-square)](https://github.com/MutantSparrow/LowBitMoss/blob/main/paper/whitepaper.pdf)
[![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](https://github.com/MutantSparrow/LowBitMoss/blob/main/LICENSE)

</div>

LowBitMoss explores block-wise quantization-aware distillation of the 5.08B-parameter MOSS-Image SFT diffusion transformer on one 16 GB GPU. Mixed-precision DiT files occupy **1.72-2.93 GB**, with low-rank corrections and a custom ComfyUI loader. This is exploratory research, with working configurations, negative results, and implementation failures documented in the paper.

The **7.18 GB small distilled package** combines a 2.14 GB DiT, a 4.93 GB pruned Qwen3.5-9B GPTQ text encoder, and a 0.11 GB decoder. That is model-file storage, not peak RAM or VRAM. It has automatic scores, but no complete-package judge pass. The proposed **7.97 GB quality package** remains unmeasured as that exact combination; its source DiT's A 4.27 and 23/24 acceptance were measured with the bf16 encoder and full VAE.

**Author's hands-on validation.** I have also manually benchmarked the models and reviewed their outputs, and I am satisfied with the results. The images have a softer rendering style, but the models retain a surprising amount of visual information and image quality. This is my qualitative assessment from hands-on use; the numerical results in the paper refer to the evaluation configurations described above.

## Where things live

| Repository | Contents |
|---|---|
| [GitHub](https://github.com/MutantSparrow/LowBitMoss) | Training and evaluation code, ComfyUI loader, format specification, abstract, whitepaper |
| [Hugging Face](https://huggingface.co/MutantSparrow/LowBitMoss) | Model files, calibration data, evaluation artifacts, abstract, whitepaper |

## Abstract

LowBitMoss explores how mixed-precision quantization can bring MOSS-Image SFT diffusion transformers to consumer hardware. We apply block-wise quantization-aware distillation on a single 16 GB GPU to the 5.08B-parameter DiT, combining learnable group scales, student-stream inputs, and low-rank corrections. Released DiT files occupy 1.72-2.93 GB, compared with 10.16 GB for the bf16 teacher. On 24 prompts at 768 px, the 2.14 GB mixG_tuned model achieves model-judge acceptance of 4.21/5 and letter-exact rendering of all six text prompts. The 1.85 GB and 1.72 GB variants broaden the available storage options.

For four-step distilled generation, the 2.93 GB folded-LoRA model reaches model-judge acceptance of 4.27/5, with 23 of 24 images accepted at 1024 px using the bf16 text encoder and full VAE. A 4.93 GB pruned GPTQ text encoder and a 0.11 GB VAE decoder bring the two recommended distilled component sets to 7.18 GB and 7.97 GB of model-file storage, compared with 29.81 GB originally. We document the training, packing, and loader choices and release code, model components, calibration captures, and evaluation artifacts under Apache 2.0, giving the community a practical starting point for further experiments with low-bit DiT models.

[Read the abstract](ABSTRACT.md) · [Whitepaper PDF](paper/whitepaper.pdf) · [Whitepaper HTML](paper/whitepaper.html) · [Whitepaper source](paper/whitepaper.md)

## Released model files

| Component | File | Size (decimal GB) | Notes |
|---|---|---:|---|
| Distilled small DiT | `distilled/tmixG_x.safetensors` | 2.14 | Four-step training; source-DiT A 4.06, 18/24 accepted with original encoder/VAE |
| Distilled quality candidate | `distilled/mixG_tuned_t.safetensors` | 2.93 | Folded turbo correction; source-DiT A 4.27, 23/24 with original encoder/VAE |
| Non-distilled, best text | `non_distilled/mixG_tuned.safetensors` | 2.14 | Six text prompts letter-exact with original components |
| Non-distilled, overall | `non_distilled/mixG_x.safetensors` | 2.14 | Source-DiT A 4.29 |
| Non-distilled, smaller | `non_distilled/mixH2_x_f8.safetensors` | 1.85 | Compact fp8 low-rank factors |
| Non-distilled, smallest | `non_distilled/mixH2_xm_f8.safetensors` | 1.72 | Slight cool, lower-contrast drift |
| Text encoder | `text_encoder/qwen3.5_9b_w4a8_gptq_pruned.safetensors` | 4.93 | Comfy-Org GPTQ, unused tensors removed; stock Load CLIP |
| Decoder only | `vae/flux2-small-decoder-vae.safetensors` | 0.11 | Text-to-image decoding only |
| Encoder + small decoder | `vae/flux2-vae-small-decoder.safetensors` | 0.25 | Keeps VAE Encode support |

## ComfyUI quick start

1. Use a ComfyUI version with native MOSS-Image support (the research used 0.43.0).
2. Copy [`custom_nodes/ComfyUI-MOSS-LowBit/`](https://github.com/MutantSparrow/LowBitMoss/tree/main/custom_nodes/ComfyUI-MOSS-LowBit) into `ComfyUI/custom_nodes/` and restart ComfyUI.
3. Download one DiT, the pruned text encoder, and a VAE from [Hugging Face](https://huggingface.co/MutantSparrow/LowBitMoss/tree/main). Place them in `models/diffusion_models/`, `models/text_encoders/`, and `models/vae/` respectively.
4. Connect **MOSS LowBit UNET Loader** to the sampler. Use stock **Load CLIP** with type `moss_image`, stock **Load VAE**, and the stock MOSS-Image empty-latent node (128 channels, 16x spatial downscale).

| Family | Steps | CFG | Sampler / scheduler | Resolution | Negative prompt |
|---|---:|---:|---|---|---|
| Distilled | 4 | 1.0 | `euler` / `simple` | 1024 x 1024; 768 also works | Empty |
| Non-distilled | 25 | 2.5 | `er_sde` / `simple` | 768 x 768 evaluated | `nsfw, worst quality, bad anatomy, watermark, jpeg artifact` |

**Six-step tip:** although trained at four steps, the very small distilled models also work well at six steps in the author's informal use, giving more contrast and a more polished look. The reported evaluation scores use four steps; this observation is not a scored comparison.

Use the custom UNET loader for all low-bit DiTs. The custom CLIP loader is only for the project's experimental encoder format; the released pruned GPTQ encoder uses the stock loader. Additional runtime LoRA patches on low-bit DiTs are refused; the folded model already includes its turbo correction.

## Research code

Install a CUDA-compatible PyTorch build and the dependencies of your ComfyUI checkout. The additional Python requirements are listed in `requirements.txt`. Set `COMFYUI_DIR` to that checkout; `LOWBITMOSS_HOME` optionally points to the directory containing `data/`, `out/`, and `eval/`.

```bash
git clone https://github.com/MutantSparrow/LowBitMoss.git
cd LowBitMoss
python -m pip install -r requirements.txt
export COMFYUI_DIR=/path/to/ComfyUI
```

Start with [METHOD](docs/METHOD.md), [TESTING](docs/TESTING.md), the [file-format specification](SPEC.md), and `scripts/example_pipeline.sh`. The scripts cover calibration capture, sensitivity/allocation, block-wise QAT, end-to-end tuning, packing, LoRA folding, encoder pruning, and evaluation. The paper records which settings produced each measured configuration.

The Hugging Face `data/` folder contains the captured calibration states and teacher outputs; `eval/` contains the saved evaluation images, latents, metric summaries, and judge reports. Prompt definitions are also in `prompts/` here. The serialized tensor data should be read with a current PyTorch release and `weights_only=True` where supported. See the [data guide](https://huggingface.co/MutantSparrow/LowBitMoss/blob/main/DATA.md) for exact capture/target pairings.

## Reading the results

- Evaluation uses 24 prompts and one seed per prompt; separate Claude visual-judge instances rated the models. Their approximately 0.2-point spread is informal, not a confidence interval.
- Local HPSv3-based scores have not been validated for numerical parity with official HPSv3. Paired intervals are exploratory; similar averages do not establish equivalence.
- Image distances measure likeness to a teacher image, not quality. Laplacian edge energy does not measure meaningful detail retained.
- The mixed 1-bit attention-and-FFN allocation fails. Larger-data runs also use more optimizer steps, so they do not isolate a data-only benefit.
- The distilled teacher is harder to quantize than the 25-step teacher in this setting. No peak-memory or wall-clock speedup is claimed.

See paper Tables I7-I9 for exact tested configurations, interval estimates, and common-preprocessing image statistics.

## License and acknowledgments

The code, whitepaper, project-authored prompt/evaluation data, and released model derivatives are distributed under **Apache License 2.0**. See [LICENSE](LICENSE), [NOTICE](NOTICE), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and modifications. Separately installed dependencies keep their own licenses; ComfyUI is not redistributed here.

Thanks to [OpenMOSS](https://huggingface.co/OpenMOSS-Team), [ComfyUI](https://huggingface.co/Comfy-Org), [Qwen](https://huggingface.co/Qwen), [akhaliq](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps), and [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder) for the models and tools that enabled these experiments. AI tools assisted the research, implementation, and drafting.
