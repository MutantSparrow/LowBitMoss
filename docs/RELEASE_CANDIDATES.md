# Model release guide

LowBitMoss explores block-wise quantization-aware distillation of the 5.08B-parameter MOSS-Image SFT diffusion transformer on one 16 GB GPU. Mixed-precision DiT files occupy **1.72-2.93 GB**, with low-rank corrections and a custom ComfyUI loader. This is exploratory research, with working configurations, negative results, and implementation failures documented in the paper.

The **7.18 GB small distilled package** combines a 2.14 GB DiT, a 4.93 GB pruned Qwen3.5-9B GPTQ text encoder, and a 0.11 GB decoder. That is model-file storage, not peak RAM or VRAM. It has automatic scores and an informal web-interface test, but no complete-package judge pass. The proposed **7.97 GB quality package** remains unmeasured as that exact combination; its source DiT's A 4.27 and 23/24 acceptance were measured with the bf16 encoder and full VAE.

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

## Reading the results

- Evaluation uses 24 prompts and one seed per prompt; separate Claude visual-judge instances rated the models. Their approximately 0.2-point spread is informal, not a confidence interval.
- Local HPSv3-based scores have not been validated for numerical parity with official HPSv3. Paired intervals are exploratory; similar averages do not establish equivalence.
- Image distances measure likeness to a teacher image, not quality. Laplacian edge energy does not measure meaningful detail retained.
- The mixed 1-bit attention-and-FFN allocation fails. Larger-data runs also use more optimizer steps, so they do not isolate a data-only benefit.
- The distilled teacher is harder to quantize than the 25-step teacher in this setting. No peak-memory or wall-clock speedup is claimed.

See paper Tables I7-I9 for exact tested configurations, interval estimates, and common-preprocessing image statistics.
