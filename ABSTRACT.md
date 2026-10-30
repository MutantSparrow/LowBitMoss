# LowBitMoss: abstract

LowBitMoss explores how mixed-precision quantization can bring MOSS-Image SFT diffusion transformers to consumer hardware. We apply block-wise quantization-aware distillation on a single 16 GB GPU to the 5.08B-parameter DiT, combining learnable group scales, student-stream inputs, and low-rank corrections. Released DiT files occupy 1.72-2.93 GB, compared with 10.16 GB for the bf16 teacher. On 24 prompts at 768 px, the 2.14 GB mixG_tuned model achieves model-judge acceptance of 4.21/5 and letter-exact rendering of all six text prompts. The 1.85 GB and 1.72 GB variants broaden the available storage options.

For four-step distilled generation, the 2.93 GB folded-LoRA model reaches model-judge acceptance of 4.27/5, with 23 of 24 images accepted at 1024 px using the bf16 text encoder and full VAE. A 4.93 GB pruned GPTQ text encoder and a 0.11 GB VAE decoder bring the two recommended distilled component sets to 7.18 GB and 7.97 GB of model-file storage, compared with 29.81 GB originally. We document the training, packing, and loader choices and release code, model components, calibration captures, and evaluation artifacts under Apache 2.0, giving the community a practical starting point for further experiments with low-bit DiT models.

[Whitepaper](paper/whitepaper.pdf) · [Code](https://github.com/MutantSparrow/LowBitMoss) · [Models and data](https://huggingface.co/MutantSparrow/LowBitMoss)

Licensed under Apache License 2.0.
