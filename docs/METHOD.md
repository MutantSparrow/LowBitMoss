# Method and findings

Condensed notes of the research log behind this repository. All numbers are from one machine (RTX 4080 16 GB), one seed per prompt and a
24-prompt held-out evaluation set (`prompts/eval_prompts.json`), so differences of a few percent are noise. A longer write-up is planned.

## Goal and setting

Make the MOSS-Image SFT diffusion transformer (5.08B parameters, 42 blocks, bf16 file 10.2 GB) much smaller while keeping the images, including
rendered text and object counts, close to the bf16 teacher. Sampling is the official one: 25 steps, cfg 2.5, `er_sde`, `simple` scheduler,
768 px for evaluation. The teacher needs classifier-free guidance (two model calls per step), which matters below.

The PrismML Bonsai-Image whitepaper describes a *format* (group-wise symmetric binary/ternary weights with fp16 group scales, small sensitive layers
kept in 16 bit) but not how its models were trained. The format is borrowed (`SPEC.md`); the training recipe here is our own and is not a reproduction.

## Format

Group size 128 along the input dimension, symmetric codes, fp16 scales, six code layouts (1-bit, ternary in 2-bit slots, int2, int4, ternary
5-per-byte, int3). Per-layer kinds can be mixed in one file; an optional bf16 low-rank branch `lora_a`/`lora_b` (rank 32 by default, about 130 MB
for all linears) is folded into the dequantized weight (`W = Q + B A`). Non-block linears (text embedder, modulation, embedders, head) are
kept bf16 unless quantized by `quantize_extras.py`.

## Training pipeline

1. **Calibration data** (`capture.py`): run the real teacher sampler on 160 prompts (plus 80 text-rich prompts) at 512 px and record
   `(x_t, timestep, text context)` at a stratified subset of steps, both cond and uncond calls. The last 5% of files (at least 2) are held out and never trained on.
2. **Teacher targets** (`teacher_targets.py`): the bf16 model's output for every captured sample. These define the *output relative-MSE proxy*
   used everywhere (`e2e_tune.py --epochs 0` evaluates it; it also reports the CFG-combined velocity error).
3. **Sensitivity** (`sensitivity.py`): quantize ONE block linear at a time (optimal-scale rounding) and measure the full-model output error.
   The last 8 blocks (34-41) and block 0 carry about 76% of the damage; blocks 1-22 are robust. The sum of single-layer damages predicts the
   all-layer PTQ error well (predicted 0.041, measured 0.042 for int4). Non-block linears (`sensitivity_extra.py`): the text-embedder blocks
   are almost insensitive (int2 is fine), modulation layers are moderately sensitive.
4. **Bit allocation** (`alloc.py`, `make_alloc.py`): greedy per-linear choice among bin / int2 / int4 (optionally ternary) that starts at int4
   and takes the cheapest-damage-per-byte downgrade until a predicted-error budget is used. Files `scripts/alloc_mix*.json`:

   | file | what |
   |---|---|
   | mixA / mixB | greedy bin/int2/int4 at predicted error 0.15 / 0.10 |
   | mixC / mixD | more aggressive (0.30 / 0.50), mostly 1-bit |
   | mixE | like mixC with ternary (5-per-byte) instead of int2 |
   | mixG | attention of mixA, FFN int3 (int4 in blocks 0, 1, 40, 41) |
   | mixH | attention of mixA, FFN int2 in blocks 6-35, int3 in 2-5 and 36-39, int4 in 0, 1, 40, 41 |
5. **Block-wise distillation QAT** (`train_blocks.py`, `qlayer.py`): for block i, the student block has fake-quantized linears (fp32 latent weights
   + learnable group scales, straight-through rounding), takes the activation produced by the already-quantized blocks < i, and is trained to match
   the bf16 teacher block applied to the teacher's own activation. Loss options: `token` (relative MSE per token), `global`, `std`
   (error relative to each channel's spread over tokens, used for all reported models). A rank-32 low-rank correction per quantized linear is initialised
   in closed form (data-aware SVD of the quantization residual with the input covariance as metric) and trained too: on block 34 ternary it cut the
   pre-training error 0.52 -> 0.09 and the final stream error 2.3x.
6. **Extras** (`quantize_extras.py`): plain optimal-scale rounding of the text-embedder linears (int2, ffn_down int4, proj int4). Free in quality, 2.54 -> 2.14 GB for mixG.
7. **End-to-end tuning** (`e2e_tune.py`): codes frozen; the group scales and the LoRA branches are trained so the full model's output matches the
   teacher on the captured trajectories. Loss per (cond, uncond) pair = rel-MSE(cond) + rel-MSE(uncond) + rel-MSE(CFG-combined velocity), weighted by
   `1 + tw * (1 - t)` so low-noise steps (fine detail) count more. Cut the holdout error by 24% on mixA.

## Results

Output relative-MSE proxy on held-out trajectories (lower is better):

| model | size | proxy | images |
|---|---|---|---|
| naive int4 rounding | 3.1 GB | 0.042 | good |
| uniform ternary, block-wise QAT, no LoRA (v1 / v2) | 1.92 GB | 0.286 / 0.136 | no structure |
| uniform binary v1 / naive 1-bit | 1.32 GB | 0.536 / 2.05 | unusable |
| mixA (bin/int2/int4 + rank-32 LoRA) | 2.07 GB | 0.032 | coherent, text glyphs degraded |

Image-level evaluation (24 held-out prompts, 768 px; teacher CLIP score 36.17):

| model | size | CLIP | DINOv2 cos | SSIM | judge F / T / C / Q / accept |
|---|---|---|---|---|---|
| mixA_lora32 | 2.07 GB (1.67 with text embedder quantized) | 35.66 | 0.81 | 0.58 | 3.50 / 1.40 / 2.14 / 3.81 / 2.90 |
| mixF_ptq (mixA attention + FFN int4 by plain rounding) | 2.42 GB | 36.07 | 0.80 | 0.44 | 4.29 / 4.36 / 2.71 / 4.29 / 4.02 |
| **mixG_x** (QAT with FFN int3) | **2.14 GB** | **36.31** | **0.884** | **0.656** | **4.50 / 4.43 / 3.36 / 4.44 / 4.29** |

The judge is an independent LLM instance that opened all 24 images and scored 0-5 on faithfulness (F), text correctness (T), counts/spatial
relations (C), quality (Q) and "would a user accept it instead of the teacher" (A). Scores come from different judge instances, so +-0.2 is
noise; the per-image side-by-side verdicts are firmer: mixG_x vs mixF_ptq is 15 better, 9 same, 0 worse. Remaining mixG_x defects: a six-object count
(fruit bowl shows 4 fruits instead of 6), the infographic layout (labels in a column instead of a cycle) and small text glitches (a dropped colon,
one extra garbled sign). Contact sheets: `docs/images/`.

## Key finding: MSE sensitivity is not text sensitivity

The first coherent model (mixA) matched the teacher on CLIP score (98.6%) and had a good output-error proxy, but an independent judge found all 6 text prompts
and both counting prompts wrong (overall acceptance 2.9/5). So CLIP score and output MSE do not see text and count errors, and "satisfactory"
has to include them.

* **Ablation** (`make_hybrid.py`, 9 critical prompts: 6 text, typewriter keys, 2 counting; `docs/images/sheet_ablation.jpg`): restoring groups of the
  tuned mixA to bf16. Restoring the FFN (3.17B of the 4.76B block parameters) fixed text, Chinese glyphs, prices, typewriter keys and counts nearly
  completely. Restoring attention made it *worse* (muddy), because FFN, LoRA and scales had co-adapted to the quantized attention. Restoring blocks
  0-13, 14-27, 28-41 or the text embedder alone did not fix text. The text/count failures come from the low-bit FFN (most middle-block FFNs of mixA were at 1 bit),
  which the output-MSE sensitivity had rated as harmless.
* **FFN precision study** (`docs/images/sheet_ffn_study.jpg`): FFN requantized by plain rounding on top of the tuned mixA. int4: text, Chinese, prices and counts
  essentially as good as bf16 (this is mixF_ptq, 2.42 GB). int2: total collapse (pure noise). Uniform int3 by plain rounding: clearly degraded
  (colour cast, "WAER CYCLE"); int4 at the ends plus int3 in the middle: better but still oversaturated with text damage.
* **QAT at 3 bits beats plain 4-bit rounding** (mixG_x, `docs/images/sheet_mixG.jpg`): block-wise QAT with FFN int3 (+LoRA, text-rich calibration prompts) is
  smaller than mixF_ptq and better on every judged category.
* A text-prompt held-out version of the proxy (`e2e_tune.py --holdout_text`) did not track text quality either: the tuned model had the lowest MSE but worse
  text than the hybrids. Do not pick allocations by output MSE alone; judge by images.

## Why the base model is hard

Bonsai's base (FLUX.2 Klein 4B) is step-distilled (4 steps, cfg 1): no CFG amplification and far fewer steps to accumulate error. MOSS-Image SFT needs
25 steps and cfg 2.5; we measured a 2.4x error amplification of the CFG-combined velocity. This is a hypothesis for why
the same bit-widths do not transfer, not a tested claim.

## Loader bug worth knowing

`F.linear(x, QuantizedTensor)` decomposes into transpose/mm and never reaches a registered `aten.linear` handler (the generic fallback dequantizes instead).
A LoRA branch hooked into that handler was silently skipped (loader proxy 0.17 vs 0.03 on the training side); an early "garbage" mixA result was this bug,
not the file. Fix: the low-rank term is folded into the layout's `dequantize()`. `test_loader_proxy.py` compares the loader path with the training-side
proxy on the same file; always run it for a new file.

## Operational lessons

* Never trust a loader/inference path without checking it against the training-side proxy on the same file (`test_loader_proxy.py`).
* Never run two GPU jobs with big VRAM on a 16 GB card: they thrash (a 23 s step became 1500 s). `gen_inproc.py` has a watchdog that restarts a stalled image.
* `python -I` ignores the script directory; scripts add their own directory to `sys.path` (and read `LOWBITMOSS_HOME` / `COMFYUI_DIR` from the environment).

## Reproducing the ablations

```
cd scripts        # BASE = a tuned moss_lowbit_v1 file in $LOWBITMOSS_HOME/out
python -I make_hybrid.py --inp $BASE --out out/hyb_ffn.safetensors --restore ffn                 # FFN back to bf16
python -I make_hybrid.py --inp $BASE --out out/hyb_b00_13.safetensors --restore blocks:0-13      # every linear of blocks 0-13 back to bf16
python -I make_hybrid.py --inp $BASE --out out/v_ffn_int4.safetensors --restore ffn --ptq int4   # FFN requantized from the teacher by plain rounding
python -I gen_inproc.py --unet out/hyb_ffn.safetensors --tag hyb_ffn --ids 0,1,2,3,4,5,10,14,15  # the 9 critical prompts
python -I sheet.py --dirs eval/teacher_bf16 eval/hyb_ffn eval/v_ffn_int4 --thumb 220 --rows 9 --out eval/sheet_ffn_study.png
```
