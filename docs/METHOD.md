# Method and findings

Condensed notes of the research log behind this repository. All numbers are from one machine (RTX 4080 16 GB), one seed per prompt and a
24-prompt held-out evaluation set (`prompts/eval_prompts.json`, 768 px), so differences of a few percent are noise. The numbers come from the project's `NOTES.md`,
the judge reports `eval/judge_*.md`, the per-model `eval/<model>/metrics.json`, `eval/hps3_summary.md` and `eval/category_winners.md`; those are work-directory files and are not part of this repository.
A longer write-up is planned.

## Goal and setting

Make the MOSS-Image SFT diffusion transformer (5.08B parameters, 42 blocks, bf16 file 10.2 GB) much smaller while keeping the images, including
rendered text and object counts, close to the bf16 teacher. Sampling is the official one: 25 steps, cfg 2.5, `er_sde`, `simple` scheduler,
768 px for evaluation. The teacher needs classifier-free guidance (two model calls per step), which matters below.

All sizes are decimal GB (10^9 bytes). Windows Explorer shows binary GiB (2^30 bytes), which look about 7% smaller (the 1.85 GB model appears as 1.72 in Explorer, the 1.72 GB model as 1.61, the 10.16 GB bf16 teacher as 9.47).

The PrismML Bonsai-Image whitepaper describes a *format* (group-wise symmetric binary/ternary weights with fp16 group scales, small sensitive layers
kept in 16 bit) but not how its models were trained. The format is borrowed (`SPEC.md`); the training recipe here is our own and is not a reproduction.

## Format

Group size 128 along the input dimension, symmetric codes, fp16 scales, six code layouts (1-bit, ternary in 2-bit slots, int2, int4, ternary
5-per-byte, int3). Per-layer kinds can be mixed in one file; an optional low-rank branch `lora_a`/`lora_b` (rank 32, or 64 on the bin/int2 linears; 35-113 MB in the first models, 204 MB in
mixH2_x) is folded into the dequantized weight (`W = Q + B A`). The branch is bf16, or float8_e4m3fn with one fp16 scale per factor (half the bytes: 204 -> 102 MB on mixH2_x,
see "Compact LoRA" in `SPEC.md`). Non-block linears (text embedder, modulation, embedders, head) are kept bf16 unless quantized by `quantize_extras.py`.

## Training pipeline

1. **Calibration data** (`capture.py`): run the real teacher sampler on 160 prompts (plus 80 text-rich prompts) at 512 px and record
   `(x_t, timestep, text context)` at a stratified subset of steps, both cond and uncond calls. The last 5% of files (at least 2) are held out and never trained on.
2. **Teacher targets** (`teacher_targets.py`): the bf16 model's output for every captured sample. These define the *output relative-MSE proxy*
   used everywhere (`e2e_tune.py --epochs 0` evaluates it; it also reports the CFG-combined velocity error).
3. **Sensitivity** (`sensitivity.py`): quantize ONE block linear at a time (optimal-scale rounding) and measure the full-model output error.
   The last 8 blocks (34-41) and block 0 carry about 80% of the damage (79.6%); blocks 1-22 are robust. The sum of single-layer damages predicts the
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
   | mixI | mixH with 1-bit qkv/o in blocks 6-35 and 1-bit ffn_down in blocks 12-29 (not acceptable, see below) |

   `make_alloc.py` reproduces all eight files byte for byte from the original sensitivity map.
5. **Block-wise distillation QAT** (`train_blocks.py`, `qlayer.py`): for block i, the student block has fake-quantized linears (fp32 latent weights
   + learnable group scales, straight-through rounding), takes the activation produced by the already-quantized blocks < i, and is trained to match
   the bf16 teacher block applied to the teacher's own activation. Loss options: `token` (relative MSE per token), `global`, `std`
   (error relative to each channel's spread over tokens, used for all reported models). A rank-32 low-rank correction per quantized linear (`--lora_rank_low 64`: rank 64 on the
   bin/int2 linears, used by the mixH family) is initialised in closed form (data-aware SVD of the quantization residual with the input covariance as metric) and trained too: on
   block 34 ternary it cut the pre-training error 0.52 -> 0.09 and the final stream error 2.3x. `--resume` continues from the partial file of an interrupted run.
6. **Extras** (`quantize_extras.py`): plain optimal-scale rounding of the text-embedder linears (int2, ffn_down int4, proj int4). Free in quality, 2.54 -> 2.14 GB for mixG.
   `--mod` / `--time` additionally round `shared_block_modulation`, `video_head_modulation` and `time_embedding.2` to int4 (see "Cheap savings" below: not free).
7. **End-to-end tuning** (`e2e_tune.py`): codes frozen; the group scales and the LoRA branches are trained so the full model's output matches the
   teacher on the captured trajectories. Loss per (cond, uncond) pair = rel-MSE(cond) + rel-MSE(uncond) + rel-MSE(CFG-combined velocity), weighted by
   `1 + tw * (1 - t)` so low-noise steps (fine detail) count more. Cut the holdout error by 24% on mixA. The tuning set of mixG_tuned and later includes the text-rich prompts.
   It reads bf16 or fp8 LoRA and writes bf16.
8. **Compact LoRA** (`compact_lora.py`): rewrites the LoRA factors as float8_e4m3fn plus a scalar fp16 scale each. Run it last.

## Results

Output relative-MSE proxy on held-out trajectories (lower is better):

| model | size | proxy | images |
|---|---|---|---|
| naive int4 rounding | 3.1 GB | 0.042 | good |
| uniform ternary, block-wise QAT, no LoRA (v1 / v2) | 1.92 GB | 0.286 / 0.136 | no structure |
| uniform binary v1 / naive 1-bit | 1.32 GB | 0.536 / 2.05 | unusable |
| mixA (bin/int2/int4 + rank-32 LoRA) | 2.07 GB | 0.032 | coherent, text glyphs degraded |

Image-level evaluation (24 held-out prompts, 768 px; teacher CLIP score 36.17; "text ok / exact" = of the 6 quoted-text prompts, how many the judge accepted (A >= 4) / how many are letter-for-letter exact):

| model | size | CLIP | DINOv2 cos | SSIM | judge F / T / C / Q / A | text ok / exact |
|---|---|---|---|---|---|---|
| mixA_lora32 | 2.07 GB (1.67 with text embedder quantized) | 35.66 | 0.81 | 0.58 | 3.50 / 1.40 / 2.14 / 3.81 / 2.90 | 0/6, - |
| mixF_ptq (mixA attention + FFN int4 by plain rounding) | 2.42 GB | 36.07 | 0.80 | 0.44 | 4.29 / 4.36 / 2.71 / 4.29 / 4.02 | 5/6, 3/6 |
| mixG_x (QAT with FFN int3) | 2.14 GB | 36.31 | 0.884 | 0.656 | 4.50 / 4.43 / 3.36 / 4.44 / 4.29 | 5/6, 4/6 |
| mixG_tuned (mixG_x + end-to-end tuning) | 2.14 GB | 36.35 | 0.901 | 0.666 | 4.48 / 4.79 / 3.64 / 4.23 / 4.21 | 5/6, 6/6 |
| mixH_x (FFN int2 in blocks 6-35, rank-64 LoRA; not judged) | 1.95 GB | 36.64 | 0.867 | 0.627 | | |
| mixH_tuned (mixH_x + tuning) | 1.95 GB | 36.11 | 0.860 | 0.615 | 4.31 / 4.00 / 3.29 / 4.29 / 4.10 | 4/6, 3/6 |
| mixH2_x (mixH allocation, ~2x calibration data, no tuning) | 1.95 GB | 36.33 | 0.877 | 0.640 | 4.33 / 4.00 / 3.43 / 4.27 / 4.21 | 4/6, 3/6 |
| mixH2_tuned (mixH2_x + tuning) | 1.95 GB | 36.26 | 0.864 | 0.622 | 4.29 / 3.93 / 3.36 / 4.29 / 4.17 | 4/6, 2/6 |
| mixH2_x_f8 (mixH2_x, fp8 LoRA) | 1.85 GB | 36.39 | 0.877 | 0.641 | pending | pending |
| mixH2_xm (mixH2_x + int4 modulation layers; not judged) | | 36.68 | 0.856 | 0.594 | | |
| mixH2_xm_f8 (+ fp8 LoRA) | 1.72 GB | 36.61 | 0.851 | 0.594 | 4.25 / 4.00 / 3.79 / 4.21 / 4.15 | 5/6, 4/6 |
| mixI_x (mixH + 1-bit qkv/o in blocks 6-35 and 1-bit ffn_down in blocks 12-29; not judged) | 1.74 GB | 35.60 | 0.745 | 0.541 | | |
| mixI_tuned (mixI_x + tuning) | 1.74 GB | 35.93 | 0.788 | 0.554 | 3.54 / 1.14 / 2.07 / 3.54 / 3.15 | 0/6, 0/6 |

The judge is a Claude model instance that opened all 24 images and scored 0-5 on faithfulness (F), text correctness (T), counts/spatial
relations (C), quality (Q) and "would a user accept it instead of the teacher" (A). Scores come from different judge instances (one per model), so +-0.2 is
noise; the per-image side-by-side verdicts are firmer: mixG_x vs mixF_ptq is 15 better, 9 same, 0 worse. `mixH2_x_f8` has metrics but no judge run yet
(pending). Remaining mixG_x defects: a six-object count (fruit bowl shows 4 fruits instead of 6), the infographic layout (labels in a column instead of a cycle)
and small text glitches (a dropped colon, one extra garbled sign). Contact sheets: `docs/images/`.

## Going below 2 GB

Per-image verdicts of the judge (better / same / worse):

| comparison | better / same / worse |
|---|---|
| mixG_x vs mixF_ptq | 15 / 9 / 0 |
| mixG_tuned vs mixG_x | 7 / 14 / 3 |
| mixH_tuned vs mixG_tuned | 5 / 12 / 7 |
| mixH2_x vs mixH_tuned | 9 / 13 / 2 |
| mixH2_x vs mixG_tuned | 4 / 17 / 3 |
| mixH2_tuned vs mixH2_x | 8 / 12 / 4 |
| mixH2_tuned vs mixG_tuned | 3 / 16 / 5 |
| mixH2_xm_f8 vs mixH2_x | 4 / 15 / 5 |
| mixI_tuned vs mixH_tuned | 0 / 4 / 20 |

* **mixG (2.14 GB)**: attention as in mixA, QAT-trained int3 FFN (int4 in blocks 0, 1, 40, 41). Smaller than mixF_ptq and better on every judged category. The end-to-end round (mixG_tuned) cut the held-out
  CFG-combined error 0.0323 -> 0.0277 and made the text exact (6/6 letter-for-letter, balloons exactly 5, fruit bowl 5 of 6) at the price of slightly softer close-up portraits
  (7 better, 14 same, 3 worse than mixG_x).
* **mixH (1.95 GB)**: FFN at 2 bits in blocks 6-35 (int3 in 2-5 and 36-39, int4 in 0, 1, 40, 41) and a rank-64 LoRA on the bin/int2 linears. Before tuning it was only eyeballed (shop sign, poster title, Chinese sign
  with sub-line and prices correct, diagram labels and one phone word slightly garbled): a *trained* 2-bit FFN with a rank-64 LoRA keeps the text where plain 2-bit rounding collapsed to noise. After tuning (mixH_tuned) the judge
  found 4/6 text prompts acceptable and 3/6 exact (mixG_tuned: 5/6 and 6/6): failures were small text on busy graphics and a price digit. Judged by criterion, the smallest acceptable model was 1.95 GB for general images and 2.14 GB for exact text and counts.
* **mixH2 (1.95 GB): more data, not more bits.** The same allocation trained with ~2x the calibration data (3000 instead of 1536 (prompt, step) samples, 2 epochs, no end-to-end tuning) reaches would-accept 4.21, equal to mixG_tuned and above
  mixH_tuned (4.10), with 20 of 24 images at A >= 4; text 4/6 acceptable, 3/6 exact, counts 3.43. An end-to-end tuning round on top (mixH2_tuned) did not help: the averages are inside the noise band (A 4.17), portraits improved, but
  letter-exact text went from 3/6 to 2/6 (the errors moved to other strings rather than disappeared) and the text-heavy tuning data did not buy better small-text spelling.
* **mixI (1.74 GB): 1-bit attention in the middle is past the cliff.** mixI = mixH with qkv/o at 1 bit in blocks 6-35 and ffn_down at 1 bit in blocks 12-29. Both mixI_x and mixI_tuned are far from the 1.95 GB models on every metric
  (DINOv2 0.745 / 0.788 against 0.860 for mixH_tuned) and the judge rated mixI_tuned not acceptable: T 1.14, 0/6 text prompts ("FPRONAGE", "WATER CCCLLE", 吉五面馆), C 2.07, A 3.15, 20 of 24 images worse than mixH_tuned.
  Natural scenes, watercolour and still life stay fine; structured objects, text, counts and anatomy drift. At this training budget 2 bits is the floor for attention and FFN, even with a rank-64 LoRA and end-to-end tuning.

## Cheap savings and what they cost

* **Size of mixH2_x by tensor group** (`size_report.py`): block codes 1420 MB (72.8%), LoRA 204 MB (10.4%), `shared_block_modulation` bf16 113 MB (5.8%), text-embedder codes 75 MB (3.9%), group scales 74 MB (3.8%),
  `video_head_modulation` 38 MB, `time_embedding.2` 19 MB (the three modulation tensors together are 170 MB, about 8.7%).
* **fp8 LoRA factors** (`compact_lora.py`): LoRA 203.7 -> 101.8 MB (file 1.951 -> 1.850 GB). Relative error of a factor 1.7-2.7%, of `B A` 3.7%, of the whole weight `Q + B A` 0.1-0.5%; the weight error against the teacher changes by <= 2e-5.
  Image metrics mixH2_x -> mixH2_x_f8: CLIP 36.33 -> 36.39, DINOv2 0.877 -> 0.877, SSIM 0.640 -> 0.641, latent cosine 0.720 -> 0.720. No measurable effect; the judge run on mixH2_x_f8 is pending.
* **int4 modulation layers** (`quantize_extras.py --mod --time`, expected ~125 MB) cost more fidelity than predicted from the sensitivity map, which had rated the modulation layers only moderately sensitive:
  DINOv2 0.877 -> 0.856, SSIM 0.640 -> 0.594, latent cosine 0.720 -> 0.659 (mixH2_x -> mixH2_xm). The judge on mixH2_xm_f8 (1.72 GB) still finds it acceptable: F 4.25, T 4.00, C 3.79, Q 4.21, A 4.15 against 4.33 / 4.00 / 3.43 / 4.27 / 4.21
  for mixH2_x (all inside the noise band except C, whose +0.36 comes from three image-specific changes), 21 of 24 images with A >= 4, 5/6 text prompts acceptable. What it measured is a small consistent drift: about 0.025 cooler (R-B)
  and about 8% less luminance contrast than the teacher on average, twice the loss of mixH2_x, scene dependent (grey-green typewriter keys, muted lanterns, paler bright scenes); no banding, blocking or noise.

## Aesthetic score (HPSv3) and category winners

CLIP score, DINOv2 and SSIM compare with the teacher or the prompt, none of them measures how pleasing an image is, so the 24 evaluation images of each model are also scored with HPSv3 (MizzenAI/HPSv3, MIT).
The results are in the README ("Aesthetic score (HPSv3)", "Category winners"); this is how they are made.

* **The scorer** (`hps3_score.py`, own minimal inference code, not the upstream package). The checkpoint is Qwen2-VL-7B-Instruct plus a new `<|Reward|>` token and a reward head (3584-1024-16-2). The score is the head's output pair
  `[mu, log_sigma]` at the `<|Reward|>` position of the last hidden state (after the final norm); the tables use mu only. The image is resized to a fixed pixel budget (256 x 28 x 28), the text is the upstream
  evaluation instruction with the prompt filled in, parsed with `ast` from the upstream collator source (never imported), and, like the upstream code, the language model gets plain sequential position ids instead of the 3D mrope index.
  Files and download: README, "Aesthetic score (HPSv3)"; they live under `$LOWBITMOSS_AUX` (default `$LOWBITMOSS_HOME/models_aux`).
* **Layer offload.** The 16.6 GB checkpoint (15.45 GiB) leaves no headroom for activations on a 16 GB card. `infer_auto_device_map` with `max_memory={0: "11GiB", "cpu": "90GiB"}` (`--gpu_gib`, default 11) assigns whole `Qwen2VLDecoderLayer` /
  `Qwen2VLVisionBlock` modules: about 11 GiB of the model on the GPU, the remaining decoder layers in pinned CPU RAM, which accelerate's `dispatch_model` streams to the GPU during each forward pass. The checkpoint is read tensor by tensor from the
  safetensors file and each tensor goes straight to its target device, so it is never held in RAM as a whole. Images are scored one at a time and cached per image in `eval/<tag>/hps3.json`; run it while no other job holds the GPU (`--wait_gpu` polls `nvidia-smi` until under 2000 MiB are in use).
* **Statistics** (`summarize` in `hps3_score.py`): mean mu over the 24 prompts with a bootstrap 95% interval over prompts, and the paired difference to the teacher (mean of the per-prompt differences, its SE, number of prompts won).
  The sd of mu across prompts is 1.34-2.03 and the bootstrap half-width of a mean 0.52-0.79. The paired SE between two candidate models (teacher and tern_v1 excluded) is 0.06-0.34, median 0.23, so a difference below about 0.46 mu (twice the median) is within
  prompt-sampling noise; with ONE seed per prompt, seed-to-seed variation is not measured at all.
* **Validation.** The upstream repository ships no reference scores, and its requirements pin `transformers` 4.45.2 while this implementation ran on 5.14.1, so the absolute values were not checked against the reference implementation. `hps3_checks.py` runs plausibility checks
  only: determinism, wrong prompts and degraded images (blurred, grey, noise) scoring lower, and the sequential position ids against the 3D mrope ids (the two give different scores; the sequential ids of the reference code are used).
* **Category winners** (`category_winners.py`, CPU only): the rules are in the script. Best overall = highest judge would-accept A (ties: within 0.2); best text = highest judge T; best aesthetics = highest HPSv3 mean mu (ties: within 0.5 mu); closest to the teacher = highest DINOv2 cosine;
  smallest acceptable = smallest file with A >= 4.0; best value = smallest file with A >= 4.0 and an HPSv3 difference to the teacher of at least -0.5. The judge scores are a table typed into the script from the judge reports (a Claude model instance per model, +-0.2 noise per average);
  `mixH2_x_f8` is not judged separately and takes the scores of `mixH2_x` (same model within fp8 rounding of the LoRA factors).

What the two tables add: the mixG, mixH and mixH2 models lie together at mu 11.70 to 12.14 against 12.38 for the teacher (plain-rounding mixF_ptq 10.86, mixI 9.66 and 10.65), and that spread is mostly inside the noise,
so "best overall" and "best aesthetics" are ties. The clear differences are exact text and closeness to the teacher (`mixG_tuned`) and size (`mixH2_x_f8` 1.85 GB, `mixH2_xm_f8` 1.72 GB). HPSv3 agrees in direction that the int4 modulation layers cost fidelity:
the difference to the teacher goes from -0.40 (mixH2_x_f8) to -0.94 (mixH2_xm_f8), a step of about 0.5 mu, at the edge of the noise level.

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
* Never run two GPU jobs with big VRAM on a 16 GB card: they thrash (a 23 s step became 1500 s).
* On Windows the driver pages VRAM to system RAM when it runs short (desktop, another job, a long-running process); sampling then becomes 15-40x slower and does not recover. `gen_inproc.py` therefore runs as a supervisor
  around a worker: the worker keeps 2.5 GB of VRAM free (`reserve_vram`), exits when one image takes longer than 150 s (a normal image takes about 30 s) and is restarted on purpose every 5 images; finished images are skipped.
* `python -I` ignores the script directory; scripts add their own directory to `sys.path` (and read `LOWBITMOSS_HOME` / `COMFYUI_DIR` from the environment).

## Reproducing the ablations

```
cd scripts        # BASE = a tuned moss_lowbit_v1 file in $LOWBITMOSS_HOME/out
python -I make_hybrid.py --inp $BASE --out out/hyb_ffn.safetensors --restore ffn                 # FFN back to bf16
python -I make_hybrid.py --inp $BASE --out out/hyb_b00_13.safetensors --restore blocks:0-13      # every linear of blocks 0-13 back to bf16
python -I make_hybrid.py --inp $BASE --out out/v_ffn_int4.safetensors --restore ffn --ptq int4   # FFN requantized from the teacher by plain rounding
python -I make_hybrid.py --inp $BASE --out out/v_ffn_mid3.safetensors --restore ffn:8-33 --ptq int3  # only the FFN of blocks 8-33 (also attn:LO-HI)
python -I gen_inproc.py --unet out/hyb_ffn.safetensors --tag hyb_ffn --ids 0,1,2,3,4,5,10,14,15  # the 9 critical prompts
python -I sheet.py --dirs eval/teacher_bf16 eval/hyb_ffn eval/v_ffn_int4 --thumb 220 --rows 9 --out eval/sheet_ffn_study.png
```
