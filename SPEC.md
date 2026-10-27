# moss_lowbit_v1: file format and loader requirements

A quantized MOSS-Image SFT diffusion transformer stored as one `.safetensors` file. The format follows the idea of the PrismML
Bonsai-Image whitepaper (group-wise symmetric low-bit weights with fp16 group scales, small sensitive tensors kept in bf16); the exact
layout below is this project's own. Reference implementation: `scripts/lowbit_pack.py` (pack/unpack/dequantize) and
`custom_nodes/ComfyUI-MOSS-LowBit/lowbit.py` (ComfyUI loader).

The teacher model (`moss_image_sft_bf16.safetensors`, 5.08B parameters, 42 blocks) has per block `layers.N.proj.{qkv,o,ffn_up_gate,ffn_down}.weight`
(bias-free) = 94% of the parameters; everything else (`shared_block_modulation`, `video_head_modulation`, `time_embedding`, `video_embedder`,
`final_linear_video`, `text_embedder.*`, norms) is small and stays bf16 unless quantized explicitly.

## Quantized linears

For each quantized linear X (the 4 matrices of any of the 42 blocks, optionally any other `nn.Linear` such as `text_embedder.*`):

    X.weight        uint8   [out, in*bits//8]  packed codes (bits per code from the kind; element j of a byte at bit j*bits, LSB first)
    X.weight_scale  float16 [out, in//128]     group size 128 along `in` (`in` must be a multiple of 128)
    X.weight_kind   uint8   scalar             1=bin, 2=tern, 3=int2, 4=int4, 5=tern5, 6=int3
    X.lora_a        bf16    [r, in]            optional low-rank correction
    X.lora_b        bf16    [out, r]           optional low-rank correction
    X.bias                                     stays bf16 under its normal name, if the linear has one

Dequantization: `w = (code - offset[kind]) * scale[:, group]` (+ `lora_b @ lora_a` if present, so `W = Q + B A`).

| kind | name | bits/code | code values | offset | weight levels (times scale) | bits/weight incl. scales |
|---|---|---|---|---|---|---|
| 1 | bin | 1 | 0, 1 | 0.5 | -0.5, +0.5 | 1.125 |
| 2 | tern | 2 (slot) | 0, 1, 2 | 1 | -1, 0, +1 | 2.125 |
| 3 | int2 | 2 | 0..3 | 1.5 | -1.5 .. +1.5 | 2.125 |
| 4 | int4 | 4 | 0..15 | 7.5 | -7.5 .. +7.5 | 4.125 |
| 5 | tern5 | 1.6 | 0, 1, 2 | 1 | -1, 0, +1 | 1.725 |
| 6 | int3 | 3 | 0..7 | 3.5 | -3.5 .. +3.5 | 3.125 |

Nothing else is required: no `X.comfy_quant` keys and no file metadata (the writers add `{"moss_lowbit_v1": "1"}` as a marker, readers ignore it).
Every other tensor keeps its original name and dtype (bf16). A linear without `X.weight_kind` is loaded as a plain bf16 linear; kinds can differ per layer.
`X.weight` keeps the original name so ComfyUI's model detection (`shared_block_modulation.weight`, `layers.0.proj.ffn_up_gate.weight`.shape[0],
`layers.0.proj.qkv.weight`) is unchanged; detection only reads shape[0], which the packing does not touch.

Dequantization in the loader is bit-identical to `lowbit_pack.dequantize`: fp32 `(code - offset) * scale`, one rounding to bf16.

## Kind 5 (tern5): ternary codes, 5 per byte

Same codes as kind 2 (code 0, 1, 2 = weight -1, 0, +1, offset 1, fp16 scales `[out, in//128]` unchanged), but 5 codes per byte in base 3
instead of 4 per byte in 2-bit slots: `byte = sum_{j=0..4} code_j * 3^j` (value 0..242, first code in the lowest digit), packed along `in` per
row. Rows are padded with code 1 (weight 0) to a multiple of 5, so `X.weight` is uint8 `[out, ceil(in / 5)]` (in=3072 -> 615 bytes,
in=8192 -> 1639). That is 1.6 bits/code instead of 2 (-20% on the packed codes). The loader derives the expected width as `ceil(in / 5)` and
rejects any other width; `lowbit_pack.packed_cols(in, kind)` gives the width of any kind. In `lowbit_pack`, `KINDS[5][1]` (= 1.6) is the `bits`
argument that selects this packing in `pack_codes` / `unpack_codes`; `unpack_codes(packed, bits, n)` takes `n` to drop the padding. A
ternary-trained layer is exported as kind 5 by setting `qlayer.EXPORT_TERN5 = True` (`train_blocks.py --tern5`); a file may mix kinds 2 and 5.

## Kind 6 (int3): 3-bit codes, 8 per 3 bytes

8-level symmetric codes 0..7 (`w = (code - 3.5) * scale`). 8 consecutive codes along `in` form one little-endian 24-bit word, code j at bit
offset 3*j (`word = sum_j code_j << 3j`), stored as 3 bytes (`word & 255`, `word >> 8 & 255`, `word >> 16`). `in` is a multiple of 128, so
there is no padding: `X.weight` is uint8 `[out, in*3//8]` (in=3072 -> 1152 bytes, in=8192 -> 3072). Relative weight MSE with optimal group
scales on `layers.20.proj.ffn_up_gate.weight` of the teacher: int4 0.0107, int3 0.0377, int2 0.122. The loader derives the expected width
`in*3//8` and rejects any other; a kind-6 linear takes `lora_a`/`lora_b` like every kind. The loader layout's `quantize()` (requantizing a
patched weight) uses 0.75 * absmean scales for int3. Tests: `scripts/test_int3.py` (CPU only).

## ComfyUI loader requirements

Load with the node "MOSS LowBit UNET Loader" (`custom_nodes/ComfyUI-MOSS-LowBit`), not the stock UNETLoader: ComfyUI core only loads quantized
layers that have a hard-coded branch in `comfy/ops.py`, so the pack passes its own `custom_operations`. The node registers a
`QuantizedTensor` layout (`MossLowBitLayout`); the packed codes and scales stay resident and the bf16 weight only exists inside the linear call.
The low-rank term is folded into the layout's `dequantize()` (`W = Q + B A`), which every code path uses (`F.linear(x, QuantizedTensor)` decomposes
before it reaches a registered `aten.linear` handler, so a LoRA branch hooked into that handler is silently skipped).
