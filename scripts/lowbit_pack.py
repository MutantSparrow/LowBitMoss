"""moss_lowbit_v1: group-wise symmetric low-bit weight format shared by the quantizer, the ComfyUI loader node and tests.

A quantized linear X (weight [out, in], group size G=128 along `in`) is stored as
    X.weight        uint8   [out, in * bits // 8]   packed codes, element j of a byte sits at bit offset j * bits (little endian)
    X.weight_scale  float16 [out, in // G]
    X.weight_kind   uint8   scalar                  KIND code, see KINDS
and dequantizes to   w = (code - OFFSET[kind]) * scale[:, group]
kind 5 (tern5) packs 5 ternary codes per byte in base 3 instead: byte = sum_j code_j * 3**j, codes padded with 1 (weight 0) to a
multiple of 5 along `in`, so X.weight is uint8 [out, ceil(in / 5)]; see packed_cols.
kind 6 (int3) packs 8 codes per 3 bytes: code j of a group of 8 sits at bit offset 3 * j of a little-endian 24-bit word, so X.weight is
uint8 [out, in * 3 // 8].
An optional rank-r correction W = Q + B A is stored as X.lora_a [r, in] and X.lora_b [out, r], either bf16 or compact: float8_e4m3fn plus a
scalar fp16 X.lora_a_scale / X.lora_b_scale, the effective factor being fp8.float() * scale (lora_to_fp8, lora_from_stored).
"""
from functools import cache

import torch

GROUP = 128
FP8_MAX = 448.0  # largest finite float8_e4m3fn; the format has no inf and rounds 464 and above to NaN
TERN5_BITS = 8 / 5  # bits per code of kind 5; pack_codes / unpack_codes take it as their `bits` argument
# kind code -> (name, bits per code, offset)
KINDS = {1: ("bin", 1, 0.5), 2: ("tern", 2, 1.0), 3: ("int2", 2, 1.5), 4: ("int4", 4, 7.5), 5: ("tern5", TERN5_BITS, 1.0), 6: ("int3", 3, 3.5)}
KIND_BY_NAME = {v[0]: k for k, v in KINDS.items()}
# integer code grids: w / scale = code - offset
#   bin  : {0,1}        -> {-0.5, +0.5}   (scale is the full distance between the two levels, alpha = scale / 2)
#   tern : {0,1,2}      -> {-1, 0, +1}    (code 3 is unused)
#   tern5: {0,1,2}      -> {-1, 0, +1}    (same codes as tern, 5 per byte)
#   int2 : {0,1,2,3}    -> {-1.5,-0.5,+0.5,+1.5}
#   int4 : {0..15}      -> {-7.5 .. +7.5}
#   int3 : {0..7}       -> {-3.5 .. +3.5}


def packed_cols(in_features, kind):
    """width of the packed uint8 weight of a linear with `in_features` inputs"""
    bits = KINDS[kind][1]
    return -(-in_features // 5) if bits == TERN5_BITS else in_features * bits // 8


@cache
def _tern5_lut(device):
    """[243, 5] uint8: the 5 trits of every tern5 byte value"""
    return (torch.arange(243, device=device).unsqueeze(1) // 3 ** torch.arange(5, device=device) % 3).to(torch.uint8)


def pack_codes(codes, bits):
    """codes: integer tensor [out, in] with values < 2**bits -> uint8 [out, in * bits // 8] (bits=TERN5_BITS: codes 0..2 -> [out, ceil(in / 5)])"""
    out, n = codes.shape
    if bits == TERN5_BITS:
        c = torch.nn.functional.pad(codes.to(torch.uint8), (0, -n % 5), value=1).reshape(out, -1, 5)
        return (c * (3 ** torch.arange(5, device=codes.device)).to(torch.uint8)).sum(-1, dtype=torch.uint8)
    if bits == 3:
        shifts = 3 * torch.arange(8, device=codes.device, dtype=torch.int32)
        word = (codes.to(torch.int32).reshape(out, n // 8, 8) << shifts).sum(-1, dtype=torch.int32)  # 24-bit word per 8 codes
        return torch.stack((word, word >> 8, word >> 16), -1).bitwise_and_(255).to(torch.uint8).reshape(out, -1)
    per_byte = 8 // bits
    c = codes.to(torch.uint8).reshape(out, n // per_byte, per_byte)
    packed = torch.zeros(out, n // per_byte, dtype=torch.uint8, device=codes.device)
    for j in range(per_byte):
        packed |= c[..., j] << (j * bits)
    return packed


def unpack_codes(packed, bits, n=None):
    """uint8 [out, m] -> uint8 codes [out, m * 8 // bits]; for bits=TERN5_BITS [out, 5 * m], or [out, n] if given (drops the padding)"""
    if bits == TERN5_BITS:
        codes = _tern5_lut(packed.device).index_select(0, packed.reshape(-1).int()).reshape(packed.shape[0], -1)
        return codes[:, :n]
    if bits == 3:
        b = packed.reshape(packed.shape[0], -1, 3).int()
        word = b[..., 0] | b[..., 1] << 8 | b[..., 2] << 16
        shifts = 3 * torch.arange(8, device=packed.device, dtype=torch.int32)
        return (word.unsqueeze(-1) >> shifts).bitwise_and_(7).to(torch.uint8).reshape(packed.shape[0], -1)
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    shifts = torch.arange(per_byte, device=packed.device, dtype=torch.uint8) * bits
    return ((packed.unsqueeze(-1) >> shifts) & mask).reshape(packed.shape[0], -1)


def dequantize(packed, scale, kind, dtype=torch.bfloat16):
    """reference dequantization: packed uint8, scale fp16 [out, in/G], kind code (int) -> [out, in]"""
    _, bits, offset = KINDS[int(kind)]
    codes = unpack_codes(packed, bits, scale.shape[1] * GROUP)
    out, n = codes.shape
    w = (codes.float() - offset).reshape(out, n // GROUP, GROUP) * scale.float().unsqueeze(-1)
    return w.reshape(out, n).to(dtype)


def quantize_to_codes(w, scale, kind):
    """nearest-code quantization of w [out, in] with given scale [out, in/G] -> uint8 codes [out, in]"""
    _, bits, offset = KINDS[int(kind)]
    out, n = w.shape
    q = w.float().reshape(out, n // GROUP, GROUP) / scale.float().unsqueeze(-1).clamp_min(1e-12)
    if kind == 1:
        codes = (q > 0).to(torch.uint8)
    else:
        codes = torch.round(q + offset).clamp_(0, 2 ** bits - 1 if kind not in (2, 5) else 2).to(torch.uint8)
    return codes.reshape(out, n)


def lora_to_fp8(t):
    """low-rank factor -> (float8_e4m3fn tensor, fp16 scalar scale) with the largest |element| at (just under) 448; lora_from_stored(*result) is the factor back"""
    t = t.float()
    # e4m3fn has no inf (464 and above becomes NaN), so the fp16 scale is rounded up, never down: |t / scale| <= 448 by construction. No other headroom is needed,
    # the error of real factors is flat (2.6% relative) for amax / scale anywhere in 14..448. The scale is floored at the smallest fp16 subnormal (all-zero factor).
    scale = (t.abs().amax() / FP8_MAX).clamp_min(2 ** -24)
    s16 = scale.half()
    s16 = torch.where(s16.float() < scale, (s16.view(torch.int16) + 1).view(torch.float16), s16)
    return (t / s16.float()).to(torch.float8_e4m3fn), s16


def lora_from_stored(t, scale=None, dtype=torch.float32):
    """stored factor (bf16, or float8_e4m3fn with its fp16 scale) -> effective factor in `dtype`; None stays None"""
    if t is None:
        return None
    t = t.float()
    if scale is not None:
        t = t * scale.float()
    return t.to(dtype)
