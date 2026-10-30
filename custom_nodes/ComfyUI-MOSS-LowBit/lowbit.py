"""moss_lowbit_v1 weights as a ComfyUI QuantizedTensor layout.

Per quantized linear X:
    X.weight       uint8   [out, in * bits // 8]  packed codes, element j of a byte at bit j * bits
    X.weight_scale float16 [out, in // 128]
    X.weight_kind  uint8   scalar                 1 bin, 2 tern, 3 int2, 4 int4, 5 tern5, 6 int3
kind 5 is ternary with 5 codes per byte in base 3 (byte = sum_j code_j * 3**j), zero-padded to a multiple of 5: X.weight is [out, ceil(in / 5)].
kind 6 is int3: 8 codes per 3 bytes (code j at bit 3 * j of a little-endian 24-bit word), X.weight is [out, in * 3 // 8].
Optional low-rank correction W = Q + B A: X.lora_a [r, in], X.lora_b [out, r], bf16, or float8_e4m3fn with a scalar fp16 X.lora_a_scale / X.lora_b_scale
(effective factor = fp8.float() * scale). fp8 factors stay fp8 in the layout params and are converted inside dequantize_packed.
The packed codes and scales stay resident; the bf16 weight only exists inside the linear call.
"""
from dataclasses import dataclass
from functools import cache

import torch

import comfy.ops
from comfy.quant_ops import QUANT_ALGOS, QuantizedLayout, QuantizedTensor, register_layout_class, register_layout_op
from comfy_kitchen.tensor.base import BaseLayoutParams

FORMAT = "moss_lowbit"
LAYOUT = "MossLowBitLayout"
GROUP = 128
TERN5_BITS = 8 / 5  # bits per code of kind 5, also how unpack_codes / pack_codes recognize it
# kind -> (bits per code, offset); w = (code - offset) * scale
KINDS = {1: (1, 0.5), 2: (2, 1.0), 3: (2, 1.5), 4: (4, 7.5), 5: (TERN5_BITS, 1.0), 6: (3, 3.5)}
# elements per fp32 temporary during dequantization
CHUNK = 1 << 24


def packed_cols(n, kind):
    bits = KINDS[kind][0]
    return -(-n // 5) if bits == TERN5_BITS else n * bits // 8


@cache
def tern5_lut(device):
    """[243, 5] uint8: the 5 trits of every tern5 byte value"""
    return (torch.arange(243, device=device).unsqueeze(1) // 3 ** torch.arange(5, device=device) % 3).to(torch.uint8)


def unpack_codes(packed, bits, n=None):
    """codes [out, n]; n is only needed for tern5, whose last byte of a row is padded"""
    if bits == TERN5_BITS:
        return tern5_lut(packed.device).index_select(0, packed.reshape(-1).int()).reshape(packed.shape[0], -1)[:, :n]
    if bits == 3:
        b = packed.reshape(packed.shape[0], -1, 3).int()
        word = b[..., 0] | b[..., 1] << 8 | b[..., 2] << 16
        return (word.unsqueeze(-1) >> 3 * torch.arange(8, device=packed.device, dtype=torch.int32)).bitwise_and_(7).to(torch.uint8).reshape(packed.shape[0], -1)
    shifts = torch.arange(0, 8, bits, device=packed.device, dtype=torch.uint8)
    return ((packed.unsqueeze(-1) >> shifts) & (2 ** bits - 1)).reshape(packed.shape[0], -1)


def pack_codes(codes, bits):
    if bits == TERN5_BITS:
        c = torch.nn.functional.pad(codes.to(torch.uint8), (0, -codes.shape[1] % 5), value=1).reshape(codes.shape[0], -1, 5)
        return (c * (3 ** torch.arange(5, device=codes.device)).to(torch.uint8)).sum(-1, dtype=torch.uint8)
    if bits == 3:
        word = (codes.to(torch.int32).reshape(codes.shape[0], -1, 8) << 3 * torch.arange(8, device=codes.device, dtype=torch.int32)).sum(-1, dtype=torch.int32)
        return torch.stack((word, word >> 8, word >> 16), -1).bitwise_and_(255).to(torch.uint8).reshape(codes.shape[0], -1)
    shifts = torch.arange(0, 8, bits, device=codes.device, dtype=torch.uint8)
    return (codes.to(torch.uint8).reshape(codes.shape[0], -1, 8 // bits) << shifts).sum(-1, dtype=torch.uint8)


def dequantize_packed(packed, scale, kind, dtype, lora_a=None, lora_b=None, lora_a_scale=None, lora_b_scale=None):
    bits, offset = KINDS[kind]
    out, n = packed.shape[0], scale.shape[1] * GROUP
    result = torch.empty(out, n, dtype=dtype, device=packed.device)
    rows = max(1, CHUNK // n)
    if lora_a is not None:
        lora_a = lora_a.to(device=packed.device, dtype=torch.float32)
        if lora_a_scale is not None:
            lora_a = lora_a * lora_a_scale.to(packed.device).float()
    for i in range(0, out, rows):
        s = scale[i:i + rows].float().unsqueeze(-1)
        codes = unpack_codes(packed[i:i + rows], bits, n).reshape(s.shape[0], n // GROUP, GROUP)
        # (code - offset) * scale is exact in fp32, so the cast to dtype is the only rounding
        w = torch.addcmul(s * -offset, codes.float(), s).reshape(-1, n)
        if lora_a is not None:  # low-rank correction folded into the weight: W = Q + B A
            b = lora_b[i:i + rows].to(device=packed.device, dtype=torch.float32)
            if lora_b_scale is not None:
                b = b * lora_b_scale.to(packed.device).float()
            w.addmm_(b, lora_a)
        result[i:i + rows] = w
    return result


class MossLowBitLayout(QuantizedLayout):
    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        kind: int = 4
        lora_a: torch.Tensor = None  # optional low-rank correction [r, in] / [out, r]: bf16, or float8_e4m3fn next to its scalar fp16 scale
        lora_b: torch.Tensor = None
        lora_a_scale: torch.Tensor = None
        lora_b_scale: torch.Tensor = None

        def _tensor_fields(self):
            return ["scale"] + [f for f in ("lora_a", "lora_b", "lora_a_scale", "lora_b_scale") if getattr(self, f) is not None]

        def _validate_tensor_fields(self):
            out, n = self.orig_shape
            if self.kind not in KINDS:
                raise ValueError(f"unknown moss_lowbit kind {self.kind}")
            if n % GROUP != 0 or tuple(self.scale.shape) != (out, n // GROUP):
                raise ValueError(f"moss_lowbit scale {tuple(self.scale.shape)} does not fit weight {tuple(self.orig_shape)} with group {GROUP}")

    @classmethod
    def quantize(cls, tensor, kind=4, **kwargs):
        # round to nearest, absmean scales (x2 bin/int2, x0.75 int3; absmax for int4); not reached from ComfyUI (requantize_kwargs refuses), only the CPU tests call it
        bits, offset = KINDS[kind]
        out, n = tensor.shape
        w = tensor.float().reshape(out, n // GROUP, GROUP)
        if kind == 4:
            scale = w.abs().amax(-1, keepdim=True) / 7.5
        else:
            scale = w.abs().mean(-1, keepdim=True) * (2.0 if kind in (1, 3) else 0.75 if kind == 6 else 1.0)
        scale = scale.half().clamp_min(1e-4)
        q = w / scale.float()
        codes = (q > 0) if kind == 1 else (q + offset).round().clamp(0, 2 if kind in (2, 5) else 2 ** bits - 1)
        params = cls.Params(scale=scale.squeeze(-1), orig_dtype=tensor.dtype, orig_shape=tuple(tensor.shape), kind=kind)
        return pack_codes(codes.reshape(out, n), bits), params

    @classmethod
    def dequantize(cls, qdata, params):
        return dequantize_packed(qdata, params.scale, params.kind, params.orig_dtype, params.lora_a, params.lora_b, params.lora_a_scale, params.lora_b_scale)

    @classmethod
    def get_plain_tensors(cls, qtensor):
        return qtensor._qdata, qtensor._params.scale

    @classmethod
    def state_dict_tensors(cls, qdata, params):
        out = {"": qdata, "_scale": params.scale, "_kind": torch.tensor(params.kind, dtype=torch.uint8)}
        for name in ("lora_a", "lora_b", "lora_a_scale", "lora_b_scale"):
            if getattr(params, name) is not None:
                out["_" + name] = getattr(params, name)
        return out

    @classmethod
    def requantize_kwargs(cls, qtensor):
        # ComfyUI re-encodes a LoRA-patched weight through here; that would re-round it with plain absmean scales and drop the trained scales and low-rank branch
        raise NotImplementedError("LoRA patches do not work on moss_lowbit weights; fold the LoRA into the file with scripts/add_turbo_factors.py")


@register_layout_op(torch.ops.aten.linear.default, MossLowBitLayout)
def _linear(qt, args, kwargs):
    x, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None
    return torch.nn.functional.linear(x, weight.dequantize(), bias)


register_layout_class(LAYOUT, MossLowBitLayout)
QUANT_ALGOS[FORMAT] = {
    "storage_t": torch.uint8,
    "parameters": {"weight_scale"},
    "comfy_tensor_layout": LAYOUT,
    "quantize_input": False,
}


def load_lora_factor(key, t, scale, device, dtype):
    """(factor, scale) as kept in the layout params: float8_e4m3fn stays fp8 next to its scalar scale, any other dtype goes to the compute dtype"""
    if t is None:
        if scale is not None:
            raise ValueError(f"{key}_scale without {key}")
        return None, None
    if (t.dtype == torch.float8_e4m3fn) != (scale is not None):
        raise ValueError(f"{key}: a float8_e4m3fn factor needs {key}_scale and a {t.dtype} one must not have it")
    if scale is None:
        return t.to(device=device, dtype=dtype), None
    if scale.numel() != 1:
        raise ValueError(f"{key}_scale must be a scalar, got shape {tuple(scale.shape)}")
    return t.to(device=device), scale.reshape(()).to(device=device)


def lowbit_ops(compute_dtype):
    """mixed precision ops whose Linear also reads moss_lowbit_v1 weights; linears without X.weight_kind load as usual"""
    base = comfy.ops.mixed_precision_ops({}, compute_dtype)

    class Ops(base):
        class Linear(base.Linear):
            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                if f"{prefix}weight_kind" not in state_dict:
                    return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

                weight = state_dict.pop(f"{prefix}weight")
                scale = state_dict.pop(f"{prefix}weight_scale")
                kind = int(state_dict.pop(f"{prefix}weight_kind"))
                lora_a, lora_b = state_dict.pop(f"{prefix}lora_a", None), state_dict.pop(f"{prefix}lora_b", None)
                lora_a_scale, lora_b_scale = state_dict.pop(f"{prefix}lora_a_scale", None), state_dict.pop(f"{prefix}lora_b_scale", None)
                out, n = self._orig_shape
                if weight.shape != (out, packed_cols(n, kind)):
                    raise ValueError(f"{prefix}weight: packed shape {tuple(weight.shape)} does not fit a [{out}, {n}] linear of kind {kind}")
                device = self.factory_kwargs["device"]
                if (lora_a is None) != (lora_b is None) or (lora_a is not None and (lora_a.shape[1] != n or lora_b.shape[0] != out or lora_a.shape[0] != lora_b.shape[1])):
                    raise ValueError(f"{prefix}lora_a/lora_b do not fit a [{out}, {n}] linear")
                lora_a, lora_a_scale = load_lora_factor(f"{prefix}lora_a", lora_a, lora_a_scale, device, compute_dtype)
                lora_b, lora_b_scale = load_lora_factor(f"{prefix}lora_b", lora_b, lora_b_scale, device, compute_dtype)
                params = MossLowBitLayout.Params(scale=scale.to(device=device), orig_dtype=compute_dtype, orig_shape=self._orig_shape, kind=kind,
                                                 lora_a=lora_a, lora_b=lora_b, lora_a_scale=lora_a_scale, lora_b_scale=lora_b_scale)
                self.quant_format = FORMAT
                self.layout_type = LAYOUT
                self.weight = torch.nn.Parameter(QuantizedTensor(weight.to(device=device), LAYOUT, params), requires_grad=False)
                super(base.Linear, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
                if f"{prefix}weight" in missing_keys:
                    missing_keys.remove(f"{prefix}weight")

    return Ops
