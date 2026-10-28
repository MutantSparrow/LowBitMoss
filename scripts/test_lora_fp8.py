"""CPU-only tests of the compact LoRA storage: lora_a / lora_b as float8_e4m3fn plus a scalar fp16 lora_a_scale / lora_b_scale.

    python -I -B test_lora_fp8.py          (ComfyUI .venv python, COMFYUI_DIR set)

Tests scripts/lowbit_pack.py, scripts/compact_lora.py, the loader pack in custom_nodes/ and the call site in e2e_tune.py.
scripts/orig holds lowbit_pack.py / lowbit.py as they were before this format: bf16-LoRA files and kinds 1-6 must behave exactly like them.
"""
import ast
import importlib.util
import os
import subprocess
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # these tests must never touch the GPU (an empty value is dropped by Windows, -1 hides all devices)
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import NODES, OUT, comfy_dir  # noqa: E402

LB_PATH = os.path.join(NODES, "ComfyUI-MOSS-LowBit", "lowbit.py")
ORIG = os.path.join(HERE, "orig")
sys.path[:0] = [comfy_dir(), HERE]
sys.argv = sys.argv[:1] + ["--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()  # without this ComfyUI ignores --cpu and queries the GPU on import

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import comfy.memory_management as mm  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.utils  # noqa: E402
import lowbit_pack as L  # noqa: E402
from comfy.quant_ops import QuantizedTensor  # noqa: E402

assert not torch.cuda.is_available()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L0 = load(os.path.join(ORIG, "lowbit_pack.py"), "lowbit_pack_orig")
lb0 = load(os.path.join(ORIG, "lowbit.py"), "moss_lowbit_orig")
lb = load(LB_PATH, "moss_lowbit_new")  # last: both register the layout class "MossLowBitLayout" in ComfyUI, the later one wins
print("testing: %s | %s | %s" % (L.__file__, LB_PATH, ORIG))
torch.manual_seed(0)
FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def raw(t):
    return t.contiguous().reshape(-1).view(torch.uint8)


def expect_error(fn, what):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("accepted: " + what)


# ---- 1. lowbit_pack helpers
amax_seen = []
for amp in (1e-7, 1e-5, 1e-3, 0.05, 1.0, 300.0, 1e4):
    for shape in ((32, 3072), (3072, 64), (64, 8192)):
        t = (torch.randn(shape) * amp).bfloat16()
        q, s = L.lora_to_fp8(t)
        assert q.dtype == FP8 and q.shape == t.shape and s.dtype == torch.float16 and s.shape == () and float(s) > 0
        qf = q.float()
        assert not bool(qf.isnan().any()) and float(qf.abs().max()) <= 448.0
        d = L.lora_from_stored(q, s)
        assert d.dtype == torch.float32 and torch.equal(d, qf * s.float())
        # element bound of e4m3: half an ulp = 2**-4 relative for normal codes, 2**-10 (in code units) for subnormal ones
        bound = 2 ** -4 * t.float().abs() + 2 ** -10 * s.float()
        assert bool((d - t.float()).abs().le(bound * (1 + 1e-6)).all()), (amp, shape)
        e = rel(d, t)
        assert 0.02 < e < 0.032 or amp == 1e-7, (amp, shape, e)  # 1e-7: scale floored at the smallest fp16 subnormal, fewer bits in use
        amax_seen.append(float(qf.abs().max()))
assert all(a >= 224 for a in amax_seen[3:])  # the range is used (at worst half of it: a subnormal fp16 scale rounded up, amp 1e-5)
assert amax_seen[-1] == 448.0 and amax_seen[-4] == 448.0
print("lora_to_fp8: dtype/shape/scale, no NaN, |code| <= 448, per-element e4m3 bound for 21 factors (amp 1e-7..1e4); rel error 2.0-3.2%")

# tensors with outliers and with a few huge values
t = torch.randn(64, 3072) * 0.01
t[3, 5] = 40.0
q, s = L.lora_to_fp8(t)
assert float(q.float().abs().max()) == 448.0 and rel(L.lora_from_stored(q, s), t) < 0.05
# all zeros: scale floored, codes 0, exact
z = torch.zeros(4, 128).bfloat16()
q, s = L.lora_to_fp8(z)
assert float(s) == 2 ** -24 and not bool(q.float().any()) and not bool(L.lora_from_stored(q, s).any())
# the scale is rounded up in fp16: for amax / 448 between subnormal scales (1.3 and 1.4 units of 2**-24 round down to 1, 2.4 to 2, ...) amax / scale must not reach 464 (NaN)
for k in (1.3, 1.4, 2.4, 3.4, 13.6, 100.4, 1000.3):
    t = torch.full((8, 8), k * 2 ** -24 * 448.0)
    q, s = L.lora_to_fp8(t)
    assert not bool(q.float().isnan().any()) and float(q.float().abs().max()) <= 448.0 and float(s) >= k * 2 ** -24 * (1 - 1e-6) and rel(L.lora_from_stored(q, s), t) < 0.07, k
q, s = L.lora_to_fp8(torch.tensor([[1.0, -448.0]]))  # scale exactly 1.0, no rounding needed: the codes are the values
assert float(s) == 1.0 and q.float().tolist() == [[1.0, -448.0]]
# re-compacting an already decoded factor is lossless
t = torch.randn(64, 3072).bfloat16() * 0.03
q, s = L.lora_to_fp8(t)
q2, s2 = L.lora_to_fp8(L.lora_from_stored(q, s))
assert torch.equal(raw(q), raw(q2)) and torch.equal(s, s2)
# every finite e4m3 code survives decode -> encode when the largest one is present (scale is recovered from amax)
codes = torch.arange(256, dtype=torch.uint8).view(FP8)
codes = codes[~codes.float().isnan()].reshape(1, -1)
q3, s3 = L.lora_to_fp8(L.lora_from_stored(codes, torch.tensor(0.0123, dtype=torch.float16)))
assert torch.equal(raw(q3), raw(codes)) and float(s3) == float(torch.tensor(0.0123, dtype=torch.float16))
# lora_from_stored: dtype argument, no scale = plain cast, None passes through
b = torch.randn(8, 16).bfloat16()
assert torch.equal(L.lora_from_stored(b), b.float()) and L.lora_from_stored(b, None, BF16).dtype == BF16 and torch.equal(L.lora_from_stored(b, None, BF16), b)
assert L.lora_from_stored(None) is None and L.lora_from_stored(None, torch.tensor(1.0)) is None
assert L.lora_from_stored(q, s, BF16).dtype == BF16 and torch.equal(L.lora_from_stored(q, s, BF16), (q.float() * s.float()).bfloat16())
print("lora_to_fp8 / lora_from_stored: outliers, zeros, tiny amax (scale rounded up, no NaN), idempotent re-compaction, all e4m3 codes, dtype argument")

# lowbit_pack otherwise untouched
assert L.KINDS == L0.KINDS and L.KIND_BY_NAME == L0.KIND_BY_NAME and L.GROUP == L0.GROUP and L.TERN5_BITS == L0.TERN5_BITS
for kind in range(1, 7):
    bits = L.KINDS[kind][1]
    c = torch.randint(0, 2 if kind == 1 else 3 if kind in (2, 5) else 2 ** bits, (24, 384), dtype=torch.uint8)
    sc = (torch.rand(24, 3) * 0.1 + 0.01).half()
    p = L.pack_codes(c, bits)
    assert torch.equal(p, L0.pack_codes(c, bits)) and L.packed_cols(384, kind) == L0.packed_cols(384, kind)
    assert torch.equal(L.unpack_codes(p, bits, 384), L0.unpack_codes(p, bits, 384))
    for dt in (torch.float32, BF16, torch.float16):
        assert torch.equal(L.dequantize(p, sc, kind, dt), L0.dequantize(p, sc, kind, dt))
    w = torch.randn(24, 384) * 0.05
    assert torch.equal(L.quantize_to_codes(w, sc, kind), L0.quantize_to_codes(w, sc, kind))
print("lowbit_pack kinds 1-6: pack, unpack, packed_cols, dequantize, quantize_to_codes identical to orig/")


# ---- 2. loader layout: fp8 factors vs bf16 factors vs dense reference, kinds 1-6
def make_q(kind, out, n):
    bits = L.KINDS[kind][1]
    c = torch.randint(0, 2 if kind == 1 else 3 if kind in (2, 5) else 2 ** bits, (out, n), dtype=torch.uint8)
    sc = (torch.rand(out, n // 128) * 0.1 + 0.01).half()
    return L.pack_codes(c, bits), sc


def params(packed, sc, kind, out, n, A=None, B=None, sa=None, sb=None):
    return lb.MossLowBitLayout.Params(scale=sc, orig_dtype=BF16, orig_shape=(out, n), kind=kind, lora_a=A, lora_b=B, lora_a_scale=sa, lora_b_scale=sb)


worst = 0.0
for kind in range(1, 7):
    for n, out, r in ((3072, 192, 64), (8192, 96, 32), (128, 40, 8)):
        packed, sc = make_q(kind, out, n)
        A, B = torch.randn(r, n).bfloat16() * 0.05, torch.randn(out, r).bfloat16() * 0.05
        A8, sa = L.lora_to_fp8(A)
        B8, sb = L.lora_to_fp8(B)
        Q = L.dequantize(packed, sc, kind, torch.float32)
        delta_bf, delta_8 = B.float() @ A.float(), L.lora_from_stored(B8, sb) @ L.lora_from_stored(A8, sa)
        x = torch.randn(3, 20, n).bfloat16()
        qt_bf = QuantizedTensor(packed, lb.LAYOUT, params(packed, sc, kind, out, n, A, B))
        qt_8 = QuantizedTensor(packed, lb.LAYOUT, params(packed, sc, kind, out, n, A8, B8, sa, sb))
        assert qt_8.params._tensor_fields() == ["scale", "lora_a", "lora_b", "lora_a_scale", "lora_b_scale"]
        assert qt_bf.params._tensor_fields() == ["scale", "lora_a", "lora_b"]
        # fp8 weight == dense fp32 reference with the decoded factors, up to the one bf16 rounding (1 ulp = 2**-7 relative)
        got = qt_8.dequantize()
        ref = Q + delta_8
        assert got.dtype == BF16 and bool((got.float() - ref).abs().le(ref.abs() * 2 ** -7 + 1e-6).all()), (kind, n)
        assert torch.equal(got, ref.bfloat16()) or rel(got, ref) < 2e-3
        # versus the bf16-factor weight (both in fp32, no bf16 rounding noise) the difference is the factor error: ||W8 - Wbf|| / ||B A|| ~ 3.7%
        e = rel(lb.dequantize_packed(packed, sc, kind, torch.float32, A8, B8, sa, sb) - Q, lb.dequantize_packed(packed, sc, kind, torch.float32, A, B) - Q)
        assert 0.02 < e < 0.06, (kind, n, e)
        worst = max(worst, e)
        # the plain-linear path (F.linear on the tensor class) sees the fp8 factors
        y = F.linear(x, qt_8)
        want = F.linear(x.float(), ref)
        assert rel(y, want) < 1e-2
        assert not torch.equal(qt_8.dequantize(), lb.dequantize_packed(packed, sc, kind, BF16))  # the low-rank term is in the weight
        # chunked dequantization with row chunks that do not divide `out`
        old_chunk, lb.CHUNK = lb.CHUNK, n * 7
        try:
            chunked = qt_8.dequantize()
        finally:
            lb.CHUNK = old_chunk
        assert torch.allclose(chunked.float(), got.float(), rtol=1e-2, atol=1e-3)
        # one factor fp8 and the other bf16 (each factor stands for itself)
        qt_mix = QuantizedTensor(packed, lb.LAYOUT, params(packed, sc, kind, out, n, A8, B, sa, None))
        assert qt_mix.params._tensor_fields() == ["scale", "lora_a", "lora_b", "lora_a_scale"]
        ref_mix = Q + B.float() @ L.lora_from_stored(A8, sa)
        assert bool((qt_mix.dequantize().float() - ref_mix).abs().le(ref_mix.abs() * 2 ** -7 + 1e-6).all())
print("loader layout kinds 1-6 x 3 shapes: fp8 weight == dense reference with decoded factors (1 bf16 ulp); vs bf16 factors ||dW|| / ||B A|| <= %.3f; F.linear, chunking, fp8/bf16 mix" % worst)

# ComfyUI's Params machinery: to_device, clone, copy_from, empty_like, flatten, gathered buffers, state_dict, byte accounting
kind, n, out, r = 4, 3072, 192, 32
packed, sc = make_q(kind, out, n)
A, B = torch.randn(r, n).bfloat16() * 0.05, torch.randn(out, r).bfloat16() * 0.05
A8, sa = L.lora_to_fp8(A)
B8, sb = L.lora_to_fp8(B)
p8 = params(packed, sc, kind, out, n, A8, B8, sa, sb)
qt = QuantizedTensor(packed, lb.LAYOUT, p8)
ref_w = qt.dequantize()
moved = p8.to_device(torch.device("cpu"))
assert all(torch.equal(raw(getattr(moved, f)), raw(getattr(p8, f))) for f in p8._tensor_fields()) and moved.lora_a.dtype == FP8 and moved.lora_a_scale.dtype == torch.float16
cl = p8.clone()
assert all(torch.equal(raw(getattr(cl, f)), raw(getattr(p8, f))) and getattr(cl, f).data_ptr() != getattr(p8, f).data_ptr() for f in p8._tensor_fields())
qt2 = qt.to(torch.float16)  # only orig_dtype changes, the fp8 factors stay
assert qt2.params.lora_a.dtype == FP8 and qt2.params.lora_a_scale.shape == () and qt2.dtype == torch.float16
dst = torch.empty_like(qt)  # what ComfyUI's cast path does: empty_like, then copy_
assert isinstance(dst, QuantizedTensor) and dst.params.lora_a.dtype == FP8
dst.copy_(qt)
assert torch.equal(dst.dequantize(), ref_w)
names, ctx = qt.__tensor_flatten__()
assert names == ["_qdata", "_param_scale", "_param_lora_a", "_param_lora_b", "_param_lora_a_scale", "_param_lora_b_scale"], names
size = mm.vram_aligned_size(qt)
assert size == sum(mm.vram_aligned_size(getattr(qt, nm)) for nm in names)
buf = torch.empty(size, dtype=torch.uint8)
gview = mm.interpret_gathered_like([qt], buf)[0]  # dynamic-VRAM style single buffer holding the qdata, scales and fp8 factors
gview.copy_(qt)
assert gview.params.lora_a.dtype == FP8 and gview.params.lora_a_scale.shape == () and torch.equal(gview.dequantize(), ref_w)
sdt = lb.MossLowBitLayout.state_dict_tensors(packed, p8)
assert list(sdt) == ["", "_scale", "_kind", "_lora_a", "_lora_b", "_lora_a_scale", "_lora_b_scale"]
sum_bytes = sum(t.nbytes for t in sdt.values())
sd_bf = lb.MossLowBitLayout.state_dict_tensors(packed, params(packed, sc, kind, out, n, A, B))
assert list(sd_bf) == ["", "_scale", "_kind", "_lora_a", "_lora_b"]
assert sum(t.nbytes for t in sd_bf.values()) - sum_bytes == A.nbytes + B.nbytes - A8.nbytes - B8.nbytes - 4
print("Params machinery: to_device, clone, .to(dtype), empty_like + copy_, flatten fields, gathered buffer (%d B), state_dict_tensors bytes: ok" % size)

# ---- 3. Linear.load_state_dict (strict): fp8 keys, bf16 keys, shapes
Ops, Ops0 = lb.lowbit_ops(BF16), lb0.lowbit_ops(BF16)
kind_t = lambda k: torch.tensor(k, dtype=torch.uint8)  # noqa: E731
for kind in range(1, 7):
    out, n, r = 64, 384, 8
    packed, sc = make_q(kind, out, n)
    A, B = torch.randn(r, n).bfloat16() * 0.05, torch.randn(out, r).bfloat16() * 0.05
    A8, sa = L.lora_to_fp8(A)
    B8, sb = L.lora_to_fp8(B)
    bias = torch.randn(out).bfloat16()
    x = torch.randn(2, 5, n).bfloat16()
    base = {"weight": packed, "weight_scale": sc, "weight_kind": kind_t(kind), "bias": bias}
    Q = L.dequantize(packed, sc, kind, torch.float32)
    # fp8 keys
    lin = Ops.Linear(n, out, bias=True, device="cpu")
    lin.load_state_dict({**base, "lora_a": A8, "lora_a_scale": sa, "lora_b": B8, "lora_b_scale": sb})
    pr = lin.weight.data.params
    assert isinstance(lin.weight.data, QuantizedTensor) and pr.kind == kind and pr.lora_a.dtype == FP8 and pr.lora_b.dtype == FP8
    assert pr.lora_a_scale.shape == () and pr.lora_a_scale.dtype == torch.float16 and float(pr.lora_b_scale) == float(sb)
    want = F.linear(x.float(), Q + L.lora_from_stored(B8, sb) @ L.lora_from_stored(A8, sa), bias.float())
    assert rel(lin(x), want) < 1e-2
    # bf16 keys: as before (cast to the compute dtype, no scales), and identical to what the original loader produces
    lin_b = Ops.Linear(n, out, bias=True, device="cpu")
    lin_b.load_state_dict({**base, "lora_a": A, "lora_b": B})
    pb = lin_b.weight.data.params
    assert pb.lora_a.dtype == BF16 and pb.lora_a_scale is None and pb.lora_b_scale is None and pb._tensor_fields() == ["scale", "lora_a", "lora_b"]
    lin_0 = Ops0.Linear(n, out, bias=True, device="cpu")
    lin_0.load_state_dict({**base, "lora_a": A, "lora_b": B})
    p0 = lin_0.weight.data.params
    assert torch.equal(pb.lora_a, p0.lora_a) and torch.equal(pb.lora_b, p0.lora_b) and torch.equal(pb.scale, p0.scale) and pb.kind == p0.kind
    assert torch.equal(lb.dequantize_packed(packed, sc, kind, BF16, pb.lora_a, pb.lora_b), lb0.dequantize_packed(packed, sc, kind, BF16, p0.lora_a, p0.lora_b))
    assert torch.equal(lin_b.weight.data.dequantize(), lb0.dequantize_packed(packed, sc, kind, BF16, p0.lora_a, p0.lora_b))  # new layout, bf16 file: bit-identical to the old loader
    # a float16 / float32 factor without scale still goes to bf16 like before
    lin_h = Ops.Linear(n, out, bias=True, device="cpu")
    lin_h.load_state_dict({**base, "lora_a": A.half(), "lora_b": B.float()})
    assert lin_h.weight.data.params.lora_a.dtype == BF16 and lin_h.weight.data.params.lora_b.dtype == BF16
    # no lora at all
    lin_n = Ops.Linear(n, out, bias=True, device="cpu")
    lin_n.load_state_dict(base)
    pn = lin_n.weight.data.params
    assert pn.lora_a is None and pn.lora_a_scale is None and pn._tensor_fields() == ["scale"]
    assert torch.equal(lin_n.weight.data.dequantize(), L.dequantize(packed, sc, kind, BF16))
    # memory accounting counts the fp8 factors and their scales (module_size sums state_dict())
    m8, mb = comfy.model_management.module_size(lin), comfy.model_management.module_size(lin_b)
    assert mb - m8 == A.nbytes + B.nbytes - A8.nbytes - B8.nbytes - 4, (mb, m8)
print("Linear.load_state_dict kinds 1-6: fp8 keys (stay fp8 + scale), bf16 keys (identical to the old loader), fp16/fp32 keys, no lora; module_size counts fp8 bytes + scales")

# wrong shapes / wrong combinations
kind, out, n, r = 6, 64, 384, 8
packed, sc = make_q(kind, out, n)
A, B = torch.randn(r, n).bfloat16() * 0.05, torch.randn(out, r).bfloat16() * 0.05
A8, sa = L.lora_to_fp8(A)
B8, sb = L.lora_to_fp8(B)
base = {"weight": packed, "weight_scale": sc, "weight_kind": kind_t(kind)}
good8 = {"lora_a": A8, "lora_a_scale": sa, "lora_b": B8, "lora_b_scale": sb}


def try_load(extra, drop=()):
    sd = {**base, **{k: v for k, v in {**good8, **extra}.items() if k not in drop and v is not None}}
    Ops.Linear(n, out, bias=False, device="cpu").load_state_dict(sd)


try_load({})  # the good set loads
bad = [("lora_a in dim", {"lora_a": A8[:, :-1].contiguous()}), ("lora_a in dim +1", {"lora_a": F.pad(A8.float(), (0, 1)).to(FP8)}),
       ("lora_b out dim", {"lora_b": B8[:-1].contiguous()}), ("rank mismatch", {"lora_b": B8[:, :-1].contiguous()}), ("transposed lora_a", {"lora_a": A8.t().contiguous()}),
       ("transposed lora_b", {"lora_b": B8.t().contiguous()}), ("lora_a fp8 without scale", {}, ("lora_a_scale",)), ("lora_b fp8 without scale", {}, ("lora_b_scale",)),
       ("bf16 lora_a with scale", {"lora_a": A}), ("scale vector", {"lora_a_scale": sa.reshape(1).repeat(2)}), ("scale without lora_a", {}, ("lora_a",)),
       ("scale without lora_b", {}, ("lora_b",)), ("only lora_a", {}, ("lora_b", "lora_b_scale")), ("only lora_b", {}, ("lora_a", "lora_a_scale")),
       ("scale but no factors", {}, ("lora_a", "lora_b", "lora_b_scale"))]
for item in bad:
    what, extra, drop = item if len(item) == 3 else (*item, ())
    expect_error(lambda: try_load(extra, drop), what)
try_load({"lora_a_scale": sa.reshape(1)})  # a [1]-shaped scale is still a scalar
print("Linear wrong-shape / wrong-combination rejection: %d cases raise ValueError; [1]-shaped scale accepted" % len(bad))

# kinds 1-5 wrong packed widths still rejected with fp8 factors present
for k in (1, 2, 3, 4, 5, 6):
    p, s = make_q(k, out, n)
    expect_error(lambda: Ops.Linear(n, out, bias=False, device="cpu").load_state_dict({"weight": p[:, :-1], "weight_scale": s, "weight_kind": kind_t(k), **good8}), "short packed width, kind %d" % k)

# ---- 4. safetensors / ComfyUI's loader read the compact form (F8_E4M3 + 0-dim F16)
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "mini.safetensors")
    save_file({"a.weight": packed, "a.weight_scale": sc, "a.weight_kind": kind_t(kind), "a.lora_a": A8, "a.lora_a_scale": sa, "a.lora_b": B8, "a.lora_b_scale": sb,
               "b.weight": packed.clone(), "b.weight_scale": sc.clone(), "b.weight_kind": kind_t(kind), "b.lora_a": A.clone(), "b.lora_b": B.clone()}, path, metadata={"moss_lowbit_v1": "1"})
    for sd in (load_file(path), comfy.utils.load_torch_file(path)):
        assert sd["a.lora_a"].dtype == FP8 and sd["a.lora_a_scale"].shape == () and sd["a.lora_a_scale"].dtype == torch.float16 and sd["b.lora_a"].dtype == BF16
        assert torch.equal(raw(sd["a.lora_b"]), raw(B8)) and float(sd["a.lora_b_scale"]) == float(sb)
        la, lb_ = Ops.Linear(n, out, bias=False, device="cpu"), Ops.Linear(n, out, bias=False, device="cpu")
        la.load_state_dict({k[2:]: v for k, v in sd.items() if k.startswith("a.")})
        lb_.load_state_dict({k[2:]: v for k, v in sd.items() if k.startswith("b.")})
        assert la.weight.data.params.lora_a.dtype == FP8 and lb_.weight.data.params.lora_a.dtype == BF16
        Q = L.dequantize(packed, sc, kind, torch.float32)
        assert rel(la.weight.data.dequantize() - Q, lb_.weight.data.dequantize() - Q) < 0.12  # includes the bf16 rounding of both weights
print("safetensors + comfy.utils.load_torch_file: F8_E4M3 factors and 0-dim fp16 scales load into the loader Linear")

# ---- 5. compact_lora.py on a small mixed file
with tempfile.TemporaryDirectory() as d:
    src, dst, dst2 = (os.path.join(d, nm) for nm in ("in.safetensors", "out.safetensors", "out2.safetensors"))
    tensors = {"norm.weight": torch.randn(256).bfloat16(), "rope.bands": torch.randn(16).bfloat16(), "x.bias": torch.full((4,), float("nan")).bfloat16()}
    for name, k, rows in (("layers.0.proj.qkv", 6, 128), ("layers.1.proj.o", 1, 64), ("layers.2.proj.o", 4, 64)):
        p_, s_ = make_q(k, rows, 256)
        tensors.update({name + ".weight": p_, name + ".weight_scale": s_, name + ".weight_kind": kind_t(k)})
    tensors.update({"layers.0.proj.qkv.lora_a": torch.randn(8, 256).bfloat16() * 0.05, "layers.0.proj.qkv.lora_b": torch.randn(128, 8).bfloat16() * 0.02,
                    "layers.1.proj.o.lora_a": torch.zeros(4, 256).bfloat16(), "layers.1.proj.o.lora_b": torch.randn(64, 4).bfloat16()})
    save_file(tensors, src, metadata={"moss_lowbit_v1": "1"})
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    run = lambda *a: subprocess.run([sys.executable, "-I", "-B", os.path.join(HERE, "compact_lora.py"), *a], capture_output=True, text=True, env=env)  # noqa: E731
    r1 = run("--inp", src, "--out", dst, "--check", "2")
    assert r1.returncode == 0, r1.stderr[-2000:]
    assert "saved" in r1.stdout and "lora_a relative error" in r1.stdout and "rel err of B A" in r1.stdout and "2 linears with a low-rank correction" in r1.stdout, r1.stdout
    o = load_file(dst)
    assert set(o) == set(tensors) | {f"layers.{i}.proj.{p}.lora_{x}_scale" for i, p in ((0, "qkv"), (1, "o")) for x in "ab"}
    for k, v in tensors.items():
        if ".lora_" in k:
            q, s = L.lora_to_fp8(v)
            assert o[k].dtype == FP8 and torch.equal(raw(o[k]), raw(q)) and torch.equal(o[k + "_scale"], s) and o[k + "_scale"].shape == ()
        else:
            assert o[k].dtype == v.dtype and torch.equal(raw(o[k]), raw(v)), k  # bit-identical, including the NaN bias
    assert not bool(L.lora_from_stored(o["layers.1.proj.o.lora_a"], o["layers.1.proj.o.lora_a_scale"]).any())
    with safe_open(dst, "pt") as f1, safe_open(src, "pt") as f2:
        assert f1.metadata() == f2.metadata() == {"moss_lowbit_v1": "1"}
    assert os.path.getsize(dst) < os.path.getsize(src) and not os.path.exists(dst + ".tmp")
    # running it on its own output changes nothing
    r2 = run("--inp", dst, "--out", dst2, "--check", "0")
    assert r2.returncode == 0, r2.stderr[-2000:]
    assert "4 of them already compact" in r2.stdout, r2.stdout
    o2 = load_file(dst2)
    assert set(o2) == set(o) and all(torch.equal(raw(o2[k]), raw(o[k])) for k in o)
    # a broken input is refused: fp8 factor without scale, scale on a bf16 factor, lora_a without lora_b
    bad_in = os.path.join(d, "bad.safetensors")
    for name, mutate in (("fp8 without scale", lambda t: t.pop("layers.0.proj.qkv.lora_a_scale")), ("scale on bf16", lambda t: t.update({"layers.2.proj.o.lora_a": torch.zeros(2, 256).bfloat16(), "layers.2.proj.o.lora_b": torch.zeros(64, 2).bfloat16(), "layers.2.proj.o.lora_a_scale": sa})),
                         ("lora_a without lora_b", lambda t: t.pop("layers.0.proj.qkv.lora_b"))):
        t = dict(o)
        mutate(t)
        save_file(t, bad_in)
        assert run("--inp", bad_in, "--out", os.path.join(d, "never.safetensors")).returncode != 0, name
        assert not os.path.exists(os.path.join(d, "never.safetensors")), name
print("compact_lora.py: mixed file converted (factors == lora_to_fp8, other tensors bit-identical incl. NaN, metadata kept), idempotent, bad inputs refused")

# ---- 6. e2e_tune PackedLinear fed through lora_from_stored (the call-site change it needs)
script = "e2e_tune.py"
tree = ast.parse(open(os.path.join(HERE, script), encoding="utf-8").read())
cls_src = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PackedLinear")
ns = {"nn": nn, "F": F, "L": L, "torch": torch}
exec(compile(ast.Module([cls_src], []), script + ".PackedLinear", "exec"), ns)
out, n, r = 32, 512, 16
packed, sc = make_q(4, out, n)
A, B = torch.randn(r, n).bfloat16() * 0.05, torch.randn(out, r).bfloat16() * 0.05
A8, sa = L.lora_to_fp8(A)
B8, sb = L.lora_to_fp8(B)
sdict = {"w.lora_a": A8, "w.lora_a_scale": sa, "w.lora_b": B8, "w.lora_b_scale": sb}
pl = ns["PackedLinear"](packed, sc, 4, None, L.lora_from_stored(sdict.get("w.lora_a"), sdict.get("w.lora_a_scale")), L.lora_from_stored(sdict.get("w.lora_b"), sdict.get("w.lora_b_scale")))
x = torch.randn(3, n)
want = F.linear(x, L.dequantize(packed, sc, 4, torch.float32) + L.lora_from_stored(B8, sb) @ L.lora_from_stored(A8, sa))
assert torch.allclose(pl(x), want, rtol=1e-3, atol=1e-4), script
pl0 = ns["PackedLinear"](packed, sc, 4, None, L.lora_from_stored(None), L.lora_from_stored(None))  # a file without lora: None, None
assert torch.allclose(pl0(x), F.linear(x, L.dequantize(packed, sc, 4, torch.float32)), rtol=1e-3, atol=1e-5)
print("e2e_tune PackedLinear with L.lora_from_stored(sd.get(...), sd.get(..._scale)) as lora_a / lora_b: ok")

# ---- 7. a real file written by compact_lora.py, if present in $LOWBITMOSS_HOME/out: loader Linear on real layers, fp8 vs bf16 original
REAL0, REAL8 = (os.path.join(OUT, nm) for nm in ("mixH2_x.safetensors", "mixH2_x_fp8lora.safetensors"))


def deq32(lin):
    qt = lin.weight.data
    p = qt.params
    return lb.dequantize_packed(qt._qdata, p.scale, p.kind, torch.float32, p.lora_a, p.lora_b, p.lora_a_scale, p.lora_b_scale), p


if os.path.exists(REAL0) and os.path.exists(REAL8):
    with safe_open(REAL0, "pt") as f0, safe_open(REAL8, "pt") as f8:
        assert f8.metadata() == f0.metadata()
        k0, k8 = set(f0.keys()), set(f8.keys())
        assert k8 - k0 == {k + "_scale" for k in k0 if ".lora_" in k} and k0 <= k8
        for name in ("layers.20.proj.ffn_up_gate", "layers.5.proj.qkv", "layers.33.proj.ffn_down"):
            assert name + ".lora_a" in k0
            sc_ = f0.get_tensor(name + ".weight_scale")
            lin0, lin8 = (Ops.Linear(sc_.shape[1] * 128, sc_.shape[0], bias=False, device="cpu") for _ in range(2))
            lin0.load_state_dict({k[len(name) + 1:]: f0.get_tensor(k) for k in k0 if k.startswith(name + ".")})
            lin8.load_state_dict({k[len(name) + 1:]: f8.get_tensor(k) for k in k8 if k.startswith(name + ".")})
            assert lin8.weight.data.params.lora_a.dtype == FP8 and lin0.weight.data.params.lora_a.dtype == BF16
            for part in ("lora_a", "lora_b"):  # the shipped file holds exactly what lora_to_fp8 makes of the original factors
                q_, s_ = L.lora_to_fp8(f0.get_tensor(f"{name}.{part}"))
                assert torch.equal(raw(f8.get_tensor(f"{name}.{part}")), raw(q_)) and torch.equal(f8.get_tensor(f"{name}.{part}_scale"), s_)
            w0, p0 = deq32(lin0)
            w8, p8_ = deq32(lin8)
            qw = lb.dequantize_packed(lin0.weight.data._qdata, p0.scale, p0.kind, torch.float32)
            # the original file through the new loader is bit-identical to the old loader function
            assert torch.equal(lin0.weight.data.dequantize(), lb0.dequantize_packed(lin0.weight.data._qdata, p0.scale, p0.kind, BF16, p0.lora_a, p0.lora_b))
            print("  %-30s %s rank %-3d ||W8 - W|| / ||W|| %.5f   ||W8 - W|| / ||B A|| %.4f" % (name, tuple(w0.shape), p0.lora_a.shape[0], rel(w8, w0), rel(w8 - qw, w0 - qw)))
            assert rel(w8, w0) < 0.01 and 0.02 < rel(w8 - qw, w0 - qw) < 0.06
    print("real mixH2_x vs mixH2_x_fp8lora: key sets, metadata, loader Linear on 3 layers: ok")
else:
    print("(no mixH2_x_fp8lora.safetensors: real-file check skipped)")
print("ALL OK")
