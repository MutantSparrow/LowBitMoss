"""CPU-only tests of format kind 6 (int3: codes 0..7, offset 3.5, 8 codes per 3 bytes along `in`).

    python -I -B test_int3.py          (ComfyUI .venv python, COMFYUI_DIR set; tests lowbit_pack.py, qlayer.py and the loader pack in custom_nodes/)
Section 7 reads one tensor of the bf16 teacher when it is present in COMFYUI_DIR/models/diffusion_models and is skipped otherwise.
"""
import ast
import importlib.util
import os
import sys
import tempfile
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # these tests must never touch the GPU (an empty value is dropped by Windows, -1 hides all devices)
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import NODES, comfy_dir  # noqa: E402

COMFY = comfy_dir()
LB_PATH = os.path.join(NODES, "ComfyUI-MOSS-LowBit", "lowbit.py")
sys.path[:0] = [COMFY, HERE]
sys.argv = sys.argv[:1] + ["--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()  # without this ComfyUI ignores --cpu and queries the GPU on import

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import lowbit_pack as L  # noqa: E402
import qlayer  # noqa: E402
from comfy.quant_ops import QuantizedTensor  # noqa: E402

assert not torch.cuda.is_available()

spec =importlib.util.spec_from_file_location("moss_lowbit_dev", LB_PATH)
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)
print("testing: %s | %s | %s" % (L.__file__, qlayer.__file__, LB_PATH))

torch.manual_seed(0)
KIND = 6
BITS = L.KINDS[KIND][1]
assert L.KINDS[KIND] == ("int3", 3, 3.5) and L.KIND_BY_NAME["int3"] == KIND and lb.KINDS[KIND] == (3, 3.5)


def int3_setup(out, n):
    codes = torch.randint(0, 8, (out, n), dtype=torch.uint8)
    scale = (torch.rand(out, n // 128) * 0.1 + 0.01).half()
    return codes, scale


def ref_pack_row(row):
    """independent definition: code j of every 8 sits at bit 3 * j of a little-endian 24-bit word -> 3 bytes"""
    out = []
    for i in range(0, len(row), 8):
        word = sum(int(row[i + j]) << (3 * j) for j in range(8))
        out += [word & 255, (word >> 8) & 255, (word >> 16) & 255]
    return out


# ---- 1. pack/unpack round trip
for n in (128, 3072, 8192):
    codes, _ = int3_setup(64, n)
    packed = L.pack_codes(codes, BITS)
    assert packed.dtype == torch.uint8 and packed.shape == (64, n * 3 // 8) == (64, L.packed_cols(n, KIND)), packed.shape
    assert torch.equal(L.unpack_codes(packed, BITS), codes)
    assert torch.equal(L.unpack_codes(packed, BITS, n), codes)
    assert L.unpack_codes(packed, BITS).dtype == torch.uint8
    assert torch.equal(lb.unpack_codes(packed, lb.KINDS[KIND][0]), codes)  # loader copy of the unpacker
    assert torch.equal(lb.pack_codes(codes, lb.KINDS[KIND][0]), packed)  # loader copy of the packer
    for r in (0, 3, 63):
        assert packed[r].tolist() == ref_pack_row(codes[r].tolist()), (n, r)
    print("kind 6 round trip in=%-5d packed %s  ok" % (n, tuple(packed.shape)))
assert [L.packed_cols(n, KIND) for n in (128, 3072, 8192)] == [48, 1152, 3072]
assert [lb.packed_cols(n, KIND) for n in (128, 3072, 8192)] == [48, 1152, 3072]
for c in (torch.zeros(2, 128, dtype=torch.uint8), torch.full((2, 128), 7, dtype=torch.uint8)):  # all-0 / all-7 words (0x000000 / 0xFFFFFF)
    p = L.pack_codes(c, BITS)
    assert int(p.max()) == (255 if c.max() else 0) and int(p.min()) == (255 if c.max() else 0) and torch.equal(L.unpack_codes(p, BITS), c)
# a non-contiguous codes tensor and an int64 one give the same bytes
codes, _ = int3_setup(8, 256)
assert torch.equal(L.pack_codes(codes.long().t().contiguous().t(), BITS), L.pack_codes(codes, BITS))

# every possible 24-bit word survives pack(unpack(.)): all 2**24 byte triples, in chunks
total, step = 1 << 24, 1 << 22
for start in range(0, total, step):
    w = torch.arange(start, start + step, dtype=torch.int32)
    b = torch.stack((w & 255, (w >> 8) & 255, (w >> 16) & 255), -1).to(torch.uint8).reshape(1, -1)
    assert torch.equal(L.pack_codes(L.unpack_codes(b, BITS), BITS), b)
    assert torch.equal(lb.pack_codes(lb.unpack_codes(b, 3), 3), b)
print("all 2**24 three-byte words: pack(unpack(b)) == b (lowbit_pack and loader)")

# ---- 2. dequantize / quantize_to_codes
for n in (128, 3072, 8192):
    codes, scale = int3_setup(48, n)
    packed = L.pack_codes(codes, BITS)
    ref = ((codes.double() - 3.5).reshape(48, n // 128, 128) * scale.double().unsqueeze(-1)).reshape(48, n)
    got32 = L.dequantize(packed, scale, KIND, torch.float32)
    assert torch.equal(got32, ref.float()), "fp32 dequantize must be exact"
    assert torch.equal(L.dequantize(packed, scale, KIND), ref.bfloat16())  # default dtype bf16: one rounding
    assert torch.equal(L.dequantize(packed, scale, KIND, torch.float16), ref.half())
    assert float(got32.abs().max()) <= 3.5 * float(scale.max()) and float(got32.min()) < 0 < float(got32.max())
    # quantize_to_codes: exact grid points come back, and nearest-level choice for arbitrary weights
    assert torch.equal(L.quantize_to_codes(got32, scale, KIND), codes)
    w = torch.randn(48, n) * 0.05
    qc = L.quantize_to_codes(w, scale, KIND)
    q = (w.reshape(48, n // 128, 128) / scale.float().unsqueeze(-1)).reshape(48, n)
    levels = torch.arange(8, dtype=torch.float32) - 3.5
    near_tie = ((q + 3.5).frac() - 0.5).abs() < 1e-5  # q + 3.5 may round across x.5 in fp32 where the exact distances are equal
    assert torch.equal(qc.long()[~near_tie], (q.unsqueeze(-1) - levels).abs().argmin(-1)[~near_tie])
    assert qc.dtype == torch.uint8 and int(qc.max()) == 7 and int(qc.min()) == 0
    # clamping: weights far outside +-3.5 scales land on codes 7 / 0
    big = torch.where(torch.rand(48, n) < 0.5, 1e3, -1e3)
    assert set(L.quantize_to_codes(big, scale, KIND).unique().tolist()) == {0, 7}
print("dequantize (fp32 exact, bf16, fp16) and quantize_to_codes (nearest, clamp 0..7) vs references: ok")

# ---- 3. loader layout: F.linear(x, QuantizedTensor) vs dense fp32 reference
for n, out in ((3072, 192), (8192, 96), (128, 40)):
    codes, scale = int3_setup(out, n)
    packed = L.pack_codes(codes, BITS)
    r = 16
    A, B = torch.randn(r, n).bfloat16() * 0.05, torch.randn(out, r).bfloat16() * 0.05
    x = torch.randn(3, 20, n).bfloat16()
    for lora in (False, True):
        params = lb.MossLowBitLayout.Params(scale=scale, orig_dtype=torch.bfloat16, orig_shape=(out, n), kind=KIND,
                                            lora_a=A if lora else None, lora_b=B if lora else None)
        qt = QuantizedTensor(packed, lb.LAYOUT, params)
        assert tuple(qt.shape) == (out, n) and tuple(qt.storage_shape) == (out, n * 3 // 8)
        got = F.linear(x, qt)
        ref = F.linear(x.float(), L.dequantize(packed, scale, KIND, torch.float32) + (B.float() @ A.float() if lora else 0))
        rel = ((got.float() - ref).norm() / ref.norm()).item()
        assert rel < 1e-2, rel
        if not lora:  # loader dequantize == lowbit_pack.dequantize (one rounding to bf16)
            assert torch.equal(qt.dequantize(), L.dequantize(packed, scale, KIND, torch.bfloat16))
        else:  # the low-rank term must actually be in the weight (it is not a no-op)
            plain = lb.dequantize_packed(packed, scale, KIND, torch.bfloat16)
            assert not torch.equal(qt.dequantize(), plain)
        # chunked dequantization with row chunks that do not divide `out`
        whole, old_chunk, lb.CHUNK = qt.dequantize(), lb.CHUNK, n * 7
        try:
            chunked = qt.dequantize()
        finally:
            lb.CHUNK = old_chunk
        if lora:  # the low-rank GEMM may round differently per chunk
            assert torch.allclose(chunked.float(), whole.float(), rtol=1e-2, atol=1e-3)
        else:
            assert torch.equal(chunked, whole)
        print("loader layout in=%-5d lora %-5s rel err vs dense fp32 %.4f" % (n, lora, rel))

# requantize path of the layout: valid shape, grid, error between int4 and int2
w = torch.randn(64, 512) * 0.05
rels = {}
for kind in (4, 6, 3):
    qdata, params = lb.MossLowBitLayout.quantize(w, kind=kind)
    assert qdata.shape == (64, lb.packed_cols(512, kind)) and params.kind == kind
    rels[kind] = ((lb.MossLowBitLayout.dequantize(qdata, params).float() - w).norm() / w.norm()).item() ** 2
assert rels[4] < rels[6] < rels[3], rels
assert lb.MossLowBitLayout.quantize(w, kind=6)[0].shape == (64, 192)
print("layout.quantize rel MSE (gaussian): int4 %.4f < int3 %.4f < int2 %.4f" % (rels[4], rels[6], rels[3]))

# ---- 4. loader Linear class: tiny synthetic state dicts
Ops = lb.lowbit_ops(torch.bfloat16)
out, n = 64, 384
codes, scale = int3_setup(out, n)
packed = L.pack_codes(codes, BITS)
x = torch.randn(2, 5, n).bfloat16()
ref = F.linear(x.float(), L.dequantize(packed, scale, KIND, torch.float32))
kind_t = torch.tensor(KIND, dtype=torch.uint8)
for bias in (False, True):
    lin = Ops.Linear(n, out, bias=bias, device="cpu")
    sd = {"weight": packed, "weight_scale": scale, "weight_kind": kind_t}
    if bias:
        sd["bias"] = torch.randn(out).bfloat16()
    lin.load_state_dict(sd)
    assert isinstance(lin.weight.data, QuantizedTensor) and lin.weight.data.params.kind == KIND
    assert tuple(lin.weight.data.shape) == (out, n) and tuple(lin.weight.data.storage_shape) == (out, n * 3 // 8)
    want = ref + (sd["bias"].float() if bias else 0)
    rel = ((lin(x).float() - want).norm() / want.norm()).item()
    assert rel < 1e-2, rel
    print("Linear.load_state_dict kind 6 bias %-5s -> forward rel err %.4f" % (bias, rel))

lin = Ops.Linear(n, out, bias=False, device="cpu")
A, B = torch.randn(8, n).bfloat16() * 0.05, torch.randn(out, 8).bfloat16() * 0.05
lin.load_state_dict({"weight": packed, "weight_scale": scale, "weight_kind": kind_t, "lora_a": A, "lora_b": B})
want = F.linear(x.float(), L.dequantize(packed, scale, KIND, torch.float32) + B.float() @ A.float())
rel = ((lin(x).float() - want).norm() / want.norm()).item()
assert rel < 1e-2, rel
print("Linear with lora_a/lora_b (W = Q + B A) kind 6 -> forward rel err %.4f" % rel)

# shape validation: the packed width must match the kind
c4 = L.pack_codes(codes % 2, 4)  # kind 4 width: n // 2
bad = [(6, c4, "int4 width"), (6, L.pack_codes(codes % 2, 2), "int2 width"), (6, L.pack_codes(codes % 3, L.KINDS[5][1]), "tern5 width"),
       (6, packed[:, :-1], "one byte short"), (6, F.pad(packed, (0, 1)), "one byte long"), (6, packed.reshape(-1)[: out * n * 3 // 8].reshape(out * 2, -1), "rows"),
       (4, packed, "int3 packed as kind 4"), (3, packed, "int3 packed as kind 3"), (5, packed, "int3 packed as kind 5"), (2, packed, "int3 packed as kind 2")]
for kind, p, what in bad:
    try:
        Ops.Linear(n, out, bias=False, device="cpu").load_state_dict({"weight": p, "weight_scale": scale, "weight_kind": torch.tensor(kind, dtype=torch.uint8)})
    except ValueError:
        pass
    else:
        raise AssertionError("wrong packed width accepted: kind %d, %s" % (kind, what))
print("Linear shape validation: %d wrong packed widths rejected" % len(bad))

# a safetensors file with kind 6 and kind 4 linears side by side keeps every dtype/shape
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "mix.safetensors")
    c, sc = int3_setup(16, 256)
    save_file({"a.weight": L.pack_codes(c, BITS), "a.weight_scale": sc, "a.weight_kind": kind_t,
               "b.weight": L.pack_codes(c % 16, 4), "b.weight_scale": sc.clone(), "b.weight_kind": torch.tensor(4, dtype=torch.uint8)}, path, metadata={"moss_lowbit_v1": "1"})
    sd = load_file(path)
    assert sd["a.weight"].shape == (16, 96) and sd["a.weight"].dtype == torch.uint8 and int(sd["a.weight_kind"]) == 6 and sd["b.weight"].shape == (16, 128)
    assert torch.equal(L.unpack_codes(sd["a.weight"], BITS, 256), c)
print("safetensors round trip with mixed kinds 6 / 4: ok")

# ---- 5. qlayer: init_scale / fake_quant / QLinear.export for kind 6 (no qlayer change needed)
w = torch.randn(64, 512) * 0.05
mse = {}
for kind in (4, 6, 3, 2, 1):
    s = qlayer.init_scale(w, kind)
    assert s.shape == (64, 4) and bool((s > 0).all())
    mse[kind] = (((qlayer.fake_quant(w, s, kind) - w) ** 2).sum() / (w ** 2).sum()).item()
assert mse[4] < mse[6] < mse[3] < mse[2] < mse[1], mse
print("qlayer rel weight MSE (gaussian, optimal scale): int4 %.4f  int3 %.4f  int2 %.4f  tern %.4f  bin %.4f" % (mse[4], mse[6], mse[3], mse[2], mse[1]))
s = qlayer.init_scale(w, 6)
fq = qlayer.fake_quant(w, s, 6)
g = fq.reshape(64, 4, 128) / s.unsqueeze(-1)  # fake-quantized weights sit on the 8 levels -3.5..+3.5
assert bool(((g + 3.5 - (g + 3.5).round()).abs() < 1e-4).all()) and float(g.min()) >= -3.5 - 1e-4 and float(g.max()) <= 3.5 + 1e-4 and set((g + 3.5).round().unique().tolist()) <= set(range(8))
# no scale of the grid search is worse than the plain absmax scale
amax = w.reshape(64, 4, 128).abs().amax(-1) / 3.5
assert ((qlayer.fake_quant(w, s, 6) - w) ** 2).sum() <= ((qlayer.fake_quant(w, amax, 6) - w) ** 2).sum()
# straight-through gradient: 1 inside the range, 0 where the weight is clamped, finite scale gradient
wt = w.clone().requires_grad_(True)
st = s.clone().requires_grad_(True)
fq = qlayer.fake_quant(wt, st, 6)
fq.sum().backward()
inside = (wt.detach().reshape(64, 4, 128) / s.unsqueeze(-1)).abs() <= 3.5
assert torch.equal(wt.grad.reshape(64, 4, 128), inside.float()) and bool(torch.isfinite(st.grad).all())

q = qlayer.QLinear(w, 6)
packed_e, sc_e, kind_e = q.export()
assert kind_e.dtype == torch.uint8 and int(kind_e) == 6 and packed_e.dtype == torch.uint8 and packed_e.shape == (64, 512 * 3 // 8) == (64, 192)
assert sc_e.dtype == torch.float16 and sc_e.shape == (64, 4)
# exported file content == what training saw (fake_quant with the fp16-rounded scale)
want = qlayer.fake_quant(q.w.detach(), sc_e.float(), 6)
assert torch.allclose(L.dequantize(packed_e, sc_e, 6, torch.float32), want, rtol=1e-5, atol=1e-7)
assert torch.equal(L.unpack_codes(packed_e, BITS), L.quantize_to_codes(q.w.detach(), sc_e, 6))
# a QLinear forward uses the same grid
xin = torch.randn(3, 512)
assert torch.allclose(q(xin), F.linear(xin, qlayer.fake_quant(q.w, q.scale(), 6)), rtol=1e-5, atol=1e-6)
# EXPORT_TERN5 does not touch kind 6
qlayer.EXPORT_TERN5 = True
try:
    assert int(qlayer.QLinear(w, 6).export()[2]) == 6 and qlayer.QLinear(w, 6).export()[0].shape == (64, 192)
finally:
    qlayer.EXPORT_TERN5 = False
# LoRA init on a kind 6 layer: W - Q - B A is smaller than W - Q
cov = torch.eye(512)
q = qlayer.QLinear(w, 6)
e0 = (q.w - q.quantized_weight()).norm().item()
q.init_lora(8, cov)
e1 = (q.w - q.quantized_weight() - q.lora_b @ q.lora_a).norm().item()
assert e1 < e0
print("QLinear(kind 6).export: packed %s, kind uint8 6, equals fake_quant with the fp16 scale; STE gradient ok" % (tuple(packed_e.shape),))

# ---- 6. e2e_tune PackedLinear (the script cannot be imported: it parses args and needs CUDA); run its class body on CPU
tree = ast.parse(open(os.path.join(HERE, "e2e_tune.py"), encoding="utf-8").read())
cls_src = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PackedLinear")
ns = {"nn": nn, "F": F, "L": L, "torch": torch}
exec(compile(ast.Module([cls_src], []), "e2e_tune.PackedLinear", "exec"), ns)
x = torch.randn(3, 512)
for kind in (6, 4, 1):
    bits = L.KINDS[kind][1]
    c = torch.randint(0, 8 if kind == 6 else 2 ** bits, (32, 512), dtype=torch.uint8)
    sc = (torch.rand(32, 4) * 0.1 + 0.01).half()
    p = L.pack_codes(c, bits)
    pl = ns["PackedLinear"](p, sc, kind)
    assert torch.allclose(pl(x), F.linear(x, L.dequantize(p, sc, kind, torch.float32)), rtol=1e-3, atol=1e-5), kind
print("e2e_tune.PackedLinear.forward kinds 6/4/1: ok")

# ---- 7. a real teacher weight: relative weight MSE of the optimal-scale quantizers (the numbers alloc.py needs) and the export round trip
TEACHER = os.path.join(COMFY, "models", "diffusion_models", "moss_image_sft_bf16.safetensors")
if os.path.exists(TEACHER):
    from safetensors import safe_open  # noqa: E402
    with safe_open(TEACHER, "pt", device="cpu") as f:
        w = f.get_tensor("layers.20.proj.ffn_up_gate.weight").float()
    rel = {}
    for kind in (4, 6, 3):
        s = qlayer.init_scale(w, kind)
        rel[kind] = (((qlayer.fake_quant(w, s, kind) - w) ** 2).sum() / (w ** 2).sum()).item()
    assert 0.03 < rel[6] < 0.045 and rel[4] < 0.012 and 0.11 < rel[3] < 0.13, rel
    q = qlayer.QLinear(w, 6)
    packed_e, sc_e, kind_e = q.export()
    assert packed_e.shape == (16384, 1152) and int(kind_e) == 6
    wd = L.dequantize(packed_e, sc_e, 6, torch.float32)
    assert torch.allclose(wd, qlayer.fake_quant(q.w.detach(), sc_e.float(), 6), rtol=1e-5, atol=1e-7)
    print("layers.20.proj.ffn_up_gate.weight %s: relative weight MSE ||wq-w||^2/||w||^2  int4 %.5f  int3 %.5f  int2 %.5f; kind 6 export (16384, 1152) round trip ok" % (
        tuple(w.shape), rel[4], rel[6], rel[3]))

# ---- 8. speed (CPU, informational) and sizes
codes, scale = int3_setup(16384, 3072)
p3 = L.pack_codes(codes, BITS)
c4 = torch.randint(0, 16, (16384, 3072), dtype=torch.uint8)
p4 = L.pack_codes(c4, 4)
for label, fn in (("unpack int4", lambda: L.unpack_codes(p4, 4)), ("unpack int3", lambda: L.unpack_codes(p3, BITS)),
                  ("pack int4", lambda: L.pack_codes(c4, 4)), ("pack int3", lambda: L.pack_codes(codes, BITS)),
                  ("loader dequantize int4", lambda: lb.dequantize_packed(p4, scale, 4, torch.bfloat16)),
                  ("loader dequantize int3", lambda: lb.dequantize_packed(p3, scale, KIND, torch.bfloat16))):
    fn()
    t0 = time.time()
    for _ in range(3):
        fn()
    print("  %-24s 16384x3072: %6.1f ms" % (label, (time.time() - t0) / 3 * 1000))

BLOCK = {"qkv": (9216, 3072), "o": (3072, 3072), "ffn_up_gate": (16384, 3072), "ffn_down": (3072, 8192)}
params = sum(o * i for o, i in BLOCK.values()) * 42
for kind in (4, 6, 3):
    packed_b = sum(o * L.packed_cols(i, kind) for o, i in BLOCK.values()) * 42
    scales_b = sum(o * (i // 128) * 2 for o, i in BLOCK.values()) * 42
    print("42 blocks, %-4s: packed %.1f MB + scales %.1f MB = %.1f MB  (%.4f bit/weight codes, %.4f with scales)" % (
        L.KINDS[kind][0], packed_b / 1e6, scales_b / 1e6, (packed_b + scales_b) / 1e6, packed_b * 8 / params, (packed_b + scales_b) * 8 / params))
assert abs(sum(o * L.packed_cols(i, KIND) + o * (i // 128) * 2 for o, i in BLOCK.values()) * 8 / sum(o * i for o, i in BLOCK.values()) - 3.125) < 1e-9
print("ALL OK")
