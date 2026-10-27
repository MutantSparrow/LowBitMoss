"""CPU-only tests of format kind 5 (tern5: ternary codes, 5 trits per byte in base 3).

    python -I -B test_tern5.py            (ComfyUI .venv python, COMFYUI_DIR set)
"""
import ast
import importlib.util
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # these tests must never touch the GPU (an empty value is dropped by Windows, -1 hides all devices)
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import NODES, comfy_dir  # noqa: E402

sys.path[:0] = [comfy_dir(), HERE]
sys.argv = sys.argv[:1] + ["--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()  # without this ComfyUI ignores --cpu and queries the GPU on import

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import lowbit_pack as L  # noqa: E402
import qlayer  # noqa: E402
from comfy.quant_ops import QuantizedTensor  # noqa: E402

LB_PATH = os.path.join(NODES, "ComfyUI-MOSS-LowBit", "lowbit.py")
spec = importlib.util.spec_from_file_location("moss_lowbit_dev", LB_PATH)
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)

BITS5 = L.KINDS[5][1]
torch.manual_seed(0)


def ternary_setup(out, n):
    """random ternary codes + fp16 group scales"""
    codes = torch.randint(0, 3, (out, n), dtype=torch.uint8)
    scale = (torch.rand(out, n // 128) * 0.1 + 0.01).half()
    return codes, scale


# ---- 1. pack/unpack round trip
for n in (128, 3072, 8192):
    codes, _ = ternary_setup(64, n)
    packed = L.pack_codes(codes, BITS5)
    assert packed.dtype == torch.uint8 and packed.shape == (64, L.packed_cols(n, 5)) == (64, -(-n // 5)), packed.shape
    assert int(packed.max()) <= 242
    assert torch.equal(L.unpack_codes(packed, BITS5, n), codes)
    full = L.unpack_codes(packed, BITS5)  # without n: padded width, pad trits are code 1 (weight 0)
    assert full.shape == (64, 5 * packed.shape[1]) and torch.equal(full[:, :n], codes) and bool((full[:, n:] == 1).all())
    assert torch.equal(lb.unpack_codes(packed, lb.KINDS[5][0], n), codes)  # loader copy of the unpacker
    assert torch.equal(lb.pack_codes(codes, lb.KINDS[5][0]), packed)  # loader copy of the packer
    # independent definition: byte = sum_j trit_j * 3**j, row-wise, along `in`
    row = codes[3].tolist() + [1] * (-n % 5)
    assert packed[3].tolist() == [sum(row[5 * i + j] * 3 ** j for j in range(5)) for i in range(len(row) // 5)]
    print("kind 5 round trip in=%-5d packed %s  ok" % (n, tuple(packed.shape)))
assert L.packed_cols(3072, 5) == 615 and L.packed_cols(8192, 5) == 1639
assert [L.packed_cols(3072, k) for k in (1, 2, 3, 4)] == [384, 768, 768, 1536]
assert [lb.packed_cols(3072, k) for k in (1, 2, 3, 4, 5)] == [384, 768, 768, 1536, 615]

# all 243 byte values re-encode to themselves
allb = torch.arange(243, dtype=torch.uint8).unsqueeze(0)
assert torch.equal(L.pack_codes(L.unpack_codes(allb, BITS5), BITS5), allb)

# kinds 1-4 behave as before: independent shift/mask reference
for kind in (1, 2, 3, 4):
    bits = L.KINDS[kind][1]
    c = torch.randint(0, 2 ** bits if kind != 2 else 3, (5, 256), dtype=torch.uint8)
    p = L.pack_codes(c, bits)
    ref = sum(c.reshape(5, -1, 8 // bits)[..., j].long() << (j * bits) for j in range(8 // bits)).to(torch.uint8)
    assert torch.equal(p, ref) and p.shape == (5, L.packed_cols(256, kind)) and torch.equal(L.unpack_codes(p, bits), c)
    assert torch.equal(lb.unpack_codes(p, bits), c) and torch.equal(lb.pack_codes(c, bits), p)
print("kinds 1-4 pack/unpack unchanged: ok")

# ---- 2. dequantize: kind 5 is bit-identical to kind 2 for the same codes/scales
for n in (128, 3072, 8192):
    codes, scale = ternary_setup(48, n)
    p2, p5 = L.pack_codes(codes, 2), L.pack_codes(codes, BITS5)
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        assert torch.equal(L.dequantize(p2, scale, 2, dtype), L.dequantize(p5, scale, 5, dtype))
    w = torch.randn(48, n) * 0.05
    assert torch.equal(L.quantize_to_codes(w, scale, 5), L.quantize_to_codes(w, scale, 2))
print("dequantize kind 2 == kind 5 (fp32, bf16, fp16): bit-identical")

# ---- 3. loader layout: F.linear(x, QuantizedTensor) vs dense fp32 reference
for n, out in ((3072, 192), (8192, 96), (128, 40)):
    codes, scale = ternary_setup(out, n)
    p2, p5 = L.pack_codes(codes, 2), L.pack_codes(codes, BITS5)
    r = 16
    A, B = torch.randn(r, n).bfloat16() * 0.05, torch.randn(out, r).bfloat16() * 0.05
    x = torch.randn(3, 20, n).bfloat16()
    for lora in (False, True):
        def make(packed, kind):
            params = lb.MossLowBitLayout.Params(scale=scale, orig_dtype=torch.bfloat16, orig_shape=(out, n), kind=kind,
                                                lora_a=A if lora else None, lora_b=B if lora else None)
            return QuantizedTensor(packed, lb.LAYOUT, params)
        qt5, qt2 = make(p5, 5), make(p2, 2)
        assert tuple(qt5.shape) == (out, n) and tuple(qt5.storage_shape) == (out, L.packed_cols(n, 5))
        got = F.linear(x, qt5)
        ref = F.linear(x.float(), L.dequantize(p5, scale, 5, torch.float32) + (B.float() @ A.float() if lora else 0))
        rel = ((got.float() - ref).norm() / ref.norm()).item()
        assert rel < 1e-2, rel
        assert torch.equal(got, F.linear(x, qt2)), "kind 5 and kind 2 layouts must give identical outputs"
        if not lora:  # loader dequantize == lowbit_pack.dequantize (one rounding to bf16)
            assert torch.equal(qt5.dequantize(), L.dequantize(p5, scale, 5, torch.bfloat16))
        # chunked dequantization with row chunks that do not divide `out` gives the same weight
        whole, old_chunk, lb.CHUNK = qt5.dequantize(), lb.CHUNK, n * 7
        try:
            chunked = qt5.dequantize()
        finally:
            lb.CHUNK = old_chunk
        if lora:  # the low-rank GEMM may round differently per chunk
            assert torch.allclose(chunked.float(), whole.float(), rtol=1e-2, atol=1e-3)
        else:
            assert torch.equal(chunked, whole)
        print("loader layout in=%-5d lora %-5s rel err vs dense fp32 %.4f  (== kind 2 output: yes)" % (n, lora, rel))

# requantize path of the layout (round to nearest) gives the same weight as kind 2
w = torch.randn(64, 256) * 0.05
qdata, params = lb.MossLowBitLayout.quantize(w, kind=5)
assert qdata.shape == (64, 52) and torch.equal(lb.MossLowBitLayout.dequantize(qdata, params), lb.MossLowBitLayout.dequantize(*lb.MossLowBitLayout.quantize(w, kind=2)))

# ---- 4. loader Linear class: tiny synthetic state dicts
Ops = lb.lowbit_ops(torch.bfloat16)
out, n = 64, 384
codes, scale = ternary_setup(out, n)
x = torch.randn(2, 5, n).bfloat16()
ref = F.linear(x.float(), L.dequantize(L.pack_codes(codes, 2), scale, 2, torch.float32))
for bias in (False, True):
    for kind in (5, 2):
        lin = Ops.Linear(n, out, bias=bias, device="cpu")
        sd = {"weight": L.pack_codes(codes, L.KINDS[kind][1]), "weight_scale": scale, "weight_kind": torch.tensor(kind, dtype=torch.uint8)}
        if bias:
            sd["bias"] = torch.randn(out).bfloat16()
        lin.load_state_dict(sd)
        assert isinstance(lin.weight.data, QuantizedTensor) and lin.weight.data.params.kind == kind
        assert tuple(lin.weight.data.shape) == (out, n) and tuple(lin.weight.data.storage_shape) == (out, L.packed_cols(n, kind))
        want = ref + (sd["bias"].float() if bias else 0)
        rel = ((lin(x).float() - want).norm() / want.norm()).item()
        assert rel < 1e-2, rel
        print("Linear.load_state_dict kind %d bias %-5s -> forward rel err %.4f" % (kind, bias, rel))

# lora branch through the Linear class
lin = Ops.Linear(n, out, bias=False, device="cpu")
A, B = torch.randn(8, n).bfloat16() * 0.05, torch.randn(out, 8).bfloat16() * 0.05
lin.load_state_dict({"weight": L.pack_codes(codes, BITS5), "weight_scale": scale, "weight_kind": torch.tensor(5, dtype=torch.uint8), "lora_a": A, "lora_b": B})
want = F.linear(x.float(), L.dequantize(L.pack_codes(codes, 2), scale, 2, torch.float32) + B.float() @ A.float())
assert ((lin(x).float() - want).norm() / want.norm()).item() < 1e-2

# shape validation: the packed width must match the kind
for kind, packed in ((5, L.pack_codes(codes, 2)), (2, L.pack_codes(codes, BITS5)), (5, L.pack_codes(codes, BITS5)[:, :-1])):
    try:
        Ops.Linear(n, out, bias=False, device="cpu").load_state_dict({"weight": packed, "weight_scale": scale, "weight_kind": torch.tensor(kind, dtype=torch.uint8)})
    except ValueError:
        pass
    else:
        raise AssertionError("wrong packed width accepted for kind %d" % kind)
print("Linear shape validation: wrong packed width rejected")

# ---- 5. QLinear.export
w = torch.randn(64, 512) * 0.05
q = qlayer.QLinear(w, 2)
assert qlayer.EXPORT_TERN5 is False
p2, sc2, k2 = q.export()
assert int(k2) == 2 and p2.shape == (64, 128)
qlayer.EXPORT_TERN5 = True
try:
    p5, sc5, k5 = q.export()
    assert int(k5) == 5 and k5.dtype == torch.uint8 and p5.shape == (64, L.packed_cols(512, 5)) and torch.equal(sc2, sc5)
    assert torch.equal(L.dequantize(p2, sc2, 2, torch.float32), L.dequantize(p5, sc5, 5, torch.float32))
    p4, _, k4 = qlayer.QLinear(w, 4).export()
    assert int(k4) == 4 and p4.shape == (64, 256)
finally:
    qlayer.EXPORT_TERN5 = False
print("QLinear.export: kind 2 by default, kind 5 with EXPORT_TERN5, other kinds unaffected")

# ---- 6. e2e_tune.PackedLinear (the script cannot be imported: it parses args and needs CUDA); run its class body on CPU
tree = ast.parse(open(os.path.join(HERE, "e2e_tune.py"), encoding="utf-8").read())
cls_src = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PackedLinear")
ns = {"nn": nn, "F": F, "L": L, "torch": torch}
exec(compile(ast.Module([cls_src], []), "e2e_tune.PackedLinear", "exec"), ns)
x = torch.randn(3, 512)
for kind in (2, 5, 4, 1):
    bits = L.KINDS[kind][1]
    codes = torch.randint(0, 3 if kind in (2, 5) else 2 ** bits, (32, 512), dtype=torch.uint8)
    scale = (torch.rand(32, 4) * 0.1 + 0.01).half()
    packed = L.pack_codes(codes, bits)
    pl = ns["PackedLinear"](packed, scale, kind)
    ref = F.linear(x, L.dequantize(packed, scale, kind, torch.float32))
    assert torch.allclose(pl(x), ref, rtol=1e-3, atol=1e-5), kind
print("e2e_tune.PackedLinear.forward kinds 1/2/4/5: ok")

# ---- 7. speed (CPU, informational) and file size
codes, scale = ternary_setup(16384, 3072)
p2, p5 = L.pack_codes(codes, 2), L.pack_codes(codes, BITS5)
for label, fn in (("unpack kind 2", lambda: L.unpack_codes(p2, 2)), ("unpack kind 5", lambda: L.unpack_codes(p5, BITS5, 3072)),
                  ("dequantize kind 2 (bf16)", lambda: L.dequantize(p2, scale, 2)), ("dequantize kind 5 (bf16)", lambda: L.dequantize(p5, scale, 5)),
                  ("loader dequantize kind 2", lambda: lb.dequantize_packed(p2, scale, 2, torch.bfloat16)),
                  ("loader dequantize kind 5", lambda: lb.dequantize_packed(p5, scale, 5, torch.bfloat16))):
    fn()
    t0 = time.time()
    for _ in range(3):
        fn()
    print("  %-26s 16384x3072: %6.1f ms" % (label, (time.time() - t0) / 3 * 1000))

BLOCK = {"qkv": (9216, 3072), "o": (3072, 3072), "ffn_up_gate": (16384, 3072), "ffn_down": (3072, 8192)}
params = sum(o * i for o, i in BLOCK.values()) * 42
sizes = {}
for kind in (2, 5):
    packed = sum(o * L.packed_cols(i, kind) for o, i in BLOCK.values()) * 42
    scales = sum(o * (i // 128) * 2 for o, i in BLOCK.values()) * 42
    sizes[kind] = (packed, scales)
    print("42 blocks, %-5s: %.3f G weights, packed %.1f MB + scales %.1f MB = %.1f MB  (%.4f bit/weight codes, %.4f with scales)" % (
        L.KINDS[kind][0], params / 1e9, packed / 1e6, scales / 1e6, (packed + scales) / 1e6, packed * 8 / params, (packed + scales) * 8 / params))
saved = sizes[2][0] - sizes[5][0]
print("tern5 saves %.1f MB (%.1f%% of the packed codes, %.1f%% of codes+scales)" % (saved / 1e6, 100 * saved / sizes[2][0], 100 * saved / sum(sizes[2])))
print("ALL OK")
