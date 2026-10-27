"""CPU-only test of the low-rank branch in the loader layout: F.linear(x, QuantizedTensor) must equal x @ (Q + B A)^T for kinds 1-4,
and the layout params must keep lora_a/lora_b through to_device / clone.

    python -I -B test_lora_layout.py          (ComfyUI .venv python, COMFYUI_DIR set)
"""
import importlib.util
import os
import sys

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
import torch.nn.functional as F  # noqa: E402

import lowbit_pack as L  # noqa: E402
from comfy.quant_ops import QuantizedTensor  # noqa: E402

spec = importlib.util.spec_from_file_location("moss_lowbit_dev", os.path.join(NODES, "ComfyUI-MOSS-LowBit", "lowbit.py"))
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)

torch.manual_seed(0)
out, n, r = 256, 512, 16
for kind in (1, 2, 3, 4):
    bits = L.KINDS[kind][1]
    w = torch.randn(out, n)
    sc = (w.reshape(out, n // 128, 128).abs().mean(-1) * (2 if kind in (1, 3) else 1)).half()
    packed = L.pack_codes(L.quantize_to_codes(w, sc, kind), bits)
    A, B = torch.randn(r, n).bfloat16() * 0.05, torch.randn(out, r).bfloat16() * 0.05
    x = torch.randn(3, 20, n).bfloat16()
    for lora in (False, True):
        p = lb.MossLowBitLayout.Params(scale=sc, orig_dtype=torch.bfloat16, orig_shape=(out, n), kind=kind, lora_a=A if lora else None, lora_b=B if lora else None)
        qt = QuantizedTensor(packed, lb.LAYOUT, p)
        got = F.linear(x, qt)
        Wq = L.dequantize(packed, sc, kind, torch.float32)
        ref = F.linear(x.float(), Wq + (B.float() @ A.float() if lora else 0))
        rel = ((got.float() - ref).norm() / ref.norm()).item()
        assert rel < 1e-2, (kind, lora, rel)
        moved, cloned = p.to_device(torch.device("cpu")), p.clone()
        assert all((q.lora_a is not None) == lora and (q.lora_b is not None) == lora for q in (moved, cloned))
        print("kind %d lora %-5s rel err vs dense fp32 %.4f  tensor fields %s" % (kind, lora, rel, p._tensor_fields()))
print("ALL OK")
