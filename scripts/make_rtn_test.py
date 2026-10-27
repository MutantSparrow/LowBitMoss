"""Naive RTN conversion of the bf16 MOSS-Image teacher to moss_lowbit_v1, for loader tests (quality is irrelevant).

    python -I make_rtn_test.py int4|bin|mixed [out.safetensors]      (needs a GPU; default output: $LOWBITMOSS_HOME/out/test_<mode>.safetensors)

int4 / bin: the 4 matrices of every block. mixed: blocks 0-1 tern, 2-3 int2, 4-5 bin, 6-40 int4, block 41 stays bf16, plus
text_embedder.blocks.0.qkv (int4, has a bias), shared_block_modulation (int2) and final_linear_video (tern).
Scales: int4 absmax / 7.5, otherwise absmean (x2 for bin and int2).
"""
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import OUT, teacher_path
from lowbit_pack import GROUP, KIND_BY_NAME, KINDS, pack_codes, quantize_to_codes

BLOCK_LINEAR = re.compile(r"layers\.(\d+)\.proj\.(qkv|o|ffn_up_gate|ffn_down)\.weight")


def kind_for(name, mode):
    m = BLOCK_LINEAR.fullmatch(name)
    if mode != "mixed":
        return KIND_BY_NAME[mode] if m else None
    extra = {"text_embedder.blocks.0.qkv.weight": "int4", "shared_block_modulation.weight": "int2", "final_linear_video.weight": "tern"}
    if name in extra:
        return KIND_BY_NAME[extra[name]]
    if not m or int(m.group(1)) == 41:
        return None
    block = int(m.group(1))
    return KIND_BY_NAME["tern" if block < 2 else "int2" if block < 4 else "bin" if block < 6 else "int4"]


def quantize(w, kind):
    out, n = w.shape
    g = w.float().reshape(out, n // GROUP, GROUP)
    scale = g.abs().amax(-1) / 7.5 if kind == 4 else g.abs().mean(-1) * (2.0 if kind in (1, 3) else 1.0)
    scale = scale.half().clamp_min(1e-4)
    return pack_codes(quantize_to_codes(w, scale, kind), KINDS[kind][1]), scale


mode = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT, f"test_{mode}.safetensors")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
sd = {}
with safe_open(teacher_path(), "pt") as f:
    for name in f.keys():
        w = f.get_tensor(name)
        kind = kind_for(name, mode)
        if kind is None:
            sd[name] = w
            continue
        packed, scale = quantize(w.cuda(), kind)
        sd[name] = packed.cpu()
        sd[name[:-len("weight")] + "weight_scale"] = scale.cpu()
        sd[name[:-len("weight")] + "weight_kind"] = torch.tensor(kind, dtype=torch.uint8)
save_file(sd, out_path, metadata={"moss_lowbit_v1": "1"})
print(out_path, "%.2f GB" % (os.path.getsize(out_path) / 1e9))
