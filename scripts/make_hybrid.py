"""Ablation helper: take a moss_lowbit_v1 file and restore selected linears to the bf16 teacher weights (drops their quant keys).
    python make_hybrid.py --inp ../out/X.safetensors --out ../out/hyb_attn.safetensors --restore attn
--restore: attn (qkv,o of all blocks) | ffn (ffn_up_gate,ffn_down of all blocks) | blocks:LO-HI (every linear of those blocks) | text_embedder"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from safetensors.torch import load_file, save_file

import lowbit_pack as L
from _paths import teacher_path
from qlayer import init_scale

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--restore", required=True)
ap.add_argument("--ptq", default=None, help="instead of bf16, requantize the selected linears from the teacher with optimal-scale rounding at this kind (bin|tern|int2|int4); their LoRA is dropped")
args = ap.parse_args()

sd = load_file(args.inp)
teacher = load_file(teacher_path())
kinds = [k[:-len(".weight_kind")] for k in sd if k.endswith(".weight_kind")]


def selected(name):
    if args.restore == "attn":
        return bool(re.fullmatch(r"layers\.\d+\.proj\.(qkv|o)", name))
    if args.restore == "ffn":
        return bool(re.fullmatch(r"layers\.\d+\.proj\.(ffn_up_gate|ffn_down)", name))
    if args.restore == "text_embedder":
        return name.startswith("text_embedder.")
    for group, pattern in (("ffn", r"layers\.(\d+)\.proj\.(ffn_up_gate|ffn_down)"), ("attn", r"layers\.(\d+)\.proj\.(qkv|o)")):
        if args.restore.startswith(group + ":"):  # e.g. ffn:8-33 = the FFN of blocks 8..33 only
            lo, hi = (int(v) for v in args.restore[len(group) + 1:].split("-"))
            m = re.fullmatch(pattern, name)
            return bool(m) and lo <= int(m.group(1)) <= hi
    if args.restore.startswith("blocks:"):
        lo, hi = (int(v) for v in args.restore[len("blocks:"):].split("-"))
        m = re.match(r"layers\.(\d+)\.", name)
        return bool(m) and lo <= int(m.group(1)) <= hi
    raise SystemExit("unknown --restore " + args.restore)


n = 0
for name in kinds:
    if selected(name):
        for suffix in (".weight_scale", ".weight_kind", ".lora_a", ".lora_b"):
            sd.pop(name + suffix, None)
        if args.ptq:
            kind = L.KIND_BY_NAME[args.ptq]
            w = teacher[name + ".weight"].float().cuda()
            sc = init_scale(w, kind).half()
            sd[name + ".weight"] = L.pack_codes(L.quantize_to_codes(w, sc, kind), L.KINDS[kind][1]).cpu()
            sd[name + ".weight_scale"] = sc.cpu()
            sd[name + ".weight_kind"] = torch.tensor(kind, dtype=torch.uint8)
        else:
            sd[name + ".weight"] = teacher[name + ".weight"]
        n += 1
save_file({k: v.contiguous() for k, v in sd.items()}, args.out, metadata={"moss_lowbit_v1": "1"})
print("restored %d of %d quantized linears to bf16 -> %s (%.2f GB)" % (n, len(kinds), args.out, os.path.getsize(args.out) / 1e9))
