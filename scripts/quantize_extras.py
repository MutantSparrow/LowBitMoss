"""Add PTQ (optimal-scale rounding, no training) quantization of the non-block linears to a moss_lowbit_v1 file.
Policy comes from logs/sens_extra.json: text_embedder blocks are almost insensitive -> int2 (ffn_down int4), text_embedder.proj int4,
optionally the modulation linears -> int4."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from safetensors.torch import load_file, save_file

import lowbit_pack as L
from qlayer import init_scale

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mod", action="store_true", help="also quantize shared_block_modulation and video_head_modulation to int4")
ap.add_argument("--time", action="store_true", help="also quantize time_embedding.2 to int4")
args = ap.parse_args()

policy = {"text_embedder.proj": "int4"}
for i in range(2):
    for n in ("qkv", "proj", "ffn_gate", "ffn_up"):
        policy["text_embedder.blocks.%d.%s" % (i, n)] = "int2"
    policy["text_embedder.blocks.%d.ffn_down" % i] = "int4"
if args.mod:
    policy["shared_block_modulation"] = "int4"
    policy["video_head_modulation"] = "int4"
if args.time:
    policy["time_embedding.2"] = "int4"

sd = load_file(args.inp)
dev = "cuda" if torch.cuda.is_available() else "cpu"
for name, kname in policy.items():
    key = name + ".weight"
    if name + ".weight_kind" in sd:
        continue
    kind = L.KIND_BY_NAME[kname]
    w = sd[key].to(dev).float()
    sc = init_scale(w, kind).half()
    sd[key] = L.pack_codes(L.quantize_to_codes(w, sc, kind), L.KINDS[kind][1]).cpu()
    sd[name + ".weight_scale"] = sc.cpu()
    sd[name + ".weight_kind"] = torch.tensor(kind, dtype=torch.uint8)
    print("quantized", name, kname)
save_file({k: v.contiguous() for k, v in sd.items()}, args.out, metadata={"moss_lowbit_v1": "1"})
print("wrote", args.out, "%.3f GB" % (os.path.getsize(args.out) / 1e9))
