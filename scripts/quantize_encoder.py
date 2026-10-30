"""Prune and post-training-quantize the Qwen3.5-9B text encoder of MOSS-Image into a moss_lowbit_v1 file the stock CLIPLoader can read (custom_operations = lowbit_ops).
Dropped: lm_head, mtp.*, model.visual.* (MOSS conditioning reads the last hidden state with the MTP head off, text only). Kept in bf16: embedding, norms, conv1d, A_log, dt_bias and the two
tiny [32, 4096] linear-attention gate projections. Quantized (group 128, MSE-optimal scales, no training): the linears of the 32 layers.

    python -I -B quantize_encoder.py --out ../out/enc_a.safetensors --mlp int4 --attn int4 --lin int4 --embed int8

kinds: int4 | int3 | int2 | bf16 (keep). Needs the GPU for the rounding. --inp and --embed-src default to files in $COMFYUI_DIR/models/text_encoders
(qwen3.5_9b_bf16.safetensors and ComfyUI's qwen3.5_9b_int8_convrot.safetensors); load the result with gen_inproc.py --clip <file> --clip-lowbit or the MOSS LowBit CLIP Loader node."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import lowbit_pack as L
from _paths import comfy_dir
from qlayer import init_scale

TE = os.path.join(comfy_dir(), "models", "text_encoders")
EMBED = "model.language_model.embed_tokens."
ap = argparse.ArgumentParser()
ap.add_argument("--inp", default=os.path.join(TE, "qwen3.5_9b_bf16.safetensors"))
ap.add_argument("--out", required=True)
ap.add_argument("--embed", default="bf16", help="bf16, or int8: ComfyUI's own int8_tensorwise (convrot) embedding table, copied from --embed-src (its Embedding reads it per row)")
ap.add_argument("--embed-src", default=os.path.join(TE, "qwen3.5_9b_int8_convrot.safetensors"))
ap.add_argument("--mlp", default="int4", help="gate_proj / up_proj / down_proj")
ap.add_argument("--attn", default="int4", help="self_attn q/k/v/o_proj (8 full-attention layers)")
ap.add_argument("--lin", default="int4", help="linear_attn in_proj_qkv / in_proj_z / out_proj (24 layers)")
args = ap.parse_args()

DROP = ("lm_head.", "mtp.", "model.visual.")
LINEAR_KIND = {"mlp": args.mlp, "self_attn": args.attn, "linear_attn": args.lin}
dev = torch.device("cuda")
out, packed_params, bf16_params = {}, 0, 0
if args.embed == "int8":
    with safe_open(args.embed_src, "pt") as g:
        for key in g.keys():
            if key.startswith(EMBED):
                out[key] = g.get_tensor(key)
with safe_open(args.inp, "pt") as f:
    for key in f.keys():
        if key.startswith(DROP) or (args.embed == "int8" and key.startswith(EMBED)):
            continue
        t = f.get_tensor(key)
        group = next((g for g in LINEAR_KIND if ".%s." % g in key), None)
        name = key[:-len(".weight")] if key.endswith(".weight") else None
        if group and name and t.dim() == 2 and t.shape[0] > 64 and ".layers." in key and LINEAR_KIND[group] != "bf16":
            kind = L.KIND_BY_NAME[LINEAR_KIND[group]]
            w = t.to(dev).float()
            sc = init_scale(w, kind).half()
            out[key] = L.pack_codes(L.quantize_to_codes(w, sc, kind), L.KINDS[kind][1]).cpu()
            out[name + ".weight_scale"] = sc.cpu()
            out[name + ".weight_kind"] = torch.tensor(kind, dtype=torch.uint8)
            packed_params += t.numel()
        else:
            out[key] = t
            bf16_params += t.numel()
save_file({k: v.contiguous() for k, v in out.items()}, args.out, metadata={"moss_lowbit_v1": "1"})
print("quantized %.3f B params, kept %.3f B in bf16 -> %s (%.2f GB)" % (packed_params / 1e9, bf16_params / 1e9, args.out, os.path.getsize(args.out) / 1e9))
