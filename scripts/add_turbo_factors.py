"""Add the turbo (DMD) LoRA to a moss_lowbit_v1 file as extra low-rank factors: W = Q + B A + dW, applied in fp32 inside the loader's dequantization.
ComfyUI's own LoRA patching cannot do this for low-bit weights: it re-encodes every patched weight with MossLowBitLayout.quantize (plain absmean rounding, no scales
from training, no low-rank branch), which destroys the QAT result. Linears kept in bf16 get the delta merged with round-to-nearest, as ComfyUI deploys it.

    python -I -B add_turbo_factors.py --inp X.safetensors --out Y.safetensors [--check 4] [--seed 0]

CPU only. Factors are stored in bf16 (any existing fp8 factors are decoded and re-stored in bf16, concatenated with the turbo factors along the rank)."""
import argparse
import os
import random
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I does not
from _paths import comfy_dir

sys.path.insert(0, comfy_dir())
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import distill_lib  # noqa: E402
import lowbit_pack as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--turbo", default=distill_lib.TURBO)
ap.add_argument("--check", type=int, default=4, help="number of random quantized linears whose folded weight is compared with Q + B A + dW")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


with safe_open(args.inp, "pt") as f:
    metadata = f.metadata()
sd = load_file(args.inp)
turbo = distill_lib.load_turbo(args.turbo)
out = dict(sd)
names = [(layer, name) for layer in range(42) for name in ("qkv", "o", "ffn_up_gate", "ffn_down")]
quantized, dense = [], []
for layer, name in names:
    key = "layers.%d.proj.%s" % (layer, name)
    a_t, b_t = distill_lib.factors(turbo, layer, name)
    if key + ".weight_kind" in sd:
        a_old = L.lora_from_stored(sd.get(key + ".lora_a"), sd.get(key + ".lora_a_scale"))
        b_old = L.lora_from_stored(sd.get(key + ".lora_b"), sd.get(key + ".lora_b_scale"))
        for part in ("lora_a", "lora_b", "lora_a_scale", "lora_b_scale"):
            out.pop("%s.%s" % (key, part), None)
        out[key + ".lora_a"] = (a_t if a_old is None else torch.cat([a_old, a_t], 0)).to(torch.bfloat16)
        out[key + ".lora_b"] = (b_t if b_old is None else torch.cat([b_old, b_t], 1)).to(torch.bfloat16)
        quantized.append(key)
    else:
        out[key + ".weight"] = distill_lib.merge(sd[key + ".weight"], b_t @ a_t)
        dense.append(key)
print("%d quantized linears got turbo factors, %d bf16 linears were merged" % (len(quantized), len(dense)))

for key in random.Random(args.seed).sample(quantized, min(args.check, len(quantized))):
    layer, name = int(key.split(".")[1]), key.split(".")[3]
    Q = L.dequantize(sd[key + ".weight"], sd[key + ".weight_scale"], int(sd[key + ".weight_kind"]), torch.float32)
    A, B = L.lora_from_stored(sd.get(key + ".lora_a"), sd.get(key + ".lora_a_scale")), L.lora_from_stored(sd.get(key + ".lora_b"), sd.get(key + ".lora_b_scale"))
    want = Q + (B @ A if A is not None else 0) + distill_lib.delta(turbo, layer, name)
    got = Q + out[key + ".lora_b"].float() @ out[key + ".lora_a"].float()
    old = Q + (B @ A if A is not None else 0)
    print("  %-34s rank %d -> %d   rel err of the folded weight vs Q + B A + dW: %.6f   (delta alone: %.6f, delta size %.6f of W)" % (
        key, 0 if A is None else A.shape[0], out[key + ".lora_a"].shape[0], rel(got, want), rel(got - old, want - old), (want - old).norm().item() / old.norm().item()))

tmp = args.out + ".tmp"
save_file({k: v.contiguous() for k, v in out.items()}, tmp, metadata=metadata)
os.replace(tmp, args.out)
with safe_open(args.out, "pt") as f:
    assert set(f.keys()) == set(out), "key set differs"
print("%s -> %s: %.3f GB -> %.3f GB" % (os.path.basename(args.inp), os.path.basename(args.out), os.path.getsize(args.inp) / 1e9, os.path.getsize(args.out) / 1e9))
