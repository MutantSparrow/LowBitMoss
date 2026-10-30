"""Merge the turbo (DMD) LoRA into the bf16 SFT checkpoint -> the distilled teacher as one bf16 file (4 steps, CFG 1, euler/simple).
The merge is bf16(W + dW) with round-to-nearest, i.e. what ComfyUI deploys when it patches the checkpoint with the LoRA (distill_lib.merge).

    python -I -B make_turbo_teacher.py [--out X.safetensors]

CPU only. Every other tensor is copied unchanged. The default output is $LOWBITMOSS_AUX/turbo4/moss_image_sft_turbo4_bf16.safetensors;
point the pipeline at the result with MOSS_TEACHER=<path>. The input is always ComfyUI's moss_image_sft_bf16.safetensors (not MOSS_TEACHER)."""
import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I does not
from _paths import comfy_dir

sys.path.insert(0, comfy_dir())
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import distill_lib  # noqa: E402

TEACHER = os.path.join(comfy_dir(), "models", "diffusion_models", "moss_image_sft_bf16.safetensors")
LINEARS = ("qkv", "o", "ffn_up_gate", "ffn_down")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(distill_lib.TURBO_DIR, "moss_image_sft_turbo4_bf16.safetensors"))
args = ap.parse_args()

with safe_open(TEACHER, "pt") as f:
    metadata = f.metadata()
sd = load_file(TEACHER)
turbo = distill_lib.load_turbo()
changed = 0
for layer in range(42):
    for name in LINEARS:
        key = "layers.%d.proj.%s.weight" % (layer, name)
        sd[key] = distill_lib.merge(sd[key], distill_lib.delta(turbo, layer, name))
        changed += 1
tmp = args.out + ".tmp"
save_file({k: v.contiguous() for k, v in sd.items()}, tmp, metadata=metadata)
os.replace(tmp, args.out)
print("merged %d linears -> %s (%.2f GB)" % (changed, args.out, os.path.getsize(args.out) / 1e9))
