"""CPU check of distill_lib on real data: delta shapes, delta size relative to the weights, and how much of the delta survives the bf16 merge ComfyUI deploys.

    python -I -B test_distill_lib.py          (ComfyUI .venv python, COMFYUI_DIR set)

Needs the 4-step LoRA in $LOWBITMOSS_AUX/turbo4 and ComfyUI's moss_image_sft_bf16.safetensors (not MOSS_TEACHER). Prints, per linear, |dW|/|W|, the part of the intended delta
the round-to-nearest bf16 merge keeps (projection and cosine) and the share of weights that change."""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I does not
from _paths import comfy_dir

sys.path.insert(0, comfy_dir())
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import distill_lib  # noqa: E402

turbo = distill_lib.load_turbo()
f = safe_open(os.path.join(comfy_dir(), "models", "diffusion_models", "moss_image_sft_bf16.safetensors"), "pt", device="cpu")
for layer in (0, 20, 41):
    for name in ("qkv", "o", "ffn_up_gate", "ffn_down"):
        w = f.get_tensor("layers.%d.proj.%s.weight" % (layer, name))
        dw = distill_lib.delta(turbo, layer, name)
        assert dw.shape == w.shape, (layer, name, dw.shape, w.shape)
        m = distill_lib.merge(w, dw)
        applied = m.float() - w.float()
        rel = (dw.norm() / w.float().norm()).item()
        kept = (applied * dw).sum().item() / (dw * dw).sum().item()  # projection of the applied delta on the intended one
        cos = ((applied * dw).sum() / (applied.norm() * dw.norm())).item()
        print("layer %2d %-12s |dW|/|W| %.4f   applied delta: kept %.2f  cos %.2f  changed weights %.1f%%" % (layer, name, rel, kept, cos, 100 * (m != w).float().mean().item()))
print("OK")
