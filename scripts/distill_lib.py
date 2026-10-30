"""Turbo (DMD) LoRA helpers for quantizing the distilled model.
The LoRA (akhaliq/MOSS-Image-SFT-4-steps lora.safetensors) holds separate q/k/v/o/ffn_up_gate/ffn_down factors for all 42 blocks, rank 128, alpha 128, fp32.
ComfyUI deploys it on bf16 weights as round-to-nearest bf16(W + alpha/rank * up @ down) (comfy.float.stochastic_rounding is a plain cast for bf16, and the
cast-time patch adds in bf16). The delta is mostly below half a bf16 step, so only the larger entries survive; the teacher blocks here use exactly that merge.
The file is read from $LOWBITMOSS_AUX/turbo4/lora.safetensors (downloaded by the user, see the README)."""
import os

import torch

import comfy.utils
from _paths import AUX

TURBO_DIR = os.path.join(AUX, "turbo4")
TURBO = os.path.join(TURBO_DIR, "lora.safetensors")


def load_turbo(path=TURBO):
    return comfy.utils.load_torch_file(path)


def factors(turbo, layer, name):
    """fp32 (A [r, in], B [out, r]) with delta = B @ A for the fused model layout (qkv = rows [q | k | v], equal thirds, so rank 3 x 128 with a block-diagonal B)"""
    if name == "qkv":
        parts = [factors(turbo, layer, n) for n in "qkv"]
        return torch.cat([a for a, _ in parts], 0), torch.block_diag(*[b for _, b in parts])
    p = "diffusion_model.layers.%d.proj.%s" % (layer, name)
    down, up = turbo[p + ".lora_down.weight"].float(), turbo[p + ".lora_up.weight"].float()
    return down, up * (float(turbo[p + ".alpha"]) / down.shape[0])


def delta(turbo, layer, name):
    """fp32 weight delta [out, in]"""
    a, b = factors(turbo, layer, name)
    return b @ a


def merge(weight_bf16, dw):
    """bf16 weight as deployed in ComfyUI"""
    return (weight_bf16.float() + dw.to(weight_bf16.device)).to(torch.bfloat16)
