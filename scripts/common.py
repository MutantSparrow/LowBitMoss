import os
import sys

import torch

from _paths import comfy_dir, teacher_path

COMFY = comfy_dir()
TEACHER = teacher_path()
for p in (COMFY, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

import comfy.model_management  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.utils  # noqa: E402
from comfy.ldm.flux.layers import timestep_embedding  # noqa: E402
from comfy.ldm.moss_image.model import MOSSImage, modulated_rms_norm  # noqa: E402

comfy.model_management.in_training = True  # differentiable, unfused paths in the block code

LINEARS = ("qkv", "o", "ffn_up_gate", "ffn_down")


def load_teacher_sd():
    return comfy.utils.load_torch_file(TEACHER)


def build_model(sd, device="cpu", dtype=torch.bfloat16):
    sd = {k: v for k, v in sd.items() if k != "rope.bands"}
    m = MOSSImage(dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, latent_channels=128,
                  fused_qkv=True, dtype=dtype, device=device, operations=comfy.ops.disable_weight_init)
    m.load_state_dict(sd, strict=True)
    return m.requires_grad_(False).eval()


def prefix(m, x, t, ctx):
    """everything before the block stack. x [B,128,H,W], t [B], ctx [B,L,4096] -> tokens, mod, pe, head mod"""
    B, _, H, W = x.shape
    pe = m.rope(H, W, ctx.shape[1], x.device)
    vec = m.time_embedding(timestep_embedding(t, 256).to(x.dtype))
    mod = m.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    head = m.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
    h = m.text_embedder.proj(ctx)
    for blk in m.text_embedder.blocks:
        h = blk(h, pe[:, H * W:])
    tok = torch.cat((m.video_embedder(x.flatten(2).transpose(1, 2)), h), dim=1)
    return tok, mod, pe, head, (H, W)


def suffix(m, tok, head, hw):
    H, W = hw
    shift, scale = head
    out = m.final_linear_video(modulated_rms_norm(tok[:, :H * W], m.final_norm_video, scale, shift))
    return out.transpose(1, 2).reshape(tok.shape[0], m.latent_channels, H, W)
