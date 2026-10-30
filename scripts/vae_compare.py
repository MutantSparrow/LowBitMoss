"""Decode the saved latents of an eval tag with several VAE files (models/vae): pixel PSNR of every VAE against the first one, decode time, peak VRAM; PNGs go to $LOWBITMOSS_HOME/eval/<tag>_<vae stem>/.
GPU. The printed peak VRAM is not reliable: ComfyUI's virtual allocator confuses torch.cuda.max_memory_allocated (it printed 95.5 GB), do not quote it.

    python -I vae_compare.py --tag enc_i4e8 --vae flux2-vae.safetensors flux2-vae-small-decoder.safetensors"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import EVAL, comfy_dir

sys.path.insert(0, comfy_dir())
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

import comfy.cli_args  # noqa: E402

comfy.cli_args.args.reserve_vram = 2.5
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--vae", nargs="+", required=True)
ap.add_argument("--cpu", action="store_true", help="decode on the CPU (while the GPU is busy); timing and VRAM are then not comparable to the GPU")
args = ap.parse_args()
comfy.cli_args.args.cpu = args.cpu
import comfy.model_management  # noqa: E402
import nodes  # noqa: E402

files = sorted(glob.glob(os.path.join(EVAL, args.tag, "latents", "*.latent")))
ref = []
for v, name in enumerate(args.vae):
    vae = nodes.VAELoader().load_vae(name)[0]
    out_dir = os.path.join(EVAL, "%s_%s" % (args.tag, os.path.splitext(name)[0]))
    os.makedirs(out_dir, exist_ok=True)
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.reset_peak_memory_stats()
    t0, psnrs, imgs = time.time(), [], []
    for i, f in enumerate(files):
        lat = {"samples": load_file(f)["latent_tensor"]}
        img = nodes.VAEDecode().decode(vae, lat)[0][0].clamp(0, 1).float().cpu()
        imgs.append(img)
        Image.fromarray((img * 255).round().to(torch.uint8).numpy()).save(os.path.join(out_dir, os.path.basename(f).replace(".latent", ".png")))
        if v:
            mse = ((img - ref[i]) ** 2).mean().item()
            psnrs.append(10 * np.log10(1.0 / max(mse, 1e-12)))
    dt = time.time() - t0
    if not v:
        ref = imgs
    print("%-36s %d images  %.1f s (%.2f s/img incl. first-call setup)  peak VRAM %.2f GB  %s" % (
        name, len(files), dt, dt / len(files), 0.0 if args.cpu else torch.cuda.max_memory_allocated() / 1e9,
        "reference" if not v else "PSNR vs %s: mean %.2f dB, min %.2f dB" % (args.vae[0], float(np.mean(psnrs)), float(np.min(psnrs)))), flush=True)
    del vae
    comfy.model_management.soft_empty_cache()
