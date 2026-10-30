"""Make a decoder-only FLUX.2 VAE file ComfyUI's VAE loader accepts from black-forest-labs/FLUX.2-small-decoder's small_decoder.safetensors:
the decoder tensors get the 'decoder.' prefix and the three batch-norm latent statistics (bn.*) are copied from the full FLUX.2 VAE. The encoder is left out (text-to-image only decodes).
CPU only. Reads $COMFYUI_DIR/models/vae/flux2-vae.safetensors and flux2-small-decoder-only.safetensors (the downloaded small_decoder.safetensors, renamed). The default output is
$LOWBITMOSS_HOME/out/flux2-small-decoder-vae.safetensors (112 MB, 143 tensors); copy it into models/vae.

    python -I -B make_small_vae.py [--bf16]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import OUT, comfy_dir

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

V = os.path.join(comfy_dir(), "models", "vae")
ap = argparse.ArgumentParser()
ap.add_argument("--bf16", action="store_true", help="store the decoder weights in bf16")
ap.add_argument("--out", default=os.path.join(OUT, "flux2-small-decoder-vae.safetensors"))
args = ap.parse_args()

full, dec = load_file(os.path.join(V, "flux2-vae.safetensors")), load_file(os.path.join(V, "flux2-small-decoder-only.safetensors"))
out = {"decoder." + k: (v.to(torch.bfloat16) if args.bf16 else v) for k, v in dec.items()}
want = {k for k in full if k.startswith("decoder.")}
assert set(out) == want, (sorted(set(out) - want)[:5], sorted(want - set(out))[:5])
out.update({k: v for k, v in full.items() if k.startswith("bn.")})
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
save_file({k: v.contiguous() for k, v in out.items()}, args.out, metadata={"format": "pt"})
print("wrote %s: %d tensors, %.1f M params, %.3f GB" % (args.out, len(out), sum(v.numel() for v in out.values()) / 1e6, os.path.getsize(args.out) / 1e9))
