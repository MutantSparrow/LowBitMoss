"""PSNR / max abs difference between the same-named PNGs of two folders.
    python -I png_psnr.py <dir_ref> <dir_other> [<dir_other> ...]"""
import os
import sys

import numpy as np
from PIL import Image

ref = sys.argv[1]
for other in sys.argv[2:]:
    ps, mx = [], []
    for n in sorted(f for f in os.listdir(ref) if f.endswith(".png")):
        p = os.path.join(other, n)
        if not os.path.exists(p):
            continue
        a, b = (np.asarray(Image.open(q).convert("RGB"), dtype=np.float32) / 255 for q in (os.path.join(ref, n), p))
        ps.append(10 * np.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12)))
        mx.append(float(np.abs(a - b).max()))
    print("%-48s %2d images  PSNR vs %s: mean %.2f dB  min %.2f dB  max abs diff %.3f" % (os.path.basename(other), len(ps), os.path.basename(ref), np.mean(ps), np.min(ps), max(mx)))
