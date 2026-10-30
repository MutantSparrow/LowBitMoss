"""List unreadable / truncated PNGs (and their .latent files) in the given eval folders; --delete removes the broken ones so a skip-existing run regenerates them.
    python -I check_pngs.py <dir> [<dir> ...] [--delete]"""
import os
import sys

from PIL import Image
from safetensors import safe_open

delete = "--delete" in sys.argv
bad = total = 0
for d in (a for a in sys.argv[1:] if a != "--delete"):
    for n in sorted(os.listdir(d)):
        if not n.endswith(".png"):
            continue
        total += 1
        p = os.path.join(d, n)
        lat = os.path.join(d, "latents", n[:-4] + ".latent")
        problems = []
        try:
            with Image.open(p) as im:
                im.load()
        except Exception as e:
            problems.append("png: %s" % e)
        if os.path.exists(lat):
            try:
                with safe_open(lat, "pt") as f:
                    f.get_tensor("latent_tensor")
            except Exception as e:
                problems.append("latent: %s" % e)
        if problems:
            bad += 1
            print("BROKEN", p, "; ".join(problems))
            if delete:
                for q in (p, lat):
                    if os.path.exists(q):
                        os.remove(q)
print("checked %d images, %d broken%s" % (total, bad, " (deleted)" if delete and bad else ""))
