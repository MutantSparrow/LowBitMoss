"""One labelled strip of the same prompt id across several eval tags, for side-by-side artifact checks.
    python -I strip_id.py <id> <tag> [<tag> ...] [--w 384] [--out path]   -> $LOWBITMOSS_HOME/eval/strip_<id>.png (6 images per row)"""
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import EVAL as EV

args = sys.argv[1:]
w = int(args[args.index("--w") + 1]) if "--w" in args else 384
out = args[args.index("--out") + 1] if "--out" in args else None
skip = {i for k, i in ((k, k + 1) for k, a in enumerate(args) if a in ("--w", "--out"))} | {k for k, a in enumerate(args) if a in ("--w", "--out")}
args = [a for k, a in enumerate(args) if k not in skip]
pid, tags = int(args[0]), args[1:]
cols = min(6, len(tags))
rows = (len(tags) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w, rows * (w + 16)), (24, 24, 24))
d = ImageDraw.Draw(sheet)
for i, tag in enumerate(tags):
    names = [n for n in os.listdir(os.path.join(EV, tag)) if n.startswith("%02d_" % pid) and n.endswith(".png")]
    x, y = (i % cols) * w, (i // cols) * (w + 16)
    d.text((x + 3, y + 2), tag, fill=(255, 255, 0))
    if names:
        sheet.paste(Image.open(os.path.join(EV, tag, names[0])).convert("RGB").resize((w, w), Image.LANCZOS), (x, y + 16))
path = out or os.path.join(EV, "strip_%02d.png" % pid)
sheet.save(path)
print(path, sheet.size)
