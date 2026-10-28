"""Where do the bytes of a moss_lowbit_v1 file go?  python -I size_report.py <file>"""
import json
import re
import struct
import sys
from collections import Counter

p = sys.argv[1]
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n))
h.pop("__metadata__", None)
size = {"DT": 0}
lora_fmt = "fp8 + scales" if any(x.endswith(".lora_a_scale") for x in h) else "bf16"
cat = Counter()
count = Counter()
for k, v in h.items():
    a, b = v["data_offsets"]
    nbytes = b - a
    if k.startswith("layers.") and k.endswith(".weight") and v["dtype"] == "U8":
        c = "block linears: packed codes"
    elif k.endswith(".weight_scale"):
        c = "group scales (fp16)" if k.startswith("layers.") else "non-block scales"
    elif ".lora_" in k:
        c = "LoRA branches (%s)" % lora_fmt + (" blocks" if k.startswith("layers.") else " non-block")
    elif k.endswith(".weight_kind"):
        c = "kind flags"
    elif k.startswith("layers.") and re.search(r"(norm|q_norm|k_norm)", k):
        c = "block norms"
    elif k.startswith("text_embedder.") and v["dtype"] == "U8":
        c = "text embedder: packed codes"
    elif k.startswith("text_embedder."):
        c = "text embedder: bf16 rest (norms, biases)"
    elif v["dtype"] == "BF16":
        c = "other bf16: " + re.sub(r"\.(weight|bias)$", "", k)
    else:
        c = "other: " + k
    cat[c] += nbytes
    count[c] += 1
tot = sum(cat.values())
print("%s  total %.3f GB" % (p.split("\\")[-1], tot / 1e9))
for c, b in sorted(cat.items(), key=lambda x: -x[1]):
    print("  %7.1f MB  %5.1f%%  %-48s (%d tensors)" % (b / 1e6, 100 * b / tot, c, count[c]))
