"""Output rel MSE vs the teacher on held-out captured samples, but through the ComfyUI loader path (MossLowBit ops).

    python -I test_loader_proxy.py MODEL [n_samples] [--data CAPTURES] [--targets TARGETS]

Defaults to the expanded calibration pool and matching teacher_out_512_plus.pt.
"""
import argparse
import glob
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA, NODES, comfy_dir

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("path", help="Packed model file")
ap.add_argument("n_samples", nargs="?", type=int, default=8)
ap.add_argument("--data", default=os.path.join(DATA, "calib_512"), help="Capture directory")
ap.add_argument("--targets", default=os.path.join(DATA, "teacher_out_512_plus.pt"), help="Teacher outputs matching capture order")
args = ap.parse_args()
if args.n_samples < 1:
    ap.error("n_samples must be positive")

def select_samples(counts, target_count, n):
    total = sum(counts)
    if not counts or total == 0:
        raise ValueError("No calibration samples found")
    if target_count != total:
        raise ValueError("Capture and teacher-target counts differ; choose the matching --data and --targets pair from DATA.md")
    start = sum(counts[:-max(2, len(counts) // 20)])
    return list(range(start, total))[::max(1, (total - start) // n)][:n]

sys.path.insert(0, comfy_dir())
sys.path.insert(0, NODES)
import torch

import comfy.model_management
import comfy.sd

path = args.path
n = args.n_samples
files = sorted(glob.glob(os.path.join(args.data, "p*.pt")))
counts = [len(torch.load(f, map_location="cpu", weights_only=True)["samples"]) for f in files]
targets = torch.load(args.targets, map_location="cpu", weights_only=True)
sel = select_samples(counts, len(targets), n)
lowbit = importlib.import_module("ComfyUI-MOSS-LowBit.lowbit")
mp = comfy.sd.load_diffusion_model(path, model_options={"dtype": torch.bfloat16, "custom_operations": lowbit.lowbit_ops(torch.bfloat16)})
comfy.model_management.load_model_gpu(mp)
dm = mp.model.diffusion_model
num = den = 0.0
off = 0
for f, c in zip(files, counts):
    ss = None
    for i in sel:
        if off <= i < off + c:
            ss = ss or torch.load(f, map_location="cpu", weights_only=True)["samples"]
            s = ss[i - off]
            x, t, ctx = s["x"].unsqueeze(0).cuda(), s["t"].reshape(1).cuda(), s["ctx"].unsqueeze(0).cuda()
            with torch.no_grad():
                o = dm(x, t, ctx).float()
            y = targets[i].cuda().float().unsqueeze(0)
            e = ((o - y) ** 2).sum().item() / (y ** 2).sum().item()
            print("sample %d  t=%.3f  rel mse %.5f  nan=%s" % (i, float(s["t"]), e, bool(torch.isnan(o).any())), flush=True)
            num += ((o - y) ** 2).sum().item()
            den += (y ** 2).sum().item()
    off += c
print("loader-path rel mse %.5f" % (num / den))
