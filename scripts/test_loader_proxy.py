"""Output rel MSE vs the teacher on held-out captured samples, but through the ComfyUI loader path (MossLowBit ops).

    python -I test_loader_proxy.py <file.safetensors> [n_samples]      (ComfyUI .venv python, COMFYUI_DIR set; needs a GPU, data/calib_512 and data/teacher_out_512.pt)
"""
import glob
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA, NODES, comfy_dir

sys.path.insert(0, comfy_dir())
sys.path.insert(0, NODES)
import torch

import comfy.model_management
import comfy.sd

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
lowbit = importlib.import_module("ComfyUI-MOSS-LowBit.lowbit")
mp = comfy.sd.load_diffusion_model(path, model_options={"dtype": torch.bfloat16, "custom_operations": lowbit.lowbit_ops(torch.bfloat16)})
comfy.model_management.load_model_gpu(mp)
dm = mp.model.diffusion_model
files = sorted(glob.glob(os.path.join(DATA, "calib_512", "p*.pt")))
counts = [len(torch.load(f)["samples"]) for f in files]
targets = torch.load(os.path.join(DATA, "teacher_out_512.pt"))
start = sum(counts[:-max(2, len(files) // 20)])
sel = list(range(start, sum(counts)))[::max(1, (sum(counts) - start) // n)][:n]
num = den = 0.0
off = 0
for f, c in zip(files, counts):
    ss = None
    for i in sel:
        if off <= i < off + c:
            ss = ss or torch.load(f)["samples"]
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
