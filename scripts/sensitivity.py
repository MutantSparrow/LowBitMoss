"""Per-layer sensitivity: quantize ONE block linear at a time (optimal-scale PTQ) in the bf16 teacher and measure the relative
MSE of the full model's output on held-out captured samples. -> logs/sens_<kind>.json  {block: {linear: rel_mse}}"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import lowbit_pack as L
from _paths import DATA, LOGS
from common import *  # noqa: F401,F403
from qlayer import fake_quant, init_scale

ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="tern")
ap.add_argument("--data", default=os.path.join(DATA, "calib_512"))
ap.add_argument("--targets", default=os.path.join(DATA, "teacher_out_512.pt"))
ap.add_argument("--n", type=int, default=24)
args = ap.parse_args()
kind = L.KIND_BY_NAME[args.kind]

files = sorted(glob.glob(os.path.join(args.data, "p*.pt")))
targets = torch.load(args.targets)
counts = [len(torch.load(f)["samples"]) for f in files]
n_hold_files = max(2, len(files) // 20)
start = sum(counts[:-n_hold_files])
idx = list(range(start, sum(counts)))
idx = idx[::max(1, len(idx) // args.n)][:args.n]
samples = {}
off = 0
for f, c in zip(files, counts):
    if off + c > start:
        ss = torch.load(f)["samples"]
        for j, s in enumerate(ss):
            if off + j in idx:
                samples[off + j] = s
    off += c

m = build_model(load_teacher_sd(), device="cuda")


def evaluate():
    num = den = 0.0
    with torch.no_grad():
        for i in idx:
            s = samples[i]
            x, t, ctx = s["x"].unsqueeze(0).cuda(), s["t"].reshape(1).cuda(), s["ctx"].unsqueeze(0).cuda()
            tok, mod, pe, head, hw = prefix(m, x, t, ctx)
            for blk in m.layers:
                tok = blk(tok, mod, pe)
            o = suffix(m, tok, head, hw)[0].float()
            y = targets[i].cuda().float()
            num += ((o - y) ** 2).sum().item()
            den += (y ** 2).sum().item()
    return num / den


print("sanity (no change): %.6f" % evaluate(), flush=True)
os.makedirs(LOGS, exist_ok=True)
res = {}
for b in range(42):
    res[b] = {}
    for name in LINEARS:
        lin = m.layers[b].proj[name]
        w0 = lin.weight.data.clone()
        wf = w0.float()
        lin.weight.data = fake_quant(wf, init_scale(wf, kind), kind).to(torch.bfloat16)
        res[b][name] = evaluate()
        lin.weight.data = w0
    print("block %2d  %s" % (b, "  ".join("%s %.5f" % (n, res[b][n]) for n in LINEARS)), flush=True)
    json.dump(res, open(os.path.join(LOGS, "sens_%s.json" % args.kind), "w"), indent=1)
