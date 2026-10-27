"""Sensitivity of the non-block linears (embedders, text-embedder blocks, modulation, head): quantize ONE of them at a time
(optimal-scale PTQ) and measure the full model's output rel MSE vs the teacher on held-out captured samples.
The teacher blocks are streamed through the GPU one at a time so this needs ~1.5 GB of VRAM. -> logs/sens_extra_<kind>.json"""
import argparse
import copy
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
ap.add_argument("--kinds", default="int4,int2")
ap.add_argument("--data", default=os.path.join(DATA, "calib_512"))
ap.add_argument("--targets", default=os.path.join(DATA, "teacher_out_512.pt"))
ap.add_argument("--n", type=int, default=24)
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.data, "p*.pt")))
targets = torch.load(args.targets)
counts = [len(torch.load(f)["samples"]) for f in files]
n_hold_files = max(2, len(files) // 20)
start = sum(counts[:-n_hold_files])
idx = list(range(start, sum(counts)))
idx = idx[::max(1, len(idx) // args.n)][:args.n]
samples, off = [], 0
for f, c in zip(files, counts):
    if off + c > start:
        ss = torch.load(f)["samples"]
        samples += [ss[i - off] for i in idx if off <= i < off + c]
    off += c

m = build_model(load_teacher_sd(), device="cpu")
blocks = list(m.layers)
small = m
small.layers = torch.nn.ModuleList()  # blocks stay on the CPU and are streamed through the GPU one at a time
small.to("cuda")

NAMES = (["video_embedder", "text_embedder.proj", "shared_block_modulation", "video_head_modulation", "time_embedding.0", "time_embedding.2", "final_linear_video"]
         + ["text_embedder.blocks.%d.%s" % (i, n) for i in range(2) for n in ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down")])


def get(root, path):
    for p in path.split("."):
        root = root[int(p)] if p.isdigit() else getattr(root, p)
    return root


def evaluate():
    with torch.no_grad():
        state = []
        for s in samples:
            x, t, ctx = s["x"].unsqueeze(0).cuda(), s["t"].reshape(1).cuda(), s["ctx"].unsqueeze(0).cuda()
            tok, mod, pe, head, hw = prefix(small, x, t, ctx)
            state.append([tok, mod, pe, head, hw])
        for blk in blocks:
            blk.cuda()
            for st in state:
                st[0] = blk(st[0], st[1], st[2])
            blk.cpu()
        num = den = 0.0
        for st, i in zip(state, idx):
            o = suffix(small, st[0], st[3], st[4])[0].float()
            y = targets[i].cuda().float()
            num += ((o - y) ** 2).sum().item()
            den += (y ** 2).sum().item()
    return num / den


print("sanity (no change): %.6f" % evaluate(), flush=True)
os.makedirs(LOGS, exist_ok=True)
res = {}
for kname in args.kinds.split(","):
    kind = L.KIND_BY_NAME[kname]
    res[kname] = {}
    for name in NAMES:
        lin = get(small, name)
        w0 = lin.weight.data.clone()
        wf = w0.float()
        lin.weight.data = fake_quant(wf, init_scale(wf, kind), kind).to(torch.bfloat16)
        res[kname][name] = evaluate()
        lin.weight.data = w0
        print("%-5s %-34s params %7.1fM  rel mse %.5f" % (kname, name, w0.numel() / 1e6, res[kname][name]), flush=True)
        json.dump(res, open(os.path.join(LOGS, "sens_extra.json"), "w"), indent=1)
