"""Run the bf16 teacher's real sampling (25 steps, cfg 2.5, er_sde) on calibration prompts and record the
(x_t, timestep, context) the diffusion model sees at a stratified subset of steps -> data/calib_<res>/p####.pt"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA, PROMPTS, comfy_dir

sys.path.insert(0, comfy_dir())
import torch

import comfy.model_management
import nodes

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--keep", type=int, default=8, help="distinct sampler steps kept per prompt")
ap.add_argument("--prompts", default=os.path.join(PROMPTS, "calib_prompts.json"))
ap.add_argument("--out", default=None)
ap.add_argument("--prefix", default="p", help="file name prefix; 'p-t' sorts before the 'p0...' files so the held-out (last) files stay the same")
args = ap.parse_args()
out_dir = args.out or os.path.join(DATA, "calib_%d" % args.res)
os.makedirs(out_dir, exist_ok=True)

prompts = json.load(open(args.prompts, encoding="utf-8"))
prompts = [p["prompt"] if isinstance(p, dict) else p for p in prompts][args.start:args.start + args.n]

NEG = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
model = nodes.UNETLoader().load_unet("moss_image_sft_bf16.safetensors", "default")[0]
clip = nodes.CLIPLoader().load_clip("qwen3.5_9b_bf16.safetensors", "moss_image")[0]
neg = nodes.CLIPTextEncode().encode(clip, NEG)[0]

calls = []
dm = model.model.diffusion_model
orig_forward = dm.forward


def recording_forward(x, timestep, context, *a, **kw):
    for b in range(x.shape[0]):
        calls.append((x[b].detach().cpu(), timestep[b].detach().float().cpu(), context[b].detach().cpu()))
    return orig_forward(x, timestep, context, *a, **kw)


dm.forward = recording_forward
rng = random.Random(0)
for i, text in enumerate(prompts, start=args.start):
    path = os.path.join(out_dir, "%s%04d.pt" % (args.prefix, i - args.start))
    if os.path.exists(path):
        continue
    calls.clear()
    pos = nodes.CLIPTextEncode().encode(clip, text)[0]
    lat = {"samples": torch.zeros(1, 128, args.res // 16, args.res // 16)}
    nodes.KSampler().sample(model, 1000 + i, 25, 2.5, "er_sde", "simple", pos, neg, lat)
    ts = sorted({round(float(c[1]), 5) for c in calls}, reverse=True)
    # stratified choice of sampler steps (always keep the first and last), both cond and uncond calls of a step
    idx = sorted({0, len(ts) - 1} | {min(len(ts) - 1, int(k * len(ts) / args.keep + rng.random() * len(ts) / args.keep)) for k in range(args.keep)})
    chosen = {ts[j] for j in idx}
    keep = [{"x": c[0].bfloat16(), "t": c[1], "ctx": c[2].bfloat16()} for c in calls if round(float(c[1]), 5) in chosen]
    torch.save({"prompt": text, "samples": keep}, path)
    print(i, len(calls), "calls ->", len(keep), "kept; steps", len(ts), flush=True)
    comfy.model_management.soft_empty_cache()
