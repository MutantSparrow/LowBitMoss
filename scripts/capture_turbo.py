"""Record the (x_t, timestep, context) the DISTILLED model (the turbo-merged bf16 checkpoint of make_turbo_teacher.py, euler/simple, 4 steps, CFG 1.0) sees on calibration prompts -> data/<out>/p####.pt
All 4 sampler steps of every prompt are kept (one conditional call per step, there is no negative branch at CFG 1).
--prompts is a json list of prompts (prompts/calib_prompts.json, prompts/calib_text_prompts.json); --prefix keeps several captures apart (files sort by name, the last ones are held out)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import AUX, DATA, comfy_dir

sys.path.insert(0, comfy_dir())
import torch

import comfy.cli_args

comfy.cli_args.args.reserve_vram = 2.5
import comfy.model_management
import comfy.sd
import nodes

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=768)
ap.add_argument("--prompts", required=True)
ap.add_argument("--prefix", default="p")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--out", default=None)
ap.add_argument("--unet", default=os.path.join(AUX, "turbo4", "moss_image_sft_turbo4_bf16.safetensors"))
ap.add_argument("--seed0", type=int, default=3000)
args = ap.parse_args()
out_dir = args.out or os.path.join(DATA, "calib_turbo_%d" % args.res)
os.makedirs(out_dir, exist_ok=True)

prompts = json.load(open(args.prompts, encoding="utf-8"))
prompts = [p["prompt"] if isinstance(p, dict) else p for p in prompts]
if args.n:
    prompts = prompts[:args.n]

model = comfy.sd.load_diffusion_model(args.unet)
clip = nodes.CLIPLoader().load_clip("qwen3.5_9b_bf16.safetensors", "moss_image")[0]
neg = nodes.CLIPTextEncode().encode(clip, "")[0]

calls = []
dm = model.model.diffusion_model
orig_forward = dm.forward


def recording_forward(x, timestep, context, *a, **kw):
    for b in range(x.shape[0]):
        calls.append((x[b].detach().cpu(), timestep[b].detach().float().cpu(), context[b].detach().cpu()))
    return orig_forward(x, timestep, context, *a, **kw)


dm.forward = recording_forward
for i, text in enumerate(prompts):
    path = os.path.join(out_dir, "%s%04d.pt" % (args.prefix, i))
    if os.path.exists(path):
        continue
    calls.clear()
    pos = nodes.CLIPTextEncode().encode(clip, text)[0]
    lat = {"samples": torch.zeros(1, 128, args.res // 16, args.res // 16)}
    nodes.KSampler().sample(model, args.seed0 + i, 4, 1.0, "euler", "simple", pos, neg, lat)
    keep = [{"x": c[0].bfloat16(), "t": c[1], "ctx": c[2].bfloat16()} for c in calls]
    torch.save({"prompt": text, "samples": keep}, path)
    print(i, len(calls), "calls kept", flush=True)
    comfy.model_management.soft_empty_cache()
