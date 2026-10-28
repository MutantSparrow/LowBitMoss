"""In-process twin of gen_eval.py (no server): same ComfyUI nodes, seeds, sampler settings, file names.

    python -I gen_inproc.py --unet out/mixG_x.safetensors --tag mixG_x [--res 768] [--prompts eval|calib] [--bf16]
Writes $LOWBITMOSS_HOME/eval/<tag>/NN_seedSSSS.png and .../latents/NN_seedSSSS.latent (same format as gen_eval.py --save-latent).

On Windows, when VRAM runs short (the desktop, another job, or a process that has run for a while) the driver pages VRAM to system RAM and sampling
gets 15-40x slower (a normal image takes about 30 s) and does not recover. So the script runs as a supervisor around a worker process: the worker leaves 2.5 GB
of VRAM free (reserve_vram), exits with code 3 when one image takes longer than --watchdog seconds, and also exits on purpose after --recycle images;
the supervisor then restarts it and it skips the images that already exist."""
import argparse
import importlib
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import EVAL, NODES, PROMPTS, comfy_dir

COMFY = comfy_dir()  # fail before the supervisor starts workers
if "--child" not in sys.argv:  # supervisor: relaunch the worker (skipping finished images) when its watchdog fires
    cmd = [sys.executable, "-I", os.path.abspath(__file__), "--child", "--skip-existing"] + sys.argv[1:]
    for attempt in range(1, 13):
        rc = subprocess.call(cmd)
        if rc == 0:
            sys.exit(0)
        print("gen_inproc: worker exited with %d, restarting (%d/12)" % (rc, attempt), flush=True)
    sys.exit(rc)

sys.path.insert(0, COMFY)
sys.path.insert(0, NODES)
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

import comfy.cli_args

comfy.cli_args.args.reserve_vram = 2.5  # headroom for the desktop: without it Windows pages VRAM to RAM and sampling slows ~40x
import comfy.sd
import nodes

ap = argparse.ArgumentParser()
ap.add_argument("--unet", required=True, help="path to a moss_lowbit_v1 file, or a bf16 file with --bf16")
ap.add_argument("--tag", required=True)
ap.add_argument("--res", type=int, default=768)
ap.add_argument("--prompts", default="eval")
ap.add_argument("--n", type=int, default=0, help="only the first n eval prompts")
ap.add_argument("--bf16", action="store_true", help="stock loader instead of the low-bit loader")
ap.add_argument("--skip-existing", action="store_true")
ap.add_argument("--ids", default="", help="comma-separated prompt ids to generate (default: all)")
ap.add_argument("--child", action="store_true", help="internal: run the worker (the supervisor adds this)")
ap.add_argument("--recycle", type=int, default=5, help="planned restart of the worker after this many images (memory state degrades over a run); 0 = never")
ap.add_argument("--watchdog", type=int, default=150, help="seconds after which a single image is considered stalled and the process exits with code 3")
a = ap.parse_args()

NEG = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
items = json.load(open(os.path.join(PROMPTS, "eval_prompts.json" if a.prompts == "eval" else "calib_prompts.json"), encoding="utf-8"))
if a.prompts != "eval":
    for it in items:
        it["seed"] = 7000 + it["id"]
if a.n:
    items = items[:a.n]
if a.ids:
    items = [it for it in items if it["id"] in {int(v) for v in a.ids.split(",")}]
out = os.path.join(EVAL, a.tag)
os.makedirs(os.path.join(out, "latents"), exist_ok=True)

if a.bf16:
    model = comfy.sd.load_diffusion_model(a.unet, model_options={"dtype": torch.bfloat16})
else:
    lowbit = importlib.import_module("ComfyUI-MOSS-LowBit.lowbit")
    model = comfy.sd.load_diffusion_model(a.unet, model_options={"dtype": torch.bfloat16, "custom_operations": lowbit.lowbit_ops(torch.bfloat16)})
clip = nodes.CLIPLoader().load_clip("qwen3.5_9b_bf16.safetensors", "moss_image")[0]
vae = nodes.VAELoader().load_vae("flux2-vae.safetensors")[0]
neg = nodes.CLIPTextEncode().encode(clip, NEG)[0]

def stalled():  # a normal image takes ~30 s; a VRAM-paging stall is 15-40x slower and never recovers, so restart the process
    print("watchdog: image took > %d s, exiting for a restart" % a.watchdog, flush=True)
    os._exit(3)


t_all = time.time()
done_here = 0
for it in items:
    name = "%02d_seed%d" % (it["id"], it["seed"])
    png = os.path.join(out, name + ".png")
    if a.skip_existing and os.path.exists(png):
        continue
    t0 = time.time()
    watchdog = threading.Timer(a.watchdog, stalled)
    watchdog.start()
    pos = nodes.CLIPTextEncode().encode(clip, it["prompt"])[0]
    lat = {"samples": torch.zeros(1, 128, a.res // 16, a.res // 16)}
    lat = nodes.KSampler().sample(model, it["seed"], 25, 2.5, "er_sde", "simple", pos, neg, lat)[0]
    img = nodes.VAEDecode().decode(vae, lat)[0]
    Image.fromarray((img[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()).save(png)
    save_file({"latent_tensor": lat["samples"].float().cpu().contiguous()}, os.path.join(out, "latents", name + ".latent"))
    watchdog.cancel()
    print("%s  %.1fs" % (name, time.time() - t0), flush=True)
    done_here += 1
    if a.recycle and done_here >= a.recycle and it is not items[-1]:
        print("recycling the worker after %d images" % done_here, flush=True)
        os._exit(3)
print("done %d images in %.1f min -> %s" % (len(items), (time.time() - t_all) / 60, out))
